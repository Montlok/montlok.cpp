# -*- coding: utf-8 -*-

"""Manifold-Constrained Hyper-Connections (mHC).

Reference: DeepSeek, *Manifold-Constrained Hyper-Connections* (arXiv:2512.24880),
building on Hyper-Connections (arXiv:2409.19606).

A residual connection is replaced by ``n`` parallel residual streams plus three
**data-dependent** maps (paper Eq. 5/7: ``b + alpha * tanh(theta @ x')``), each
projected onto a bounded manifold so the composite gain of a deep / recurrent
stack stays bounded:

* ``H_pre = sigmoid(.)``      — per-stream input aggregation weights in ``(0, 1)``.
* ``H_post = 2 * sigmoid(.)`` — per-stream output write weights in ``(0, 2)``.
* ``H_res = sinkhorn(.)``     — ``n x n`` doubly-stochastic cross-stream mixing
  (``||H_res||_2 <= 1`` by Birkhoff), which is what prevents the write term
  ``H_post * F`` from accumulating and exploding across layers.

Each map's pre-activation logits are ``static_bias + alpha * tanh(proj(x_ref))``
where ``x_ref`` is the RMS-normalized per-token aggregate of the streams, so the
coefficients are conditioned on the input (token-wise; no cross-position mixing,
so causality is preserved). The dynamic gate ``alpha`` is initialized to zero, so
at init the maps collapse to their static bias and the whole connection reduces
to the vanilla residual ``x + fn(x)`` with unit gain.

The connection is meant to live *inside* a layer (one instance per residual),
not wrapped around a recurrent loop. Expand once, keep ``n`` streams across
layers, collapse once.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.layers.rmsnorm import RMSNorm


def sinkhorn_knopp(log_alpha: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """Project ``exp(log_alpha)`` onto the Birkhoff polytope (doubly stochastic).

    Operates in log space for numerical stability by alternating row and column
    log-normalization. The last two dims are treated as the ``n x n`` matrix; any
    leading dims are batched.

    The iteration always runs in float32 (the result is cast back to the input
    dtype). Under bf16/fp16 autocast — as used during pretraining — repeated
    ``logsumexp`` in low precision drifts enough to break the doubly-stochastic
    guarantee, so the projection is computed in float32 regardless of input dtype.
    At least one normalization step is performed so the output is always (close
    to) doubly stochastic rather than an unnormalized ``exp(log_alpha)``.
    """

    if log_alpha.ndim < 2:
        raise ValueError("log_alpha must have at least 2 dims")
    if log_alpha.shape[-1] != log_alpha.shape[-2]:
        raise ValueError("log_alpha must be square in its last two dims")
    if n_iters < 0:
        raise ValueError("n_iters must be non-negative")

    in_dtype = log_alpha.dtype
    work = log_alpha.float()

    for _ in range(max(n_iters, 1)):
        work = work - torch.logsumexp(work, dim=-1, keepdim=True)
        work = work - torch.logsumexp(work, dim=-2, keepdim=True)

    return torch.exp(work).to(in_dtype)


class ManifoldHyperConnection(nn.Module):
    """A single manifold-constrained hyper-connection over ``n`` streams.

    ``forward(streams, fn)`` aggregates the streams into one tensor, applies the
    wrapped layer ``fn`` once, mixes the streams through the doubly-stochastic
    residual matrix, and writes the layer output back into every stream:

        agg          = sum_i H_pre_i * streams_i
        out          = fn(agg)
        new_streams  = H_res @ streams + H_post (x) out

    At initialization the maps reduce to the vanilla residual ``x + fn(x)`` with
    unit gain (see the init notes below), so an mHC layer is a drop-in for a
    standard pre-norm residual layer.
    """

    def __init__(
        self,
        d_model: int,
        n_streams: int = 4,
        sinkhorn_iters: int = 20,
        constrain: bool = True,
        rmsnorm_eps: float = 1e-5,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if n_streams <= 0:
            raise ValueError("n_streams must be positive")
        if sinkhorn_iters < 0:
            raise ValueError("sinkhorn_iters must be non-negative")

        self.d_model = d_model
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.constrain = constrain

        n = n_streams

        # --- Static biases (``b`` in ``b + alpha * tanh(theta @ x')``) ---
        # H_pre: init so sum_i sigmoid(pre_i) == 1 with all streams equal, i.e.
        # sigmoid(pre_i) == 1 / n, so the aggregation of ``n`` copies of x is x.
        # For n == 1 that means sigmoid(pre) == 1, approximated by a large logit.
        if n > 1:
            pre_value = float(torch.log(torch.tensor(1.0 / (n - 1))))
            pre_init = torch.full((n,), pre_value)
        else:
            pre_init = torch.full((1,), 20.0)
        self.pre_bias = nn.Parameter(pre_init)

        # H_post: init at logit 0 -> 2 * sigmoid(0) == 1, so the layer output is
        # written into each stream exactly once (vanilla residual).
        self.post_bias = nn.Parameter(torch.zeros(n))

        # H_res: init near identity (doubly stochastic) so streams stay distinct
        # at the start; Sinkhorn of a strongly diagonal matrix is ~identity.
        self.res_bias = nn.Parameter(torch.eye(n) * 3.0)

        # --- Data-dependent term (``alpha * tanh(theta @ x')``) ---
        # ``x'`` is the RMS-normalized per-token aggregate of the streams; the
        # projections produce per-token logit offsets for each map.
        self.dyn_norm = RMSNorm(d_model, eps=rmsnorm_eps)
        self.pre_proj = nn.Linear(d_model, n, bias=False)
        self.post_proj = nn.Linear(d_model, n, bias=False)
        self.res_proj = nn.Linear(d_model, n * n, bias=False)

        # Per-map dynamic gate ``alpha``, initialized to zero so the maps start
        # at their static bias and the connection reduces to the vanilla
        # residual; the gates are learned to introduce data-dependence.
        self.pre_alpha = nn.Parameter(torch.zeros(()))
        self.post_alpha = nn.Parameter(torch.zeros(()))
        self.res_alpha = nn.Parameter(torch.zeros(()))

    def _dyn_ref(self, streams: torch.Tensor) -> torch.Tensor:
        """RMS-normalized per-token aggregate used to condition the maps."""

        return self.dyn_norm(streams.mean(dim=-2))

    def _res_matrix(self, res_logits: torch.Tensor) -> torch.Tensor:
        """Apply the manifold projection to ``[..., n, n]`` residual logits."""

        if self.constrain:
            return sinkhorn_knopp(res_logits, self.sinkhorn_iters)
        return F.softplus(res_logits)

    def residual_matrix(self, ref: torch.Tensor | None = None) -> torch.Tensor:
        """Return the cross-stream mixing matrix.

        With ``ref=None`` returns the static ``n x n`` matrix (zero dynamic
        term). With a ``[B, L, d_model]`` reference returns the data-dependent
        ``[B, L, n, n]`` matrices. Doubly stochastic when ``constrain`` is True.
        """

        if ref is None:
            return self._res_matrix(self.res_bias)

        dyn = self.res_alpha * torch.tanh(self.res_proj(ref))
        dyn = dyn.view(*ref.shape[:-1], self.n_streams, self.n_streams)
        return self._res_matrix(self.res_bias + dyn)

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """Replicate ``[B, L, d]`` into ``n`` streams ``[B, L, n, d]``."""

        if x.ndim != 3:
            raise ValueError("x must have shape [B, L, d_model]")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {x.shape[-1]}")

        return x.unsqueeze(-2).expand(-1, -1, self.n_streams, -1).contiguous()

    def collapse(self, streams: torch.Tensor) -> torch.Tensor:
        """Reduce ``n`` streams ``[B, L, n, d]`` back to ``[B, L, d]`` (mean)."""

        self._check_streams(streams)
        return streams.mean(dim=-2)

    def forward(self, streams: torch.Tensor, fn) -> torch.Tensor:
        self._check_streams(streams)

        ref = self._dyn_ref(streams)

        pre_logits = self.pre_bias + self.pre_alpha * torch.tanh(self.pre_proj(ref))
        pre = torch.sigmoid(pre_logits).to(dtype=streams.dtype)
        agg = torch.einsum("bln,blnd->bld", pre, streams)

        out = fn(agg)
        if out.shape != agg.shape:
            raise ValueError("wrapped fn must preserve [B, L, d_model] shape")

        res = self.residual_matrix(ref).to(dtype=streams.dtype)
        mixed = torch.einsum("blij,bljd->blid", res, streams)

        post_logits = self.post_bias + self.post_alpha * torch.tanh(self.post_proj(ref))
        post = (2.0 * torch.sigmoid(post_logits)).to(dtype=streams.dtype)
        write = post.unsqueeze(-1) * out.unsqueeze(-2)

        return mixed + write

    def _check_streams(self, streams: torch.Tensor) -> None:
        if streams.ndim != 4:
            raise ValueError("streams must have shape [B, L, n_streams, d_model]")
        if streams.shape[-2] != self.n_streams:
            raise ValueError(
                f"expected n_streams={self.n_streams}, got {streams.shape[-2]}"
            )
        if streams.shape[-1] != self.d_model:
            raise ValueError(
                f"expected d_model={self.d_model}, got {streams.shape[-1]}"
            )


def _check() -> None:
    torch.manual_seed(0)

    n = 4
    mat = sinkhorn_knopp(torch.randn(n, n), n_iters=20)
    print("sinkhorn_knopp")
    print(f"  row_sums: {mat.sum(dim=-1).tolist()}")
    print(f"  col_sums: {mat.sum(dim=-2).tolist()}")

    d_model = 16
    hc = ManifoldHyperConnection(d_model, n_streams=n, sinkhorn_iters=20)
    hc.eval()

    x = torch.randn(2, 5, d_model)
    streams = hc.expand(x)

    def zero_fn(inp):
        return torch.zeros_like(inp)

    out = hc(streams, zero_fn)
    print("ManifoldHyperConnection")
    print(f"  streams: {tuple(streams.shape)} -> {tuple(out.shape)}")

    # Composite gain of pure cross-stream mixing (F = 0) over 60 layers.
    s = streams
    for _ in range(60):
        s = hc(s, zero_fn)
    gain = hc.collapse(s).norm() / hc.collapse(streams).norm()
    print(f"  composite_gain_60_layers: {gain.item():.4f}")

    xg = torch.randn(2, 5, d_model, requires_grad=True)
    loss = hc.collapse(hc(hc.expand(xg), lambda inp: inp)).sum()
    loss.backward()
    print(f"  grad_norm: {xg.grad.norm().item():.6f}")


if __name__ == "__main__":
    _check()
