"""CPU-optimised Manifold-Constrained Hyper-Connection (mHC).

``ManifoldHyperConnectionCPU`` is a drop-in subclass of the upstream
``Model.layers.mhc.ManifoldHyperConnection`` (same parameters, same state
dict). With ``MONTLOK_CPP=1`` its forward runs the two fused C++ ops from
``montlok.cpp`` instead of the ~100 tiny PyTorch ops of the reference (20
Sinkhorn iterations x logsumexp/subtract, two batched matmuls over B*L tiny
n x n matrices, ...). Every intermediate is accumulated in double precision
and rounded to float32 once, so the outputs agree with the fp32 reference to
the reference's own rounding (a few ulp).

``forward_tail(streams, fn, n_last)`` evaluates the residual for the trailing
``n_last`` positions only. The aggregate passed to ``fn`` still covers the
whole sequence (attention needs every key), ``fn`` returns the outputs of the
trailing positions, and the mixing/write step runs on those positions alone.

``MONTLOK_CPP_MHC=0`` runs this layer with the pure PyTorch reference while
keeping the other kernels, for A/B checks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

NATIVE = Path(__file__).resolve().parents[1] / "rdt4quant" / "native"
if str(NATIVE) not in sys.path:
    sys.path.insert(0, str(NATIVE))

from Model.layers.mhc import ManifoldHyperConnection  # noqa: E402

from mamba3_cpu import cpp_backend  # noqa: E402


def mhc_cpp_enabled() -> bool:
    return cpp_backend() is not None and os.environ.get("MONTLOK_CPP_MHC") != "0"


class ManifoldHyperConnectionCPU(ManifoldHyperConnection):
    """Upstream mHC with the coefficient/mixing math in fused C++ ops."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._proj_w64_key = None
        self._proj_w64 = None

    def _projection_block(self) -> torch.Tensor:
        """``cat(pre, post, res).weight`` widened to float64, cached per weights.

        The cache key includes ``_version`` so ``load_state_dict`` (which copies
        in place) invalidates it.
        """
        weights = (self.pre_proj.weight, self.post_proj.weight, self.res_proj.weight)
        key = tuple((w.data_ptr(), w._version) for w in weights)
        if key != self._proj_w64_key:
            with torch.no_grad():
                self._proj_w64 = torch.cat(weights, 0).detach().to(torch.float64).contiguous()
            self._proj_w64_key = key
        return self._proj_w64

    def _fast_path(self, streams: torch.Tensor) -> bool:
        return (
            mhc_cpp_enabled()
            and self.constrain
            and streams.dtype == torch.float32
            and streams.device.type == "cpu"
            and not torch.is_grad_enabled()
        )

    def _prepare_args(self) -> tuple:
        """Arguments shared by ``mhc_prepare`` and ``mhc_combine_prepare``."""
        return (
            self.dyn_norm.weight,
            self.dyn_norm.eps,
            self._projection_block(),
            self.pre_bias,
            self.post_bias,
            self.res_bias,
            self.pre_alpha,
            self.post_alpha,
            self.res_alpha,
            self.sinkhorn_iters,
        )

    def _prepare(self, streams: torch.Tensor, *, allow_broadcast: bool = False, output_norm=None):
        from montlok_loader import load_montlok

        if streams.ndim == 3:
            if not allow_broadcast:
                raise ValueError("broadcast streams are only valid at the start of a fused stage-2 pass")
            if streams.shape[-1] != self.d_model:
                raise ValueError(f"expected d_model={self.d_model}, got {streams.shape[-1]}")
            n_streams = self.n_streams
        else:
            self._check_streams(streams)
            n_streams = 0
        streams = streams.contiguous()
        ext = load_montlok()
        args = (streams, *self._prepare_args(), n_streams)
        if output_norm is not None:
            args += (output_norm.weight, output_norm.eps)
        return streams, ext.mhc_prepare(*args)

    @staticmethod
    def _prepared_parts(prepared):
        _agg, _pre, post, res = prepared
        return post, res

    def _combine(self, streams: torch.Tensor, prepared, out: torch.Tensor) -> torch.Tensor:
        from montlok_loader import load_montlok

        post, res = self._prepared_parts(prepared)
        return load_montlok().mhc_combine(streams, res, post, out.contiguous())

    def _combine_collapse(self, streams: torch.Tensor, prepared, out: torch.Tensor) -> torch.Tensor:
        from montlok_loader import load_montlok

        post, res = self._prepared_parts(prepared)
        return load_montlok().mhc_combine_collapse(streams, res, post, out.contiguous())

    def _combine_prepare(self, streams: torch.Tensor, prepared, out: torch.Tensor, next_hc, output_norm=None):
        """Write this residual and prepare ``next_hc`` in one C++ traversal."""
        from montlok_loader import load_montlok

        if not isinstance(next_hc, ManifoldHyperConnectionCPU):
            raise TypeError("next_hc must be a ManifoldHyperConnectionCPU")
        post, res = self._prepared_parts(prepared)
        args = (
            streams,
            res,
            post,
            out.contiguous(),
            *next_hc._prepare_args(),
        )
        if output_norm is not None:
            args += (output_norm.weight, output_norm.eps)
        values = load_montlok().mhc_combine_prepare(*args)
        new_streams, agg, pre, next_post, next_res = values
        return new_streams, (agg, pre, next_post, next_res)

    def _combine_tail(self, streams: torch.Tensor, prepared, out: torch.Tensor, n_last: int) -> torch.Tensor:
        """Write only the trailing ``n_last`` positions from prepared coefficients."""
        length = streams.shape[1]
        if not 1 <= n_last <= length:
            raise ValueError("n_last must be between 1 and the sequence length")
        if out.shape != (streams.shape[0], n_last, streams.shape[-1]):
            raise ValueError("wrapped fn must return [B, n_last, d_model]")
        post, res = self._prepared_parts(prepared)
        tail = slice(length - n_last, length)
        tail_streams = streams[:, tail].contiguous()
        tail_prepared = (
            None,
            None,
            post[:, tail].contiguous(),
            res[:, tail].contiguous(),
        )
        return self._combine(tail_streams, tail_prepared, out)

    def forward(self, streams: torch.Tensor, fn) -> torch.Tensor:
        if not self._fast_path(streams):
            return super().forward(streams, fn)
        self._check_streams(streams)
        streams, prepared = self._prepare(streams)
        agg = prepared[0]
        out = fn(agg)
        if out.shape != agg.shape:
            raise ValueError("wrapped fn must preserve [B, L, d_model] shape")
        return self._combine(streams, prepared, out)

    def forward_tail(self, streams: torch.Tensor, fn, n_last: int = 1) -> torch.Tensor:
        """Residual output ``[B, n_last, n, d]`` for the trailing positions.

        ``fn`` receives the aggregate of every position ``[B, L, d]`` and must
        return ``[B, n_last, d]``.
        """
        self._check_streams(streams)
        length = streams.shape[1]
        if not 1 <= n_last <= length:
            raise ValueError("n_last must be between 1 and the sequence length")
        tail = slice(length - n_last, length)
        if not self._fast_path(streams):
            ref = self._dyn_ref(streams)
            pre_logits = self.pre_bias + self.pre_alpha * torch.tanh(self.pre_proj(ref))
            pre = torch.sigmoid(pre_logits).to(dtype=streams.dtype)
            agg = torch.einsum("bln,blnd->bld", pre, streams)
            out = fn(agg)
            if out.shape != (streams.shape[0], n_last, streams.shape[-1]):
                raise ValueError("wrapped fn must return [B, n_last, d_model]")
            ref, streams = ref[:, tail], streams[:, tail]
            res = self.residual_matrix(ref).to(dtype=streams.dtype)
            mixed = torch.einsum("blij,bljd->blid", res, streams)
            post_logits = self.post_bias + self.post_alpha * torch.tanh(self.post_proj(ref))
            post = (2.0 * torch.sigmoid(post_logits)).to(dtype=streams.dtype)
            return mixed + post.unsqueeze(-1) * out.unsqueeze(-2)
        streams, prepared = self._prepare(streams)
        agg = prepared[0]
        out = fn(agg)
        if out.shape != (streams.shape[0], n_last, streams.shape[-1]):
            raise ValueError("wrapped fn must return [B, n_last, d_model]")
        return self._combine_tail(streams, prepared, out, n_last)
