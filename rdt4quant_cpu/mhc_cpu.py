"""CPU-optimised Manifold-Constrained Hyper-Connection (mHC).

``ManifoldHyperConnectionCPU`` is a drop-in subclass of the upstream
``Model.layers.mhc.ManifoldHyperConnection`` (same parameters, same state
dict). With ``MONTLOK_CPP=1`` its forward runs the two fused C++ ops from
``montlok.cpp`` instead of the ~100 tiny PyTorch ops of the reference (20
Sinkhorn iterations x logsumexp/subtract, two batched matmuls over B*L tiny
n x n matrices, ...). Every intermediate is accumulated in double precision
and rounded to float32 once, so the outputs agree with the fp32 reference to
the reference's own rounding (a few ulp).

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

    def forward(self, streams: torch.Tensor, fn) -> torch.Tensor:
        if (
            not mhc_cpp_enabled()
            or not self.constrain
            or streams.dtype != torch.float32
            or streams.device.type != "cpu"
            or torch.is_grad_enabled()
        ):
            return super().forward(streams, fn)

        from montlok_loader import load_montlok

        ext = load_montlok()
        self._check_streams(streams)
        streams = streams.contiguous()
        agg, _pre, post, res = ext.mhc_prepare(
            streams,
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
        out = fn(agg)
        if out.shape != agg.shape:
            raise ValueError("wrapped fn must preserve [B, L, d_model] shape")
        return ext.mhc_combine(streams, res, post, out.contiguous())
