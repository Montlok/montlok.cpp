"""CPU-optimised RMSNorm / GroupedRMSNorm.

Drop-in subclasses of the upstream ``Model.layers.rmsnorm`` modules (same
parameters, same state dict). With ``MONTLOK_CPP=1`` the forward is one fused
C++ op (``montlok_cpp.rms_norm``) instead of the seven tiny PyTorch ops of the
reference; the arithmetic is done in double precision and rounded to float32
once. ``install_cpu_norms(model)`` swaps the class of every upstream norm
instance in a built model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch import nn

NATIVE = Path(__file__).resolve().parents[1] / "rdt4quant" / "native"
if str(NATIVE) not in sys.path:
    sys.path.insert(0, str(NATIVE))

from Model.layers.rmsnorm import GroupedRMSNorm, RMSNorm  # noqa: E402

from mamba3_cpu import cpp_backend  # noqa: E402


def norm_cpp_enabled(x: torch.Tensor) -> bool:
    return (
        cpp_backend() is not None
        and os.environ.get("MONTLOK_CPP_LAYERS") != "0"
        and x.dtype == torch.float32
        and x.device.type == "cpu"
        and not torch.is_grad_enabled()
    )


def _fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, groups: int) -> torch.Tensor:
    from montlok_loader import load_montlok

    return load_montlok().rms_norm(x, weight, eps, groups)


class RMSNormCPU(RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not norm_cpp_enabled(x):
            return super().forward(x)
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dim {self.dim}, got {x.shape[-1]}")
        return _fused_rms_norm(x, self.weight, self.eps, 1)


class GroupedRMSNormCPU(GroupedRMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not norm_cpp_enabled(x):
            return super().forward(x)
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dim {self.dim}, got {x.shape[-1]}")
        return _fused_rms_norm(x, self.weight, self.eps, self.num_groups)


def install_cpu_norms(module: nn.Module) -> int:
    """Re-class every upstream RMSNorm/GroupedRMSNorm under ``module``.

    Parameters and buffers are shared with the upstream class, so state dicts load unchanged.
    Returns the number of instances swapped.
    """
    swapped = 0
    for child in module.modules():
        if type(child) is RMSNorm:
            child.__class__ = RMSNormCPU
            swapped += 1
        elif type(child) is GroupedRMSNorm:
            child.__class__ = GroupedRMSNormCPU
            swapped += 1
    return swapped
