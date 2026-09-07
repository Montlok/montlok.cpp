"""CPU-optimised per-token layers: RMSNorm / GroupedRMSNorm, LayerNorm, SwiGLU.

Drop-in subclasses of the upstream modules (same parameters, same state dict).
With ``MONTLOK_CPP=1`` each forward is one fused C++ op from ``montlok.cpp``
instead of the handful of tiny PyTorch ops of the reference (an RMSNorm is
seven, a SwiGLU gate two); the arithmetic is done in double precision and
rounded to float32 once. ``install_cpu_layers(model)`` swaps the class of
every such instance in a built model.

``MONTLOK_CPP_LAYERS=0`` keeps these layers on the upstream forward, for A/B
checks.
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
from Model.layers.swiglu import SwiGLU  # noqa: E402

from mamba3_cpu import cpp_backend  # noqa: E402


def layers_cpp_enabled(x: torch.Tensor) -> bool:
    return (
        cpp_backend() is not None
        and os.environ.get("MONTLOK_CPP_LAYERS") != "0"
        and x.dtype == torch.float32
        and x.device.type == "cpu"
        and not torch.is_grad_enabled()
    )


def _ext():
    from montlok_loader import load_montlok

    return load_montlok()


class RMSNormCPU(RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not layers_cpp_enabled(x):
            return super().forward(x)
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dim {self.dim}, got {x.shape[-1]}")
        return _ext().rms_norm(x, self.weight, self.eps, 1)


class GroupedRMSNormCPU(GroupedRMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not layers_cpp_enabled(x):
            return super().forward(x)
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dim {self.dim}, got {x.shape[-1]}")
        return _ext().rms_norm(x, self.weight, self.eps, self.num_groups)


class LayerNormCPU(nn.LayerNorm):
    """``torch.nn.LayerNorm`` over a single trailing dimension in one op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not layers_cpp_enabled(x) or len(self.normalized_shape) != 1 or self.weight is None:
            return super().forward(x)
        if x.shape[-1] != self.normalized_shape[0]:
            raise ValueError(f"expected last dim {self.normalized_shape[0]}, got {x.shape[-1]}")
        bias = self.bias if self.bias is not None else x.new_empty(0)
        return _ext().layer_norm(x, self.weight, bias, self.eps)


class SwiGLUCPU(SwiGLU):
    """Upstream SwiGLU with ``silu(gate) * up`` as one op between the two projections."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not layers_cpp_enabled(x):
            return super().forward(x)
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {x.shape[-1]}")
        return _ext().swiglu_mlp(
            x,
            self.w_in.weight,
            self.w_in.bias,
            self.w_down.weight,
            self.w_down.bias,
        )


_SWAPS = {
    RMSNorm: RMSNormCPU,
    GroupedRMSNorm: GroupedRMSNormCPU,
    nn.LayerNorm: LayerNormCPU,
    SwiGLU: SwiGLUCPU,
}


def install_cpu_layers(module: nn.Module) -> int:
    """Re-class every upstream RMSNorm/GroupedRMSNorm/LayerNorm/SwiGLU under ``module``.

    Parameters and buffers are shared with the upstream class, so state dicts
    load unchanged. Returns the number of instances swapped.
    """
    swapped = 0
    for child in module.modules():
        replacement = _SWAPS.get(type(child))
        if replacement is not None:
            child.__class__ = replacement
            swapped += 1
    return swapped
