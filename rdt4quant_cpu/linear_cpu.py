"""Inference-only oneDNN linear dispatch for the montlok CPU runtime."""

from __future__ import annotations

import os

import torch
from torch.nn import functional as F


def native_linear_enabled(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        os.environ.get("MONTLOK_CPP") == "1"
        and x.device.type == "cpu"
        and weight.device.type == "cpu"
        and x.dtype == torch.float32
        and weight.dtype == torch.float32
        and not torch.is_grad_enabled()
    )


def cpu_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a dense linear layer, using oneDNN's cached CPU primitive when available."""
    if native_linear_enabled(x, weight):
        from montlok_loader import load_montlok

        return load_montlok().linear(x, weight, bias)
    return F.linear(x, weight, bias)
