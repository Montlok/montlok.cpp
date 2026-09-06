# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dim {self.dim}, got {x.shape[-1]}")

        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.to(dtype) * self.weight.to(dtype)


class GroupedRMSNorm(nn.Module):
    def __init__(self, dim: int, num_groups: int = 1, eps: float = 1e-5) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_groups <= 0:
            raise ValueError("num_groups must be positive")
        if dim % num_groups != 0:
            raise ValueError("dim must be divisible by num_groups")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.dim = dim
        self.num_groups = num_groups
        self.group_size = dim // num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dim {self.dim}, got {x.shape[-1]}")

        dtype = x.dtype
        shape = x.shape

        y = x.float().reshape(*shape[:-1], self.num_groups, self.group_size)
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = y.reshape(*shape)

        return y.to(dtype) * self.weight.to(dtype)


def _check() -> None:
    torch.manual_seed(0)

    x = torch.randn(2, 16, 512)

    norm = RMSNorm(512)
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()

    print("RMSNorm")
    print(f"  shape: {tuple(x.shape)} -> {tuple(y.shape)}")
    print(f"  rms_mean: {rms.mean().item():.6f}")

    gnorm = GroupedRMSNorm(512, num_groups=8)
    yg = gnorm(x)

    print("GroupedRMSNorm")
    print(f"  groups: {gnorm.num_groups}")
    print(f"  shape: {tuple(x.shape)} -> {tuple(yg.shape)}")

    x2 = torch.randn(2, 16, 512, requires_grad=True)
    loss = gnorm(x2).sum()
    loss.backward()

    print("Backward")
    print(f"  grad_norm: {x2.grad.norm().item():.6f}")

    g1 = GroupedRMSNorm(512, num_groups=1)
    r1 = RMSNorm(512)
    with torch.no_grad():
        g1.weight.copy_(r1.weight)

    diff = (g1(x) - r1(x)).abs().max().item()

    print("Equivalence")
    print(f"  grouped_1_vs_rms: {diff:.6e}")


if __name__ == "__main__":
    _check()
