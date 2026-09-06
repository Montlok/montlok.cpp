"""Weight-compatible, pure PyTorch CPU reference for upstream Mamba-3 SISO.

Copyright 2026 OpenAI. Modified implementation derived from the Apache-2.0
Mamba-3 reference formulas published by Dao AI Lab / GoombaLab in
state-spaces/mamba tests/ops/triton/test_mamba3_siso.py.
"""

from __future__ import annotations

import math
import os

import torch
from torch import nn
from torch.nn import functional as F


def heavy_tail_activation(value: torch.Tensor) -> torch.Tensor:
    negative = value.clamp_max(0)
    positive = value.clamp_min(0)
    return positive + torch.reciprocal(1 - negative)


class BCNorm(nn.Module):
    """The ungated last-dimension RMS norm used for Mamba-3 B and C."""

    def __init__(self, dimension: int, eps: float = 1e-5, **_: object):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(value.dtype)


def _rotary(value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    paired = value.reshape(*value.shape[:-1], -1, 2)
    left, right = paired[..., 0], paired[..., 1]
    if cosine.shape[-1] < left.shape[-1]:
        padding = left.shape[-1] - cosine.shape[-1]
        cosine = F.pad(cosine, (0, padding), value=1.0)
        sine = F.pad(sine, (0, padding), value=0.0)
    return torch.stack((left * cosine - right * sine, left * sine + right * cosine), dim=-1).reshape_as(value)


def _segment_sum(value: torch.Tensor) -> torch.Tensor:
    """Return causal exclusive segment sums for [batch, head, time]."""
    length = value.shape[-1]
    expanded = value.unsqueeze(-1).expand(*value.shape, length)
    strict_lower = torch.tril(torch.ones(length, length, device=value.device, dtype=torch.bool), diagonal=-1)
    result = torch.cumsum(expanded.masked_fill(~strict_lower, 0), dim=-2)
    lower = torch.tril(torch.ones(length, length, device=value.device, dtype=torch.bool))
    return result.masked_fill(~lower, -torch.inf)


class Mamba3CPUReference(nn.Module):
    """SISO Mamba-3 forward with upstream-compatible parameter names/shapes."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        is_outproj_norm: bool = False,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        layer_idx: int | None = None,
        device=None,
        dtype=None,
        **_: object,
    ):
        super().__init__()
        if is_mimo or is_outproj_norm:
            raise ValueError("CPU reference currently supports the trained SISO/no-outproj-norm contract only")
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.d_inner = d_model * expand
        if self.d_inner % headdim:
            raise ValueError("d_model * expand must be divisible by headdim")
        self.nheads = self.d_inner // headdim
        self.num_bc_heads = ngroups
        self.mimo_rank = 1
        self.layer_idx = layer_idx
        self.A_floor = A_floor
        split = int(d_state * rope_fraction)
        if split % 2:
            split -= 1
        self.num_rope_angles = split // 2
        factory = {"device": device, "dtype": dtype}
        width = 2 * self.d_inner + 2 * d_state * ngroups + 3 * self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(d_model, width, bias=False, **factory)
        log_low, log_high = math.log(dt_min), math.log(dt_max)
        dt = torch.exp(torch.rand(self.nheads, device=device) * (log_high - log_low) + log_low).clamp_min(dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.B_bias = nn.Parameter(torch.ones(self.nheads, 1, d_state, device=device))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, 1, d_state, device=device))
        self.B_norm = BCNorm(d_state, eps=1e-5)
        self.C_norm = BCNorm(d_state, eps=1e-5)
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **factory)

    def forward(self, inputs: torch.Tensor, **_: object) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.d_model:
            raise ValueError("Mamba3CPUReference expects [batch, length, d_model]")
        batch, length, _ = inputs.shape
        projected = self.in_proj(inputs.float())
        z, x, b_value, c_value, raw_dt, raw_a, raw_trap, raw_angles = torch.split(
            projected,
            [
                self.d_inner,
                self.d_inner,
                self.d_state * self.num_bc_heads,
                self.d_state * self.num_bc_heads,
                self.nheads,
                self.nheads,
                self.nheads,
                self.num_rope_angles,
            ],
            dim=-1,
        )
        z = z.reshape(batch, length, self.nheads, self.headdim)
        x = x.reshape(batch, length, self.nheads, self.headdim)
        b_value = self.B_norm(b_value.reshape(batch, length, self.num_bc_heads, self.d_state))
        c_value = self.C_norm(c_value.reshape(batch, length, self.num_bc_heads, self.d_state))
        if self.num_bc_heads != self.nheads:
            if self.nheads % self.num_bc_heads:
                raise ValueError("nheads must be divisible by grouped B/C heads")
            repeats = self.nheads // self.num_bc_heads
            b_value = b_value.repeat_interleave(repeats, dim=2)
            c_value = c_value.repeat_interleave(repeats, dim=2)
        q_unrotated = c_value + self.C_bias.squeeze(1).view(1, 1, self.nheads, self.d_state)
        k_unrotated = b_value + self.B_bias.squeeze(1).view(1, 1, self.nheads, self.d_state)
        dt = F.softplus(raw_dt.float() + self.dt_bias.float())
        a_value = -heavy_tail_activation(raw_a.float()).clamp_min(self.A_floor)
        adt = a_value * dt
        angles = torch.tanh(raw_angles.float()).unsqueeze(2).expand(-1, -1, self.nheads, -1) * math.pi
        cumulative = torch.cumsum(angles * dt.unsqueeze(-1), dim=1)
        cumulative = cumulative - 2 * math.pi * torch.floor(cumulative / (2 * math.pi))
        q_value = _rotary(q_unrotated, torch.cos(cumulative), torch.sin(cumulative))
        k_value = _rotary(k_unrotated, torch.cos(cumulative), torch.sin(cumulative))
        trap = torch.sigmoid(raw_trap.float())
        shifted_dt = F.pad(dt[:, 1:].transpose(1, 2), (0, 1)).transpose(1, 2)
        shifted_trap = F.pad(trap[:, 1:].transpose(1, 2), (0, 1)).transpose(1, 2)
        shifted_gamma = shifted_dt * (1 - shifted_trap)
        scale = dt * trap + shifted_gamma
        if os.environ.get("MONTLOK_CPP") == "1":
            from montlok_loader import load_montlok

            output = load_montlok().mamba3_siso_recurrent(
                q_value.float().contiguous(),
                k_value.float().contiguous(),
                x.float().contiguous(),
                adt.transpose(1, 2).float().contiguous(),
                dt.transpose(1, 2).float().contiguous(),
                raw_trap.transpose(1, 2).float().contiguous(),
                self.D.float().contiguous(),
                z.float().contiguous(),
            )
        else:
            qk_skip = (q_unrotated * k_unrotated).sum(dim=-1) * shifted_gamma
            k_scaled = k_value * scale.unsqueeze(-1)
            qk = torch.einsum("bthd,bshd->bhts", q_value, k_scaled)
            decay = torch.exp(_segment_sum(adt.transpose(1, 2)))
            causal = torch.tril(torch.ones(length, length, device=inputs.device, dtype=torch.bool))
            qk = qk.masked_fill(~causal, 0) * decay
            output = torch.einsum("bhts,bshd->bthd", qk, x)
            output = output + self.D.float().view(1, 1, self.nheads, 1) * x
            output = output - x * qk_skip.unsqueeze(-1)
            output = output * z * torch.sigmoid(z)
        return self.out_proj(output.reshape(batch, length, self.d_inner).to(self.out_proj.weight.dtype))
