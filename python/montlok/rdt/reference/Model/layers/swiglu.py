# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        bias: bool = False,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if ffn_hidden <= 0:
            raise ValueError("ffn_hidden must be positive")

        self.d_model = d_model
        self.ffn_hidden = ffn_hidden

        self.w_in = nn.Linear(d_model, 2 * ffn_hidden, bias=bias)
        self.w_down = nn.Linear(ffn_hidden, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {x.shape[-1]}")

        gate, up = self.w_in(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


def _check() -> None:
    torch.manual_seed(0)

    x = torch.randn(2, 16, 512)
    ffn = SwiGLU(512, 1536)

    y = ffn(x)

    print("SwiGLU")
    print(f"  shape: {tuple(x.shape)} -> {tuple(y.shape)}")
    print(f"  params: {sum(p.numel() for p in ffn.parameters()):,}")

    x2 = torch.randn(2, 16, 512, requires_grad=True)
    loss = ffn(x2).sum()
    loss.backward()

    print(f"  grad_norm: {x2.grad.norm().item():.6f}")


if __name__ == "__main__":
    _check()
