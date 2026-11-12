# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn

from Model.layers.mla import MLA
from Model.layers.mamba3_layer import Mamba3Layer
from Model.layers.mixture_lora_ffn import MixtureLoRAFFN
from Model.layers.rmsnorm import RMSNorm
from Model.layers.swiglu import SwiGLU


def _build_ffn(cfg):
    """Recurrent-block FFN: step-aware Mixture-of-LoRAs when ``use_mol``, else the
    plain shared SwiGLU. Prelude/coda (``StandardBlock``) always use plain SwiGLU
    -- they are non-recurrent, so step-aware expert routing has no step to
    condition on."""
    if getattr(cfg, "use_mol", False):
        return MixtureLoRAFFN(
            cfg.d_model,
            cfg.ffn_hidden,
            cfg.mol_experts,
            cfg.mol_rank,
            cfg.recurrence_step_table,
            step_aware=cfg.mol_step_aware,
            router_temp=cfg.mol_router_temp,
            top_k=cfg.mol_top_k,
            init_std=cfg.init_std,
            rmsnorm_eps=cfg.rmsnorm_eps,
        )
    return SwiGLU(cfg.d_model, cfg.ffn_hidden)


def _apply_ffn(ffn, x, step):
    """Forward a ``_build_ffn`` FFN, threading the recurrent step index only into
    a MixtureLoRAFFN (plain SwiGLU has no ``step`` kwarg and ignores it)."""
    if isinstance(ffn, MixtureLoRAFFN):
        return ffn(x, step=step)
    return ffn(x)


class StandardBlock(nn.Module):
    def __init__(self, cfg, layer_idx: int | None = None):
        super().__init__()

        self.layer_idx = layer_idx

        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)
        self.attn = MLA(cfg)

        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(
        self,
        x: torch.Tensor,
        word_pos: torch.Tensor | None = None,
        morph_depth: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        causal: bool = True,
        cache=None,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.attn_norm(x),
            word_pos=word_pos,
            morph_depth=morph_depth,
            attn_mask=attn_mask,
            causal=causal,
            cache=cache,
            pos_offset=pos_offset,
        )
        x = x + self.ffn(self.ffn_norm(x))
        return x


class AttnSubLayer(nn.Module):
    def __init__(self, cfg, layer_idx: int | None = None):
        super().__init__()

        self.layer_idx = layer_idx

        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)
        self.attn = MLA(cfg)

        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)
        self.ffn = _build_ffn(cfg)

    def forward(
        self,
        x: torch.Tensor,
        word_pos: torch.Tensor | None = None,
        morph_depth: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        causal: bool = True,
        cache=None,
        pos_offset: int = 0,
        step: int | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.attn_norm(x),
            word_pos=word_pos,
            morph_depth=morph_depth,
            attn_mask=attn_mask,
            causal=causal,
            cache=cache,
            pos_offset=pos_offset,
        )
        x = x + _apply_ffn(self.ffn, self.ffn_norm(x), step)
        return x


class MambaSubLayer(nn.Module):
    def __init__(self, cfg, layer_idx: int | None = None):
        super().__init__()

        self.layer_idx = layer_idx

        self.mamba = Mamba3Layer(cfg, layer_idx=layer_idx)
        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)
        self.ffn = _build_ffn(cfg)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        cache=None,
        step: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        x = self.mamba(x, attn_mask=attn_mask, cache=cache)
        x = x + _apply_ffn(self.ffn, self.ffn_norm(x), step)

        if attn_mask is not None:
            x = x * attn_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)

        return x


class RecurrentBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.layer_types = self._interleave(
            cfg.mamba_per_block,
            cfg.attn_per_block,
        )

        self.layers = nn.ModuleList(
            self._make_layer(cfg, kind, idx)
            for idx, kind in enumerate(self.layer_types)
        )

    @staticmethod
    def _interleave(n_mamba: int, n_attn: int) -> list[str]:
        if n_mamba < 0 or n_attn < 0:
            raise ValueError("layer counts must be non-negative")

        total = n_mamba + n_attn

        if total <= 0:
            raise ValueError("recurrent block cannot be empty")

        if n_attn == 0:
            return ["mamba"] * n_mamba

        if n_mamba == 0:
            return ["attn"] * n_attn

        result: list[str] = []
        m_used = 0
        a_used = 0

        for pos in range(total):
            target_attn = round((pos + 1) * n_attn / total)

            if a_used < target_attn and a_used < n_attn:
                result.append("attn")
                a_used += 1
            elif m_used < n_mamba:
                result.append("mamba")
                m_used += 1
            else:
                result.append("attn")
                a_used += 1

        return result

    @staticmethod
    def _make_layer(cfg, kind: str, layer_idx: int) -> nn.Module:
        if kind == "mamba":
            return MambaSubLayer(cfg, layer_idx=layer_idx)

        if kind == "attn":
            return AttnSubLayer(cfg, layer_idx=layer_idx)

        raise ValueError(f"unknown layer type: {kind}")

    def forward(
        self,
        x: torch.Tensor,
        word_pos: torch.Tensor | None = None,
        morph_depth: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        causal: bool = True,
        step: int | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attn_mask,
                causal=causal,
                step=step,
            )

        return x


def _check() -> None:
    from Model.config import tiny_config

    torch.manual_seed(0)

    cfg = tiny_config()

    bsz, seq_len = 2, 16
    x = torch.randn(bsz, seq_len, cfg.d_model)

    word_pos = torch.arange(seq_len).unsqueeze(0).expand(bsz, seq_len)
    morph_depth = torch.zeros(bsz, seq_len, dtype=torch.long)

    block = StandardBlock(cfg)
    y = block(
        x,
        word_pos=word_pos,
        morph_depth=morph_depth,
    )

    print("StandardBlock")
    print(f"  shape: {tuple(x.shape)} -> {tuple(y.shape)}")

    recurrent = RecurrentBlock(cfg)
    y2 = recurrent(
        x,
        word_pos=word_pos,
        morph_depth=morph_depth,
    )

    print("RecurrentBlock")
    print(f"  layer_types: {recurrent.layer_types}")
    print(f"  shape: {tuple(x.shape)} -> {tuple(y2.shape)}")
    print(f"  params: {sum(p.numel() for p in recurrent.parameters()):,}")

    x2 = torch.randn(bsz, seq_len, cfg.d_model, requires_grad=True)
    loss = recurrent(
        x2,
        word_pos=word_pos,
        morph_depth=morph_depth,
    ).sum()
    loss.backward()

    print(f"  grad_norm: {x2.grad.norm().item():.6f}")


if __name__ == "__main__":
    _check()
