# -*- coding: utf-8 -*-

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.layers.rope import MorphologicalRoPE, apply_rope


class MLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.nope_dim = cfg.nope_head_dim
        self.rope_dim = cfg.rope_head_dim
        self.kv_lora_rank = cfg.kv_lora_rank
        self.q_lora_rank = cfg.q_lora_rank
        self.dropout = cfg.dropout
        self.use_sdpa = bool(
            getattr(cfg, "use_sdpa_attention", True)
            and hasattr(F, "scaled_dot_product_attention")
        )

        if self.head_dim != self.nope_dim + self.rope_dim:
            raise ValueError("head_dim must equal nope_head_dim + rope_head_dim")
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError("d_model must equal n_heads * head_dim")
        if self.rope_dim % 2 != 0:
            raise ValueError("rope_head_dim must be even")
        if self.kv_lora_rank <= 0:
            raise ValueError("kv_lora_rank must be positive")
        if self.q_lora_rank < 0:
            raise ValueError("q_lora_rank must be non-negative")

        if self.q_lora_rank > 0:
            self.q_down = nn.Linear(self.d_model, self.q_lora_rank, bias=False)
            self.q_norm = nn.LayerNorm(self.q_lora_rank)
            self.q_up = nn.Linear(
                self.q_lora_rank,
                self.n_heads * self.head_dim,
                bias=False,
            )
        else:
            self.q_proj = nn.Linear(
                self.d_model,
                self.n_heads * self.head_dim,
                bias=False,
            )

        self.kv_down = nn.Linear(self.d_model, self.kv_lora_rank, bias=False)
        self.kv_norm = nn.LayerNorm(self.kv_lora_rank)
        self.kv_up = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.nope_dim + self.head_dim),
            bias=False,
        )

        self.k_rope_proj = nn.Linear(self.d_model, self.rope_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        self.rope = MorphologicalRoPE(
            rope_dim=self.rope_dim,
            theta=cfg.rope_theta,
            max_morph_depth=cfg.max_morph_depth,
            use_morphological=cfg.use_morphological_rope,
        )

        self.scale = 1.0 / math.sqrt(self.head_dim)

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
        if x.ndim != 3:
            raise ValueError("x must have shape [B, L, d_model]")

        bsz, seq_len, dim = x.shape

        if dim != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {dim}")

        if attn_mask is not None and attn_mask.shape != (bsz, seq_len):
            raise ValueError("attn_mask must have shape [B, L]")

        if word_pos is not None and word_pos.shape != (bsz, seq_len):
            raise ValueError("word_pos must have shape [B, L]")

        if morph_depth is not None and morph_depth.shape != (bsz, seq_len):
            raise ValueError("morph_depth must have shape [B, L]")

        q = self._project_q(x)
        k, v = self._project_kv(x)

        q_nope, q_rope = q.split([self.nope_dim, self.rope_dim], dim=-1)
        k_nope, k_rope = k.split([self.nope_dim, self.rope_dim], dim=-1)

        cos, sin = self.rope(
            seq_len,
            word_pos=word_pos,
            morph_depth=morph_depth,
            device=x.device,
            pos_offset=pos_offset,
        )

        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        if cache is not None:
            if not causal:
                raise ValueError("cached MLA decode requires causal=True")
            past_len = cache.length
            k, v = cache.append(k, v)
            out = self._attention_cached(q, k, v, past_len=past_len)
            out = out.transpose(1, 2).reshape(
                bsz, seq_len, self.n_heads * self.head_dim
            )
            return self.o_proj(out)

        out = self._attention(q, k, v, attn_mask=attn_mask, causal=causal)
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.n_heads * self.head_dim)
        out = self.o_proj(out)

        if attn_mask is not None:
            out = out * attn_mask.to(dtype=out.dtype).unsqueeze(-1)

        return out

    def _attention_cached(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_len: int,
    ) -> torch.Tensor:
        """Causal attention of ``m`` new queries over ``past_len + m`` keys.

        Query ``i`` (absolute position ``past_len + i``) attends keys
        ``0 .. past_len + i``. For single-token decode (``m == 1``) the mask is
        all-ones, matching the last row of the full causal forward exactly.
        """

        q_len = q.shape[-2]
        k_len = k.shape[-2]
        device = q.device

        rows = torch.arange(q_len, device=device).unsqueeze(-1) + past_len
        cols = torch.arange(k_len, device=device).unsqueeze(0)
        allow = (cols <= rows).view(1, 1, q_len, k_len)

        if self.use_sdpa:
            dropout_p = self.dropout if self.training and self.dropout > 0 else 0.0
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=allow,
                dropout_p=dropout_p,
                is_causal=False,
                scale=self.scale,
            )

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~allow, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        return torch.matmul(attn, v)

    def _project_q(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        if self.q_lora_rank > 0:
            q = self.q_up(self.q_norm(self.q_down(x)))
        else:
            q = self.q_proj(x)

        return q.reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _project_kv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape

        kv_latent = self.kv_norm(self.kv_down(x))
        kv = self.kv_up(kv_latent)
        kv = kv.reshape(
            bsz,
            seq_len,
            self.n_heads,
            self.nope_dim + self.head_dim,
        ).transpose(1, 2)

        k_nope, v = kv.split([self.nope_dim, self.head_dim], dim=-1)

        k_rope = self.k_rope_proj(x)
        k_rope = k_rope.unsqueeze(1).expand(
            bsz,
            self.n_heads,
            seq_len,
            self.rope_dim,
        )

        k = torch.cat([k_nope, k_rope], dim=-1)
        return k, v

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        causal: bool,
    ) -> torch.Tensor:
        if self.use_sdpa:
            return self._attention_sdpa(q, k, v, attn_mask=attn_mask, causal=causal)

        return self._attention_math(q, k, v, attn_mask=attn_mask, causal=causal)

    def _attention_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        causal: bool,
    ) -> torch.Tensor:
        bsz, _heads, q_len, _dim = q.shape
        k_len = k.shape[-2]
        dropout_p = self.dropout if self.training and self.dropout > 0 else 0.0

        if attn_mask is None:
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=self.scale,
            )

        key_mask = attn_mask.to(device=q.device, dtype=torch.bool).view(
            bsz,
            1,
            1,
            k_len,
        )
        if causal:
            causal_mask = torch.ones(
                q_len,
                k_len,
                dtype=torch.bool,
                device=q.device,
            ).tril()
            sdpa_mask = key_mask & causal_mask.view(1, 1, q_len, k_len)
        else:
            sdpa_mask = key_mask

        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=self.scale,
        )

    def _attention_math(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        causal: bool,
    ) -> torch.Tensor:
        bsz, _heads, q_len, _dim = q.shape
        k_len = k.shape[-2]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            mask = torch.triu(
                torch.ones(q_len, k_len, dtype=torch.bool, device=q.device),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        if attn_mask is not None:
            key_mask = attn_mask.to(dtype=torch.bool).view(bsz, 1, 1, k_len)
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)

        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        return torch.matmul(attn, v)


def _check() -> None:
    from Model.config import tiny_config

    torch.manual_seed(0)

    cfg = tiny_config()
    mla = MLA(cfg)
    mla.eval()

    bsz, seq_len = 2, 16
    x = torch.randn(bsz, seq_len, cfg.d_model)

    word_pos = torch.arange(seq_len).unsqueeze(0).expand(bsz, seq_len)
    morph_depth = torch.zeros(bsz, seq_len, dtype=torch.long)
    attn_mask = torch.ones(bsz, seq_len, dtype=torch.long)

    y = mla(
        x,
        word_pos=word_pos,
        morph_depth=morph_depth,
        attn_mask=attn_mask,
    )

    print("MLA")
    print(f"  shape: {tuple(x.shape)} -> {tuple(y.shape)}")
    print(f"  params: {sum(p.numel() for p in mla.parameters()):,}")

    full_kv = cfg.n_heads * cfg.head_dim * 2
    ratio = cfg.kv_lora_rank / full_kv
    print(f"  kv_cache: {full_kv}/token -> {cfg.kv_lora_rank}/token ({ratio:.1%})")

    x2 = torch.randn(bsz, seq_len, cfg.d_model, requires_grad=True)
    loss = mla(
        x2,
        word_pos=word_pos,
        morph_depth=morph_depth,
        attn_mask=attn_mask,
    ).sum()
    loss.backward()
    print(f"  grad_norm: {x2.grad.norm().item():.6f}")

    x3 = torch.randn(1, 8, cfg.d_model)
    wp = torch.arange(8).unsqueeze(0)
    md = torch.zeros(1, 8, dtype=torch.long)

    out_full = mla(x3, word_pos=wp, morph_depth=md)
    x3_mod = x3.clone()
    x3_mod[0, 5:] += 10.0
    out_mod = mla(x3_mod, word_pos=wp, morph_depth=md)

    diff = (out_full[0, :5] - out_mod[0, :5]).abs().max().item()
    print(f"  causal_diff_before_changed_span: {diff:.6e}")


if __name__ == "__main__":
    _check()
