"""CPU-optimised Multi-head Latent Attention (MLA) forward.

``MLACPU`` is a drop-in subclass of the upstream ``Model.layers.mla.MLA`` (same
parameters, same state dict). For the plain full-sequence case (no cache, no
padding mask, token-index RoPE) and ``MONTLOK_CPP=1`` it

* caches the RoPE cos/sin tables per sequence length instead of rebuilding
  them (arange, mul, cat, cos, sin) on every call,
* rotates ``q`` in place and assembles ``k = cat(k_nope, rope(k_rope))`` in one
  C++ op (``montlok_cpp.mla_rope_qk``) instead of ~15 split/cat/mul ops,
* feeds ``v`` to scaled_dot_product_attention as a view of the kv_up output.

The attention itself is the same ``F.scaled_dot_product_attention`` call as
upstream; the rotary arithmetic is done in double and rounded to float32 once.
Anything else (KV cache, attention masks, morphological RoPE, training) falls
back to the upstream forward.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

NATIVE = Path(__file__).resolve().parents[1] / "rdt4quant" / "native"
if str(NATIVE) not in sys.path:
    sys.path.insert(0, str(NATIVE))

from Model.layers.mla import MLA  # noqa: E402

from mamba3_cpu import cpp_backend  # noqa: E402


def mla_cpp_enabled() -> bool:
    return cpp_backend() is not None and os.environ.get("MONTLOK_CPP_LAYERS") != "0"


class MLACPU(MLA):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._rope_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _rope_tables(self, seq_len: int, pos_offset: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (seq_len, pos_offset)
        tables = self._rope_cache.get(key)
        if tables is None:
            cos, sin = self.rope(seq_len, device=device, pos_offset=pos_offset)
            tables = (cos.contiguous(), sin.contiguous())
            if len(self._rope_cache) > 8:
                self._rope_cache.clear()
            self._rope_cache[key] = tables
        return tables

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
        if (
            not mla_cpp_enabled()
            or cache is not None
            or attn_mask is not None
            or word_pos is not None
            or morph_depth is not None
            or self.training
            or x.dtype != torch.float32
            or x.device.type != "cpu"
            or torch.is_grad_enabled()
        ):
            return super().forward(
                x,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attn_mask,
                causal=causal,
                cache=cache,
                pos_offset=pos_offset,
            )
        if x.ndim != 3:
            raise ValueError("x must have shape [B, L, d_model]")
        bsz, seq_len, dim = x.shape
        if dim != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {dim}")

        from montlok_loader import load_montlok

        ext = load_montlok()
        if self.q_lora_rank > 0:
            q = self.q_up(self.q_norm(self.q_down(x)))
        else:
            q = self.q_proj(x)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim)
        kv = self.kv_up(self.kv_norm(self.kv_down(x)))
        kv = kv.view(bsz, seq_len, self.n_heads, self.nope_dim + self.head_dim)
        k_rope = self.k_rope_proj(x)
        cos, sin = self._rope_tables(seq_len, pos_offset, x.device)
        # Rotates q's rope lanes in place (q is a fresh projection output).
        k = ext.mla_rope_qk(q, kv, k_rope, cos, sin, self.nope_dim)
        v = kv[..., self.nope_dim:]

        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            is_causal=causal,
            scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(out)
