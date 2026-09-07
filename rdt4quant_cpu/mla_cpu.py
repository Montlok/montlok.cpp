"""CPU-optimised Multi-head Latent Attention (MLA) forward.

``MLACPU`` is a drop-in subclass of the upstream ``Model.layers.mla.MLA`` (same
parameters, same state dict). For the plain full-sequence case (no cache, no
padding mask, token-index RoPE) and ``MONTLOK_CPP=1`` it

* caches the RoPE cos/sin tables per sequence length instead of rebuilding
  them (arange, mul, cat, cos, sin) on every call,
* runs ``q_proj``, ``kv_down`` and ``k_rope_proj`` as one GEMM over the shared
  input (the weights are stacked once and cached per parameter version),
* rotates ``q`` in place and assembles ``k = cat(k_nope, rope(k_rope))`` in one
  C++ op (``montlok_cpp.mla_rope_qk``) instead of ~15 split/cat/mul ops,
* feeds ``v`` to scaled_dot_product_attention as a view of the kv_up output.

``forward_last(x, n_last)`` returns the attention output of the trailing
``n_last`` positions only: keys and values are projected for the whole
sequence, queries, attention and the output projection for those positions.
Row for row this is the same arithmetic as the full forward.

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
        self._in_proj_cache: dict[bool, tuple[tuple, torch.Tensor]] = {}

    def _in_proj_weight(self, with_q: bool) -> torch.Tensor:
        """Rows of ``[q_proj;] kv_down; k_rope_proj`` stacked into one weight.

        The three input projections read the same activations, so one GEMM
        replaces three launches. Cached per parameter version (``load_state_dict``
        copies in place and bumps ``_version``).
        """
        weights: tuple[torch.Tensor, ...] = (
            self.kv_down.weight,
            self.k_rope_proj.weight,
        )
        if with_q:
            weights = (self.q_proj.weight, *weights)
        key = tuple((w.data_ptr(), w._version) for w in weights)
        cached = self._in_proj_cache.get(with_q)
        if cached is None or cached[0] != key:
            with torch.no_grad():
                cached = (key, torch.cat(weights, 0).detach().contiguous())
            self._in_proj_cache[with_q] = cached
        return cached[1]

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

    def _fast_path(
        self,
        x: torch.Tensor,
        word_pos,
        morph_depth,
        attn_mask,
        cache,
    ) -> bool:
        return not (
            not mla_cpp_enabled()
            or cache is not None
            or attn_mask is not None
            or word_pos is not None
            or morph_depth is not None
            or self.training
            or x.dtype != torch.float32
            or x.device.type != "cpu"
            or torch.is_grad_enabled()
        )

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
        if not self._fast_path(x, word_pos, morph_depth, attn_mask, cache):
            return super().forward(
                x,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attn_mask,
                causal=causal,
                cache=cache,
                pos_offset=pos_offset,
            )
        return self._attend(x, x.shape[1], causal, pos_offset)

    def forward_last(self, x: torch.Tensor, n_last: int = 1, pos_offset: int = 0) -> torch.Tensor:
        """Causal attention output ``[B, n_last, d_model]`` of the trailing positions."""
        if x.ndim != 3:
            raise ValueError("x must have shape [B, L, d_model]")
        if not 1 <= n_last <= x.shape[1]:
            raise ValueError("n_last must be between 1 and the sequence length")
        if not self._fast_path(x, None, None, None, None):
            return super().forward(x, causal=True, pos_offset=pos_offset)[:, -n_last:]
        return self._attend(x, n_last, True, pos_offset)

    def _project_q_rows(self, x_rows: torch.Tensor) -> torch.Tensor:
        if self.q_lora_rank > 0:
            return self.q_up(self.q_norm(self.q_down(x_rows)))
        return self.q_proj(x_rows)

    def _attend(self, x: torch.Tensor, n_last: int, causal: bool, pos_offset: int) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        if dim != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {dim}")

        from montlok_loader import load_montlok

        ext = load_montlok()
        fuse_q = self.q_lora_rank == 0 and n_last == seq_len
        projected = F.linear(x, self._in_proj_weight(fuse_q))
        q_width = self.n_heads * self.head_dim if fuse_q else 0
        if fuse_q:
            q = projected[..., :q_width]
        else:
            q = self._project_q_rows(x if n_last == seq_len else x[:, seq_len - n_last :])
        q = q.view(bsz, n_last, self.n_heads, self.head_dim)
        kv_latent = projected[..., q_width : q_width + self.kv_lora_rank]
        k_rope = projected[..., q_width + self.kv_lora_rank :]
        kv = self.kv_up(self.kv_norm(kv_latent))
        kv = kv.view(bsz, seq_len, self.n_heads, self.nope_dim + self.head_dim)
        cos, sin = self._rope_tables(seq_len, pos_offset, x.device)
        # Rotates q's rope lanes in place (q is a fresh projection output).
        k = ext.mla_rope_qk(q, kv, k_rope, cos, sin, self.nope_dim)
        v = kv[..., self.nope_dim :]

        attn_mask = None
        is_causal = causal
        if n_last != seq_len:
            # Trailing queries: row i sits at absolute position L - n_last + i.
            # SDPA's is_causal aligns the mask to the top-left corner, so a
            # single trailing query attends every key and several trailing
            # queries need an explicit bottom-right aligned mask.
            is_causal = False
            if causal and n_last > 1:
                rows = torch.arange(n_last, device=x.device).unsqueeze(-1) + (seq_len - n_last)
                cols = torch.arange(seq_len, device=x.device).unsqueeze(0)
                attn_mask = (cols <= rows).view(1, 1, n_last, seq_len)

        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(bsz, n_last, self.n_heads * self.head_dim)
        return self.o_proj(out)
