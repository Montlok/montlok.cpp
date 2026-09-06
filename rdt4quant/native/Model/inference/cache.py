# -*- coding: utf-8 -*-

"""Incremental decode caches for bit-exact autoregressive generation.

The Daffodils RDT has exactly one cross-position operation: causal MLA
attention. Every other sub-layer is per-position (mHC stream mixing, SwiGLU
FFN, RMSNorm) or a sequential recurrence (the NaiveSSM Mamba scan). Therefore
incremental decoding -- appending one token at a time while reusing frozen
earlier state -- is *numerically identical* to re-running the full forward on
the growing prefix, and these caches make that explicit.

Two primitive caches:

* :class:`MLACache` -- post-RoPE ``K``/``V`` of one attention call, grown by
  ``append``. Because each refinement pass of the Stage-2 loop is itself a
  fresh causal attention over the frozen earlier positions, every
  ``(step, layer)`` pair owns its own :class:`MLACache`.
* :class:`MambaCache` -- either the constant-size NaiveSSM recurrent state or
  one call site's isolated upstream ``InferenceParams`` object.

:class:`DecodeCache` is a flat keyed container the orchestrators
(``RDTForCausalLM`` / ``TwoStageCore``) populate lazily; keys encode the call
site (e.g. ``"prelude.0"``, ``"stage1.2"``, ``"stage2.s3.l0"``, ``"coda.1"``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class MLACache:
    """Per-attention-call key/value cache (post-RoPE, ``[B, H, T, head_dim]``)."""

    k: torch.Tensor | None = None
    v: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[-2]

    def append(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append the new tokens' ``k``/``v`` and return the full cached tensors."""

        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=-2)
            self.v = torch.cat([self.v, v], dim=-2)
        return self.k, self.v


@dataclass
class MambaCache:
    """Per-call-site recurrent state for exactly one Mamba backend.

    ``conv_window`` holds the last ``d_conv - 1`` *pre-convolution* columns
    (shape ``[B, d_inner, d_conv - 1]``); ``ssm_state`` is the selective-scan
    state ``[B, nheads, headdim, d_state]`` in float32. The ``official_*``
    fields hold an isolated upstream cache; they must never be shared between
    logical call sites because upstream indexes state by ``layer_idx``.
    """

    conv_window: torch.Tensor | None = None
    ssm_state: torch.Tensor | None = None
    backend: str | None = None
    official_inference_params: object | None = None
    official_owner: object | None = field(default=None, repr=False)
    official_batch_size: int | None = None
    official_max_seqlen: int | None = None
    official_device: torch.device | None = None
    official_dtype: torch.dtype | None = None
    official_poisoned: bool = False


@dataclass
class DecodeCache:
    """Flat keyed container of per-layer caches for one decode stream."""

    mla: dict[str, MLACache] = field(default_factory=dict)
    mamba: dict[str, MambaCache] = field(default_factory=dict)
    seq_len: int = 0

    def mla_cache(self, key: str) -> MLACache:
        cache = self.mla.get(key)
        if cache is None:
            cache = self.mla[key] = MLACache()
        return cache

    def mamba_cache(self, key: str) -> MambaCache:
        cache = self.mamba.get(key)
        if cache is None:
            cache = self.mamba[key] = MambaCache()
        return cache

    @property
    def past_len(self) -> int:
        """Number of tokens already cached (0 before the prefill call)."""

        return self.seq_len
