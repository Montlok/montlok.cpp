# -*- coding: utf-8 -*-

"""Two-stage recurrent core for the Daffodils RDT.

Language understanding is modeled as estimating a continuous latent semantic
trajectory that runs through the context. Surface tokens are discrete, noisy
expression points; the hidden state should form a smooth trajectory toward a
global semantic target. Under this assumption the core is split into two stages:

* **Stage 1 (pure Mamba)** performs an order-preserving, lossy, morphology-aware
  compression of the *raw* context, extracting the semantic backbone. It is
  equal-length (no downsampling on the causal pretraining path).
* **Stage 2 (pure attention / MLA)** performs recurrent refinement and relational
  reasoning *on top of the Mamba-processed representation*. The Transformer never
  touches raw context.

Drift control for the refinement loop is selected by ``cfg.recurrent_drift_mode``.
The ``"mhc"`` mode sinks Manifold-Constrained Hyper-Connections (arXiv:2512.24880)
into every attention/ffn residual via :class:`MHCAttnSubLayer`; stability then
comes from the per-layer doubly-stochastic constraint rather than from any
loop-level injection or boundary normalization.

``TwoStageCore`` is a drop-in replacement for ``RecurrentCore``: same forward
signature and return contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from Model.blocks import AttnSubLayer, MambaSubLayer
from Model.layers.mhc import ManifoldHyperConnection
from Model.layers.mla import MLA
from Model.layers.rmsnorm import RMSNorm
from Model.layers.swiglu import SwiGLU


@dataclass
class TwoStageOutput:
    """Auxiliary outputs of :class:`TwoStageCore`.

    ``hidden`` is the equal-length refined representation fed to the coda /
    LM head. ``global_semantic`` is a masked mean-pool of the Stage-1 backbone
    (a diagnostic / auxiliary summary, never fed back into per-position logits,
    so causality is preserved). ``compressed`` / ``seg_ids`` are populated only
    when order-preserving downsampling is enabled.
    """

    hidden: torch.Tensor
    global_semantic: torch.Tensor
    compressed: torch.Tensor | None = None
    seg_ids: torch.Tensor | None = None


class MHCAttnSubLayer(nn.Module):
    """``AttnSubLayer`` variant with mHC sunk into each residual.

    Operates on ``n``-stream hidden states ``[B, L, n, d]``. The attention
    residual and the ffn residual each own one :class:`ManifoldHyperConnection`,
    so the write term ``H_post * F`` is redistributed by a doubly-stochastic
    cross-stream matrix at every layer instead of accumulating.
    """

    def __init__(self, cfg, layer_idx: int | None = None):
        super().__init__()

        self.layer_idx = layer_idx

        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)
        self.attn = MLA(cfg)

        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)

        self.attn_hc = ManifoldHyperConnection(
            cfg.d_model,
            n_streams=cfg.mhc_n_streams,
            sinkhorn_iters=cfg.mhc_sinkhorn_iters,
        )
        self.ffn_hc = ManifoldHyperConnection(
            cfg.d_model,
            n_streams=cfg.mhc_n_streams,
            sinkhorn_iters=cfg.mhc_sinkhorn_iters,
        )

    def forward(
        self,
        streams: torch.Tensor,
        word_pos: torch.Tensor | None = None,
        morph_depth: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        causal: bool = True,
        cache=None,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        def attn_fn(inp: torch.Tensor) -> torch.Tensor:
            return self.attn(
                self.attn_norm(inp),
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attn_mask,
                causal=causal,
                cache=cache,
                pos_offset=pos_offset,
            )

        streams = self.attn_hc(streams, attn_fn)

        def ffn_fn(inp: torch.Tensor) -> torch.Tensor:
            return self.ffn(self.ffn_norm(inp))

        streams = self.ffn_hc(streams, ffn_fn)
        return streams


class TwoStageCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        if cfg.use_act:
            raise ValueError("TwoStageCore does not support use_act=True")

        self.drift_mode = cfg.recurrent_drift_mode
        self.n_streams = cfg.mhc_n_streams
        self.inject_scale = cfg.inject_scale
        self.inject_decay = cfg.recurrent_inject_decay
        self.downsample = cfg.two_stage_downsample
        self.max_segments = cfg.two_stage_max_segments

        if self.downsample:
            raise NotImplementedError(
                "two_stage_downsample=True is not implemented for the causal "
                "two-stage core (it would leak intra-word future on a causal "
                "path); keep two_stage_downsample=False"
            )

        self.grad_ckpt_stage1 = bool(getattr(cfg, "grad_ckpt_blocks", False))
        self.grad_ckpt_stage2 = bool(getattr(cfg, "grad_ckpt_recurrent", False))

        if self.inject_scale < 0:
            raise ValueError("inject_scale must be non-negative")
        if cfg.recurrent_steps <= 0:
            raise ValueError("recurrent_steps must be positive")

        self.stage1 = nn.ModuleList(
            MambaSubLayer(cfg, layer_idx=i) for i in range(cfg.stage1_mamba_layers)
        )

        if self.drift_mode == "mhc":
            self.stage2 = nn.ModuleList(
                MHCAttnSubLayer(cfg, layer_idx=i)
                for i in range(cfg.stage2_attn_layers)
            )
        else:
            self.stage2 = nn.ModuleList(
                AttnSubLayer(cfg, layer_idx=i) for i in range(cfg.stage2_attn_layers)
            )

        if self.drift_mode in {"norm", "both"}:
            self.boundary_norm = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)
        else:
            self.boundary_norm = None

    def forward(
        self,
        e0: torch.Tensor,
        word_pos: torch.Tensor | None = None,
        morph_depth: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        causal: bool = True,
        steps: int | None = None,
        bptt_window: int | None = None,
        cache=None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, dict]:
        self._check_inputs(e0, word_pos, morph_depth, attn_mask)

        total_steps = int(steps if steps is not None else self.cfg.recurrent_steps)
        if total_steps <= 0:
            raise ValueError("steps must be positive")

        if cache is not None:
            return self._forward_cached(
                e0,
                word_pos=word_pos,
                morph_depth=morph_depth,
                total_steps=total_steps,
                cache=cache,
                pos_offset=pos_offset,
            )

        backbone = self._run_stage1(
            e0,
            word_pos=word_pos,
            morph_depth=morph_depth,
            attn_mask=attn_mask,
            causal=causal,
        )

        global_semantic = self._global_semantic(backbone, attn_mask)

        if self.drift_mode == "mhc":
            hidden = self._refine_mhc(
                backbone,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attn_mask,
                causal=causal,
                total_steps=total_steps,
                bptt_window=bptt_window,
            )
        else:
            hidden = self._refine_plain(
                backbone,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attn_mask,
                causal=causal,
                total_steps=total_steps,
                bptt_window=bptt_window,
            )

        out = TwoStageOutput(hidden=hidden, global_semantic=global_semantic)

        info = {
            "steps_used": total_steps,
            "ponder_cost": e0.new_tensor(0.0),
            "global_semantic": out.global_semantic,
        }
        return out.hidden, info

    def _forward_cached(
        self,
        e0: torch.Tensor,
        word_pos: torch.Tensor | None,
        morph_depth: torch.Tensor | None,
        total_steps: int,
        cache,
        pos_offset: int,
    ) -> tuple[torch.Tensor, dict]:
        """Incremental decode path (bit-exact with :meth:`forward`).

        Stage-1 Mamba layers carry a constant-size recurrent state; each
        ``(step, layer)`` of the Stage-2 refinement loop owns its own causal
        MLA cache, since every pass re-attends over the frozen earlier
        positions. ``attn_mask`` is omitted (pad-free generation).
        """

        backbone = e0
        for i, layer in enumerate(self.stage1):
            backbone = layer(
                backbone,
                attn_mask=None,
                cache=cache.mamba_cache(f"stage1.{i}"),
            )

        if self.drift_mode == "mhc":
            streams = self._expand(backbone)
            for step in range(total_steps):
                for li, layer in enumerate(self.stage2):
                    streams = layer(
                        streams,
                        word_pos=word_pos,
                        morph_depth=morph_depth,
                        attn_mask=None,
                        causal=True,
                        cache=cache.mla_cache(f"stage2.s{step}.l{li}"),
                        pos_offset=pos_offset,
                    )
            hidden = self._collapse(streams)
        else:
            inject = self.drift_mode in {"decay", "both"}
            hidden = backbone
            for step in range(total_steps):
                if inject:
                    scale = self.inject_scale * (self.inject_decay ** step)
                    hidden = hidden + scale * backbone
                for li, layer in enumerate(self.stage2):
                    hidden = layer(
                        hidden,
                        word_pos=word_pos,
                        morph_depth=morph_depth,
                        attn_mask=None,
                        causal=True,
                        cache=cache.mla_cache(f"stage2.s{step}.l{li}"),
                        pos_offset=pos_offset,
                    )
                if self.boundary_norm is not None:
                    hidden = self.boundary_norm(hidden)

        info = {
            "steps_used": total_steps,
            "ponder_cost": e0.new_tensor(0.0),
            "global_semantic": backbone.mean(dim=1),
        }
        return hidden, info

    def _run_stage1(
        self,
        x: torch.Tensor,
        word_pos: torch.Tensor | None,
        morph_depth: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        causal: bool,
    ) -> torch.Tensor:
        for layer in self.stage1:
            x = self._maybe_ckpt(
                layer,
                x,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attn_mask,
                causal=causal,
                enabled=self.grad_ckpt_stage1,
            )
        return x

    def _refine_plain(
        self,
        backbone: torch.Tensor,
        word_pos: torch.Tensor | None,
        morph_depth: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        causal: bool,
        total_steps: int,
        bptt_window: int | None,
    ) -> torch.Tensor:
        if bptt_window is not None:
            if bptt_window <= 0:
                raise ValueError("bptt_window must be positive")
            bptt_window = min(bptt_window, total_steps)

        inject = self.drift_mode in {"decay", "both"}
        h = backbone

        for step in range(total_steps):
            if bptt_window is not None and step < total_steps - bptt_window:
                h = h.detach()

            if inject:
                scale = self.inject_scale * (self.inject_decay ** step)
                h = h + scale * backbone

            for layer in self.stage2:
                h = self._maybe_ckpt(
                    layer,
                    h,
                    word_pos=word_pos,
                    morph_depth=morph_depth,
                    attn_mask=attn_mask,
                    causal=causal,
                    enabled=self.grad_ckpt_stage2,
                )

            if self.boundary_norm is not None:
                h = self.boundary_norm(h)

        return h

    def _refine_mhc(
        self,
        backbone: torch.Tensor,
        word_pos: torch.Tensor | None,
        morph_depth: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        causal: bool,
        total_steps: int,
        bptt_window: int | None,
    ) -> torch.Tensor:
        # Expand once, keep n streams across every step/layer, collapse once.
        # bptt_window truncates backprop through the recurrent steps: streams are
        # detached before the last ``bptt_window`` steps so activation memory
        # stays bounded during training (drop-in with RecurrentCore). Per-step
        # stability still comes from the per-layer doubly-stochastic constraint.
        if bptt_window is not None:
            if bptt_window <= 0:
                raise ValueError("bptt_window must be positive")
            bptt_window = min(bptt_window, total_steps)

        streams = self._expand(backbone)

        for step in range(total_steps):
            if bptt_window is not None and step < total_steps - bptt_window:
                streams = streams.detach()

            for layer in self.stage2:
                streams = self._maybe_ckpt(
                    layer,
                    streams,
                    word_pos=word_pos,
                    morph_depth=morph_depth,
                    attn_mask=attn_mask,
                    causal=causal,
                    enabled=self.grad_ckpt_stage2,
                )

        return self._collapse(streams)

    def _expand(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-2).expand(-1, -1, self.n_streams, -1).contiguous()

    def _collapse(self, streams: torch.Tensor) -> torch.Tensor:
        return streams.mean(dim=-2)

    @staticmethod
    def _global_semantic(
        backbone: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attn_mask is None:
            return backbone.mean(dim=1)

        mask = attn_mask.to(dtype=backbone.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (backbone * mask).sum(dim=1) / denom

    def _maybe_ckpt(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        word_pos: torch.Tensor | None,
        morph_depth: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        causal: bool,
        enabled: bool,
    ) -> torch.Tensor:
        if enabled and self.training and x.requires_grad:
            def _fn(x_in):
                return layer(
                    x_in,
                    word_pos=word_pos,
                    morph_depth=morph_depth,
                    attn_mask=attn_mask,
                    causal=causal,
                )

            return checkpoint(_fn, x, use_reentrant=False)

        return layer(
            x,
            word_pos=word_pos,
            morph_depth=morph_depth,
            attn_mask=attn_mask,
            causal=causal,
        )

    def _check_inputs(
        self,
        e0: torch.Tensor,
        word_pos: torch.Tensor | None,
        morph_depth: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
    ) -> None:
        if e0.ndim != 3:
            raise ValueError("e0 must have shape [B, L, d_model]")

        bsz, seq_len, dim = e0.shape

        if dim != self.cfg.d_model:
            raise ValueError(f"expected d_model={self.cfg.d_model}, got {dim}")

        if word_pos is not None and word_pos.shape != (bsz, seq_len):
            raise ValueError("word_pos must have shape [B, L]")

        if morph_depth is not None and morph_depth.shape != (bsz, seq_len):
            raise ValueError("morph_depth must have shape [B, L]")

        if attn_mask is not None and attn_mask.shape != (bsz, seq_len):
            raise ValueError("attn_mask must have shape [B, L]")

    # ------------------------------------------------------------------
    # Order-preserving downsampling helpers.
    #
    # Disabled by default (``two_stage_downsample=False``): on a causal
    # pretraining path, mean-pooling a word's characters into one segment would
    # leak that word's future characters into earlier positions. Retained for
    # non-causal / encoder-style use.
    # ------------------------------------------------------------------
    @staticmethod
    def _segment_mean_pool(
        x: torch.Tensor,
        seg_ids: torch.Tensor,
        max_segments: int,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, L, d_model]")
        if seg_ids.shape != x.shape[:2]:
            raise ValueError("seg_ids must have shape [B, L]")
        if max_segments <= 0:
            raise ValueError("max_segments must be positive for pooling")

        bsz, _seq_len, dim = x.shape
        device = x.device

        pooled = x.new_zeros(bsz, max_segments, dim)
        counts = x.new_zeros(bsz, max_segments, 1)

        idx = seg_ids.clamp(min=0, max=max_segments - 1).unsqueeze(-1).expand(-1, -1, dim)
        pooled.scatter_add_(1, idx, x)
        counts.scatter_add_(
            1,
            seg_ids.clamp(min=0, max=max_segments - 1).unsqueeze(-1),
            torch.ones(bsz, seg_ids.shape[1], 1, device=device, dtype=x.dtype),
        )

        return pooled / counts.clamp(min=1.0)

    @staticmethod
    def _segment_scatter_back(
        pooled: torch.Tensor,
        seg_ids: torch.Tensor,
    ) -> torch.Tensor:
        if pooled.ndim != 3:
            raise ValueError("pooled must have shape [B, S, d_model]")
        if seg_ids.ndim != 2:
            raise ValueError("seg_ids must have shape [B, L]")

        bsz, max_segments, dim = pooled.shape
        idx = seg_ids.clamp(min=0, max=max_segments - 1).unsqueeze(-1).expand(-1, -1, dim)
        return torch.gather(pooled, 1, idx)

    @staticmethod
    @torch.no_grad()
    def slice_orientation(
        hidden: torch.Tensor,
        word_pos: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Probe: mean cosine similarity of adjacent same-word hidden states.

        Validates the core assumption that hidden states within one word slice
        form a same-direction (oriented) trajectory. Returns a scalar tensor in
        ``[-1, 1]``; higher means more strongly oriented.
        """

        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [B, L, d_model]")
        if word_pos.shape != hidden.shape[:2]:
            raise ValueError("word_pos must have shape [B, L]")

        if hidden.shape[1] < 2:
            return hidden.new_tensor(0.0)

        a = hidden[:, :-1]
        b = hidden[:, 1:]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)

        same_word = word_pos[:, :-1] == word_pos[:, 1:]
        valid = same_word
        if attn_mask is not None:
            pair_mask = (attn_mask[:, :-1] > 0) & (attn_mask[:, 1:] > 0)
            valid = valid & pair_mask

        valid = valid.to(dtype=cos.dtype)
        denom = valid.sum().clamp(min=1.0)
        return (cos * valid).sum() / denom


def _check() -> None:
    from Model.config import two_stage_tiny_config

    torch.manual_seed(0)

    cfg = two_stage_tiny_config()
    core = TwoStageCore(cfg)
    core.eval()

    bsz, seq_len = 2, 16
    e0 = torch.randn(bsz, seq_len, cfg.d_model)
    word_pos = torch.arange(seq_len).unsqueeze(0).expand(bsz, seq_len)
    morph_depth = torch.zeros(bsz, seq_len, dtype=torch.long)

    h, info = core(e0, word_pos=word_pos, morph_depth=morph_depth)

    print("TwoStageCore")
    print(f"  drift_mode: {cfg.recurrent_drift_mode}")
    print(f"  shape: {tuple(e0.shape)} -> {tuple(h.shape)}")
    print(f"  steps_used: {info['steps_used']}")
    print(f"  global_semantic: {tuple(info['global_semantic'].shape)}")

    orient = TwoStageCore.slice_orientation(h, word_pos)
    print(f"  slice_orientation: {orient.item():.4f}")

    e0g = torch.randn(bsz, seq_len, cfg.d_model, requires_grad=True)
    h, _info = core(e0g, word_pos=word_pos, morph_depth=morph_depth)
    h.sum().backward()
    print(f"  grad_norm: {e0g.grad.norm().item():.6f}")


if __name__ == "__main__":
    _check()
