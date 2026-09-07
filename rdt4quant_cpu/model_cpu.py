"""CPU runtime for the trained RDT4quant Mamba-3 SISO checkpoints.

The forecast heads read the hidden state of the last position only. With the
C++ backend enabled, ``forward`` therefore evaluates the final application of
the stage-2 layer for that position alone (keys/values still cover the whole
sequence), which is row for row the arithmetic of the full pass. ``encode``
keeps returning the hidden states of every position. ``MONTLOK_CPP_TAIL=0``
disables the trailing-position path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


PIPELINES = Path(__file__).resolve().parents[1]
NATIVE = PIPELINES / "rdt4quant" / "native"
sys.path.insert(0, str(NATIVE))

from Model.config import RDTConfig  # noqa: E402
from Model.layers.rmsnorm import RMSNorm  # noqa: E402
import Model.layers.mamba3_layer as mamba_layer  # noqa: E402

from linear_cpu import cpu_linear  # noqa: E402
from mamba3_cpu import Mamba3CPUReference, cpp_backend  # noqa: E402
from mhc_cpu import ManifoldHyperConnectionCPU  # noqa: E402
from mla_cpu import MLACPU  # noqa: E402
from layers_cpu import SwiGLUCPU, install_cpu_layers  # noqa: E402

mamba_layer.OfficialMamba3 = Mamba3CPUReference

import Model.two_stage as two_stage  # noqa: E402
from Model.two_stage import TwoStageCore  # noqa: E402

# Stage-2 layers build their hyper-connections and attention from these
# module-level names; the CPU subclasses keep the upstream parameters.
two_stage.ManifoldHyperConnection = ManifoldHyperConnectionCPU
two_stage.MLA = MLACPU


def tail_enabled() -> bool:
    return cpp_backend() is not None and os.environ.get("MONTLOK_CPP_TAIL") != "0"


def _native_stage1_ready(core, inputs: torch.Tensor) -> bool:
    return (
        os.environ.get("MONTLOK_CPP_NATIVE_STAGE1") != "0"
        and os.environ.get("MONTLOK_CPP_LAYERS") != "0"
        and cpp_backend() == "fused"
        and bool(core.stage1)
        and inputs.dtype == torch.float32
        and inputs.device.type == "cpu"
        and not torch.is_grad_enabled()
        and all(
            isinstance(layer.mamba.mamba, Mamba3CPUReference) and isinstance(layer.ffn, SwiGLUCPU)
            for layer in core.stage1
        )
    )


def native_stage1(core, inputs: torch.Tensor) -> torch.Tensor:
    """Bind the full Mamba/FFN stack to one native inference call."""
    from montlok_loader import load_montlok

    packs = core.__dict__.get("_montlok_stage1_parameter_pack")
    if packs is None or os.environ.get("MONTLOK_CACHE_PARAMETER_PACKS") == "0":
        empty = inputs.new_empty(0)
        packs = []
        for layer in core.stage1:
            mixer = layer.mamba.mamba
            packs.append(
                (
                    layer.mamba.norm.weight,
                    mixer.in_proj.weight,
                    mixer.out_proj.weight,
                    mixer.B_bias,
                    mixer.C_bias,
                    mixer.B_norm.weight,
                    mixer.C_norm.weight,
                    mixer.dt_bias,
                    mixer.D,
                    layer.ffn_norm.weight,
                    layer.ffn.w_in.weight,
                    layer.ffn.w_in.bias if layer.ffn.w_in.bias is not None else empty,
                    layer.ffn.w_down.weight,
                    layer.ffn.w_down.bias if layer.ffn.w_down.bias is not None else empty,
                )
            )
        if os.environ.get("MONTLOK_CACHE_PARAMETER_PACKS") != "0":
            core.__dict__["_montlok_stage1_parameter_pack"] = packs
    first = core.stage1[0]
    mixer = first.mamba.mamba
    return load_montlok().stage1_forward(
        inputs,
        packs,
        first.mamba.norm.eps,
        first.mamba.norm.num_groups,
        first.ffn_norm.eps,
        mixer.d_state,
        mixer.num_bc_heads,
        mixer.nheads,
        mixer.headdim,
        mixer.num_rope_angles,
        mixer.B_norm.eps,
        mixer.C_norm.eps,
        mixer.A_floor,
    )


def stage2_tail(layer, streams: torch.Tensor, n_last: int = 1) -> torch.Tensor:
    """One ``MHCAttnSubLayer`` application restricted to the trailing positions.

    Attention keys/values are built from every position; queries, the attention
    output, the feed-forward sub-layer and both hyper-connection writes are
    evaluated for the last ``n_last`` positions only. Returns ``[B, n_last, n, d]``.
    """

    def attn_fn(agg: torch.Tensor) -> torch.Tensor:
        return layer.attn.forward_last(layer.attn_norm(agg), n_last)

    streams = layer.attn_hc.forward_tail(streams, attn_fn, n_last)

    def ffn_fn(agg: torch.Tensor) -> torch.Tensor:
        return layer.ffn(layer.ffn_norm(agg))

    return layer.ffn_hc(streams, ffn_fn)


def _fused_stage2_ready(applications: list, backbone: torch.Tensor) -> bool:
    """Whether every mHC residual can use the fused inference-only chain."""
    return bool(applications) and all(
        isinstance(hc, ManifoldHyperConnectionCPU) and hc._fast_path(backbone)
        for layer in applications
        for hc in (layer.attn_hc, layer.ffn_hc)
    )


def _native_stage2_ready(core, backbone: torch.Tensor) -> bool:
    if (
        os.environ.get("MONTLOK_CPP_NATIVE_STAGE2") == "0"
        or os.environ.get("MONTLOK_CPP_LAYERS") == "0"
        or len(core.stage2) != 1
    ):
        return False
    layer = core.stage2[0]
    return (
        isinstance(layer.attn, MLACPU)
        and layer.attn.q_lora_rank == 0
        and isinstance(layer.ffn, SwiGLUCPU)
        and _fused_stage2_ready([layer], backbone)
    )


def native_stage2_tail(layer, backbone: torch.Tensor, depth: int) -> torch.Tensor:
    """Bind one recurrent mHC layer to the complete native stage-2 tail loop."""
    from montlok_loader import load_montlok

    attn = layer.attn
    cache_key = (backbone.shape[1], backbone.device.type, backbone.device.index)
    cache = layer.__dict__.get("_montlok_stage2_parameter_packs")
    packed = None if cache is None else cache.get(cache_key)
    if packed is None or os.environ.get("MONTLOK_CACHE_PARAMETER_PACKS") == "0":
        def hc_pack(hc):
            return (
                hc.dyn_norm.weight,
                hc._projection_block(),
                hc.pre_bias,
                hc.post_bias,
                hc.res_bias,
                hc.pre_alpha,
                hc.post_alpha,
                hc.res_alpha,
            )

        cos, sin = attn._rope_tables(backbone.shape[1], 0, backbone.device)
        empty = backbone.new_empty(0)
        packed = (
            hc_pack(layer.attn_hc),
            (
                attn._in_proj_weight(True),
                attn._in_proj_weight(False),
                attn.q_proj.weight,
                attn.kv_norm.weight,
                attn.kv_norm.bias if attn.kv_norm.bias is not None else empty,
                attn.kv_up.weight,
                attn.o_proj.weight,
                cos,
                sin,
            ),
            hc_pack(layer.ffn_hc),
            (
                layer.ffn.w_in.weight,
                layer.ffn.w_in.bias if layer.ffn.w_in.bias is not None else empty,
                layer.ffn.w_down.weight,
                layer.ffn.w_down.bias if layer.ffn.w_down.bias is not None else empty,
            ),
        )
        if os.environ.get("MONTLOK_CACHE_PARAMETER_PACKS") != "0":
            if cache is None:
                cache = {}
                layer.__dict__["_montlok_stage2_parameter_packs"] = cache
            cache[cache_key] = packed
    attn_hc_pack, mla_pack, ffn_hc_pack, ffn_pack = packed
    return load_montlok().stage2_tail_forward(
        backbone,
        depth,
        layer.attn_hc.n_streams,
        attn_hc_pack,
        layer.attn_hc.dyn_norm.eps,
        layer.attn_hc.sinkhorn_iters,
        layer.attn_norm.weight,
        layer.attn_norm.eps,
        mla_pack,
        attn.kv_norm.eps,
        attn.n_heads,
        attn.head_dim,
        attn.nope_dim,
        attn.kv_lora_rank,
        attn.scale,
        ffn_hc_pack,
        layer.ffn_hc.dyn_norm.eps,
        layer.ffn_hc.sinkhorn_iters,
        layer.ffn_norm.weight,
        layer.ffn_norm.eps,
        ffn_pack,
    )


def fused_stage2(applications: list, backbone: torch.Tensor, n_last: int | None = None) -> torch.Tensor:
    """Run a prepared mHC chain and collapse it without redundant memory passes.

    The first stream expansion is represented as a broadcast view in C++. Each
    residual write is fused with preparation of the next residual. When
    ``n_last`` is set, only those queries and writes are evaluated in the final
    attention/FFN application.
    """
    if n_last is not None and not 1 <= n_last <= backbone.shape[1]:
        raise ValueError("n_last must be between 1 and the sequence length")
    if not applications:
        return backbone if n_last is None else backbone[:, -n_last:]
    if not _fused_stage2_ready(applications, backbone):
        raise RuntimeError("the fused stage-2 chain requires inference-mode float32 CPU mHC layers")

    first_hc = applications[0].attn_hc
    streams, prepared = first_hc._prepare(
        backbone,
        allow_broadcast=True,
        output_norm=applications[0].attn_norm,
    )
    for index, layer in enumerate(applications):
        final_application = index == len(applications) - 1
        agg = prepared[0]

        if final_application and n_last is not None:
            attn_out = layer.attn.forward_last(agg, n_last)
            streams = layer.attn_hc._combine_tail(streams, prepared, attn_out, n_last)
            streams, prepared = layer.ffn_hc._prepare(streams, output_norm=layer.ffn_norm)
            ffn_out = layer.ffn(prepared[0])
            return layer.ffn_hc._combine_collapse(streams, prepared, ffn_out)

        attn_out = layer.attn(agg, causal=True)
        streams, prepared = layer.attn_hc._combine_prepare(
            streams,
            prepared,
            attn_out,
            layer.ffn_hc,
            output_norm=layer.ffn_norm,
        )
        ffn_out = layer.ffn(prepared[0])
        if final_application:
            return layer.ffn_hc._combine_collapse(streams, prepared, ffn_out)
        streams, prepared = layer.ffn_hc._combine_prepare(
            streams,
            prepared,
            ffn_out,
            applications[index + 1].attn_hc,
            output_norm=applications[index + 1].attn_norm,
        )

    raise AssertionError("unreachable empty stage-2 chain")


class RDT4QuantCPU(nn.Module):
    def __init__(self, n_features: int, config: dict):
        super().__init__()
        self.cfg = RDTConfig(**config["model"])
        if not self.cfg.use_official_mamba:
            raise ValueError("the CPU runtime requires a checkpoint trained with use_official_mamba=True")
        self.projection = nn.Linear(n_features, self.cfg.d_model)
        self.input_norm = nn.LayerNorm(self.cfg.d_model)
        self.prelude, self.coda = nn.ModuleList(), nn.ModuleList()
        self.recurrent = TwoStageCore(self.cfg)
        if any(not isinstance(layer.mamba.mamba, Mamba3CPUReference) for layer in self.recurrent.stage1):
            raise RuntimeError("CPU reference backend was not installed explicitly")
        if self.cfg.recurrent_drift_mode == "mhc" and any(
            not isinstance(layer.attn_hc, ManifoldHyperConnectionCPU) for layer in self.recurrent.stage2
        ):
            raise RuntimeError("CPU mHC backend was not installed explicitly")
        self.final_norm = RMSNorm(self.cfg.d_model, eps=self.cfg.rmsnorm_eps)
        self.head = nn.Linear(self.cfg.d_model, len(config["horizons_minutes"]) * 3)
        self.n_horizons = len(config["horizons_minutes"])
        install_cpu_layers(self)

    def _clear_native_parameter_packs(self) -> None:
        self.recurrent.__dict__.pop("_montlok_stage1_parameter_pack", None)
        for layer in self.recurrent.stage2:
            layer.__dict__.pop("_montlok_stage2_parameter_packs", None)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._clear_native_parameter_packs()
        return result

    def train(self, mode: bool = True):
        result = super().train(mode)
        self._clear_native_parameter_packs()
        return result

    def encode(self, inputs: torch.Tensor, depth: int):
        projected = cpu_linear(inputs.float(), self.projection.weight, self.projection.bias)
        embedded = self.input_norm(projected)
        return self.recurrent(embedded, steps=depth, bptt_window=None, causal=True)

    def _tail_path(self, embedded: torch.Tensor) -> bool:
        return (
            tail_enabled()
            and self.recurrent.drift_mode == "mhc"
            and embedded.dtype == torch.float32
            and embedded.device.type == "cpu"
            and not torch.is_grad_enabled()
            and not self.training
        )

    def _hidden_tail(self, embedded: torch.Tensor, depth: int, n_last: int = 1) -> torch.Tensor:
        """Hidden states ``[B, n_last, d]`` of the trailing positions after ``depth`` steps."""
        core = self.recurrent
        core._check_inputs(embedded, None, None, None)
        depth = int(depth)
        if depth <= 0:
            raise ValueError("steps must be positive")
        if not 1 <= n_last <= embedded.shape[1]:
            raise ValueError("n_last must be between 1 and the sequence length")
        if _native_stage1_ready(core, embedded):
            backbone = native_stage1(core, embedded)
        else:
            backbone = core._run_stage1(embedded, word_pos=None, morph_depth=None, attn_mask=None, causal=True)
        applications = [layer for _ in range(depth) for layer in core.stage2]
        if not applications:
            return backbone[:, -n_last:]
        if n_last == 1 and _native_stage2_ready(core, backbone):
            return native_stage2_tail(core.stage2[0], backbone, depth)
        if _fused_stage2_ready(applications, backbone):
            return fused_stage2(applications, backbone, n_last)
        streams = core._expand(backbone)
        for layer in applications[:-1]:
            streams = layer(streams, causal=True)
        streams = stage2_tail(applications[-1], streams, n_last)
        return core._collapse(streams)

    def _hidden_for_head(self, embedded: torch.Tensor, depth: int) -> torch.Tensor:
        if self._tail_path(embedded):
            return self._hidden_tail(embedded, depth)
        hidden, _ = self.recurrent(embedded, steps=depth, bptt_window=None, causal=True)
        return hidden

    @staticmethod
    def _quantiles(raw: torch.Tensor) -> torch.Tensor:
        center = raw[..., 0]
        return torch.stack(
            (
                center - F.softplus(raw[..., 1]),
                center,
                center + F.softplus(raw[..., 2]),
            ),
            dim=-1,
        )

    def forward(self, inputs: torch.Tensor, depth: int):
        projected = cpu_linear(inputs.float(), self.projection.weight, self.projection.bias)
        embedded = self.input_norm(projected)
        hidden = self._hidden_for_head(embedded, depth)
        last = self.final_norm(hidden[:, -1].float())
        raw = cpu_linear(last, self.head.weight, self.head.bias).reshape(-1, self.n_horizons, 3)
        return self._quantiles(raw), hidden.detach().float().square().mean().sqrt()


class MultiAssetRDTCPU(RDT4QuantCPU):
    def __init__(self, base_config: dict, asset_count: int):
        super().__init__(57, base_config)
        self.stock_projection = nn.Linear(23, self.cfg.d_model)
        self.stock_norm = nn.LayerNorm(self.cfg.d_model)
        self.asset_embedding = nn.Embedding(asset_count, self.cfg.d_model)
        self.period_embedding = nn.Embedding(2, self.cfg.d_model)
        self.stock_heads = nn.ModuleDict(
            {name: nn.Linear(self.cfg.d_model, 9) for name in ("equity_daily", "token_hour")}
        )

    def forward(self, inputs: torch.Tensor, depth: int, domain: str = "crypto", asset_id=None):
        if domain == "crypto":
            return super().forward(inputs, depth)
        if domain not in self.stock_heads or asset_id is None:
            raise ValueError("typed stock domain and asset identifiers are required")
        identity = self.asset_embedding(asset_id)
        period = self.period_embedding.weight[0 if domain == "equity_daily" else 1]
        projected = cpu_linear(inputs.float(), self.stock_projection.weight, self.stock_projection.bias)
        embedded = self.stock_norm(projected + identity[:, None] + period)
        hidden = self._hidden_for_head(embedded, depth)
        last = self.final_norm(hidden[:, -1].float())
        head = self.stock_heads[domain]
        raw = cpu_linear(last, head.weight, head.bias).reshape(-1, 3, 3)
        return self._quantiles(raw), hidden.detach().float().square().mean().sqrt()
