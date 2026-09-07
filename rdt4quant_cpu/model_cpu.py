"""CPU runtime for the trained RDT4quant Mamba-3 SISO checkpoints."""

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

from mamba3_cpu import Mamba3CPUReference  # noqa: E402
from mhc_cpu import ManifoldHyperConnectionCPU  # noqa: E402
from mla_cpu import MLACPU  # noqa: E402
from rmsnorm_cpu import install_cpu_norms  # noqa: E402

mamba_layer.OfficialMamba3 = Mamba3CPUReference

import Model.two_stage as two_stage  # noqa: E402
from Model.two_stage import TwoStageCore  # noqa: E402

# Stage-2 layers build their hyper-connections and attention from these
# module-level names; the CPU subclasses keep the upstream parameters.
two_stage.ManifoldHyperConnection = ManifoldHyperConnectionCPU
two_stage.MLA = MLACPU


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
        install_cpu_norms(self)

    def encode(self, inputs: torch.Tensor, depth: int):
        embedded = self.input_norm(self.projection(inputs.float()))
        return self.recurrent(embedded, steps=depth, bptt_window=None, causal=True)

    def forward(self, inputs: torch.Tensor, depth: int):
        hidden, _ = self.encode(inputs, depth)
        last = self.final_norm(hidden[:, -1].float())
        raw = self.head(last).reshape(-1, self.n_horizons, 3)
        center = raw[..., 0]
        quantiles = torch.stack(
            (center - F.softplus(raw[..., 1]), center, center + F.softplus(raw[..., 2])),
            dim=-1,
        )
        return quantiles, hidden.detach().float().square().mean().sqrt()


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
        embedded = self.stock_norm(self.stock_projection(inputs.float()) + identity[:, None] + period)
        hidden, _ = self.recurrent(embedded, steps=depth, bptt_window=None, causal=True)
        raw = self.stock_heads[domain](self.final_norm(hidden[:, -1].float())).reshape(-1, 3, 3)
        center = raw[..., 0]
        quantiles = torch.stack(
            (center - F.softplus(raw[..., 1]), center, center + F.softplus(raw[..., 2])), dim=-1
        )
        return quantiles, hidden.detach().float().square().mean().sqrt()
