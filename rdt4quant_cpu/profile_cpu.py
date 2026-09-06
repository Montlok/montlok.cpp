#!/usr/bin/env python3
"""Attribute full RDT CPU latency to Mamba and Transformer components."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model_cpu import MultiAssetRDTCPU

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = json.loads(args.base_config.read_text())
    with np.load(args.fixture, allow_pickle=False) as source:
        inputs = torch.as_tensor(source["inputs"][:1]).float()
        assets = torch.as_tensor(source["asset_id"][:1]).long()
    model = MultiAssetRDTCPU(config, 31).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    times = defaultdict(float)
    calls = defaultdict(int)
    starts = {}

    def attach(name: str, module: torch.nn.Module) -> None:
        def before(_module, _args):
            starts[name] = time.perf_counter()

        def after(_module, _args, _output):
            times[name] += (time.perf_counter() - starts.pop(name)) * 1000
            calls[name] += 1

        module.register_forward_pre_hook(before)
        module.register_forward_hook(after)

    for index, layer in enumerate(model.recurrent.stage1):
        attach(f"mamba_block_{index}", layer)
    transformer = model.recurrent.stage2[0]
    attach("transformer_layer_total", transformer)
    attach("transformer_attention", transformer.attn)
    attach("transformer_ffn", transformer.ffn)
    attach("transformer_attn_hyperconnection", transformer.attn_hc)
    attach("transformer_ffn_hyperconnection", transformer.ffn_hc)
    with torch.inference_mode():
        model(inputs, 4, "crypto", assets)
        times.clear()
        calls.clear()
        totals = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            model(inputs, 4, "crypto", assets)
            totals.append((time.perf_counter() - start) * 1000)
    report = {
        "iterations": args.iterations,
        "full_model_p50_ms": float(np.quantile(totals, 0.5)),
        "components_mean_total_ms": {name: times[name] / args.iterations for name in sorted(times)},
        "calls_per_inference": {name: calls[name] / args.iterations for name in sorted(calls)},
        "orders_sent": False,
    }
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
