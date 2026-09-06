#!/usr/bin/env python3
"""Create official CUDA predictions and CPU-reference parity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


PIPELINES = Path(__file__).resolve().parents[1]
os.environ.setdefault("RDT_BASE_CODE", str(PIPELINES / "rdt4quant"))
sys.path.insert(0, str(PIPELINES / "rdt4quant_fullpass"))
sys.path.insert(0, str(PIPELINES / "rdt4quant_full_backtest"))

from multiasset_model import MultiAssetRDT  # noqa: E402
from train_walkforward import window  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cpu-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base_config = json.loads(args.base_config.read_text())
    with np.load(args.data, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    if "asset_id" not in data:
        data["asset_id"] = np.zeros(len(data["x"]), dtype="int64")
    candidates = data["test_indices"]
    selected = np.linspace(0, len(candidates) - 1, args.samples, dtype=int)
    indices = candidates[selected]
    inputs, assets = window(data, indices, 480, "cuda")
    official = MultiAssetRDT(base_config, 31).cuda().eval()
    official.load_state_dict(checkpoint["model"], strict=True)
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.perf_counter()
        official_prediction, _ = official(inputs, 4, "crypto", assets)
        torch.cuda.synchronize()
        official_ms = (time.perf_counter() - start) * 1000
    official_prediction = official_prediction.float().cpu().numpy()
    sys.path.insert(0, str(args.cpu_code))
    from model_cpu import MultiAssetRDTCPU

    cpu = MultiAssetRDTCPU(base_config, 31).cpu().eval()
    cpu.load_state_dict(checkpoint["model"], strict=True)
    cpu_inputs = inputs.float().cpu()
    with torch.no_grad():
        start = time.perf_counter()
        cpu_prediction, _ = cpu(cpu_inputs, 4, "crypto", assets.cpu())
        cpu_ms = (time.perf_counter() - start) * 1000
    cpu_prediction = cpu_prediction.numpy()
    y_scale = data["y_scale"] if "y_scale" in data else np.ones(3)
    difference_bps = np.abs(cpu_prediction - official_prediction) * y_scale[None, :, None]
    np.savez_compressed(
        args.output / "fixture.npz",
        inputs=cpu_inputs.numpy(),
        asset_id=assets.cpu().numpy(),
        official_prediction=official_prediction,
        cpu_prediction_spark=cpu_prediction,
        indices=indices,
        y_scale=y_scale,
    )
    report = {
        "checkpoint_sha256": sha256(args.checkpoint),
        "data_sha256": sha256(args.data),
        "samples": len(indices),
        "official_cuda_ms_batch": official_ms,
        "cpu_reference_ms_batch_spark": cpu_ms,
        "max_abs_normalized_difference": float(np.max(np.abs(cpu_prediction - official_prediction))),
        "max_abs_bps_difference": float(difference_bps.max()),
        "median_abs_bps_difference": float(np.median(difference_bps)),
        "official_prediction_finite": bool(np.isfinite(official_prediction).all()),
        "cpu_prediction_finite": bool(np.isfinite(cpu_prediction).all()),
    }
    (args.output / "spark_parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
