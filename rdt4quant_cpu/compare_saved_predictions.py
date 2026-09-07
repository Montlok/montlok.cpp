#!/usr/bin/env python3
"""Compare CPU outputs with saved CUDA predictions in chronological order."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--official-predictions", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--compile-transformer", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model_cpu import MultiAssetRDTCPU

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base_config = json.loads(args.base_config.read_text())
    manifest = json.loads(args.data_manifest.read_text())
    with np.load(args.data, allow_pickle=False) as source:
        features = source["x"]
    with np.load(args.official_predictions, allow_pickle=False) as source:
        official = source["prediction_normalized"]
        indices = source["indices"]
        asset_ids = source["asset_id"]
    if official.shape != (len(indices), 3, 3) or np.any(indices < 479):
        raise ValueError("invalid saved prediction contract")
    model = MultiAssetRDTCPU(base_config, 31).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    if args.compile_transformer:
        model.recurrent.stage2[0] = torch.compile(
            model.recurrent.stage2[0], mode="reduce-overhead", fullgraph=False, dynamic=False
        )
    predictions, times = [], []
    offsets = np.arange(-479, 1, dtype="int64")
    with torch.inference_mode():
        for start in range(0, len(indices), args.batch_size):
            selected = indices[start : start + args.batch_size]
            batch = torch.as_tensor(features[selected[:, None] + offsets[None, :]]).float()
            assets = torch.as_tensor(asset_ids[start : start + len(selected)]).long()
            before = time.perf_counter()
            value, _ = model(batch, 4, "crypto", assets)
            times.append((time.perf_counter() - before) * 1000)
            predictions.append(value.numpy())
    cpu = np.concatenate(predictions)
    scale = np.asarray(manifest["y_scale"], dtype="float64")
    difference = np.abs(cpu - official) * scale[None, :, None]
    cpu_median = cpu[:, :, 1] * scale
    official_median = official[:, :, 1] * scale
    thresholds = {}
    for threshold in (0.0, 12.0, 24.0, 40.0):
        official_above = official_median[:, 2] > threshold
        cpu_above = cpu_median[:, 2] > threshold
        thresholds[str(threshold)] = {
            "agreement": float(np.mean(official_above == cpu_above)),
            "disagreements": int(np.sum(official_above != cpu_above)),
            "official_above": int(official_above.sum()),
            "cpu_above": int(cpu_above.sum()),
        }
    report = {
        "kind": "chronological_saved_official_cuda_vs_cpu_reference",
        "samples": len(indices),
        "threads": args.threads,
        "batch_size": args.batch_size,
        "compiled_transformer": args.compile_transformer,
        "checkpoint_sha256": sha256(args.checkpoint),
        "data_sha256": sha256(args.data),
        "data_manifest_sha256": sha256(args.data_manifest),
        "official_predictions_sha256": sha256(args.official_predictions),
        "latency_batch_p50_ms": float(np.quantile(times, 0.5)),
        "latency_batch_p95_ms": float(np.quantile(times, 0.95)),
        "latency_per_window_p50_ms": float(np.quantile(times, 0.5) / args.batch_size),
        "max_abs_bps_difference": float(difference.max()),
        "p95_abs_bps_difference": float(np.quantile(difference, 0.95)),
        "median_abs_bps_difference": float(np.median(difference)),
        "per_horizon_max_abs_bps": difference.max(axis=(0, 2)).tolist(),
        "median_prediction_correlation": [
            float(np.corrcoef(cpu_median[:, horizon], official_median[:, horizon])[0, 1])
            for horizon in range(3)
        ],
        "thresholds_240m": thresholds,
    }
    args.output.mkdir(parents=True)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.output / "cpu_predictions.npz", prediction_normalized=cpu, indices=indices, asset_id=asset_ids)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
