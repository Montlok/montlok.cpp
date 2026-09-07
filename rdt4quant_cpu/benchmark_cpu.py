#!/usr/bin/env python3
"""Benchmark an RDT4quant checkpoint on CPU and check its outputs against a fixture."""

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


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument(
        "--compile-transformer",
        action="store_true",
        help="torch.compile the stage-2 layer (slower than eager once MONTLOK_CPP=1 fused ops are active)",
    )
    parser.add_argument(
        "--no-flush-denormal",
        action="store_true",
        help="keep IEEE denormals (default enables FTZ/DAZ, which avoids ~100x slow paths on decaying states)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if not args.no_flush_denormal:
        torch.set_flush_denormal(True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mamba3_cpu import cpp_backend
    from model_cpu import MultiAssetRDTCPU

    montlok_build = None
    if cpp_backend() is not None:
        from montlok_loader import load_montlok

        montlok_build = load_montlok().build_info()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = json.loads(args.base_config.read_text())
    with np.load(args.fixture, allow_pickle=False) as source:
        inputs = torch.as_tensor(source["inputs"][:1]).float()
        assets = torch.as_tensor(source["asset_id"][:1]).long()
        official = source["official_prediction"][:1]
        y_scale = source["y_scale"]
    if args.data_manifest:
        y_scale = np.asarray(json.loads(args.data_manifest.read_text())["y_scale"], dtype="float64")
    model = MultiAssetRDTCPU(config, 31).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    if args.compile_transformer:
        model.recurrent.stage2[0] = torch.compile(
            model.recurrent.stage2[0],
            mode="reduce-overhead",
            fullgraph=False,
            dynamic=False,
        )
    times, prediction = [], None
    with torch.inference_mode():
        start = time.perf_counter()
        prediction, _ = model(inputs, 4, "crypto", assets)
        warmup_ms = (time.perf_counter() - start) * 1000
        for _ in range(args.iterations):
            start = time.perf_counter()
            prediction, _ = model(inputs, 4, "crypto", assets)
            times.append((time.perf_counter() - start) * 1000)
    prediction = prediction.numpy()
    difference = np.abs(prediction - official) * y_scale[None, :, None]
    report = {
        "device": "cpu",
        "threads": args.threads,
        "compiled_transformer": args.compile_transformer,
        "flush_denormal": not args.no_flush_denormal,
        "montlok_backend": cpp_backend(),
        "montlok_build": montlok_build,
        "montlok_env": {
            name: os.environ.get(name)
            for name in (
                "MONTLOK_CPP_FUSED",
                "MONTLOK_CPP_MHC",
                "MONTLOK_CPP_LAYERS",
                "MONTLOK_CPP_DNNL",
                "MONTLOK_DNNL_MIN_ROWS",
                "MONTLOK_CPP_NATIVE_STAGE1",
                "MONTLOK_CPP_NATIVE_STAGE2",
                "MONTLOK_STAGE2_INPLACE",
                "MONTLOK_OPENMP",
                "MONTLOK_SERIAL_WORK",
            )
        },
        "checkpoint_sha256": sha256(args.checkpoint),
        "fixture_sha256": sha256(args.fixture),
        "data_manifest_sha256": sha256(args.data_manifest) if args.data_manifest else None,
        "warmup_ms": warmup_ms,
        "iterations": args.iterations,
        "latency_ms": times,
        "p50_ms": float(np.quantile(times, 0.5)),
        "p95_ms": float(np.quantile(times, 0.95)),
        "max_abs_bps_difference_vs_official_cuda": float(difference.max()),
        "prediction": prediction.tolist(),
    }
    serialized = json.dumps(report, allow_nan=False)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(serialized, flush=True)


if __name__ == "__main__":
    main()
