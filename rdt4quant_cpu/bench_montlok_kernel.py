#!/usr/bin/env python3
"""Micro- and full-model benchmarks for the montlok CPU backend (random weights).

Runs with random weights anywhere the extension builds. It reports, at the
model geometry:

* the Mamba-3 layer with the pure PyTorch reference, the recurrence-only C++
  path and the fused path, plus the fused op and the bare recurrence kernel
* one mHC hyper-connection, RMSNorm and the MLA attention block, reference vs C++
* the whole MultiAssetRDTCPU forward (depth 4) with per-stage attribution,
  including full-sequence and trailing-position fused paths

    python bench_montlok_kernel.py --threads 16 --iterations 50
    python bench_montlok_kernel.py --threads 32 --skip-reference   # fast, C++ only

Compare several --threads values: with SMT the physical core count is often
the sweet spot, and OMP_PROC_BIND=close OMP_PLACES=cores helps.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / "rdt4quant" / "native"
if str(NATIVE) not in sys.path:
    sys.path.insert(0, str(NATIVE))

from Model.config import RDTConfig  # noqa: E402
from Model.layers.mhc import ManifoldHyperConnection  # noqa: E402
from Model.layers.mla import MLA  # noqa: E402
from Model.layers.rmsnorm import RMSNorm  # noqa: E402

from mamba3_cpu import Mamba3CPUReference  # noqa: E402
from mhc_cpu import ManifoldHyperConnectionCPU  # noqa: E402
from mla_cpu import MLACPU  # noqa: E402
from layers_cpu import install_cpu_layers  # noqa: E402

BACKEND_VARS = (
    "MONTLOK_CPP",
    "MONTLOK_CPP_FUSED",
    "MONTLOK_CPP_MHC",
    "MONTLOK_CPP_LAYERS",
    "MONTLOK_CPP_TAIL",
    "MONTLOK_CPP_DNNL",
    "MONTLOK_DNNL_MIN_ROWS",
    "MONTLOK_CPP_NATIVE_STAGE1",
    "MONTLOK_CPP_NATIVE_STAGE2",
    "MONTLOK_STAGE2_INPLACE",
    "MONTLOK_STAGE2_REUSE_COEFF",
    "MONTLOK_MAMBA_SLAB",
    "MONTLOK_CACHE_PARAMETER_PACKS",
)


@contextmanager
def backend(**flags: str | None):
    saved = {name: os.environ.get(name) for name in BACKEND_VARS}
    for name in BACKEND_VARS:
        os.environ.pop(name, None)
    for name, value in flags.items():
        if value is not None:
            os.environ[name] = value
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class Bench:
    def __init__(self, iterations: int, warmup: int):
        self.iterations = iterations
        self.warmup = warmup
        self.results: dict[str, dict[str, float]] = {}

    def time(self, name: str, fn, iterations: int | None = None) -> float:
        iterations = iterations or self.iterations
        with torch.inference_mode():
            for _ in range(self.warmup):
                fn()
            samples = []
            for _ in range(iterations):
                start = time.perf_counter()
                fn()
                samples.append((time.perf_counter() - start) * 1e3)
        median = statistics.median(samples)
        self.results[name] = {
            "median_ms": median,
            "min_ms": min(samples),
            "p95_ms": sorted(samples)[int(0.95 * (len(samples) - 1))],
        }
        print(f"{name:<52s} median {median:8.3f} ms   min {min(samples):8.3f} ms")
        return median


def bench_mamba(bench: Bench, skip_reference: bool) -> None:
    print("--- Mamba-3 layer [1, 480, 256], d_state 64, headdim 64 (8 heads)")
    torch.manual_seed(0)
    layer = Mamba3CPUReference(256, d_state=64, expand=2, headdim=64).eval()
    x = torch.randn(1, 480, 256)
    if not skip_reference:
        with backend():
            bench.time("mamba layer: PyTorch reference", lambda: layer(x))
    with backend(MONTLOK_CPP="1", MONTLOK_CPP_FUSED="0"):
        bench.time("mamba layer: C++ recurrence, PyTorch prep", lambda: layer(x))
    with backend(MONTLOK_CPP="1"):
        bench.time("mamba layer: C++ fused", lambda: layer(x))
        with torch.inference_mode():
            projected = layer.in_proj(x)
        bench.time("  in_proj GEMM 480x256x1560", lambda: layer.in_proj(x))
        bench.time("  fused op (prep + recurrence)", lambda: layer._forward_fused(projected))
        from montlok_loader import load_montlok

        ext = load_montlok()
        B, L, H, N, D = 1, 480, 8, 64, 64
        torch.manual_seed(1)
        q, k = torch.randn(B, L, H, N), torch.randn(B, L, H, N)
        v, gate = torch.randn(B, L, H, D), torch.randn(B, L, H, D)
        adt, dt, trap, skip = (
            -torch.rand(B, L, H) * 0.1,
            torch.rand(B, L, H) * 0.1,
            torch.randn(B, L, H),
            torch.randn(H),
        )
        bench.time(
            "  recurrence kernel only",
            lambda: ext.mamba3_siso_recurrent(q, k, v, gate, adt, dt, trap, skip),
        )


def bench_stage2_parts(bench: Bench, skip_reference: bool) -> None:
    print("--- stage-2 pieces [1, 480, 4 streams, 256]")
    torch.manual_seed(2)
    hc = ManifoldHyperConnectionCPU(256, n_streams=4, sinkhorn_iters=20).eval()
    with torch.no_grad():
        for alpha in (hc.pre_alpha, hc.post_alpha, hc.res_alpha):
            alpha.fill_(0.5)
    streams = torch.randn(1, 480, 4, 256)
    identity = lambda a: a  # noqa: E731
    if not skip_reference:
        bench.time(
            "mHC hyper-connection: PyTorch reference",
            lambda: ManifoldHyperConnection.forward(hc, streams, identity),
        )
    with backend(MONTLOK_CPP="1"):
        bench.time("mHC hyper-connection: C++ fused", lambda: hc(streams, identity))

    norm = RMSNorm(256, eps=1e-6).eval()
    holder = torch.nn.Sequential(norm)
    x = torch.randn(1, 480, 256)
    if not skip_reference:
        bench.time("RMSNorm [1,480,256]: PyTorch reference", lambda: holder(x))
    install_cpu_layers(holder)
    with backend(MONTLOK_CPP="1"):
        bench.time("RMSNorm [1,480,256]: C++", lambda: holder(x))

    cfg = RDTConfig(**json.loads((HERE.parent / "rdt4quant" / "config.json").read_text())["model"])
    mla = MLACPU(cfg).eval()
    if not skip_reference:
        bench.time("MLA attention block: PyTorch reference", lambda: MLA.forward(mla, x))
    with backend(MONTLOK_CPP="1"):
        bench.time("MLA attention block: C++ rope + SDPA", lambda: mla(x))


def bench_full_model(bench: Bench, skip_reference: bool, depth: int) -> None:
    from model_cpu import MultiAssetRDTCPU

    print(f"--- full MultiAssetRDTCPU forward, random weights, inputs [1, 480, 57], depth {depth}")
    config = json.loads((HERE.parent / "rdt4quant" / "config.json").read_text())
    torch.manual_seed(0)
    model = MultiAssetRDTCPU(config, 31).eval()
    with torch.no_grad():
        for layer in model.recurrent.stage2:
            for hc in (layer.attn_hc, layer.ffn_hc):
                for alpha in (hc.pre_alpha, hc.post_alpha, hc.res_alpha):
                    alpha.fill_(0.5)
    inputs = torch.randn(1, 480, 57)
    assets = torch.zeros(1, dtype=torch.long)

    totals = defaultdict(float)
    calls = defaultdict(int)
    starts = {}

    def attach(name, module):
        def before(_m, _a):
            starts[name] = time.perf_counter()

        def after(_m, _a, _o):
            totals[name] += (time.perf_counter() - starts.pop(name)) * 1e3
            calls[name] += 1

        module.register_forward_pre_hook(before)
        module.register_forward_hook(after)

    for index, layer in enumerate(model.recurrent.stage1):
        attach(f"stage1[{index}]", layer)
    stage2 = model.recurrent.stage2[0]
    attach("stage2 layer", stage2)
    attach("stage2 attn_hc (incl. attention)", stage2.attn_hc)
    attach("stage2 ffn_hc (incl. ffn)", stage2.ffn_hc)
    attach("stage2 attention", stage2.attn)
    attach("stage2 FFN", stage2.ffn)

    def run():
        with torch.inference_mode():
            return model(inputs, depth, "crypto", assets)[0]

    def attributed(label: str) -> None:
        totals.clear()
        calls.clear()
        median = bench.time(label, run)
        n = bench.iterations + bench.warmup
        for name in totals:
            print(f"    {name:<44s} {totals[name] / n:8.3f} ms per forward ({calls[name] // n} calls)")
        bench.results[label]["components_ms"] = {name: totals[name] / n for name in totals}
        return median

    if not skip_reference:
        with backend():
            attributed("full model: PyTorch reference")
    with backend(
        MONTLOK_CPP="1",
        MONTLOK_CPP_MHC="0",
        MONTLOK_CPP_LAYERS="0",
        MONTLOK_CPP_TAIL="0",
    ):
        attributed("full model: Mamba kernel only")
    with backend(MONTLOK_CPP="1", MONTLOK_CPP_TAIL="0"):
        attributed("full model: all fused ops, full sequence")
    with backend(MONTLOK_CPP="1", MONTLOK_CPP_NATIVE_STAGE1="0", MONTLOK_CPP_NATIVE_STAGE2="0"):
        attributed("full model: fused tail, Python orchestration")
    with backend(MONTLOK_CPP="1"):
        attributed("full model: native Stage-1/Stage-2 tail")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--depth", type=int, default=4, help="recurrent stage-2 steps")
    parser.add_argument("--skip-reference", action="store_true", help="only time the C++ paths")
    parser.add_argument("--no-flush-denormal", action="store_true")
    parser.add_argument("--only", choices=("mamba", "stage2", "model"), action="append")
    parser.add_argument("--output", type=Path, help="write all timings as JSON")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if not args.no_flush_denormal:
        torch.set_flush_denormal(True)

    from montlok_loader import load_montlok

    info = load_montlok().build_info()
    print(f"threads {torch.get_num_threads()}  torch {torch.__version__}  build {info}")
    bench = Bench(args.iterations, args.warmup)
    sections = set(args.only or ("mamba", "stage2", "model"))
    if "mamba" in sections:
        bench_mamba(bench, args.skip_reference)
    if "stage2" in sections:
        bench_stage2_parts(bench, args.skip_reference)
    if "model" in sections:
        bench_full_model(bench, args.skip_reference, args.depth)
    if args.output:
        payload = {
            "threads": args.threads,
            "torch": torch.__version__,
            "build": info,
            "timings": bench.results,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
