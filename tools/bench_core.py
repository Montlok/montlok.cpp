#!/usr/bin/env python3
"""Latency of montlok and of the frameworks the models come from.

Random weights, one process, medians and 99th percentiles over repeated calls
made one after another from Python (the way an engine worker calls them).
Prints JSON. Run on an otherwise idle machine.

    python tools/bench_core.py [--threads N] [--iterations 2000]
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import numpy as np

import montlok
from montlok import export


def timed(fn, iterations: int) -> dict:
    for _ in range(max(10, iterations // 20)):
        fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1000)
    samples.sort()
    return {"p50_us": round(statistics.median(samples), 2), "p99_us": round(samples[int(0.99 * (len(samples) - 1))], 2)}


def bench_gru(iterations: int, torch) -> dict:
    nn = torch.nn
    torch.manual_seed(0)
    gru = nn.GRU(40, 64, batch_first=True).eval()
    head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 1)).eval()
    model = export.from_torch([gru, export.LAST, head], features=40, sequence=32).load()
    out = {}
    for batch in (1, 16):
        x = np.random.default_rng(batch).normal(size=(batch, 32, 40)).astype(np.float32)
        xt = torch.from_numpy(x)

        def run_torch():
            with torch.inference_mode():
                return head(gru(xt)[1][-1])

        out[f"batch{batch}"] = {"montlok": timed(lambda: model.predict(x), iterations),
                                "torch": timed(run_torch, iterations)}
    live = model.stream()
    bar = np.random.default_rng(9).normal(size=40)
    out["stream_step"] = {"montlok": timed(lambda: live.step(bar), iterations)}
    return out


def bench_trees(iterations: int) -> dict:
    from sklearn import ensemble

    rng = np.random.default_rng(0)
    x = rng.normal(size=(5000, 20))
    y = x[:, 0] * x[:, 1] + np.sin(x[:, 2]) + rng.normal(size=5000) * 0.1
    gbr = ensemble.GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=0).fit(x, y)
    model = export.from_sklearn(gbr).load()
    row = x[:1]
    big = np.tile(x, (20, 1))
    return {
        "gbr300_row": {"montlok": timed(lambda: model.predict(row), iterations),
                       "sklearn": timed(lambda: gbr.predict(row), max(50, iterations // 20))},
        "gbr300_100k_rows_ms": {
            "montlok": round(timed(lambda: model.predict(big), 5)["p50_us"] / 1000, 2),
            "sklearn": round(timed(lambda: gbr.predict(big), 3)["p50_us"] / 1000, 2),
        },
    }


def bench_micro(iterations: int) -> dict:
    from montlok import micro

    ms = 1_000_000
    n = 200_000
    rng = np.random.default_rng(0)
    t = (np.cumsum(rng.integers(1, 5, size=n)) * ms + 10**15).astype(np.int64)
    kind = rng.choice([0, 1, 1, 2], size=n).astype(np.int32)
    kind[:3] = [0, 1, 1]
    mid = 100 + np.cumsum(rng.normal(scale=0.01, size=n))
    side = np.where(kind == 2, 1, 0).astype(np.int32)
    a = np.where(kind == 1, mid - 0.01, mid)
    b = np.where(kind == 1, mid + 0.01, np.where(kind == 2, 3.0, mid))
    c = np.where(kind == 0, 1.0, np.where(kind == 1, 10.0, 0.0))
    d = np.where(kind == 1, 12.0, 0.0)
    engine = micro.FeatureEngine()
    start = time.perf_counter()
    engine.replay(t, kind, side, a, b, c, d)
    replay_ns = (time.perf_counter() - start) * 1e9 / n
    live = micro.FeatureEngine()
    events = list(zip(t.tolist(), kind.tolist(), side.tolist(), a.tolist(), b.tolist(), c.tolist(), d.tolist()))
    position = iter(range(n))

    def one():
        e = events[next(position)]
        live.update(*e)
        live.snapshot()

    return {"update_and_snapshot": timed(one, min(iterations * 10, n - 1000)),
            "replay_ns_per_event": round(replay_ns, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=1, help="montlok and torch intra-op threads")
    args = parser.parse_args()
    montlok.set_threads(args.threads)
    report: dict = {"montlok": montlok.__version__, "machine": platform.machine(), "processor": platform.processor(),
                    "threads": args.threads}
    try:
        import torch

        torch.set_num_threads(args.threads)
        report["torch"] = torch.__version__
        report["gru_40x32_h64"] = bench_gru(args.iterations, torch)
    except ImportError:
        report["gru_40x32_h64"] = "torch not installed"
    try:
        report["trees"] = bench_trees(args.iterations)
    except ImportError:
        report["trees"] = "scikit-learn not installed"
    report["micro"] = bench_micro(args.iterations)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
