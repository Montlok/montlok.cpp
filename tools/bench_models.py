#!/usr/bin/env python3
"""Write the models of the README's latency table into a directory.

    python tools/bench_models.py build/models
    build/montlok_bench build/models/xgb1000_d8.montlok

* ``xgb1000_d8``: XGBoost-format regression, 100 features, 1000 complete trees
  of depth 8 with random splits.
* ``gbr300``: scikit-learn GradientBoostingRegressor, 300 trees of depth 5 on
  20 features (needs scikit-learn).
* ``gru``: GRU 40 -> 64, LayerNorm and Linear head over 32 bars (needs PyTorch).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from montlok import export


def xgboost_tree(rng: np.random.Generator, depth: int, features: int) -> dict:
    count = 0

    def node(level: int) -> dict:
        nonlocal count
        nodeid = count
        count += 1
        if level == depth:
            return {"nodeid": nodeid, "leaf": float(rng.normal(scale=0.01))}
        yes, no = node(level + 1), node(level + 1)
        missing = yes["nodeid"] if rng.random() < 0.5 else no["nodeid"]
        return {"nodeid": nodeid, "split": f"f{int(rng.integers(features))}",
                "split_condition": float(np.float32(rng.normal())), "yes": yes["nodeid"], "no": no["nodeid"],
                "missing": missing, "children": [yes, no]}

    return node(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    dumps = [xgboost_tree(rng, 8, 100) for _ in range(1000)]
    export.from_xgboost(dumps, features=100, objective="reg:squarederror", base_score=0.5).save(
        args.directory / "xgb1000_d8.montlok")
    written = ["xgb1000_d8"]

    try:
        from sklearn import ensemble

        x = rng.normal(size=(5000, 20))
        y = x[:, 0] * x[:, 1] + np.sin(x[:, 2]) + rng.normal(size=5000) * 0.1
        gbr = ensemble.GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=0).fit(x, y)
        export.from_sklearn(gbr).save(args.directory / "gbr300.montlok")
        written.append("gbr300")
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(0)
        gru = torch.nn.GRU(40, 64, batch_first=True).eval()
        head = torch.nn.Sequential(torch.nn.LayerNorm(64), torch.nn.Linear(64, 1)).eval()
        export.from_torch([gru, export.LAST, head], features=40, sequence=32).save(args.directory / "gru.montlok")
        written.append("gru")
    except ImportError:
        pass

    print(" ".join(str(args.directory / f"{name}.montlok") for name in written))


if __name__ == "__main__":
    main()
