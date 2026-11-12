#!/usr/bin/env python3
"""Regenerate the cross-language fixtures in tests/data (needs torch and scikit-learn).

For each model ``<name>.montlok`` there is ``<name>.io``: little-endian float64
values, first the inputs of a batch of 3 (3 * max(sequence, 1) * features),
then the outputs montlok's Python binding produced for them (3 * outputs). The
C++ (tests/cpp) and Rust (bindings/rust) tests replay them through the C ABI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import montlok
from montlok import export

HERE = Path(__file__).resolve().parent
BATCH = 3


def write(name: str, exported: export.Exported, x: np.ndarray) -> None:
    exported = exported.with_metadata(fixture=name)
    exported.save(HERE / f"{name}.montlok")
    model = montlok.load(HERE / f"{name}.montlok")
    y = model.predict(x)
    np.concatenate([x.reshape(-1), y.reshape(-1)]).astype("<f8").tofile(HERE / f"{name}.io")
    print(name, model.family, model.features, model.sequence, model.outputs)


def main() -> None:
    import torch
    from sklearn.ensemble import GradientBoostingRegressor

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    gru = torch.nn.GRU(5, 8, num_layers=2, batch_first=True)
    head = torch.nn.Sequential(torch.nn.LayerNorm(8), torch.nn.Linear(8, 2))
    mean, scale = rng.normal(size=5) * 10, rng.uniform(0.5, 3, size=5)
    write("gru", export.from_torch([gru, export.LAST, head], features=5, sequence=6,
                                   preprocess={"mean": mean, "scale": scale, "clip": (-6, 6)}),
          rng.normal(size=(BATCH, 6, 5)) * 5 + mean)

    x = rng.normal(size=(300, 4))
    y = x[:, 0] * x[:, 1] + np.sin(x[:, 2])
    gbr = GradientBoostingRegressor(n_estimators=25, max_depth=3, random_state=0).fit(x, y)
    test = rng.normal(size=(BATCH, 4))
    test[1, 2] = np.nan
    write("gbr", export.from_sklearn(gbr), test)


if __name__ == "__main__":
    main()
