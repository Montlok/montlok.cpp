"""The montlok command."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import montlok
from montlok.cli import main

DATA = Path(__file__).resolve().parents[1] / "data"
sklearn = pytest.importorskip("sklearn")


@pytest.fixture()
def gbr(tmp_path):
    import joblib
    from sklearn.ensemble import GradientBoostingRegressor

    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 4))
    y = np.sin(x[:, 0]) + x[:, 1] * x[:, 2]
    estimator = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=0).fit(x, y)
    joblib.dump(estimator, tmp_path / "gbr.joblib")
    return estimator, x


def test_convert_check_info_and_predict(tmp_path, gbr, capsys):
    estimator, x = gbr
    # columns in another order than the model's features: the header selects them
    order = [2, 0, 3, 1]
    np.savetxt(tmp_path / "rows.csv", x[:, order], delimiter=",", header="c,a,d,b", comments="")
    np.save(tmp_path / "rows.npy", x)
    assert main(["convert", str(tmp_path / "gbr.joblib"), "-o", str(tmp_path / "gbr.montlok"),
                 "--feature-names", "a,b,c,d", "--check", str(tmp_path / "rows.npy")]) == 0
    assert "max |difference|" in capsys.readouterr().out
    assert main(["info", str(tmp_path / "gbr.montlok")]) == 0
    assert "tree_ensemble" in capsys.readouterr().out
    assert main(["predict", str(tmp_path / "gbr.montlok"), str(tmp_path / "rows.csv"),
                 "-o", str(tmp_path / "y.csv")]) == 0
    got = np.loadtxt(tmp_path / "y.csv", delimiter=",", skiprows=1)
    np.testing.assert_allclose(got, estimator.predict(x), rtol=1e-12, atol=1e-12)


def test_predict_reads_sliding_windows_for_sequence_models(tmp_path, capsys):
    model = montlok.load(DATA / "gru.montlok")
    rows = np.random.default_rng(1).normal(size=(model.sequence + 4, model.features))
    np.save(tmp_path / "rows.npy", rows)
    assert main(["predict", str(DATA / "gru.montlok"), str(tmp_path / "rows.npy"), "-o", str(tmp_path / "y.npy")]) == 0
    windows = np.stack([rows[i:i + model.sequence] for i in range(5)])
    np.testing.assert_array_equal(np.load(tmp_path / "y.npy"), model.predict(windows))


def test_bench_micro_serve_and_env(tmp_path, gbr, capsys, monkeypatch):
    main(["convert", str(tmp_path / "gbr.joblib"), "-o", str(tmp_path / "gbr.montlok")])
    capsys.readouterr()
    assert main(["bench", str(tmp_path / "gbr.montlok"), "--iterations", "300", "--latency-threads", "1"]) == 0
    assert "p50" in capsys.readouterr().out
    assert montlok.latency_threads() == 0

    t = (np.arange(50) * 1_000_000 + 10**15).astype(np.int64)
    kind = np.array([0, 1] + [1, 2] * 24)
    events = np.column_stack([t, kind, (kind == 2).astype(int), np.full(50, 100.0), np.full(50, 100.02),
                              np.full(50, 5.0), np.full(50, 6.0)])
    np.savetxt(tmp_path / "events.csv", events, delimiter=",", header="receive_ns,kind,side,a,b,c,d", comments="",
               fmt="%.17g")
    assert main(["micro", str(tmp_path / "events.csv"), "-o", str(tmp_path / "features.csv")]) == 0
    header = (tmp_path / "features.csv").read_text().splitlines()[0].split(",")
    assert header[0] == "ready" and len(header) == 33

    requests = '{"id": 1, "x": [0, 0, 0, 0]}\n{"id": 2, "x": [1, 2]}\n'
    monkeypatch.setattr(sys, "stdin", io.StringIO(requests))
    assert main(["serve", str(tmp_path / "gbr.montlok")]) == 0
    replies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert replies[0]["id"] == 1 and len(replies[0]["y"]) == 1
    assert replies[1]["id"] == 2 and "error" in replies[1]

    assert main(["env"]) == 0
    assert "families" in capsys.readouterr().out


def test_python_dash_m():
    out = subprocess.run([sys.executable, "-m", "montlok", "--version"], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == f"montlok {montlok.__version__}"
