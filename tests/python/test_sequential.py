"""The sequential family against PyTorch: linear, MLP, GRU, LSTM, norms, pooling, streaming."""

from __future__ import annotations

import numpy as np
import pytest

import montlok
from montlok import export

torch = pytest.importorskip("torch")
nn = torch.nn


def torch_out(modules, x: np.ndarray) -> np.ndarray:
    """Runs a module chain with montlok's markers on float32 input."""
    value = torch.from_numpy(x.astype(np.float32))
    with torch.no_grad():
        for module in modules:
            if module == export.LAST:
                value = value[:, -1]
            elif module == export.MEAN:
                value = value.mean(dim=1)
            elif module == export.FLATTEN:
                value = value.flatten(1)
            elif isinstance(module, (nn.GRU, nn.LSTM)):
                value = module(value)[0]
            else:
                value = module(value)
    return value.double().numpy()


def assert_close(actual, expected, tol=2e-5):
    scale = max(1.0, float(np.abs(expected).max()))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tol * scale)


@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(7)


def test_mlp_matches_torch():
    modules = [nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 16), nn.GELU(), nn.Dropout(0.3), nn.Linear(16, 3)]
    for m in modules:
        m.eval()
    model = export.from_torch(modules, features=12).load()
    x = np.random.default_rng(0).normal(size=(64, 12))
    assert_close(model.predict(x), torch_out(modules, x))


@pytest.mark.parametrize("activation", [nn.ReLU(), nn.LeakyReLU(0.2), nn.ELU(0.7), nn.GELU(), nn.GELU("tanh"),
                                        nn.Tanh(), nn.Sigmoid(), nn.SiLU(), nn.Softplus(), nn.Softmax(dim=-1)])
def test_activations(activation):
    modules = [nn.Linear(5, 7), activation]
    model = export.from_torch(modules, features=5).load()
    x = np.random.default_rng(1).normal(size=(32, 5)) * 3
    assert_close(model.predict(x), torch_out(modules, x))


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_gru_head_matches_torch(layers):
    gru = nn.GRU(6, 16, num_layers=layers, batch_first=True)
    head = nn.Sequential(nn.LayerNorm(16), nn.Linear(16, 2))
    modules = [gru, export.LAST, head]
    model = export.from_torch(modules, features=6, sequence=20).load()
    assert model.streamable
    x = np.random.default_rng(2).normal(size=(9, 20, 6))
    assert_close(model.predict(x), torch_out(modules, x))


@pytest.mark.parametrize("layers", [1, 2])
def test_lstm_matches_torch(layers):
    lstm = nn.LSTM(4, 12, num_layers=layers, batch_first=True)
    modules = [lstm, export.LAST, nn.Linear(12, 1)]
    model = export.from_torch(modules, features=4, sequence=15).load()
    x = np.random.default_rng(3).normal(size=(5, 15, 4))
    assert_close(model.predict(x), torch_out(modules, x))


def test_recurrent_without_bias_and_rmsnorm():
    if not hasattr(nn, "RMSNorm"):
        pytest.skip("torch without nn.RMSNorm")
    gru = nn.GRU(3, 8, batch_first=True, bias=False)
    modules = [gru, export.LAST, nn.RMSNorm(8), nn.Linear(8, 2, bias=False)]
    model = export.from_torch(modules, features=3, sequence=11).load()
    x = np.random.default_rng(4).normal(size=(4, 11, 3))
    assert_close(model.predict(x), torch_out(modules, x))


@pytest.mark.parametrize("marker", [export.MEAN, export.FLATTEN])
def test_pooling(marker):
    width = 3 * 10 if marker == export.FLATTEN else 3
    modules = [nn.Linear(4, 3), nn.Tanh(), marker, nn.Linear(width, 2)]
    model = export.from_torch(modules, features=4, sequence=10).load()
    assert not model.streamable
    x = np.random.default_rng(5).normal(size=(6, 10, 4))
    assert_close(model.predict(x), torch_out(modules, x))
    with pytest.raises(NotImplementedError):
        model.stream()


def test_stream_equals_full_sequence():
    gru = nn.GRU(5, 10, num_layers=2, batch_first=True)
    model = export.from_torch([gru, export.LAST, nn.Linear(10, 3)], features=5, sequence=25).load()
    x = np.random.default_rng(6).normal(size=(3, 25, 5))
    live = model.stream(batch=3)
    for t in range(25):
        y = live.step(x[:, t])
    np.testing.assert_array_equal(y, model.predict(x))
    live.reset()
    single = model.stream()
    for t in range(25):
        y0 = single.step(x[0, t])
    np.testing.assert_array_equal(y0, model.predict(x[:1])[0])


def test_preprocess_matches_engine_contract():
    """Raw float64 features -> clip((x - mean) / scale) in float64 -> float32 network."""
    rng = np.random.default_rng(8)
    mean, scale, clip = rng.normal(size=6) * 100, rng.uniform(0.5, 50, size=6), (-4.0, 4.0)
    modules = [nn.Linear(6, 2)]
    model = export.from_torch(modules, features=6, preprocess={"mean": mean, "scale": scale, "clip": clip}).load()
    raw = rng.normal(size=(50, 6)) * 300 + 60000
    normalized = np.clip((raw - mean) / scale, *clip)
    assert_close(model.predict(raw), torch_out(modules, normalized))
    nan_row = raw[:1].copy()
    nan_row[0, 2] = np.nan
    assert np.isnan(model.predict(nan_row)).all()


def test_float32_and_float64_inputs_agree():
    model = export.from_torch([nn.Linear(4, 2)], features=4).load()
    x = np.random.default_rng(9).normal(size=(10, 4)).astype(np.float32)
    np.testing.assert_array_equal(model.predict(x), model.predict(x.astype(np.float64)))


def test_threading_is_deterministic():
    gru = nn.GRU(8, 24, batch_first=True)
    model = export.from_torch([gru, export.LAST, nn.Linear(24, 1)], features=8, sequence=32).load()
    x = np.random.default_rng(10).normal(size=(500, 32, 8))
    try:
        montlok.set_threads(1)
        serial = model.predict(x)
        montlok.set_threads(8)
        parallel = model.predict(x)
    finally:
        montlok.set_threads(0)
    np.testing.assert_array_equal(serial, parallel)


def test_concurrent_predict_from_python_threads():
    from concurrent.futures import ThreadPoolExecutor

    model = export.from_torch([nn.GRU(4, 8, batch_first=True), export.LAST, nn.Linear(8, 1)],
                              features=4, sequence=16).load()
    inputs = [np.random.default_rng(i).normal(size=(3, 16, 4)) for i in range(16)]
    expected = [model.predict(x) for x in inputs]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(model.predict, inputs * 4))
    for i, result in enumerate(results):
        np.testing.assert_array_equal(result, expected[i % 16])


def test_unsupported_modules_are_named():
    with pytest.raises(TypeError, match="Conv1d"):
        export.from_torch([nn.Conv1d(3, 3, 2)], features=3)
    with pytest.raises(ValueError, match="time axis"):
        export.from_torch([nn.GRU(3, 4, batch_first=True)], features=3, sequence=5)
    with pytest.raises(ValueError, match="bidirectional"):
        export.from_torch([nn.GRU(3, 4, bidirectional=True)], features=3, sequence=5)
