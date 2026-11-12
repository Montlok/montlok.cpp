"""Plugin model families (examples/plugin/ema_dot.c) and the model C ABI."""

from __future__ import annotations

import ctypes as ct
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import montlok
from montlok import export

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def plugin(tmp_path_factory) -> Path:
    compiler = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if not compiler:
        pytest.skip("no C compiler")
    out = tmp_path_factory.mktemp("plugin") / ("libema_dot.dylib" if sys.platform == "darwin" else "libema_dot.so")
    subprocess.run([compiler, "-O2", "-std=c11", "-Wall", "-Werror", "-shared", "-fPIC", "-I", str(ROOT / "include"),
                    str(ROOT / "examples/plugin/ema_dot.c"), "-o", str(out)], check=True)
    if "ema_dot" not in montlok.families():
        montlok.load_plugin(out)
    return out


def ema_model(features=4, sequence=6, alpha=0.8, bias=0.1, **kwargs):
    weight = np.linspace(-1, 1, features)
    tensors = {"ema.weight": weight, "ema.alpha": np.array([alpha]), "ema.bias": np.array([bias])}
    return export.make("ema_dot", {}, tensors, features=features, sequence=sequence, outputs=1, **kwargs), weight


def reference(x, weight, alpha, bias):
    state = np.zeros(len(x))
    for t in range(x.shape[1]):
        state = alpha * state + (1 - alpha) * (x[:, t] @ weight + bias)
    return state


def test_plugin_family_predicts_and_streams(plugin):
    assert "ema_dot" in montlok.families()
    exported, weight = ema_model()
    model = exported.load()
    assert model.family == "ema_dot" and model.streamable
    x = np.random.default_rng(0).normal(size=(5, 6, 4))
    np.testing.assert_allclose(model.predict(x)[:, 0], reference(x, weight, 0.8, 0.1), rtol=1e-13)
    live = model.stream(batch=5)
    for t in range(6):
        y = live.step(x[:, t])
    np.testing.assert_array_equal(y, model.predict(x))


def test_plugin_sees_preprocessed_inputs(plugin):
    mean, scale = np.arange(4.0), np.full(4, 2.0)
    exported, weight = ema_model(preprocess={"mean": mean, "scale": scale, "clip": (-1, 1)})
    x = np.random.default_rng(1).normal(size=(3, 6, 4)) * 5
    expected = reference(np.clip((x - mean) / scale, -1, 1), weight, 0.8, 0.1)
    np.testing.assert_allclose(exported.load().predict(x)[:, 0], expected, rtol=1e-13)


def test_plugin_errors_surface(plugin):
    exported, _ = ema_model()
    del exported.tensors["ema.alpha"]
    with pytest.raises(montlok.FormatError, match="ema_dot: needs ema.alpha"):
        exported.load()
    with pytest.raises(ValueError, match="already registered"):
        montlok.load_plugin(plugin)
    with pytest.raises(OSError):
        montlok.load_plugin(ROOT / "README.md")


# --- the model C ABI, called the way C, Rust or Qt code calls it ---

class Info(ct.Structure):
    _fields_ = [("features", ct.c_uint32), ("sequence", ct.c_uint32), ("outputs", ct.c_uint32),
                ("flags", ct.c_uint32)]


@pytest.fixture(scope="module")
def lib():
    lib = ct.CDLL(montlok._core.__file__)
    lib.montlok_version.restype = ct.c_char_p
    lib.montlok_last_error.restype = ct.c_char_p
    lib.montlok_model_open_memory.argtypes = [ct.c_char_p, ct.c_size_t, ct.POINTER(ct.c_void_p)]
    lib.montlok_model_get_info.argtypes = [ct.c_void_p, ct.POINTER(Info)]
    lib.montlok_model_predict.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_size_t,
                                          ct.POINTER(ct.c_double)]
    lib.montlok_model_close.argtypes = [ct.c_void_p]
    lib.montlok_stream_open.argtypes = [ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_void_p)]
    lib.montlok_stream_step.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.POINTER(ct.c_double)]
    lib.montlok_stream_close.argtypes = [ct.c_void_p]
    return lib


def test_c_abi_model_roundtrip(lib):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    exported = export.from_torch([torch.nn.GRU(3, 5, batch_first=True), export.LAST, torch.nn.Linear(5, 2)],
                                 features=3, sequence=4)
    data = exported.to_bytes()
    handle = ct.c_void_p()
    assert lib.montlok_model_open_memory(data, len(data), ct.byref(handle)) == 0
    try:
        info = Info()
        assert lib.montlok_model_get_info(handle, ct.byref(info)) == 0
        assert (info.features, info.sequence, info.outputs, info.flags & 1) == (3, 4, 2, 1)
        x = np.random.default_rng(0).normal(size=(2, 4, 3)).astype(np.float32)
        out = (ct.c_double * 4)()
        assert lib.montlok_model_predict(handle, 1, x.ctypes.data, 2, out) == 0
        np.testing.assert_array_equal(np.array(out).reshape(2, 2), exported.load().predict(x))
        stream = ct.c_void_p()
        assert lib.montlok_stream_open(handle, 1, ct.byref(stream)) == 0
        step_out = (ct.c_double * 2)()
        for t in range(4):
            row = np.ascontiguousarray(x[0, t], dtype=np.float64)
            assert lib.montlok_stream_step(stream, 2, row.ctypes.data, step_out) == 0
        np.testing.assert_array_equal(np.array(step_out), np.array(out)[:2])
        lib.montlok_stream_close(stream)
        assert lib.montlok_model_predict(handle, 7, x.ctypes.data, 2, out) == 1
        assert b"dtype" in lib.montlok_last_error()
    finally:
        lib.montlok_model_close(handle)
    bad = ct.c_void_p(123)
    assert lib.montlok_model_open_memory(b"garbage!", 8, ct.byref(bad)) == 3
    assert bad.value is None
    assert lib.montlok_version().decode() == montlok.__version__
