"""Streaming microstructure features through the binding and the C ABI."""

from __future__ import annotations

import ctypes as ct

import numpy as np
import pytest

import montlok
from montlok import micro

MS = 1_000_000


def synthetic_events(n=6000, seed=0):
    """Receive-ordered reference / L1 / trade events around a drifting price."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.integers(1, 15, size=n)) * MS + 10**15
    kind = rng.choice([0, 1, 1, 2], size=n).astype(np.int32)
    kind[:3] = [0, 1, 1]
    mid = 100 + np.cumsum(rng.normal(scale=0.01, size=n))
    spread = rng.uniform(0.01, 0.05, size=n)
    side = np.where(kind == 2, rng.choice([-1, 1], size=n), 0).astype(np.int32)
    a = np.where(kind == 0, mid + rng.normal(scale=0.01, size=n), np.where(kind == 1, mid - spread / 2, mid))
    b = np.where(kind == 0, mid, np.where(kind == 1, mid + spread / 2, rng.uniform(1, 20, size=n)))
    c = np.where(kind == 0, rng.uniform(0.5, 2, size=n), np.where(kind == 1, rng.uniform(1, 50, size=n), 0.0))
    d = np.where(kind == 1, rng.uniform(1, 50, size=n), 0.0)
    return t.astype(np.int64), kind, side, a, b, c, d


def test_replay_equals_event_by_event():
    columns = synthetic_events()
    features, ready = micro.FeatureEngine().replay(*columns)
    engine = micro.FeatureEngine()
    for i, event in enumerate(zip(*columns)):
        engine.update(int(event[0]), int(event[1]), int(event[2]), *map(float, event[3:]))
        snapshot = engine.snapshot()
        assert (snapshot is not None) == bool(ready[i])
        if snapshot is not None:
            np.testing.assert_array_equal(snapshot, features[i])
    assert ready[-1] and ready.mean() > 0.8


def test_feature_definitions():
    engine = micro.FeatureEngine()
    t = 10**15
    engine.update(t, micro.REFERENCE, 0, 100.0, 100.02, 1.5)
    engine.update(t + 1 * MS, micro.BOOK, 0, 99.99, 100.03, 30.0, 10.0)
    assert engine.snapshot() is None  # no 2 s of history yet
    engine.update(t + 2500 * MS, micro.BOOK, 0, 99.99, 100.03, 30.0, 10.0)
    values = dict(zip(micro.FEATURES, engine.snapshot()))
    mid = (99.99 + 100.03) / 2
    assert values["spread_bps"] == pytest.approx(10000 * 0.04 / mid, rel=1e-12)
    assert values["imbalance_l1"] == pytest.approx((30 - 10) / 40, rel=1e-12)
    micro_price = (100.03 * 30 + 99.99 * 10) / 40
    assert values["micro_minus_mid_bps"] == pytest.approx(10000 * (micro_price / mid - 1), rel=1e-12)
    assert values["fair_gap_mid_bps"] == pytest.approx(10000 * (mid / 100.02 - 1), rel=1e-12)
    assert values["fair_uncertainty_bps"] == 1.5
    assert values["book_receive_age_ms"] == 0.0


def test_rejected_event_requires_reset():
    engine = micro.FeatureEngine()
    engine.update(10**15, micro.BOOK, 0, 100.0, 100.1, 1.0, 1.0)
    with pytest.raises(ValueError, match="nonmonotonic"):
        engine.update(10**15 - 1, micro.BOOK, 0, 100.0, 100.1, 1.0, 1.0)
    with pytest.raises(ValueError, match="reset"):
        engine.update(10**15 + 1, micro.BOOK, 0, 100.0, 100.1, 1.0, 1.0)
    engine.reset()
    engine.update(10**15 + 1, micro.BOOK, 0, 100.0, 100.1, 1.0, 1.0)
    with pytest.raises(TypeError):
        engine.update(2**63, micro.BOOK)
    columns = list(synthetic_events(100))
    columns[3] = columns[3].copy()
    columns[3][50] = np.nan
    with pytest.raises(ValueError, match="event 50"):
        micro.FeatureEngine().replay(*columns)


def test_capacity_overflow_is_reported():
    engine = micro.FeatureEngine(capacity=4)
    t = 10**15
    with pytest.raises(ValueError, match="capacity"):
        for i in range(10):
            engine.update(t + i * MS, micro.BOOK, 0, 100.0, 100.1, 1.0, 1.0)


def test_linear_heads():
    rng = np.random.default_rng(0)
    x, w, bias = rng.normal(size=32), rng.normal(size=(3, 32)), rng.normal(size=3)
    np.testing.assert_allclose(micro.linear_heads(x, w, bias), w @ x + bias, rtol=1e-13)


class CEvent(ct.Structure):
    _fields_ = [("receive_ns", ct.c_int64), ("kind", ct.c_int32), ("side", ct.c_int32),
                ("a", ct.c_double), ("b", ct.c_double), ("c", ct.c_double), ("d", ct.c_double)]


def test_c_abi_matches_binding():
    lib = ct.CDLL(montlok._core.__file__)
    lib.montlok_micro_schema.restype = ct.c_char_p
    lib.montlok_micro_event_size.restype = ct.c_size_t
    lib.montlok_micro_create.restype = ct.c_void_p
    lib.montlok_micro_create.argtypes = [ct.c_size_t]
    lib.montlok_micro_update.argtypes = [ct.c_void_p, ct.POINTER(CEvent)]
    lib.montlok_micro_snapshot.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double), ct.c_size_t]
    lib.montlok_micro_destroy.argtypes = [ct.c_void_p]
    assert lib.montlok_micro_schema().decode() == micro.SCHEMA
    assert lib.montlok_micro_event_size() == ct.sizeof(CEvent)
    columns = synthetic_events(2000, seed=3)
    features, ready = micro.FeatureEngine().replay(*columns)
    handle = lib.montlok_micro_create(65536)
    out = (ct.c_double * 32)()
    try:
        for i, event in enumerate(zip(*columns)):
            assert lib.montlok_micro_update(handle, ct.byref(CEvent(*[v.item() for v in event]))) == 0
            status = lib.montlok_micro_snapshot(handle, out, 32)
            assert status == int(ready[i])
            if status:
                np.testing.assert_array_equal(np.array(out), features[i])
    finally:
        lib.montlok_micro_destroy(handle)
