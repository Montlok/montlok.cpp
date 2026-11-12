"""Model file format: safetensors compatibility, spec validation, damaged files."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

import montlok
from montlok import export


def tiny_linear() -> export.Exported:
    rng = np.random.default_rng(0)
    return export.sequential(
        [{"type": "linear", "weight": "w", "bias": "b"}],
        {"w": rng.normal(size=(2, 3)), "b": rng.normal(size=2)},
        features=3, outputs=2, precision="f64", feature_names=["a", "b", "c"],
        preprocess={"mean": [1.0, 2.0, 3.0], "scale": [2.0, 4.0, 8.0], "clip": (-3, 3)},
        metadata={"release": "unit"},
    )


def test_roundtrip_and_spec():
    exported = tiny_linear()
    data = exported.to_bytes()
    spec, tensors = export.read(data)
    assert spec == exported.spec
    assert set(tensors) == set(exported.tensors)
    for name, value in exported.tensors.items():
        np.testing.assert_array_equal(tensors[name], value)
    model = montlok.load(data)
    assert (model.family, model.features, model.sequence, model.outputs) == ("sequential", 3, 0, 2)
    assert model.preprocess and model.streamable
    assert model.spec["inputs"]["names"] == ["a", "b", "c"]
    assert model.spec["metadata"]["release"] == "unit"
    assert model.spec["metadata"]["exporter"].startswith("montlok ")


def test_tensors_are_aligned_and_readable_by_safetensors(tmp_path):
    safetensors_numpy = pytest.importorskip("safetensors.numpy")
    exported = tiny_linear()
    exported.tensors["mask"] = np.array([1, 0, 1], dtype=np.uint8)
    exported.tensors["index"] = np.array([7, 9], dtype=np.int32)
    path = exported.save(tmp_path / "m.montlok")
    data = path.read_bytes()
    (length,) = struct.unpack_from("<Q", data)
    assert (8 + length) % 8 == 0
    header = json.loads(data[8:8 + length])
    for name, entry in header.items():
        if name != "__metadata__":
            itemsize = np.dtype(export._DTYPES[entry["dtype"]]).itemsize
            assert entry["data_offsets"][0] % itemsize == 0, name
    loaded = safetensors_numpy.load(data)
    for name, value in exported.tensors.items():
        np.testing.assert_array_equal(loaded[name], value)
    montlok.load(path)  # the extra tensors do not disturb the model


def test_path_bytes_and_file_object(tmp_path):
    path = tiny_linear().save(tmp_path / "m.montlok")
    x = np.array([[1.0, 2.0, 3.0]])
    expected = montlok.load(path).predict(x)
    np.testing.assert_array_equal(montlok.load(str(path)).predict(x), expected)
    np.testing.assert_array_equal(montlok.load(path.read_bytes()).predict(x), expected)
    with path.open("rb") as handle:
        np.testing.assert_array_equal(montlok.load(handle).predict(x), expected)


def test_missing_file():
    with pytest.raises(OSError):
        montlok.load("/nonexistent/model.montlok")


def replace_spec(data: bytes, mutate) -> bytes:
    spec, tensors = export.read(data)
    mutate(spec)
    return export._encode(spec, tensors)


@pytest.mark.parametrize(
    "mutate, error",
    [
        (lambda s: s.update(schema="montlok.model/9"), NotImplementedError),
        (lambda s: s.update(family="no_such_family"), NotImplementedError),
        (lambda s: s["inputs"].update(features=4), ValueError),
        (lambda s: s["outputs"].update(size=3), ValueError),
        (lambda s: s["config"]["layers"].append({"type": "warp_drive"}), NotImplementedError),
        (lambda s: s["config"].update(precision="f16"), ValueError),
        (lambda s: s["preprocess"].update(clip=[3, -3]), ValueError),
        (lambda s: s["config"]["layers"][0].update(weight="nope"), ValueError),
        (lambda s: s["inputs"].update(names=["a"]), ValueError),
    ],
)
def test_invalid_specs_are_rejected(mutate, error):
    with pytest.raises(error):
        montlok.load(replace_spec(tiny_linear().to_bytes(), mutate))


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x01",
        struct.pack("<Q", 1 << 40) + b"{}",
        struct.pack("<Q", 2) + b"{}",
        struct.pack("<Q", 2) + b"[]",
        struct.pack("<Q", 5) + b"{\"a\":",
        struct.pack("<Q", 47) + b'{"t":{"dtype":"F64","shape":[2],"data_offsets":[0,16]}}',
        struct.pack("<Q", 56) + b'{"__metadata__":{"montlok":"{\\"schema\\":1}"},"x":1,"y":2}',
    ],
)
def test_garbage_is_a_clean_error(data):
    with pytest.raises((ValueError, NotImplementedError)):
        montlok.load(data)


def test_random_corruption_is_a_clean_error():
    data = bytearray(tiny_linear().to_bytes())
    rng = np.random.default_rng(1)
    for _ in range(600):
        mutated = bytearray(data)
        for position in rng.integers(0, len(mutated), size=rng.integers(1, 4)):
            mutated[position] = int(rng.integers(0, 256))
        try:
            montlok.load(bytes(mutated)).predict(np.zeros((1, 3)))
        except (ValueError, NotImplementedError, OSError):
            pass
    for cut in range(0, len(data), 7):
        with pytest.raises((ValueError, NotImplementedError)):
            montlok.load(bytes(data[:cut]))


def test_input_validation():
    model = tiny_linear().load()
    with pytest.raises(ValueError):
        model.predict(np.zeros((2, 4)))
    with pytest.raises(ValueError):
        model.predict(np.zeros((2, 3, 1)))
    assert model.predict(np.zeros(3)).shape == (2,)
    assert model.predict(np.zeros((0, 3))).shape == (0, 2)
    np.testing.assert_array_equal(model.predict(np.zeros((1, 3), dtype=np.int64)),
                                  model.predict(np.zeros((1, 3))))
    strided = np.zeros((4, 6))[:, ::2]
    np.testing.assert_array_equal(model.predict(strided), model.predict(np.zeros((4, 3))))


def test_build_info():
    info = montlok.build_info()
    assert info["version"] == montlok.__version__
    assert info["abi"] == 1
    assert {"sequential", "tree_ensemble"} <= set(info["families"])
