"""Write montlok model files and convert trained models into them.

A montlok model is a safetensors file: every weight is a named tensor, and
``__metadata__["montlok"]`` holds the model spec (schema ``montlok.model/1``)::

    {"schema": "montlok.model/1", "family": "sequential",
     "inputs": {"features": 40, "sequence": 32, "names": [...]},
     "outputs": {"size": 1},
     "preprocess": {"mean": "preprocess.mean", "scale": "preprocess.scale", "clip": [-8, 8]},
     "config": {"precision": "f32", "layers": [...]},
     "metadata": {...}}

The converters take PyTorch, scikit-learn, LightGBM and XGBoost models, or
their dumps::

    from montlok import export
    export.from_torch([net.gru, export.LAST, net.head], features=40, sequence=32,
                      preprocess={"mean": mean, "scale": scale, "clip": (-8, 8)}).save("gru.montlok")
    export.from_sklearn(pipeline).save("ridge.montlok")
    export.from_lightgbm(booster).save("lgbm.montlok")
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "FLATTEN",
    "LAST",
    "MEAN",
    "SCHEMA",
    "Exported",
    "from_lightgbm",
    "from_sklearn",
    "from_torch",
    "from_xgboost",
    "make",
    "read",
    "sequential",
    "tree_ensemble",
]

SCHEMA = "montlok.model/1"

# Markers for from_torch: reduce [batch, steps, width] to one step.
LAST = "last"
MEAN = "mean"
FLATTEN = "flatten"

_CODES = {
    np.dtype("float64"): "F64",
    np.dtype("float32"): "F32",
    np.dtype("int64"): "I64",
    np.dtype("int32"): "I32",
    np.dtype("uint8"): "U8",
    np.dtype("bool"): "BOOL",
}
_DTYPES = {code: dtype for dtype, code in _CODES.items()}


# ---------------------------------------------------------------------------
# File format
# ---------------------------------------------------------------------------


def _encode(spec: Mapping[str, Any], tensors: Mapping[str, Any]) -> bytes:
    arrays: dict[str, np.ndarray] = {}
    for name, value in tensors.items():
        array = np.asarray(value)
        if array.dtype not in _CODES:
            raise TypeError(f"tensor {name!r} has unsupported dtype {array.dtype}")
        arrays[name] = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    # Wider elements first: every tensor then starts at a multiple of its
    # element size, and the 8-byte aligned data section keeps that in memory.
    order = sorted(arrays, key=lambda n: (-arrays[n].dtype.itemsize, n))
    header: dict[str, Any] = {}
    offset = 0
    for name in order:
        array = arrays[name]
        header[name] = {"dtype": _CODES[array.dtype.newbyteorder("=")], "shape": list(array.shape),
                        "data_offsets": [offset, offset + array.nbytes]}
        offset += array.nbytes
    header["__metadata__"] = {"montlok": json.dumps(spec, separators=(",", ":"), allow_nan=False,
                                                    ensure_ascii=False)}
    text = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    text += b" " * (-(8 + len(text)) % 8)
    return struct.pack("<Q", len(text)) + text + b"".join(arrays[name].tobytes() for name in order)


def read(source: str | os.PathLike | bytes) -> tuple[dict, dict[str, np.ndarray]]:
    """Return ``(spec, tensors)`` of a montlok model file, tensors as NumPy arrays."""
    data = bytes(source) if isinstance(source, (bytes, bytearray, memoryview)) else Path(source).read_bytes()
    (length,) = struct.unpack_from("<Q", data, 0)
    header = json.loads(data[8:8 + length])
    metadata = header.pop("__metadata__", {})
    if "montlok" not in metadata:
        raise ValueError("not a montlok model: __metadata__.montlok is missing")
    base = 8 + length
    tensors = {}
    for name, entry in header.items():
        dtype = np.dtype(_DTYPES[entry["dtype"]]).newbyteorder("<")
        begin, end = entry["data_offsets"]
        array = np.frombuffer(data, dtype=dtype, count=(end - begin) // dtype.itemsize, offset=base + begin)
        tensors[name] = array.reshape(entry["shape"]).astype(dtype.newbyteorder("="))
    return json.loads(metadata["montlok"]), tensors


@dataclass
class Exported:
    """A model spec and its tensors, ready to write or to load."""

    spec: dict
    tensors: dict[str, np.ndarray] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        return _encode(self.spec, self.tensors)

    def save(self, path: str | os.PathLike) -> Path:
        """Write atomically (temporary file + rename)."""
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(self.to_bytes())
        os.replace(temporary, path)
        return path

    def load(self):
        """Build the ``montlok.Model`` in memory."""
        from . import load

        return load(self.to_bytes())

    def with_metadata(self, **items: Any) -> "Exported":
        spec = dict(self.spec)
        spec["metadata"] = {**spec.get("metadata", {}), **items}
        return Exported(spec, dict(self.tensors))


def _version() -> str:
    try:
        from . import __version__

        return __version__
    except ImportError:  # pragma: no cover
        return "unknown"


def make(family: str, config: Mapping[str, Any], tensors: Mapping[str, Any], *, features: int,
         outputs: int, sequence: int = 0, preprocess: Mapping[str, Any] | None = None,
         feature_names: Sequence[str] | None = None, output_names: Sequence[str] | None = None,
         metadata: Mapping[str, Any] | None = None) -> Exported:
    """Assemble a model of any registered family, plugins included.

    ``preprocess`` takes ``mean`` and ``scale`` (one value per feature) and/or
    ``clip=(low, high)``; inputs become ``clip((x - mean) / scale)`` in float64
    before the family sees them, matching the engine's feature contracts.
    """
    tensors = dict(tensors)
    spec: dict[str, Any] = {
        "schema": SCHEMA,
        "family": family,
        "inputs": {"features": int(features), "sequence": int(sequence)},
        "outputs": {"size": int(outputs)},
        "config": dict(config),
    }
    if feature_names is not None:
        names = [str(name) for name in feature_names]
        if len(names) != features:
            raise ValueError(f"{len(names)} feature names for {features} features")
        spec["inputs"]["names"] = names
    if output_names is not None:
        names = [str(name) for name in output_names]
        if len(names) != outputs:
            raise ValueError(f"{len(names)} output names for {outputs} outputs")
        spec["outputs"]["names"] = names
    if preprocess:
        block: dict[str, Any] = {}
        mean, scale = preprocess.get("mean"), preprocess.get("scale")
        if (mean is None) != (scale is None):
            raise ValueError("preprocess needs both mean and scale")
        if mean is not None:
            mean = np.asarray(mean, dtype=np.float64).reshape(-1)
            scale = np.asarray(scale, dtype=np.float64).reshape(-1)
            if mean.shape != (features,) or scale.shape != (features,):
                raise ValueError("preprocess mean/scale need one value per feature")
            if not (np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all()):
                raise ValueError("preprocess mean must be finite and scale finite and positive")
            tensors["preprocess.mean"], tensors["preprocess.scale"] = mean, scale
            block.update(mean="preprocess.mean", scale="preprocess.scale")
        clip = preprocess.get("clip")
        if clip is not None:
            low, high = (float(v) for v in clip)
            if not low < high:
                raise ValueError("preprocess clip needs low < high")
            block["clip"] = [low, high]
        spec["preprocess"] = block
    spec["metadata"] = {"exporter": f"montlok {_version()}", **(metadata or {})}
    return Exported(spec, tensors)


def sequential(layers: Sequence[Mapping[str, Any]], tensors: Mapping[str, Any], *, features: int,
               outputs: int, sequence: int = 0, precision: str = "f32", **kwargs: Any) -> Exported:
    """A ``sequential`` model from explicit layer specs."""
    if precision not in ("f32", "f64"):
        raise ValueError("precision must be 'f32' or 'f64'")
    return make("sequential", {"precision": precision, "layers": [dict(layer) for layer in layers]}, tensors,
                features=features, outputs=outputs, sequence=sequence, **kwargs)


def _f32(tensor: Any) -> np.ndarray:
    return tensor.detach().to("cpu").float().numpy().astype(np.float32, copy=True)


def _f64(value: Any) -> np.ndarray:
    return np.array(value, dtype=np.float64, copy=True)


# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------


def _torch_items(modules: Any) -> list:
    from torch import nn

    items: list = []
    stack = [modules] if not isinstance(modules, (list, tuple)) else list(modules)
    for item in stack:
        if isinstance(item, (list, tuple)):
            items.extend(_torch_items(item))
        elif isinstance(item, nn.Sequential):
            items.extend(_torch_items(list(item.children())))
        else:
            items.append(item)
    return items


def from_torch(modules: Any, *, features: int, sequence: int = 0, precision: str = "f32",
               preprocess: Mapping[str, Any] | None = None, feature_names: Sequence[str] | None = None,
               output_names: Sequence[str] | None = None, metadata: Mapping[str, Any] | None = None) -> Exported:
    """Export a chain of PyTorch modules as a ``sequential`` model.

    ``modules`` is a module or a list of modules and the markers ``LAST``,
    ``MEAN`` and ``FLATTEN``. Supported: Linear, LayerNorm, RMSNorm, GRU,
    LSTM (unidirectional, ``proj_size=0``), ReLU, LeakyReLU, ELU, GELU, Tanh,
    Sigmoid, SiLU, Softplus (beta 1, threshold 20), Softmax over the last
    dimension, Flatten(1, -1); Dropout and Identity are skipped. A model whose
    forward does more than chain these has to be listed by its parts, e.g.
    ``[net.gru, LAST, net.head]`` for ``head(gru(x)[1][-1])``.
    """
    import torch
    from torch import nn

    steps, width = max(int(sequence), 1), int(features)
    layers: list[dict] = []
    tensors: dict[str, np.ndarray] = {}
    for index, item in enumerate(_torch_items(modules)):
        p = f"l{index}"
        if isinstance(item, str):
            if item not in (LAST, MEAN, FLATTEN):
                raise ValueError(f"unknown marker {item!r}")
            layers.append({"type": item})
            width = width * steps if item == FLATTEN else width
            steps = 1
        elif isinstance(item, (nn.Dropout, nn.Identity)):
            continue
        elif isinstance(item, nn.Linear):
            if item.in_features != width:
                raise ValueError(f"{p}: Linear expects {item.in_features} inputs, previous layer gives {width}")
            tensors[f"{p}.weight"] = _f32(item.weight)
            layer = {"type": "linear", "weight": f"{p}.weight"}
            if item.bias is not None:
                tensors[f"{p}.bias"] = _f32(item.bias)
                layer["bias"] = f"{p}.bias"
            layers.append(layer)
            width = item.out_features
        elif isinstance(item, nn.LayerNorm) or (hasattr(nn, "RMSNorm") and isinstance(item, nn.RMSNorm)):
            rms = not isinstance(item, nn.LayerNorm)
            if tuple(item.normalized_shape) != (width,):
                raise ValueError(f"{p}: only normalization over the last dimension ({width}) is supported")
            eps = item.eps if item.eps is not None else float(torch.finfo(torch.float32).eps)
            layer = {"type": "rmsnorm" if rms else "layernorm", "eps": float(eps)}
            if getattr(item, "weight", None) is not None:
                tensors[f"{p}.weight"] = _f32(item.weight)
                layer["weight"] = f"{p}.weight"
            if not rms and getattr(item, "bias", None) is not None:
                tensors[f"{p}.bias"] = _f32(item.bias)
                layer["bias"] = f"{p}.bias"
            layers.append(layer)
        elif isinstance(item, (nn.GRU, nn.LSTM)):
            if item.bidirectional:
                raise ValueError(f"{p}: bidirectional recurrent layers are not supported")
            if isinstance(item, nn.LSTM) and item.proj_size:
                raise ValueError(f"{p}: LSTM projections are not supported")
            if item.input_size != width:
                raise ValueError(f"{p}: {type(item).__name__} expects {item.input_size} inputs, previous layer gives {width}")
            for k in range(item.num_layers):
                for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    name = f"{kind}_l{k}"
                    if getattr(item, name, None) is not None:
                        tensors[f"{p}.{name}"] = _f32(getattr(item, name))
            layers.append({"type": "gru" if isinstance(item, nn.GRU) else "lstm", "hidden": item.hidden_size,
                           "layers": item.num_layers, "weights": p})
            width = item.hidden_size
        elif isinstance(item, nn.ReLU):
            layers.append({"type": "relu"})
        elif isinstance(item, nn.LeakyReLU):
            layers.append({"type": "leaky_relu", "alpha": float(item.negative_slope)})
        elif isinstance(item, nn.ELU):
            layers.append({"type": "elu", "alpha": float(item.alpha)})
        elif isinstance(item, nn.GELU):
            layers.append({"type": "gelu_tanh" if item.approximate == "tanh" else "gelu"})
        elif isinstance(item, nn.Tanh):
            layers.append({"type": "tanh"})
        elif isinstance(item, nn.Sigmoid):
            layers.append({"type": "sigmoid"})
        elif isinstance(item, nn.SiLU):
            layers.append({"type": "silu"})
        elif isinstance(item, nn.Softplus):
            if item.beta != 1 or item.threshold != 20:
                raise ValueError(f"{p}: only Softplus(beta=1, threshold=20) is supported")
            layers.append({"type": "softplus"})
        elif isinstance(item, nn.Softmax):
            # Input is [batch, width] after a reduction, [batch, steps, width] before.
            last = 1 if steps == 1 else 2
            if item.dim not in (-1, last) and not (item.dim is None and steps == 1):
                raise ValueError(f"{p}: Softmax must run over the last dimension")
            layers.append({"type": "softmax"})
        elif isinstance(item, nn.Flatten):
            if (item.start_dim, item.end_dim) != (1, -1):
                raise ValueError(f"{p}: only Flatten(1, -1) is supported")
            layers.append({"type": FLATTEN})
            width, steps = width * steps, 1
        else:
            raise TypeError(f"{p}: cannot export {type(item).__name__}; list its supported parts, or implement "
                            "the model as a plugin family")
    if steps != 1:
        raise ValueError("the model keeps a time axis; add montlok.export.LAST, MEAN or FLATTEN")
    return sequential(layers, tensors, features=features, sequence=sequence, outputs=width, precision=precision,
                      preprocess=preprocess, feature_names=feature_names, output_names=output_names,
                      metadata={"source": "torch", **(metadata or {})})


# ---------------------------------------------------------------------------
# Tree ensembles
# ---------------------------------------------------------------------------

# flags: bit 0 = missing goes left; bits 1-2 = missing mode.
_AS_ZERO, _NAN_DEFAULT, _ZERO_OR_NAN_DEFAULT = 0, 1, 2


class _Nodes:
    def __init__(self) -> None:
        self.feature: list[int] = []
        self.threshold: list[float] = []
        self.left: list[int] = []
        self.right: list[int] = []
        self.value: list[float] = []
        self.flags: list[int] = []

    def add(self, feature: int = -1, threshold: float = 0.0, value: float = 0.0, flags: int = 0) -> int:
        self.feature.append(int(feature))
        self.threshold.append(float(threshold))
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(float(value))
        self.flags.append(int(flags))
        return len(self.feature) - 1

    def tensors(self, prefix: str = "trees") -> dict[str, np.ndarray]:
        return {
            f"{prefix}.feature": np.asarray(self.feature, dtype=np.int32),
            f"{prefix}.threshold": np.asarray(self.threshold, dtype=np.float64),
            f"{prefix}.left": np.asarray(self.left, dtype=np.int32),
            f"{prefix}.right": np.asarray(self.right, dtype=np.int32),
            f"{prefix}.value": np.asarray(self.value, dtype=np.float64),
            f"{prefix}.flags": np.asarray(self.flags, dtype=np.uint8),
        }


def tree_ensemble(nodes: _Nodes, roots: Sequence[int], tree_outputs: Sequence[int], *, features: int,
                  outputs: int = 1, decision: str = "le", feature_cast: str = "f64", aggregate: str = "sum",
                  link: str = "identity", base_score: float | Sequence[float] = 0.0, sigmoid_scale: float = 1.0,
                  **kwargs: Any) -> Exported:
    """A ``tree_ensemble`` model from flattened nodes (see src/core/trees.cpp)."""
    tensors = nodes.tensors()
    tensors["trees.root"] = np.asarray(roots, dtype=np.int32)
    tensors["trees.output"] = np.asarray(tree_outputs, dtype=np.int32)
    base = [float(base_score)] * outputs if np.ndim(base_score) == 0 else [float(v) for v in base_score]
    config = {"decision": decision, "feature_cast": feature_cast, "aggregate": aggregate, "link": link,
              "base_score": base, "sigmoid_scale": float(sigmoid_scale)}
    return make("tree_ensemble", config, tensors, features=features, outputs=outputs, **kwargs)


def _sk_tree(nodes: _Nodes, tree: Any, scale: float = 1.0) -> int:
    """Append one fitted sklearn ``Tree``; returns its root index."""
    if tree.n_outputs != 1:
        raise ValueError("multi-output trees are not supported")
    offset = len(nodes.feature)
    missing_left = getattr(tree, "missing_go_to_left", None)
    values = tree.value[:, 0, 0]
    for i in range(tree.node_count):
        if tree.children_left[i] == -1:
            nodes.add(value=scale * values[i] if scale != 1.0 else values[i])
        else:
            default_left = int(missing_left[i]) if missing_left is not None else 1
            index = nodes.add(feature=tree.feature[i], threshold=tree.threshold[i],
                              flags=default_left | (_NAN_DEFAULT << 1))
            nodes.left[index] = offset + int(tree.children_left[i])
            nodes.right[index] = offset + int(tree.children_right[i])
    return offset


def _hgb_predictor(nodes: _Nodes, predictor: Any) -> int:
    array = predictor.nodes
    if array["is_categorical"].any():
        raise ValueError("categorical HistGradientBoosting splits are not supported")
    offset = len(nodes.feature)
    for row in array:
        if row["is_leaf"]:
            nodes.add(value=float(row["value"]))
        else:
            index = nodes.add(feature=int(row["feature_idx"]), threshold=float(row["num_threshold"]),
                              flags=int(row["missing_go_to_left"]) | (_NAN_DEFAULT << 1))
            nodes.left[index] = offset + int(row["left"])
            nodes.right[index] = offset + int(row["right"])
    return offset


def _sk_linear(coef: Any, intercept: Any, link: str, *, features: int, **kwargs: Any) -> Exported:
    weight = np.atleast_2d(np.asarray(coef, dtype=np.float64))
    bias = np.atleast_1d(np.asarray(intercept, dtype=np.float64)) * np.ones(weight.shape[0])
    if weight.shape[1] != features:
        raise ValueError("coefficients do not match the number of features")
    layers: list[dict] = [{"type": "linear", "weight": "linear.weight", "bias": "linear.bias"}]
    if link != "identity":
        layers.append({"type": link})
    return sequential(layers, {"linear.weight": weight, "linear.bias": bias}, features=features,
                      outputs=weight.shape[0], precision="f64", **kwargs)


def from_sklearn(estimator: Any, *, feature_names: Sequence[str] | None = None,
                 metadata: Mapping[str, Any] | None = None) -> Exported:
    """Export a fitted scikit-learn regressor (or a Pipeline ending in one).

    Supported: linear regressors with ``coef_``/``intercept_`` (LinearRegression,
    Ridge, Lasso, ElasticNet, SGDRegressor, ...), LogisticRegression
    (probabilities), MLPRegressor, DecisionTree/ExtraTree, RandomForest/ExtraTrees,
    GradientBoosting and HistGradientBoosting regressors. A leading
    StandardScaler becomes the model's float64 preprocessing.
    """
    preprocess = None
    if type(estimator).__name__ == "Pipeline":
        steps = [step for _, step in estimator.steps if step not in (None, "passthrough")]
        if len(steps) == 2 and type(steps[0]).__name__ == "StandardScaler":
            scaler = steps[0]
            n = scaler.n_features_in_
            mean = scaler.mean_ if scaler.with_mean else np.zeros(n)
            scale = scaler.scale_ if scaler.with_std else np.ones(n)
            preprocess = {"mean": mean, "scale": scale}
            estimator = steps[1]
        elif len(steps) != 1:
            raise TypeError("only Pipeline([StandardScaler, model]) or a single estimator are supported")
        else:
            estimator = steps[0]
    name = type(estimator).__name__
    features = int(estimator.n_features_in_)
    if feature_names is None and hasattr(estimator, "feature_names_in_"):
        feature_names = [str(f) for f in estimator.feature_names_in_]
    common = dict(feature_names=feature_names, preprocess=preprocess,
                  metadata={"source": f"sklearn.{name}", **(metadata or {})})

    if name in ("DecisionTreeRegressor", "ExtraTreeRegressor"):
        nodes = _Nodes()
        root = _sk_tree(nodes, estimator.tree_)
        return tree_ensemble(nodes, [root], [0], features=features, feature_cast="f32", **common)
    if name in ("RandomForestRegressor", "ExtraTreesRegressor"):
        nodes = _Nodes()
        roots = [_sk_tree(nodes, tree.tree_) for tree in estimator.estimators_]
        return tree_ensemble(nodes, roots, [0] * len(roots), features=features, feature_cast="f32",
                             aggregate="mean", **common)
    if name == "GradientBoostingRegressor":
        init = estimator.init_
        if isinstance(init, str) and init == "zero":
            base = 0.0
        elif type(init).__name__ == "DummyRegressor":
            base = float(np.asarray(init.constant_).reshape(-1)[0])
        else:
            raise TypeError("GradientBoostingRegressor with a non-constant init estimator is not supported")
        nodes = _Nodes()
        lr = float(estimator.learning_rate)
        roots = [_sk_tree(nodes, stage[0].tree_, scale=lr) for stage in estimator.estimators_]
        return tree_ensemble(nodes, roots, [0] * len(roots), features=features, feature_cast="f32",
                             base_score=base, **common)
    if name == "HistGradientBoostingRegressor":
        loss = getattr(estimator, "loss", "squared_error")
        link = "exp" if loss in ("poisson", "gamma") else "identity"
        nodes = _Nodes()
        roots = [_hgb_predictor(nodes, predictors[0]) for predictors in estimator._predictors]
        base = float(np.asarray(estimator._baseline_prediction).reshape(-1)[0])
        return tree_ensemble(nodes, roots, [0] * len(roots), features=features, base_score=base, link=link,
                             **common)
    if name == "LogisticRegression":
        classes = len(estimator.classes_)
        return _sk_linear(estimator.coef_, estimator.intercept_, "sigmoid" if classes == 2 else "softmax",
                          features=features, **common)
    if name == "MLPRegressor":
        activation = {"relu": "relu", "tanh": "tanh", "logistic": "sigmoid", "identity": None}[estimator.activation]
        layers, tensors = [], {}
        for i, (w, b) in enumerate(zip(estimator.coefs_, estimator.intercepts_)):
            tensors[f"l{i}.weight"] = _f64(np.asarray(w).T)
            tensors[f"l{i}.bias"] = _f64(b)
            layers.append({"type": "linear", "weight": f"l{i}.weight", "bias": f"l{i}.bias"})
            if i + 1 < len(estimator.coefs_) and activation:
                layers.append({"type": activation})
        return sequential(layers, tensors, features=features, outputs=int(estimator.n_outputs_), precision="f64",
                          **common)
    if hasattr(estimator, "coef_") and hasattr(estimator, "intercept_") and not hasattr(estimator, "classes_"):
        return _sk_linear(estimator.coef_, estimator.intercept_, "identity", features=features, **common)
    raise TypeError(f"cannot export {name}")


def _lightgbm_dump(model: Any) -> dict:
    if isinstance(model, Mapping):
        return dict(model)
    if isinstance(model, (str, bytes)):
        return json.loads(model)
    booster = getattr(model, "booster_", model)
    if hasattr(booster, "dump_model"):
        return booster.dump_model()
    raise TypeError("expected a LightGBM Booster/LGBMModel, its dump_model() dict, or that dict as JSON")


def from_lightgbm(model: Any, *, feature_names: Sequence[str] | None = None,
                  metadata: Mapping[str, Any] | None = None) -> Exported:
    """Export a LightGBM model with numerical splits. Binary and multiclass
    objectives predict probabilities."""
    dump = _lightgbm_dump(model)
    if dump.get("is_linear"):
        raise ValueError("linear trees are not supported")
    features = int(dump["max_feature_idx"]) + 1
    per_iteration = int(dump.get("num_tree_per_iteration", 1))
    objective = str(dump.get("objective", "regression")).split()
    kind = objective[0] if objective else "regression"
    params = dict(token.split(":", 1) for token in objective[1:] if ":" in token)
    link, scale = "identity", 1.0
    if kind in ("binary", "cross_entropy", "xentropy"):
        link, scale = "sigmoid", float(params.get("sigmoid", 1.0))
    elif kind in ("multiclass", "softmax"):
        link = "softmax"
    elif kind in ("poisson", "gamma", "tweedie"):
        link = "exp"
    elif kind in ("multiclassova", "multiclass_ova", "ova", "ovr"):
        raise ValueError("one-vs-all multiclass objectives are not supported")
    nodes = _Nodes()
    roots, tree_outputs = [], []
    missing = {"None": _AS_ZERO, "NaN": _NAN_DEFAULT, "Zero": _ZERO_OR_NAN_DEFAULT}
    for t, info in enumerate(dump["tree_info"]):
        roots.append(len(nodes.feature))
        tree_outputs.append(t % per_iteration)
        pending = [(info["tree_structure"], None, False)]
        while pending:
            node, parent, is_right = pending.pop()
            if "split_feature" in node:
                if node.get("decision_type", "<=") != "<=":
                    raise ValueError("categorical LightGBM splits are not supported")
                flags = (1 if node.get("default_left", True) else 0) | (missing[node.get("missing_type", "None")] << 1)
                index = nodes.add(feature=node["split_feature"], threshold=float(node["threshold"]), flags=flags)
                # Right is pushed first so the left subtree is numbered first.
                pending.append((node["right_child"], index, True))
                pending.append((node["left_child"], index, False))
            else:
                if "leaf_coeff" in node:
                    raise ValueError("linear trees are not supported")
                index = nodes.add(value=float(node["leaf_value"]))
            if parent is not None:
                if is_right:
                    nodes.right[parent] = index
                else:
                    nodes.left[parent] = index
    names = feature_names if feature_names is not None else dump.get("feature_names")
    return tree_ensemble(nodes, roots, tree_outputs, features=features, outputs=per_iteration, link=link,
                         sigmoid_scale=scale, aggregate="mean" if dump.get("average_output") else "sum",
                         feature_names=names, metadata={"source": "lightgbm", "objective": " ".join(objective),
                                                         **(metadata or {})})


_XGB_LOGISTIC = ("binary:logistic", "reg:logistic")
_XGB_LOG_LINK = ("count:poisson", "reg:gamma", "reg:tweedie", "survival:cox")


def _xgboost_margin(objective: str, base_score: float) -> float:
    if objective in _XGB_LOGISTIC:
        return math.log(base_score / (1.0 - base_score))
    if objective in _XGB_LOG_LINK:
        return math.log(base_score)
    return base_score


def from_xgboost(model: Any, *, features: int | None = None, objective: str | None = None,
                 base_score: float | None = None, base_margin: float | None = None, classes: int | None = None,
                 parallel_trees: int = 1, feature_names: Sequence[str] | None = None,
                 metadata: Mapping[str, Any] | None = None) -> Exported:
    """Export an XGBoost gbtree model.

    Pass a Booster/XGBModel, or the list of JSON tree dumps with ``features``,
    ``objective`` and ``base_score`` (prediction space) or ``base_margin``.
    Leaves are summed in float64, XGBoost's own sums in float32; the two agree
    to about 1e-6 relative.
    """
    if hasattr(model, "get_booster"):
        model = model.get_booster()
    if hasattr(model, "get_dump"):
        config = json.loads(model.save_config())
        learner = config["learner"]
        objective = objective or learner["objective"]["name"]
        params = learner["learner_model_param"]
        base_score = float(str(params["base_score"]).strip("[]")) if base_score is None else base_score
        features = features or int(params["num_feature"])
        classes = classes or int(params.get("num_class", 0))
        booster_params = learner.get("gradient_booster", {}).get("gbtree_model_param", {})
        parallel_trees = int(booster_params.get("num_parallel_tree", parallel_trees))
        feature_names = feature_names or model.feature_names
        dumps = model.get_dump(dump_format="json")
    else:
        dumps = list(model)
    if features is None or objective is None or (base_score is None and base_margin is None):
        raise ValueError("tree dumps need features=, objective= and base_score= or base_margin=")
    margin = base_margin if base_margin is not None else _xgboost_margin(objective, float(base_score))
    outputs = classes if classes and classes > 1 else 1
    index_of = {str(n): i for i, n in enumerate(feature_names)} if feature_names else {}

    def feature_index(split: str) -> int:
        if split in index_of:
            return index_of[split]
        match = re.fullmatch(r"f(\d+)", split)
        if not match:
            raise ValueError(f"unknown feature {split!r}; pass feature_names=")
        return int(match.group(1))

    nodes = _Nodes()
    roots, tree_outputs = [], []
    for t, dump in enumerate(dumps):
        tree = json.loads(dump) if isinstance(dump, (str, bytes)) else dump
        roots.append(len(nodes.feature))
        tree_outputs.append((t // parallel_trees) % outputs)
        pending = [(tree, None, False)]
        while pending:
            node, parent, is_right = pending.pop()
            if "children" in node:
                children = {child["nodeid"]: child for child in node["children"]}
                default_left = 1 if node["missing"] == node["yes"] else 0
                threshold = float(np.float32(node["split_condition"]))
                index = nodes.add(feature=feature_index(str(node["split"])), threshold=threshold,
                                  flags=default_left | (_NAN_DEFAULT << 1))
                pending.append((children[node["no"]], index, True))
                pending.append((children[node["yes"]], index, False))
            else:
                index = nodes.add(value=float(node["leaf"]))
            if parent is not None:
                if is_right:
                    nodes.right[parent] = index
                else:
                    nodes.left[parent] = index
    link = "identity"
    if objective in _XGB_LOGISTIC:
        link = "sigmoid"
    elif objective.startswith("multi:"):
        link = "softmax"
    elif objective in _XGB_LOG_LINK:
        link = "exp"
    return tree_ensemble(nodes, roots, tree_outputs, features=int(features), outputs=outputs, decision="lt",
                         feature_cast="f32", link=link, base_score=margin, feature_names=feature_names,
                         metadata={"source": "xgboost", "objective": objective, **(metadata or {})})
