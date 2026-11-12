"""montlok - CPU runtime for quantitative trading models.

    import montlok

    model = montlok.load("gru.montlok")      # any family: sequential, tree_ensemble, plugins
    y = model.predict(x)                      # [batch, sequence, features] -> [batch, outputs]
    live = model.stream()                     # recurrent state carried from bar to bar
    y_now = live.step(features_now)           # [features] -> [outputs]

:mod:`montlok.export` writes model files from PyTorch, scikit-learn, LightGBM
and XGBoost models, :mod:`montlok.micro` computes streaming microstructure
features, and :mod:`montlok.rdt` runs RDT4quant checkpoints with fused kernels.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import platform
from typing import Any

from . import _core
from ._core import FormatError, Model, Stream, families, latency_threads, set_latency_threads, set_threads, threads

__version__: str = _core.version()

__all__ = [
    "FormatError",
    "Model",
    "Stream",
    "__version__",
    "build_info",
    "export",
    "families",
    "latency_threads",
    "load",
    "load_plugin",
    "micro",
    "rdt",
    "set_latency_threads",
    "set_threads",
    "threads",
]

_SUBMODULES = ("export", "micro", "rdt")


def load(source: Any) -> Model:
    """Open a model from a path, the file's bytes, or a binary file object."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        return Model.from_bytes(bytes(source))
    if hasattr(source, "read"):
        return Model.from_bytes(source.read())
    return Model.open(os.fspath(source))


def load_plugin(path: str | os.PathLike) -> None:
    """Load a shared library exporting ``montlok_plugin_init`` and register its
    model families for this process (see examples/plugin/ema_dot.c)."""
    _core.load_plugin(os.fspath(path))


def _spec(self: Model) -> dict:
    """The model spec as a dict (inputs, outputs, family config, metadata)."""
    return json.loads(self.spec_json)


Model.spec = property(_spec)  # type: ignore[attr-defined]


def build_info() -> dict:
    """Version, ABI, families and thread settings, for logs and status pages."""
    info: dict[str, Any] = {
        "version": __version__,
        "abi": _core.abi_version(),
        "families": families(),
        "threads": threads(),
        "latency_threads": latency_threads(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    rdt = importlib.util.find_spec("montlok._C")
    info["rdt_extension"] = rdt is not None
    return info


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'montlok' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SUBMODULES))
