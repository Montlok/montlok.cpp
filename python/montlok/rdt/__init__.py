"""RDT4quant Mamba-3 CPU runtime: fused C++ kernels behind the upstream modules.

Needs PyTorch. Checkpoints load unchanged (``strict=True``)::

    from montlok import rdt

    rdt.use_reference("/reviewed/pipelines/rdt4quant/native")  # optional, before the model import
    model = rdt.MultiAssetRDTCPU(config, asset_count=31).eval()
    quantiles, hidden_rms = model(x, depth=4)

Importing this package enables the fused kernels unless ``MONTLOK_CPP`` is
already set; ``MONTLOK_CPP=0`` keeps the pure PyTorch reference path. The
other ``MONTLOK_CPP_*`` switches are described in the README.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from ._reference import BUNDLED, ensure_reference

use_reference = ensure_reference
REFERENCE_CONFIG = BUNDLED / "config.json"

os.environ.setdefault("MONTLOK_CPP", "1")

_EXPORTS = {
    "MultiAssetRDTCPU": "model",
    "Mamba3CPUReference": "mamba3",
    "ManifoldHyperConnectionCPU": "mhc",
    "MLACPU": "mla",
    "install_cpu_layers": "layers",
    "cpu_linear": "linear",
}

__all__ = ["REFERENCE_CONFIG", "build_info", "use_reference", *_EXPORTS]


def build_info() -> dict:
    """Kernel variant, ISA list, CPU features and thread dispatch of montlok._C."""
    from ._loader import load_montlok

    return dict(load_montlok().build_info())


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module 'montlok.rdt' has no attribute {name!r}")
