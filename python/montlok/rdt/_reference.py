"""Where the upstream RDT4quant ``Model`` package comes from.

The CPU classes subclass the upstream modules and keep their parameters and
state-dict keys, so the upstream package has to be importable as ``Model``.
The copy bundled under ``reference/`` is used unless ``use_reference(path)``
(or ``MONTLOK_RDT_REFERENCE``) names another tree before the first model
import, for example a host-reviewed research checkout whose file hashes the
caller has already verified. One process uses one tree.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BUNDLED = Path(__file__).resolve().parent / "reference"
_active: Path | None = None


def ensure_reference(path: str | os.PathLike | None = None) -> Path:
    """Make ``Model`` importable from `path` (default: env var, then bundled)."""
    global _active
    requested = Path(path or os.environ.get("MONTLOK_RDT_REFERENCE") or BUNDLED).resolve()
    if _active is not None:
        if path is not None and requested != _active:
            raise ImportError(f"the RDT reference is already {_active}; one process cannot switch to {requested}")
        return _active
    if not (requested / "Model" / "__init__.py").is_file():
        raise ImportError(f"no RDT4quant Model package under {requested}")
    loaded = sys.modules.get("Model")
    if loaded is not None:
        origin = Path(getattr(loaded, "__file__", None) or "").resolve()
        if not origin.is_relative_to(requested):
            raise ImportError(f"another Model package is already imported from {origin}")
    if str(requested) not in sys.path:
        sys.path.insert(0, str(requested))
    _active = requested
    return requested
