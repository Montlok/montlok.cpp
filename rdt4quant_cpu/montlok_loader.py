"""Build/load the local montlok.cpp PyTorch CPU extension once per process."""

from __future__ import annotations

from functools import lru_cache
import importlib
from pathlib import Path


@lru_cache(maxsize=1)
def load_montlok():
    try:
        return importlib.import_module("montlok_cpp_v1")
    except ImportError:
        pass
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("montlok.cpp")
    return load(
        name="montlok_cpp_v1",
        sources=[str(source)],
        extra_cflags=["-O3", "-march=native", "-DNDEBUG"],
        with_cuda=False,
        verbose=False,
    )
