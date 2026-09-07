"""Build/load the local montlok.cpp PyTorch CPU extension once per process.

Resolution order:

1. A prebuilt ``montlok_cpp_v2`` module (``python build_montlok.py build_ext
   --inplace``).
2. A JIT build with ``torch.utils.cpp_extension.load``. The JIT module name
   embeds a hash of ``montlok.cpp`` + every ``montlok_*.h`` header + the
   compile flags, so every source change produces a fresh build.

``MONTLOK_MARCH`` overrides the ``-march`` value (default ``native``), e.g.
``znver2`` when cross-building for another x86-64 microarchitecture.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib
import os
from pathlib import Path
import sys

PREBUILT_MODULE = "montlok_cpp_v2"
SOURCE_DIR = Path(__file__).resolve().parent
SOURCES = [SOURCE_DIR / "montlok.cpp"]
HEADERS = [SOURCE_DIR / "montlok_kernel.h", SOURCE_DIR / "montlok_mhc.h", SOURCE_DIR / "montlok_layers.h"]


def compile_flags() -> list[str]:
    """Flags shared by the JIT loader and build_montlok.py.

    ``-ffp-contract=fast`` lets both GCC and Clang fuse the recurrence into
    FMAs (GCC already does this by default), ``-fno-math-errno`` removes the
    errno bookkeeping around exp/log/tanh so they can be inlined. No
    ``-ffast-math``: reassociation and flush semantics stay IEEE.
    """

    march = os.environ.get("MONTLOK_MARCH", "native")
    return ["-O3", f"-march={march}", "-DNDEBUG", "-fno-math-errno", "-ffp-contract=fast", *openmp_flags()]


def openmp_flags() -> list[str]:
    if sys.platform.startswith("linux") and os.environ.get("MONTLOK_OPENMP") != "0":
        return ["-fopenmp"]
    return []


def _jit_module_name(flags: list[str]) -> str:
    digest = hashlib.sha256()
    for path in [*SOURCES, *HEADERS]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(" ".join(flags).encode())
    return f"montlok_cpp_jit_{digest.hexdigest()[:12]}"


@lru_cache(maxsize=1)
def load_montlok():
    try:
        return importlib.import_module(PREBUILT_MODULE)
    except ImportError:
        pass
    from torch.utils.cpp_extension import load

    flags = compile_flags()
    return load(
        name=_jit_module_name(flags),
        sources=[str(path) for path in SOURCES],
        extra_cflags=flags,
        extra_ldflags=openmp_flags(),
        extra_include_paths=[str(SOURCE_DIR)],
        with_cuda=False,
        verbose=os.environ.get("MONTLOK_VERBOSE_BUILD") == "1",
    )
