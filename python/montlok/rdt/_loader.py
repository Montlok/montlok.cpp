"""Load the RDT4quant kernels of montlok.cpp (the PyTorch extension montlok._C).

Resolution order:

1. ``montlok._C``, built when montlok is installed with PyTorch present
   (``pip install --no-build-isolation .``; see setup.py).
2. In a source checkout, a JIT build with ``torch.utils.cpp_extension.load``
   from rdt4quant_cpu/. The JIT module name embeds a hash of every source and
   header plus the compile flags, so every source change produces a fresh
   build. ``MONTLOK_REQUIRE_PREBUILT=1`` restricts loading to step 1.

One build runs on every CPU of its architecture: the kernels are compiled once
per instruction set (baseline SSE2, AVX2+FMA and AVX-512 on x86-64; NEON on
arm64) and the variant is picked at run time from cpuid (``montlok_isa.h``;
``MONTLOK_ISA`` or ``set_isa()`` override it). ``MONTLOK_MARCH`` adds
``-march=<value>`` (``native``, ``znver2``, ``x86-64-v3`` ...) for a
machine-specific build; it then applies to the baseline translation unit too,
so such a build runs only on CPUs with those features.

OpenMP: on Linux the extension is compiled with ``-fopenmp`` and binds to the
libgomp/libomp the PyTorch wheel already loads; on macOS Apple clang gets
``-Xpreprocessor -fopenmp`` with the ``omp.h`` shipped inside the torch package
and binds at load time to the ``libomp.dylib`` libtorch_cpu loads. Either way ``at::parallel_for`` forks on torch's own
pool. ``MONTLOK_OPENMP=0`` disables this and the extension dispatches through
``TensorIterator`` instead (slightly more overhead per operator, same threads).
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib
import os
from pathlib import Path
import platform
import subprocess
import sys

PREBUILT_MODULE = "montlok._C"
# The kernel sources sit next to the Python package in a source checkout.
SOURCE_DIR = Path(__file__).resolve().parents[3] / "rdt4quant_cpu"
SOURCES = [
    SOURCE_DIR / name
    for name in (
        "montlok.cpp",
        "montlok_isa.cpp",
        "montlok_isa_baseline.cpp",
        "montlok_isa_avx2.cpp",
        "montlok_isa_avx512.cpp",
    )
]
HEADERS = [
    SOURCE_DIR / name
    for name in (
        "montlok_types.h",
        "montlok_isa.h",
        "montlok_isa_table.inc",
        "montlok_kernel.h",
        "montlok_mhc.h",
        "montlok_layers.h",
    )
]


def _torch_dir() -> Path:
    import torch

    return Path(torch.__file__).resolve().parent


def cxx_compiler() -> str:
    """The C++ compiler torch.utils.cpp_extension will invoke."""
    try:
        from torch.utils.cpp_extension import get_cxx_compiler

        return get_cxx_compiler()
    except ImportError:  # older torch
        return os.environ.get("CXX", "c++")


@lru_cache(maxsize=1)
def _compiler_is_clang() -> bool:
    try:
        version = subprocess.run([cxx_compiler(), "--version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "clang" in version.lower()


def is_x86() -> bool:
    return platform.machine().lower() in ("x86_64", "amd64", "i386", "i686", "x86")


def openmp_cflags() -> list[str]:
    """Compiler flags that let at::parallel_for fork inside the extension."""
    if os.environ.get("MONTLOK_OPENMP") == "0":
        return []
    if sys.platform.startswith("linux"):
        return ["-fopenmp"]
    if sys.platform == "darwin" and _compiler_is_clang():
        torch_dir = _torch_dir()
        if (torch_dir / "include" / "omp.h").exists() and (torch_dir / "lib" / "libomp.dylib").exists():
            return ["-Xpreprocessor", "-fopenmp"]
    return []


def openmp_ldflags() -> list[str]:
    if not openmp_cflags():
        return []
    if sys.platform == "darwin":
        # Extension modules on macOS link with -undefined dynamic_lookup: the
        # OpenMP symbols bind at load time to the libomp.dylib that
        # libtorch_cpu has loaded from the torch wheel.
        return []
    return ["-fopenmp"]


def link_flags() -> list[str]:
    """Linker flags shared by the JIT loader and setup.py.

    The rpath lets the prebuilt module find libc10/libtorch even when it is
    imported before ``torch`` (setuptools does not add one on macOS).
    """
    lib = _torch_dir() / "lib"
    return [f"-Wl,-rpath,{lib}", *openmp_ldflags()]


def compile_flags() -> list[str]:
    """Flags shared by the JIT loader and setup.py.

    ``-ffp-contract=fast`` lets GCC and Clang fuse the recurrence into FMAs
    where the target has them, and ``-fno-math-errno`` lets exp/log/tanh
    inline; reassociation and denormal handling stay IEEE. The AVX2 and
    AVX-512 kernels are compiled with target pragmas and chosen at run time.
    """

    flags = ["-O3", "-DNDEBUG", "-fno-math-errno", "-ffp-contract=fast"]
    march = os.environ.get("MONTLOK_MARCH")
    if march:
        flags.append(f"-march={march}")
    return [*flags, *openmp_cflags()]


def _jit_module_name(flags: list[str]) -> str:
    digest = hashlib.sha256()
    for path in [*SOURCES, *HEADERS]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(" ".join(flags).encode())
    return f"montlok_cpp_jit_{digest.hexdigest()[:12]}"


@lru_cache(maxsize=1)
def load_montlok():
    import torch  # noqa: F401  (the extension binds against torch's own types)

    try:
        return importlib.import_module(PREBUILT_MODULE)
    except ImportError as error:
        prebuilt_error = error
    if os.environ.get("MONTLOK_REQUIRE_PREBUILT") == "1":
        raise ImportError(
            "montlok._C is not built: reinstall montlok where PyTorch is installed "
            "(pip install --no-build-isolation montlok)"
        ) from prebuilt_error
    if not all(path.exists() for path in [*SOURCES, *HEADERS]):
        raise ImportError(
            "montlok._C is not built and the kernel sources for a JIT build are not available; "
            "reinstall montlok where PyTorch is installed (pip install --no-build-isolation montlok)"
        ) from prebuilt_error
    from torch.utils.cpp_extension import load

    flags = compile_flags()
    return load(
        name=_jit_module_name(flags),
        sources=[str(path) for path in SOURCES],
        extra_cflags=flags,
        extra_ldflags=link_flags(),
        extra_include_paths=[str(SOURCE_DIR)],
        with_cuda=False,
        verbose=os.environ.get("MONTLOK_VERBOSE_BUILD") == "1",
    )
