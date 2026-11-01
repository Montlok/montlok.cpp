"""Build/load the local montlok.cpp PyTorch CPU extension once per process.

Resolution order:

1. A prebuilt ``montlok_cpp_v2`` module (``python build_montlok.py build_ext
   --inplace``).
2. A JIT build with ``torch.utils.cpp_extension.load``. The JIT module name
   embeds a hash of every source and header plus the compile flags, so every
   source change produces a fresh build.

One build runs on every CPU of its architecture: the kernels are compiled once
per instruction set (baseline SSE2, AVX2+FMA and AVX-512 on x86-64; NEON on
arm64) and the variant is picked at run time from cpuid (``montlok_isa.h``;
``MONTLOK_ISA`` or ``set_isa()`` override it). No ``-march`` flag is passed by
default. ``MONTLOK_MARCH`` adds ``-march=<value>`` (``native``, ``znver2``,
``x86-64-v3`` ...) for a machine-specific build; it then applies to the
baseline translation unit too, so such a build only runs on CPUs with those
features.

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

PREBUILT_MODULE = "montlok_cpp_v2"
SOURCE_DIR = Path(__file__).resolve().parent
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
        # Not linked explicitly: the libomp.dylib in the torch wheel carries an
        # absolute install name that does not exist on the machine. Python
        # extension modules on macOS are linked with -undefined dynamic_lookup,
        # so the OpenMP runtime symbols bind at load time to the libomp that
        # libtorch_cpu (a declared dependency) has already loaded.
        return []
    return ["-fopenmp"]


def link_flags() -> list[str]:
    """Linker flags shared by the JIT loader and build_montlok.py.

    The rpath lets the prebuilt module find libc10/libtorch even when it is
    imported before ``torch`` (setuptools does not add one on macOS).
    """
    lib = _torch_dir() / "lib"
    return [f"-Wl,-rpath,{lib}", *openmp_ldflags()]


def compile_flags() -> list[str]:
    """Flags shared by the JIT loader and build_montlok.py.

    ``-ffp-contract=fast`` lets both GCC and Clang fuse the recurrence into
    FMAs where the target has them (GCC already does this by default),
    ``-fno-math-errno`` removes the errno bookkeeping around exp/log/tanh so
    they can be inlined. No ``-ffast-math``: reassociation and flush semantics
    stay IEEE. No ``-march`` unless ``MONTLOK_MARCH`` asks for one: the AVX2
    and AVX-512 kernels are compiled with target pragmas and chosen at run time.
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
        extra_ldflags=link_flags(),
        extra_include_paths=[str(SOURCE_DIR)],
        with_cuda=False,
        verbose=os.environ.get("MONTLOK_VERBOSE_BUILD") == "1",
    )
