"""Build the montlok Python package.

Two native modules:

* ``montlok._core`` - the runtime (model families, C ABI, plugin host,
  microstructure features).
* ``montlok._C`` - the RDT4quant kernels, a PyTorch extension built against the
  PyTorch importable at build time:

      pip install .                          # montlok._core
      pip install --no-build-isolation .     # montlok._core and montlok._C for the installed PyTorch

``MONTLOK_BUILD_TORCH=0`` skips montlok._C, ``=1`` makes a missing PyTorch an
error. Other flags (MONTLOK_MARCH, MONTLOK_OPENMP) are in python/montlok/rdt/_loader.py.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
import sysconfig
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension
from pybind11.setup_helpers import build_ext as pybind11_build_ext
from setuptools import setup

ROOT = Path(__file__).resolve().parent

# Build and tag the wheel for the running architecture (PyTorch for macOS is
# arm64).
if sys.platform == "darwin" and "ARCHFLAGS" not in os.environ:
    machine = platform.machine()
    os.environ["ARCHFLAGS"] = f"-arch {machine}"
    host = sysconfig.get_platform()
    if host.endswith("universal2") and "_PYTHON_HOST_PLATFORM" not in os.environ:
        version = "11.0" if machine == "arm64" else host.split("-")[1]
        os.environ["_PYTHON_HOST_PLATFORM"] = f"macosx-{version}-{machine}"


def relative(paths):
    return [str(Path(p).resolve().relative_to(ROOT)) for p in paths]


def core_extension() -> Pybind11Extension:
    sources = sorted((ROOT / "src" / "core").glob("*.cpp"))
    sources += [ROOT / "microstructure" / "bindings.cpp", ROOT / "src" / "python" / "module.cpp"]
    flags = [] if sys.platform == "win32" else ["-O3", "-fno-math-errno", "-ffp-contract=fast", "-fvisibility=hidden"]
    return Pybind11Extension(
        "montlok._core",
        relative(sources),
        include_dirs=["include", "microstructure"],
        depends=relative([*(ROOT / "src").rglob("*.hpp"), *(ROOT / "include").rglob("*.h*"),
                          *(ROOT / "microstructure").glob("*.h")]),
        cxx_std=17,
        extra_compile_args=flags,
        libraries=["dl"] if sys.platform.startswith("linux") else [],
    )


def torch_extension():
    mode = os.environ.get("MONTLOK_BUILD_TORCH", "auto")
    if mode == "0":
        return None, None
    try:
        from torch.utils.cpp_extension import BuildExtension, CppExtension
    except ImportError:
        if mode == "1":
            raise
        print("montlok: PyTorch is not importable at build time; montlok._C (RDT kernels) is skipped",
              file=sys.stderr)
        return None, None
    spec = importlib.util.spec_from_file_location("montlok_rdt_loader", ROOT / "python/montlok/rdt/_loader.py")
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    extension = CppExtension(
        "montlok._C",
        relative(loader.SOURCES),
        depends=relative(loader.HEADERS),
        include_dirs=[str(loader.SOURCE_DIR.relative_to(ROOT))],
        extra_compile_args=loader.compile_flags(),
        extra_link_args=loader.link_flags(),
    )
    return extension, BuildExtension.with_options(use_ninja=False)


extensions = [core_extension()]
torch_ext, torch_build = torch_extension()
if torch_ext is not None:
    extensions.append(torch_ext)

setup(ext_modules=extensions, cmdclass={"build_ext": torch_build or pybind11_build_ext})
