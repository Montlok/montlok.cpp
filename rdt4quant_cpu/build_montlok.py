#!/usr/bin/env python3
"""Build montlok.cpp without requiring Ninja.

    python build_montlok.py build_ext --inplace

The module contains every instruction-set variant of the kernels and picks one
at run time, so the default build runs on any CPU of its architecture.
``MONTLOK_MARCH`` adds ``-march=<value>`` (e.g. ``native``) for a build tied to
one machine. On macOS the extension is built for the running architecture
only: a universal2 Python would otherwise ask for an x86_64 slice as well,
which torch itself does not ship.
"""

import os
from pathlib import Path
import platform
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

from montlok_loader import HEADERS, PREBUILT_MODULE, SOURCES, compile_flags, link_flags

if sys.platform == "darwin" and "ARCHFLAGS" not in os.environ:
    os.environ["ARCHFLAGS"] = f"-arch {platform.machine()}"

root = Path(__file__).resolve().parent
setup(
    name=PREBUILT_MODULE,
    ext_modules=[
        CppExtension(
            name=PREBUILT_MODULE,
            sources=[str(path) for path in SOURCES],
            depends=[str(path) for path in HEADERS],
            include_dirs=[str(root)],
            extra_compile_args=compile_flags(),
            extra_link_args=link_flags(),
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
