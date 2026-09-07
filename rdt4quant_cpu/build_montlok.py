#!/usr/bin/env python3
"""Build montlok.cpp without requiring Ninja.

    python build_montlok.py build_ext --inplace

``MONTLOK_MARCH`` selects the ``-march`` value (default ``native``), e.g.
``MONTLOK_MARCH=znver2`` when cross-building for another x86-64 microarchitecture.
"""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

from montlok_loader import HEADERS, PREBUILT_MODULE, SOURCES, compile_flags, openmp_flags


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
            extra_link_args=openmp_flags(),
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
