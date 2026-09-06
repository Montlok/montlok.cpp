#!/usr/bin/env python3
"""Build montlok.cpp without requiring Ninja."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


root = Path(__file__).resolve().parent
setup(
    name="montlok_cpp_v1",
    ext_modules=[
        CppExtension(
            name="montlok_cpp_v1",
            sources=[str(root / "montlok.cpp")],
            extra_compile_args=["-O3", "-march=native", "-DNDEBUG"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
