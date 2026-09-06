#!/usr/bin/env python3
"""Compare montlok.cpp recurrence with the PyTorch quadratic reference."""

from __future__ import annotations

import os

import torch

from mamba3_cpu import Mamba3CPUReference


def main() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(8)
    model = Mamba3CPUReference(64, d_state=16, expand=2, headdim=16).eval()
    values = torch.randn(2, 48, 64)
    os.environ.pop("MONTLOK_CPP", None)
    with torch.inference_mode():
        reference = model(values)
    os.environ["MONTLOK_CPP"] = "1"
    with torch.inference_mode():
        native = model(values)
    difference = (reference - native).abs()
    print(
        {
            "max_abs_difference": float(difference.max()),
            "mean_abs_difference": float(difference.mean()),
            "allclose_1e-4": bool(torch.allclose(reference, native, atol=1e-4, rtol=1e-4)),
        }
    )


if __name__ == "__main__":
    main()
