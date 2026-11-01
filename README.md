<div align="center">

# montlok.cpp

**CPU inference backend for RDT4quant Mamba-3 models**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c.svg)](https://pytorch.org)

</div>

`montlok.cpp` runs RDT4quant checkpoints on CPUs with fused C++17 kernels for
the Mamba-3 recurrence, Manifold-Constrained Hyper-Connections, RMSNorm and
MLA rotary embeddings. Checkpoints load unchanged, and every kernel accumulates
in double precision before rounding to `float32`. One build runs on any x86-64
or arm64 CPU: the kernels are compiled once per instruction set and the best
variant the machine supports is selected at run time.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Supported CPUs](#supported-cpus)
- [Configuration](#configuration)
- [License](#license)

## Features

- **Fused Mamba-3 SISO recurrence** — pre-processing, rotary embedding and the
  selective-scan recurrence in one operator.
- **Fused mHC hyper-connections** — RMS statistics, projections, Sinkhorn
  normalisation and stream mixing in two operators.
- **Fused RMSNorm and MLA rotary/key assembly.**
- **Drop-in** — CPU classes keep upstream parameters and state-dict keys;
  checkpoints load with `strict=True`.
- **Double-precision accumulation** — each operator returns the correctly
  rounded `float32` result.
- **One build for every CPU** — baseline (SSE2), AVX2 + FMA and AVX-512
  variants of every kernel on x86-64, NEON on arm64; `cpuid` picks the variant
  at run time, so a prebuilt module or wheel runs on any machine of its
  architecture. Variants with the same FMA availability return identical bits.
- **Multi-threaded** on PyTorch's intra-op thread pool, controlled by
  `torch.set_num_threads`. Work is handed out dynamically, so mixed
  performance/efficiency cores (Apple M-series, Intel Core 12th generation and
  later) and descheduled threads do not stall an operator; on macOS the
  extension forks through the OpenMP runtime of the PyTorch wheel.

## Requirements

| Component | Version |
|---|---|
| Python | ≥ 3.12 |
| PyTorch | 2.11 (CPU) |
| NumPy | ≥ 2.2, < 3 |
| Compiler | GCC ≥ 11 or Clang ≥ 15 (Apple clang included) |
| CPU | any x86-64 (SSE2 and up) or arm64; AVX2/FMA and AVX-512 are used when present |

## Getting Started

### Installation

```bash
git clone https://github.com/Montlok/montlok.cpp.git
cd montlok.cpp
pip install -r requirements.txt
cd rdt4quant_cpu && python build_montlok.py build_ext --inplace
```

```bash
python -c "import montlok_cpp_v2; print(montlok_cpp_v2.build_info())"
```

Without the prebuilt module the extension is compiled just-in-time on first
import. `build_info()` reports the variant in use (`isa`), every variant in the
build (`isa_variants`), the CPU features found and whether the thread pool is
reached through OpenMP or `TensorIterator` (`parallel_dispatch`). No `-march`
flag is used by default; `MONTLOK_MARCH=native` (or `znver2`, `x86-64-v3`, ...)
builds for one machine only.

The torch-free kernel self-test checks every compiled variant against a
double-precision reference and compares the variants bit for bit:

```bash
cd rdt4quant_cpu && ./run_selftest.sh
```

### Quickstart

```python
import json, os, sys
import torch

os.environ["MONTLOK_CPP"] = "1"
sys.path.insert(0, "rdt4quant_cpu")
from model_cpu import MultiAssetRDTCPU

config = json.load(open("rdt4quant/config.json"))
model = MultiAssetRDTCPU(config, asset_count=31).eval()
model.load_state_dict(torch.load("model.pt", map_location="cpu")["model"], strict=True)

x = torch.randn(1, 480, 57)                        # [batch, sequence, features]
with torch.inference_mode():
    quantiles, hidden_rms = model(x, depth=4)      # quantiles: [batch, horizons, 3]
```

## Usage

### Command-line tools

| Script | Purpose |
|---|---|
| `benchmark_cpu.py` | End-to-end latency with a checkpoint and fixture |
| `profile_cpu.py` | Per-layer latency attribution |
| `bench_montlok_kernel.py` | Component and full-model benchmarks with random weights |
| `compare_saved_predictions.py` | Chronological comparison with saved CUDA predictions |
| `make_parity_fixture.py` | CUDA reference fixture generator |

```bash
MONTLOK_CPP=1 OMP_PROC_BIND=close OMP_PLACES=cores \
python benchmark_cpu.py \
  --checkpoint model.pt \
  --base-config ../rdt4quant/config.json \
  --fixture fixture.npz \
  --threads 16 --iterations 100
```

### Performance tips

- Compare physical-core and logical-core thread counts. The kernels are
  FMA-bound, so SMT siblings rarely help; on hybrid CPUs try the
  performance-core count and the full count (on an M4 Max, 12 threads beat 16).
- Pin threads with `OMP_PROC_BIND=close OMP_PLACES=cores`.
- Keep `--compile-transformer` off; eager execution is faster with the fused
  operators.
- `bench_montlok_kernel.py --isa baseline` (or `avx2`, `avx512`) times another
  kernel variant for an A/B comparison on the same machine.

### Measured on an Apple M4 Max

Random weights, 12 threads, medians over 30 iterations of
`bench_montlok_kernel.py`, same machine, same torch 2.11 wheel. "Before" is
the previous `-march=native` build; "after" is the portable multi-variant build
of this version (`neon` variant, OpenMP dispatch).

| Operator | Before | After |
|---|---:|---:|
| Mamba-3 recurrence kernel | 0.224 ms | 0.199 ms |
| Fused Mamba-3 op (pre-processing + recurrence) | 0.439 ms | 0.406 ms |
| Mamba-3 layer (with both GEMMs) | 0.692 ms | 0.654 ms |
| mHC hyper-connection | 0.166 ms | 0.128 ms |
| RMSNorm [1, 480, 256] | 0.052 ms | 0.037 ms |
| MLA attention block | 0.406 ms | 0.408 ms |
| Full model, native stage-1/stage-2 tail, depth 4 | 6.67 ms | 6.44 ms |

The arithmetic of every kernel is unchanged; the differences come from the
dispatch. With `MONTLOK_OPENMP=0` (TensorIterator dispatch, dynamic chunking
only) the same build measures 0.152 ms for the hyper-connection and 0.051 ms
for the RMSNorm, so the OpenMP path accounts for most of the gain on the small
operators. x86-64 numbers depend on the variant the CPU gets (`avx512` halves
the recurrence instruction count against `avx2`) and were not measured for
this table.

## Supported CPUs

| CPU | Kernel variant | Notes |
|---|---|---|
| x86-64 with AVX-512 F/VL/BW/DQ (Zen 4/5, Intel Ice Lake, Sapphire Rapids, ...) | `avx512` | one 16-lane zmm register per recurrence block |
| x86-64 with AVX2 + FMA (Haswell, Zen 1-3, Intel N100 and later, ...) | `avx2` | the kernels of the earlier `-march=native` builds |
| any other x86-64 (older Core 2 / Atom / Pentium, early Zhaoxin, VMs without AVX) | `baseline` | SSE2, no fused multiply-add |
| arm64 (Apple M-series, Snapdragon X, Graviton, Raspberry Pi 4/5, Phytium, Kunpeng) | `neon` | |
| other architectures with GCC/Clang (LoongArch, RISC-V, ppc64le) | `baseline` | generic vector extensions |

The variant is chosen once per process from `cpuid` (including the XCR0 state
the operating system enables); `MONTLOK_ISA` overrides it. GEMMs stay in
PyTorch (oneDNN on x86-64 Linux, Accelerate on macOS, oneDNN + Arm Compute
Library on aarch64 Linux), which dispatch on the CPU themselves. MSVC builds
fall back to scalar code for every kernel.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `MONTLOK_CPP` | unset | `1` enables the C++ backend |
| `MONTLOK_CPP_FUSED` | `1` | `0` runs only the recurrence in C++ |
| `MONTLOK_CPP_MHC` | `1` | `0` runs hyper-connections in PyTorch |
| `MONTLOK_CPP_LAYERS` | `1` | `0` runs RMSNorm and MLA rotary in PyTorch |
| `MONTLOK_ISA` | unset | Kernel variant to use (`baseline`, `avx2`, `avx512`, `neon`); `force:<name>` skips the CPU check |
| `MONTLOK_SERIAL_WORK` | `150000` | Work threshold below which an operator runs single-threaded |
| `MONTLOK_MARCH` | unset | Adds `-march=<value>` to the build; the result runs only on CPUs with those features |
| `MONTLOK_OPENMP` | `1` | `0` builds without OpenMP and dispatches through `TensorIterator` |
| `MONTLOK_VERBOSE_BUILD` | unset | `1` prints compiler output during JIT builds |

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
