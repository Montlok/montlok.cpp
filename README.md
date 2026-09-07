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
in double precision before rounding to `float32`.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Usage](#usage)
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
- **Portable SIMD** — AVX2 + FMA on x86-64, NEON on arm64.
- **Multi-threaded** on PyTorch's intra-op thread pool, controlled by
  `torch.set_num_threads`.

## Requirements

| Component | Version |
|---|---|
| Python | ≥ 3.12 |
| PyTorch | 2.11 (CPU) |
| NumPy | ≥ 2.2, < 3 |
| Compiler | GCC ≥ 11 or Clang ≥ 15 |
| CPU | x86-64 with AVX2 + FMA, or arm64 |

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
import. Set `MONTLOK_MARCH` (for example `znver2`) to build for a different
x86-64 microarchitecture.

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

- Compare physical-core and logical-core thread counts.
- Pin threads with `OMP_PROC_BIND=close OMP_PLACES=cores`.
- Keep `--compile-transformer` off; eager execution is faster with the fused
  operators.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `MONTLOK_CPP` | unset | `1` enables the C++ backend |
| `MONTLOK_CPP_FUSED` | `1` | `0` runs only the recurrence in C++ |
| `MONTLOK_CPP_MHC` | `1` | `0` runs hyper-connections in PyTorch |
| `MONTLOK_CPP_LAYERS` | `1` | `0` runs RMSNorm and MLA rotary in PyTorch |
| `MONTLOK_SERIAL_WORK` | `100000` | Work threshold below which an operator runs single-threaded |
| `MONTLOK_MARCH` | `native` | `-march` value for the extension build |
| `MONTLOK_VERBOSE_BUILD` | unset | `1` prints compiler output during JIT builds |

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
