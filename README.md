# montlok.cpp

High-performance CPU inference backend for RDT4quant Mamba-3 SISO models.

`montlok.cpp` loads RDT4quant checkpoints unchanged and replaces the
latency-critical parts of the PyTorch graph with fused C++17 kernels. Every
kernel accumulates in double precision and rounds to `float32` once, so the
results are at least as accurate as the pure PyTorch implementation.

## Features

- **Fused Mamba-3 recurrence** — BCNorm, biases, `dt`/`A`/trapezoid
  coefficients, rotary embedding and the selective-scan recurrence in a single
  operator between `in_proj` and `out_proj`.
- **Fused Manifold-Constrained Hyper-Connections (mHC)** — stream RMS
  statistics, projections, Sinkhorn normalisation and stream mixing in two
  operators.
- **Fused RMSNorm / GroupedRMSNorm** and **MLA rotary embedding + key
  assembly**.
- **Drop-in installation** — CPU subclasses keep the upstream parameters and
  state-dict keys; checkpoints load with `strict=True`.
- **Portable SIMD** — GCC/Clang vector extensions target AVX2 + FMA on x86-64
  and NEON on arm64. No AVX-512 requirement.
- **Torch-free kernel headers** with a standalone self-test.
- **Per-component fallback** to the PyTorch reference through environment
  variables for A/B validation.

## Requirements

| Component | Version |
|---|---|
| Python | ≥ 3.12 |
| PyTorch | 2.11 (CPU) |
| NumPy | ≥ 2.2, < 3 |
| Compiler | GCC ≥ 11 or Clang ≥ 15 with C++17 |
| CPU | x86-64 with AVX2 + FMA, or arm64 |

## Installation

```bash
pip install -r requirements.txt
cd rdt4quant_cpu
python build_montlok.py build_ext --inplace     # builds montlok_cpp_v2*.so
```

Verify the build:

```bash
python -c "import montlok_cpp_v2; print(montlok_cpp_v2.build_info())"
# {'vector_extensions': True, 'avx2': True, 'fma': True, ...}
python test_montlok_cpp.py
```

If the prebuilt module is absent, `montlok_loader.py` compiles the extension
just-in-time into the PyTorch extension cache under a content-hashed module
name. The first build takes one to two minutes.

To build for a different x86-64 microarchitecture than the build machine, set
`MONTLOK_MARCH` (for example `MONTLOK_MARCH=znver2`).

## Usage

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

Enable the C++ backend with `MONTLOK_CPP=1` in the environment before the
model is constructed. Set `torch.set_num_threads(n)` to control parallelism;
the kernels run on PyTorch's intra-op thread pool.

### Command-line tools

| Script | Purpose |
|---|---|
| `benchmark_cpu.py` | End-to-end latency and output check against a fixture |
| `profile_cpu.py` | Per-layer latency attribution |
| `bench_montlok_kernel.py` | Micro- and full-model benchmarks with random weights (no checkpoint required) |
| `compare_saved_predictions.py` | Chronological comparison with saved CUDA predictions |
| `make_parity_fixture.py` | Generates a CUDA reference fixture |
| `test_montlok_cpp.py` | Parity tests for every fused operator and the whole model |

Example:

```bash
MONTLOK_CPP=1 OMP_PROC_BIND=close OMP_PLACES=cores \
python benchmark_cpu.py \
  --checkpoint model.pt \
  --base-config ../rdt4quant/config.json \
  --fixture fixture.npz \
  --data-manifest data-manifest.json \
  --threads 16 --iterations 100
```

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `MONTLOK_CPP` | unset | `1` enables the C++ backend; unset runs the PyTorch reference |
| `MONTLOK_CPP_FUSED` | `1` | `0` runs the C++ recurrence with PyTorch pre-processing |
| `MONTLOK_CPP_MHC` | `1` | `0` runs the hyper-connections in PyTorch |
| `MONTLOK_CPP_LAYERS` | `1` | `0` runs RMSNorm and MLA rotary in PyTorch |
| `MONTLOK_SERIAL_WORK` | `100000` | Estimated flop count below which an operator runs single-threaded |
| `MONTLOK_MARCH` | `native` | `-march` value for the extension build |
| `MONTLOK_VERBOSE_BUILD` | unset | `1` prints compiler output during JIT builds |

## Performance

Apple M4 Max, 12 threads, PyTorch 2.11, random weights, model geometry
(`d_model` 256, `d_inner` 512, `d_state` 64, sequence 480, recurrent depth 4).
Reproduce with `python bench_montlok_kernel.py --threads 12`.

| Component | PyTorch | `MONTLOK_CPP=1` | Speed-up |
|---|---:|---:|---:|
| Mamba-3 layer | 4.7 ms | 0.77 ms | 6.1× |
| mHC hyper-connection | 4.7 ms | 0.20 ms | 23× |
| MLA attention block | 1.3 ms | 0.54 ms | 2.4× |
| RMSNorm `[1, 480, 256]` | 0.19 ms | 0.06 ms | 3.2× |
| **Full forward, depth 4** | **46–58 ms** | **~10 ms** | **~5×** |

Time per forward with the backend enabled: ~1.3 ms per Mamba layer (three
layers) and ~1.4 ms per recurrent Transformer step (four steps). Within the
Transformer step roughly 40% is GEMMs, 20% scaled-dot-product attention and
40% the remaining small PyTorch operators, whose cost is dominated by thread
fork/join rather than arithmetic.

`torch.compile` on the Transformer layer is slower than eager execution with
the fused operators enabled; leave `--compile-transformer` off.

### Tuning

- **Thread count.** Compare the physical-core count with the logical-core
  count (`--threads 16` vs `32` on a 16-core/32-thread part). The kernels are
  FMA-bound, SMT siblings share FMA pipes, and fork/join cost grows with
  thread count.
- **Pinning.** `OMP_PROC_BIND=close OMP_PLACES=cores` keeps each thread's state
  block in its own L1. On multi-socket or multi-NUMA machines add
  `numactl --cpunodebind=0 --membind=0`.
- **Denormals.** `benchmark_cpu.py` and `profile_cpu.py` enable
  `torch.set_flush_denormal(True)`; `--no-flush-denormal` opts out. The
  recurrence kernel sets FTZ/DAZ for its own work items regardless.
- **Serial threshold.** Raise `MONTLOK_SERIAL_WORK` (e.g. `1000000`) if
  profiling shows small operators spending their time in fork/join.

## Numerical accuracy

Each fused operator returns the correctly rounded `float32` value of the
reference expression (all intermediates in `float64`). Maximum absolute
difference against the PyTorch implementation with random weights
(`python test_montlok_cpp.py`):

| Comparison | Max abs difference | Value range |
|---|---:|---:|
| Mamba-3 layer | 1.6e-6 | ±3 |
| mHC output | 1.3e-6 | ±6 |
| RMSNorm, MLA block | < 1e-6 | — |
| Full model hidden state | 1.1e-5 | ±12 |
| Full model quantile output | ~1e-6 | — |

`montlok_selftest.cpp` additionally validates every kernel against a naive
double-precision implementation on arm64 and x86-64 AVX2/FMA. The vectorised
`sin`/`cos` agrees with libm bit-for-bit after rounding to `float32` on
5 × 10⁵ samples.

For reference, the CPU `float32` path and the BF16 Triton kernels used in
training differ by 2.52 bp median / 27.73 bp P95 over 2,000 chronological
predictions (`evidence/comparison.json`, `compare_saved_predictions.py`).

## Architecture

```text
rdt4quant_cpu/
  montlok.cpp                  PyTorch bindings and thread dispatch
  montlok_kernel.h             Mamba-3 recurrence and fused pre-processing (torch-free)
  montlok_mhc.h                mHC coefficients, Sinkhorn and stream mixing (torch-free)
  montlok_layers.h             RMSNorm and MLA rotary/key assembly (torch-free)
  montlok_selftest.cpp         Standalone kernel self-test and micro-benchmark
  montlok_loader.py            Extension loader (prebuilt module or content-hashed JIT)
  build_montlok.py             setuptools build (module montlok_cpp_v2)
  mamba3_cpu.py                Mamba-3 SISO reference layer and backend selection
  mhc_cpu.py                   ManifoldHyperConnectionCPU
  mla_cpu.py                   MLACPU
  rmsnorm_cpu.py               RMSNormCPU, GroupedRMSNormCPU, install_cpu_norms
  model_cpu.py                 RDT4quant multi-domain model with CPU classes installed
  test_montlok_cpp.py          Parity tests
  bench_montlok_kernel.py      Random-weight benchmarks
  benchmark_cpu.py             End-to-end benchmark
  profile_cpu.py               Latency attribution
  compare_saved_predictions.py CUDA/CPU output comparison
  make_parity_fixture.py       CUDA fixture generator
rdt4quant/
  config.json                  Model configuration
  native/Model/                Upstream RDT runtime (Apache-2.0)
evidence/                      Recorded benchmark and comparison reports
```

### Operators

| Operator | Replaces |
|---|---|
| `mamba3_siso_forward` | ~60 PyTorch ops between `in_proj` and `out_proj` |
| `mamba3_siso_recurrent` | The recurrence alone (pre-processing stays in PyTorch) |
| `mhc_prepare`, `mhc_combine` | ~100 PyTorch ops per hyper-connection |
| `rms_norm` | 7 PyTorch ops |
| `mla_rope_qk` | ~15 PyTorch ops |
| `build_info` | Reports the SIMD path of the compiled extension |

### Installation of the CPU classes

`model_cpu.py` binds the CPU subclasses into the upstream module namespace
before the model is constructed (`two_stage.ManifoldHyperConnection`,
`two_stage.MLA`, `mamba_layer.OfficialMamba3`) and re-classes every
`RMSNorm`/`GroupedRMSNorm` instance afterwards. Each subclass delegates to the
upstream forward whenever the fast path does not apply (autograd enabled,
non-`float32` inputs, KV cache, attention masks, morphological RoPE).

### Design notes

- **Recurrence.** The `[d_state, headdim]` state of one `(batch, head)` is
  split into 16-lane headdim blocks, giving `B · H · headdim / 16` independent
  work items (32 at the model shape). Each step costs one multiply and
  three FMAs per 8 state elements in one 256-bit register; two accumulators
  hide FMA latency on the `q · state` reduction. The 4 KiB state block stays
  in L1.
- **Fused pre-processing.** The rotary cumulative sum accumulates in double,
  matching ATen's `cumsum`. `sin`/`cos` use a vectorised double-precision
  polynomial (Cody–Waite reduction, Taylor to `r¹³`/`r¹⁴`).
- **mHC.** The 24 projection weights are widened to `float64` once and cached.
  Rows are shared across blocks of 4 tokens so the dot-product loop is
  FMA-bound rather than load-bound, and Sinkhorn iterates 4 tokens per SIMD
  register in linear space.
- **Thread dispatch.** `parallel_items()` dispatches through
  `at::TensorIterator::for_each`, which executes inside `libtorch_cpu` on
  PyTorch's OpenMP pool and honours `torch.set_num_threads`. (The inline
  `at::parallel_for` only parallelises when the extension itself is compiled
  with OpenMP.) Operators below `MONTLOK_SERIAL_WORK` estimated flops run
  inline because a fork/join costs tens of microseconds.

## Testing

```bash
cd rdt4quant_cpu
python test_montlok_cpp.py                                   # operator and model parity

g++ -O3 -std=c++17 -march=native -fno-math-errno -ffp-contract=fast \
    montlok_selftest.cpp -o montlok_selftest && ./montlok_selftest   # torch-free kernels

python bench_montlok_kernel.py --threads 16                  # performance regression
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
