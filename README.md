# montlok.cpp

`montlok.cpp` is the x86-64 CPU inference backend for Montlok's RDT4quant
checkpoints. It keeps the trained official Mamba-3 SISO parameter layout,
implements the public recurrent equations in C++17, and compiles the repeated
Transformer/MLA layer with PyTorch Inductor's CPU backend.

The first target is the Tokyo trading host: AMD EPYC 7K62, 32 vCPU, Ubuntu
24.04, Python 3.12 and PyTorch 2.11 CPU. Training remains on CUDA. The runtime
is designed for local Shadow inference and does not contain an exchange client,
account credential, order API, model weight or market dataset.

## Measured Tokyo performance

One inference consumes a `480 x 57` crypto feature window and produces ordered
q10/q50/q90 forecasts for 15, 60 and 240 minutes at recurrent depth 4.

| Runtime | P50 | P95 |
|---|---:|---:|
| Eager PyTorch CPU reference | ~157.0 ms | ~161.1 ms |
| `montlok.cpp` Mamba recurrence | ~132.7 ms | ~143.8 ms |
| C++ Mamba + cached compiled Transformer | **110.8 ms** | **118.2 ms** |

The persistent Transformer graph cache takes about 4.15 seconds to load and
warm in a fresh process. The initial release-time graph build takes about 26
seconds. The C++ recurrence agreed with the PyTorch reference to a maximum
absolute difference of `2.38e-7` in the deterministic kernel test.

## Numerical boundary

The CPU FP32 reference is weight-compatible but not bit-identical to the
training-time BF16 Triton kernel. On 2,000 chronological saved official-CUDA
predictions, the 240-minute 24-bp action agreement was 91.65% (167 flips), the
median absolute forecast difference was 2.52 bp and P95 was 27.73 bp. Treat the
CPU backend as its own reviewed runtime version and run strategy evaluation on
its outputs before activation.

The matching seed-7 diagnostic at 12 bp per side produced -19.90% for the saved
CUDA predictions and -19.87% for CPU. Both mappings are rejected; these files
prove runtime feasibility and measure backend drift, not Alpha.

## Layout

```text
rdt4quant_cpu/
  montlok.cpp                  C++17 recurrent kernel
  montlok_loader.py            prebuilt/JIT extension loader
  mamba3_cpu.py                official-weight CPU reference layer
  model_cpu.py                 RDT4quant multi-domain CPU model
  build_montlok.py             no-Ninja CppExtension build
  benchmark_cpu.py             full-model latency and output check
  profile_cpu.py               Mamba/Transformer attribution
  compare_saved_predictions.py chronological CUDA/CPU comparison
  make_parity_fixture.py       CUDA teacher fixture generator
  test_montlok_cpp.py          kernel parity test
rdt4quant/
  config.json
  native/Model/                minimal Apache-2.0 RDT runtime
evidence/
  benchmark_cached_start.json
  benchmark_optimized_corrected.json
  comparison.json
```

## Build on the target host

Use the same Python and PyTorch ABI as production:

```bash
cd rdt4quant_cpu
python build_montlok.py build_ext --inplace
MONTLOK_CPP=1 python test_montlok_cpp.py
```

Run full-model measurement with an administrator-reviewed checkpoint and
fixture kept outside Git:

```bash
MONTLOK_CPP=1 \
TORCHINDUCTOR_CACHE_DIR=/var/cache/montlok/rdt4quant-inductor \
python benchmark_cpu.py \
  --checkpoint /secure/model.pt \
  --base-config ../rdt4quant/config.json \
  --fixture /secure/fixture.npz \
  --data-manifest /secure/data-manifest.json \
  --threads 32 \
  --iterations 100 \
  --compile-transformer
```

Production release must pin hashes for the checkpoint, source tree, compiled
extension, feature/scaler manifest and Inductor cache. Start with Shadow mode
and order creation disabled.
