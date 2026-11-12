<div align="center">

# montlok.cpp

**CPU runtime for quantitative trading models**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![C ABI](https://img.shields.io/badge/C%20ABI-v1-555.svg)](include/montlok/montlok.h)

</div>

montlok runs trained quant models on the CPU, inside the process that uses the
prediction. Models from PyTorch, scikit-learn, LightGBM and XGBoost become one
`.montlok` file, executed by a C++17 core that the `montlok` command, Python,
Rust and C++ all reach through the same C ABI.

- [Install](#install)
- [Commands](#commands)
- [Library](#library)
- [Model files](#model-files)
- [Families and exporters](#families-and-exporters)
- [Latency threads](#latency-threads)
- [Plugins](#plugins)
- [Microstructure features](#microstructure-features)
- [RDT4quant kernels](#rdt4quant-kernels)
- [Performance](#performance)
- [Building and testing](#building-and-testing)
- [Configuration](#configuration)

## Install

```bash
pip install "montlok @ git+https://github.com/Montlok/montlok.cpp"
pip install --no-build-isolation "montlok @ git+https://github.com/Montlok/montlok.cpp"   # also montlok.rdt, for the installed PyTorch
```

## Commands

| Command | Does |
|---|---|
| `montlok convert SOURCE -o MODEL` | Converts scikit-learn (`.joblib`, `.pkl`), LightGBM (`.txt`, `dump_model()` `.json`) or XGBoost (`.json`, `.ubj`) to `.montlok`. `--check INPUT` compares predictions with the source model. |
| `montlok info MODEL` | Inputs, outputs, preprocessing, layers or trees. `--json` prints the spec. |
| `montlok predict MODEL INPUT` | Predictions for a CSV or `.npy`. A CSV header is matched to the model's feature names; sequence models read every window of rows. |
| `montlok bench MODEL` | In-process latency (p50 to p99.9) and calls per second. `--batch`, `--threads`, `--latency-threads`, `--stream`. |
| `montlok serve MODEL` | Reads one request per line on stdin, `{"id": 1, "x": [...]}`, and writes `{"id": 1, "y": [...]}`. `--stream` keeps recurrent state from line to line. |
| `montlok micro EVENTS` | The 32 microstructure features after each event of a CSV or `.npz` with `receive_ns, kind, side, a, b, c, d`. |
| `montlok env` | Versions, CPU, thread settings and the RDT kernel variant. |

```bash
montlok convert gbr.joblib -o gbr.montlok --check rows.csv
montlok info gbr.montlok
montlok predict gbr.montlok rows.csv -o predictions.csv
montlok bench gbr.montlok --latency-threads 3
```

`python -m montlok` runs the same command. PyTorch models are exported from
Python ([Library](#library)).

## Library

```python
import montlok
from montlok import export

export.from_torch([net.gru, export.LAST, net.head], features=40, sequence=32,
                  preprocess={"mean": mean, "scale": scale, "clip": (-8, 8)}).save("gru.montlok")
export.from_sklearn(pipeline).save("ridge.montlok")          # a leading StandardScaler becomes preprocessing
export.from_lightgbm(booster).save("lgbm.montlok")

model = montlok.load("gru.montlok")
y = model.predict(window)              # [batch, 32, 40] raw features -> [batch, 1]
model.predict(window, out)             # writes into an existing float64 array
live = model.stream()                  # carries the GRU state from bar to bar
y_now = live.step(features_now)        # [40] -> [1]
```

Rust (`montlok = { git = "https://github.com/Montlok/montlok.cpp" }`):

```rust
let model = montlok::Model::open("gru.montlok")?;
let y = model.predict(&window, 1)?;            // window: &[f32] or &[f64]
let mut live = model.stream(1)?;
let y_now = live.step(&features_now)?;
```

C++ (`add_subdirectory(montlok.cpp)`, `target_link_libraries(app PRIVATE montlok::montlok)`):

```cpp
#include <montlok/montlok.hpp>
montlok::Model model = montlok::Model::open("gru.montlok");
std::vector<double> y = model.predict(window, 1);
```

A loaded model is immutable and safe to call from many threads; a stream
belongs to one thread at a time.

## Model files

A `.montlok` file is a [safetensors](https://github.com/huggingface/safetensors)
file. Weights are named tensors; `__metadata__["montlok"]` holds the spec:

```json
{"schema": "montlok.model/1", "family": "sequential",
 "inputs": {"features": 40, "sequence": 32, "names": ["ret_1", "..."]},
 "outputs": {"size": 1},
 "preprocess": {"mean": "preprocess.mean", "scale": "preprocess.scale", "clip": [-8, 8]},
 "config": {"precision": "f32", "layers": [{"type": "gru", "hidden": 64, "weights": "l0"}, {"type": "last"},
                                            {"type": "layernorm", "weight": "l2.weight", "bias": "l2.bias"},
                                            {"type": "linear", "weight": "l3.weight", "bias": "l3.bias"}]},
 "metadata": {"exporter": "montlok 0.2.0", "source": "torch"}}
```

`sequence` is 0 for tabular models (`[batch, features]` inputs). `preprocess`
applies `clip((x - mean) / scale)` in float64 before the family sees the values.
The files also open with the `safetensors` library, and `montlok.export.read`
returns their spec and NumPy tensors. Loading validates every offset, shape,
dtype and node index before the model is built.

## Families and exporters

| Family | Layers / options | Exporter |
|---|---|---|
| `sequential` | `linear`, `layernorm`, `rmsnorm`, `gru`, `lstm`, `last`, `mean`, `flatten`, `softmax`, `affine`, activations `relu` `leaky_relu` `elu` `gelu` `gelu_tanh` `tanh` `sigmoid` `silu` `softplus` | `from_torch` (module chains), `from_sklearn` (linear, logistic, MLP) |
| `tree_ensemble` | `<=`/`<` splits, per-node missing handling (LightGBM None/Zero/NaN, XGBoost and scikit-learn default directions), float32 feature casting, sum or mean, identity/sigmoid/softmax/exp link | `from_lightgbm`, `from_xgboost`, `from_sklearn` (trees, forests, GradientBoosting, HistGradientBoosting) |

`sequential` has two precisions. `f32` (PyTorch exports) rounds the input and
every operator's output to float32, as the float32 module does, and stores the
weights as float32. `f64` (scikit-learn) keeps every value in double. Dot
products accumulate in double in a fixed order. Recurrent layers follow
PyTorch's gate order and `h' = n + z (h - n)` update; `stream()` after T steps
returns what `predict()` returns for those T steps.

`tree_ensemble` adds each output's leaves in tree order within blocks of 64
trees, starting from the base score, and then adds the block sums in order.

Parity in the test suite: PyTorch modules within 2e-5 of the output scale;
scikit-learn trees, forests, boosting and linear models within 1e-12; LightGBM
and XGBoost against reference implementations of their decision rules,
including NaN, zero and `-0.0` inputs.

## Latency threads

`montlok.set_latency_threads(n)` (C: `montlok_set_latency_threads`, C++ and
Rust: `montlok::set_latency_threads`) starts n threads that compute a single
prediction together with the calling thread: a row of a tree ensemble divides
its blocks of trees among them. Each latency thread spins on its own core until
`set_latency_threads(0)`. On Linux, `MONTLOK_LATENCY_CPUS` places them on given
CPUs, such as cores reserved with `isolcpus`.

| Latency threads | `xgb1000_d8` p50 | `gbr300` p50 |
|---:|---:|---:|
| 0 | 6.2 µs | 0.75 µs |
| 1 | 3.4 µs | 0.54 µs |
| 3 | 2.0 µs | 0.46 µs |
| 7 | 1.5 µs | 0.63 µs |

One row on an Apple M4 Max: `montlok bench MODEL --latency-threads N`.

## Plugins

A model family is a `montlok_family_v1` vtable (`create`, `destroy`,
`predict`, and optionally `stream_*`). It receives preprocessed float64 inputs
and reads its tensors and config through the `montlok_host_v1` table it is
given:

```bash
cc -O2 -shared -fPIC -Iinclude examples/plugin/ema_dot.c -o libema_dot.so
```

```python
montlok.load_plugin("./libema_dot.so")
model = export.make("ema_dot", {}, {"ema.weight": w, "ema.alpha": [0.8], "ema.bias": [0.0]},
                    features=4, sequence=6, outputs=1).load()
```

`montlok_register_family` registers a family from inside the process, for
example one implemented in Rust or C++ by the engine itself. Built-in families
register in `src/core/runtime.cpp`; layers of `sequential` in
`src/core/sequential.cpp`.

## Microstructure features

`montlok.micro.FeatureEngine` keeps fixed-capacity rolling state over
receive-ordered reference, L1 book and trade events and returns the 32
features of schema `xstock-observable-l1-v2-32` (`montlok.micro.FEATURES`).
`replay(...)` feeds whole event columns in C++, as `montlok micro` does; a
rejected event (non-monotonic time, non-finite values, capacity overflow)
faults the engine until `reset()`. The same engine is the `montlok_micro_*`
part of the C ABI and `montlok::micro` in Rust.

## RDT4quant kernels

`montlok.rdt` runs RDT4quant Mamba-3 SISO checkpoints with fused C++17 kernels
for the Mamba-3 recurrence, Manifold-Constrained Hyper-Connections, RMSNorm
and MLA rotary embeddings (`rdt4quant_cpu/`); every kernel accumulates in
double and rounds to float32.

```python
from montlok import rdt
model = rdt.MultiAssetRDTCPU(config, asset_count=31).eval()
model.load_state_dict(torch.load("model.pt", map_location="cpu")["model"], strict=True)
with torch.inference_mode():
    quantiles, hidden_rms = model(torch.randn(1, 480, 57), depth=4)
```

The CPU classes subclass the upstream RDT4quant modules; `rdt.use_reference(path)`
selects which upstream tree they extend (default: the copy under
`python/montlok/rdt/reference`). The kernels are compiled once per instruction
set, and `cpuid` picks the variant at run time (`montlok env` shows it):

| CPU | Kernel variant |
|---|---|
| x86-64 with AVX-512 F/VL/BW/DQ (Zen 4/5, Ice Lake, Sapphire Rapids, ...) | `avx512` |
| x86-64 with AVX2 + FMA (Haswell, Zen 1-3, N100, ...) | `avx2` |
| other x86-64 | `baseline` (SSE2) |
| arm64 (Apple M-series, Graviton, Snapdragon X, ...) | `neon` |

Tools in `tools/rdt/`: `benchmark_cpu.py` (latency with a checkpoint),
`profile_cpu.py` (per layer), `bench_montlok_kernel.py` (per operator, `--isa`
to compare variants), `compare_saved_predictions.py`, `make_parity_fixture.py`.
`rdt4quant_cpu/run_selftest.sh` checks every variant against a double-precision
reference.

## Performance

Apple M4 Max, one thread, one random row per call, in the process
(`python tools/bench_models.py models`, then `montlok bench models/<name>.montlok`):

| Model | p50 | p99 |
|---|---:|---:|
| `gbr300`: GradientBoosting, 300 trees of depth 5 | 0.75 µs | 0.92 µs |
| `xgb1000_d8`: XGBoost regression, 100 features, 1000 trees of depth 8 | 6.2 µs | 9.4 µs |
| `gru`: GRU 40 -> 64, LayerNorm+Linear head, one `stream` step | 1.8 µs | 2.3 µs |
| `gru`, a 32-bar window | 48 µs | 62 µs |

Compared with the training frameworks, from Python (`python tools/bench_core.py`):

| Workload | montlok | Framework |
|---|---:|---:|
| GRU window above | 49 µs | PyTorch 2.11: 198 µs |
| 300-tree GradientBoosting, one row | 1.2 µs | scikit-learn: 92 µs |
| 300-tree GradientBoosting, 100k rows, 8 threads | 10 ms | scikit-learn: 188 ms |

Tree ensembles are laid out at load time as packed 16-byte nodes; one row
walks 16 trees at a time, and a batch walks 64 rows per tree.
`montlok.set_threads(n)` spreads batches over a worker pool (16 GRU windows:
111 µs on 8 threads), and RDT4quant runs kernels compiled for the CPU's
instruction set (full model: 6.4 ms on 12 threads).

## Building and testing

| Target | Command |
|---|---|
| Python package | `pip install --no-build-isolation -e .` |
| C/C++ library | `cmake -S . -B build && cmake --build build && ctest --test-dir build` (also builds `montlok_bench`) |
| Rust crate | `cargo test --manifest-path bindings/rust/Cargo.toml` |

Requirements: Python 3.12+, NumPy, a C++17 compiler (GCC 11+, Clang 15+,
Apple clang), pybind11 for the Python build, and PyTorch 2.11 for
`montlok._C`. Tests: `pytest`, `python tests/rdt/test_rdt_kernels.py`, the
CMake test, `cargo test`; `tests/data/make_fixtures.py` regenerates the
cross-language fixtures.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `MONTLOK_LATENCY_CPUS` | unset | Linux: CPUs for the latency threads, such as `2-5,8` |
| `MONTLOK_CPP` | `1` once `montlok.rdt` is imported | `0` runs RDT4quant on the PyTorch reference |
| `MONTLOK_CPP_FUSED` | `1` | `0` runs only the Mamba-3 recurrence in C++ |
| `MONTLOK_CPP_MHC` | `1` | `0` runs hyper-connections in PyTorch |
| `MONTLOK_CPP_LAYERS` | `1` | `0` runs RMSNorm and MLA rotary in PyTorch |
| `MONTLOK_ISA` | unset | RDT kernel variant (`baseline`, `avx2`, `avx512`, `neon`); `force:<name>` skips the CPU check |
| `MONTLOK_REQUIRE_PREBUILT` | unset | `1` loads the RDT kernels from the installed `montlok._C` only |
| `MONTLOK_RDT_REFERENCE` | bundled | directory holding the upstream `Model` package |
| `MONTLOK_SERIAL_WORK` | `150000` | work threshold below which an RDT operator runs single-threaded |
| `MONTLOK_MARCH` | unset | adds `-march=<value>` to the RDT kernel build |
| `MONTLOK_OPENMP` | `1` | `0` builds the RDT kernels with ATen's own thread dispatch |
| `MONTLOK_BUILD_TORCH` | auto | `0` skips `montlok._C` in `pip install`, `1` requires it |

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
