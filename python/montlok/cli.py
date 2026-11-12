"""The `montlok` command.

    montlok info MODEL                  what a model file holds
    montlok convert SOURCE -o MODEL     scikit-learn, LightGBM or XGBoost model -> .montlok
    montlok predict MODEL INPUT         predictions for a CSV/NPY file
    montlok bench MODEL                 in-process latency of one model
    montlok micro EVENTS                microstructure features for an event file
    montlok serve MODEL                 JSON lines on stdin -> predictions on stdout
    montlok env                         versions, CPU and kernel variants

`python -m montlok` runs the same command.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import montlok
from montlok import export

# ---------------------------------------------------------------------------
# Tables (CSV with an optional header row, .npy, .npz)
# ---------------------------------------------------------------------------


def read_table(path: str | Path) -> tuple[np.ndarray | dict[str, np.ndarray], list[str] | None]:
    """A .npy array, the arrays of a .npz, or a CSV and its header (None when the
    first row is numeric)."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path), None
    if path.suffix == ".npz":
        with np.load(path) as data:
            return {name: data[name] for name in data.files}, None
    with path.open() as stream:
        first = stream.readline()
    header = [token.strip() for token in first.split(",")]
    try:
        [float(token) for token in header]
        header = None
    except ValueError:
        pass
    table = np.loadtxt(path, delimiter=",", ndmin=2, skiprows=1 if header else 0)
    return table, header


def write_table(values: np.ndarray, names: Sequence[str], path: str | None) -> None:
    if path and path.endswith(".npy"):
        np.save(path, values)
        return
    stream = open(path, "w", newline="") if path and path != "-" else sys.stdout
    try:
        np.savetxt(stream, np.atleast_2d(values), delimiter=",", header=",".join(names), comments="", fmt="%.17g")
    finally:
        if stream is not sys.stdout:
            stream.close()


def model_inputs(model: montlok.Model, table: np.ndarray, header: list[str] | None) -> np.ndarray:
    """Rows of `table` as the model's input. A header selects the model's named
    features in order; a sequence model reads every complete window of rows."""
    names = model.spec["inputs"].get("names")
    if header and names:
        missing = [name for name in names if name not in header]
        if missing:
            raise SystemExit(f"input lacks feature columns: {', '.join(missing[:5])}")
        table = table[:, [header.index(name) for name in names]]
    if model.sequence and table.ndim == 2:
        if len(table) < model.sequence:
            raise SystemExit(f"input has {len(table)} rows; a window takes {model.sequence}")
        windows = np.lib.stride_tricks.sliding_window_view(table, (model.sequence, table.shape[1]))
        return np.ascontiguousarray(windows[:, 0])
    return table


def output_names(model: montlok.Model) -> list[str]:
    return ["y"] if model.outputs == 1 else [f"y{k}" for k in range(model.outputs)]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _size(path: Path) -> str:
    size = float(path.stat().st_size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return ""


def cmd_info(args: argparse.Namespace) -> int:
    model = montlok.load(args.model)
    spec = model.spec
    if args.json:
        print(json.dumps(spec, indent=2))
        return 0
    _, tensors = export.read(args.model)
    config = spec.get("config", {})
    rows = [("model", f"{args.model} ({_size(Path(args.model))})"), ("family", model.family),
            ("inputs", f"{model.features} features" + (f" x {model.sequence} steps" if model.sequence else "")),
            ("outputs", str(model.outputs))]
    pre = spec.get("preprocess")
    if pre:
        parts = (["standardize"] if pre.get("mean") else []) + ([f"clip {pre['clip']}"] if pre.get("clip") else [])
        rows.append(("preprocess", ", ".join(parts)))
    if model.family == "sequential":
        layers = []
        for layer in config.get("layers", []):
            kind = layer["type"]
            if kind in ("gru", "lstm"):
                kind += f"({layer['hidden']})"
            elif kind == "linear":
                kind += f"({tensors[layer['weight']].shape[0]})"
            layers.append(kind)
        rows += [("precision", config.get("precision", "f64")), ("layers", " -> ".join(layers))]
    elif model.family == "tree_ensemble":
        prefix = config.get("tensors", "trees")
        rows += [("trees", f"{len(tensors[prefix + '.root'])} ({len(tensors[prefix + '.feature'])} nodes)"),
                 ("splits", f"{'<=' if config.get('decision', 'le') == 'le' else '<'} on "
                            f"{'float32' if config.get('feature_cast') == 'f32' else 'float64'} features"),
                 ("link", f"{config.get('link', 'identity')}, {config.get('aggregate', 'sum')} of trees")]
    meta = spec.get("metadata", {})
    if meta:
        rows.append(("exported", ", ".join(f"{k}={v}" for k, v in meta.items() if isinstance(v, (str, int, float)))))
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value}")
    return 0


def load_source(path: Path) -> tuple[Any, str]:
    """The trained model in `path` and its kind: sklearn, lightgbm or xgboost."""
    suffix = path.suffix.lower()
    if suffix in (".joblib", ".pkl", ".pickle"):
        try:
            import joblib

            obj = joblib.load(path)
        except ImportError:
            import pickle

            with path.open("rb") as stream:
                obj = pickle.load(stream)
        module = type(obj).__module__.split(".")[0]
        return obj, module if module in ("lightgbm", "xgboost") else "sklearn"
    if suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "tree_info" in data:
            return data, "lightgbm"
        if isinstance(data, dict) and "learner" in data:
            return _xgboost_booster(path), "xgboost"
        raise SystemExit(f"{path}: expected a LightGBM dump_model() or an XGBoost save_model() JSON file")
    if suffix in (".ubj", ".bin", ".model"):
        return _xgboost_booster(path), "xgboost"
    if suffix == ".txt":
        try:
            import lightgbm
        except ImportError:
            raise SystemExit("LightGBM text models load with the lightgbm package; "
                             "booster.dump_model() saved as JSON converts with numpy alone") from None
        return lightgbm.Booster(model_file=str(path)), "lightgbm"
    raise SystemExit(f"{path}: use .joblib/.pkl (scikit-learn and its LightGBM/XGBoost wrappers), "
                     ".txt or dump_model() .json (LightGBM), .json/.ubj (XGBoost)")


def _xgboost_booster(path: Path) -> Any:
    try:
        import xgboost
    except ImportError:
        raise SystemExit("XGBoost model files load with the xgboost package") from None
    return xgboost.Booster(model_file=str(path))


def _reference(source: Any, x: np.ndarray, outputs: int) -> np.ndarray:
    if hasattr(source, "predict_proba"):
        p = np.asarray(source.predict_proba(x), dtype=np.float64)
        return p[:, 1:] if p.shape[1] == 2 and outputs == 1 else p
    if type(source).__module__.startswith("xgboost") and not hasattr(source, "fit"):
        import xgboost

        return np.asarray(source.predict(xgboost.DMatrix(x)), dtype=np.float64).reshape(len(x), -1)
    return np.asarray(source.predict(x), dtype=np.float64).reshape(len(x), -1)


def cmd_convert(args: argparse.Namespace) -> int:
    source, kind = load_source(Path(args.source))
    names = args.feature_names.split(",") if args.feature_names else None
    converter = {"sklearn": export.from_sklearn, "lightgbm": export.from_lightgbm, "xgboost": export.from_xgboost}[kind]
    exported = converter(source, feature_names=names)
    exported.save(args.output)
    model = montlok.load(args.output)
    print(f"{args.output}: {model.family}, {model.features} features -> {model.outputs} output(s), from {kind}")
    if args.check:
        if isinstance(source, dict):
            raise SystemExit("--check needs the model object; a dump file holds no predict()")
        table, header = read_table(args.check)
        x = model_inputs(model, table, header)
        got = model.predict(x).reshape(len(x), -1)
        want = _reference(source, x, model.outputs)
        difference = float(np.max(np.abs(got - want))) if len(x) else 0.0
        scale = max(1.0, float(np.max(np.abs(want)))) if len(x) else 1.0
        print(f"check: {len(x)} rows, max |difference| {difference:.3g} (scale {scale:.3g})")
        return 0 if difference <= args.tolerance * scale else 1
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    montlok.set_threads(args.threads)
    model = montlok.load(args.model)
    table, header = read_table(args.input)
    if isinstance(table, dict):
        table = table["x"] if "x" in table else next(iter(table.values()))
    y = model.predict(model_inputs(model, table, header))
    write_table(y.reshape(len(y), -1) if y.ndim > 1 else y.reshape(1, -1), output_names(model), args.output)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    montlok.set_threads(args.threads)
    montlok.set_latency_threads(args.latency_threads)
    try:
        model = montlok.load(args.model)
        width = args.batch * (model.features if args.stream else model.features * max(model.sequence, 1))
        rows = np.random.default_rng(7).standard_normal((args.rows, width))
        samples, mean = montlok._core.bench(model, rows, args.batch, args.iterations, args.stream)
    finally:
        montlok.set_latency_threads(0)
    p50, p90, p99, p999 = np.percentile(samples, [50, 90, 99, 99.9])
    call = "stream step" if args.stream else "predict"
    print(f"{args.model}: {call}, batch {args.batch}, {args.threads} thread(s), "
          f"{args.latency_threads} latency thread(s), {args.iterations} calls")
    print(f"p50 {_ns(p50)}  p90 {_ns(p90)}  p99 {_ns(p99)}  p99.9 {_ns(p999)}  mean {_ns(mean)}")
    print(f"{1e9 / mean:,.0f} calls/s, {args.batch * 1e9 / mean:,.0f} rows/s")
    return 0


def _ns(value: float) -> str:
    return f"{value:.0f} ns" if value < 1000 else f"{value / 1000:.2f} µs" if value < 1e6 else f"{value / 1e6:.2f} ms"


def cmd_micro(args: argparse.Namespace) -> int:
    from montlok import micro

    table, header = read_table(args.events)
    columns = ("receive_ns", "kind", "side", "a", "b", "c", "d")
    if isinstance(table, dict):
        data = table
    elif header:
        data = {name: table[:, header.index(name)] for name in columns if name in header}
    else:
        data = {name: table[:, i] for i, name in enumerate(columns[: table.shape[1]])}
    if "receive_ns" not in data or "kind" not in data:
        raise SystemExit("events need receive_ns and kind columns (then side, a, b, c, d)")
    n = len(data["receive_ns"])
    get = lambda name: np.asarray(data.get(name, np.zeros(n)))  # noqa: E731
    engine = micro.FeatureEngine(args.capacity)
    features, ready = engine.replay(get("receive_ns").astype(np.int64), get("kind").astype(np.int32),
                                    get("side").astype(np.int32), get("a"), get("b"), get("c"), get("d"))
    if args.output and args.output.endswith(".npz"):
        np.savez(args.output, features=features, ready=ready)
    else:
        write_table(np.column_stack([ready, features]), ["ready", *micro.FEATURES], args.output)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    montlok.set_threads(args.threads)
    model = montlok.load(args.model)
    stream = model.stream() if args.stream else None
    for line in sys.stdin:
        if not line.strip():
            continue
        request: Any = None
        try:
            request = json.loads(line)
            x = np.asarray(request["x"], dtype=np.float64)
            y = stream.step(x) if stream is not None else model.predict(x)
            reply = {"id": request.get("id"), "y": y.tolist()}
        except Exception as failure:  # one bad line answers with its error
            reply = {"id": request.get("id") if isinstance(request, dict) else None, "error": str(failure)}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()
    return 0


def cmd_env(args: argparse.Namespace) -> int:
    info = montlok.build_info()
    rows = [("montlok", f"{info['version']} (C ABI {info['abi']})"),
            ("python", f"{info['python']} ({info['machine']})"), ("cpu", _cpu()),
            ("families", ", ".join(info["families"])),
            ("threads", f"{info['threads']} for batches, {info['latency_threads']} latency")]
    if info["rdt_extension"]:
        try:
            from montlok import rdt

            kernels = rdt.build_info()
            rows.append(("rdt kernels", f"{kernels['isa']} of {', '.join(kernels['isa_variants'])}, "
                                        f"{kernels['threads']} threads ({kernels['parallel_dispatch']})"))
        except Exception as failure:
            rows.append(("rdt kernels", f"montlok._C present, {failure}"))
    else:
        rows.append(("rdt kernels", "montlok._C absent (pip install --no-build-isolation montlok with PyTorch)"))
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value}")
    return 0


def _cpu() -> str:
    name = platform.processor() or platform.machine()
    try:
        if sys.platform == "darwin":
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  check=True).stdout.strip() or name
        elif Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
    except (OSError, subprocess.SubprocessError):
        pass
    import os

    return f"{name}, {os.cpu_count()} hardware threads"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="montlok", description="CPU runtime for quantitative trading models.")
    root.add_argument("--version", action="version", version=f"montlok {montlok.__version__}")
    commands = root.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = commands.add_parser("info", help="what a model file holds")
    p.add_argument("model")
    p.add_argument("--json", action="store_true", help="print the full spec")
    p.set_defaults(run=cmd_info)

    p = commands.add_parser("convert", help="scikit-learn, LightGBM or XGBoost model -> .montlok")
    p.add_argument("source", help=".joblib/.pkl, LightGBM .txt or dump_model() .json, XGBoost .json/.ubj")
    p.add_argument("-o", "--output", required=True, help="the .montlok file to write")
    p.add_argument("--feature-names", help="comma-separated names stored in the model")
    p.add_argument("--check", metavar="INPUT", help="compare predictions with the source model on this CSV/NPY")
    p.add_argument("--tolerance", type=float, default=1e-6, help="largest difference --check accepts, relative "
                                                                 "to the output scale (default 1e-6)")
    p.set_defaults(run=cmd_convert)

    p = commands.add_parser("predict", help="predictions for a CSV/NPY file")
    p.add_argument("model")
    p.add_argument("input", help="CSV (header optional) or .npy; sequence models read sliding windows of rows")
    p.add_argument("-o", "--output", help="CSV or .npy (default: CSV on stdout)")
    p.add_argument("--threads", type=int, default=0, help="worker threads (default: one per hardware thread)")
    p.set_defaults(run=cmd_predict)

    p = commands.add_parser("bench", help="in-process latency of one model")
    p.add_argument("model")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--iterations", type=int, default=20000)
    p.add_argument("--rows", type=int, default=1024, help="distinct random inputs used in turn")
    p.add_argument("--threads", type=int, default=1, help="worker threads for batches")
    p.add_argument("--latency-threads", type=int, default=0, help="threads that split one row across cores")
    p.add_argument("--stream", action="store_true", help="time stream steps of a recurrent model")
    p.set_defaults(run=cmd_bench)

    p = commands.add_parser("micro", help="microstructure features for an event file")
    p.add_argument("events", help="CSV/.npz with receive_ns, kind, side, a, b, c, d")
    p.add_argument("-o", "--output", help="CSV, .npy or .npz (default: CSV on stdout)")
    p.add_argument("--capacity", type=int, default=65536, help="events kept per rolling window")
    p.set_defaults(run=cmd_micro)

    p = commands.add_parser("serve", help="JSON lines on stdin -> predictions on stdout")
    p.add_argument("model")
    p.add_argument("--stream", action="store_true", help="each line is the next step of one stream")
    p.add_argument("--threads", type=int, default=1)
    p.set_defaults(run=cmd_serve)

    p = commands.add_parser("env", help="versions, CPU and kernel variants")
    p.set_defaults(run=cmd_env)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
