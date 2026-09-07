#!/usr/bin/env python3
"""Parity tests for the montlok C++ CPU backend against the pure PyTorch code.

Every fused op is compared with the upstream / reference implementation it
replaces, on the model geometry and on odd shapes that exercise the
scalar and partial-block code paths:

* Mamba-3 SISO layer: reference vs ``MONTLOK_CPP_FUSED=0`` (recurrence only)
  vs fused (recurrence + in_proj post-processing)
* ManifoldHyperConnectionCPU vs Model.layers.mhc.ManifoldHyperConnection
* fused RMSNorm / GroupedRMSNorm / LayerNorm / SwiGLU vs their PyTorch modules
* MLACPU vs Model.layers.mla.MLA
* fused mHC chaining, final-position attention and the whole MultiAssetRDTCPU
  model (random weights) with and without the backend

The script toggles the MONTLOK_* environment variables itself; it builds the
extension on first use (see montlok_loader.py). Exits non-zero on failure.

    python test_montlok_cpp.py
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / "rdt4quant" / "native"
if str(NATIVE) not in sys.path:
    sys.path.insert(0, str(NATIVE))

from Model.config import RDTConfig  # noqa: E402
from Model.layers.mhc import ManifoldHyperConnection  # noqa: E402
from Model.layers.mla import MLA  # noqa: E402
from Model.layers.rmsnorm import GroupedRMSNorm, RMSNorm  # noqa: E402
from Model.layers.swiglu import SwiGLU  # noqa: E402

from mamba3_cpu import Mamba3CPUReference  # noqa: E402
from mhc_cpu import ManifoldHyperConnectionCPU  # noqa: E402
from mla_cpu import MLACPU  # noqa: E402
from layers_cpu import install_cpu_layers  # noqa: E402

FAILURES: list[str] = []
BACKEND_VARS = (
    "MONTLOK_CPP",
    "MONTLOK_CPP_FUSED",
    "MONTLOK_CPP_MHC",
    "MONTLOK_CPP_LAYERS",
    "MONTLOK_CPP_TAIL",
)


@contextmanager
def backend(**flags: str | None):
    """Temporarily set the MONTLOK_* variables (None removes one)."""
    saved = {name: os.environ.get(name) for name in BACKEND_VARS}
    for name in BACKEND_VARS:
        os.environ.pop(name, None)
    for name, value in flags.items():
        if value is not None:
            os.environ[name] = value
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def report(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float) -> None:
    diff = (actual.double() - expected.double()).abs()
    max_diff = float(diff.max()) if diff.numel() else 0.0
    scale = float(expected.abs().max()) if expected.numel() else 0.0
    ok = torch.isfinite(actual).all().item() and actual.shape == expected.shape and max_diff <= atol
    status = "ok" if ok else "FAIL"
    print(f"{name:<58s} max|diff| {max_diff:.2e} (absmax {scale:7.3f}, tol {atol:.0e}) {status}")
    if not ok:
        FAILURES.append(name)


def test_mamba_layer() -> None:
    # (d_model, d_state, expand, headdim, ngroups, batch, length, note)
    cases = [
        (64, 16, 2, 16, 1, 2, 48, "16-lane blocks"),
        (48, 14, 2, 8, 2, 2, 37, "8-lane blocks, 2 B/C groups"),
        (36, 6, 2, 12, 1, 1, 20, "scalar path (headdim 12)"),
        (256, 64, 2, 64, 1, 1, 480, "model geometry"),
    ]
    for d_model, d_state, expand, headdim, ngroups, batch, length, note in cases:
        torch.manual_seed(7)
        model = Mamba3CPUReference(d_model, d_state=d_state, expand=expand, headdim=headdim, ngroups=ngroups).eval()
        with torch.no_grad():
            # Move every parameter away from its init so all code paths carry signal.
            model.B_bias.add_(torch.randn_like(model.B_bias) * 0.3)
            model.C_bias.add_(torch.randn_like(model.C_bias) * 0.3)
            model.B_norm.weight.mul_(torch.rand_like(model.B_norm.weight) + 0.5)
            model.C_norm.weight.mul_(torch.rand_like(model.C_norm.weight) + 0.5)
            model.D.add_(torch.randn_like(model.D) * 0.5)
            model.dt_bias.add_(torch.randn_like(model.dt_bias) * 0.5)
        values = torch.randn(batch, length, d_model)
        name = f"mamba d{d_model} N{d_state} hd{headdim} G{ngroups} B{batch} L{length} ({note})"
        with torch.inference_mode():
            with backend():
                reference = model(values)
            with backend(MONTLOK_CPP="1", MONTLOK_CPP_FUSED="0"):
                recurrent = model(values)
            with backend(MONTLOK_CPP="1"):
                fused = model(values)
        # fp32 recurrence over `length` steps against the fp32 quadratic reference.
        tol = 2e-5 * (1.0 + float(reference.abs().max()))
        report(name + " recurrent", recurrent, reference, tol)
        report(name + " fused", fused, reference, tol)


def test_mhc() -> None:
    cases = [
        (4, 256, 1, 480, 20),
        (4, 256, 2, 37, 20),
        (3, 100, 1, 50, 5),
        (5, 33, 2, 17, 0),
        (1, 16, 1, 9, 3),
        (9, 20, 1, 6, 2),
    ]
    for n, d, batch, length, iters in cases:
        torch.manual_seed(11)
        hc = ManifoldHyperConnectionCPU(d, n_streams=n, sinkhorn_iters=iters).eval()
        with torch.no_grad():
            for alpha in (hc.pre_alpha, hc.post_alpha, hc.res_alpha):
                alpha.fill_(0.7)
            hc.res_bias.add_(torch.randn(n, n))
            hc.pre_bias.add_(torch.randn(n) * 0.5)
            hc.post_bias.add_(torch.randn(n) * 0.5)
            hc.dyn_norm.weight.mul_(torch.rand(d) + 0.5)
            for lin in (hc.pre_proj, hc.post_proj, hc.res_proj):
                lin.weight.mul_(4.0)
        streams = torch.randn(batch, length, n, d) * 1.5
        mix = torch.randn(d, d) / d**0.5

        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.tanh(x @ mix)

        with torch.inference_mode():
            expected = ManifoldHyperConnection.forward(hc, streams, fn)
            with backend(MONTLOK_CPP="1"):
                actual = hc(streams, fn)
            with backend(MONTLOK_CPP="1", MONTLOK_CPP_MHC="0"):
                disabled = hc(streams, fn)
        report(f"mhc n{n} d{d} B{batch} L{length} iters{iters}", actual, expected, 1e-5)
        report(
            f"mhc n{n} d{d} B{batch} L{length} MONTLOK_CPP_MHC=0 passthrough",
            disabled,
            expected,
            0.0,
        )
        n_last = min(5, length)
        with torch.inference_mode():
            with backend(MONTLOK_CPP="1"):
                tail = hc.forward_tail(streams, lambda agg: fn(agg)[:, -n_last:], n_last)
            with backend(MONTLOK_CPP="1", MONTLOK_CPP_MHC="0"):
                tail_disabled = hc.forward_tail(streams, lambda agg: fn(agg)[:, -n_last:], n_last)
        report(
            f"mhc n{n} d{d} B{batch} L{length} tail{n_last}",
            tail,
            expected[:, -n_last:],
            1e-5,
        )
        report(
            f"mhc n{n} d{d} B{batch} L{length} tail{n_last} fallback",
            tail_disabled,
            expected[:, -n_last:],
            1e-5,
        )


def test_mhc_chain() -> None:
    """Exercise broadcast prepare, fused write/read and fused final collapse."""
    torch.manual_seed(12)
    n, d, batch, length = 4, 64, 2, 19
    first = ManifoldHyperConnectionCPU(d, n_streams=n, sinkhorn_iters=7).eval()
    second = ManifoldHyperConnectionCPU(d, n_streams=n, sinkhorn_iters=7).eval()
    for hc in (first, second):
        with torch.no_grad():
            for alpha in (hc.pre_alpha, hc.post_alpha, hc.res_alpha):
                alpha.fill_(0.6)
            hc.res_bias.add_(torch.randn_like(hc.res_bias))
            for linear in (hc.pre_proj, hc.post_proj, hc.res_proj):
                linear.weight.mul_(3.0)

    backbone = torch.randn(batch, length, d)
    expanded = backbone.unsqueeze(-2).expand(-1, -1, n, -1).contiguous()
    mix1 = torch.randn(d, d) / d**0.5
    mix2 = torch.randn(d, d) / d**0.5
    with torch.inference_mode(), backend(MONTLOK_CPP="1"):
        broadcast_streams, broadcast_prepared = first._prepare(backbone, allow_broadcast=True)
        dense_streams, dense_prepared = first._prepare(expanded)
        for index, (broadcast_value, dense_value) in enumerate(zip(broadcast_prepared, dense_prepared, strict=True)):
            report(
                f"mhc broadcast prepare output {index}",
                broadcast_value,
                dense_value,
                0.0,
            )

        out1 = torch.tanh(broadcast_prepared[0] @ mix1)
        fused_streams, fused_prepared = first._combine_prepare(
            broadcast_streams,
            broadcast_prepared,
            out1,
            second,
        )
        separate_streams = first._combine(broadcast_streams, broadcast_prepared, out1)
        separate_streams, separate_prepared = second._prepare(separate_streams)
        report("mhc combine_prepare streams", fused_streams, separate_streams, 0.0)
        for index, (fused_value, separate_value) in enumerate(zip(fused_prepared, separate_prepared, strict=True)):
            report(f"mhc combine_prepare output {index}", fused_value, separate_value, 0.0)

        out2 = torch.tanh(fused_prepared[0] @ mix2)
        collapsed = second._combine_collapse(fused_streams, fused_prepared, out2)
        materialized = second._combine(fused_streams, fused_prepared, out2)
        report(
            "mhc combine_collapse",
            collapsed,
            materialized.double().mean(dim=-2).float(),
            3e-7,
        )


def test_cpu_layers() -> None:
    torch.manual_seed(13)
    cases = [
        ("RMSNorm [2,37,256]", RMSNorm(256, eps=1e-6), torch.randn(2, 37, 256) * 3),
        ("RMSNorm [480,256]", RMSNorm(256, eps=1e-5), torch.randn(480, 256)),
        (
            "GroupedRMSNorm g8 [1,480,256]",
            GroupedRMSNorm(256, 8, eps=1e-6),
            torch.randn(1, 480, 256),
        ),
        (
            "GroupedRMSNorm g4 [3,5,100]",
            GroupedRMSNorm(100, 4, eps=1e-6),
            torch.randn(3, 5, 100),
        ),
        (
            "RMSNorm strided input [37,2,64]->transposed",
            RMSNorm(64, eps=1e-6),
            torch.randn(64, 37).t(),
        ),
        (
            "LayerNorm [2,37,64]",
            nn.LayerNorm(64, eps=1e-5),
            torch.randn(2, 37, 64) * 2 + 0.5,
        ),
        (
            "LayerNorm no bias, strided [37,64]",
            nn.LayerNorm(64, eps=1e-5, bias=False),
            torch.randn(64, 37).t(),
        ),
        (
            "LayerNorm multidimensional fallback",
            nn.LayerNorm((4, 8)),
            torch.randn(3, 4, 8),
        ),
    ]
    for name, norm, x in cases:
        norm = norm.eval()
        with torch.no_grad():
            if norm.weight is not None:
                norm.weight.mul_(torch.rand_like(norm.weight) + 0.5)
            if getattr(norm, "bias", None) is not None:
                norm.bias.add_(torch.randn_like(norm.bias) * 0.25)
        holder = nn.Sequential(norm)
        with torch.inference_mode():
            expected = holder(x)
            swapped = install_cpu_layers(holder)
            swapped_again = install_cpu_layers(holder)
            with backend(MONTLOK_CPP="1"):
                actual = holder(x)
            with backend(MONTLOK_CPP="1", MONTLOK_CPP_LAYERS="0"):
                disabled = holder(x)
        if swapped != 1 or swapped_again != 0:
            FAILURES.append(name + " install")
            print(f"{name}: install_cpu_layers swaps {swapped}, then {swapped_again} (expected 1, then 0) FAIL")
        report(name, actual, expected, 3e-6 * (1.0 + float(expected.abs().max())))
        report(name + " MONTLOK_CPP_LAYERS=0 passthrough", disabled, expected, 0.0)

    for d_model, hidden, shape, bias in (
        (64, 100, (2, 37, 64), True),
        (33, 51, (3, 11, 33), False),
    ):
        layer = SwiGLU(d_model, hidden, bias=bias).eval()
        x = torch.randn(*shape)
        holder = nn.Sequential(layer)
        with torch.inference_mode():
            expected = holder(x)
            swapped = install_cpu_layers(holder)
            with backend(MONTLOK_CPP="1"):
                actual = holder(x)
            with backend(MONTLOK_CPP="1", MONTLOK_CPP_LAYERS="0"):
                disabled = holder(x)
        name = f"SwiGLU d{d_model} hidden{hidden} bias{int(bias)}"
        if swapped != 1:
            FAILURES.append(name + " install")
            print(f"{name}: install_cpu_layers swapped {swapped} modules (expected 1) FAIL")
        report(name, actual, expected, 3e-6 * (1.0 + float(expected.abs().max())))
        report(name + " MONTLOK_CPP_LAYERS=0 passthrough", disabled, expected, 0.0)


def load_config() -> dict:
    return json.loads((HERE.parent / "rdt4quant" / "config.json").read_text())


def test_mla() -> None:
    cfg = RDTConfig(**load_config()["model"])
    for batch, length in ((1, 480), (2, 37)):
        torch.manual_seed(17)
        mla = MLACPU(cfg).eval()
        x = torch.randn(batch, length, cfg.d_model)
        with torch.inference_mode():
            expected = MLA.forward(mla, x)
            expected_noncausal = MLA.forward(mla, x, causal=False)
            with backend(MONTLOK_CPP="1"):
                actual = mla(x)
                again = mla(x)  # second call uses the cached rope tables
                noncausal = mla(x, causal=False)
                last = mla.forward_last(x)
                n_last = min(5, length)
                trailing = mla.forward_last(x, n_last)
            with backend(MONTLOK_CPP="1", MONTLOK_CPP_LAYERS="0"):
                disabled = mla(x)
                trailing_disabled = mla.forward_last(x, n_last)
        name = f"mla heads{cfg.n_heads} hd{cfg.head_dim} B{batch} L{length}"
        tol = 4e-6 * (1.0 + float(expected.abs().max()))
        report(name, actual, expected, tol)
        report(name + " cached tables", again, actual, 0.0)
        report(name + " noncausal", noncausal, expected_noncausal, tol)
        report(name + " last query", last, expected[:, -1:], tol)
        report(name + f" trailing {n_last} queries", trailing, expected[:, -n_last:], tol)
        report(
            name + f" trailing {n_last} fallback",
            trailing_disabled,
            expected[:, -n_last:],
            0.0,
        )
        report(name + " MONTLOK_CPP_LAYERS=0 passthrough", disabled, expected, 0.0)

        with torch.no_grad():
            mla.kv_down.weight.add_(0.01)
        with torch.inference_mode():
            expected_updated = MLA.forward(mla, x)
            with backend(MONTLOK_CPP="1"):
                actual_updated = mla(x)
        report(
            name + " projection cache invalidation",
            actual_updated,
            expected_updated,
            tol,
        )


def test_full_model() -> None:
    from model_cpu import MultiAssetRDTCPU, fused_stage2

    config = load_config()
    torch.manual_seed(0)
    model = MultiAssetRDTCPU(config, 31).eval()
    with torch.no_grad():
        # The dynamic mHC maps initialise to zero; move them so they are exercised.
        for layer in model.recurrent.stage2:
            for hc in (layer.attn_hc, layer.ffn_hc):
                for alpha in (hc.pre_alpha, hc.post_alpha, hc.res_alpha):
                    alpha.fill_(0.5)
    torch.manual_seed(1)
    crypto = torch.randn(2, 480, 57)
    equities = torch.randn(2, 480, 23)
    assets = torch.tensor([3, 17])

    def run():
        with torch.inference_mode():
            quantiles, hidden_rms = model(crypto, 4, "crypto")
            hidden, _ = model.encode(crypto, 4)
            stock, stock_hidden_rms = model(equities, 4, "equity_daily", assets)
        return quantiles, hidden_rms, hidden, stock, stock_hidden_rms

    with backend():
        q_ref, rms_ref, h_ref, s_ref, stock_rms_ref = run()
    with backend(MONTLOK_CPP="1", MONTLOK_CPP_TAIL="0"):
        q_full, rms_full, h_full, s_full, stock_rms_full = run()
    with backend(MONTLOK_CPP="1"):
        q_tail, rms_tail, h_tail_encode, s_tail, stock_rms_tail = run()
        with torch.inference_mode():
            embedded = model.input_norm(model.projection(crypto.float()))
            backbone = model.recurrent._run_stage1(
                embedded,
                word_pos=None,
                morph_depth=None,
                attn_mask=None,
                causal=True,
            )
            applications = [layer for _ in range(4) for layer in model.recurrent.stage2]
            fused_full_hidden = fused_stage2(applications, backbone)
            fused_tail_hidden = fused_stage2(applications, backbone, 1)
    with backend(MONTLOK_CPP="1", MONTLOK_CPP_MHC="0", MONTLOK_CPP_LAYERS="0"):
        q_mamba, _rms_mamba, h_mamba, _s_mamba, _stock_rms_mamba = run()
    hidden_tol = 1e-4 * (1.0 + float(h_ref.abs().max()))
    quantile_tol = 1e-5 * (1.0 + float(q_ref.abs().max()))
    stock_tol = 1e-5 * (1.0 + float(s_ref.abs().max()))
    report("full model hidden states (all fused ops)", h_full, h_ref, hidden_tol)
    report("full model crypto quantiles (full fused ops)", q_full, q_ref, quantile_tol)
    report("full model crypto quantiles (tail fused chain)", q_tail, q_ref, quantile_tol)
    report("full model crypto tail vs full C++", q_tail, q_full, quantile_tol)
    report("full model equity quantiles (full fused ops)", s_full, s_ref, stock_tol)
    report("full model equity quantiles (tail fused chain)", s_tail, s_ref, stock_tol)
    report(
        "full model full hidden fused chain",
        fused_full_hidden,
        h_tail_encode,
        hidden_tol,
    )
    report(
        "full model tail hidden fused chain",
        fused_tail_hidden,
        h_tail_encode[:, -1:],
        hidden_tol,
    )
    report("full model full hidden RMS", rms_full, rms_ref, hidden_tol)
    report(
        "full model tail hidden RMS",
        rms_tail,
        h_tail_encode[:, -1:].detach().float().square().mean().sqrt(),
        hidden_tol,
    )
    report("full model stock full hidden RMS", stock_rms_full, stock_rms_ref, hidden_tol)
    if not torch.isfinite(stock_rms_tail):
        FAILURES.append("full model stock tail hidden RMS")
        print("full model stock tail hidden RMS is not finite FAIL")
    report(
        "full model hidden states (Mamba kernel only)",
        h_mamba,
        h_ref,
        1e-4 * (1.0 + float(h_ref.abs().max())),
    )
    report("full model crypto quantiles (Mamba kernel only)", q_mamba, q_ref, quantile_tol)


def main() -> int:
    torch.set_num_threads(int(os.environ.get("MONTLOK_TEST_THREADS", min(8, os.cpu_count() or 1))))
    from montlok_loader import load_montlok

    info = load_montlok().build_info()
    print(f"montlok extension: {info}")
    for test in (
        test_mamba_layer,
        test_mhc,
        test_mhc_chain,
        test_cpu_layers,
        test_mla,
        test_full_model,
    ):
        print(f"--- {test.__name__}")
        test()
    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) FAILED:")
        for name in FAILURES:
            print(f"  {name}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
