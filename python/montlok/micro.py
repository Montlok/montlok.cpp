"""Streaming L1 / trade microstructure features (schema ``xstock-observable-l1-v2-32``).

Events arrive in receive order on one collector clock:

* ``REFERENCE`` (0): a = raw reference price, b = fair value, c = uncertainty (bps)
* ``BOOK`` (1): a = bid, b = ask, c = bid size, d = ask size
* ``TRADE`` (2): a = price, b = quantity, side = +1 buy / -1 sell aggressor

``FeatureEngine.update`` feeds one event; ``snapshot`` returns the 32 features
(``None`` while warming up). ``replay`` feeds whole columns in C++ and returns
``(features[N, 32], ready[N])``. A rejected event (non-monotonic time,
non-finite values, capacity overflow) raises ``ValueError`` and the engine
must be ``reset()``.
"""

from __future__ import annotations

from ._core.micro import FEATURE_COUNT, SCHEMA, FeatureEngine, linear_heads

REFERENCE, BOOK, TRADE = 0, 1, 2

FEATURES: tuple[str, ...] = (
    "fair_gap_mid_bps", "buy_gap_bps", "sell_gap_bps", "fair_uncertainty_bps",
    "ref_ret_100ms_bps", "ref_ret_500ms_bps", "ref_ret_2000ms_bps", "ref_accel_100ms_bps",
    "spread_bps", "imbalance_l1", "micro_minus_mid_bps",
    "ofi_100ms_over_depth", "ofi_500ms_over_depth", "trade_imbalance_100ms",
    "trade_imbalance_500ms", "volume_100ms_over_depth", "volume_500ms_over_depth",
    "trade_messages_100ms", "trade_messages_500ms", "signed_flow_ema_100ms_over_depth",
    "signed_flow_ema_500ms_over_depth", "signed_flow_ema_2000ms_over_depth",
    "depth_change_100ms", "depth_change_500ms", "bid_price_age_ms", "ask_price_age_ms",
    "ref_receive_age_ms", "book_receive_age_ms", "last_refill_duration_ms",
    "last_refill_completed_age_ms", "refill_pending_age_ms", "has_refill_observation",
)

if len(FEATURES) != FEATURE_COUNT:  # pragma: no cover - guards a header/name drift
    raise ImportError("montlok.micro feature names do not match the native schema")

__all__ = ["BOOK", "FEATURES", "FEATURE_COUNT", "FeatureEngine", "REFERENCE", "SCHEMA", "TRADE", "linear_heads"]
