from __future__ import annotations

import numpy as np
import pandas as pd

from factor_forge.ml.bullish_divergence_v2 import (
    BullishDivergenceV2Config,
    build_bullish_divergence_v2_features,
    build_v2_post_signal_retest_events,
)
from factor_forge.research.bullish_divergence_v2_event_study import (
    BullishDivergenceV2EventStudyConfig,
    _bootstrap_complete_case,
)


def _v2_panel() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    from conftest import make_panel

    panel = make_panel(days=130, stocks=4)
    dates = pd.bdate_range("2024-01-02", periods=130)
    code = "000000.SZ"
    mask = panel["ts_code"].eq(code)
    baseline = 10.0
    panel.loc[mask, ["raw_open", "raw_close", "adj_open", "adj_close"]] = baseline
    panel.loc[mask, ["raw_high", "adj_high"]] = 10.10
    panel.loc[mask, ["raw_low", "adj_low"]] = 9.90

    def candle(position: int, open_: float, high: float, low: float, close: float) -> None:
        row = mask & panel["trade_date"].eq(dates[position])
        panel.loc[row, ["raw_open", "adj_open"]] = open_
        panel.loc[row, ["raw_high", "adj_high"]] = high
        panel.loc[row, ["raw_low", "adj_low"]] = low
        panel.loc[row, ["raw_close", "adj_close"]] = close

    candle(37, 9.4, 9.5, 9.3, 9.35)
    candle(38, 9.0, 9.1, 8.9, 8.95)
    candle(39, 8.6, 8.7, 8.5, 8.55)
    candle(40, 8.3, 8.4, 8.0, 8.25)
    candle(55, 9.4, 9.8, 9.3, 9.6)
    candle(67, 8.8, 8.9, 8.7, 8.75)
    candle(68, 8.5, 8.6, 8.4, 8.45)
    candle(69, 8.2, 8.3, 8.1, 8.15)
    candle(70, 8.0, 8.2, 7.9, 8.10)
    return panel, dates


def test_v2_geometry_uses_strict_20_to_60_day_trough_gap():
    panel, _ = _v2_panel()
    features, _ = build_bullish_divergence_v2_features(panel)
    geometry = features.loc[features["div_v2__geometry_candidate"]]
    assert len(geometry) > 0
    assert geometry["div__trough_gap_days"].between(20, 60).all()
    assert geometry["div__descent_into_a_3d"].lt(0).all()
    assert geometry["div__descent_into_b_3d"].lt(0).all()
    assert geometry["div__price_lower_low_atr"].between(-.25, 1.0).all()


def test_v2_score_is_component_scaled_and_does_not_multiply_reliability():
    panel, _ = _v2_panel()
    config = BullishDivergenceV2Config()
    features, _ = build_bullish_divergence_v2_features(panel, config)
    rows = features.loc[features["div_v2__geometry_candidate"]]
    assert len(rows) > 0
    expected = 100 * (
        config.oscillator_weight * rows["div_v2__oscillator_strength"]
        + config.confirmation_weight * rows["div_v2__confirmation_strength"]
    )
    assert np.allclose(rows["div_v2__score"], expected, equal_nan=True)


def test_post_signal_retest_clock_ignores_pre_signal_touches():
    from conftest import make_panel

    panel = make_panel(days=80, stocks=1)
    dates = pd.bdate_range("2024-01-02", periods=80)
    code = "000000.SZ"
    signal = dates[50]
    # A pre-signal intersection must not become the retest event.
    pre = panel["trade_date"].eq(dates[49])
    panel.loc[pre, ["adj_low", "raw_low"]] = 9.95
    panel.loc[pre, ["adj_high", "raw_high"]] = 10.05
    trigger = panel["trade_date"].eq(dates[52])
    panel.loc[trigger, ["adj_low", "raw_low"]] = 9.95
    panel.loc[trigger, ["adj_high", "raw_high"]] = 10.20
    panel.loc[trigger, ["adj_close", "raw_close"]] = 10.10
    episodes = pd.DataFrame({
        "event_id": ["v2:000000.SZ:20240312"],
        "trade_date": [signal],
        "ts_code": [code],
        "touch__level_adj_pit": [10.0],
        "touch__zone_width_adj": [.05],
        "div_v2__score": [75.0],
        "div_v2__score_rank": [.8],
    })
    retests = build_v2_post_signal_retest_events(panel, episodes)
    assert len(retests) == 1
    assert retests.iloc[0]["trade_date"] == dates[52]
    assert retests.iloc[0]["retest_v2__age_days"] == 2
    assert bool(retests.iloc[0]["retest_v2__reclaimed"])


def test_v2_bootstrap_contrast_uses_same_date_complete_cases():
    dates = pd.bdate_range("2025-01-02", periods=6)
    daily = pd.DataFrame({
        "Q1": [0.0, 0.1, np.nan, 0.2, np.nan, 0.3],
        "Q5": [0.2, np.nan, 0.4, 0.5, np.nan, 0.7],
    }, index=dates)
    result = _bootstrap_complete_case(
        daily, "Q5", "Q1", "Q5_minus_Q1",
        BullishDivergenceV2EventStudyConfig(
            bootstrap_samples=20, block_length=2
        ),
    )
    expected = np.mean([0.2, 0.3, 0.4])
    assert result["usable_dates"] == 3
    assert np.isclose(result["estimate"], expected)

