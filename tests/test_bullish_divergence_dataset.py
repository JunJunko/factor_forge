from __future__ import annotations

import numpy as np
import pandas as pd

from factor_forge.ml.bullish_divergence_config import BullishDivergenceFeatureConfig
from factor_forge.ml.bullish_divergence_dataset import (
    DIVERGENCE_FEATURES,
    TOUCH_MODEL_FEATURES,
    build_bullish_divergence_features,
    build_divergence_episodes,
    bullish_divergence_feature_manifest,
)


def _divergence_panel(*, include_retest: bool = True) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    from conftest import make_panel

    panel = make_panel(days=110, stocks=3)
    dates = pd.bdate_range("2024-01-02", periods=110)
    code = "000000.SZ"
    mask = panel["ts_code"].eq(code)
    panel.loc[mask, ["raw_open", "raw_close", "adj_open", "adj_close"]] = 10.0
    panel.loc[mask, ["raw_high", "adj_high"]] = 10.10
    panel.loc[mask, ["raw_low", "adj_low"]] = 9.90
    panel.loc[mask, "pre_close"] = 10.0

    def candle(position: int, *, open_: float, high: float, low: float, close: float) -> None:
        row = panel["ts_code"].eq(code) & panel["trade_date"].eq(dates[position])
        panel.loc[row, ["raw_open", "adj_open"]] = open_
        panel.loc[row, ["raw_high", "adj_high"]] = high
        panel.loc[row, ["raw_low", "adj_low"]] = low
        panel.loc[row, ["raw_close", "adj_close"]] = close

    candle(50, open_=8.50, high=8.70, low=8.00, close=8.30)  # prior trough A
    candle(60, open_=9.40, high=9.80, low=9.30, close=9.60)  # intervening rebound
    candle(75, open_=8.20, high=8.40, low=7.80, close=8.10)  # current trough B
    if include_retest:
        candle(77, open_=8.05, high=8.20, low=7.82, close=7.98)  # post-B touch/reclaim
    candle(78, open_=8.30, high=8.55, low=8.25, close=8.45)
    return panel, dates


def _target(features: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    row = features.loc[
        features["ts_code"].eq("000000.SZ") & features["trade_date"].eq(date)
    ]
    assert len(row) == 1
    return row.iloc[0]


def test_touch_factor_emits_numeric_anchor_and_post_b_retest():
    panel, dates = _divergence_panel(include_retest=True)
    features, names = build_bullish_divergence_features(panel)
    row = _target(features, dates[78])

    assert set(DIVERGENCE_FEATURES + TOUCH_MODEL_FEATURES) == set(names)
    assert row["div__pivot_a_date"] == dates[50]
    assert row["div__pivot_b_date"] == dates[75]
    assert np.isclose(row["touch__level_raw"], 7.80)
    assert np.isclose(row["touch__level_adj_pit"], 7.80)
    assert np.isclose(row["touch__level_to_close"], 7.80 / 8.45 - 1.0)
    assert row["touch__occurred_10d"] == 1.0
    assert row["touch__post_b_count"] >= 1.0
    assert row["touch__age_days"] == 1.0
    assert row["touch__last_close_reclaim_atr"] > 0
    assert 0 <= row["touch__acceptance_score"] <= 100


def test_anchor_candle_is_excluded_from_touch_count():
    panel, dates = _divergence_panel(include_retest=False)
    features, _ = build_bullish_divergence_features(panel)
    row = _target(features, dates[78])

    assert row["div__pivot_b_date"] == dates[75]
    assert row["touch__occurred_10d"] == 0.0
    assert row["touch__count_10d"] == 0.0
    assert row["touch__post_b_count"] == 0.0


def test_same_day_trough_marks_post_b_touch_as_unobservable_not_zero():
    panel, dates = _divergence_panel(include_retest=False)
    features, _ = build_bullish_divergence_features(panel)
    row = _target(features, dates[75])

    assert row["div__pivot_b_date"] == dates[75]
    assert row["touch__post_b_observable"] == 0.0
    assert pd.isna(row["touch__post_b_count"])


def test_future_mutation_does_not_change_prior_divergence_or_touch_features():
    panel, dates = _divergence_panel(include_retest=True)
    before, names = build_bullish_divergence_features(panel)
    mutated = panel.copy()
    future = mutated["trade_date"].gt(dates[78]) & mutated["ts_code"].eq("000000.SZ")
    for column in ("raw_open", "raw_high", "raw_low", "raw_close", "adj_open", "adj_high", "adj_low", "adj_close"):
        mutated.loc[future, column] *= 3.0
    mutated.loc[future, "amount_cny"] *= 10.0
    after, _ = build_bullish_divergence_features(mutated)

    left = _target(before, dates[78])
    right = _target(after, dates[78])
    compare = names + ["touch__level_raw", "touch__level_adj_pit"]
    assert np.allclose(
        pd.to_numeric(left[compare], errors="coerce"),
        pd.to_numeric(right[compare], errors="coerce"),
        equal_nan=True,
    )
    assert left["div__pivot_a_date"] == right["div__pivot_a_date"]
    assert left["div__pivot_b_date"] == right["div__pivot_b_date"]


def test_invalid_stock_does_not_change_valid_stock_cross_section_scores():
    panel, dates = _divergence_panel(include_retest=True)
    invalid = panel["ts_code"].eq("000002.SZ")
    panel.loc[invalid, "is_st"] = True
    before, _ = build_bullish_divergence_features(panel)
    mutated = panel.copy()
    for column in ("raw_high", "adj_high"):
        mutated.loc[invalid, column] *= 20.0
    for column in ("raw_low", "adj_low"):
        mutated.loc[invalid, column] *= 0.05
    after, _ = build_bullish_divergence_features(mutated)
    left = _target(before, dates[78])
    right = _target(after, dates[78])
    assert np.isclose(left["div__score"], right["div__score"], equal_nan=True)
    assert np.isclose(
        left["touch__acceptance_score"], right["touch__acceptance_score"], equal_nan=True
    )


def test_episode_builder_keeps_first_signal_inside_cooldown():
    dates = pd.bdate_range("2025-01-02", periods=20)
    features = pd.DataFrame({
        "trade_date": dates,
        "ts_code": ["000001.SZ"] * len(dates),
        "div__event_candidate": [index in {2, 3, 8, 19} for index in range(len(dates))],
        "div__score": np.arange(len(dates), dtype=float),
    })
    episodes = build_divergence_episodes(
        features, BullishDivergenceFeatureConfig(episode_cooldown_days=5)
    )
    assert episodes["trade_date"].tolist() == [dates[2], dates[8], dates[19]]
    assert episodes["event_id"].is_unique


def test_manifest_keeps_raw_touch_prices_out_of_predictors():
    manifest = pd.DataFrame(bullish_divergence_feature_manifest()).set_index("name")
    assert manifest.loc["touch__level_raw", "role"] == "diagnostic_not_predictor"
    assert manifest.loc["touch__level_adj_pit", "role"] == "diagnostic_not_predictor"
    assert manifest.loc["touch__level_to_close", "role"] == "predictor"
