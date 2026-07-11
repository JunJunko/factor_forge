from __future__ import annotations

import numpy as np
import pandas as pd

from factor_forge.ml.atr_reversion_config import ATRReversionFeatureConfig, ATRReversionLabelConfig
from factor_forge.ml.atr_reversion_dataset import FEATURE_GROUPS, CORE_FEATURES, build_atr_reversion_dataset
from factor_forge.ml.atr_reversion_features import rolling_percentile


def test_rolling_percentile_uses_trailing_window_only():
    idx = pd.RangeIndex(6)
    value = pd.Series([1, 2, 3, 4, 5, 100], index=idx, dtype=float)
    stocks = pd.Series(["A"] * 6, index=idx)
    pct_a = rolling_percentile(value, stocks, window=4)
    mutated = value.copy()
    mutated.iloc[-1] = -100
    pct_b = rolling_percentile(mutated, stocks, window=4)
    assert np.allclose(pct_a.iloc[:5], pct_b.iloc[:5], equal_nan=True)


def test_build_atr_reversion_dataset_emits_core_features():
    from conftest import make_panel

    panel = make_panel(days=180, stocks=5)
    # Create one obvious long-lower-shadow observation so the core signal is computable.
    mask = (panel["ts_code"].eq("000000.SZ")) & (panel["trade_date"].eq(panel["trade_date"].iloc[150]))
    panel.loc[mask, "raw_low"] = panel.loc[mask, "raw_close"] * 0.92
    panel.loc[mask, "adj_low"] = panel.loc[mask, "adj_close"] * 0.92
    ds, names = build_atr_reversion_dataset(
        panel,
        ATRReversionFeatureConfig(cross_sectional_zscore=False, winsor_quantile=0),
        ATRReversionLabelConfig(horizons=[3, 5], primary_horizon=5),
    )
    assert set(CORE_FEATURES) <= set(names)
    assert set(CORE_FEATURES) <= set(ds.columns)
    assert "label" in ds.columns and "label_3" in ds.columns
    assert np.isfinite(ds[CORE_FEATURES].to_numpy()).sum() > 0


def test_bollinger_touch_uses_only_prior_close_history_and_marks_missing_flow():
    from conftest import make_panel

    panel = make_panel(days=180, stocks=5)
    features = ATRReversionFeatureConfig(cross_sectional_zscore=False, winsor_quantile=0)
    label = ATRReversionLabelConfig(horizons=[5], primary_horizon=5)
    before, names = build_atr_reversion_dataset(panel, features, label)
    target_date = panel.loc[panel["ts_code"].eq("000000.SZ"), "trade_date"].iloc[140]
    target = (before["instrument"].eq("000000.SZ") & before["datetime"].eq(target_date))

    # Changing today's close must not change today's lower band / touch depth:
    # those use the trailing close window ending at t-1 and ATR(t-1).
    mutated = panel.copy()
    mask = mutated["ts_code"].eq("000000.SZ") & mutated["trade_date"].eq(target_date)
    mutated.loc[mask, "adj_close"] *= 1.4
    after, _ = build_atr_reversion_dataset(mutated, features, label)
    assert np.isclose(before.loc[target, "touch_depth_atr"].iloc[0], after.loc[target, "touch_depth_atr"].iloc[0])
    assert set(FEATURE_GROUPS["all"]) == set(names)
    assert before["net_flow_available"].eq(False).all()
    assert before[FEATURE_GROUPS["F"][:2]].isna().all().all()
