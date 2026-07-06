from __future__ import annotations

import numpy as np
import pandas as pd

from factor_forge.ml.atr_reversion_config import ATRReversionFeatureConfig, ATRReversionLabelConfig
from factor_forge.ml.atr_reversion_dataset import CORE_FEATURES, build_atr_reversion_dataset
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

