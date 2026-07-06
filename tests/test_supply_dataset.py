"""Dataset-layer tests: shape, sample filter, A/B group resolution, anti-leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_forge.ml.supply_config import SupplyFeatureConfig, SupplyLabelConfig
from factor_forge.ml.supply_dataset import (
    FEATURE_GROUP_REGISTRY,
    build_supply_dataset,
    features_for_groups,
)


def _make_supply_panel(days: int = 150, n_stocks: int = 5) -> pd.DataFrame:
    """Synthetic panel long enough for the 120-day volume-residual window."""
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2022-01-03", periods=days)
    rows = []
    for s in range(n_stocks):
        code = f"{s:06d}.SZ"
        price = 10.0 + s
        for d, date in enumerate(dates):
            ret = rng.normal(0, 0.02)
            price = max(1.0, price * (1 + ret))
            raw_open = price * (1 + rng.normal(0, 0.005))
            raw_high = max(price, raw_open) * (1 + abs(rng.normal(0, 0.01)))
            raw_low = min(price, raw_open) * (1 - abs(rng.normal(0, 0.01)))
            rows.append({
                "trade_date": date,
                "ts_code": code,
                "raw_open": raw_open, "raw_high": raw_high, "raw_low": raw_low,
                "raw_close": price, "pre_close": price / (1 + ret),
                "adj_factor": 1.0,
                "adj_open": raw_open, "adj_high": raw_high, "adj_low": raw_low,
                "adj_close": price,
                "volume_shares": 1_000_000.0, "amount_cny": 50_000_000.0 * (1 + s),
                "turnover_rate": max(0.1, rng.uniform(0.5, 5.0)),
                "total_mv_cny": 1e9 * (s + 1), "circ_mv_cny": 8e8 * (s + 1),
                "industry_l1_code": f"I{s % 3}",
                "limit_up_price": price * 1.1, "limit_down_price": price * 0.9,
                "is_suspended": False, "is_limit_up_open": False, "is_limit_down_open": False,
                # Mark one stock as ST to exercise the sample filter.
                "is_st": s == 0,
                "is_delisting_period": False,
                "listing_trade_days": 200 + d,
                "is_factor_eligible": True, "is_tradeable": True, "is_liquid": True,
                "st_status_known": True,
            })
    return pd.DataFrame(rows)


def test_build_supply_dataset_shape_and_columns():
    panel = _make_supply_panel()
    features = SupplyFeatureConfig()
    label = SupplyLabelConfig()
    ds, names = build_supply_dataset(panel, index_daily=None, features=features, label=label)
    assert {"datetime", "instrument", "label"} <= set(ds.columns)
    # Every registered feature (both groups) is produced.
    all_features = set().union(*FEATURE_GROUP_REGISTRY.values())
    assert all_features <= set(names)
    assert len(ds) == len(panel)


def test_excluded_samples_are_nan_masked():
    panel = _make_supply_panel()
    ds, names = build_supply_dataset(
        panel, index_daily=None, features=SupplyFeatureConfig(), label=SupplyLabelConfig()
    )
    # Stock 0 is marked is_st=True everywhere -> its feature rows must be NaN.
    st_rows = ds[ds["instrument"] == "000000.SZ"]
    feature_cols = [c for c in names if c in ds.columns]
    assert st_rows[feature_cols].isna().all().all()
    assert st_rows["label"].isna().all()
    # A clean stock should have at least some finite features after warmup.
    clean = ds[ds["instrument"] == "000001.SZ"]
    finite_share = clean[feature_cols].notna().mean().mean()
    assert finite_share > 0.5


def test_feature_groups_resolve_and_partition():
    a = set(features_for_groups(["controls"]))
    b = set(features_for_groups(["controls", "supply_core"]))
    # Model B is a strict superset of Model A.
    assert a <= b
    # The signature feature is exclusive to the supply core.
    assert "volume_residual" in b and "volume_residual" not in a
    assert "excess_ret_5" in b and "excess_ret_5" not in a
    # Controls carry the size / liquidity / environment family.
    for col in ("log_float_market_cap", "amihud_illiquidity_20", "market_breadth", "volatility_20"):
        assert col in a


def test_features_have_no_future_leakage():
    """Recomputing after dropping the last 10 days must not change any feature at prior dates."""
    panel = _make_supply_panel()
    full, names = build_supply_dataset(
        panel, index_daily=None, features=SupplyFeatureConfig(), label=SupplyLabelConfig()
    )
    cutoff = panel["trade_date"].max() - pd.Timedelta(days=14)
    truncated = panel[panel["trade_date"] <= cutoff].copy()
    trunc, _ = build_supply_dataset(
        truncated, index_daily=None, features=SupplyFeatureConfig(), label=SupplyLabelConfig()
    )
    feature_cols = [c for c in names if c in full.columns]
    # Compare on the common (date <= cutoff) keys; features must be identical.
    left = full[full["datetime"] <= cutoff].set_index(["datetime", "instrument"])[feature_cols]
    right = trunc.set_index(["datetime", "instrument"])[feature_cols]
    common = left.index.intersection(right.index)
    a = left.loc[common].sort_index()
    b = right.loc[common].sort_index()
    diff = (a - b).abs()
    # Allow tiny float noise from rolling/clipping; no feature may move materially.
    assert diff.max().max() < 1e-9


def test_label_method_open_to_close_runs():
    panel = _make_supply_panel()
    label = SupplyLabelConfig(label_method="open_to_close", horizon=5)
    ds, _ = build_supply_dataset(panel, index_daily=None, features=SupplyFeatureConfig(), label=label)
    # Just assert it produces a finite label column on clean samples (no crash).
    clean = ds[ds["instrument"] == "000001.SZ"]
    assert clean["label"].notna().any()
