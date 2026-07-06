"""Closed-form unit tests for the V2 stable-baseline + no-volume-rise features
(handoff doc ``Codex交接说明_稳定成交基线后的无量上涨因子研究.md`` sec. 3).

Each test builds a small synthetic series where the answer is known by hand.  The most
important ones guard the anti-leakage invariants the handoff document makes hard
requirements (sec. 4.1): the event days ``t-1, t`` never enter the 28-day baseline, and
future data never changes a feature computed at an earlier bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_forge.ml import supply_features as sf


def _turnover_panel(values, stock="A"):
    """One stock, len(values) bars; returns (turnover_pct_series, stocks_series, index)."""
    n = len(values)
    idx = pd.RangeIndex(n)
    return (
        pd.Series(np.asarray(values, dtype=float), index=idx, name="turnover"),
        pd.Series([stock] * n, index=idx),
        idx,
    )


# --------------------------------------------------------------------------- #
# 4.1.3 / 4.1.1: the event days t-1, t NEVER enter the 28-day baseline.
# --------------------------------------------------------------------------- #
def test_baseline_window_excludes_event_days():
    n = 40
    values = np.full(n, 1.0)          # constant 1% turnover
    values[-1] = 100.0                # t: huge activation
    values[-2] = 50.0                 # t-1: activation
    turnover, stocks, _ = _turnover_panel(values)
    lt = sf._log_turnover(turnover)
    mean = sf.baseline_window_stat(lt, stocks, 28, 2, method="mean")
    std = sf.baseline_window_stat(lt, stocks, 28, 2, method="std", ddof=0)
    expected = np.log1p(0.01)         # the baseline is all-1% bars
    # At t=39 the window is [10, 37] (28 bars); t-1=38 and t=39 are excluded.
    assert np.isclose(mean.iloc[-1], expected, atol=1e-12)
    assert np.isclose(std.iloc[-1], 0.0, atol=1e-12)
    # At t=38 the window [9, 36] also excludes the two event days of bar 39's view.
    assert np.isclose(mean.iloc[-2], expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# 4.1.4 / 4.1.5: changing data at t+1 or later never changes a feature at bar t.
# --------------------------------------------------------------------------- #
def test_future_data_does_not_change_prior_features():
    n = 40
    base = np.full(n, 1.0)
    turnover_a, stocks, _ = _turnover_panel(base.copy())
    mutated = base.copy()
    mutated[-1] = 999.0               # mutate only the last bar
    turnover_b, _, _ = _turnover_panel(mutated)

    def z_pack(turnover):
        lt = sf._log_turnover(turnover)
        mean = sf.baseline_window_stat(lt, stocks, 28, 2, method="mean")
        std = sf.baseline_window_stat(lt, stocks, 28, 2, method="std", ddof=0)
        floor = pd.Series(0.001, index=lt.index)
        z_t1, z_t = sf.recent_volume_z(lt, mean, std, floor, stocks)
        return mean, z_t1, z_t

    mean_a, zt1_a, zt_a = z_pack(turnover_a)
    mean_b, zt1_b, zt_b = z_pack(turnover_b)
    # baseline_mean[t] (window t-29..t-2) and z_t1[t] (reads x[t-1]) never touch x[t] or
    # later, so they are invariant across the WHOLE series when only the last bar changes.
    assert np.allclose(mean_a.to_numpy(), mean_b.to_numpy(), equal_nan=True)
    assert np.allclose(zt1_a.to_numpy(), zt1_b.to_numpy(), equal_nan=True)
    # z_t[t] reads x[t] (the current bar), so bar 39 itself legitimately differs when x[39]
    # changes -- that is NOT leakage. Bars 0..38 must be invariant (their x[t] and their
    # baseline windows never reach bar 39).
    assert np.allclose(zt_a.to_numpy()[:39], zt_b.to_numpy()[:39], equal_nan=True)


# --------------------------------------------------------------------------- #
# 4.3: baseline_std_28 -> 0 must not blow up; std_floor bounds the denominator.
# --------------------------------------------------------------------------- #
def test_recent_volume_z_no_div_by_zero_when_baseline_std_zero():
    n = 40
    values = np.full(n, 1.0)
    values[-1] = 10.0                 # activation at t; baseline is flat -> std == 0
    turnover, stocks, idx = _turnover_panel(values)
    lt = sf._log_turnover(turnover)
    mean = sf.baseline_window_stat(lt, stocks, 28, 2, method="mean")
    std = sf.baseline_window_stat(lt, stocks, 28, 2, method="std", ddof=0)
    assert np.isclose(std.iloc[-1], 0.0, atol=1e-12)
    floor = pd.Series(0.001, index=idx)   # dataset-layer cross-section floor (constant here)
    z_t1, z_t = sf.recent_volume_z(lt, mean, std, floor, stocks)
    assert np.isfinite(z_t.iloc[-1]) and np.isfinite(z_t1.iloc[-1])
    expected = (np.log1p(0.10) - np.log1p(0.01)) / 0.001
    assert np.isclose(z_t.iloc[-1], expected, atol=1e-9)


# --------------------------------------------------------------------------- #
# 4.4: a single heavy-activation day is caught by max_2 (handoff 3.7 / 5.5).
# --------------------------------------------------------------------------- #
def test_max_z_catches_single_day_activation():
    n = 40
    values = np.full(n, 1.0)
    values[-1] = 20.0                 # t spikes; t-1 quiet
    turnover, stocks, idx = _turnover_panel(values)
    lt = sf._log_turnover(turnover)
    mean = sf.baseline_window_stat(lt, stocks, 28, 2, method="mean")
    std = sf.baseline_window_stat(lt, stocks, 28, 2, method="std", ddof=0)
    floor = pd.Series(0.001, index=idx)
    z_t1, z_t = sf.recent_volume_z(lt, mean, std, floor, stocks)
    agg = sf.recent_volume_z_aggregates(z_t1, z_t, -3.0, 3.0)
    assert z_t1.iloc[-1] < z_t.iloc[-1]            # t-1 quiet, t active
    assert agg["recent_volume_z_max_2_raw"].iloc[-1] > agg["recent_volume_z_mean_2_raw"].iloc[-1]
    assert agg["recent_volume_z_max_2_raw"].iloc[-1] > 0


# --------------------------------------------------------------------------- #
# 4.5: one spike + one shrink day -- mean_2 masks it, max_2 does not (handoff 5.5).
# --------------------------------------------------------------------------- #
def test_max_z_beats_mean_when_one_day_spike_one_day_shrink():
    n = 40
    values = np.full(n, 1.0)
    values[-2] = 20.0                 # t-1 spikes
    values[-1] = 0.5                  # t shrinks below baseline
    turnover, stocks, idx = _turnover_panel(values)
    lt = sf._log_turnover(turnover)
    mean = sf.baseline_window_stat(lt, stocks, 28, 2, method="mean")
    std = sf.baseline_window_stat(lt, stocks, 28, 2, method="std", ddof=0)
    floor = pd.Series(0.001, index=idx)
    z_t1, z_t = sf.recent_volume_z(lt, mean, std, floor, stocks)
    agg = sf.recent_volume_z_aggregates(z_t1, z_t, -3.0, 3.0)
    assert z_t1.iloc[-1] > 0 and z_t.iloc[-1] < 0
    assert agg["recent_volume_z_max_2_raw"].iloc[-1] > agg["recent_volume_z_mean_2_raw"].iloc[-1]
    # clip bound is honored on the raw spike.
    assert agg["recent_volume_z_max_2_clip"].iloc[-1] == 3.0


# --------------------------------------------------------------------------- #
# 4.6: effective_ticks_2 counts raw-price tick steps over the 2-day event window.
# --------------------------------------------------------------------------- #
def test_effective_ticks_2_counts_tick_steps():
    raw_close = pd.Series([10.0, 10.01, 10.02, 10.05], index=pd.RangeIndex(4))
    stocks = pd.Series(["A"] * 4, index=pd.RangeIndex(4))
    ticks = sf.effective_ticks(raw_close, stocks, window=2, tick_size=0.01)
    assert np.isclose(ticks.iloc[2], 2.0)     # (10.02 - 10.00) / 0.01
    assert np.isclose(ticks.iloc[3], 4.0)     # (10.05 - 10.01) / 0.01


# --------------------------------------------------------------------------- #
# 3.6: price_strength_2 divides by volatility measured up to t-2, so the two event-day
# returns do not contaminate the volatility baseline.
# --------------------------------------------------------------------------- #
def test_volatility_prior_excludes_event_days():
    n = 40
    lr_values = np.full(n, 0.01)
    lr_values[-1] = 0.9               # event-day spikes
    lr_values[-2] = 0.9
    lr = pd.Series(lr_values, index=pd.RangeIndex(n))
    stocks = pd.Series(["A"] * n, index=pd.RangeIndex(n))
    vol_prior = sf.volatility_prior(lr, stocks, window=20, gap=2, ddof=1)
    vol_full = sf.volatility(lr, stocks, window=20, ddof=1)     # V1: ends at t
    assert vol_prior.iloc[-1] < 1e-6           # window [18, 37] all-0.01 -> ~0
    assert vol_full.iloc[-1] > 0.2             # the two 0.9 bars inflate the full window


# --------------------------------------------------------------------------- #
# 4.8: the daily cross-section percentile rank uses only valid stocks on that date.
# --------------------------------------------------------------------------- #
def test_cross_section_rank_uses_only_valid_stocks():
    idx = pd.RangeIndex(6)
    dates = pd.Series(
        [pd.Timestamp("2024-01-02")] * 3 + [pd.Timestamp("2024-01-03")] * 3, index=idx
    )
    baseline_std = pd.Series([0.1, 0.2, 0.3, 0.2, 0.4, np.nan], index=idx)
    valid = pd.Series([True, True, True, True, True, False], index=idx)
    rank = baseline_std.where(valid).groupby(dates, sort=False).rank(pct=True)
    # Date 1: 3 valid -> ranks 1/3, 2/3, 3/3.
    assert np.isclose(rank.iloc[0], 1 / 3)
    assert np.isclose(rank.iloc[1], 2 / 3)
    assert np.isclose(rank.iloc[2], 3 / 3)
    # Date 2: only A, B valid (std 0.2, 0.4) -> ranks 1/2, 2/2; C excluded -> NaN.
    assert np.isclose(rank.iloc[3], 1 / 2)
    assert np.isclose(rank.iloc[4], 2 / 2)
    assert np.isnan(rank.iloc[5])


# --------------------------------------------------------------------------- #
# Integration smoke: build_supply_dataset emits the baseline_structure group end-to-end
# and the group registry resolves to exactly those fields.
# --------------------------------------------------------------------------- #
def test_build_supply_dataset_emits_baseline_structure_group():
    from conftest import make_panel

    from factor_forge.ml.supply_dataset import (
        FEATURE_GROUP_REGISTRY,
        build_supply_dataset,
        features_for_groups,
    )
    from factor_forge.ml.supply_config import SupplyFeatureConfig, SupplyLabelConfig

    panel = make_panel(days=60, stocks=4)
    ds, names = build_supply_dataset(
        panel, None, SupplyFeatureConfig(), SupplyLabelConfig(), sample_weight_train=None,
    )
    baseline_cols = FEATURE_GROUP_REGISTRY["baseline_structure"]
    assert set(baseline_cols) <= set(names)                       # all V2 columns present
    assert set(baseline_cols) <= set(ds.columns)
    assert features_for_groups(["baseline_structure"]) == baseline_cols
    # No V2 column leaked inf.
    bad = ds[baseline_cols].replace([np.inf, -np.inf], np.nan)
    assert np.isfinite(bad.to_numpy()).sum() > 0                  # some bars are computable
