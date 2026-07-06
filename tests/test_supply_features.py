"""Closed-form unit tests for the supply-contraction feature primitives.

Each test constructs a small synthetic panel where the answer is known by hand and
asserts the exact numeric result.  The ``volume_residual`` tests are the most important:
they verify (a) the rolling OLS recovers the known conditional relationship and (b) the
fit window never includes the current bar (anti-leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_forge.ml import supply_features as sf


def _series(arr, name="x"):
    arr = np.asarray(arr)
    if arr.dtype.kind in "iuf":  # integer / unsigned / float only
        arr = arr.astype(float)
    return pd.Series(arr, name=name)


# --------------------------------------------------------------------------- #
# volume_residual: closed-form recovery + anti-leakage
# --------------------------------------------------------------------------- #
def _make_exact_stock(n: int = 200, shock_bar: int | None = None, shock: float = 0.5):
    """One stock, n bars, where log_turnover is an exact linear function of 5 regressors.

    ``shock_bar`` adds ``shock`` to log_turnover at exactly one bar.  Putting the shock
    at the last bar means every earlier bar's fit window [t-120, t-1] excludes it, so the
    OLS recovers the exact coefficients and the residual is ~0 everywhere except the
    shock bar where it equals ``shack``.
    """
    rng = np.random.default_rng(7)
    # Well-conditioned, varying regressors.
    x1 = np.abs(rng.normal(0, 0.02, n))          # |excess_ret_1|
    x2 = rng.uniform(0.01, 0.05, n)              # intraday_range
    x3 = rng.normal(0, 1, n)                      # market_turnover_z
    x4 = rng.normal(0, 1, n)                      # industry_turnover_z
    x5 = rng.uniform(0.01, 0.03, n)              # volatility_20
    coef = np.array([0.4, 1.5, 0.3, 0.2, -0.2, 0.1])  # intercept + 5 slopes
    y = coef[0] + coef[1] * x1 + coef[2] * x2 + coef[3] * x3 + coef[4] * x4 + coef[5] * x5
    if shock_bar is not None:
        y[shock_bar] += shock
    turnover = (np.exp(y) - 1.0) * 100.0  # invert log1p(turnover/100)
    idx = pd.RangeIndex(n)
    return {
        "turnover": pd.Series(turnover, index=idx, name="turnover"),
        "abs_excess_ret_1": pd.Series(x1, index=idx),
        "intraday_range": pd.Series(x2, index=idx),
        "market_turnover_z": pd.Series(x3, index=idx),
        "industry_turnover_z": pd.Series(x4, index=idx),
        "volatility_20": pd.Series(x5, index=idx),
        "stocks": pd.Series(["000001.SZ"] * n, index=idx),
        "y_true": y,
    }


def test_volume_residual_recovers_zero_when_no_noise():
    data = _make_exact_stock(n=200)
    res = sf.volume_residual(
        data["turnover"], data["abs_excess_ret_1"], data["intraday_range"],
        data["market_turnover_z"], data["industry_turnover_z"], data["volatility_20"],
        data["stocks"], window=120, min_periods=80,
    )
    # Bars after the warmup have an exact fit window of clean data → residual ~ 0.
    stable = res.iloc[120:199]
    assert np.allclose(stable.dropna().to_numpy(), 0.0, atol=1e-4)


def test_volume_residual_recovers_known_shock_at_last_bar():
    shock = 0.5
    data = _make_exact_stock(n=200, shock_bar=199, shock=shock)
    res = sf.volume_residual(
        data["turnover"], data["abs_excess_ret_1"], data["intraday_range"],
        data["market_turnover_z"], data["industry_turnover_z"], data["volatility_20"],
        data["stocks"], window=120, min_periods=80,
    )
    assert np.isclose(res.iloc[199], shock, atol=1e-4)
    # Earlier bars are unaffected by the last-bar shock (anti-leakage).
    assert np.allclose(res.iloc[120:199].dropna().to_numpy(), 0.0, atol=1e-4)


def test_volume_residual_anti_leakage_spike_does_not_reach_earlier_bars():
    # Spike at bar 150: the fit window for bar 149 is [29, 148] and must not see it.
    data = _make_exact_stock(n=200, shock_bar=150, shock=0.9)
    res = sf.volume_residual(
        data["turnover"], data["abs_excess_ret_1"], data["intraday_range"],
        data["market_turnover_z"], data["industry_turnover_z"], data["volatility_20"],
        data["stocks"], window=120, min_periods=80,
    )
    assert np.isclose(res.iloc[150], 0.9, atol=1e-4)
    # Bar 149's window ends at 148, before the spike.
    assert np.isclose(res.iloc[149], 0.0, atol=1e-4)


def test_scarcity_sign():
    data = _make_exact_stock(n=200, shock_bar=199, shock=-0.5)  # turnover below expected
    res = sf.volume_residual(
        data["turnover"], data["abs_excess_ret_1"], data["intraday_range"],
        data["market_turnover_z"], data["industry_turnover_z"], data["volatility_20"],
        data["stocks"], window=120, min_periods=80,
    )
    scarcity = sf.scarcity(res)
    # residual < 0 (turnover below expected)  ->  scarcity > 0 (supply squeeze).
    assert res.iloc[199] < 0
    assert scarcity.iloc[199] > 0
    assert np.isclose(scarcity.iloc[199], 0.5, atol=1e-4)


# --------------------------------------------------------------------------- #
# Microstructure
# --------------------------------------------------------------------------- #
def test_tick_noise_price_scaling():
    raw_close = _series([1.0, 20.0])
    vol = _series([0.01, 0.01])
    noise = sf.tick_noise(raw_close, vol, tick_size=0.01)
    # 1 yuan: tick_return = 0.01/1 = 1%   -> noise = 0.01/0.01 = 1.0
    # 20 yuan: tick_return = 0.01/20=0.05% -> noise = 0.0005/0.01 = 0.05  (20x smaller)
    assert np.isclose(noise.iloc[0], 1.0, atol=1e-6)
    assert np.isclose(noise.iloc[1], 0.05, atol=1e-6)
    assert np.isclose(noise.iloc[0] / noise.iloc[1], 20.0, atol=1e-6)


def test_tick_return_units():
    raw_close = _series([1.0, 20.0])
    tr = sf.tick_return(raw_close, tick_size=0.01)
    assert np.isclose(tr.iloc[0], 0.01, atol=1e-9)
    assert np.isclose(tr.iloc[1], 0.0005, atol=1e-9)


# --------------------------------------------------------------------------- #
# K-line
# --------------------------------------------------------------------------- #
def test_close_location_edges():
    raw_close = _series([10.0, 10.0, 9.0, 9.5])
    raw_low = _series([9.0, 10.0, 9.0, 9.0])
    raw_high = _series([10.0, 10.0, 10.0, 10.0])
    loc = sf.close_location(raw_close, raw_low, raw_high)
    assert np.isclose(loc.iloc[0], 1.0)        # close == high
    assert np.isclose(loc.iloc[1], 0.5)        # high == low (one-word board)
    assert np.isclose(loc.iloc[2], 0.0)        # close == low
    assert np.isclose(loc.iloc[3], 0.5)        # midpoint


def test_intraday_range():
    span = sf.intraday_range(_series([11.0]), _series([9.0]), _series([10.0]))
    assert np.isclose(span.iloc[0], 0.2, atol=1e-6)


def test_upper_shadow_ratio():
    ratio = sf.upper_shadow_ratio(
        _series([11.0]), _series([10.0]), _series([10.0]), _series([9.0])
    )
    assert np.isclose(ratio.iloc[0], 0.5, atol=1e-6)


def test_body_ratio():
    # open=9, close=11, high=11, low=9 -> body=2, span=2 -> 1.0
    ratio = sf.body_ratio(_series([9.0]), _series([11.0]), _series([11.0]), _series([9.0]))
    assert np.isclose(ratio.iloc[0], 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# Industry leave-one-out aggregate
# --------------------------------------------------------------------------- #
def test_industry_loo_mean_excludes_self():
    # 3 stocks in industry I0 on one date with values 1, 2, 3.
    value = _series([1.0, 2.0, 3.0])
    dates = _series([pd.Timestamp("2024-01-02")] * 3)
    industries = _series(["I0", "I0", "I0"])
    loo = sf._industry_loo_mean(value, dates, industries)
    assert np.isclose(loo.iloc[0], 2.5)  # (2+3)/2
    assert np.isclose(loo.iloc[1], 2.0)  # (1+3)/2
    assert np.isclose(loo.iloc[2], 1.5)  # (1+2)/2


def test_excess_returns_subtracts_industry_loo():
    # Two stocks in the same industry; identical close path => identical N-day return,
    # so the leave-one-out industry return equals the stock return and excess == 0.
    n = 6
    close = pd.Series(np.linspace(10, 11, n).tolist() * 2)
    stocks = pd.Series(["A"] * n + ["B"] * n)
    dates = pd.Series([pd.Timestamp("2024-01-02") + pd.Timedelta(days=i) for i in range(n)] * 2)
    industries = pd.Series(["I0"] * (2 * n))
    out = sf.excess_returns(close, stocks, dates, industries, windows=[3])
    excess = out[3]
    # Same path → same return → excess vs leave-one-out peer == 0.
    finite = excess.dropna()
    assert np.allclose(finite.to_numpy(), 0.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# Volatility ddof
# --------------------------------------------------------------------------- #
def test_volatility_ddof_controls_denominator():
    # 3 bars: log_return = [nan, +0.01, -0.01].  Rolling(2) over [0.01, -0.01]:
    #   ddof=1 (sample) = sqrt(0.0002/1) = 0.014142...
    #   ddof=0 (pop)    = sqrt(0.0002/2) = 0.01
    close = pd.Series([10.0, 10.1, 9.9])  # log returns ~ +0.00995, -0.00995
    stocks = pd.Series(["A", "A", "A"])
    lr = sf.log_returns(close, stocks)
    vol1 = sf.volatility(lr, stocks, window=2, ddof=1).iloc[2]
    vol0 = sf.volatility(lr, stocks, window=2, ddof=0).iloc[2]
    assert vol1 > vol0
    # sample std exceeds population std by sqrt(2) for a 2-observation window
    assert np.isclose(vol1 / vol0, np.sqrt(2.0), atol=1e-2)


# --------------------------------------------------------------------------- #
# Breadth
# --------------------------------------------------------------------------- #
def test_market_breadth_counts_up_fraction():
    # 6 stocks on one date: 3 up, 2 down, 1 NaN.  Breadth = 3 / 5 = 0.6.
    lr = _series([0.01, 0.02, 0.03, -0.01, -0.02, np.nan])
    dates = _series([pd.Timestamp("2024-01-02")] * 6)
    valid = _series([True] * 6)
    breadth = sf.market_breadth(lr, dates, valid)
    assert np.allclose(breadth.to_numpy(), 0.6, atol=1e-9)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_scarcity_days_ratio_in_unit_interval():
    rng = np.random.default_rng(3)
    n = 200
    res = pd.Series(rng.normal(0, 1, n), index=pd.RangeIndex(n))
    stocks = pd.Series(["A"] * n, index=pd.RangeIndex(n))
    ratio = sf.scarcity_days_ratio(res, stocks, window=5)
    finite = ratio.dropna()
    assert (finite >= 0).all() and (finite <= 1).all()
    # Known window: all-negative residual -> ratio == 1.
    res_neg = pd.Series([-0.1] * 10, index=pd.RangeIndex(10))
    stocks_neg = pd.Series(["A"] * 10, index=pd.RangeIndex(10))
    ratio_neg = sf.scarcity_days_ratio(res_neg, stocks_neg, window=5)
    assert np.isclose(ratio_neg.iloc[9], 1.0)


def test_scarcity_slope_detects_trend():
    # scarcity rising linearly 0,1,2,3,4 -> slope == 1.0
    res = pd.Series([0.0, -1.0, -2.0, -3.0, -4.0], index=pd.RangeIndex(5))  # scarcity = -res = 0,1,2,3,4
    stocks = pd.Series(["A"] * 5, index=pd.RangeIndex(5))
    slope = sf.scarcity_slope(res, stocks, window=5)
    assert np.isclose(slope.iloc[4], 1.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# Composite factors (document sec. 13) + sample weights (sec. 8.6 / 9.5 / 9.6)
# --------------------------------------------------------------------------- #
def test_price_weight_clips_and_inverts():
    # tick_noise = 0 -> full weight 1.0; huge tick_noise -> floored at 0.1
    pw = sf.price_weight(_series([0.0, 1.0, 1e6]))
    assert np.isclose(pw.iloc[0], 1.0, atol=1e-9)
    assert np.isclose(pw.iloc[1], 1.0 / (1 + 2.0 * 1.0), atol=1e-9)  # lambda default 2
    assert np.isclose(pw.iloc[2], 0.1, atol=1e-9)  # clipped


def test_liquidity_weight_linear_ramp():
    lw = sf.liquidity_weight(_series([0.0, 1.0, 2.0, 3.0, 4.0]), a_low=1.0, a_full=3.0)
    assert np.isclose(lw.iloc[0], 0.0)            # below A_low
    assert np.isclose(lw.iloc[1], 0.0)            # at A_low
    assert np.isclose(lw.iloc[2], 0.5)            # midpoint
    assert np.isclose(lw.iloc[3], 1.0)            # at A_full
    assert np.isclose(lw.iloc[4], 1.0)            # above A_full


def test_liquidity_weight_degenerate_thresholds_return_full_weight():
    lw = sf.liquidity_weight(_series([1.0, 2.0]), a_low=2.0, a_full=2.0)  # non-increasing
    assert np.allclose(lw.to_numpy(), 1.0)


def test_sample_weight_product_clipped():
    pw = _series([0.5, 0.1])
    liw = _series([0.5, 0.1])
    sw = sf.sample_weight(pw, liw)
    assert np.isclose(sw.iloc[0], 0.25, atol=1e-9)   # within [0.1, 1]
    assert np.isclose(sw.iloc[1], 0.1, atol=1e-9)    # 0.01 clipped up to 0.1


def test_conditional_scarcity_factor_only_positive_legs():
    rar5 = _series([2.0, -1.0, 2.0])
    vres = _series([-1.0, -1.0, 1.0])
    f = sf.conditional_scarcity_factor(rar5, vres)
    assert np.isclose(f.iloc[0], 2.0)    # max(2,0)*max(1,0) = 2
    assert np.isclose(f.iloc[1], 0.0)    # negative return leg -> 0
    assert np.isclose(f.iloc[2], 0.0)    # positive residual (no squeeze) -> 0


def test_close_quality_scarcity_factor():
    cond = _series([2.0])
    close_loc = _series([0.5])
    upper_shadow = _series([0.25])
    f = sf.close_quality_scarcity_factor(cond, close_loc, upper_shadow)
    assert np.isclose(f.iloc[0], 2.0 * 0.5 * (1 - 0.25), atol=1e-9)  # 0.75


def test_persistent_and_price_adjusted_composites():
    cond = _series([2.0])
    scar_days = _series([0.5])
    up_days = _series([0.4])
    persistent = sf.persistent_scarcity_factor(cond, scar_days, up_days)
    assert np.isclose(persistent.iloc[0], 2.0 * 0.5 * 0.4, atol=1e-9)  # 0.4
    pw = _series([0.5])
    pa = sf.price_adjusted_scarcity_factor(persistent, pw)
    assert np.isclose(pa.iloc[0], 0.4 * 0.5, atol=1e-9)  # 0.2


def test_simple_low_volume_rise_nonneg_and_positive_on_rise_plus_squeeze():
    # Increasing excess_ret_5 -> last bar above its rolling mean -> positive ts-zscore.
    # turnover z = -1 everywhere (squeeze) -> max(-tz, 0) = 1.
    n = 80
    excess = pd.Series(np.linspace(0.0, 0.02, n), index=pd.RangeIndex(n))
    tz60 = pd.Series([-1.0] * n, index=pd.RangeIndex(n))
    stocks = pd.Series(["A"] * n, index=pd.RangeIndex(n))
    out = sf.simple_low_volume_rise(excess, tz60, stocks, ret_z_window=60)
    assert (out.dropna() >= 0).all()                        # product of positive parts
    assert np.isfinite(out.iloc[-1]) and out.iloc[-1] > 0   # rising + squeeze -> positive

