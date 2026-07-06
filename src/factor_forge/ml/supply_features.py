"""Feature primitives for the low-volume-rise supply-contraction factor (document sec. 4-13).

All functions are pure: they take aligned :class:`pandas.Series` (indexed like the long
panel, sorted by ``(ts_code, trade_date)``) plus ``stocks``/``dates``/``industries`` group
keys and return a Series on the same index.  They return *raw* values; the daily
1%/99% winsorize + cross-sectional z-score (document sec. 3.4) is applied once at the
dataset layer, except where the document bakes a clip into the definition
(e.g. ``risk_adjusted_ret_5`` is clipped to ``[-3, 3]``).

Price convention (document sec. 3.1/3.2): returns and volatility use **adjusted** prices;
tick / log_raw_price / K-line / gap features use **raw** prices.  ``turnover_rate`` is a
percent in the panel (``1.23`` means ``1.23%``) so it is divided by 100 wherever a
fraction is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Rolling / cross-sectional helpers
# --------------------------------------------------------------------------- #
def _rolling(
    value: pd.Series,
    stocks: pd.Series,
    window: int,
    *,
    method: str = "mean",
    min_periods: int | None = None,
    ddof: int = 1,
    shift: int = 0,
) -> pd.Series:
    """Per-stock rolling statistic.

    ``shift > 0`` shifts the source by ``shift`` bars *before* rolling so the window
    excludes the most recent bars (the anti-leakage knob used by ``volume_residual``).
    """
    min_periods = window if min_periods is None else max(2, min_periods)
    src = value.groupby(stocks, sort=False).shift(shift) if shift else value
    rolled = src.groupby(stocks, sort=False).rolling(window, min_periods=min_periods)
    if method == "std":
        out = rolled.std(ddof=ddof)
    elif method == "mean":
        out = rolled.mean()
    elif method == "sum":
        out = rolled.sum()
    elif method == "max":
        out = rolled.max()
    elif method == "min":
        out = rolled.min()
    else:
        raise ValueError(f"unknown rolling method: {method}")
    return out.reset_index(level=0, drop=True).reindex(value.index)


def _industry_loo_mean(value: pd.Series, dates: pd.Series, industries: pd.Series) -> pd.Series:
    """Daily within-industry equal-weight mean that excludes the stock itself."""
    keys = [dates, industries]
    count = value.notna().groupby(keys, sort=False, dropna=False).transform("sum")
    total = value.fillna(0.0).groupby(keys, sort=False, dropna=False).transform("sum")
    return (total - value) / (count - 1).replace(0, np.nan)


def _daily_cross_section_mean(value: pd.Series, dates: pd.Series) -> pd.Series:
    return value.groupby(dates, sort=False).transform("mean")


# --------------------------------------------------------------------------- #
# Returns & volatility (adjusted prices)
# --------------------------------------------------------------------------- #
def log_returns(adj_close: pd.Series, stocks: pd.Series) -> pd.Series:
    close = adj_close.where(adj_close > 0)
    return np.log(close).groupby(stocks, sort=False).diff()


def excess_returns(
    adj_close: pd.Series,
    stocks: pd.Series,
    dates: pd.Series,
    industries: pd.Series,
    windows: list[int],
) -> dict[int, pd.Series]:
    """``excess_ret_N = ret_stock_N - ret_industry_N`` (simple returns, leave-one-out).

    The industry N-day return is the leave-one-out equal-weight mean of members' N-day
    returns on the same date, so a stock never contributes to its own benchmark.
    """
    close = adj_close.where(adj_close > 0)
    out: dict[int, pd.Series] = {}
    for n in windows:
        ret_n = close.groupby(stocks, sort=False).pct_change(n, fill_method=None)
        industry_ret_n = _industry_loo_mean(ret_n, dates, industries)
        out[n] = ret_n - industry_ret_n
    return out


def volatility(log_return: pd.Series, stocks: pd.Series, window: int, ddof: int = 1) -> pd.Series:
    min_periods = max(2, int(round(window * 0.75)))
    return _rolling(log_return, stocks, window, method="std", min_periods=min_periods, ddof=ddof)


def risk_adjusted_ret_5(excess_ret_5: pd.Series, volatility_20: pd.Series) -> pd.Series:
    """``excess_ret_5 / (volatility_20 * sqrt(5) + eps)``, clipped to ``[-3, 3]``."""
    out = excess_ret_5 / (volatility_20 * np.sqrt(5.0) + EPS)
    return out.clip(-3.0, 3.0)


# --------------------------------------------------------------------------- #
# Activity (turnover / amount)
# --------------------------------------------------------------------------- #
def _log_turnover(turnover_rate_pct: pd.Series) -> pd.Series:
    return np.log1p(turnover_rate_pct / 100.0)


def log_turnover_zscore(
    turnover_rate_pct: pd.Series, stocks: pd.Series, window: int
) -> pd.Series:
    min_periods = max(5, int(round(window * 0.75)))
    lt = _log_turnover(turnover_rate_pct)
    mean = _rolling(lt, stocks, window, method="mean", min_periods=min_periods)
    std = _rolling(lt, stocks, window, method="std", min_periods=min_periods, ddof=0)
    return (lt - mean) / std.replace(0, np.nan)


def amount_zscore(amount_cny: pd.Series, stocks: pd.Series, window: int) -> pd.Series:
    min_periods = max(5, int(round(window * 0.75)))
    la = np.log1p(amount_cny.where(amount_cny > 0))
    mean = _rolling(la, stocks, window, method="mean", min_periods=min_periods)
    std = _rolling(la, stocks, window, method="std", min_periods=min_periods, ddof=0)
    return (la - mean) / std.replace(0, np.nan)


def log_avg_amount(amount_cny: pd.Series, stocks: pd.Series, window: int) -> pd.Series:
    min_periods = max(5, int(round(window * 0.75)))
    mean = _rolling(
        amount_cny.where(amount_cny > 0), stocks, window, method="mean", min_periods=min_periods
    )
    return np.log1p(mean)


# --------------------------------------------------------------------------- #
# Activity-regime regressors for ``volume_residual`` (PIT, document sec. 6.1)
# --------------------------------------------------------------------------- #
def market_turnover_z(
    turnover_rate_pct: pd.Series, dates: pd.Series, stocks: pd.Series, window: int = 60
) -> pd.Series:
    """Time-series z of the daily market equal-weight mean log turnover.

    A single market-activity-regime value per day (same for every stock), measuring
    whether the whole market is more active than its recent norm.
    """
    lt = _log_turnover(turnover_rate_pct)
    daily_mean = lt.groupby(dates, sort=False).mean()
    min_periods = max(5, int(round(window * 0.75)))
    rolling = daily_mean.rolling(window, min_periods=min_periods)
    z = (daily_mean - rolling.mean()) / rolling.std(ddof=0).replace(0, np.nan)
    return dates.map(z).astype(float).reindex(turnover_rate_pct.index)


def industry_turnover_z(
    turnover_rate_pct: pd.Series,
    dates: pd.Series,
    industries: pd.Series,
    stocks: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Time-series z of the daily within-industry mean log turnover (per industry)."""
    lt = _log_turnover(turnover_rate_pct)
    df = pd.DataFrame(
        {"date": dates.to_numpy(), "industry": industries.to_numpy(), "v": lt.to_numpy()},
        index=turnover_rate_pct.index,
    )
    ind_daily = (
        df.groupby(["date", "industry"], sort=False, dropna=False)["v"].mean().reset_index()
    )
    min_periods = max(5, int(round(window * 0.75)))

    def _z(s: pd.Series) -> pd.Series:
        r = s.rolling(window, min_periods=min_periods)
        return (s - r.mean()) / r.std(ddof=0).replace(0, np.nan)

    ind_daily["z"] = ind_daily.groupby("industry", sort=False)["v"].transform(_z)
    merged = df.reset_index().merge(ind_daily[["date", "industry", "z"]], on=["date", "industry"], how="left")
    return pd.Series(merged["z"].to_numpy(), index=turnover_rate_pct.index, dtype=float)


# --------------------------------------------------------------------------- #
# Conditional volume residual (the signature feature, document sec. 6)
# --------------------------------------------------------------------------- #
def _rolling_ols_residual(
    y: np.ndarray,
    design: np.ndarray,
    stock_ids: np.ndarray,
    unique_stocks: np.ndarray,
    window: int,
    min_periods: int,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Per-stock rolling OLS residual with anti-leakage.

    For each bar ``t`` the coefficients are fit on the design rows ``[t-window, t-1]``
    (never including bar ``t``) and used to predict ``y[t]``; the residual is
    ``y[t] - prediction``.  Windows with fewer than ``min_periods`` finite rows yield NaN.
    """
    residual = np.full(y.shape, np.nan, dtype=float)
    k = design.shape[1]
    eye = np.eye(k)
    for stock in unique_stocks:
        idx = np.nonzero(stock_ids == stock)[0]
        if idx.size < min_periods + 1:
            continue
        ys = y[idx]
        Xs = design[idx]
        # Shift by one bar so a rolling window ending at t covers original [t-window, t-1].
        y_shift = np.full_like(ys, np.nan)
        y_shift[1:] = ys[:-1]
        X_shift = np.full_like(Xs, np.nan)
        X_shift[1:] = Xs[:-1]
        finite = np.isfinite(X_shift).all(axis=1) & np.isfinite(y_shift)
        # Rolling sums of outer products via cumulative sums (O(L) per stock).
        Xc = np.where(finite[:, None], X_shift, 0.0)
        yc = np.where(finite, y_shift, 0.0)
        outer = Xc[:, :, None] * Xc[:, None, :]  # (L, k, k)
        cum_xtx = np.cumsum(outer, axis=0)
        cum_xty = np.cumsum(Xc * yc[:, None], axis=0)
        cum_count = np.cumsum(finite.astype(np.float64))
        L = len(idx)
        # Windowed sums over the last `window` bars: sum[t] = cum[t] - cum[t-window].
        sum_xtx = cum_xtx.copy()
        sum_xty = cum_xty.copy()
        sum_count = cum_count.copy()
        if L > window:
            sum_xtx[window:] -= cum_xtx[:-window]
            sum_xty[window:] -= cum_xty[:-window]
            sum_count[window:] -= cum_count[:-window]
        # Batched solve for all bars at once (leading dim = L).  Ridge keeps degenerate
        # windows solvable; invalid windows are masked out via ``sum_count`` afterward.
        A = sum_xtx + ridge * eye  # (L, k, k)
        with np.errstate(all="ignore"):
            betas = np.linalg.solve(A, sum_xty[..., None])[..., 0]  # (L, k)
        pred = (Xs * betas).sum(axis=1)
        cand = ys - pred
        cur_finite = np.isfinite(Xs).all(axis=1) & np.isfinite(ys)
        valid = (sum_count >= min_periods) & cur_finite & (np.arange(L) >= 1)
        residual[idx] = np.where(valid, cand, np.nan)
    return residual


def volume_residual(
    turnover_rate_pct: pd.Series,
    abs_excess_ret_1: pd.Series,
    intraday_range: pd.Series,
    market_turnover_z_series: pd.Series,
    industry_turnover_z_series: pd.Series,
    volatility_20: pd.Series,
    stocks: pd.Series,
    window: int = 120,
    min_periods: int = 80,
) -> pd.Series:
    """Conditional turnover residual: actual log turnover minus OLS-predicted, fit on t-1.."""
    y = _log_turnover(turnover_rate_pct).to_numpy(dtype=float)
    design = np.column_stack(
        [
            np.ones(len(y)),
            abs_excess_ret_1.to_numpy(dtype=float),
            intraday_range.to_numpy(dtype=float),
            market_turnover_z_series.to_numpy(dtype=float),
            industry_turnover_z_series.to_numpy(dtype=float),
            volatility_20.to_numpy(dtype=float),
        ]
    )
    stock_ids = stocks.to_numpy()
    unique_stocks = pd.unique(stock_ids)
    res = _rolling_ols_residual(y, design, stock_ids, unique_stocks, window, min_periods)
    return pd.Series(res, index=turnover_rate_pct.index)


def scarcity(volume_residual_series: pd.Series) -> pd.Series:
    """``scarcity = -volume_residual`` (larger = stronger supply contraction)."""
    return -volume_residual_series


def volume_residual_mean(
    volume_residual_series: pd.Series, stocks: pd.Series, window: int
) -> pd.Series:
    return _rolling(volume_residual_series, stocks, window, method="mean", min_periods=window)


def scarcity_days_ratio(
    volume_residual_series: pd.Series, stocks: pd.Series, window: int
) -> pd.Series:
    """Fraction of the last ``window`` bars with ``volume_residual < 0``."""
    flag = (volume_residual_series < 0).astype(float)
    return _rolling(flag, stocks, window, method="mean", min_periods=window)


def scarcity_slope(volume_residual_series: pd.Series, stocks: pd.Series, window: int = 5) -> pd.Series:
    """OLS slope of ``scarcity`` (== -residual) over the last ``window`` bars, per stock."""
    scarcity_series = -volume_residual_series
    out = scarcity_series.copy().astype(float)
    out[:] = np.nan
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denom = (x_centered ** 2).sum()
    stock_ids = stocks.to_numpy()
    arr = scarcity_series.to_numpy(dtype=float)
    for stock in pd.unique(stock_ids):
        idx = np.nonzero(stock_ids == stock)[0]
        if idx.size < window:
            continue
        view = np.lib.stride_tricks.sliding_window_view(arr[idx], window)
        finite = np.isfinite(view).all(axis=1)
        slopes = np.full(view.shape[0], np.nan)
        valid = np.where(finite)[0]
        if valid.size:
            y = view[valid]
            slopes[valid] = ((y - y.mean(axis=1, keepdims=True)) * x_centered).sum(axis=1) / denom
        out.iloc[idx[window - 1 :]] = slopes
    return pd.Series(out, index=scarcity_series.index)


# --------------------------------------------------------------------------- #
# Microstructure (raw prices, document sec. 8)
# --------------------------------------------------------------------------- #
def log_raw_price(raw_close: pd.Series) -> pd.Series:
    return np.log(raw_close.where(raw_close > 0))


def tick_return(raw_close: pd.Series, tick_size: float = 0.01) -> pd.Series:
    return tick_size / raw_close.where(raw_close > 0)


def tick_noise(raw_close: pd.Series, volatility_20: pd.Series, tick_size: float = 0.01) -> pd.Series:
    return tick_return(raw_close, tick_size) / (volatility_20 + EPS)


def effective_ticks(
    raw_close: pd.Series, stocks: pd.Series, window: int, tick_size: float = 0.01
) -> pd.Series:
    delta = raw_close.groupby(stocks, sort=False).diff(window)
    return delta / tick_size


# --------------------------------------------------------------------------- #
# K-line structure (raw prices, document sec. 10)
# --------------------------------------------------------------------------- #
def close_location(raw_close: pd.Series, raw_low: pd.Series, raw_high: pd.Series) -> pd.Series:
    span = raw_high - raw_low
    loc = (raw_close - raw_low) / (span + EPS)
    return loc.where(span > EPS, 0.5)  # one-word board (high == low) → 0.5


def upper_shadow_ratio(
    raw_high: pd.Series, raw_open: pd.Series, raw_close: pd.Series, raw_low: pd.Series
) -> pd.Series:
    body_top = np.maximum(raw_open, raw_close)
    return (raw_high - body_top) / ((raw_high - raw_low) + EPS)


def body_ratio(raw_open: pd.Series, raw_close: pd.Series, raw_high: pd.Series, raw_low: pd.Series) -> pd.Series:
    return (raw_close - raw_open).abs() / ((raw_high - raw_low) + EPS)


def intraday_range(raw_high: pd.Series, raw_low: pd.Series, pre_close: pd.Series) -> pd.Series:
    return (raw_high - raw_low) / (pre_close.where(pre_close > 0) + EPS)


def gap_return(raw_open: pd.Series, pre_close: pd.Series, stocks: pd.Series) -> pd.Series:
    prev_close = pre_close.groupby(stocks, sort=False).shift(1)
    return raw_open / prev_close.where(prev_close > 0) - 1


# --------------------------------------------------------------------------- #
# Liquidity & size (document sec. 9)
# --------------------------------------------------------------------------- #
def log_float_market_cap(circ_mv_cny: pd.Series) -> pd.Series:
    return np.log1p(circ_mv_cny.where(circ_mv_cny > 0))


def amihud_illiquidity(
    abs_log_return: pd.Series, amount_cny: pd.Series, stocks: pd.Series, window: int, scale: float = 1e8
) -> pd.Series:
    ratio = abs_log_return / (amount_cny.where(amount_cny > 0) + EPS)
    return _rolling(ratio, stocks, window, method="mean", min_periods=max(5, window // 2)) * scale


def zero_return_days(log_return: pd.Series, stocks: pd.Series, window: int) -> pd.Series:
    flag = (log_return.abs() < 1e-8).astype(float)
    return _rolling(flag, stocks, window, method="sum", min_periods=window)


# --------------------------------------------------------------------------- #
# Price impact & breadth (document sec. 7, 12)
# --------------------------------------------------------------------------- #
def price_impact_1(excess_ret_1: pd.Series, turnover_rate_pct: pd.Series, denom_floor: float) -> pd.Series:
    denom = (turnover_rate_pct / 100.0).clip(lower=denom_floor)
    return excess_ret_1 / denom


def price_impact_5(
    excess_ret_5: pd.Series, turnover_rate_pct: pd.Series, stocks: pd.Series, denom_floor: float
) -> pd.Series:
    turnover_frac = (turnover_rate_pct / 100.0).groupby(stocks, sort=False).rolling(5, min_periods=5).sum()
    turnover_frac = turnover_frac.reset_index(level=0, drop=True).reindex(turnover_rate_pct.index)
    return excess_ret_5 / turnover_frac.clip(lower=denom_floor)


def up_days_ratio(excess_ret_1: pd.Series, stocks: pd.Series, window: int) -> pd.Series:
    flag = (excess_ret_1 > 0).astype(float)
    return _rolling(flag, stocks, window, method="mean", min_periods=window)


def market_breadth(log_return: pd.Series, dates: pd.Series, valid_mask: pd.Series) -> pd.Series:
    """Fraction of valid stocks with positive daily log return (broadcast to all rows)."""
    eligible_return = log_return.where(valid_mask)
    up = (eligible_return > 0).groupby(dates, sort=False).sum()
    count = eligible_return.notna().groupby(dates, sort=False).sum()
    breadth = (up / count.replace(0, np.nan)).fillna(0.0)
    return dates.map(breadth).astype(float).reindex(log_return.index)


def industry_breadth(
    log_return: pd.Series, dates: pd.Series, industries: pd.Series, valid_mask: pd.Series
) -> pd.Series:
    eligible = log_return.where(valid_mask)
    up = (eligible > 0).groupby([dates, industries], sort=False, dropna=False).transform("sum")
    count = eligible.notna().groupby([dates, industries], sort=False, dropna=False).transform("sum")
    return (up / count.replace(0, np.nan)).fillna(0.0)


# --------------------------------------------------------------------------- #
# Composite factors (document sec. 13) -- interaction terms built from the raw
# component primitives.  Trees can learn these interactions, but the document
# recommends keeping the composites AND their components (sec. 13.5).
# --------------------------------------------------------------------------- #
def ts_zscore(value: pd.Series, stocks: pd.Series, window: int) -> pd.Series:
    """Per-stock rolling z-score (time-series, not cross-sectional)."""
    min_periods = max(5, window // 2)
    mean = _rolling(value, stocks, window, method="mean", min_periods=min_periods)
    std = _rolling(value, stocks, window, method="std", min_periods=min_periods, ddof=0)
    return (value - mean) / std.replace(0, np.nan)


def simple_low_volume_rise(
    excess_ret_5: pd.Series, turnover_zscore_60: pd.Series, stocks: pd.Series, ret_z_window: int = 60
) -> pd.Series:
    """``max(ret_z, 0) * max(-turnover_z, 0)`` where ``ret_z`` is the 60d ts-zscore of excess_ret_5."""
    ret_z = ts_zscore(excess_ret_5, stocks, ret_z_window)
    return ret_z.clip(lower=0.0) * (-turnover_zscore_60).clip(lower=0.0)


def conditional_scarcity_factor(
    risk_adjusted_ret_5: pd.Series, volume_residual: pd.Series
) -> pd.Series:
    """``max(risk_adjusted_ret_5, 0) * max(-volume_residual, 0)``."""
    return risk_adjusted_ret_5.clip(lower=0.0) * (-volume_residual).clip(lower=0.0)


def close_quality_scarcity_factor(
    conditional: pd.Series, close_location: pd.Series, upper_shadow_ratio: pd.Series
) -> pd.Series:
    """``conditional * close_location * (1 - upper_shadow_ratio)``."""
    return conditional * close_location * (1.0 - upper_shadow_ratio)


def persistent_scarcity_factor(
    conditional: pd.Series, scarcity_days_ratio_5: pd.Series, up_days_ratio_5: pd.Series
) -> pd.Series:
    """``conditional * scarcity_days_ratio_5 * up_days_ratio_5``."""
    return conditional * scarcity_days_ratio_5 * up_days_ratio_5


def price_adjusted_scarcity_factor(persistent: pd.Series, price_weight_series: pd.Series) -> pd.Series:
    """``persistent * price_weight``."""
    return persistent * price_weight_series


# --------------------------------------------------------------------------- #
# Sample weights (document sec. 8.6, 9.5, 9.6) -- training weights, NOT model
# inputs.  Down-weight low-price tick-noise samples and illiquid samples.
# --------------------------------------------------------------------------- #
def price_weight(tick_noise_series: pd.Series, lam: float = 2.0) -> pd.Series:
    """``1 / (1 + lam * tick_noise)``, clipped to ``[0.1, 1]``."""
    return (1.0 / (1.0 + lam * tick_noise_series)).clip(0.1, 1.0)


def liquidity_weight(log_avg_amount: pd.Series, a_low: float, a_full: float) -> pd.Series:
    """Linear ramp of ``log(avg_amount_20)`` from ``a_low`` (0 weight) to ``a_full`` (1 weight).

    ``a_low`` / ``a_full`` must be estimated on the TRAINING segment only (document sec. 9.5).
    A degenerate (non-increasing) threshold returns full weight everywhere.
    """
    denom = a_full - a_low
    if not np.isfinite(denom) or denom <= 0:
        return pd.Series(1.0, index=log_avg_amount.index)
    return ((log_avg_amount - a_low) / denom).clip(0.0, 1.0)


def sample_weight(price_weight_series: pd.Series, liquidity_weight_series: pd.Series) -> pd.Series:
    """``clip(price_weight * liquidity_weight, 0.1, 1)``."""
    return (price_weight_series * liquidity_weight_series).clip(0.1, 1.0)


# --------------------------------------------------------------------------- #
# V2: stable turnover baseline + recent no-volume rise (handoff doc sec. 3).
# Baseline window = t-29..t-2 (``baseline_window`` bars), event window = t-1..t
# (``event_window`` bars).  The event days NEVER enter the baseline mean/std
# (handoff 4.1.3.3).  These return raw per-stock values; the daily cross-section
# percentile rank, std_floor, winsor and zscore are applied at the dataset layer.
# --------------------------------------------------------------------------- #
def baseline_window_stat(
    value: pd.Series,
    stocks: pd.Series,
    baseline_window: int,
    event_window: int,
    *,
    method: str = "mean",
    ddof: int = 0,
) -> pd.Series:
    """Rolling stat over the baseline window ``[t - bw - ew + 1, t - ew]``.

    For ``baseline_window=28, event_window=2`` the window at bar ``t`` is ``t-29..t-2``
    (28 bars), leaving ``t-1, t`` out of the baseline.  Implemented as
    ``rolling(baseline_window)`` on the series shifted by ``event_window``.  ``ddof=0``
    for std matches the project ``ts_std`` population-std convention.
    """
    min_periods = max(2, int(round(baseline_window * 0.75)))
    return _rolling(
        value, stocks, baseline_window, method=method, min_periods=min_periods,
        ddof=ddof, shift=event_window,
    )


def volatility_prior(
    log_return: pd.Series, stocks: pd.Series, window: int, gap: int, ddof: int = 1
) -> pd.Series:
    """Volatility whose window ends ``gap`` bars before the current bar.

    ``price_strength_2`` divides by ``volatility_20`` measured up to ``t-2`` so the last
    two event-day returns do not contaminate the volatility baseline (handoff 3.6).
    """
    min_periods = max(2, int(round(window * 0.75)))
    return _rolling(log_return, stocks, window, method="std", min_periods=min_periods, ddof=ddof, shift=gap)


def price_strength_2(excess_ret_2: pd.Series, volatility_prior_20: pd.Series) -> pd.Series:
    """``excess_ret_2 / (vol20_[..t-2] * sqrt(2) + eps)`` (handoff 3.6)."""
    return excess_ret_2 / (volatility_prior_20 * np.sqrt(2.0) + EPS)


def recent_volume_z(
    log_turnover: pd.Series,
    baseline_mean: pd.Series,
    baseline_std: pd.Series,
    std_floor: pd.Series,
    stocks: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Raw z of the last two event-day log turnovers vs the baseline mean/std.

    ``z[t-1], z[t] = (x - baseline_mean_28) / max(baseline_std_28, std_floor)`` (handoff 3.7).
    ``std_floor`` is the daily cross-section floor (computed at the dataset layer) that
    prevents division by ~0 when a stock's 28-day turnover is near-constant.  Returns
    ``(z_t1_raw, z_t_raw)``; clipping is the caller's job (handoff 3.8 keeps raw + clip).
    """
    denom_arr = np.maximum(baseline_std.to_numpy(), std_floor.to_numpy())
    denom = pd.Series(denom_arr, index=baseline_std.index).replace(0.0, np.nan)
    x_prev = log_turnover.groupby(stocks, sort=False).shift(1)
    z_t1 = (x_prev - baseline_mean) / denom
    z_t = (log_turnover - baseline_mean) / denom
    return z_t1, z_t


def recent_volume_z_aggregates(
    z_t1_raw: pd.Series, z_t_raw: pd.Series, clip_lower: float, clip_upper: float
) -> dict[str, pd.Series]:
    """Build ``mean_2`` / ``max_2`` in raw and clipped form (handoff 3.7/3.8).

    ``max_2`` takes priority over ``mean_2`` in the thesis (handoff 3.7/5.5): a single
    heavy-activation day should not be masked by averaging it with a quiet day.
    """
    mean_2_raw = (z_t1_raw + z_t_raw) / 2.0
    max_2_raw = pd.Series(
        np.maximum(z_t1_raw.to_numpy(), z_t_raw.to_numpy()), index=z_t1_raw.index
    )
    return {
        "recent_volume_z_mean_2_raw": mean_2_raw,
        "recent_volume_z_mean_2_clip": mean_2_raw.clip(clip_lower, clip_upper),
        "recent_volume_z_max_2_raw": max_2_raw,
        "recent_volume_z_max_2_clip": max_2_raw.clip(clip_lower, clip_upper),
    }
