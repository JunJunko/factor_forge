"""Pure feature primitives for ATR lower-shadow reversion research."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import supply_features as sf

EPS = 1e-12


def true_range(high: pd.Series, low: pd.Series, close: pd.Series, stocks: pd.Series) -> pd.Series:
    prev_close = close.groupby(stocks, sort=False).shift(1)
    arr = np.maximum.reduce(
        [
            (high - low).abs().to_numpy(dtype=float),
            (high - prev_close).abs().to_numpy(dtype=float),
            (low - prev_close).abs().to_numpy(dtype=float),
        ]
    )
    return pd.Series(arr, index=high.index)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, stocks: pd.Series, window: int) -> pd.Series:
    return sf._rolling(true_range(high, low, close, stocks), stocks, window, method="mean", min_periods=max(5, window // 2))


def rolling_mean(value: pd.Series, stocks: pd.Series, window: int) -> pd.Series:
    return sf._rolling(value, stocks, window, method="mean", min_periods=max(5, window // 2))


def rolling_percentile(value: pd.Series, stocks: pd.Series, window: int) -> pd.Series:
    """Per-stock rolling percentile rank of the current value inside the trailing window."""
    min_periods = min(window, max(2, min(20, window // 2)))

    def _last_pct(x: np.ndarray) -> float:
        last = x[-1]
        if not np.isfinite(last):
            return np.nan
        finite = x[np.isfinite(x)]
        if finite.size < min_periods:
            return np.nan
        return float((finite <= last).mean())

    out = (
        value.groupby(stocks, sort=False)
        .rolling(window, min_periods=min_periods)
        .apply(_last_pct, raw=True)
        .reset_index(level=0, drop=True)
    )
    return out.reindex(value.index)


def downside_deviation(close: pd.Series, ma: pd.Series, atr_value: pd.Series) -> pd.Series:
    return ((ma - close) / (atr_value + EPS)).clip(lower=0.0)


def lower_shadow(open_: pd.Series, close: pd.Series, low: pd.Series, atr_value: pd.Series) -> pd.Series:
    body_bottom = pd.Series(np.minimum(open_.to_numpy(), close.to_numpy()), index=open_.index)
    return ((body_bottom - low) / (atr_value + EPS)).clip(lower=0.0)


def intraday_repair(close: pd.Series, low: pd.Series, high: pd.Series) -> pd.Series:
    span = high - low
    return ((close - low) / (span + EPS)).where(span > EPS, 0.5).clip(0.0, 1.0)


def trend_state(close: pd.Series, ma20: pd.Series, ma60: pd.Series) -> pd.Series:
    return ma20 / ma60.where(ma60 > 0) - 1.0


def amount_shock(amount: pd.Series, stocks: pd.Series, window: int) -> pd.Series:
    base = sf._rolling(amount.where(amount > 0), stocks, window, method="mean", min_periods=max(5, window // 2))
    return np.log((amount.where(amount > 0) / base.where(base > 0)).replace([np.inf, -np.inf], np.nan))


def rolling_rank_by_date(value: pd.Series, dates: pd.Series) -> pd.Series:
    return value.groupby(dates, sort=False).rank(pct=True)


def core_signal(down_dev: pd.Series, lower_shadow_atr: pd.Series, repair: pd.Series) -> pd.Series:
    return down_dev.clip(0.0, 2.5) * lower_shadow_atr.clip(0.0, 1.5) * repair
