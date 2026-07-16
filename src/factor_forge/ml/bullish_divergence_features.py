"""Point-in-time indicator primitives for bullish-divergence research."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import atr_reversion_features as af
from . import supply_features as sf


EPS = 1e-12


def wilder_rsi(close: pd.Series, stocks: pd.Series, window: int) -> pd.Series:
    """Causal Wilder-style RSI for a sorted long panel."""
    delta = close.groupby(stocks, sort=False).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    average_gain = gain.groupby(stocks, sort=False).transform(
        lambda values: values.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    )
    average_loss = loss.groupby(stocks, sort=False).transform(
        lambda values: values.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    )
    relative_strength = average_gain / average_loss.where(average_loss > EPS)
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    rsi = rsi.where(average_loss > EPS, 100.0)
    return rsi.where((average_gain + average_loss) > EPS, 50.0)


def macd_histogram(
    close: pd.Series,
    stocks: pd.Series,
    *,
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> pd.Series:
    fast = af.ema(close, stocks, fast_span)
    slow = af.ema(close, stocks, slow_span)
    macd = fast - slow
    signal = af.ema(macd, stocks, signal_span)
    return macd - signal


def trailing_amount_ratio(amount: pd.Series, stocks: pd.Series, window: int = 20) -> pd.Series:
    baseline = sf._rolling(
        amount.where(amount > 0), stocks, window, method="mean",
        min_periods=max(5, window // 2), shift=1,
    )
    return amount / baseline.where(baseline > 0)


def cross_section_percentile(value: pd.Series, dates: pd.Series) -> pd.Series:
    return value.groupby(dates, sort=False).rank(pct=True)


def interval_distance(level: float, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Unsigned distance from a price to candle intervals; zero means intersection."""
    return np.maximum(np.maximum(low - level, level - high), 0.0)

