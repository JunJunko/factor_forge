from __future__ import annotations

import numpy as np
import pandas as pd


def _by_security(value: pd.Series):
    return value.groupby(level="ts_code", sort=False, group_keys=False)


def ret(value: pd.Series, periods: int) -> pd.Series:
    """Point return: x[t] / x[t-periods] - 1."""
    return value / _by_security(value).shift(int(periods)) - 1.0


def lag(value: pd.Series, periods: int) -> pd.Series:
    return _by_security(value).shift(int(periods))


def delta(value: pd.Series, periods: int) -> pd.Series:
    return value - lag(value, periods)


def _rolling(value: pd.Series, window: int, method: str) -> pd.Series:
    grouped = _by_security(value)
    rolling = grouped.rolling(int(window), min_periods=int(window))
    result = getattr(rolling, method)()
    return result.droplevel(0).reindex(value.index)


def ts_mean(value: pd.Series, window: int) -> pd.Series:
    return _rolling(value, window, "mean")


def ts_std(value: pd.Series, window: int) -> pd.Series:
    # pandas rolling std is sample std; V1 explicitly uses population std.
    grouped = _by_security(value)
    result = grouped.rolling(int(window), min_periods=int(window)).std(ddof=0)
    return result.droplevel(0).reindex(value.index)


def ts_max(value: pd.Series, window: int) -> pd.Series:
    return _rolling(value, window, "max")


def ts_min(value: pd.Series, window: int) -> pd.Series:
    return _rolling(value, window, "min")


def slope(value: pd.Series, window: int) -> pd.Series:
    """OLS coefficient on positions 0..window-1, requiring a full window."""
    window = int(window)
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))

    def coefficient(items: np.ndarray) -> float:
        if len(items) != window or not np.isfinite(items).all():
            return np.nan
        return float(np.dot(items - items.mean(), x_centered) / denominator)

    result = _by_security(value).rolling(window, min_periods=window).apply(
        coefficient, raw=True
    )
    return result.droplevel(0).reindex(value.index)


def _cross_section_keys(value: pd.Series, by: pd.Series | None) -> list[pd.Series]:
    dates = pd.Series(value.index.get_level_values("trade_date"), index=value.index)
    return [dates] if by is None else [dates, by]


def _mask_small_groups(
    result: pd.Series, value: pd.Series, by: pd.Series | None, min_group_size: int
) -> pd.Series:
    if by is None:
        return result
    counts = value.notna().groupby(_cross_section_keys(value, by), dropna=False).transform("sum")
    return result.where((counts >= min_group_size) & by.notna())


def cs_rank(
    value: pd.Series, by: pd.Series | None = None, min_group_size: int = 2
) -> pd.Series:
    keys = _cross_section_keys(value, by)
    result = value.groupby(keys, dropna=False).rank(method="average", pct=True)
    return _mask_small_groups(result, value, by, min_group_size)


def cs_zscore(
    value: pd.Series, by: pd.Series | None = None, min_group_size: int = 2
) -> pd.Series:
    keys = _cross_section_keys(value, by)
    grouped = value.groupby(keys, dropna=False)
    mean = grouped.transform("mean")
    std = grouped.transform(lambda x: x.std(ddof=0))
    result = (value - mean) / std.replace(0, np.nan)
    return _mask_small_groups(result, value, by, min_group_size)


def group_mean(
    value: pd.Series, by: pd.Series, min_group_size: int = 2
) -> pd.Series:
    numeric = value.astype(float)
    keys = _cross_section_keys(numeric, by)
    result = numeric.groupby(keys, dropna=False).transform("mean")
    return _mask_small_groups(result, numeric, by, min_group_size)


def winsorize_mad(value: pd.Series, scale: float = 5.0) -> pd.Series:
    dates = value.index.get_level_values("trade_date")

    def clip_one(items: pd.Series) -> pd.Series:
        median = items.median()
        mad = (items - median).abs().median()
        if not np.isfinite(mad) or mad == 0:
            return items
        return items.clip(median - scale * mad, median + scale * mad)

    return value.groupby(dates, group_keys=False).apply(clip_one).reindex(value.index)


def standardize_zscore(value: pd.Series) -> pd.Series:
    return cs_zscore(value)

