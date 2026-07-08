from __future__ import annotations

import math

import numpy as np
import pandas as pd


def to_datetime_series(value: pd.Series) -> pd.Series:
    """Parse Tushare-style YYYYMMDD strings and normal dates into pandas timestamps."""
    text = value.astype("string")
    compact = text.str.fullmatch(r"\d{8}").fillna(False)
    monthly = text.str.fullmatch(r"\d{6}").fillna(False)
    parsed = pd.Series(pd.NaT, index=value.index, dtype="datetime64[ns]")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    if monthly.any():
        parsed.loc[monthly] = pd.to_datetime(text.loc[monthly], format="%Y%m", errors="coerce")
    remaining = ~(compact | monthly)
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(text.loc[remaining], errors="coerce")
    return parsed


def clean_numeric(value: pd.Series) -> pd.Series:
    return pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan)


def first_existing(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    lower_map = {str(column).lower(): column for column in frame.columns}
    for column in candidates:
        match = lower_map.get(column.lower())
        if match is not None:
            return str(match)
    return None


def rolling_mad_zscore(
    value: pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
    clip: float | None = 3.0,
) -> pd.Series:
    min_periods = min(window, min_periods or max(20, window // 4))
    median = value.rolling(window, min_periods=min_periods).median()
    mad = (value - median).abs().rolling(window, min_periods=min_periods).median()
    z = 0.67448975 * (value - median) / mad.replace(0, np.nan)
    if clip is not None:
        z = z.clip(-clip, clip)
    return z


def rolling_percentile(
    value: pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
    clip: tuple[float, float] | None = (0.01, 0.99),
) -> pd.Series:
    min_periods = min(window, min_periods or max(20, window // 4))

    def pct_rank(items: np.ndarray) -> float:
        current = items[-1]
        finite = items[np.isfinite(items)]
        if not np.isfinite(current) or len(finite) < min_periods:
            return np.nan
        return float((finite <= current).sum() / len(finite))

    result = value.rolling(window, min_periods=min_periods).apply(pct_rank, raw=True)
    if clip is not None:
        result = result.clip(clip[0], clip[1])
    return result


def add_rolling_normalizations(
    data: pd.DataFrame,
    column: str,
    *,
    z_window: int,
    pct_window: int,
    z_clip: float = 3.0,
    pct_clip: tuple[float, float] = (0.01, 0.99),
) -> list[str]:
    names: list[str] = []
    z_name = f"{column}_z_{z_window}"
    pct_name = f"{column}_pct_{pct_window}"
    data[z_name] = rolling_mad_zscore(data[column], z_window, clip=z_clip)
    data[pct_name] = rolling_percentile(data[column], pct_window, clip=pct_clip)
    names.extend([z_name, pct_name])
    return names


def add_changes(data: pd.DataFrame, column: str, windows: tuple[int, ...]) -> list[str]:
    names: list[str] = []
    for window in windows:
        name = f"{column}_chg_{window}d"
        data[name] = data[column] - data[column].shift(window)
        names.append(name)
    return names


def add_extreme_flags(
    data: pd.DataFrame,
    pct_column: str,
    *,
    prefix: str,
    low: tuple[float, ...] = (0.05, 0.10),
    high: tuple[float, ...] = (0.90, 0.95),
) -> list[str]:
    names: list[str] = []
    for threshold in low:
        name = f"{prefix}_low_{int(threshold * 100)}"
        data[name] = data[pct_column].lt(threshold).astype(float).where(data[pct_column].notna())
        names.append(name)
    for threshold in high:
        name = f"{prefix}_high_{int(threshold * 100)}"
        data[name] = data[pct_column].gt(threshold).astype(float).where(data[pct_column].notna())
        names.append(name)
    return names


def lag_non_date_columns(data: pd.DataFrame, lag: int) -> pd.DataFrame:
    if lag <= 0 or data.empty:
        return data
    result = data.sort_values("trade_date").copy()
    columns = [column for column in result.columns if column != "trade_date"]
    result[columns] = result[columns].shift(lag)
    return result


def norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def black_scholes_price(
    *,
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    volatility: float,
    option_type: str,
) -> float:
    if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or volatility <= 0:
        return np.nan
    sigma_sqrt_t = volatility * math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility * volatility) * time_to_expiry) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    discount = math.exp(-rate * time_to_expiry)
    normalized = option_type.lower()
    if normalized in {"c", "call", "认购"}:
        return spot * norm_cdf(d1) - strike * discount * norm_cdf(d2)
    if normalized in {"p", "put", "认沽"}:
        return strike * discount * norm_cdf(-d2) - spot * norm_cdf(-d1)
    return np.nan


def implied_volatility_bisection(
    *,
    price: float,
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    option_type: str,
    low: float = 0.01,
    high: float = 1.50,
    tolerance: float = 1e-5,
    max_iter: int = 80,
) -> float:
    if not all(np.isfinite(item) for item in [price, spot, strike, rate, time_to_expiry]):
        return np.nan
    if price <= 0 or spot <= 0 or strike <= 0 or time_to_expiry <= 0:
        return np.nan
    low_price = black_scholes_price(
        spot=spot, strike=strike, rate=rate, time_to_expiry=time_to_expiry,
        volatility=low, option_type=option_type,
    )
    high_price = black_scholes_price(
        spot=spot, strike=strike, rate=rate, time_to_expiry=time_to_expiry,
        volatility=high, option_type=option_type,
    )
    if not np.isfinite(low_price) or not np.isfinite(high_price):
        return np.nan
    if price < low_price or price > high_price:
        return np.nan
    left, right = low, high
    for _ in range(max_iter):
        middle = (left + right) / 2
        model = black_scholes_price(
            spot=spot, strike=strike, rate=rate, time_to_expiry=time_to_expiry,
            volatility=middle, option_type=option_type,
        )
        if not np.isfinite(model):
            return np.nan
        if abs(model - price) < tolerance:
            return float(middle)
        if model < price:
            left = middle
        else:
            right = middle
    return float((left + right) / 2)
