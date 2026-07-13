from __future__ import annotations

import numpy as np
import pandas as pd

from .event_factor_sensitivity_config import FACTOR_BASIS


def build_event_factor_basis(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the seven frozen, named, point-in-time event factor axes."""
    required = {
        "trade_date", "ts_code", "adj_open", "adj_high", "adj_low", "adj_close",
        "amount_cny", "turnover_rate", "industry_l1_code",
    }
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"event factor basis missing panel fields: {sorted(missing)}")
    data = panel[list(required)].copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data["ts_code"] = data["ts_code"].astype(str)
    data = data.sort_values(["ts_code", "trade_date"], kind="mergesort")
    grouped = data.groupby("ts_code", sort=False)
    close = pd.to_numeric(data["adj_close"], errors="coerce").where(lambda x: x > 0)
    ret_1 = grouped["adj_close"].pct_change(fill_method=None)
    ret_5 = grouped["adj_close"].pct_change(5, fill_method=None)
    ret_20 = grouped["adj_close"].pct_change(20, fill_method=None)
    vol_5 = ret_1.groupby(data["ts_code"]).rolling(5, min_periods=3).std(ddof=0).reset_index(level=0, drop=True)
    vol_20 = ret_1.groupby(data["ts_code"]).rolling(20, min_periods=10).std(ddof=0).reset_index(level=0, drop=True)
    amount = pd.to_numeric(data["amount_cny"], errors="coerce").where(lambda x: x > 0)
    amount_20 = amount.groupby(data["ts_code"]).rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    turnover = pd.to_numeric(data["turnover_rate"], errors="coerce").clip(lower=0)
    industry_mean = ret_5.groupby([data["trade_date"], data["industry_l1_code"]]).transform("mean")
    bar_range = (data["adj_high"] - data["adj_low"]).replace(0, np.nan)
    close_location = (data["adj_close"] - data["adj_low"]) / bar_range

    result = data[["trade_date", "ts_code"]].copy()
    result["short_reversal"] = -ret_5
    result["trend_acceleration"] = ret_5 - ret_20 / 4.0
    result["volume_price_efficiency"] = ret_5.abs() / np.log1p(amount_20).replace(0, np.nan)
    result["volatility_compression"] = -(vol_5 / vol_20.replace(0, np.nan) - 1.0)
    result["industry_relative_return"] = ret_5 - industry_mean
    result["liquidity_displacement"] = ret_5.abs() / np.log1p(turnover).replace(0, np.nan)
    result["intraday_rejection"] = close_location - 0.5
    result = result.replace([np.inf, -np.inf], np.nan)
    for name in FACTOR_BASIS:
        result[name] = _daily_robust_zscore(result[name], result["trade_date"])
    return result.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)


def _daily_robust_zscore(values: pd.Series, dates: pd.Series) -> pd.Series:
    grouped = values.groupby(dates)
    lower = grouped.transform("quantile", q=0.01)
    upper = grouped.transform("quantile", q=0.99)
    clipped = values.clip(lower, upper)
    mean = clipped.groupby(dates).transform("mean")
    std = clipped.groupby(dates).transform("std").replace(0, np.nan)
    return (clipped - mean) / std
