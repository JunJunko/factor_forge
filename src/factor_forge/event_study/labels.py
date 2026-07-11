from __future__ import annotations

import numpy as np
import pandas as pd


def build_point_in_time_features_and_labels(
    panel: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """Build close-T matching covariates and T+1-open to T+h+1-open labels."""
    required = {
        "trade_date", "ts_code", "adj_open", "adj_close", "amount_cny",
        "log_total_mv", "industry_l1_code", "is_liquid",
    }
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"event-study panel missing columns: {sorted(missing)}")
    if panel.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("event-study panel has duplicate trade_date/ts_code rows")
    data = panel.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    close = pd.to_numeric(data["adj_close"], errors="coerce")
    amount = pd.to_numeric(data["amount_cny"], errors="coerce")
    grouped_close = close.groupby(data["ts_code"], sort=False)
    daily_return = grouped_close.pct_change(1, fill_method=None)
    data["prior_return_5d"] = grouped_close.pct_change(5, fill_method=None)
    data["volatility_20d"] = daily_return.groupby(data["ts_code"], sort=False).transform(
        lambda values: values.rolling(20, min_periods=10).std(ddof=0)
    )
    avg_amount = amount.groupby(data["ts_code"], sort=False).transform(
        lambda values: values.rolling(20, min_periods=10).mean()
    )
    data["log_avg_amount_20d"] = np.log(avg_amount.where(avg_amount > 0))
    opens = pd.to_numeric(data["adj_open"], errors="coerce")
    grouped_open = opens.groupby(data["ts_code"], sort=False)
    entry = grouped_open.shift(-1)
    for horizon in horizons:
        data[f"forward_return_{horizon}"] = grouped_open.shift(-(horizon + 1)) / entry - 1.0
        data[f"label_mature_{horizon}"] = (
            entry.notna() & grouped_open.shift(-(horizon + 1)).notna()
        )
    return data


def build_market_regimes(data: pd.DataFrame) -> pd.DataFrame:
    # Recompute one-day cross-sectional market return from close prices.
    ordered = data.sort_values(["ts_code", "trade_date"], kind="mergesort").copy()
    one_day = pd.to_numeric(ordered["adj_close"], errors="coerce").groupby(
        ordered["ts_code"], sort=False
    ).pct_change(1, fill_method=None)
    ordered["one_day_return"] = one_day
    market = (
        ordered.loc[ordered["is_liquid"].fillna(False).astype(bool)]
        .groupby("trade_date")["one_day_return"].mean().sort_index()
    )
    result = market.rename("market_daily_return").to_frame()
    result["market_return_20d"] = result["market_daily_return"].rolling(20, min_periods=10).sum()
    result["market_volatility_20d"] = result["market_daily_return"].rolling(20, min_periods=10).std(ddof=0)
    prior_median = result["market_volatility_20d"].shift(1).rolling(252, min_periods=60).median()
    result["market_direction"] = np.where(result["market_return_20d"].ge(0), "up", "down")
    result["market_volatility"] = np.where(
        result["market_volatility_20d"].ge(prior_median), "high", "low"
    )
    return result.reset_index()
