from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig, LabelConfig


BASE_FEATURES = ["log_total_mv", "log_circ_mv", "turnover_rate"]


def build_dataset(panel: pd.DataFrame, features: FeatureConfig, label: LabelConfig) -> tuple[pd.DataFrame, list[str]]:
    """Build point-in-time features and a future label in Qlib's datetime/instrument shape."""
    required = {
        "trade_date", "ts_code", "adj_open", "adj_close", "amount_cny",
        "is_tradeable", "is_liquid", *BASE_FEATURES,
    }
    missing = {"trade_date", "ts_code", "adj_open", "adj_close", "amount_cny"} - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing ML fields: {sorted(missing)}")
    # The curated panel is deliberately wide. Keeping only ML columns prevents
    # a full-market run from duplicating several gigabytes of execution fields.
    data = panel[[c for c in panel.columns if c in required]].copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data.sort_values(["ts_code", "trade_date"])
    grouped = data.groupby("ts_code", sort=False)
    names: list[str] = []
    close = data["adj_close"].where(data["adj_close"] > 0)
    amount = data["amount_cny"].where(data["amount_cny"] > 0)
    for window in sorted(set(features.windows)):
        minp = max(2, window // 2)
        name = f"ret_{window}d"
        data[name] = grouped["adj_close"].pct_change(window, fill_method=None)
        names.append(name)
        name = f"vol_{window}d"
        daily_ret = grouped["adj_close"].pct_change(fill_method=None)
        data[name] = daily_ret.groupby(data["ts_code"]).rolling(window, min_periods=minp).std().reset_index(level=0, drop=True)
        names.append(name)
        name = f"amount_mean_{window}d"
        data[name] = amount.groupby(data["ts_code"]).rolling(window, min_periods=minp).mean().reset_index(level=0, drop=True)
        data[name] = np.log1p(data[name])
        names.append(name)
        name = f"price_ma_ratio_{window}d"
        ma = close.groupby(data["ts_code"]).rolling(window, min_periods=minp).mean().reset_index(level=0, drop=True)
        data[name] = close / ma - 1
        names.append(name)
    names.extend([c for c in BASE_FEATURES if c in data])
    # Signal is formed after T close. Execution buys at T+1 open and the
    # existing backtester exits after ``horizon`` complete trading days, at
    # T+(horizon+1) open.
    future = grouped[label.price].shift(-(label.horizon + 1)) / grouped[label.price].shift(-1) - 1
    if label.excess_over_universe:
        eligible = data.get("is_tradeable", pd.Series(True, index=data.index)).eq(True)
        daily_mean = future.where(eligible).groupby(data["trade_date"]).transform("mean")
        future = future - daily_mean
    data["label"] = future
    data = data.replace([np.inf, -np.inf], np.nan)
    if features.winsor_quantile:
        q = features.winsor_quantile
        for column in names:
            group = data.groupby("trade_date")[column]
            lower = group.transform("quantile", q=q)
            upper = group.transform("quantile", q=1 - q)
            data[column] = data[column].clip(lower, upper)
    if features.cross_sectional_zscore:
        for column in names:
            group = data.groupby("trade_date")[column]
            mean, std = group.transform("mean"), group.transform("std")
            data[column] = (data[column] - mean) / std.replace(0, np.nan)
    data = data.rename(columns={"trade_date": "datetime", "ts_code": "instrument"})
    return data, names


def to_qlib_frame(dataset: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Return the canonical Qlib MultiIndex/column-group representation."""
    frame = dataset.set_index(["datetime", "instrument"])[feature_names + ["label"]].sort_index()
    frame.columns = pd.MultiIndex.from_tuples(
        [("feature", c) for c in feature_names] + [("label", "LABEL0")]
    )
    return frame
