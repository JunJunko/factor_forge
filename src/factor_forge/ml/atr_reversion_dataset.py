"""Dataset assembly for ATR lower-shadow reversion experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import atr_reversion_features as af
from . import supply_features as sf
from .atr_reversion_config import ATRReversionFeatureConfig, ATRReversionLabelConfig


CORE_FEATURES = [
    "down_deviation_atr",
    "down_deviation_pct",
    "lower_shadow_atr",
    "lower_shadow_pct",
    "intraday_repair",
    "core_signal",
]
STATE_FEATURES = [
    "trend_state",
    "vol_state",
    "vol_state_pct",
    "amount_shock",
    "liquidity_log_amount_20",
    "limit_flag",
    "near_down_limit_flag",
]
CONTEXT_FEATURES = [
    "market_ret_1d",
    "market_ret_5d",
    "industry_ret_1d",
    "industry_ret_5d",
    "stock_minus_industry_5d",
]
FEATURE_GROUPS = {
    "core": CORE_FEATURES,
    "state": STATE_FEATURES,
    "context": CONTEXT_FEATURES,
    "all": CORE_FEATURES + STATE_FEATURES + CONTEXT_FEATURES,
}
NO_CROSS_SECTION_ZSCORE = frozenset({
    "down_deviation_pct",
    "lower_shadow_pct",
    "vol_state_pct",
    "limit_flag",
    "near_down_limit_flag",
    "market_ret_1d",
    "market_ret_5d",
})

REQUIRED_PANEL_COLUMNS = {
    "trade_date", "ts_code",
    "raw_open", "raw_high", "raw_low", "raw_close", "pre_close",
    "adj_open", "adj_high", "adj_low", "adj_close",
    "amount_cny", "turnover_rate", "industry_l1_code",
    "is_st", "is_delisting_period", "is_suspended", "is_tradeable", "listing_trade_days",
}


def build_atr_reversion_dataset(
    panel: pd.DataFrame,
    features: ATRReversionFeatureConfig,
    label: ATRReversionLabelConfig,
) -> tuple[pd.DataFrame, list[str]]:
    missing = REQUIRED_PANEL_COLUMNS - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing ATR reversion fields: {sorted(missing)}")

    data = panel.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    stocks = data["ts_code"]
    dates = data["trade_date"]
    industries = data["industry_l1_code"]
    valid_mask = (
        data["is_tradeable"].eq(True)
        & ~data["is_suspended"]
        & ~data["is_st"]
        & ~data["is_delisting_period"]
        & data["listing_trade_days"].ge(features.min_listing_days)
        & industries.notna()
    )

    atr20 = af.atr(data["adj_high"], data["adj_low"], data["adj_close"], stocks, features.atr_window)
    ma20 = af.rolling_mean(data["adj_close"], stocks, features.ma_window)
    ma60 = af.rolling_mean(data["adj_close"], stocks, features.long_ma_window)
    down = af.downside_deviation(data["adj_close"], ma20, atr20)
    wick = af.lower_shadow(data["raw_open"], data["raw_close"], data["raw_low"], atr20)
    repair = af.intraday_repair(data["raw_close"], data["raw_low"], data["raw_high"])
    down_pct = af.rolling_percentile(down, stocks, features.percentile_window)
    wick_pct = af.rolling_percentile(wick, stocks, features.percentile_window)
    vol_state = atr20 / data["adj_close"].where(data["adj_close"] > 0)
    vol_pct = af.rolling_percentile(vol_state, stocks, features.percentile_window)
    log_amt_20 = sf.log_avg_amount(data["amount_cny"], stocks, features.amount_window)
    amt_shock = af.amount_shock(data["amount_cny"], stocks, features.amount_shock_window)

    ret1 = data["adj_close"].groupby(stocks, sort=False).pct_change(1, fill_method=None)
    ret5 = data["adj_close"].groupby(stocks, sort=False).pct_change(features.market_window, fill_method=None)
    market_ret_1d = ret1.where(valid_mask).groupby(dates, sort=False).transform("mean")
    market_ret_5d = ret5.where(valid_mask).groupby(dates, sort=False).transform("mean")
    industry_ret_1d = sf._industry_loo_mean(ret1, dates, industries)
    industry_ret_5d = sf._industry_loo_mean(ret5, dates, industries)

    lower_limit_price = data.get("limit_down_price")
    upper_limit_price = data.get("limit_up_price")
    if lower_limit_price is None:
        lower_limit_price = data["pre_close"] * 0.9
    if upper_limit_price is None:
        upper_limit_price = data["pre_close"] * 1.1
    limit_flag = (
        data.get("is_limit_up_open", pd.Series(False, index=data.index)).astype(bool)
        | data.get("is_limit_down_open", pd.Series(False, index=data.index)).astype(bool)
        | (data["raw_high"] >= upper_limit_price * 0.999)
        | (data["raw_low"] <= lower_limit_price * 1.001)
    )
    near_down_limit = data["raw_close"] <= lower_limit_price * 1.02

    cols = {
        "down_deviation_atr": down,
        "down_deviation_pct": down_pct,
        "lower_shadow_atr": wick,
        "lower_shadow_pct": wick_pct,
        "intraday_repair": repair,
        "core_signal": af.core_signal(down, wick, repair),
        "trend_state": af.trend_state(data["adj_close"], ma20, ma60),
        "vol_state": vol_state,
        "vol_state_pct": vol_pct,
        "amount_shock": amt_shock,
        "liquidity_log_amount_20": log_amt_20,
        "limit_flag": limit_flag.astype(float),
        "near_down_limit_flag": near_down_limit.astype(float),
        "market_ret_1d": market_ret_1d,
        "market_ret_5d": market_ret_5d,
        "industry_ret_1d": industry_ret_1d,
        "industry_ret_5d": industry_ret_5d,
        "stock_minus_industry_5d": ret5 - industry_ret_5d,
    }
    feature_names = FEATURE_GROUPS["all"]
    out = pd.DataFrame({"datetime": dates.to_numpy(), "instrument": stocks.to_numpy()})
    for name in feature_names:
        out[name] = cols[name].to_numpy()
    for horizon in sorted(set(label.horizons + [label.primary_horizon])):
        raw = _forward_label(data, stocks, dates, industries, horizon, label)
        name = "label" if horizon == label.primary_horizon else f"label_{horizon}"
        out[name] = raw.to_numpy()
        if label.cross_sectional_rank_label:
            ranked = out.groupby("datetime")[name].rank(pct=True) - 0.5
            out[name] = ranked.where(out[name].notna())

    scale_targets = [n for n in feature_names if n not in NO_CROSS_SECTION_ZSCORE]
    out[scale_targets] = out[scale_targets].replace([np.inf, -np.inf], np.nan)
    if features.winsor_quantile:
        q = features.winsor_quantile
        grouped = out.groupby("datetime")
        lower = grouped[scale_targets].transform(lambda s: s.quantile(q))
        upper = grouped[scale_targets].transform(lambda s: s.quantile(1 - q))
        out[scale_targets] = out[scale_targets].clip(lower, upper)
    if features.cross_sectional_zscore:
        grouped = out.groupby("datetime")[scale_targets]
        mean = grouped.transform("mean")
        std = grouped.transform("std", ddof=0)
        out[scale_targets] = (out[scale_targets] - mean) / std.replace(0, np.nan)

    exclude = ~valid_mask.to_numpy()
    label_cols = [c for c in out.columns if c == "label" or c.startswith("label_")]
    out.loc[exclude, feature_names + label_cols] = np.nan
    out.loc[out["limit_flag"].eq(1.0), feature_names + label_cols] = np.nan

    if features.use_sample_weight:
        weight = pd.Series(1.0, index=data.index)
        weight = weight.mask(near_down_limit, 0.35)
        weight = weight.mask(vol_pct >= features.extreme_vol_quantile, 0.5)
        weight = weight.mask(log_amt_20 < log_amt_20.groupby(dates, sort=False).transform(lambda s: s.quantile(0.1)), 0.5)
        out["sample_weight"] = weight.to_numpy()
        out.loc[exclude | limit_flag.to_numpy(), "sample_weight"] = np.nan
    else:
        out["sample_weight"] = np.nan
    return out, feature_names


def _forward_label(
    data: pd.DataFrame,
    stocks: pd.Series,
    dates: pd.Series,
    industries: pd.Series,
    horizon: int,
    label: ATRReversionLabelConfig,
) -> pd.Series:
    if label.label_method == "open_to_open":
        fwd = data["adj_open"].groupby(stocks, sort=False).shift(-(horizon + 1)) / data["adj_open"].groupby(stocks, sort=False).shift(-1) - 1.0
    else:
        fwd = data["adj_close"].groupby(stocks, sort=False).shift(-horizon) / data["adj_open"].groupby(stocks, sort=False).shift(-1) - 1.0
    if label.industry_neutralize:
        return fwd - sf._industry_loo_mean(fwd, dates, industries)
    return fwd
