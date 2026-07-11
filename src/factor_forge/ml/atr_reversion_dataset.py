"""Dataset assembly for ATR lower-shadow reversion experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import atr_reversion_features as af
from . import supply_features as sf
from .atr_reversion_config import ATRReversionFeatureConfig, ATRReversionLabelConfig


SHAPE_FEATURES = [
    "touch_depth_atr", "reclaim_atr", "lower_wick_share", "lower_wick_atr",
    "upper_wick_share", "upper_wick_atr", "close_location", "body_share",
]
PRICE_FEATURES = ["price_velocity_3_atr"]
VOLATILITY_FEATURES = ["atr_pct_120", "atr_velocity_5"]
FLOW_FEATURES = ["flow_intensity", "flow_change_3_5", "amount_ratio_20"]
ACCELERATION_FEATURES = ["atr_acceleration_5_5", "price_acceleration_3_3_atr"]

# Kept as a public alias for existing callers of the original research module.
CORE_FEATURES = SHAPE_FEATURES
FEATURE_GROUPS = {
    "S": SHAPE_FEATURES,
    "P": PRICE_FEATURES,
    "V": VOLATILITY_FEATURES,
    "F": FLOW_FEATURES,
    "A": ACCELERATION_FEATURES,
    "all": SHAPE_FEATURES + PRICE_FEATURES + VOLATILITY_FEATURES + FLOW_FEATURES + ACCELERATION_FEATURES,
}
NO_CROSS_SECTION_ZSCORE = frozenset({"lower_wick_share", "upper_wick_share", "close_location", "body_share", "atr_pct_120"})

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

    # All intra-day geometric measures use adjusted OHLC.  This avoids mixing a
    # corporate-action-adjusted Bollinger band with raw candle prices.
    close, open_, high, low = (data[c] for c in ("adj_close", "adj_open", "adj_high", "adj_low"))
    atr20 = af.atr(high, low, close, stocks, features.atr_window)
    atr_lag = af.lag(atr20, stocks)
    close_lag = af.lag(close, stocks)
    mid = sf._rolling(close, stocks, features.ma_window, method="mean", min_periods=max(5, features.ma_window // 2), shift=1)
    std = af.rolling_std(close, stocks, features.ma_window, shift=1)
    lower_band = mid - features.bollinger_std * std
    denom_atr = atr_lag.where(atr_lag > af.EPS)
    intraday_range = high - low
    body_bottom = pd.Series(np.minimum(open_.to_numpy(), close.to_numpy()), index=data.index)
    body_top = pd.Series(np.maximum(open_.to_numpy(), close.to_numpy()), index=data.index)
    lower_wick = (body_bottom - low).clip(lower=0.0)
    upper_wick = (high - body_top).clip(lower=0.0)
    range_denom = intraday_range.where(intraday_range > af.EPS)

    touch = ((lower_band - low) / denom_atr).clip(-features.shape_clip, features.shape_clip)
    reclaim = ((close - lower_band) / denom_atr).clip(-features.shape_clip, features.shape_clip)
    natr = atr20 / close.where(close > 0)
    smooth_natr = af.ema(natr, stocks, features.natr_ema_span)
    log_smooth_natr = np.log(smooth_natr.where(smooth_natr > af.EPS))
    velocity_window = features.velocity_window
    acceleration_window = features.acceleration_window
    atr_velocity = af.rolling_slope(log_smooth_natr, stocks, velocity_window)
    atr_accel = atr_velocity - af.lag(atr_velocity, stocks, acceleration_window)
    log_close = np.log(close.where(close > 0))
    price_slope_3 = af.rolling_slope(log_close, stocks, 3)
    price_accel = price_slope_3 - af.lag(price_slope_3, stocks, 3)
    price_velocity = ((log_close - af.lag(log_close, stocks, 3)) / 3.0) / (atr_lag / close_lag).where(close_lag > 0)
    price_accel = price_accel / (atr_lag / close_lag).where(close_lag > 0)
    amount_base = sf._rolling(data["amount_cny"], stocks, features.amount_window, method="mean", min_periods=max(5, features.amount_window // 2), shift=1)
    amount_ratio = data["amount_cny"] / amount_base.where(amount_base > 0)
    flow_col = features.net_flow_column or next(
        (name for name in ("net_flow_cny", "net_mf_amount", "main_net_inflow", "main_net_amount", "net_amount") if name in data),
        None,
    )
    if flow_col is None:
        flow_intensity = pd.Series(np.nan, index=data.index)
        flow_change = pd.Series(np.nan, index=data.index)
    else:
        flow_intensity = data[flow_col] / data["amount_cny"].where(data["amount_cny"] > 0)
        recent = sf._rolling(flow_intensity, stocks, 3, method="mean", min_periods=3)
        previous = sf._rolling(flow_intensity, stocks, 5, method="mean", min_periods=5, shift=3)
        flow_change = recent - previous

    cols = {
        "touch_depth_atr": touch,
        "reclaim_atr": reclaim,
        "lower_wick_share": lower_wick / range_denom,
        "lower_wick_atr": lower_wick / denom_atr,
        "upper_wick_share": upper_wick / range_denom,
        "upper_wick_atr": upper_wick / denom_atr,
        "close_location": (close - low) / range_denom,
        "body_share": (close - open_) / range_denom,
        "price_velocity_3_atr": price_velocity.clip(-features.shape_clip, features.shape_clip),
        "atr_pct_120": af.rolling_percentile(natr, stocks, features.percentile_window),
        "atr_velocity_5": atr_velocity,
        "flow_intensity": flow_intensity,
        "flow_change_3_5": flow_change,
        "amount_ratio_20": np.log(amount_ratio.where(amount_ratio > 0)),
        "atr_acceleration_5_5": atr_accel,
        "price_acceleration_3_3_atr": price_accel.clip(-features.shape_clip, features.shape_clip),
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

    out["event_pool"] = touch.ge(features.event_pool_threshold).to_numpy()
    out["net_flow_available"] = bool(flow_col)
    exclude = ~valid_mask.to_numpy()
    label_cols = [c for c in out.columns if c == "label" or c.startswith("label_")]
    out.loc[exclude, feature_names + label_cols] = np.nan
    if features.use_sample_weight:
        out["sample_weight"] = 1.0
        out.loc[exclude, "sample_weight"] = np.nan
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
