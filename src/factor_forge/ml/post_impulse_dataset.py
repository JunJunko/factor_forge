from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Segments
from .post_impulse_config import PostImpulseFeatureConfig, PostImpulseLabelConfig


EPS = 1e-12


@dataclass(frozen=True)
class PostImpulseDataset:
    events: pd.DataFrame
    path: pd.DataFrame
    feature_blocks: dict[str, list[str]]
    feature_manifest: list[dict]


REQUIRED_COLUMNS = [
    "trade_date", "ts_code", "raw_open", "raw_high", "raw_low", "raw_close",
    "adj_open", "adj_high", "adj_low", "adj_close", "volume_shares", "amount_cny",
    "turnover_rate", "circ_mv_cny", "industry_l1_code", "is_liquid", "is_tradeable",
    "is_suspended", "is_st", "is_delisting_period", "listing_trade_days",
]


def _lag(value: pd.Series, codes: pd.Series, periods: int = 1) -> pd.Series:
    return value.groupby(codes, sort=False).shift(periods)


def _rolling_mean(
    value: pd.Series, codes: pd.Series, window: int, *, min_periods: int | None = None
) -> pd.Series:
    result = value.groupby(codes, sort=False).rolling(
        window, min_periods=min_periods or window
    ).mean()
    return result.reset_index(level=0, drop=True).reindex(value.index)


def _rolling_std(
    value: pd.Series, codes: pd.Series, window: int, *, min_periods: int | None = None
) -> pd.Series:
    result = value.groupby(codes, sort=False).rolling(
        window, min_periods=min_periods or window
    ).std(ddof=0)
    return result.reset_index(level=0, drop=True).reindex(value.index)


def _ts_rank(
    value: pd.Series, codes: pd.Series, window: int, *, min_periods: int
) -> pd.Series:
    ranked = value.groupby(codes, sort=False).rolling(
        window, min_periods=min_periods
    ).rank(pct=True)
    return ranked.reset_index(level=0, drop=True).reindex(value.index)


def _future_extreme(
    value: pd.Series, codes: pd.Series, horizon: int, method: str
) -> pd.Series:
    """Extreme over T+1..T+horizon aligned to T."""
    next_value = _lag(value, codes, -1)
    reverse_value, reverse_codes = next_value.iloc[::-1], codes.iloc[::-1]
    rolled = reverse_value.groupby(reverse_codes, sort=False).rolling(
        horizon, min_periods=horizon
    ).agg(method)
    rolled = rolled.reset_index(level=0, drop=True)
    return rolled.iloc[::-1].reindex(value.index)


def _group_slope(frame: pd.DataFrame, column: str) -> pd.Series:
    """Vectorized OLS slope by event using the integer path offset."""
    valid = frame[column].notna() & frame["offset"].notna()
    sample = frame.loc[valid, ["event_id", "offset", column]].copy()
    if sample.empty:
        return pd.Series(dtype=float, name=column)
    sample["xy"] = sample["offset"] * sample[column]
    sample["x2"] = sample["offset"] ** 2
    grouped = sample.groupby("event_id", sort=False)
    stats = grouped.agg(
        n=(column, "size"), sum_x=("offset", "sum"), sum_y=(column, "sum"),
        sum_xy=("xy", "sum"), sum_x2=("x2", "sum"),
    )
    denominator = stats["n"] * stats["sum_x2"] - stats["sum_x"] ** 2
    slope = (
        stats["n"] * stats["sum_xy"] - stats["sum_x"] * stats["sum_y"]
    ) / denominator.where(denominator.abs() > EPS)
    return slope.where(stats["n"].ge(2)).rename(column)


def _monotonicity(frame: pd.DataFrame, column: str, *, increasing: bool) -> pd.Series:
    ordered = frame.sort_values(["event_id", "offset"], kind="stable")
    delta = ordered.groupby("event_id", sort=False)[column].diff()
    observed = delta.notna()
    conforms = delta.gt(0) if increasing else delta.lt(0)
    numerator = conforms.where(observed).groupby(ordered["event_id"], sort=False).sum(min_count=1)
    denominator = observed.groupby(ordered["event_id"], sort=False).sum()
    return (numerator / denominator.where(denominator > 0)).rename(column)


def _build_market_state(data: pd.DataFrame, spec: PostImpulseFeatureConfig) -> pd.DataFrame:
    eligible = data["is_valid_universe"] & data["return_1d"].notna()
    sample = data.loc[eligible, [
        "trade_date", "ts_code", "industry_l1_code", "return_1d", "circ_mv_cny",
    ]].copy()
    sample["positive"] = sample["return_1d"].gt(0).astype(float)
    market = sample.groupby("trade_date", sort=True).agg(
        market_return=("return_1d", "mean"), breadth=("positive", "mean")
    )
    market["regime__market_trend_20d"] = np.log1p(
        market["market_return"].clip(lower=-0.999)
    ).rolling(spec.market_window, min_periods=spec.market_window).sum()
    market["regime__market_breadth_20d"] = market["breadth"].rolling(
        spec.market_window, min_periods=spec.market_window
    ).mean()
    volatility = market["market_return"].rolling(
        spec.market_window, min_periods=spec.market_window
    ).std(ddof=0)
    market["regime__market_volatility_pct"] = volatility.rolling(
        spec.history_window, min_periods=spec.min_history
    ).rank(pct=True)

    sample["size_pct"] = sample.groupby("trade_date", sort=False)["circ_mv_cny"].rank(pct=True)
    small = sample["return_1d"].where(sample["size_pct"].le(0.3)).groupby(
        sample["trade_date"], sort=True
    ).mean()
    large = sample["return_1d"].where(sample["size_pct"].ge(0.7)).groupby(
        sample["trade_date"], sort=True
    ).mean()
    market["regime__small_minus_large_20d"] = (small - large).rolling(
        spec.market_window, min_periods=spec.market_window
    ).sum()
    market = market.reset_index()[[
        "trade_date", "regime__market_trend_20d", "regime__market_breadth_20d",
        "regime__market_volatility_pct", "regime__small_minus_large_20d",
    ]]

    industry = sample.groupby(
        ["industry_l1_code", "trade_date"], sort=True, as_index=False
    )["positive"].mean()
    industry["regime__industry_breadth_5d"] = industry.groupby(
        "industry_l1_code", sort=False
    )["positive"].transform(
        lambda value: value.rolling(
            spec.industry_breadth_window,
            min_periods=spec.industry_breadth_window,
        ).mean()
    )
    industry = industry[[
        "industry_l1_code", "trade_date", "regime__industry_breadth_5d",
    ]]
    return data.merge(market, on="trade_date", how="left", validate="many_to_one").merge(
        industry, on=["industry_l1_code", "trade_date"], how="left", validate="many_to_one"
    )


def _build_daily_features(
    panel: pd.DataFrame,
    features: PostImpulseFeatureConfig,
    labels: PostImpulseLabelConfig,
) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(panel.columns)
    if missing:
        raise ValueError("post-impulse panel missing columns: " + ", ".join(sorted(missing)))
    data = panel.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data.sort_values(["ts_code", "trade_date"], kind="stable").reset_index(drop=True)
    data["stock_row"] = data.groupby("ts_code", sort=False).cumcount()
    codes = data["ts_code"]
    close = pd.to_numeric(data["adj_close"], errors="coerce")
    high = pd.to_numeric(data["adj_high"], errors="coerce")
    low = pd.to_numeric(data["adj_low"], errors="coerce")
    prev_close = _lag(close, codes)
    data["return_1d"] = close / prev_close - 1.0
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    data["atr"] = _rolling_mean(true_range, codes, features.atr_window)
    atr = data["atr"].where(data["atr"] > EPS)

    data["is_valid_universe"] = (
        data["is_liquid"].fillna(False).astype(bool)
        & data["is_tradeable"].fillna(False).astype(bool)
        & ~data["is_suspended"].fillna(True).astype(bool)
        & ~data["is_st"].fillna(True).astype(bool)
        & ~data["is_delisting_period"].fillna(True).astype(bool)
        & data["listing_trade_days"].ge(features.min_listing_days)
    )

    data["event__return_3d"] = close / _lag(close, codes, features.impulse_return_window) - 1.0
    data["event__return_3d_ts_pct"] = _ts_rank(
        data["event__return_3d"], codes, features.history_window,
        min_periods=features.min_history,
    )
    data["event__return_3d_industry_pct"] = data.groupby(
        ["trade_date", "industry_l1_code"], sort=False
    )["event__return_3d"].rank(pct=True)
    turnover = pd.to_numeric(data["turnover_rate"], errors="coerce").clip(lower=0.0)
    turnover_mean = _rolling_mean(turnover, codes, features.atr_window)
    data["event__turnover_shock_ts_pct"] = _ts_rank(
        turnover / turnover_mean.where(turnover_mean > EPS), codes,
        features.history_window, min_periods=features.min_history,
    )

    raw_range = (data["raw_high"] - data["raw_low"]).where(lambda value: value > EPS)
    data["__close_location"] = (data["raw_close"] - data["raw_low"]) / raw_range
    data["event__close_location"] = data["__close_location"]
    data["event__distance_ma20_atr"] = (
        close - _rolling_mean(close, codes, features.market_window)
    ) / atr

    volatility = _rolling_std(data["return_1d"], codes, features.atr_window)
    liquidity = _rolling_mean(np.log1p(data["amount_cny"].clip(lower=0.0)), codes, features.atr_window)
    data["coord__log_circ_mv"] = np.log(data["circ_mv_cny"].where(data["circ_mv_cny"] > 0))
    data["coord__volatility_20d_ts_pct"] = _ts_rank(
        volatility, codes, features.history_window, min_periods=features.min_history
    )
    data["coord__liquidity_20d_ts_pct"] = _ts_rank(
        liquidity, codes, features.history_window, min_periods=features.min_history
    )
    market_return = data["return_1d"].where(data["is_valid_universe"]).groupby(
        data["trade_date"], sort=False
    ).transform("mean")
    mean_x = _rolling_mean(data["return_1d"], codes, features.beta_window)
    mean_y = _rolling_mean(market_return, codes, features.beta_window)
    mean_xy = _rolling_mean(data["return_1d"] * market_return, codes, features.beta_window)
    mean_y2 = _rolling_mean(market_return**2, codes, features.beta_window)
    covariance, variance = mean_xy - mean_x * mean_y, mean_y2 - mean_y**2
    data["coord__beta_60d"] = covariance / variance.where(variance.abs() > EPS)

    upper_shadow = (
        data["raw_high"] - pd.concat([data["raw_open"], data["raw_close"]], axis=1).max(axis=1)
    ).clip(lower=0.0) / raw_range
    daily_vwap = data["amount_cny"] / data["volume_shares"].where(data["volume_shares"] > EPS)
    data["__close_vwap_gap"] = data["raw_close"] / daily_vwap.where(daily_vwap > EPS) - 1.0
    down_turnover = turnover.where(data["return_1d"].lt(0), 0.0)
    pressure_sources = {
        "__p_down_turnover": _rolling_mean(down_turnover, codes, features.observation_days),
        "__p_upper_shadow": upper_shadow * turnover,
        "__p_close_below_vwap": (-data["__close_vwap_gap"]).clip(lower=0.0),
        "__p_high_to_close": (data["adj_high"] - data["adj_close"]).clip(lower=0.0) / atr,
    }
    pressure_rank_columns = []
    for name, value in pressure_sources.items():
        rank_name = name + "_pct"
        data[rank_name] = _ts_rank(
            value, codes, features.history_window, min_periods=features.min_history
        )
        pressure_rank_columns.append(rank_name)
    data["__pressure_daily"] = data[pressure_rank_columns].mean(axis=1, skipna=False)
    data["__pressure_components_daily"] = data[pressure_rank_columns].ge(
        features.pressure_threshold
    ).sum(axis=1)

    downside_impact = (-data["return_1d"].clip(upper=0.0)) / turnover.where(
        data["return_1d"].lt(0) & turnover.gt(EPS)
    )
    # No down-close observation remains missing. It is never encoded as perfect absorption.
    data["__downside_impact_pct"] = _ts_rank(
        downside_impact, codes, features.history_window,
        min_periods=max(10, features.min_history // 3),
    )
    data["__range_atr"] = (data["adj_high"] - data["adj_low"]) / atr

    data = _build_market_state(data, features)

    entry = _lag(pd.to_numeric(data["adj_open"], errors="coerce"), codes, -1)
    exit_open = _lag(pd.to_numeric(data["adj_open"], errors="coerce"), codes, -(labels.horizon + 1))
    data["label__return_10d"] = exit_open / entry - 1.0
    industry_benchmark = data["label__return_10d"].where(data["is_valid_universe"]).groupby(
        [data["trade_date"], data["industry_l1_code"]], sort=False
    ).transform("mean")
    data["label__industry_excess_10d"] = data["label__return_10d"] - industry_benchmark
    future_high = _future_extreme(data["adj_high"], codes, labels.horizon, "max")
    future_low = _future_extreme(data["adj_low"], codes, labels.horizon, "min")
    atr_fraction = atr / close
    data["label__mfe_atr"] = (future_high / entry - 1.0) / atr_fraction
    data["label__mae_atr"] = (entry / future_low - 1.0) / atr_fraction
    data["label__quality_atr"] = (
        data["label__mfe_atr"] - labels.quality_mae_penalty * data["label__mae_atr"]
    )
    data["__future_high"] = future_high
    return data


def aggregate_event_path(
    path: pd.DataFrame, features: PostImpulseFeatureConfig
) -> pd.DataFrame:
    """Aggregate T+1..T+observation_days without hiding prerequisite failures."""
    if path.empty:
        return pd.DataFrame(columns=["event_id"])
    group = path.groupby("event_id", sort=False)
    component_columns = [
        "__p_down_turnover_pct", "__p_upper_shadow_pct",
        "__p_close_below_vwap_pct", "__p_high_to_close_pct",
    ]
    result = pd.DataFrame(index=group.size().index)
    for column in component_columns:
        result[f"pressure__{column[4:-4]}_level"] = group[column].mean()
    pressure_components = [column for column in result if column.startswith("pressure__")]
    result["pressure__level"] = result[pressure_components].mean(axis=1, skipna=False)
    result["pressure__component_count"] = result[pressure_components].ge(
        features.pressure_threshold
    ).sum(axis=1)
    result["pressure__active_days"] = group["__pressure_daily"].apply(
        lambda value: int(value.ge(features.pressure_threshold).sum())
    )
    result["pressure__slope"] = _group_slope(path, "__pressure_daily")
    result["pressure__turnover_slope"] = _group_slope(path, "turnover_rate")
    result["pressure__present"] = (
        result["pressure__level"].ge(features.pressure_threshold)
        & result["pressure__component_count"].ge(features.pressure_min_components)
    ).astype(float)

    result["absorb__impact_level"] = group["__downside_impact_pct"].mean()
    result["absorb__impact_observed_days"] = group["__downside_impact_pct"].count().astype(float)
    result["absorb__impact_slope"] = _group_slope(path, "__downside_impact_pct")
    result["absorb__impact_resilience"] = 1.0 - result["absorb__impact_level"]
    result["absorb__low_slope_atr"] = _group_slope(path, "__low_from_event_atr")
    result["absorb__close_location_slope"] = _group_slope(path, "__close_location")
    result["absorb__close_vwap_slope"] = _group_slope(path, "__close_vwap_gap")
    result["absorb__range_slope_atr"] = _group_slope(path, "__range_atr")
    result["absorb__low_monotonicity"] = _monotonicity(
        path, "__low_from_event_atr", increasing=True
    )
    result["absorb__close_monotonicity"] = _monotonicity(
        path, "__close_location", increasing=True
    )
    result["absorb__impact_missing"] = result["absorb__impact_observed_days"].eq(0).astype(float)

    absorption_columns = [
        "absorb__impact_level", "absorb__impact_slope", "absorb__impact_resilience",
        "absorb__low_slope_atr", "absorb__close_location_slope",
        "absorb__close_vwap_slope", "absorb__range_slope_atr",
        "absorb__low_monotonicity", "absorb__close_monotonicity",
    ]
    no_pressure = result["pressure__present"].ne(1.0)
    result.loc[no_pressure, absorption_columns] = np.nan
    return result.reset_index()


def _rank_path_confirmation(events: pd.DataFrame) -> pd.DataFrame:
    data = events.copy()
    rank_sources = {
        "absorb__low_slope_rank": data["absorb__low_slope_atr"],
        "absorb__close_slope_rank": data["absorb__close_location_slope"],
        "absorb__range_contraction_rank": -data["absorb__range_slope_atr"],
        "absorb__impact_decay_rank": -data["absorb__impact_slope"],
    }
    for name, source in rank_sources.items():
        data[name] = source.groupby(data["signal_date"], sort=False).rank(pct=True)
    core = [
        "absorb__low_slope_rank", "absorb__close_slope_rank",
        "absorb__range_contraction_rank",
    ]
    data["absorb__path_confirmation"] = data[core].mean(axis=1, skipna=False)
    pressure = data["pressure__present"].eq(1.0)
    data["interaction__pressure_impact"] = (
        data["pressure__level"] * data["absorb__impact_resilience"]
    ).where(pressure)
    data["interaction__absorption_strength"] = (
        data["pressure__level"]
        * data["absorb__impact_resilience"]
        * data["absorb__path_confirmation"]
    ).where(pressure)
    data["interaction__absorption_industry_breadth"] = (
        data["interaction__absorption_strength"] * data["regime__industry_breadth_5d"]
    )
    data["interaction__impact_market_volatility"] = (
        data["absorb__impact_resilience"] * data["regime__market_volatility_pct"]
    ).where(pressure)
    data["interaction__profit_pressure_turnover"] = (
        data["event__return_3d_ts_pct"] * data["event__turnover_shock_ts_pct"]
    )
    return data


def _feature_contract(events: pd.DataFrame) -> tuple[dict[str, list[str]], list[dict]]:
    blocks = {
        block: sorted(column for column in events.columns if column.startswith(block + "__"))
        for block in ["coord", "event", "pressure", "absorb", "regime", "interaction"]
    }
    # The prerequisite flag defines the population and is not offered as a predictive shortcut.
    blocks["pressure"] = [
        column for column in blocks["pressure"] if column != "pressure__present"
    ]
    manifest = []
    for block, columns in blocks.items():
        for column in columns:
            manifest.append({
                "name": column,
                "block": block,
                "availability_time": "event_close" if block == "event" else "signal_close",
                "fit_required": False,
                "missing_policy": (
                    "missing_indicator_and_train_median"
                    if block in {"absorb", "interaction"}
                    else "train_median"
                ),
                "role": "risk_coordinate" if block == "coord" else "predictor",
            })
    return blocks, manifest


def build_post_impulse_dataset(
    panel: pd.DataFrame,
    features: PostImpulseFeatureConfig,
    labels: PostImpulseLabelConfig,
) -> PostImpulseDataset:
    data = _build_daily_features(panel, features, labels)
    codes = data["ts_code"]
    qualified = (
        data["is_valid_universe"]
        & data["event__return_3d_ts_pct"].ge(features.impulse_percentile)
        & data["event__return_3d_industry_pct"].ge(features.industry_percentile)
    )
    prior = qualified.groupby(codes, sort=False).shift(1).eq(True).astype(float)
    prior_recent = prior.groupby(codes, sort=False).rolling(
        features.event_cooldown_days, min_periods=1
    ).max().reset_index(level=0, drop=True).reindex(data.index)
    starts = qualified & prior_recent.fillna(0.0).eq(0.0)

    event_columns = [column for column in data.columns if column.startswith("event__")]
    source = data.loc[starts, [
        "ts_code", "trade_date", "stock_row", "adj_high", "adj_close", "atr",
        *event_columns,
    ]].copy()
    source = source.rename(columns={
        "trade_date": "event_date", "stock_row": "event_row", "adj_high": "event_high",
        "adj_close": "event_close", "atr": "event_atr",
    })
    source["event_id"] = (
        source["ts_code"].astype(str) + "_" + source["event_date"].dt.strftime("%Y%m%d")
    )
    source["signal_row"] = source["event_row"] + features.observation_days

    signal_columns = [
        "ts_code", "stock_row", "trade_date", "industry_l1_code", "is_valid_universe",
        "__future_high", "label__return_10d", "label__industry_excess_10d",
        "label__mfe_atr", "label__mae_atr", "label__quality_atr",
        *[column for column in data.columns if column.startswith("coord__")],
        *[column for column in data.columns if column.startswith("regime__")],
    ]
    signal = data[signal_columns].rename(columns={
        "stock_row": "signal_row", "trade_date": "signal_date",
    })
    events = source.merge(
        signal, on=["ts_code", "signal_row"], how="left", validate="one_to_one"
    )
    events = events.loc[events["signal_date"].notna() & events["is_valid_universe"].eq(True)].copy()

    path_parts = []
    daily_path_columns = [
        "ts_code", "stock_row", "trade_date", "turnover_rate", "adj_low",
        "__close_location", "__close_vwap_gap", "__range_atr", "__pressure_daily",
        "__pressure_components_daily", "__downside_impact_pct",
        "__p_down_turnover_pct", "__p_upper_shadow_pct",
        "__p_close_below_vwap_pct", "__p_high_to_close_pct",
    ]
    daily_path = data[daily_path_columns]
    for offset in range(1, features.observation_days + 1):
        keys = events[["event_id", "ts_code", "event_row", "event_close", "event_atr"]].copy()
        keys["stock_row"] = keys["event_row"] + offset
        keys["offset"] = offset
        path_parts.append(keys.merge(
            daily_path, on=["ts_code", "stock_row"], how="left", validate="one_to_one"
        ))
    path = pd.concat(path_parts, ignore_index=True) if path_parts else pd.DataFrame()
    path["__low_from_event_atr"] = (
        path["adj_low"] - path["event_close"]
    ) / path["event_atr"].where(path["event_atr"].abs() > EPS)
    aggregates = aggregate_event_path(path, features)
    events = events.merge(aggregates, on="event_id", how="left", validate="one_to_one")
    events = _rank_path_confirmation(events)

    required_label = events[[
        "__future_high", "label__mfe_atr", "label__mae_atr",
    ]].notna().all(axis=1)
    breakout_level = events["event_high"] + labels.breakout_buffer_atr * events["event_atr"]
    success = (
        events["__future_high"].ge(breakout_level)
        & events["label__mfe_atr"].ge(labels.mfe_atr_threshold)
        & events["label__mae_atr"].le(labels.mae_atr_limit)
    )
    events["label__success"] = success.astype(float).where(required_label)
    events = events.drop(columns=["__future_high"])
    events = events.sort_values(["signal_date", "ts_code"], kind="stable").reset_index(drop=True)
    path = path.loc[path["event_id"].isin(set(events["event_id"]))].sort_values(
        ["event_id", "offset"], kind="stable"
    ).reset_index(drop=True)
    blocks, manifest = _feature_contract(events)
    return PostImpulseDataset(events=events, path=path, feature_blocks=blocks, feature_manifest=manifest)


def assign_purged_splits(
    events: pd.DataFrame,
    segments: Segments,
    trading_calendar: pd.Index,
    *,
    horizon: int,
) -> pd.DataFrame:
    """Assign chronological splits and remove boundary rows whose labels overlap."""
    data = events.copy()
    dates = pd.to_datetime(data["signal_date"])
    data["split"] = "outside"
    for name, segment in [
        ("train", segments.train), ("valid", segments.valid), ("test", segments.test)
    ]:
        data.loc[dates.between(pd.Timestamp(segment.start), pd.Timestamp(segment.end)), "split"] = name
    calendar = pd.Index(sorted(pd.to_datetime(trading_calendar).unique()))
    ordinal = pd.Series(np.arange(len(calendar), dtype=int), index=calendar)
    event_ordinal = dates.map(ordinal)
    if event_ordinal.loc[data["split"].isin(["train", "valid", "test"])].isna().any():
        raise ValueError("signal_date is missing from the supplied trading calendar")

    purge = horizon + 1
    valid_start = int(np.searchsorted(calendar.values, np.datetime64(segments.valid.start), side="left"))
    test_start = int(np.searchsorted(calendar.values, np.datetime64(segments.test.start), side="left"))
    train_overlap = data["split"].eq("train") & event_ordinal.gt(valid_start - purge - 1)
    valid_overlap = data["split"].eq("valid") & event_ordinal.gt(test_start - purge - 1)
    data["purged"] = train_overlap | valid_overlap
    data.loc[data["purged"], "split"] = "purged"
    return data
