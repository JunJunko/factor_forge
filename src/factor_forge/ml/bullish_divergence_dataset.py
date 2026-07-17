"""Causal bullish-divergence and support-touch feature assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import atr_reversion_features as af
from . import bullish_divergence_features as bf
from . import supply_features as sf
from .bullish_divergence_config import BullishDivergenceFeatureConfig


DIVERGENCE_FEATURES = [
    "div__price_lower_low_atr",
    "div__low_similarity_atr",
    "div__trough_gap_days",
    "div__intervening_rebound_atr",
    "div__b_age",
    "div__rsi6_higher_low",
    "div__rsi14_higher_low",
    "div__macd_hist_higher_low",
    "div__macd_hist_higher_low_atr",
    "div__macd_slope_b",
    "div__ret5_higher_low",
    "div__downside_velocity_change",
    "div__descent_into_a_3d",
    "div__descent_into_b_3d",
    "structure__triple_history_available",
    "structure__first_divergence_present",
    "structure__second_divergence_present",
    "structure__double_divergence_present",
    "structure__first_price_lower_low_atr",
    "structure__second_price_lower_low_atr",
    "structure__price_geometry_trend",
    "structure__first_rsi14_improvement",
    "structure__second_rsi14_improvement",
    "structure__rsi14_improvement_trend",
    "structure__first_macd_atr_improvement",
    "structure__second_macd_atr_improvement",
    "structure__macd_atr_improvement_trend",
    "structure__first_velocity_improvement",
    "structure__second_velocity_improvement",
    "structure__velocity_improvement_trend",
    "structure__first_volume_dryup",
    "structure__second_volume_dryup",
    "structure__volume_dryup_trend",
    "structure__first_rebound_atr",
    "structure__second_rebound_atr",
    "structure__rebound_trend",
    "structure__first_agreement_count",
    "structure__second_agreement_count",
    "structure__agreement_trend",
    "structure__trend_positive_count",
    "structure__double_divergence_trend_score",
    "div__down_volume_change",
    "div__turnover_change",
    "div__main_sell_ratio_change",
    "div__range_contraction",
    "div__natr_change",
    "div__close_location",
    "div__lower_shadow_atr",
    "div__reclaim_ma5_atr",
    "div__indicator_agreement_count",
    "div__history_valid_ratio",
    "div__reliability",
    "div__score",
    "div__score_rank",
]

TOUCH_MODEL_FEATURES = [
    "touch__level_to_close",
    "touch__zone_width_atr",
    "touch__occurred_10d",
    "touch__count_10d",
    "touch__nearest_distance_atr_10d",
    "touch__age_days",
    "touch__pre_b_count",
    "touch__post_b_count",
    "touch__last_penetration_atr",
    "touch__last_close_reclaim_atr",
    "touch__post_touch_return_to_t",
    "touch__false_break_reclaim",
    "touch__post_b_observable",
    "touch__acceptance_score",
]

TOUCH_DIAGNOSTIC_FIELDS = [
    "touch__level_raw",
    "touch__level_raw_origin",
    "touch__level_adj_pit",
    "touch__zone_width_adj",
    "div__pivot_a_date",
    "div__pivot_b_date",
]

STRUCTURE_DIAGNOSTIC_FIELDS = ["structure__pivot_p_date"]

REQUIRED_PANEL_COLUMNS = {
    "trade_date", "ts_code",
    "raw_open", "raw_high", "raw_low", "raw_close",
    "adj_open", "adj_high", "adj_low", "adj_close",
    "amount_cny", "turnover_rate",
    "is_st", "is_delisting_period", "is_suspended", "is_tradeable",
    "listing_trade_days",
}


def build_bullish_divergence_features(
    panel: pd.DataFrame,
    config: BullishDivergenceFeatureConfig = BullishDivergenceFeatureConfig(),
) -> tuple[pd.DataFrame, list[str]]:
    """Build daily D/T features using only each row's contemporaneous prefix.

    The support anchor is the lowest adjusted low B inside the current trailing
    window.  A is the prior trough and P is A's prior trough.  P->A and A->B
    therefore describe two causal divergence legs.  The anchor candle B is
    always excluded from touch counts.
    """
    missing = REQUIRED_PANEL_COLUMNS - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing bullish-divergence fields: {sorted(missing)}")

    data = panel.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    stocks = data["ts_code"]
    close = pd.to_numeric(data["adj_close"], errors="coerce")
    open_ = pd.to_numeric(data["adj_open"], errors="coerce")
    high = pd.to_numeric(data["adj_high"], errors="coerce")
    low = pd.to_numeric(data["adj_low"], errors="coerce")
    raw_close = pd.to_numeric(data["raw_close"], errors="coerce")

    atr = af.atr(high, low, close, stocks, config.atr_window)
    rsi_fast = bf.wilder_rsi(close, stocks, config.rsi_fast_window)
    rsi_slow = bf.wilder_rsi(close, stocks, config.rsi_slow_window)
    macd_hist = bf.macd_histogram(close, stocks)
    ret1 = close.groupby(stocks, sort=False).pct_change(fill_method=None)
    ret5 = close.groupby(stocks, sort=False).pct_change(5, fill_method=None)
    velocity3 = (close / close.groupby(stocks, sort=False).shift(3) - 1.0) / (
        atr / close.where(close > 0)
    ).where(close > 0)
    down_amount = pd.to_numeric(data["amount_cny"], errors="coerce").where(ret1.lt(0), 0.0)
    down_volume = sf._rolling(
        down_amount, stocks, 5, method="mean", min_periods=3
    ) / sf._rolling(
        pd.to_numeric(data["amount_cny"], errors="coerce"), stocks, config.atr_window,
        method="mean", min_periods=max(5, config.atr_window // 2), shift=1,
    ).where(lambda values: values > 0)
    ma5 = sf._rolling(close, stocks, 5, method="mean", min_periods=3)
    candle_range_atr = (high - low) / atr.where(atr > bf.EPS)
    natr = atr / close.where(close > 0)
    main_sell_ratio = _main_sell_ratio(data)

    work = data[[
        "trade_date", "ts_code", "raw_low", "raw_close", "turnover_rate",
        "is_st", "is_delisting_period", "is_suspended", "is_tradeable",
        "listing_trade_days",
    ]].copy()
    work["_open"] = open_
    work["_high"] = high
    work["_low"] = low
    work["_close"] = close
    work["_raw_close"] = raw_close
    work["_atr"] = atr
    work["_rsi_fast"] = rsi_fast
    work["_rsi_slow"] = rsi_slow
    work["_macd_hist"] = macd_hist
    work["_ret5"] = ret5
    work["_velocity3"] = velocity3
    work["_down_volume"] = down_volume
    work["_ma5"] = ma5
    work["_range_atr"] = candle_range_atr
    work["_natr"] = natr
    work["_main_sell_ratio"] = main_sell_ratio

    frames: list[pd.DataFrame] = []
    for _, stock in work.groupby("ts_code", sort=False, observed=True):
        frames.append(_build_stock_features(stock.reset_index(drop=True), config))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        return result, DIVERGENCE_FEATURES + TOUCH_MODEL_FEATURES

    valid = (
        result["is_tradeable"].fillna(False)
        & ~result["is_suspended"].fillna(True)
        & ~result["is_st"].fillna(False)
        & ~result["is_delisting_period"].fillna(False)
        & result["listing_trade_days"].ge(config.minimum_listing_days)
    )
    result = _attach_scores(result, valid)
    oscillator = result[[
        "div__rsi14_higher_low", "div__macd_hist_higher_low",
        "div__downside_velocity_change",
    ]].max(axis=1)
    result["div__event_candidate"] = (
        valid
        & result["div__b_age"].le(config.maximum_current_trough_age)
        & result["div__price_lower_low_atr"].ge(-config.lower_low_tolerance_atr)
        & oscillator.gt(0)
        & result["div__intervening_rebound_atr"].ge(config.minimum_intervening_rebound_atr)
        & result["div__history_valid_ratio"].ge(config.minimum_history_valid_ratio)
    )
    model_features = DIVERGENCE_FEATURES + TOUCH_MODEL_FEATURES
    result[model_features] = result[model_features].replace([np.inf, -np.inf], np.nan)
    # ``frames`` are already emitted in ts_code/trade_date order.  Sorting the
    # multi-million-row, highly fragmented result here forces pandas to
    # consolidate every float block and can require several extra GiB.
    return result, model_features


def build_divergence_episodes(
    features: pd.DataFrame,
    config: BullishDivergenceFeatureConfig = BullishDivergenceFeatureConfig(),
    *,
    candidate_field: str = "div__event_candidate",
    event_prefix: str = "",
) -> pd.DataFrame:
    """Keep the first causal signal in each per-stock cooldown episode."""
    required = {"trade_date", "ts_code", candidate_field}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"feature table is missing episode fields: {sorted(missing)}")
    dates = pd.to_datetime(features["trade_date"])
    ordering = pd.MultiIndex.from_arrays([features["ts_code"], dates])
    data = features if ordering.is_monotonic_increasing else features.sort_values(
        ["ts_code", "trade_date"], kind="stable"
    )
    ordinals = data.groupby("ts_code", sort=False).cumcount()
    candidate_mask = data[candidate_field].fillna(False)
    candidates = data.loc[candidate_mask].copy()
    candidates["trade_date"] = pd.to_datetime(candidates["trade_date"])
    candidates["_stock_ordinal"] = ordinals.loc[candidate_mask].to_numpy()
    outputs: list[pd.DataFrame] = []
    for code, stock in candidates.groupby("ts_code", sort=False, observed=True):
        stock = stock.sort_values("trade_date").copy()
        previous_kept_ordinal: int | None = None
        episode_start = False
        starts: list[bool] = []
        for ordinal in stock["_stock_ordinal"].astype(int):
            episode_start = (
                previous_kept_ordinal is None
                or ordinal - previous_kept_ordinal > config.episode_cooldown_days
            )
            starts.append(episode_start)
            if episode_start:
                previous_kept_ordinal = ordinal
        first = stock.loc[starts].copy()
        first["event_id"] = [
            f"{event_prefix}{code}:{date:%Y%m%d}" for date in first["trade_date"]
        ]
        first["episode_id"] = first["event_id"]
        outputs.append(first)
    if not outputs:
        columns = list(features.columns) + ["event_id", "episode_id"]
        return pd.DataFrame(columns=list(dict.fromkeys(columns)))
    return pd.concat(outputs, ignore_index=True).drop(columns="_stock_ordinal")


def bullish_divergence_feature_manifest() -> list[dict]:
    rows = []
    for name in DIVERGENCE_FEATURES:
        rows.append({
            "name": name,
            "group": (
                "S_low_structure_transition"
                if name.startswith("structure__")
                else "D_divergence"
            ),
            "role": "predictor",
            "clock": "signal_close_T", "missing_policy": "preserve_nan",
        })
    for name in TOUCH_MODEL_FEATURES:
        rows.append({
            "name": name, "group": "T_touch_retest", "role": "predictor",
            "clock": "signal_close_T", "missing_policy": "preserve_nan",
        })
    for name in TOUCH_DIAGNOSTIC_FIELDS:
        rows.append({
            "name": name, "group": "T_touch_retest", "role": "diagnostic_not_predictor",
            "clock": "signal_close_T", "missing_policy": "preserve_nan",
        })
    for name in STRUCTURE_DIAGNOSTIC_FIELDS:
        rows.append({
            "name": name,
            "group": "S_double_divergence_trend",
            "role": "diagnostic_not_predictor",
            "clock": "signal_close_T",
            "missing_policy": "preserve_nan",
        })
    return rows


def _build_stock_features(
    stock: pd.DataFrame,
    config: BullishDivergenceFeatureConfig,
) -> pd.DataFrame:
    n = len(stock)
    output = stock[[
        "trade_date", "ts_code", "is_st", "is_delisting_period", "is_suspended",
        "is_tradeable", "listing_trade_days",
    ]].copy()
    numeric_columns = [
        *[
            name
            for name in DIVERGENCE_FEATURES
            if name
            not in {
                "div__score",
                "div__score_rank",
                "structure__double_divergence_trend_score",
            }
        ],
        *[name for name in TOUCH_MODEL_FEATURES if name != "touch__acceptance_score"],
        "touch__level_raw", "touch__level_raw_origin", "touch__level_adj_pit",
        "touch__zone_width_adj",
    ]
    low = stock["_low"].to_numpy(float)
    high = stock["_high"].to_numpy(float)
    open_ = stock["_open"].to_numpy(float)
    close = stock["_close"].to_numpy(float)
    raw_low = pd.to_numeric(stock["raw_low"], errors="coerce").to_numpy(float)
    raw_close = stock["_raw_close"].to_numpy(float)
    atr = stock["_atr"].to_numpy(float)
    arrays = {
        "rsi_fast": stock["_rsi_fast"].to_numpy(float),
        "rsi_slow": stock["_rsi_slow"].to_numpy(float),
        "macd": stock["_macd_hist"].to_numpy(float),
        "ret5": stock["_ret5"].to_numpy(float),
        "velocity": stock["_velocity3"].to_numpy(float),
        "down_volume": stock["_down_volume"].to_numpy(float),
        "turnover": pd.to_numeric(stock["turnover_rate"], errors="coerce").to_numpy(float),
        "ma5": stock["_ma5"].to_numpy(float),
        "range_atr": stock["_range_atr"].to_numpy(float),
        "natr": stock["_natr"].to_numpy(float),
        "main_sell": stock["_main_sell_ratio"].to_numpy(float),
    }
    dates = pd.to_datetime(stock["trade_date"]).to_numpy()
    values_by_name = {name: np.full(n, np.nan, dtype=float) for name in numeric_columns}
    p_index, a_index, b_index = _causal_pivot_indices(low, config)
    row_index = np.arange(n)
    valid = (
        (a_index >= 0) & (b_index >= 0)
        & np.isfinite(atr[row_index]) & (atr[row_index] > bf.EPS)
    )
    safe_rows = row_index[valid]
    if len(safe_rows):
        p = p_index[safe_rows]
        a = a_index[safe_rows]
        b = b_index[safe_rows]
        denom = atr[a]
        finite_a_atr = np.isfinite(denom) & (denom > bf.EPS)
        safe_rows, p, a, b, denom = (
            values[finite_a_atr] for values in (safe_rows, p, a, b, denom)
        )
        gaps = b - a
        rebound_high = _range_maximum(high, a + 1, b)
        history_valid = _trailing_valid_ratio(
            low, safe_rows, config.previous_trough_lookback + 1
        )
        rsi_fast_delta = arrays["rsi_fast"][b] - arrays["rsi_fast"][a]
        rsi_slow_delta = arrays["rsi_slow"][b] - arrays["rsi_slow"][a]
        macd_delta = arrays["macd"][b] - arrays["macd"][a]
        macd_slope = arrays["macd"][b] - arrays["macd"][np.maximum(b - 3, 0)]
        ret5_delta = arrays["ret5"][b] - arrays["ret5"][a]
        velocity_delta = arrays["velocity"][b] - arrays["velocity"][a]
        down_volume_delta = arrays["down_volume"][b] - arrays["down_volume"][a]
        turnover_delta = arrays["turnover"][b] - arrays["turnover"][a]
        sell_delta = arrays["main_sell"][b] - arrays["main_sell"][a]
        range_contraction = arrays["range_atr"][a] - arrays["range_atr"][b]
        natr_delta = arrays["natr"][b] - arrays["natr"][a]
        descent_a = close[a] / close[np.maximum(a - 3, 0)] - 1.0
        descent_b = close[b] / close[np.maximum(b - 3, 0)] - 1.0
        candle_span = high[safe_rows] - low[safe_rows]
        close_location = np.divide(
            close[safe_rows] - low[safe_rows], candle_span,
            out=np.full(len(safe_rows), 0.5), where=candle_span > bf.EPS,
        )
        lower_shadow = np.maximum(np.minimum(open_[safe_rows], close[safe_rows]) - low[safe_rows], 0.0)
        denom_t = atr[safe_rows]
        agreements = (
            (rsi_slow_delta > 0).astype(float)
            + (macd_delta > 0).astype(float)
            + (velocity_delta > 0).astype(float)
            + (down_volume_delta < 0).astype(float)
        )
        calculated = {
            "div__price_lower_low_atr": (low[a] - low[b]) / denom,
            "div__low_similarity_atr": -np.abs(low[b] - low[a]) / denom,
            "div__trough_gap_days": gaps.astype(float),
            "div__intervening_rebound_atr": (rebound_high - np.maximum(low[a], low[b])) / denom,
            "div__b_age": (safe_rows - b).astype(float),
            "div__rsi6_higher_low": rsi_fast_delta,
            "div__rsi14_higher_low": rsi_slow_delta,
            "div__macd_hist_higher_low": macd_delta,
            "div__macd_hist_higher_low_atr": macd_delta / denom,
            "div__macd_slope_b": macd_slope,
            "div__ret5_higher_low": ret5_delta,
            "div__downside_velocity_change": velocity_delta,
            "div__descent_into_a_3d": descent_a,
            "div__descent_into_b_3d": descent_b,
            "div__down_volume_change": down_volume_delta,
            "div__turnover_change": turnover_delta,
            "div__main_sell_ratio_change": sell_delta,
            "div__range_contraction": range_contraction,
            "div__natr_change": natr_delta,
            "div__close_location": close_location,
            "div__lower_shadow_atr": lower_shadow / denom_t,
            "div__reclaim_ma5_atr": (close[safe_rows] - arrays["ma5"][safe_rows]) / denom_t,
            "div__indicator_agreement_count": agreements,
            "div__history_valid_ratio": history_valid,
            "div__reliability": history_valid * np.clip(gaps / 20.0, 0.0, 1.0),
        }
        for name, feature_values in calculated.items():
            values_by_name[name][safe_rows] = feature_values
        _assign_double_divergence_structure(
            values_by_name,
            rows=safe_rows,
            p=p,
            a=a,
            b=b,
            low=low,
            high=high,
            close=close,
            atr=atr,
            rsi_slow=arrays["rsi_slow"],
            macd=arrays["macd"],
            velocity=arrays["velocity"],
            down_volume=arrays["down_volume"],
            config=config,
        )
        _assign_touch_features_vectorized(
            values_by_name, rows=safe_rows, b=b, low=low, high=high, close=close,
            raw_low=raw_low, raw_close=raw_close, atr=atr, config=config,
        )
    output = pd.concat([output, pd.DataFrame(values_by_name)], axis=1)
    pivot_a_dates = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    pivot_b_dates = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    pivot_p_dates = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    valid_pivots = (a_index >= 0) & (b_index >= 0)
    pivot_a_dates[valid_pivots] = dates[a_index[valid_pivots]]
    pivot_b_dates[valid_pivots] = dates[b_index[valid_pivots]]
    valid_triples = valid_pivots & (p_index >= 0)
    pivot_p_dates[valid_triples] = dates[p_index[valid_triples]]
    output["div__pivot_a_date"] = pivot_a_dates
    output["div__pivot_b_date"] = pivot_b_dates
    output["structure__pivot_p_date"] = pivot_p_dates
    return output


def _assign_double_divergence_structure(
    output: dict[str, np.ndarray],
    *,
    rows: np.ndarray,
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    rsi_slow: np.ndarray,
    macd: np.ndarray,
    velocity: np.ndarray,
    down_volume: np.ndarray,
    config: BullishDivergenceFeatureConfig,
) -> None:
    if len(rows) == 0:
        return
    output["structure__triple_history_available"][rows] = 0.0

    atr_a = atr[a]
    second_price = (low[a] - low[b]) / atr_a
    second_rsi = rsi_slow[b] - rsi_slow[a]
    second_macd = (macd[b] - macd[a]) / atr_a
    second_velocity = velocity[b] - velocity[a]
    second_volume = down_volume[a] - down_volume[b]
    second_rebound_high = _range_maximum(high, a + 1, b)
    second_rebound = (second_rebound_high - np.maximum(low[a], low[b])) / atr_a
    second_agreement = (
        (second_rsi > 0).astype(float)
        + (second_macd > 0).astype(float)
        + (second_velocity > 0).astype(float)
        + (second_volume > 0).astype(float)
    )
    second_gap = b - a
    descent_a = close[a] / close[np.maximum(a - 3, 0)] - 1.0
    descent_b = close[b] / close[np.maximum(b - 3, 0)] - 1.0
    second_geometry = (
        (second_gap >= config.minimum_trough_separation)
        & (second_gap <= config.previous_trough_lookback)
        & (second_price >= -config.lower_low_tolerance_atr)
        & (second_price <= config.maximum_lower_low_atr)
        & (second_rebound >= config.minimum_intervening_rebound_atr)
        & (descent_a < 0)
        & (descent_b < 0)
    )
    second_present = second_geometry & (
        (second_rsi > 0) | (second_macd > 0) | (second_velocity > 0)
    )
    output["structure__second_divergence_present"][rows] = second_present.astype(float)

    history = (
        (p >= 0)
        & np.isfinite(atr[np.maximum(p, 0)])
        & (atr[np.maximum(p, 0)] > bf.EPS)
    )
    if not history.any():
        return
    positions = np.flatnonzero(history)
    history_rows = rows[positions]
    hp = p[positions]
    ha = a[positions]
    atr_p = atr[hp]
    output["structure__triple_history_available"][history_rows] = 1.0

    first_price = (low[hp] - low[ha]) / atr_p
    first_rsi = rsi_slow[ha] - rsi_slow[hp]
    first_macd = (macd[ha] - macd[hp]) / atr_p
    first_velocity = velocity[ha] - velocity[hp]
    first_volume = down_volume[hp] - down_volume[ha]
    first_rebound_high = _range_maximum(high, hp + 1, ha)
    first_rebound = (first_rebound_high - np.maximum(low[hp], low[ha])) / atr_p
    first_agreement = (
        (first_rsi > 0).astype(float)
        + (first_macd > 0).astype(float)
        + (first_velocity > 0).astype(float)
        + (first_volume > 0).astype(float)
    )
    first_gap = ha - hp
    descent_p = close[hp] / close[np.maximum(hp - 3, 0)] - 1.0
    descent_a_history = close[ha] / close[np.maximum(ha - 3, 0)] - 1.0
    first_geometry = (
        (first_gap >= config.minimum_trough_separation)
        & (first_gap <= config.previous_trough_lookback)
        & (first_price >= -config.lower_low_tolerance_atr)
        & (first_price <= config.maximum_lower_low_atr)
        & (first_rebound >= config.minimum_intervening_rebound_atr)
        & (descent_p < 0)
        & (descent_a_history < 0)
    )
    first_present = first_geometry & (
        (first_rsi > 0) | (first_macd > 0) | (first_velocity > 0)
    )
    double_present = first_present & second_present[positions]
    output["structure__first_divergence_present"][history_rows] = first_present.astype(float)
    output["structure__double_divergence_present"][history_rows] = double_present.astype(float)
    if not double_present.any():
        return

    selected = np.flatnonzero(double_present)
    selected_rows = history_rows[selected]
    first_price = first_price[selected]
    first_rsi = first_rsi[selected]
    first_macd = first_macd[selected]
    first_velocity = first_velocity[selected]
    first_volume = first_volume[selected]
    first_rebound = first_rebound[selected]
    first_agreement = first_agreement[selected]
    second_price = second_price[positions][selected]
    second_rsi = second_rsi[positions][selected]
    second_macd = second_macd[positions][selected]
    second_velocity = second_velocity[positions][selected]
    second_volume = second_volume[positions][selected]
    second_rebound = second_rebound[positions][selected]
    second_agreement = second_agreement[positions][selected]

    trends = {
        "structure__price_geometry_trend": second_price - first_price,
        "structure__rsi14_improvement_trend": second_rsi - first_rsi,
        "structure__macd_atr_improvement_trend": second_macd - first_macd,
        "structure__velocity_improvement_trend": second_velocity - first_velocity,
        "structure__volume_dryup_trend": second_volume - first_volume,
        "structure__rebound_trend": second_rebound - first_rebound,
        "structure__agreement_trend": second_agreement - first_agreement,
    }
    values = {
        "structure__first_price_lower_low_atr": first_price,
        "structure__second_price_lower_low_atr": second_price,
        "structure__first_rsi14_improvement": first_rsi,
        "structure__second_rsi14_improvement": second_rsi,
        "structure__first_macd_atr_improvement": first_macd,
        "structure__second_macd_atr_improvement": second_macd,
        "structure__first_velocity_improvement": first_velocity,
        "structure__second_velocity_improvement": second_velocity,
        "structure__first_volume_dryup": first_volume,
        "structure__second_volume_dryup": second_volume,
        "structure__first_rebound_atr": first_rebound,
        "structure__second_rebound_atr": second_rebound,
        "structure__first_agreement_count": first_agreement,
        "structure__second_agreement_count": second_agreement,
        **trends,
    }
    for name, feature_values in values.items():
        output[name][selected_rows] = feature_values
    favorable_trends = np.column_stack([
        trends["structure__rsi14_improvement_trend"],
        trends["structure__macd_atr_improvement_trend"],
        trends["structure__velocity_improvement_trend"],
        trends["structure__volume_dryup_trend"],
        trends["structure__rebound_trend"],
        trends["structure__agreement_trend"],
    ])
    output["structure__trend_positive_count"][selected_rows] = (
        favorable_trends > 0
    ).sum(axis=1).astype(float)


def _causal_pivot_indices(
    low: np.ndarray,
    config: BullishDivergenceFeatureConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(low)
    p_index = np.full(n, -1, dtype=int)
    a_index = np.full(n, -1, dtype=int)
    b_index = np.full(n, -1, dtype=int)
    current = config.current_trough_window
    if n >= current:
        windows = np.lib.stride_tricks.sliding_window_view(low, current)
        finite = np.isfinite(windows)
        relative = np.argmin(np.where(finite, windows, np.inf), axis=1)
        rows = np.arange(current - 1, n)
        valid = finite.any(axis=1)
        b_index[rows[valid]] = rows[valid] - current + 1 + relative[valid]

    lookback = config.previous_trough_lookback
    separation = config.minimum_trough_separation
    a_for_b = np.full(n, -1, dtype=int)
    if n >= lookback + 1:
        history = np.lib.stride_tricks.sliding_window_view(low, lookback + 1)
        candidates = history[:, :lookback - separation + 1]
        finite = np.isfinite(candidates)
        relative = np.argmin(np.where(finite, candidates, np.inf), axis=1)
        b_rows = np.arange(lookback, n)
        valid = finite.any(axis=1)
        a_for_b[b_rows[valid]] = b_rows[valid] - lookback + relative[valid]
    has_b = b_index >= 0
    a_index[has_b] = a_for_b[b_index[has_b]]
    has_a = a_index >= 0
    p_index[has_a] = a_for_b[a_index[has_a]]
    return p_index, a_index, b_index


def _range_maximum(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Vectorized half-open range maximum using a sparse table."""
    output = np.full(len(starts), np.nan, dtype=float)
    lengths = ends - starts
    usable = lengths > 0
    if not usable.any():
        return output
    clean = np.where(np.isfinite(values), values, -np.inf)
    table = [clean]
    width = 2
    while width <= len(values):
        half = width // 2
        previous = table[-1]
        table.append(np.maximum(previous[:-half], previous[half:]))
        width *= 2
    positions = np.flatnonzero(usable)
    usable_lengths = lengths[positions]
    powers = np.floor(np.log2(usable_lengths)).astype(int)
    for power in np.unique(powers):
        selected = positions[powers == power]
        block = 1 << int(power)
        left = starts[selected]
        right = ends[selected] - block
        output[selected] = np.maximum(table[int(power)][left], table[int(power)][right])
    output[np.isneginf(output)] = np.nan
    return output


def _trailing_valid_ratio(values: np.ndarray, rows: np.ndarray, window: int) -> np.ndarray:
    finite = np.isfinite(values).astype(int)
    cumulative = np.concatenate([[0], np.cumsum(finite)])
    starts = np.maximum(rows - window + 1, 0)
    counts = cumulative[rows + 1] - cumulative[starts]
    lengths = rows - starts + 1
    return counts / lengths


def _assign_touch_features_vectorized(
    output: dict[str, np.ndarray],
    *,
    rows: np.ndarray,
    b: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    close: np.ndarray,
    raw_low: np.ndarray,
    raw_close: np.ndarray,
    atr: np.ndarray,
    config: BullishDivergenceFeatureConfig,
) -> None:
    window = config.touch_lookback
    eligible = rows >= window - 1
    rows = rows[eligible]
    b = b[eligible]
    if len(rows) == 0:
        return
    level = low[b]
    denom = atr[rows]
    scale = np.divide(
        close[rows], raw_close[rows], out=np.full(len(rows), np.nan),
        where=raw_close[rows] > bf.EPS,
    )
    tick_width = config.touch_minimum_ticks * config.tick_size * np.nan_to_num(scale)
    zone_width = np.maximum(tick_width, config.touch_zone_atr_fraction * denom)
    all_low_windows = np.lib.stride_tricks.sliding_window_view(low, window)
    all_high_windows = np.lib.stride_tricks.sliding_window_view(high, window)
    low_windows = all_low_windows[rows - window + 1]
    high_windows = all_high_windows[rows - window + 1]
    start = rows - window + 1
    absolute_positions = start[:, None] + np.arange(window)[None, :]
    anchor_mask = absolute_positions == b[:, None]
    finite = np.isfinite(low_windows) & np.isfinite(high_windows) & ~anchor_mask
    touches = (
        (low_windows <= level[:, None] + zone_width[:, None])
        & (high_windows >= level[:, None] - zone_width[:, None])
        & finite
    )
    distances = np.maximum(
        np.maximum(low_windows - level[:, None], level[:, None] - high_windows), 0.0
    )
    distances[~finite] = np.nan
    counts = touches.sum(axis=1)
    occurred = counts > 0
    pre_counts = (touches & (absolute_positions < b[:, None])).sum(axis=1)
    post_counts = (touches & (absolute_positions > b[:, None])).sum(axis=1).astype(float)
    post_observable = b < rows
    post_counts[~post_observable] = np.nan

    output["touch__level_raw"][rows] = np.divide(
        level, scale, out=raw_low[b].copy(), where=np.isfinite(scale) & (scale > bf.EPS)
    )
    output["touch__level_raw_origin"][rows] = raw_low[b]
    output["touch__level_adj_pit"][rows] = level
    output["touch__zone_width_adj"][rows] = zone_width
    output["touch__level_to_close"][rows] = level / close[rows] - 1.0
    output["touch__zone_width_atr"][rows] = zone_width / denom
    output["touch__post_b_observable"][rows] = post_observable.astype(float)
    output["touch__occurred_10d"][rows] = occurred.astype(float)
    output["touch__count_10d"][rows] = counts.astype(float)
    finite_distance = np.isfinite(distances)
    nearest = np.min(np.where(finite_distance, distances, np.inf), axis=1)
    nearest[~finite_distance.any(axis=1)] = np.nan
    output["touch__nearest_distance_atr_10d"][rows] = nearest / denom
    output["touch__pre_b_count"][rows] = pre_counts.astype(float)
    output["touch__post_b_count"][rows] = post_counts

    if not occurred.any():
        return
    touched_rows = rows[occurred]
    last_relative = window - 1 - np.argmax(touches[occurred, ::-1], axis=1)
    last = start[occurred] + last_relative
    touched_level = level[occurred]
    touched_denom = denom[occurred]
    output["touch__age_days"][touched_rows] = touched_rows - last
    output["touch__last_penetration_atr"][touched_rows] = (touched_level - low[last]) / touched_denom
    output["touch__last_close_reclaim_atr"][touched_rows] = (close[last] - touched_level) / touched_denom
    output["touch__post_touch_return_to_t"][touched_rows] = close[touched_rows] / close[last] - 1.0
    output["touch__false_break_reclaim"][touched_rows] = (
        (low[last] < touched_level) & (close[last] > touched_level)
    ).astype(float)


def _attach_scores(result: pd.DataFrame, valid: pd.Series) -> pd.DataFrame:
    dates = result["trade_date"]
    double_divergence = valid & result["structure__double_divergence_present"].eq(1.0)
    structure_trends = pd.DataFrame({
        "rsi14": result["structure__rsi14_improvement_trend"],
        "macd_atr": result["structure__macd_atr_improvement_trend"],
        "velocity": result["structure__velocity_improvement_trend"],
        "volume_dryup": result["structure__volume_dryup_trend"],
        "rebound": result["structure__rebound_trend"],
        "agreement": result["structure__agreement_trend"],
    })
    observed_structure = structure_trends.notna().sum(axis=1)
    structure_q = pd.DataFrame({
        column: bf.cross_section_percentile(values.where(double_divergence), dates)
        for column, values in structure_trends.items()
    })
    result["structure__double_divergence_trend_score"] = (
        100.0 * structure_q.mean(axis=1)
    ).where(
        double_divergence & observed_structure.ge(4)
    )

    q_price = bf.cross_section_percentile(result["div__price_lower_low_atr"].where(valid), dates)
    oscillator = result[[
        "div__rsi14_higher_low", "div__macd_hist_higher_low",
        "div__downside_velocity_change",
    ]].max(axis=1)
    q_oscillator = bf.cross_section_percentile(oscillator.where(valid), dates)
    core = np.sqrt(q_price.clip(lower=0) * q_oscillator.clip(lower=0))
    confirmations = pd.concat([
        bf.cross_section_percentile(result["div__intervening_rebound_atr"].where(valid), dates),
        bf.cross_section_percentile(result["div__close_location"].where(valid), dates),
        bf.cross_section_percentile(result["div__lower_shadow_atr"].where(valid), dates),
        bf.cross_section_percentile((-result["div__down_volume_change"]).where(valid), dates),
    ], axis=1).mean(axis=1)
    result["div__score"] = 100.0 * result["div__reliability"].clip(0, 1) * (
        0.70 * core + 0.30 * confirmations
    )
    result["div__score_rank"] = bf.cross_section_percentile(result["div__score"].where(valid), dates)

    recency = np.exp(-result["touch__age_days"] / 5.0)
    touch_components = pd.concat([
        recency,
        bf.cross_section_percentile((-result["touch__nearest_distance_atr_10d"]).where(valid), dates),
        bf.cross_section_percentile(result["touch__last_close_reclaim_atr"].where(valid), dates),
        bf.cross_section_percentile(result["touch__post_touch_return_to_t"].where(valid), dates),
    ], axis=1).mean(axis=1)
    result["touch__acceptance_score"] = (
        100.0 * result["touch__occurred_10d"].fillna(0.0) * touch_components
    ).where(result["touch__occurred_10d"].notna())
    return result


def _main_sell_ratio(data: pd.DataFrame) -> pd.Series:
    direct = next(
        (name for name in ("main_sell_ratio", "sell_main_ratio", "sell_pressure_ratio") if name in data),
        None,
    )
    if direct is not None:
        return pd.to_numeric(data[direct], errors="coerce")
    sell_columns = [
        name for name in (
            "sell_lg_amount", "sell_elg_amount", "sell_lg_amount_cny", "sell_elg_amount_cny",
        ) if name in data
    ]
    if sell_columns:
        sell = sum((pd.to_numeric(data[name], errors="coerce") for name in sell_columns), start=pd.Series(0.0, index=data.index))
        amount = pd.to_numeric(data["amount_cny"], errors="coerce")
        return sell / amount.where(amount > 0)
    return pd.Series(np.nan, index=data.index, dtype=float)
