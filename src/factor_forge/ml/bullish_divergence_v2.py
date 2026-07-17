"""Mechanism-aligned v2 bullish-divergence features and event clocks.

This revision is intentionally separate from v1.  It treats price geometry as
an eligibility condition, scores oscillator improvement only after each
component has been put on a comparable cross-sectional scale, and separates
historical support clustering from a future post-signal retest event.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .bullish_divergence_config import BullishDivergenceFeatureConfig
from .bullish_divergence_dataset import (
    build_bullish_divergence_features,
    build_divergence_episodes,
)
from .bullish_divergence_features import cross_section_percentile


V2_FEATURES = [
    "div_v2__q_rsi14_improvement",
    "div_v2__q_macd_atr_improvement",
    "div_v2__q_velocity_improvement",
    "div_v2__oscillator_strength",
    "div_v2__confirmation_strength",
    "div_v2__score",
    "div_v2__score_rank",
    "div_v2__geometry_candidate",
    "div_v2__event_candidate",
    "support_v2__pre_b_present",
    "support_v2__pre_b_count",
]


@dataclass(frozen=True)
class BullishDivergenceV2Config:
    minimum_trough_separation: int = 20
    previous_trough_lookback: int = 60
    minimum_lower_low_atr: float = -0.25
    maximum_lower_low_atr: float = 1.00
    minimum_intervening_rebound_atr: float = 0.50
    require_decline_into_both_troughs: bool = True
    oscillator_weight: float = 0.75
    confirmation_weight: float = 0.25
    post_signal_retest_horizon_days: int = 10

    def __post_init__(self) -> None:
        if self.minimum_trough_separation < 2:
            raise ValueError("minimum_trough_separation must be at least 2")
        if self.previous_trough_lookback <= self.minimum_trough_separation:
            raise ValueError("previous_trough_lookback must exceed the minimum separation")
        if self.maximum_lower_low_atr <= self.minimum_lower_low_atr:
            raise ValueError("lower-low eligibility band is invalid")
        if self.minimum_intervening_rebound_atr <= 0:
            raise ValueError("minimum_intervening_rebound_atr must be positive")
        if not np.isclose(self.oscillator_weight + self.confirmation_weight, 1.0):
            raise ValueError("v2 score weights must sum to one")
        if self.post_signal_retest_horizon_days < 1:
            raise ValueError("post_signal_retest_horizon_days must be positive")


def v2_base_feature_config(
    config: BullishDivergenceV2Config = BullishDivergenceV2Config(),
) -> BullishDivergenceFeatureConfig:
    return replace(
        BullishDivergenceFeatureConfig(),
        previous_trough_lookback=config.previous_trough_lookback,
        minimum_trough_separation=config.minimum_trough_separation,
        minimum_intervening_rebound_atr=config.minimum_intervening_rebound_atr,
        lower_low_tolerance_atr=-config.minimum_lower_low_atr,
        maximum_lower_low_atr=config.maximum_lower_low_atr,
    )


def build_bullish_divergence_v2_features(
    panel: pd.DataFrame,
    config: BullishDivergenceV2Config = BullishDivergenceV2Config(),
) -> tuple[pd.DataFrame, list[str]]:
    """Build the strict 20-60 day v2 event pool and orthogonal mechanism score."""
    base_config = v2_base_feature_config(config)
    result, base_features = build_bullish_divergence_features(panel, base_config)
    if result.empty:
        return result, [*base_features, *V2_FEATURES]

    valid = (
        result["is_tradeable"].fillna(False)
        & ~result["is_suspended"].fillna(True)
        & ~result["is_st"].fillna(False)
        & ~result["is_delisting_period"].fillna(False)
        & result["listing_trade_days"].ge(base_config.minimum_listing_days)
    )
    dates = result["trade_date"]
    q_rsi = cross_section_percentile(
        result["div__rsi14_higher_low"].where(valid), dates
    )
    q_macd = cross_section_percentile(
        result["div__macd_hist_higher_low_atr"].where(valid), dates
    )
    q_velocity = cross_section_percentile(
        result["div__downside_velocity_change"].where(valid), dates
    )
    oscillator_strength = pd.concat(
        [q_rsi, q_macd, q_velocity], axis=1
    ).max(axis=1)
    confirmation_strength = pd.concat([
        cross_section_percentile(result["div__close_location"].where(valid), dates),
        cross_section_percentile(result["div__lower_shadow_atr"].where(valid), dates),
        cross_section_percentile((-result["div__down_volume_change"]).where(valid), dates),
    ], axis=1).mean(axis=1)

    geometry = (
        valid
        & result["div__b_age"].le(base_config.maximum_current_trough_age)
        & result["div__trough_gap_days"].between(
            config.minimum_trough_separation, config.previous_trough_lookback
        )
        & result["div__price_lower_low_atr"].between(
            config.minimum_lower_low_atr, config.maximum_lower_low_atr
        )
        & result["div__intervening_rebound_atr"].ge(
            config.minimum_intervening_rebound_atr
        )
        & result["div__history_valid_ratio"].ge(base_config.minimum_history_valid_ratio)
    )
    if config.require_decline_into_both_troughs:
        geometry &= (
            result["div__descent_into_a_3d"].lt(0)
            & result["div__descent_into_b_3d"].lt(0)
        )
    oscillator_positive = (
        result["div__rsi14_higher_low"].gt(0)
        | result["div__macd_hist_higher_low_atr"].gt(0)
        | result["div__downside_velocity_change"].gt(0)
    )
    score = 100.0 * (
        config.oscillator_weight * oscillator_strength
        + config.confirmation_weight * confirmation_strength
    )

    result["div_v2__q_rsi14_improvement"] = q_rsi
    result["div_v2__q_macd_atr_improvement"] = q_macd
    result["div_v2__q_velocity_improvement"] = q_velocity
    result["div_v2__oscillator_strength"] = oscillator_strength
    result["div_v2__confirmation_strength"] = confirmation_strength
    result["div_v2__score"] = score.where(geometry)
    result["div_v2__score_rank"] = cross_section_percentile(
        result["div_v2__score"].where(geometry), dates
    )
    result["div_v2__geometry_candidate"] = geometry
    result["div_v2__event_candidate"] = geometry & oscillator_positive
    result["support_v2__pre_b_present"] = result["touch__pre_b_count"].fillna(0).gt(0)
    result["support_v2__pre_b_count"] = result["touch__pre_b_count"]
    return result, [*base_features, *V2_FEATURES]


def build_v2_divergence_episodes(
    features: pd.DataFrame,
    config: BullishDivergenceV2Config = BullishDivergenceV2Config(),
) -> pd.DataFrame:
    return build_divergence_episodes(
        features,
        v2_base_feature_config(config),
        candidate_field="div_v2__event_candidate",
        event_prefix="v2:",
    )


def build_v2_post_signal_retest_events(
    panel: pd.DataFrame,
    origin_episodes: pd.DataFrame,
    config: BullishDivergenceV2Config = BullishDivergenceV2Config(),
) -> pd.DataFrame:
    """Create a new signal clock at the first anchor touch after the origin signal.

    The search uses future rows only to locate the future event date.  Every
    retest event is evaluated from that event's close and traded from its next
    open, so no future data enters the retest-day predictor.
    """
    required_panel = {
        "trade_date", "ts_code", "adj_high", "adj_low", "adj_close",
        "raw_close",
    }
    required_events = {
        "event_id", "trade_date", "ts_code", "touch__level_adj_pit",
        "touch__zone_width_adj", "div_v2__score", "div_v2__score_rank",
    }
    if missing := required_panel - set(panel.columns):
        raise ValueError(f"panel is missing post-retest fields: {sorted(missing)}")
    if missing := required_events - set(origin_episodes.columns):
        raise ValueError(f"origin episodes are missing post-retest fields: {sorted(missing)}")

    prices = panel[list(required_panel)].copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices = prices.sort_values(["ts_code", "trade_date"])
    events = origin_episodes.copy()
    events["trade_date"] = pd.to_datetime(events["trade_date"])
    outputs: list[dict] = []
    price_groups = prices.groupby("ts_code", sort=False, observed=True)
    for code, stock_events in events.groupby("ts_code", sort=False, observed=True):
        if code not in price_groups.groups:
            continue
        stock = price_groups.get_group(code).reset_index(drop=True)
        ordinal = pd.Series(stock.index.to_numpy(), index=stock["trade_date"]).to_dict()
        high = pd.to_numeric(stock["adj_high"], errors="coerce").to_numpy(float)
        low = pd.to_numeric(stock["adj_low"], errors="coerce").to_numpy(float)
        close = pd.to_numeric(stock["adj_close"], errors="coerce").to_numpy(float)
        raw_close = pd.to_numeric(stock["raw_close"], errors="coerce").to_numpy(float)
        dates = stock["trade_date"].to_numpy()
        for event in stock_events.itertuples(index=False):
            start = ordinal.get(pd.Timestamp(event.trade_date))
            level = float(event.touch__level_adj_pit)
            zone = float(event.touch__zone_width_adj)
            if start is None or not np.isfinite(level) or not np.isfinite(zone):
                continue
            stop = min(start + config.post_signal_retest_horizon_days + 1, len(stock))
            positions = np.arange(start + 1, stop)
            if not len(positions):
                continue
            touched = (
                np.isfinite(low[positions]) & np.isfinite(high[positions])
                & (low[positions] <= level + zone)
                & (high[positions] >= level - zone)
            )
            if not touched.any():
                continue
            trigger = int(positions[np.flatnonzero(touched)[0]])
            reclaimed = bool(close[trigger] > level)
            false_break = bool(low[trigger] < level and reclaimed)
            scale = close[trigger] / raw_close[trigger] if raw_close[trigger] > 0 else np.nan
            outputs.append({
                "event_id": f"v2r:{code}:{pd.Timestamp(dates[trigger]):%Y%m%d}",
                "episode_id": event.event_id,
                "origin_event_id": event.event_id,
                "origin_trade_date": pd.Timestamp(event.trade_date),
                "trade_date": pd.Timestamp(dates[trigger]),
                "ts_code": code,
                "div_v2__score": event.div_v2__score,
                "div_v2__score_rank": event.div_v2__score_rank,
                "retest_v2__age_days": trigger - start,
                "retest_v2__level_adj": level,
                "retest_v2__level_raw": level / scale if np.isfinite(scale) and scale > 0 else np.nan,
                "retest_v2__zone_width_adj": zone,
                "retest_v2__penetration_atr_zone": (level - low[trigger]) / zone if zone > 0 else np.nan,
                "retest_v2__close_reclaim_zone": (close[trigger] - level) / zone if zone > 0 else np.nan,
                "retest_v2__reclaimed": reclaimed,
                "retest_v2__false_break_reclaim": false_break,
            })
    return pd.DataFrame(outputs)
