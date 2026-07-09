from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


REGIME_COLUMNS = [
    "market_ret_20",
    "market_ret_60",
    "market_vol_20",
    "market_breadth_20",
    "market_xsec_vol_20",
]


@dataclass(frozen=True)
class ReliabilityFeatureConfig:
    cost_buffer: float = 0.002
    min_periods_short: int = 5
    min_periods_medium: int = 10
    min_periods_long: int = 30
    peak_window: int = 120
    regime_lookback: int = 252
    regime_effective_quantile: float = 0.70
    regime_columns: tuple[str, ...] = field(default_factory=lambda: tuple(REGIME_COLUMNS))


def build_reliability_features(
    factor_health_daily: pd.DataFrame,
    config: ReliabilityFeatureConfig | None = None,
) -> pd.DataFrame:
    """Build short-horizon factor reliability features using only current/past data."""
    cfg = config or ReliabilityFeatureConfig()
    health = _normalize_health(factor_health_daily)
    frames = []
    for factor_name, group in health.groupby("factor_name", sort=False):
        g = group.sort_values("date").copy()
        g = _add_ic_features(g, cfg)
        g = _add_spread_features(g, cfg)
        g = _add_decay_features(g, cfg)
        g = _add_regime_compatibility(g, cfg)
        frames.append(g)
    return pd.concat(frames, ignore_index=True).sort_values(["date", "factor_name"]).reset_index(drop=True)


def reliability_feature_columns(frame: pd.DataFrame | None = None, max_features: int = 40) -> list[str]:
    cols = [
        "rolling_rank_ic_10",
        "rolling_rank_ic_20",
        "rolling_rank_ic_60",
        "ic_std_20",
        "icir_20",
        "ic_velocity_10_20",
        "ic_velocity_20_60",
        "spread_10",
        "spread_20",
        "spread_60",
        "spread_velocity_10_20",
        "spread_velocity_20_60",
        "spread_acceleration",
        "decile_monotonicity",
        "score_dispersion",
        "top_bottom_score_gap",
        "factor_peak_distance",
        "factor_age",
        "days_since_last_failure",
        "top_decile_turnover",
        "holding_overlap",
        "top_stock_concentration",
        "regime_compatibility",
        "market_ret_20",
        "market_ret_60",
        "market_vol_20",
        "market_breadth_20",
        "market_xsec_vol_20",
    ]
    if frame is not None:
        cols = [col for col in cols if col in frame.columns]
    return cols[:max_features]


def _normalize_health(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "factor_name", "rank_ic", "decile_spread"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required factor health columns: {sorted(missing)}")
    health = frame.copy()
    health["date"] = pd.to_datetime(health["date"])
    for col in health.columns:
        if col not in {"date", "factor_name"}:
            health[col] = pd.to_numeric(health[col], errors="coerce")
    return health


def _add_ic_features(frame: pd.DataFrame, cfg: ReliabilityFeatureConfig) -> pd.DataFrame:
    g = frame.copy()
    g["rolling_rank_ic_10"] = g["rank_ic"].rolling(10, min_periods=cfg.min_periods_short).mean()
    g["rolling_rank_ic_20"] = g["rank_ic"].rolling(20, min_periods=cfg.min_periods_medium).mean()
    g["rolling_rank_ic_60"] = g["rank_ic"].rolling(60, min_periods=cfg.min_periods_long).mean()
    g["ic_std_20"] = g["rank_ic"].rolling(20, min_periods=cfg.min_periods_medium).std(ddof=1)
    g["icir_20"] = g["rolling_rank_ic_20"] / g["ic_std_20"].replace(0, np.nan)
    g["ic_velocity_10_20"] = g["rolling_rank_ic_10"] - g["rolling_rank_ic_20"]
    g["ic_velocity_20_60"] = g["rolling_rank_ic_20"] - g["rolling_rank_ic_60"]
    return g


def _add_spread_features(frame: pd.DataFrame, cfg: ReliabilityFeatureConfig) -> pd.DataFrame:
    g = frame.copy()
    g["spread_10"] = g["decile_spread"].rolling(10, min_periods=cfg.min_periods_short).mean()
    g["spread_20"] = g["decile_spread"].rolling(20, min_periods=cfg.min_periods_medium).mean()
    g["spread_60"] = g["decile_spread"].rolling(60, min_periods=cfg.min_periods_long).mean()
    g["spread_velocity_10_20"] = g["spread_10"] - g["spread_20"]
    g["spread_velocity_20_60"] = g["spread_20"] - g["spread_60"]
    g["spread_acceleration"] = g["spread_velocity_10_20"] - g["spread_velocity_20_60"]
    return g


def _add_decay_features(frame: pd.DataFrame, cfg: ReliabilityFeatureConfig) -> pd.DataFrame:
    g = frame.copy()
    rolling_peak = g["spread_20"].rolling(cfg.peak_window, min_periods=cfg.min_periods_long).max()
    denominator = rolling_peak.abs().replace(0, np.nan)
    g["factor_peak_distance"] = (rolling_peak - g["spread_20"]) / denominator
    effective_today = g["decile_spread"].gt(cfg.cost_buffer) & g["rank_ic"].gt(0.0)
    g["factor_age"] = _consecutive_true_count(effective_today)
    g["days_since_last_failure"] = _days_since_failure(~effective_today)
    return g


def _add_regime_compatibility(frame: pd.DataFrame, cfg: ReliabilityFeatureConfig) -> pd.DataFrame:
    g = frame.copy()
    available = [col for col in cfg.regime_columns if col in g.columns]
    g["regime_compatibility"] = np.nan
    if not available:
        return g
    for idx in range(len(g)):
        start = max(0, idx - cfg.regime_lookback)
        history = g.iloc[start:idx].copy()
        current = g.iloc[idx]
        if len(history) < cfg.min_periods_long:
            continue
        threshold = history["decile_spread"].quantile(cfg.regime_effective_quantile)
        effective = history.loc[history["decile_spread"].ge(threshold)]
        if len(effective) < cfg.min_periods_short:
            continue
        means = effective[available].mean()
        stds = history[available].std(ddof=1).replace(0, np.nan)
        current_values = pd.to_numeric(current[available], errors="coerce")
        z_distance = ((current_values - means).abs() / stds)
        z_distance = z_distance.where(np.isfinite(z_distance), np.nan)
        distance = float(z_distance.mean(skipna=True))
        g.iat[idx, g.columns.get_loc("regime_compatibility")] = np.exp(-distance) if np.isfinite(distance) else np.nan
    return g


def _consecutive_true_count(mask: pd.Series) -> pd.Series:
    values = []
    count = 0
    for item in mask.fillna(False).astype(bool):
        count = count + 1 if item else 0
        values.append(count)
    return pd.Series(values, index=mask.index, dtype=float)


def _days_since_failure(failure: pd.Series) -> pd.Series:
    values = []
    count = np.nan
    for item in failure.fillna(False).astype(bool):
        if item:
            count = 0
        elif np.isnan(count):
            count = np.nan
        else:
            count += 1
        values.append(count)
    return pd.Series(values, index=failure.index, dtype=float)
