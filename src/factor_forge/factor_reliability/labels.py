from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReliabilityLabelConfig:
    horizons: tuple[int, ...] = (5, 10, 20)
    cost_buffer: float = 0.002
    min_future_observations: int = 3


def build_reliability_labels(
    factor_health_daily: pd.DataFrame,
    config: ReliabilityLabelConfig | None = None,
) -> pd.DataFrame:
    """Build short-horizon validity labels from future factor payoff.

    Future spread/IC columns are target diagnostics only and must not be used as
    model input features.
    """
    cfg = config or ReliabilityLabelConfig()
    required = {"date", "factor_name", "rank_ic", "decile_spread"}
    missing = required - set(factor_health_daily.columns)
    if missing:
        raise ValueError(f"missing required factor health columns: {sorted(missing)}")
    health = factor_health_daily.copy()
    health["date"] = pd.to_datetime(health["date"])
    frames = []
    for factor_name, group in health.groupby("factor_name", sort=False):
        g = group.sort_values("date")[["date", "factor_name", "rank_ic", "decile_spread"]].copy()
        for horizon in cfg.horizons:
            min_obs = min(horizon, max(cfg.min_future_observations, horizon // 2))
            spread_col = f"future_top_bottom_spread_{horizon}d"
            ic_col = f"future_rank_ic_{horizon}d"
            label_col = f"validity_label_{horizon}d"
            g[spread_col] = _future_mean(g["decile_spread"], horizon, min_obs)
            g[ic_col] = _future_mean(g["rank_ic"], horizon, min_obs)
            valid = g[spread_col].gt(cfg.cost_buffer) & g[ic_col].gt(0.0)
            has_target = g[spread_col].notna() & g[ic_col].notna()
            g[label_col] = np.where(has_target, valid.astype(int), np.nan)
        frames.append(g.drop(columns=["rank_ic", "decile_spread"]))
    return pd.concat(frames, ignore_index=True).sort_values(["date", "factor_name"]).reset_index(drop=True)


def reliability_label_columns(horizons: tuple[int, ...] = (5, 10, 20)) -> list[str]:
    return [f"validity_label_{horizon}d" for horizon in horizons]


def future_diagnostic_columns(horizons: tuple[int, ...] = (5, 10, 20)) -> list[str]:
    cols: list[str] = []
    for horizon in horizons:
        cols.extend([f"future_top_bottom_spread_{horizon}d", f"future_rank_ic_{horizon}d"])
    return cols


def _future_mean(series: pd.Series, horizon: int, min_obs: int) -> pd.Series:
    return series.shift(-1).rolling(horizon, min_periods=min_obs).mean().shift(-(horizon - 1))
