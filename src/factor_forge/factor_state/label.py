from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd


class FactorState(IntEnum):
    Healthy = 0
    Weakening = 1
    Broken = 2
    Recovery = 3


STATE_NAMES = {state.value: state.name for state in FactorState}


def build_factor_state_labels(
    health: pd.DataFrame,
    *,
    forward_window: int = 60,
    spread_threshold: float = 0.0,
    monotonicity_threshold: float = 0.0,
    past_broken_window: int = 20,
) -> pd.DataFrame:
    """Create factor lifecycle state labels from future factor-health outcomes.

    Future spread/IC/monotonicity are label targets only. They must not be used as
    model input features.
    """
    required = {"date", "factor_name", "rank_ic", "decile_spread", "decile_monotonicity"}
    missing = required - set(health.columns)
    if missing:
        raise ValueError(f"missing required health columns: {sorted(missing)}")
    frames = []
    for factor_name, group in health.groupby("factor_name", sort=False):
        g = group.sort_values("date").copy()
        g["future_spread_60"] = _future_mean(g["decile_spread"], forward_window)
        g["future_rank_ic_60"] = _future_mean(g["rank_ic"], forward_window)
        g["future_monotonicity_60"] = _future_mean(g["decile_monotonicity"], forward_window)
        g["past_broken"] = (
            (g["decile_spread"].le(spread_threshold) | g["rank_ic"].le(0.0))
            .shift(1)
            .rolling(past_broken_window, min_periods=1)
            .max()
            .fillna(0)
            .astype(bool)
        )
        g["state"] = _assign_state(g, spread_threshold, monotonicity_threshold)
        g["state_name"] = g["state"].map(STATE_NAMES)
        frames.append(g)
    return pd.concat(frames, ignore_index=True).sort_values(["date", "factor_name"]).reset_index(drop=True)


def _future_mean(series: pd.Series, window: int) -> pd.Series:
    return series.shift(-1).rolling(window, min_periods=max(5, window // 2)).mean().shift(-(window - 1))


def _assign_state(frame: pd.DataFrame, spread_threshold: float, monotonicity_threshold: float) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float")
    future_good = frame["future_spread_60"].gt(spread_threshold) & frame["future_rank_ic_60"].gt(0.0)
    future_broken = frame["future_spread_60"].le(spread_threshold) | frame["future_rank_ic_60"].le(0.0)
    velocity_down = frame.get("spread_velocity_20_60", pd.Series(index=frame.index, dtype=float)).lt(0.0) | frame.get(
        "ic_velocity_20_60", pd.Series(index=frame.index, dtype=float)
    ).lt(0.0)
    velocity_up = frame.get("spread_velocity_20_60", pd.Series(index=frame.index, dtype=float)).gt(0.0) & frame.get(
        "ic_velocity_20_60", pd.Series(index=frame.index, dtype=float)
    ).gt(0.0)
    recovery = frame["past_broken"] & velocity_up & future_good
    healthy = future_good & frame["future_monotonicity_60"].gt(monotonicity_threshold)
    weakening = future_good & velocity_down

    out.loc[future_broken] = FactorState.Broken.value
    out.loc[healthy] = FactorState.Healthy.value
    out.loc[weakening] = FactorState.Weakening.value
    out.loc[recovery] = FactorState.Recovery.value
    return out
