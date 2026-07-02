from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import FactorStage, OperatorContext


def _finite(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _slope(values: np.ndarray) -> float:
    if len(values) < 2 or not np.isfinite(values).all():
        return np.nan
    x = np.arange(len(values), dtype=float)
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    return float(np.dot(values - values.mean(), centered) / denominator)


def _normalized_closes(context: OperatorContext) -> np.ndarray:
    closes = _finite(context.history[context.columns.close])
    return (closes - context.box.upper) / context.box.frozen_atr


def range_compactness(context: OperatorContext) -> float:
    return -(context.box.upper - context.box.lower) / context.box.frozen_atr


def volatility_contraction(context: OperatorContext) -> float:
    closes = _finite(context.history[context.columns.close])
    returns = pd.Series(closes).pct_change().dropna().to_numpy(dtype=float)
    short = context.config.volatility_short_window
    long = context.config.volatility_long_window
    if len(returns) < long:
        return np.nan
    short_vol = float(np.std(returns[-short:], ddof=0))
    long_vol = float(np.std(returns[-long:], ddof=0))
    if long_vol == 0:
        return 0.0 if short_vol == 0 else -np.inf
    ratio = short_vol / long_vol
    if ratio == 0:
        return np.inf
    return float(-np.log(ratio))


def trend_flatness(context: OperatorContext) -> float:
    closes = _finite(context.history[context.columns.close])
    closes = closes[-context.config.box_lookback :]
    return -abs(_slope(closes)) / context.box.frozen_atr


def approach_velocity(context: OperatorContext) -> float:
    values = _normalized_closes(context)
    window = context.config.process_window
    if len(values) < window:
        return np.nan
    return _slope(values[-window:])


def pre_acceleration(context: OperatorContext) -> float:
    values = _normalized_closes(context)
    window = context.config.acceleration_window
    if len(values) < 2 * window + 1:
        return np.nan
    differences = np.diff(values[-(2 * window + 1) :])
    early_velocity = float(np.mean(differences[:window]))
    recent_velocity = float(np.mean(differences[window:]))
    return (recent_velocity - early_velocity) / window


def direction_persistence(context: OperatorContext) -> float:
    values = _normalized_closes(context)
    window = context.config.process_window
    if len(values) < window + 1:
        return np.nan
    differences = np.diff(values[-(window + 1) :])
    return float(np.mean(differences > 0))


def consolidation_age(context: OperatorContext) -> float:
    return float(context.config.box_lookback + context.box_age)


def breakout_strength(context: OperatorContext) -> float:
    if context.current is None:
        return np.nan
    close = float(context.current[context.columns.close])
    return (close - context.box.upper) / context.box.frozen_atr


def breakout_velocity(context: OperatorContext) -> float:
    if context.current is None or context.history.empty:
        return np.nan
    close = float(context.current[context.columns.close])
    previous_close = float(context.history.iloc[-1][context.columns.close])
    return (close - previous_close) / context.box.frozen_atr


def breakout_acceleration(context: OperatorContext) -> float:
    if context.current is None:
        return np.nan
    values = _normalized_closes(context)
    window = context.config.acceleration_window
    if len(values) < window + 1:
        return np.nan
    recent_velocity = float(np.mean(np.diff(values[-(window + 1) :])))
    return breakout_velocity(context) - recent_velocity


def relative_volume(context: OperatorContext) -> float:
    if context.current is None:
        return np.nan
    history = _finite(context.history[context.columns.volume])
    window = context.config.volume_window
    if len(history) < window:
        return np.nan
    baseline = float(np.nanmedian(history[-window:]))
    current = float(context.current[context.columns.volume])
    if not np.isfinite(baseline) or baseline <= 0 or current <= 0:
        return np.nan
    return float(np.log(current / baseline))


def gap_atr(context: OperatorContext) -> float:
    if context.current is None or context.history.empty:
        return np.nan
    open_price = float(context.current[context.columns.open])
    previous_close = float(context.history.iloc[-1][context.columns.close])
    return (open_price - previous_close) / context.box.frozen_atr


@dataclass(frozen=True)
class FactorOperator:
    name: str
    stage: FactorStage
    function: Callable[[OperatorContext], float]
    direction: str = "unsigned"

    def compute(self, context: OperatorContext) -> float:
        value = self.function(context)
        return float(value) if value is not None else np.nan


class OperatorRegistry:
    """A small injectable registry; it deliberately has no dependency on FactorEngine."""

    def __init__(self, operators: Iterable[FactorOperator] = ()) -> None:
        self._operators: dict[str, FactorOperator] = {}
        for operator in operators:
            self.register(operator)

    def register(self, operator: FactorOperator, *, replace: bool = False) -> None:
        if operator.name in self._operators and not replace:
            raise ValueError(f"operator already registered: {operator.name}")
        self._operators[operator.name] = operator

    def for_stage(self, stage: FactorStage) -> tuple[FactorOperator, ...]:
        return tuple(op for op in self._operators.values() if op.stage == stage)

    def compute(self, stage: FactorStage, context: OperatorContext) -> dict[str, float]:
        return {op.name: op.compute(context) for op in self.for_stage(stage)}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._operators)


def default_operator_registry() -> OperatorRegistry:
    return OperatorRegistry(
        (
            FactorOperator("range_compactness", FactorStage.SETUP, range_compactness, "positive"),
            FactorOperator(
                "volatility_contraction",
                FactorStage.SETUP,
                volatility_contraction,
                "positive",
            ),
            FactorOperator("trend_flatness", FactorStage.SETUP, trend_flatness, "positive"),
            FactorOperator(
                "approach_velocity",
                FactorStage.PRE_BREAKOUT,
                approach_velocity,
                "positive",
            ),
            FactorOperator(
                "pre_acceleration",
                FactorStage.PRE_BREAKOUT,
                pre_acceleration,
                "positive",
            ),
            FactorOperator(
                "direction_persistence",
                FactorStage.PRE_BREAKOUT,
                direction_persistence,
                "positive",
            ),
            FactorOperator(
                "consolidation_age",
                FactorStage.PRE_BREAKOUT,
                consolidation_age,
                "unsigned",
            ),
            FactorOperator(
                "breakout_strength",
                FactorStage.BREAKOUT,
                breakout_strength,
                "positive",
            ),
            FactorOperator(
                "breakout_velocity",
                FactorStage.BREAKOUT,
                breakout_velocity,
                "positive",
            ),
            FactorOperator(
                "breakout_acceleration",
                FactorStage.BREAKOUT,
                breakout_acceleration,
                "positive",
            ),
            FactorOperator(
                "relative_volume",
                FactorStage.BREAKOUT,
                relative_volume,
                "positive",
            ),
            FactorOperator("gap_atr", FactorStage.BREAKOUT, gap_atr, "unsigned"),
        )
    )
