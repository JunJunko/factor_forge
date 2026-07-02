from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd


class FactorStage(StrEnum):
    SETUP = "setup"
    PRE_BREAKOUT = "pre_breakout"
    BREAKOUT = "breakout"


class BoxState(StrEnum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    CLOSED = "closed"


@dataclass(frozen=True)
class ColumnMap:
    date: str = "trade_date"
    security: str = "ts_code"
    open: str = "adj_open"
    high: str = "adj_high"
    low: str = "adj_low"
    close: str = "adj_close"
    volume: str = "volume_shares"

    def required(self) -> tuple[str, ...]:
        return (
            self.date,
            self.security,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
        )


@dataclass(frozen=True)
class BreakoutConfig:
    box_lookback: int = 40
    atr_window: int = 20
    volatility_short_window: int = 10
    volatility_long_window: int = 40
    process_window: int = 10
    acceleration_window: int = 3
    volume_window: int = 20
    max_active_days: int = 20
    max_box_width_atr: float = 8.0
    max_abs_slope_atr: float = 0.10
    max_volatility_ratio: float | None = 1.0
    breakout_buffer_atr: float = 0.0
    failure_buffer_atr: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = (
            "box_lookback",
            "atr_window",
            "volatility_short_window",
            "volatility_long_window",
            "process_window",
            "acceleration_window",
            "volume_window",
            "max_active_days",
        )
        for name in integer_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.volatility_short_window > self.volatility_long_window:
            raise ValueError("volatility_short_window cannot exceed volatility_long_window")
        if self.process_window < 2:
            raise ValueError("process_window must be at least 2")
        if self.max_box_width_atr <= 0:
            raise ValueError("max_box_width_atr must be positive")
        if self.max_abs_slope_atr < 0:
            raise ValueError("max_abs_slope_atr cannot be negative")
        if self.max_volatility_ratio is not None and self.max_volatility_ratio < 0:
            raise ValueError("max_volatility_ratio cannot be negative")
        if self.breakout_buffer_atr < 0 or self.failure_buffer_atr < 0:
            raise ValueError("breakout and failure buffers cannot be negative")


@dataclass
class ActiveBox:
    box_id: str
    security: str
    upper: float
    lower: float
    frozen_atr: float
    source_start: pd.Timestamp
    source_end: pd.Timestamp
    created_at: pd.Timestamp
    created_position: int
    setup_features: dict[str, float] = field(default_factory=dict)
    state: BoxState = BoxState.ACTIVE
    closed_at: pd.Timestamp | None = None
    close_reason: str | None = None


@dataclass(frozen=True)
class OperatorContext:
    history: pd.DataFrame
    box: ActiveBox
    columns: ColumnMap
    config: BreakoutConfig
    as_of: pd.Timestamp
    current: pd.Series | None = None
    box_age: int = 0


@dataclass(frozen=True)
class BreakoutRunResult:
    boxes: pd.DataFrame
    daily_features: pd.DataFrame
    events: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "boxes": self.boxes,
            "daily_features": self.daily_features,
            "events": self.events,
        }


JsonScalar = str | int | float | bool | None | pd.Timestamp
Record = dict[str, Any]
