"""Configuration for causal bullish-divergence and support-touch features."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BullishDivergenceFeatureConfig:
    current_trough_window: int = 5
    previous_trough_lookback: int = 60
    minimum_trough_separation: int = 5
    maximum_current_trough_age: int = 3
    minimum_intervening_rebound_atr: float = 0.5
    lower_low_tolerance_atr: float = 0.25
    maximum_lower_low_atr: float = 1.0
    atr_window: int = 20
    rsi_fast_window: int = 6
    rsi_slow_window: int = 14
    touch_lookback: int = 10
    touch_zone_atr_fraction: float = 0.15
    touch_minimum_ticks: int = 2
    tick_size: float = 0.01
    minimum_history_valid_ratio: float = 0.90
    minimum_listing_days: int = 60
    episode_cooldown_days: int = 10

    def __post_init__(self) -> None:
        if self.current_trough_window < 2:
            raise ValueError("current_trough_window must be at least 2")
        if self.previous_trough_lookback <= self.minimum_trough_separation:
            raise ValueError("previous_trough_lookback must exceed minimum_trough_separation")
        if self.maximum_lower_low_atr <= -self.lower_low_tolerance_atr:
            raise ValueError("lower-low eligibility band is invalid")
        if not 0 <= self.maximum_current_trough_age < self.current_trough_window:
            raise ValueError("maximum_current_trough_age must be inside the current trough window")
        if self.atr_window < 2 or self.rsi_fast_window < 2 or self.rsi_slow_window < 2:
            raise ValueError("indicator windows must be at least 2")
        if self.touch_lookback < 2:
            raise ValueError("touch_lookback must be at least 2")
        if self.touch_zone_atr_fraction < 0 or self.touch_minimum_ticks < 0:
            raise ValueError("touch-zone widths cannot be negative")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if not 0 < self.minimum_history_valid_ratio <= 1:
            raise ValueError("minimum_history_valid_ratio must be in (0, 1]")
