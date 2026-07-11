"""Configuration for ATR lower-shadow reversion research."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from .config import ModelConfig, Segment, Segments, StrictModel


class ATRReversionFeatureConfig(StrictModel):
    atr_window: int = Field(default=20, ge=2)
    ma_window: int = Field(default=20, ge=2)
    long_ma_window: int = Field(default=60, ge=2)
    percentile_window: int = Field(default=120, ge=20)
    amount_window: int = Field(default=20, ge=2)
    amount_shock_window: int = Field(default=20, ge=2)
    market_window: int = Field(default=5, ge=1)
    bollinger_std: float = Field(default=2.0, gt=0)
    event_pool_threshold: float = Field(default=-0.5)
    shape_clip: float = Field(default=3.0, gt=0)
    velocity_window: int = Field(default=5, ge=2)
    acceleration_window: int = Field(default=5, ge=2)
    natr_ema_span: int = Field(default=3, ge=1)
    net_flow_column: str | None = None
    min_listing_days: int = Field(default=60, ge=1)
    winsor_quantile: float = Field(default=0.01, ge=0, lt=0.5)
    cross_sectional_zscore: bool = True
    use_sample_weight: bool = True
    extreme_vol_quantile: float = Field(default=0.95, gt=0.5, lt=1.0)


class ATRReversionLabelConfig(StrictModel):
    horizons: list[int] = Field(default_factory=lambda: [3, 5, 10])
    primary_horizon: int = Field(default=5, ge=1, le=60)
    label_method: Literal["open_to_open", "open_to_close"] = "open_to_open"
    industry_neutralize: bool = True
    cross_sectional_rank_label: bool = True


class ATRReversionPipelineConfig(StrictModel):
    version: int = 1
    name: str = "atr_lower_shadow_reversion_v1"
    project_config: Path = Path("configs/project.yaml")
    data_version: str = "latest"
    require_full_segment_coverage: bool = True
    segments: Segments
    features: ATRReversionFeatureConfig = Field(default_factory=ATRReversionFeatureConfig)
    label: ATRReversionLabelConfig = Field(default_factory=ATRReversionLabelConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    output_root: Path = Path("artifacts/atr_reversion_runs")
    universe_top_n: int | None = 1000
    cache_dataset: bool = True
    random_seeds: list[int] = Field(default_factory=lambda: [42, 2026, 3407, 8888, 10086])
    model_variants: list[str] | None = None


def load_atr_reversion_config(path: str | Path) -> ATRReversionPipelineConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return ATRReversionPipelineConfig.model_validate(yaml.safe_load(handle) or {})
