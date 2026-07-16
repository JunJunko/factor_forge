from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from .config import ModelConfig, Segments, StrictModel


class PostImpulseFeatureConfig(StrictModel):
    """Point-in-time feature contract for one post-impulse event snapshot."""

    history_window: int = Field(default=252, ge=5, le=1260)
    min_history: int = Field(default=120, ge=3, le=1260)
    atr_window: int = Field(default=20, ge=3, le=120)
    beta_window: int = Field(default=60, ge=10, le=252)
    impulse_return_window: int = Field(default=3, ge=2, le=20)
    impulse_percentile: float = Field(default=0.90, gt=0.5, lt=1.0)
    industry_percentile: float = Field(default=0.80, gt=0.5, lt=1.0)
    observation_days: int = Field(default=3, ge=2, le=10)
    event_cooldown_days: int = Field(default=5, ge=1, le=30)
    pressure_threshold: float = Field(default=0.60, gt=0.0, lt=1.0)
    pressure_min_components: int = Field(default=2, ge=1, le=4)
    min_listing_days: int = Field(default=60, ge=1)
    market_window: int = Field(default=20, ge=5, le=120)
    industry_breadth_window: int = Field(default=5, ge=2, le=60)

    @model_validator(mode="after")
    def valid_history(self):
        if self.min_history > self.history_window:
            raise ValueError("features.min_history cannot exceed history_window")
        return self


class PostImpulseLabelConfig(StrictModel):
    horizon: Literal[10] = 10
    mfe_atr_threshold: float = Field(default=1.5, gt=0)
    mae_atr_limit: float = Field(default=1.0, gt=0)
    quality_mae_penalty: float = Field(default=0.5, ge=0)
    breakout_buffer_atr: float = Field(default=0.0, ge=0)


class PostImpulseTrainingConfig(StrictModel):
    sample_scope: Literal["pressure_events", "all_events"] = "pressure_events"
    regression_models: list[Literal["ridge", "lightgbm"]] = Field(
        default_factory=lambda: ["ridge"]
    )
    run_success_classifier: bool = True
    ridge_alpha: float = Field(default=1000.0, gt=0)
    logistic_c: float = Field(default=1.0, gt=0)
    minimum_train_events: int = Field(default=300, ge=20)
    minimum_daily_events: int = Field(default=5, ge=3)
    arms: list[Literal["m0", "m1", "m2", "m3", "m4", "m5"]] = Field(
        default_factory=lambda: ["m0", "m1", "m2", "m3", "m4", "m5"]
    )
    lightgbm: ModelConfig = Field(default_factory=ModelConfig)

    @model_validator(mode="after")
    def unique_entries(self):
        if len(set(self.regression_models)) != len(self.regression_models):
            raise ValueError("training.regression_models must be unique")
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("training.arms must be unique")
        return self


class PostImpulseMLConfig(StrictModel):
    version: Literal[1] = 1
    name: str = "post_impulse_path_ml_v1"
    project_config: Path = Path("configs/project.yaml")
    data_version: str = "latest"
    history_start_date: str = "2019-01-01"
    segments: Segments
    features: PostImpulseFeatureConfig = Field(default_factory=PostImpulseFeatureConfig)
    labels: PostImpulseLabelConfig = Field(default_factory=PostImpulseLabelConfig)
    training: PostImpulseTrainingConfig = Field(default_factory=PostImpulseTrainingConfig)
    output_root: Path = Path("artifacts/post_impulse_ml_runs")

    @model_validator(mode="after")
    def history_precedes_training(self):
        if self.history_start_date > self.segments.train.start:
            raise ValueError("history_start_date must not be after the training start")
        return self


def load_post_impulse_ml_config(path: str | Path) -> PostImpulseMLConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return PostImpulseMLConfig.model_validate(yaml.safe_load(handle) or {})
