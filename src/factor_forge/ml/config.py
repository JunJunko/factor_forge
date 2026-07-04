from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Segment(StrictModel):
    start: str
    end: str

    @model_validator(mode="after")
    def ordered(self):
        if self.start > self.end:
            raise ValueError("segment start must not be after end")
        return self


class Segments(StrictModel):
    train: Segment
    valid: Segment
    test: Segment

    @model_validator(mode="after")
    def chronological(self):
        if not (self.train.end < self.valid.start <= self.valid.end < self.test.start):
            raise ValueError("segments must be disjoint and ordered train < valid < test")
        return self


class FeatureConfig(StrictModel):
    windows: list[int] = Field(default_factory=lambda: [5, 10, 20, 60])
    winsor_quantile: float = Field(default=0.01, ge=0, lt=0.5)
    cross_sectional_zscore: bool = True


class LabelConfig(StrictModel):
    horizon: int = Field(default=5, ge=1, le=60)
    price: Literal["adj_open", "adj_close"] = "adj_open"
    excess_over_universe: bool = True


class ModelConfig(StrictModel):
    objective: Literal["regression"] = "regression"
    learning_rate: float = Field(default=0.03, gt=0)
    num_leaves: int = Field(default=31, ge=2)
    max_depth: int = -1
    n_estimators: int = Field(default=500, ge=1)
    subsample: float = Field(default=0.8, gt=0, le=1)
    colsample_bytree: float = Field(default=0.8, gt=0, le=1)
    reg_alpha: float = Field(default=0.1, ge=0)
    reg_lambda: float = Field(default=0.1, ge=0)
    random_state: int = 42
    n_jobs: int = -1


class PortfolioConfig(StrictModel):
    universe: Literal["tradeable", "liquid"] = "liquid"
    top_n: int = Field(default=50, ge=1)
    holding_days: int = Field(default=5, ge=1, le=60)
    initial_cash: float = Field(default=10_000_000, gt=0)
    lot_size: int = Field(default=100, ge=1)
    cost_bps: float = Field(default=15, ge=0)


class MLExperimentConfig(StrictModel):
    version: int = 1
    name: str
    project_config: Path = Path("configs/project.yaml")
    data_version: str = "latest"
    require_full_segment_coverage: bool = True
    segments: Segments
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    label: LabelConfig = Field(default_factory=LabelConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    output_root: Path = Path("artifacts/ml_runs")


def load_ml_config(path: str | Path) -> MLExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return MLExperimentConfig.model_validate(yaml.safe_load(handle) or {})
