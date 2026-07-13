from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from .config import FeatureConfig, ModelConfig, StrictModel
from .mamba_state_config import EncoderConfig, EncoderTrainingConfig


class RecentStructureConfig(StrictModel):
    history_trading_days: int = Field(default=756, ge=252, le=1260)
    sequence_length: int = Field(default=60, ge=10, le=252)
    min_valid_days: int = Field(default=40, ge=5)
    dedup_trading_days: int = Field(default=5, ge=1, le=20)
    efficacy_windows: list[int] = Field(default_factory=lambda: [20, 60, 120])
    minimum_mature_events: int = Field(default=20, ge=5)
    factor_columns: list[str] = Field(
        default_factory=lambda: ["severity", "ret_5d", "vol_20d", "turnover_rate"]
    )

    @model_validator(mode="after")
    def validate_structure(self):
        if self.min_valid_days > self.sequence_length:
            raise ValueError("recent_structure.min_valid_days cannot exceed sequence_length")
        if sorted(set(self.efficacy_windows)) != self.efficacy_windows:
            raise ValueError("efficacy_windows must be unique and increasing")
        if len(self.factor_columns) > 6:
            raise ValueError("at most six conditional factor columns are allowed")
        return self


class WalkForwardConfig(StrictModel):
    training_days: int = Field(default=126, ge=60, le=1008)
    validation_days: int = Field(default=40, ge=20, le=252)
    test_days: int = Field(default=20, ge=5, le=126)
    step_days: int = Field(default=20, ge=5, le=126)
    minimum_train_events: int = Field(default=1000, ge=100)
    minimum_test_dates: int = Field(default=20, ge=5)
    maximum_folds: int = Field(default=3, ge=2, le=12)


class RecentAnomalyStructureConfig(StrictModel):
    version: Literal[1] = 1
    name: str
    project_config: Path = Path("configs/project.yaml")
    data_version: str = "latest"
    scan_summary: Path
    as_of_date: str
    event_templates: list[Path] = Field(min_length=2, max_length=8)
    primary_horizon: Literal[5] = 5
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    recent_structure: RecentStructureConfig = Field(default_factory=RecentStructureConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    training: EncoderTrainingConfig = Field(default_factory=EncoderTrainingConfig)
    lightgbm: ModelConfig = Field(default_factory=ModelConfig)
    primary_metric: Literal["paired_daily_rank_ic_delta_adaptive_vs_static"] = (
        "paired_daily_rank_ic_delta_adaptive_vs_static"
    )
    gate_min_nw_t: float = Field(default=2.0, ge=0)
    gate_min_positive_ratio: float = Field(default=0.50, ge=0, le=1)
    output_root: Path = Path("artifacts/recent_anomaly_structure_runs")

    @model_validator(mode="after")
    def frozen_contract(self):
        paths = list(map(str, self.event_templates))
        if len(paths) != len(set(paths)):
            raise ValueError("event_templates contains duplicate paths")
        if self.walk_forward.step_days != self.walk_forward.test_days:
            raise ValueError("v1 requires non-overlapping test folds: step_days == test_days")
        if self.training.random_seeds != [17]:
            raise ValueError("v1 freezes the rolling CPU encoder to seed 17")
        return self


def load_recent_anomaly_structure_config(path: str | Path) -> RecentAnomalyStructureConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return RecentAnomalyStructureConfig.model_validate(yaml.safe_load(handle) or {})
