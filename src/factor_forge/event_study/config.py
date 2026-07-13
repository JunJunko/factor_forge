from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


MATCHING_CONTROLS = [
    "prior_return_5d",
    "volatility_20d",
    "log_avg_amount_20d",
    "log_total_mv",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MatchStage(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    controls: list[str]


class MatchingConfig(StrictModel):
    exact: list[Literal["trade_date", "industry_l1_code"]]
    stages: list[MatchStage] = Field(min_length=4, max_length=4)
    neighbors: int = Field(default=3, ge=1, le=10)
    caliper: float = Field(default=3.0, gt=0)
    allow_control_reuse: Literal[True] = True
    min_match_rate: float = Field(default=0.70, ge=0, le=1)

    @model_validator(mode="after")
    def frozen_progression(self):
        expected = [
            [],
            ["prior_return_5d"],
            ["prior_return_5d", "volatility_20d"],
            MATCHING_CONTROLS,
        ]
        if [stage.controls for stage in self.stages] != expected:
            raise ValueError(f"matching stages must use frozen progression: {expected}")
        if len({stage.id for stage in self.stages}) != 4:
            raise ValueError("matching stage ids must be unique")
        return self


class InferenceConfig(StrictModel):
    primary_horizon: Literal[3, 5, 10] = 5
    primary_stage: str = "full_controls"
    nw_lag_rule: Literal["horizon_minus_one"] = "horizon_minus_one"
    fdr_alpha: float = Field(default=0.10, gt=0, lt=1)
    min_mature_events: int = Field(default=200, ge=20)
    min_regime_events: int = Field(default=50, ge=20)
    severity_groups: int = Field(default=3, ge=3, le=5)


class GateConfig(StrictModel):
    min_abs_nw_t: float = Field(default=2.0, gt=0)
    max_fdr_q: float = Field(default=0.10, gt=0, lt=1)
    min_daily_direction_ratio: float = Field(default=0.52, ge=0.5, le=1)
    max_abs_smd: float = Field(default=0.20, gt=0, le=1)


class ResearchLineageConfig(StrictModel):
    idea_id: str = Field(pattern=r"^[a-z][a-z0-9_-]+$")
    hypothesis_id: str = Field(pattern=r"^[a-z][a-z0-9_-]+$")
    plan_id: str = Field(pattern=r"^[a-z][a-z0-9_-]+$")
    trial_id: str = Field(pattern=r"^[a-z][a-z0-9_-]+$")
    primary_metric: Literal["full_controls_5d_daily_mean_paired_excess"]


class EventStudyConfig(StrictModel):
    version: Literal[1]
    name: str
    project_config: Path = Path("configs/project.yaml")
    observation_dir: Path
    label_data_version: str = "latest"
    horizons: list[Literal[3, 5, 10]]
    universe_field: Literal["is_liquid"] = "is_liquid"
    matching: MatchingConfig
    inference: InferenceConfig
    gate: GateConfig
    mechanism_feature_set: Literal["turnover_concentration_v1"] | None = None
    lineage: ResearchLineageConfig | None = None
    output_root: Path = Path("artifacts/radar_event_studies")
    research_db: Path | None = None

    @model_validator(mode="after")
    def frozen_horizons_and_primary(self):
        if self.horizons != [3, 5, 10]:
            raise ValueError("Phase 3 horizons are frozen to [3, 5, 10]")
        stage_ids = {stage.id for stage in self.matching.stages}
        if self.inference.primary_stage not in stage_ids:
            raise ValueError("primary_stage must reference a matching stage")
        return self


def load_event_study_config(path: str | Path) -> EventStudyConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return EventStudyConfig.model_validate(yaml.safe_load(handle))
