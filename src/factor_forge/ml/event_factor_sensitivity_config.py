from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from .config import FeatureConfig, ModelConfig, StrictModel
from .mamba_state_config import EncoderConfig, EncoderTrainingConfig


FACTOR_BASIS = [
    "short_reversal",
    "trend_acceleration",
    "volume_price_efficiency",
    "volatility_compression",
    "industry_relative_return",
    "liquidity_displacement",
    "intraday_rejection",
]


class EventEpisodeContract(StrictModel):
    history_trading_days: int = Field(default=252, ge=200, le=756)
    sequence_length: int = Field(default=60, ge=20, le=126)
    min_valid_days: int = Field(default=40, ge=10)
    dedup_trading_days: int = Field(default=5, ge=1, le=20)
    factor_basis: list[str] = Field(default_factory=lambda: list(FACTOR_BASIS))

    @model_validator(mode="after")
    def fixed_basis(self):
        if self.factor_basis != FACTOR_BASIS:
            raise ValueError(f"v1 factor basis is frozen to {FACTOR_BASIS}")
        if self.min_valid_days > self.sequence_length:
            raise ValueError("event.min_valid_days cannot exceed sequence_length")
        return self


class ChronologicalOOFConfig(StrictModel):
    training_days: int = Field(default=126, ge=60, le=504)
    validation_days: int = Field(default=20, ge=10, le=63)
    block_days: int = Field(default=20, ge=5, le=63)
    minimum_train_events: int = Field(default=5000, ge=500)
    minimum_prior_oof_rows: int = Field(default=5000, ge=500)
    evaluation_blocks: int = Field(default=3, ge=2, le=6)


class EventMambaHeadConfig(StrictModel):
    template_embedding_dim: int = Field(default=4, ge=2, le=32)
    residual_embedding_dim: int = Field(default=4, ge=0, le=32)
    supervised_loss_weight: float = Field(default=1.0, gt=0)
    reconstruction_loss_weight: float = Field(default=0.10, ge=0)
    beta_l1_weight: float = Field(default=0.001, ge=0)


class EventFactorSensitivityConfig(StrictModel):
    version: Literal[1] = 1
    name: str
    project_config: Path = Path("configs/project.yaml")
    data_version: str = "latest"
    scan_summary: Path
    as_of_date: str
    event_templates: list[Path] = Field(min_length=2, max_length=8)
    primary_horizon: Literal[5] = 5
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    event: EventEpisodeContract = Field(default_factory=EventEpisodeContract)
    oof: ChronologicalOOFConfig = Field(default_factory=ChronologicalOOFConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    event_mamba: EventMambaHeadConfig = Field(default_factory=EventMambaHeadConfig)
    training: EncoderTrainingConfig = Field(default_factory=EncoderTrainingConfig)
    lightgbm: ModelConfig = Field(default_factory=ModelConfig)
    primary_metric: Literal["paired_daily_rank_ic_delta_e2_vs_e1"] = (
        "paired_daily_rank_ic_delta_e2_vs_e1"
    )
    gate_min_nw_t: float = Field(default=2.0, ge=0)
    gate_min_positive_ratio: float = Field(default=0.50, ge=0, le=1)
    output_root: Path = Path("artifacts/event_factor_sensitivity_runs")

    @model_validator(mode="after")
    def frozen_contract(self):
        paths = list(map(str, self.event_templates))
        if len(paths) != len(set(paths)):
            raise ValueError("event_templates contains duplicates")
        if self.training.random_seeds != [17]:
            raise ValueError("v1 freezes Event-Mamba to seed 17")
        return self


def load_event_factor_sensitivity_config(path: str | Path) -> EventFactorSensitivityConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return EventFactorSensitivityConfig.model_validate(yaml.safe_load(handle) or {})
