from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from factor_forge.event_study.config import MatchingConfig

from .config import FeatureConfig, ModelConfig, StrictModel
from .mamba_state_config import EncoderConfig, EncoderTrainingConfig


class EpisodeWindowConfig(StrictModel):
    history_trading_days: int = Field(default=504, ge=120, le=1260)
    sequence_length: int = Field(default=60, ge=10, le=252)
    min_valid_days: int = Field(default=40, ge=5)
    dedup_trading_days: int = Field(default=5, ge=1, le=20)
    decay_half_life_days: int = Field(default=126, ge=20, le=504)

    @model_validator(mode="after")
    def valid_sequence(self):
        if self.min_valid_days > self.sequence_length:
            raise ValueError("episode.min_valid_days cannot exceed sequence_length")
        return self


class EpisodeSplitConfig(StrictModel):
    train_fraction: float = Field(default=0.60, gt=0, lt=1)
    valid_fraction: float = Field(default=0.20, gt=0, lt=1)
    test_fraction: float = Field(default=0.20, gt=0, lt=1)

    @model_validator(mode="after")
    def sums_to_one(self):
        if abs(self.train_fraction + self.valid_fraction + self.test_fraction - 1.0) > 1e-9:
            raise ValueError("episode split fractions must sum to one")
        return self


class EventEpisodeConfig(StrictModel):
    version: Literal[1] = 1
    name: str
    project_config: Path = Path("configs/project.yaml")
    data_version: str = "latest"
    template: Path
    source_observation_dir: Path
    as_of_date: str | None = None
    horizons: list[Literal[1, 3, 5, 10]] = Field(default_factory=lambda: [1, 3, 5, 10])
    primary_horizon: Literal[1, 3, 5, 10] = 5
    episode: EpisodeWindowConfig = Field(default_factory=EpisodeWindowConfig)
    split: EpisodeSplitConfig = Field(default_factory=EpisodeSplitConfig)
    matching: MatchingConfig
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    training: EncoderTrainingConfig = Field(default_factory=EncoderTrainingConfig)
    lightgbm: ModelConfig = Field(default_factory=ModelConfig)
    minimum_matched_episodes: int = Field(default=300, ge=50)
    output_root: Path = Path("artifacts/event_episode_runs")

    @model_validator(mode="after")
    def frozen_contract(self):
        if self.horizons != [1, 3, 5, 10]:
            raise ValueError("event episode horizons are frozen to [1, 3, 5, 10]")
        if self.primary_horizon not in self.horizons:
            raise ValueError("primary_horizon must be in horizons")
        return self


def load_event_episode_config(path: str | Path) -> EventEpisodeConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return EventEpisodeConfig.model_validate(yaml.safe_load(handle) or {})
