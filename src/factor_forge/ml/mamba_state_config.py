from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from .config import (
    FeatureConfig,
    LabelConfig,
    ModelConfig,
    PortfolioConfig,
    Segments,
    StrictModel,
)


class SequenceConfig(StrictModel):
    length: int = Field(default=60, ge=5, le=252)
    min_valid_days: int = Field(default=40, ge=2)
    include_event_channels: bool = True
    event_templates: list[Path] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def valid_sequence_contract(self):
        if self.min_valid_days > self.length:
            raise ValueError("sequence.min_valid_days cannot exceed sequence.length")
        if self.include_event_channels and len(self.event_templates) != 8:
            raise ValueError("event-conditioned pilot requires exactly eight frozen templates")
        if len(set(map(str, self.event_templates))) != len(self.event_templates):
            raise ValueError("sequence.event_templates contains duplicate paths")
        return self


class EncoderConfig(StrictModel):
    backend: Literal["torch_reference"] = "torch_reference"
    d_model: int = Field(default=32, ge=4, le=256)
    d_state: int = Field(default=16, ge=2, le=128)
    layers: int = Field(default=2, ge=1, le=8)
    embedding_dim: int = Field(default=16, ge=2, le=128)
    dropout: float = Field(default=0.10, ge=0, lt=1)
    mask_probability: float = Field(default=0.20, gt=0, lt=1)


class EncoderTrainingConfig(StrictModel):
    epochs: int = Field(default=30, ge=1, le=500)
    batch_size: int = Field(default=512, ge=4)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    patience: int = Field(default=5, ge=1)
    validation_fraction: float = Field(default=0.15, gt=0, lt=0.5)
    max_train_samples: int | None = Field(default=200_000, ge=100)
    max_valid_samples: int | None = Field(default=50_000, ge=100)
    random_seeds: list[int] = Field(default_factory=lambda: [17, 29, 43], min_length=1, max_length=5)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    num_workers: int = Field(default=0, ge=0, le=16)


class MambaStatePilotConfig(StrictModel):
    version: Literal[1] = 1
    name: str
    project_config: Path = Path("configs/project.yaml")
    data_version: str = "latest"
    require_full_segment_coverage: bool = True
    segments: Segments
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    label: LabelConfig = Field(default_factory=LabelConfig)
    sequence: SequenceConfig
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    training: EncoderTrainingConfig = Field(default_factory=EncoderTrainingConfig)
    lightgbm: ModelConfig = Field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    output_root: Path = Path("artifacts/mamba_state_runs")


def load_mamba_state_config(path: str | Path) -> MambaStatePilotConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return MambaStatePilotConfig.model_validate(yaml.safe_load(handle) or {})
