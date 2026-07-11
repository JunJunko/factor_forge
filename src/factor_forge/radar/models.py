from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


FORBIDDEN_LABEL_KEYS = {
    "forward_return",
    "future_return",
    "forward_excess_return",
    "target",
    "label",
    "rank_ic",
    "pearson_ic",
    "icir",
    "sharpe",
}


def _assert_label_free(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if (
                normalized != "future_label_fields_present"
                and (
                    normalized in FORBIDDEN_LABEL_KEYS
                    or normalized.startswith(("forward_", "future_"))
                    or normalized.endswith(("_target", "_label"))
                )
            ):
                raise ValueError(f"future-label field is forbidden in ObservationCard: {path}.{key}")
            _assert_label_free(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_label_free(child, f"{path}[{index}]")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObservationDefinition(StrictModel):
    id: str
    version: int
    kind: str
    description: str
    definition_hash: str


class ObservationEvidence(StrictModel):
    event_count: int = Field(ge=0)
    unique_entities: int = Field(ge=0)
    unique_industries: int = Field(ge=0)
    industry_coverage: float | None = Field(default=None, ge=0, le=1)
    recent_event_rate: float | None = Field(default=None, ge=0, le=1)
    historical_event_rate: float | None = Field(default=None, ge=0, le=1)
    event_rate_ratio: float | None = Field(default=None, ge=0)
    severity_median: float | None = None
    severity_p90: float | None = None
    max_entity_share: float | None = Field(default=None, ge=0, le=1)
    max_industry_share: float | None = Field(default=None, ge=0, le=1)
    event_date_start: str | None = None
    event_date_end: str | None = None
    scan_date_event_count: int = Field(default=0, ge=0)
    scan_date_event_rate: float | None = Field(default=None, ge=0, le=1)
    rolling_event_rate_zscore: float | None = None


class ObservationQuality(StrictModel):
    input_rows: int = Field(ge=0)
    eligible_rows: int = Field(ge=0)
    duplicate_keys: int = Field(ge=0)
    measurement_missing_rates: dict[str, float]
    future_label_fields_present: Literal[False] = False
    strict_prior_history: Literal[True] = True
    temporal_audit_passed: bool
    quality_gate_passed: bool = True
    quality_gate_failures: list[str] = Field(default_factory=list)


class ObservationCard(StrictModel):
    schema_version: Literal[1] = 1
    observation_id: str
    definition: ObservationDefinition
    discovered_at: str
    data_version: str
    as_of_date: str
    observation_type: Literal["relation_anomaly"] = "relation_anomaly"
    entity_scope: Literal["stock"] = "stock"
    universe: str
    discovery_window_days: int = Field(ge=1)
    recent_window_days: int = Field(ge=1)
    conditions: dict[str, float | int | str]
    evidence: ObservationEvidence
    quality: ObservationQuality
    event_fields: list[str]
    status: Literal["registered"] = "registered"

    @model_validator(mode="before")
    @classmethod
    def forbid_future_labels(cls, value):
        _assert_label_free(value)
        return value


class RadarScanResult:
    def __init__(self, card: ObservationCard, events: pd.DataFrame):
        self.card = card
        self.events = events


class ObservationArtifact(StrictModel):
    observation_id: str
    artifact_path: Path
    card_path: Path
    events_path: Path
    manifest_path: Path
    card_sha256: str
    event_count: int
