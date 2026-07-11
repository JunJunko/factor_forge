from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RelationDriftEvidence(StrictModel):
    relation_id: str
    predictor: str
    target: str
    metric: str
    effective_as_of_date: str
    baseline_days: int
    recent_days: int
    baseline_mean: float | None = None
    medium_mean: float | None = None
    recent_mean: float | None = None
    delta: float | None = None
    robust_delta_zscore: float | None = None
    cusum_score: float | None = None
    persistence_days: int = 0
    valid_days_recent: int = 0
    is_drift: bool = False
    direction: Literal["strengthening", "weakening", "none"] = "none"
    regime_residualized: bool = False


class DriftQuality(StrictModel):
    label_maturity_enforced: bool
    future_incomplete_days_excluded: int = Field(ge=0)
    temporal_audit_passed: bool
    quality_gate_passed: bool
    quality_gate_failures: list[str]


class DriftCard(StrictModel):
    schema_version: Literal[1] = 1
    drift_id: str
    template_id: str
    template_kind: Literal["feature_return_relation_drift", "variable_relation_drift"]
    definition_hash: str
    discovered_at: str
    data_version: str
    scan_date: str
    entity_scope: Literal["market_relation"] = "market_relation"
    relations: list[RelationDriftEvidence]
    drift_count: int = Field(ge=0)
    quality: DriftQuality
    status: Literal["registered"] = "registered"
