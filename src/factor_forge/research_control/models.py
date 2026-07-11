from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IdeaStatus(StrEnum):
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    CLOSED = "CLOSED"


class HypothesisStatus(StrEnum):
    PROPOSED = "PROPOSED"
    TESTING = "TESTING"
    REJECTED = "REJECTED"
    SUPPORTED = "SUPPORTED"


class PlanStatus(StrEnum):
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class TrialStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class DataRole(StrEnum):
    DISCOVERY = "discovery"
    VALIDATION = "validation"
    SEALED_TEST = "sealed_test"
    FORWARD = "forward"


class DecisionAction(StrEnum):
    REJECT = "reject"
    OBSERVE_FORWARD = "observe_forward"
    REVISE_ONE_HYPOTHESIS = "revise_one_hypothesis"
    PROMOTE_CANDIDATE = "promote_candidate"
    RETIRE = "retire"


class ResearchIdea(StrictModel):
    id: str
    title: str
    thesis: str
    family_id: str
    target_horizon: int | None = Field(default=None, ge=1)
    status: IdeaStatus = IdeaStatus.DRAFT
    created_at: str
    updated_at: str


class ResearchHypothesis(StrictModel):
    id: str
    idea_id: str
    statement: str
    alternative_to: str | None = None
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    created_at: str


class ExperimentPlan(StrictModel):
    id: str
    idea_id: str
    hypothesis_id: str | None = None
    name: str
    primary_metric: str
    config_path: str | None = None
    status: PlanStatus = PlanStatus.READY
    created_at: str


class TrialRun(StrictModel):
    id: str
    plan_id: str
    external_run_id: str | None = None
    artifact_path: str | None = None
    data_role: DataRole
    status: TrialStatus
    validation_peek: bool = False
    revision: bool = False
    created_at: str


class ResearchDecision(StrictModel):
    id: str
    trial_id: str
    action: DecisionAction
    reason: str
    decided_by: str
    created_at: str


class ResearchBudget(StrictModel):
    scope_type: str
    scope_id: str
    max_trials: int
    max_revisions: int
    max_validation_peeks: int
    trials_used: int
    revisions_used: int
    validation_peeks_used: int
    version: int


class IndexedArtifact(StrictModel):
    run_id: str
    runner_type: str
    status: str
    manifest_path: str
    artifact_path: str
    manifest_sha256: str
    data_version: str | None = None
    code_version: str | None = None
    factor_name: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    idea_id: str | None = None
    indexed_at: str


class SealedAccessAudit(StrictModel):
    id: str
    idea_id: str
    requested_by: str
    approved_by: str
    reason: str
    data_start: str
    data_end: str
    accessed_at: str
    artifact_path: str | None = None


def normalize_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).resolve())
