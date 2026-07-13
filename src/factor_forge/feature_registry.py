"""Auditable metadata registry for reusable, non-executable research features."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FeatureProvenance(_StrictModel):
    idea_id: str
    plan_id: str
    trial_id: str
    decision_id: str
    source_run_id: str
    source_artifact: Path


class FeatureRegistryEntry(_StrictModel):
    version: Literal[1] = 1
    kind: Literal["feature_registry_entry"]
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,95}$")
    name: str
    lifecycle: Literal["research_only", "retired"]
    expression_route: Literal["evaluation_variant"]
    source_factor_config: Path
    source_variant: Literal["industry_size_neutralized"]
    semantics: str
    point_in_time_inputs: list[str] = Field(min_length=1)
    neutralization_controls: list[str] = Field(min_length=1)
    provenance: FeatureProvenance
    allowed_uses: list[Literal["research_diagnostic", "ml_candidate"]] = Field(min_length=1)
    prohibited_uses: list[Literal["standalone_alpha", "trade_signal", "validation_substitute"]] = Field(min_length=1)
    rationale: str

    @model_validator(mode="after")
    def enforce_downgrade_boundary(self) -> "FeatureRegistryEntry":
        required = {"standalone_alpha", "trade_signal", "validation_substitute"}
        if self.lifecycle != "research_only" or not required <= set(self.prohibited_uses):
            raise ValueError("downgraded feature entries must remain research_only and prohibit Alpha, trading, and validation substitution")
        return self


class FeatureRegistry(_StrictModel):
    version: Literal[1] = 1
    kind: Literal["feature_registry"]
    entries: list[Path] = Field(min_length=1)


def _read(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_feature_entry(path: str | Path) -> FeatureRegistryEntry:
    path = Path(path)
    entry = FeatureRegistryEntry.model_validate(_read(path))
    root = path.parent.parent.parent
    for referenced in (entry.source_factor_config, entry.provenance.source_artifact):
        candidate = referenced if referenced.is_absolute() else root / referenced
        if not candidate.exists():
            raise ValueError(f"Feature registry source does not exist: {referenced}")
    return entry


def load_feature_registry(path: str | Path) -> list[FeatureRegistryEntry]:
    path = Path(path)
    registry = FeatureRegistry.model_validate(_read(path))
    entries = [load_feature_entry(path.parent / entry_path) for entry_path in registry.entries]
    ids = [entry.id for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Feature registry entry IDs must be unique")
    return entries
