from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import DecisionAction


class StrictProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RadarScope(StrictProtocolModel):
    market: Literal["CN_A"]
    frequency: Literal["daily"]
    universe: str
    inputs: list[str] = Field(min_length=1)


class DataRolePolicy(StrictProtocolModel):
    generator_access: bool
    may_modify_hypothesis: bool = False
    access_counts_as_peek: bool = False
    requires_distinct_approver: bool = False
    one_time_gate: bool = False
    starts_at_discovered_at: bool = False


class BudgetLimit(StrictProtocolModel):
    max_trials: int = Field(ge=0)
    max_revisions: int = Field(ge=0)
    max_validation_peeks: int = Field(ge=0)


class ResearchBudgets(StrictProtocolModel):
    idea: BudgetLimit
    family: BudgetLimit

    @model_validator(mode="after")
    def family_covers_idea(self):
        for field in ("max_trials", "max_revisions", "max_validation_peeks"):
            if getattr(self.family, field) < getattr(self.idea, field):
                raise ValueError(f"family {field} cannot be lower than idea {field}")
        return self


class EvaluationBaseline(StrictProtocolModel):
    id: Literal["random_template", "radar_skill", "human_research"]
    description: str


class Phase0Protocol(StrictProtocolModel):
    version: Literal[1]
    name: str
    scope: RadarScope
    data_roles: dict[Literal["discovery", "validation", "sealed_test", "forward"], DataRolePolicy]
    budgets: ResearchBudgets
    event_templates: list[str] = Field(min_length=1, max_length=20)
    relation_monitors: list[str] = Field(min_length=1, max_length=10)
    primary_horizons: list[int] = Field(min_length=1)
    decision_actions: list[DecisionAction]
    evaluation_baselines: list[EvaluationBaseline] = Field(min_length=3, max_length=3)
    success_metrics: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def frozen_invariants(self):
        required_roles = {"discovery", "validation", "sealed_test", "forward"}
        if set(self.data_roles) != required_roles:
            raise ValueError(f"data_roles must be exactly {sorted(required_roles)}")
        sealed = self.data_roles["sealed_test"]
        if sealed.generator_access or not sealed.requires_distinct_approver or not sealed.one_time_gate:
            raise ValueError("sealed_test must deny generator access and require one-time distinct approval")
        if not self.data_roles["validation"].access_counts_as_peek:
            raise ValueError("validation access must count as a peek")
        if len(set(self.event_templates)) != len(self.event_templates):
            raise ValueError("event_templates must be unique")
        if len(set(self.relation_monitors)) != len(self.relation_monitors):
            raise ValueError("relation_monitors must be unique")
        if sorted(set(self.primary_horizons)) != sorted(self.primary_horizons):
            raise ValueError("primary_horizons must be unique")
        expected_actions = set(DecisionAction)
        if set(self.decision_actions) != expected_actions:
            raise ValueError("decision_actions must contain the complete frozen action set")
        expected_baselines = {"random_template", "radar_skill", "human_research"}
        if {item.id for item in self.evaluation_baselines} != expected_baselines:
            raise ValueError("evaluation_baselines must contain random, radar, and human groups")
        return self


def load_phase0_protocol(path: str | Path) -> Phase0Protocol:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return Phase0Protocol.model_validate(payload)
