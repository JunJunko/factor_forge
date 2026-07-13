from __future__ import annotations

import json
from pathlib import Path

import pytest

from factor_forge.research_control import ArtifactIndexer, BudgetExceededError, ResearchControlStore
from factor_forge.research_control import load_phase0_protocol
from factor_forge.research_control.models import DataRole, DecisionAction, TrialStatus
from factor_forge.research_control.store import ResearchControlError


def test_research_lineage_and_budget_are_recorded_atomically(tmp_path):
    store = ResearchControlStore(tmp_path / "research.sqlite3")
    store.initialize()
    idea = store.create_idea(
        title="急跌未放量",
        thesis="关系异常可能包含增量",
        family_id="price_volume",
        target_horizon=5,
        max_trials=2,
        max_validation_peeks=1,
    )
    hypothesis = store.add_hypothesis(idea.id, "卖压有限")
    plan = store.create_plan(
        idea.id, "matched_control", "forward_5d_excess", hypothesis_id=hypothesis.id
    )
    trial = store.record_trial(
        plan.id, DataRole.VALIDATION, TrialStatus.SUCCESS, external_run_id="run_001"
    )
    decision = store.save_decision(
        trial.id, DecisionAction.OBSERVE_FORWARD, "需要发现后确认", "researcher"
    )

    summary = store.idea_summary(idea.id)
    assert summary["idea"]["family_id"] == "price_volume"
    assert summary["idea_budget"]["trials_used"] == 1
    assert summary["idea_budget"]["validation_peeks_used"] == 1
    assert summary["family_budget"]["trials_used"] == 1
    assert summary["trials"][0]["validation_peek"] is True
    assert summary["decisions"][0]["id"] == decision.id

    with pytest.raises(BudgetExceededError, match="validation_peeks"):
        store.record_trial(plan.id, DataRole.VALIDATION, TrialStatus.FAILED)
    budget = store.get_budget("idea", idea.id)
    assert budget.trials_used == 1  # failed transaction does not partially consume trial budget


def test_sealed_access_requires_distinct_approver_and_is_audited(tmp_path):
    store = ResearchControlStore(tmp_path / "research.sqlite3")
    store.initialize()
    idea = store.create_idea("关系漂移", "近期关系发生变化", "relation_drift")
    plan = store.create_plan(idea.id, "drift_validation", "delta_rank_ic")

    with pytest.raises(ResearchControlError, match="sealed_test"):
        store.record_trial(plan.id, DataRole.SEALED_TEST, TrialStatus.SUCCESS)
    with pytest.raises(ResearchControlError, match="distinct approver"):
        store.record_sealed_access(
            idea.id, "same", "same", "final gate", "20250101", "20251231"
        )

    audit = store.record_sealed_access(
        idea.id, "agent", "human", "one-time final gate", "20250101", "20251231"
    )
    with pytest.raises(ResearchControlError, match="one-time"):
        store.record_sealed_access(
            idea.id, "agent", "human", "second peek", "20260101", "20261231"
        )
    with store.connect() as connection:
        saved = connection.execute(
            "SELECT * FROM sealed_access_audit WHERE id=?", (audit.id,)
        ).fetchone()
    assert saved["approved_by"] == "human"


def test_artifact_indexer_normalizes_heterogeneous_manifests_and_is_idempotent(tmp_path):
    artifacts = tmp_path / "artifacts"
    factor_run = artifacts / "runs" / "factor_abc"
    ml_run = artifacts / "ml_runs" / "model_xyz"
    factor_run.mkdir(parents=True)
    ml_run.mkdir(parents=True)
    (factor_run / "manifest.json").write_text(
        json.dumps({
            "run_id": "factor_abc", "status": "SUCCESS", "data_version": "data_v1",
            "code_version": "abc123", "factor_name": "momentum", "started_at": "2026-01-01"
        }), encoding="utf-8"
    )
    (ml_run / "manifest.json").write_text(
        json.dumps({"status": "COMPLETED", "model_name": "lgbm"}), encoding="utf-8"
    )

    store = ResearchControlStore(tmp_path / "research.sqlite3")
    store.initialize()
    indexer = ArtifactIndexer(store, artifacts)
    first = indexer.index()
    second = indexer.index()

    assert first["found"] == 2
    assert first["indexed"] == 2
    assert first["errors"] == []
    assert second["indexed"] == 0
    assert second["unchanged"] == 2
    summary = {(row["runner_type"], row["status"]): row["count"] for row in second["summary"]}
    assert summary[("runs", "SUCCESS")] == 1
    assert summary[("ml_runs", "COMPLETED")] == 1


def test_status_transitions_are_explicit_and_terminal(tmp_path):
    store = ResearchControlStore(tmp_path / "research.sqlite3")
    store.initialize()
    idea = store.create_idea("状态测试", "验证状态机", "state_test")
    assert store.set_idea_status(idea.id, "ACTIVE").status.value == "ACTIVE"
    assert store.set_idea_status(idea.id, "PAUSED").status.value == "PAUSED"
    assert store.set_idea_status(idea.id, "CLOSED").status.value == "CLOSED"
    with pytest.raises(ResearchControlError, match="invalid idea transition"):
        store.set_idea_status(idea.id, "ACTIVE")

    second = store.create_idea("试验状态", "验证试验状态机", "trial_state")
    plan = store.create_plan(second.id, "queued", "primary")
    trial = store.record_trial(plan.id, "discovery", "QUEUED")
    assert store.set_trial_status(trial.id, "RUNNING").status == TrialStatus.RUNNING
    assert store.set_trial_status(trial.id, "SUCCESS").status == TrialStatus.SUCCESS
    with pytest.raises(ResearchControlError, match="invalid trial transition"):
        store.set_trial_status(trial.id, "RUNNING")


def test_superseded_pretrial_plan_can_be_cancelled_without_budget_consumption(tmp_path):
    store = ResearchControlStore(tmp_path / "research.sqlite3")
    store.initialize()
    idea = store.create_idea("supersede", "freeze a corrected plan", "plan_state")
    plan = store.create_plan(idea.id, "v2", "primary")
    cancelled = store.set_plan_status(plan.id, "CANCELLED")
    assert cancelled.status.value == "CANCELLED"
    assert store.get_budget("idea", idea.id).trials_used == 0
    with pytest.raises(ResearchControlError, match="invalid plan transition"):
        store.set_plan_status(plan.id, "READY")


def test_phase0_protocol_is_strict_and_freezes_research_boundaries():
    protocol = load_phase0_protocol("configs/research/phase0_protocol_v1.yaml")
    assert len(protocol.event_templates) == 8
    assert len(protocol.relation_monitors) == 4
    assert protocol.data_roles["sealed_test"].generator_access is False
    assert protocol.data_roles["validation"].access_counts_as_peek is True
    assert {item.id for item in protocol.evaluation_baselines} == {
        "random_template", "radar_skill", "human_research"
    }


def test_budget_expansion_is_one_way_and_audited(tmp_path):
    store = ResearchControlStore(tmp_path / "research.sqlite3")
    store.initialize()
    idea = store.create_idea("预算扩展", "实现错误后的透明重跑", "budget_audit")
    expanded = store.expand_budget_limits(
        "idea", idea.id, max_validation_peeks=3,
        reason="superseded implementation run", approved_by="human",
    )
    assert expanded.max_validation_peeks == 3
    with store.connect() as connection:
        audit = connection.execute(
            "SELECT * FROM research_budget_adjustment WHERE scope_id=?", (idea.id,)
        ).fetchone()
    assert audit["old_max_validation_peeks"] == 2
    assert audit["new_max_validation_peeks"] == 3
    with pytest.raises(ResearchControlError, match="only expand"):
        store.expand_budget_limits(
            "idea", idea.id, max_validation_peeks=2, reason="shrink", approved_by="human"
        )
