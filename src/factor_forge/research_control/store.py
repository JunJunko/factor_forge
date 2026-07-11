from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from uuid import uuid4

from .models import (
    DataRole,
    DecisionAction,
    ExperimentPlan,
    HypothesisStatus,
    IdeaStatus,
    IndexedArtifact,
    PlanStatus,
    ResearchBudget,
    ResearchDecision,
    ResearchHypothesis,
    ResearchIdea,
    SealedAccessAudit,
    TrialRun,
    TrialStatus,
    normalize_path,
    utc_now,
)


class ResearchControlError(RuntimeError):
    pass


class BudgetExceededError(ResearchControlError):
    pass


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS research_schema_version (
    version INTEGER PRIMARY KEY,
    installed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_idea (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    thesis TEXT NOT NULL,
    family_id TEXT NOT NULL,
    target_horizon INTEGER,
    status TEXT NOT NULL CHECK (status IN ('DRAFT','ACTIVE','PAUSED','CLOSED')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_idea_family ON research_idea(family_id);
CREATE TABLE IF NOT EXISTS research_hypothesis (
    id TEXT PRIMARY KEY,
    idea_id TEXT NOT NULL REFERENCES research_idea(id),
    statement TEXT NOT NULL,
    alternative_to TEXT REFERENCES research_hypothesis(id),
    status TEXT NOT NULL CHECK (status IN ('PROPOSED','TESTING','REJECTED','SUPPORTED')),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiment_plan (
    id TEXT PRIMARY KEY,
    idea_id TEXT NOT NULL REFERENCES research_idea(id),
    hypothesis_id TEXT REFERENCES research_hypothesis(id),
    name TEXT NOT NULL,
    primary_metric TEXT NOT NULL,
    config_path TEXT,
    status TEXT NOT NULL CHECK (status IN ('READY','RUNNING','COMPLETED','CANCELLED')),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trial_run (
    id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL REFERENCES experiment_plan(id),
    external_run_id TEXT,
    artifact_path TEXT,
    data_role TEXT NOT NULL CHECK (data_role IN ('discovery','validation','sealed_test','forward')),
    status TEXT NOT NULL CHECK (status IN ('QUEUED','RUNNING','SUCCESS','FAILED','CANCELLED')),
    validation_peek INTEGER NOT NULL CHECK (validation_peek IN (0,1)),
    revision INTEGER NOT NULL CHECK (revision IN (0,1)),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trial_external_run ON trial_run(external_run_id);
CREATE TABLE IF NOT EXISTS research_decision (
    id TEXT PRIMARY KEY,
    trial_id TEXT NOT NULL UNIQUE REFERENCES trial_run(id),
    action TEXT NOT NULL CHECK (action IN ('reject','observe_forward','revise_one_hypothesis','promote_candidate','retire')),
    reason TEXT NOT NULL,
    decided_by TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_budget (
    scope_type TEXT NOT NULL CHECK (scope_type IN ('idea','family')),
    scope_id TEXT NOT NULL,
    max_trials INTEGER NOT NULL CHECK (max_trials >= 0),
    max_revisions INTEGER NOT NULL CHECK (max_revisions >= 0),
    max_validation_peeks INTEGER NOT NULL CHECK (max_validation_peeks >= 0),
    trials_used INTEGER NOT NULL DEFAULT 0,
    revisions_used INTEGER NOT NULL DEFAULT 0,
    validation_peeks_used INTEGER NOT NULL DEFAULT 0,
    version INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (scope_type, scope_id)
);
CREATE TABLE IF NOT EXISTS research_budget_adjustment (
    id TEXT PRIMARY KEY,
    scope_type TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    old_max_trials INTEGER NOT NULL,
    new_max_trials INTEGER NOT NULL,
    old_max_revisions INTEGER NOT NULL,
    new_max_revisions INTEGER NOT NULL,
    old_max_validation_peeks INTEGER NOT NULL,
    new_max_validation_peeks INTEGER NOT NULL,
    reason TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifact_index (
    manifest_path TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    runner_type TEXT NOT NULL,
    status TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    data_version TEXT,
    code_version TEXT,
    factor_name TEXT,
    started_at TEXT,
    finished_at TEXT,
    idea_id TEXT REFERENCES research_idea(id),
    indexed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifact_run_id ON artifact_index(run_id);
CREATE INDEX IF NOT EXISTS idx_artifact_runner ON artifact_index(runner_type, status);
CREATE TABLE IF NOT EXISTS observation_card (
    observation_id TEXT PRIMARY KEY,
    template_id TEXT NOT NULL,
    definition_hash TEXT NOT NULL,
    data_version TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    card_sha256 TEXT NOT NULL,
    discovered_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('registered')),
    UNIQUE(template_id, definition_hash, data_version, as_of_date)
);
CREATE INDEX IF NOT EXISTS idx_observation_template_date
ON observation_card(template_id, as_of_date);
CREATE TABLE IF NOT EXISTS drift_card (
    drift_id TEXT PRIMARY KEY,
    template_id TEXT NOT NULL,
    definition_hash TEXT NOT NULL,
    data_version TEXT NOT NULL,
    scan_date TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    card_sha256 TEXT NOT NULL,
    discovered_at TEXT NOT NULL,
    drift_count INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('registered')),
    UNIQUE(template_id, definition_hash, data_version, scan_date)
);
CREATE TABLE IF NOT EXISTS event_study_run (
    run_id TEXT PRIMARY KEY,
    observation_id TEXT NOT NULL REFERENCES observation_card(observation_id),
    idea_id TEXT NOT NULL REFERENCES research_idea(id),
    plan_id TEXT NOT NULL REFERENCES experiment_plan(id),
    trial_id TEXT NOT NULL REFERENCES trial_run(id),
    config_hash TEXT NOT NULL,
    label_data_version TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS event_study_invalidation (
    run_id TEXT PRIMARY KEY REFERENCES event_study_run(run_id),
    reason TEXT NOT NULL,
    invalidated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sealed_access_audit (
    id TEXT PRIMARY KEY,
    idea_id TEXT NOT NULL REFERENCES research_idea(id),
    requested_by TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    reason TEXT NOT NULL,
    data_start TEXT NOT NULL,
    data_end TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    artifact_path TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sealed_access_once_per_idea
ON sealed_access_audit(idea_id);
"""


class ResearchControlStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            connection.execute(
                "INSERT OR IGNORE INTO research_schema_version(version, installed_at) VALUES (?, ?)",
                (1, utc_now()),
            )

    def create_idea(
        self,
        title: str,
        thesis: str,
        family_id: str,
        target_horizon: int | None = None,
        idea_id: str | None = None,
        max_trials: int = 5,
        max_revisions: int = 2,
        max_validation_peeks: int = 2,
        family_max_trials: int = 25,
        family_max_revisions: int = 10,
        family_max_validation_peeks: int = 8,
    ) -> ResearchIdea:
        family_id = self._slug(family_id, "family_id")
        idea_id = self._slug(idea_id, "idea_id") if idea_id else self._new_id("idea")
        now = utc_now()
        idea = ResearchIdea(
            id=idea_id,
            title=title.strip(),
            thesis=thesis.strip(),
            family_id=family_id,
            target_horizon=target_horizon,
            status=IdeaStatus.DRAFT,
            created_at=now,
            updated_at=now,
        )
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO research_idea VALUES (?,?,?,?,?,?,?,?)",
                (idea.id, idea.title, idea.thesis, idea.family_id, idea.target_horizon,
                 idea.status.value, idea.created_at, idea.updated_at),
            )
            connection.execute(
                "INSERT INTO research_budget VALUES ('idea',?,?,?,?,0,0,0,1)",
                (idea.id, max_trials, max_revisions, max_validation_peeks),
            )
            connection.execute(
                "INSERT OR IGNORE INTO research_budget VALUES ('family',?,?,?,?,0,0,0,1)",
                (family_id, family_max_trials, family_max_revisions, family_max_validation_peeks),
            )
        return idea

    def add_hypothesis(
        self,
        idea_id: str,
        statement: str,
        alternative_to: str | None = None,
        hypothesis_id: str | None = None,
    ) -> ResearchHypothesis:
        self.get_idea(idea_id)
        hypothesis = ResearchHypothesis(
            id=self._slug(hypothesis_id, "hypothesis_id") if hypothesis_id else self._new_id("hyp"),
            idea_id=idea_id,
            statement=statement.strip(),
            alternative_to=alternative_to,
            status=HypothesisStatus.PROPOSED,
            created_at=utc_now(),
        )
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO research_hypothesis VALUES (?,?,?,?,?,?)",
                (hypothesis.id, hypothesis.idea_id, hypothesis.statement,
                 hypothesis.alternative_to, hypothesis.status.value, hypothesis.created_at),
            )
        return hypothesis

    def create_plan(
        self,
        idea_id: str,
        name: str,
        primary_metric: str,
        hypothesis_id: str | None = None,
        config_path: str | Path | None = None,
        plan_id: str | None = None,
    ) -> ExperimentPlan:
        self.get_idea(idea_id)
        if hypothesis_id:
            hypothesis = self.get_hypothesis(hypothesis_id)
            if hypothesis.idea_id != idea_id:
                raise ResearchControlError("hypothesis does not belong to idea")
        plan = ExperimentPlan(
            id=self._slug(plan_id, "plan_id") if plan_id else self._new_id("plan"),
            idea_id=idea_id,
            hypothesis_id=hypothesis_id,
            name=name.strip(),
            primary_metric=primary_metric.strip(),
            config_path=normalize_path(config_path),
            status=PlanStatus.READY,
            created_at=utc_now(),
        )
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO experiment_plan VALUES (?,?,?,?,?,?,?,?)",
                (plan.id, plan.idea_id, plan.hypothesis_id, plan.name, plan.primary_metric,
                 plan.config_path, plan.status.value, plan.created_at),
            )
        return plan

    def record_trial(
        self,
        plan_id: str,
        data_role: DataRole | str,
        status: TrialStatus | str,
        external_run_id: str | None = None,
        artifact_path: str | Path | None = None,
        validation_peek: bool = False,
        revision: bool = False,
        trial_id: str | None = None,
    ) -> TrialRun:
        role = DataRole(data_role)
        status_value = TrialStatus(status)
        if role == DataRole.SEALED_TEST:
            raise ResearchControlError(
                "sealed_test cannot be registered as a normal trial; record an approved sealed access first"
            )
        plan = self.get_plan(plan_id)
        idea = self.get_idea(plan.idea_id)
        if idea.status == IdeaStatus.CLOSED:
            raise ResearchControlError("cannot add a trial to a closed idea")
        trial = TrialRun(
            id=self._slug(trial_id, "trial_id") if trial_id else self._new_id("trial"),
            plan_id=plan_id,
            external_run_id=external_run_id,
            artifact_path=normalize_path(artifact_path),
            data_role=role,
            status=status_value,
            validation_peek=validation_peek or role == DataRole.VALIDATION,
            revision=revision,
            created_at=utc_now(),
        )
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for scope_type, scope_id in (("idea", idea.id), ("family", idea.family_id)):
                self._consume_budget(
                    connection, scope_type, scope_id,
                    trials=1,
                    revisions=int(trial.revision),
                    validation_peeks=int(trial.validation_peek),
                )
            connection.execute(
                "INSERT INTO trial_run VALUES (?,?,?,?,?,?,?,?,?)",
                (trial.id, trial.plan_id, trial.external_run_id, trial.artifact_path,
                 trial.data_role.value, trial.status.value, int(trial.validation_peek),
                 int(trial.revision), trial.created_at),
            )
            connection.execute(
                "UPDATE experiment_plan SET status=? WHERE id=?",
                (PlanStatus.RUNNING.value if trial.status in {TrialStatus.QUEUED, TrialStatus.RUNNING}
                 else PlanStatus.COMPLETED.value, plan_id),
            )
        return trial

    def set_idea_status(self, idea_id: str, status: IdeaStatus | str) -> ResearchIdea:
        idea = self.get_idea(idea_id)
        target = IdeaStatus(status)
        allowed = {
            IdeaStatus.DRAFT: {IdeaStatus.ACTIVE, IdeaStatus.CLOSED},
            IdeaStatus.ACTIVE: {IdeaStatus.PAUSED, IdeaStatus.CLOSED},
            IdeaStatus.PAUSED: {IdeaStatus.ACTIVE, IdeaStatus.CLOSED},
            IdeaStatus.CLOSED: set(),
        }
        if target != idea.status and target not in allowed[idea.status]:
            raise ResearchControlError(f"invalid idea transition: {idea.status.value} -> {target.value}")
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                "UPDATE research_idea SET status=?,updated_at=? WHERE id=?",
                (target.value, now, idea_id),
            )
        return self.get_idea(idea_id)

    def set_trial_status(self, trial_id: str, status: TrialStatus | str) -> TrialRun:
        trial = self.get_trial(trial_id)
        target = TrialStatus(status)
        allowed = {
            TrialStatus.QUEUED: {TrialStatus.RUNNING, TrialStatus.FAILED, TrialStatus.CANCELLED},
            TrialStatus.RUNNING: {TrialStatus.SUCCESS, TrialStatus.FAILED, TrialStatus.CANCELLED},
            TrialStatus.SUCCESS: set(),
            TrialStatus.FAILED: set(),
            TrialStatus.CANCELLED: set(),
        }
        if target != trial.status and target not in allowed[trial.status]:
            raise ResearchControlError(f"invalid trial transition: {trial.status.value} -> {target.value}")
        with self.connect() as connection:
            connection.execute("UPDATE trial_run SET status=? WHERE id=?", (target.value, trial_id))
            if target in {TrialStatus.SUCCESS, TrialStatus.FAILED, TrialStatus.CANCELLED}:
                connection.execute(
                    "UPDATE experiment_plan SET status=? WHERE id=?",
                    (PlanStatus.COMPLETED.value, trial.plan_id),
                )
        return self.get_trial(trial_id)

    def save_decision(
        self,
        trial_id: str,
        action: DecisionAction | str,
        reason: str,
        decided_by: str,
        decision_id: str | None = None,
    ) -> ResearchDecision:
        self.get_trial(trial_id)
        decision = ResearchDecision(
            id=self._slug(decision_id, "decision_id") if decision_id else self._new_id("decision"),
            trial_id=trial_id,
            action=DecisionAction(action),
            reason=reason.strip(),
            decided_by=decided_by.strip(),
            created_at=utc_now(),
        )
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO research_decision VALUES (?,?,?,?,?,?)",
                (decision.id, decision.trial_id, decision.action.value, decision.reason,
                 decision.decided_by, decision.created_at),
            )
        return decision

    def record_sealed_access(
        self,
        idea_id: str,
        requested_by: str,
        approved_by: str,
        reason: str,
        data_start: str,
        data_end: str,
        artifact_path: str | Path | None = None,
    ) -> SealedAccessAudit:
        self.get_idea(idea_id)
        if requested_by.strip() == approved_by.strip():
            raise ResearchControlError("sealed access requires a distinct approver")
        if data_start > data_end:
            raise ResearchControlError("sealed access requires data_start <= data_end")
        with self.connect() as connection:
            previous = connection.execute(
                "SELECT id FROM sealed_access_audit WHERE idea_id=?", (idea_id,)
            ).fetchone()
        if previous is not None:
            raise ResearchControlError("sealed access is one-time per idea")
        audit = SealedAccessAudit(
            id=self._new_id("sealed"), idea_id=idea_id,
            requested_by=requested_by.strip(), approved_by=approved_by.strip(),
            reason=reason.strip(), data_start=data_start, data_end=data_end,
            accessed_at=utc_now(), artifact_path=normalize_path(artifact_path),
        )
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO sealed_access_audit VALUES (?,?,?,?,?,?,?,?,?)",
                (audit.id, audit.idea_id, audit.requested_by, audit.approved_by, audit.reason,
                 audit.data_start, audit.data_end, audit.accessed_at, audit.artifact_path),
            )
        return audit

    def upsert_artifact(self, artifact: IndexedArtifact) -> None:
        idea_id = artifact.idea_id
        if idea_id is not None:
            with self.connect() as connection:
                known = connection.execute(
                    "SELECT 1 FROM research_idea WHERE id=?", (idea_id,)
                ).fetchone()
            if known is None:
                idea_id = None
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO artifact_index(
                    manifest_path,run_id,runner_type,status,artifact_path,manifest_sha256,
                    data_version,code_version,factor_name,started_at,finished_at,idea_id,indexed_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(manifest_path) DO UPDATE SET
                    run_id=excluded.run_id, runner_type=excluded.runner_type,
                    status=excluded.status, artifact_path=excluded.artifact_path,
                    manifest_sha256=excluded.manifest_sha256, data_version=excluded.data_version,
                    code_version=excluded.code_version, factor_name=excluded.factor_name,
                    started_at=excluded.started_at, finished_at=excluded.finished_at,
                    idea_id=COALESCE(excluded.idea_id, artifact_index.idea_id),
                    indexed_at=excluded.indexed_at""",
                (artifact.manifest_path, artifact.run_id, artifact.runner_type, artifact.status,
                 artifact.artifact_path, artifact.manifest_sha256, artifact.data_version,
                 artifact.code_version, artifact.factor_name, artifact.started_at,
                 artifact.finished_at, idea_id, artifact.indexed_at),
            )

    def register_observation(
        self,
        *,
        observation_id: str,
        template_id: str,
        definition_hash: str,
        data_version: str,
        as_of_date: str,
        artifact_path: str | Path,
        card_sha256: str,
        discovered_at: str,
        status: str = "registered",
    ) -> dict:
        normalized_path = normalize_path(artifact_path)
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT * FROM observation_card WHERE observation_id=?", (observation_id,)
            ).fetchone()
            if existing is not None:
                saved = dict(existing)
                expected = {
                    "observation_id": observation_id,
                    "template_id": template_id,
                    "definition_hash": definition_hash,
                    "data_version": data_version,
                    "as_of_date": as_of_date,
                    "artifact_path": normalized_path,
                    "card_sha256": card_sha256,
                    "status": status,
                }
                mismatched = [key for key, value in expected.items() if saved[key] != value]
                if mismatched:
                    raise ResearchControlError(
                        f"immutable observation registration conflict: {observation_id}; fields={mismatched}"
                    )
                return saved
            connection.execute(
                "INSERT INTO observation_card VALUES (?,?,?,?,?,?,?,?,?)",
                (observation_id, template_id, definition_hash, data_version, as_of_date,
                 normalized_path, card_sha256, discovered_at, status),
            )
        return {
            "observation_id": observation_id, "template_id": template_id,
            "definition_hash": definition_hash, "data_version": data_version,
            "as_of_date": as_of_date, "artifact_path": normalized_path,
            "card_sha256": card_sha256, "discovered_at": discovered_at, "status": status,
        }

    def get_observation(self, observation_id: str) -> dict:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM observation_card WHERE observation_id=?", (observation_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"observation not found: {observation_id}")
        return dict(row)

    def register_drift_card(
        self,
        *,
        drift_id: str,
        template_id: str,
        definition_hash: str,
        data_version: str,
        scan_date: str,
        artifact_path: str | Path,
        card_sha256: str,
        discovered_at: str,
        drift_count: int,
        status: str = "registered",
    ) -> dict:
        values = (
            drift_id, template_id, definition_hash, data_version, scan_date,
            normalize_path(artifact_path), card_sha256, discovered_at, drift_count, status,
        )
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT * FROM drift_card WHERE drift_id=?", (drift_id,)
            ).fetchone()
            if existing is not None:
                return dict(existing)
            connection.execute("INSERT INTO drift_card VALUES (?,?,?,?,?,?,?,?,?,?)", values)
        return dict(zip([
            "drift_id", "template_id", "definition_hash", "data_version", "scan_date",
            "artifact_path", "card_sha256", "discovered_at", "drift_count", "status",
        ], values, strict=True))

    def register_event_study(
        self,
        *,
        run_id: str,
        observation_id: str,
        idea_id: str,
        plan_id: str,
        trial_id: str,
        config_hash: str,
        label_data_version: str,
        artifact_path: str | Path,
        status: str,
        created_at: str,
    ) -> dict:
        values = (
            run_id, observation_id, idea_id, plan_id, trial_id, config_hash,
            label_data_version, normalize_path(artifact_path), status, created_at,
        )
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT * FROM event_study_run WHERE run_id=?", (run_id,)
            ).fetchone()
            if existing is not None:
                return dict(existing)
            connection.execute(
                "INSERT INTO event_study_run VALUES (?,?,?,?,?,?,?,?,?,?)", values
            )
        return {
            "run_id": run_id, "observation_id": observation_id, "idea_id": idea_id,
            "plan_id": plan_id, "trial_id": trial_id, "config_hash": config_hash,
            "label_data_version": label_data_version,
            "artifact_path": normalize_path(artifact_path), "status": status,
            "created_at": created_at,
        }

    def invalidate_event_study(self, run_id: str, reason: str) -> dict:
        invalidated_at = utc_now()
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT 1 FROM event_study_run WHERE run_id=?", (run_id,)
            ).fetchone()
            if existing is None:
                raise KeyError(f"event study not found: {run_id}")
            connection.execute(
                "UPDATE event_study_run SET status='INVALIDATED' WHERE run_id=?", (run_id,)
            )
            connection.execute(
                "INSERT OR REPLACE INTO event_study_invalidation VALUES (?,?,?)",
                (run_id, reason.strip(), invalidated_at),
            )
        return {"run_id": run_id, "status": "INVALIDATED", "reason": reason,
                "invalidated_at": invalidated_at}

    def get_idea(self, idea_id: str) -> ResearchIdea:
        return self._get_model("research_idea", idea_id, ResearchIdea)

    def get_hypothesis(self, hypothesis_id: str) -> ResearchHypothesis:
        return self._get_model("research_hypothesis", hypothesis_id, ResearchHypothesis)

    def get_plan(self, plan_id: str) -> ExperimentPlan:
        return self._get_model("experiment_plan", plan_id, ExperimentPlan)

    def get_trial(self, trial_id: str) -> TrialRun:
        return self._get_model("trial_run", trial_id, TrialRun, bool_fields=("validation_peek", "revision"))

    def get_decision_for_trial(self, trial_id: str) -> ResearchDecision | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM research_decision WHERE trial_id=?", (trial_id,)
            ).fetchone()
        return ResearchDecision.model_validate(dict(row)) if row is not None else None

    def get_budget(self, scope_type: str, scope_id: str) -> ResearchBudget:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM research_budget WHERE scope_type=? AND scope_id=?",
                (scope_type, scope_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"budget not found: {scope_type}/{scope_id}")
        return ResearchBudget.model_validate(dict(row))

    def expand_budget_limits(
        self,
        scope_type: str,
        scope_id: str,
        *,
        max_trials: int | None = None,
        max_revisions: int | None = None,
        max_validation_peeks: int | None = None,
        reason: str,
        approved_by: str,
    ) -> ResearchBudget:
        current = self.get_budget(scope_type, scope_id)
        new_trials = current.max_trials if max_trials is None else max_trials
        new_revisions = current.max_revisions if max_revisions is None else max_revisions
        new_peeks = current.max_validation_peeks if max_validation_peeks is None else max_validation_peeks
        if (
            new_trials < current.max_trials
            or new_revisions < current.max_revisions
            or new_peeks < current.max_validation_peeks
        ):
            raise ResearchControlError("budget adjustment can only expand limits")
        if not reason.strip() or not approved_by.strip():
            raise ResearchControlError("budget expansion requires reason and approved_by")
        adjustment_id = self._new_id("budget")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """UPDATE research_budget SET max_trials=?,max_revisions=?,
                   max_validation_peeks=?,version=version+1
                   WHERE scope_type=? AND scope_id=?""",
                (new_trials, new_revisions, new_peeks, scope_type, scope_id),
            )
            connection.execute(
                "INSERT INTO research_budget_adjustment VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    adjustment_id, scope_type, scope_id,
                    current.max_trials, new_trials,
                    current.max_revisions, new_revisions,
                    current.max_validation_peeks, new_peeks,
                    reason.strip(), approved_by.strip(), utc_now(),
                ),
            )
        return self.get_budget(scope_type, scope_id)

    def idea_summary(self, idea_id: str) -> dict:
        idea = self.get_idea(idea_id)
        with self.connect() as connection:
            hypotheses = [dict(row) for row in connection.execute(
                "SELECT * FROM research_hypothesis WHERE idea_id=? ORDER BY created_at", (idea_id,)
            )]
            plans = [dict(row) for row in connection.execute(
                "SELECT * FROM experiment_plan WHERE idea_id=? ORDER BY created_at", (idea_id,)
            )]
            trials = [dict(row) for row in connection.execute(
                """SELECT t.* FROM trial_run t JOIN experiment_plan p ON p.id=t.plan_id
                   WHERE p.idea_id=? ORDER BY t.created_at""", (idea_id,)
            )]
            decisions = [dict(row) for row in connection.execute(
                """SELECT d.* FROM research_decision d JOIN trial_run t ON t.id=d.trial_id
                   JOIN experiment_plan p ON p.id=t.plan_id WHERE p.idea_id=? ORDER BY d.created_at""",
                (idea_id,),
            )]
        for trial in trials:
            trial["validation_peek"] = bool(trial["validation_peek"])
            trial["revision"] = bool(trial["revision"])
        return {
            "idea": idea.model_dump(mode="json"),
            "idea_budget": self.get_budget("idea", idea.id).model_dump(),
            "family_budget": self.get_budget("family", idea.family_id).model_dump(),
            "hypotheses": hypotheses,
            "plans": plans,
            "trials": trials,
            "decisions": decisions,
        }

    def artifact_summary(self) -> list[dict]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(
                """SELECT runner_type,status,COUNT(*) AS count
                   FROM artifact_index GROUP BY runner_type,status
                   ORDER BY count DESC,runner_type,status"""
            )]

    def _get_model(self, table: str, object_id: str, model, bool_fields: tuple[str, ...] = ()):
        with self.connect() as connection:
            row = connection.execute(f"SELECT * FROM {table} WHERE id=?", (object_id,)).fetchone()
        if row is None:
            raise KeyError(f"{table} not found: {object_id}")
        value = dict(row)
        for field in bool_fields:
            value[field] = bool(value[field])
        return model.model_validate(value)

    @staticmethod
    def _consume_budget(
        connection: sqlite3.Connection,
        scope_type: str,
        scope_id: str,
        trials: int,
        revisions: int,
        validation_peeks: int,
    ) -> None:
        row = connection.execute(
            "SELECT * FROM research_budget WHERE scope_type=? AND scope_id=?",
            (scope_type, scope_id),
        ).fetchone()
        if row is None:
            raise ResearchControlError(f"missing budget: {scope_type}/{scope_id}")
        checks = (
            ("trials", row["trials_used"] + trials, row["max_trials"]),
            ("revisions", row["revisions_used"] + revisions, row["max_revisions"]),
            ("validation_peeks", row["validation_peeks_used"] + validation_peeks,
             row["max_validation_peeks"]),
        )
        exceeded = [f"{name} {used}>{limit}" for name, used, limit in checks if used > limit]
        if exceeded:
            raise BudgetExceededError(
                f"research budget exceeded for {scope_type}/{scope_id}: {', '.join(exceeded)}"
            )
        connection.execute(
            """UPDATE research_budget SET trials_used=trials_used+?,
               revisions_used=revisions_used+?, validation_peeks_used=validation_peeks_used+?,
               version=version+1 WHERE scope_type=? AND scope_id=?""",
            (trials, revisions, validation_peeks, scope_type, scope_id),
        )

    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix}_{uuid4().hex[:12]}"

    @staticmethod
    def _slug(value: str | None, field: str) -> str:
        if value is None or not re.fullmatch(r"[a-z][a-z0-9_\-]{1,63}", value):
            raise ValueError(f"{field} must match [a-z][a-z0-9_-]{{1,63}}")
        return value
