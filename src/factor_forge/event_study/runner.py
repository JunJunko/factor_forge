from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data import DataVersionRepository
from factor_forge.radar.models import ObservationCard
from factor_forge.research_control import ArtifactIndexer, ResearchControlStore
from factor_forge.research_control.models import utc_now
from factor_forge.research_control.store import ResearchControlError

from .analysis import analyze_event_study
from .config import EventStudyConfig, load_event_study_config
from .labels import build_market_regimes, build_point_in_time_features_and_labels
from .matching import mark_frozen_events, match_all_stages
from .mechanism_features import (
    TURNOVER_CONCENTRATION_AGGREGATE_FIELDS,
    build_turnover_concentration_aggregate_features,
    turnover_concentration_prefix_audit,
)


EVENT_STUDY_ENGINE_VERSION = "1.2.0"


class EventStudyRunner:
    def run(self, config_path: str | Path) -> dict:
        cfg = load_event_study_config(config_path)
        project = load_project(cfg.project_config)
        observation_dir = Path(cfg.observation_dir)
        card_path = observation_dir / "observation_card.json"
        events_path = observation_dir / "events.parquet"
        source_manifest_path = observation_dir / "manifest.json"
        card, events, source_manifest, e0 = self._load_frozen_observation(
            card_path, events_path, source_manifest_path
        )
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        label_version, label_manifest = repository.load_manifest(cfg.label_data_version)
        config_hash = self._config_hash(cfg)
        run_id = self._run_id(card.observation_id, source_manifest["card_sha256"], label_version, config_hash)
        output = cfg.output_root / run_id
        research_db = cfg.research_db or project.paths.data_root / "research.sqlite3"
        store = ResearchControlStore(research_db)
        store.initialize()
        registered = store.get_observation(card.observation_id)
        if registered["card_sha256"] != source_manifest["card_sha256"]:
            raise ResearchControlError("registered observation card hash differs from frozen artifact")

        cached = self._cached_result(output, run_id, config_hash, source_manifest, label_version)
        if cached is not None:
            trial_id = cached.pop("_trial_id")
            decision = store.get_decision_for_trial(trial_id)
            cached["decision_id"] = decision.id if decision is not None else None
            ArtifactIndexer(store, cfg.output_root.parent).index()
            return cached

        idea_id, hypothesis_id, plan_id, trial_id = self._start_lineage(
            store, card, run_id, cfg
        )
        try:
            panel = self._load_label_slice(
                repository, label_version, label_manifest, events, cfg
            )
            mechanism_features = self._mechanism_features(panel, events, cfg)
            if mechanism_features is not None:
                e0["mechanism_feature_prefix_audit"] = True
                e0["mechanism_feature_set"] = cfg.mechanism_feature_set
            enriched = build_point_in_time_features_and_labels(panel, cfg.horizons)
            marked = mark_frozen_events(enriched, events)
            event_rows = marked.loc[marked["is_frozen_event"]].copy()
            missing_event_rows = len(events) - len(event_rows)
            e0.update({
                "label_data_version": label_version,
                "label_data_end": label_manifest["end_date"],
                "frozen_events": len(events),
                "events_found_in_label_panel": len(event_rows),
                "missing_event_rows": missing_event_rows,
                "label_semantics": "adj_open(T+h+1) / adj_open(T+1) - 1",
                "maturity": {
                    str(horizon): {
                        "mature": int(event_rows[f"label_mature_{horizon}"].sum()),
                        "censored": int(len(event_rows) - event_rows[f"label_mature_{horizon}"].sum()),
                    }
                    for horizon in cfg.horizons
                },
            })
            if missing_event_rows:
                raise ValueError(f"label panel is missing {missing_event_rows} frozen event rows")
            matched = match_all_stages(marked, cfg.matching, cfg.horizons)
            regimes = build_market_regimes(enriched)
            summary, tables = analyze_event_study(event_rows, matched, regimes, cfg)
            summary.update({
                "run_id": run_id,
                "observation_id": card.observation_id,
                "definition_id": card.definition.id,
                "observation_as_of_date": card.as_of_date,
                "observation_data_version": card.data_version,
                "label_data_version": label_version,
                "source_event_count": len(events),
                "lineage": {
                    "idea_id": idea_id, "hypothesis_id": hypothesis_id,
                    "plan_id": plan_id, "trial_id": trial_id,
                },
            })
            if mechanism_features is not None:
                event_dates = set(pd.to_datetime(events["trade_date"]).unique())
                summary["mechanism_features"] = {
                    "feature_set": cfg.mechanism_feature_set,
                    "fields": list(TURNOVER_CONCENTRATION_AGGREGATE_FIELDS),
                    "prefix_audit_passed": True,
                    "daily_rows": len(mechanism_features),
                    "event_date_rows": int(mechanism_features["trade_date"].isin(event_dates).sum()),
                    "role": "diagnostic_only_cannot_rescue_primary_gate",
                }
            self._write_artifacts(
                output, cfg, config_hash, card, source_manifest, label_version,
                e0, summary, matched, tables, mechanism_features,
            )
            store.set_trial_status(trial_id, "SUCCESS")
            decision = store.save_decision(
                trial_id, summary["gate"]["next_action"], summary["gate"]["reason"],
                "deterministic_event_study_gate_v1",
            )
            store.register_event_study(
                run_id=run_id, observation_id=card.observation_id, idea_id=idea_id,
                plan_id=plan_id, trial_id=trial_id, config_hash=config_hash,
                label_data_version=label_version, artifact_path=output, status="COMPLETED",
                created_at=summary["created_at"],
            )
            ArtifactIndexer(store, cfg.output_root.parent).index()
            return {
                "run_id": run_id, "artifact_path": str(output.resolve()),
                "observation_id": card.observation_id, "label_data_version": label_version,
                "primary_result": summary["primary_result"], "gate": summary["gate"],
                "decision_id": decision.id, "cached": False,
            }
        except Exception:
            try:
                store.set_trial_status(trial_id, "FAILED")
            except Exception:
                pass
            raise

    @staticmethod
    def _load_frozen_observation(card_path: Path, events_path: Path, manifest_path: Path):
        for path in (card_path, events_path, manifest_path):
            if not path.exists():
                raise FileNotFoundError(f"frozen observation artifact missing: {path}")
        card_bytes, events_bytes = card_path.read_bytes(), events_path.read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        card = ObservationCard.model_validate_json(card_bytes)
        events = pd.read_parquet(events_path)
        checks = {
            "card_sha256_matches": hashlib.sha256(card_bytes).hexdigest() == manifest.get("card_sha256"),
            "events_sha256_matches": hashlib.sha256(events_bytes).hexdigest() == manifest.get("events_sha256"),
            "observation_id_matches": card.observation_id == manifest.get("observation_id"),
            "event_count_matches": len(events) == card.evidence.event_count == manifest.get("event_count"),
            "event_columns_match": list(events.columns) == card.event_fields,
            "event_keys_unique": not events.duplicated(["trade_date", "ts_code"]).any(),
            "source_contains_future_labels": manifest.get("contains_future_labels") is True,
            "source_temporal_audit_passed": card.quality.temporal_audit_passed,
        }
        # This field is phrased as a positive violation flag; it must remain False.
        checks["source_contains_future_labels"] = manifest.get("contains_future_labels") is False
        if not all(checks.values()):
            failed = [key for key, value in checks.items() if not value]
            raise ValueError(f"frozen observation audit failed: {failed}")
        return card, events, manifest, {
            "status": "PASSED", "source_card": str(card_path.resolve()),
            "source_events": str(events_path.resolve()), **checks,
        }

    @staticmethod
    def _load_label_slice(
        repository: DataVersionRepository,
        label_version: str,
        manifest: dict,
        events: pd.DataFrame,
        cfg: EventStudyConfig,
    ) -> pd.DataFrame:
        event_dates = pd.to_datetime(events["trade_date"])
        history_days = 550 if cfg.mechanism_feature_set else 120
        start = event_dates.min() - pd.Timedelta(days=history_days)
        end = min(event_dates.max() + pd.Timedelta(days=40), pd.Timestamp(manifest["end_date"]))
        path = repository.root / "versions" / label_version / "curated" / "stock_daily_panel.parquet"
        columns = [
            "trade_date", "ts_code", "adj_open", "adj_close", "amount_cny",
            "log_total_mv", "industry_l1_code", cfg.universe_field,
        ]
        return pd.read_parquet(
            path, columns=columns,
            filters=[("trade_date", ">=", start), ("trade_date", "<=", end)],
        )

    @staticmethod
    def _mechanism_features(panel, events, cfg):
        if cfg.mechanism_feature_set is None:
            return None
        if cfg.mechanism_feature_set != "turnover_concentration_v1":  # pragma: no cover
            raise ValueError(f"unsupported mechanism feature set: {cfg.mechanism_feature_set}")
        if not turnover_concentration_prefix_audit(panel):
            raise ValueError("turnover concentration mechanism Features failed PIT-prefix audit")
        features = build_turnover_concentration_aggregate_features(panel)
        event_dates = pd.to_datetime(events["trade_date"])
        return features.loc[
            features["trade_date"].between(event_dates.min(), event_dates.max())
        ].reset_index(drop=True)

    @staticmethod
    def _config_hash(cfg: EventStudyConfig) -> str:
        payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _run_id(observation_id: str, card_hash: str, label_version: str, config_hash: str) -> str:
        digest = hashlib.sha256(
            f"{EVENT_STUDY_ENGINE_VERSION}|{observation_id}|{card_hash}|{label_version}|{config_hash}".encode("utf-8")
        ).hexdigest()[:16]
        return f"event_study_{digest}"

    @staticmethod
    def _start_lineage(store, card, run_id, cfg):
        if cfg.lineage is not None:
            lineage = cfg.lineage
            idea = store.get_idea(lineage.idea_id)
            hypothesis = store.get_hypothesis(lineage.hypothesis_id)
            plan = store.get_plan(lineage.plan_id)
            if hypothesis.idea_id != idea.id or plan.idea_id != idea.id:
                raise ResearchControlError("pre-registered Event Study lineage crosses Idea boundaries")
            if plan.hypothesis_id != hypothesis.id:
                raise ResearchControlError("pre-registered Plan does not reference the configured Hypothesis")
            if plan.primary_metric != lineage.primary_metric:
                raise ResearchControlError("pre-registered Plan primary metric differs from Event Study config")
            try:
                store.get_trial(lineage.trial_id)
            except KeyError:
                store.record_trial(
                    plan.id, "validation", "RUNNING", external_run_id=run_id,
                    artifact_path=cfg.output_root / run_id, trial_id=lineage.trial_id,
                )
            else:
                raise ResearchControlError(
                    "pre-registered Trial already exists without a matching immutable cache"
                )
            return idea.id, hypothesis.id, plan.id, lineage.trial_id
        digest = hashlib.sha256(card.observation_id.encode("utf-8")).hexdigest()[:12]
        idea_id, hypothesis_id = f"idea_obs_{digest}", f"hyp_obs_{digest}"
        plan_id, trial_id = f"plan_event_{digest}", f"trial_{run_id[-16:]}"
        try:
            idea = store.get_idea(idea_id)
        except KeyError:
            idea = store.create_idea(
                title=f"Observation Event Study: {card.definition.id}",
                thesis="冻结异常事件相对同日同行业匹配对照是否存在未来收益差异",
                family_id=card.definition.id, target_horizon=cfg.inference.primary_horizon,
                idea_id=idea_id,
            )
        if idea.status.value == "DRAFT":
            store.set_idea_status(idea_id, "ACTIVE")
        try:
            store.get_hypothesis(hypothesis_id)
        except KeyError:
            store.add_hypothesis(
                idea_id, "该冻结关系异常在完整匹配控制后仍存在5日配对收益差异",
                hypothesis_id=hypothesis_id,
            )
        try:
            store.get_plan(plan_id)
        except KeyError:
            store.create_plan(
                idea_id, "phase3_matched_event_study_v1",
                "full_controls_5d_daily_mean_paired_excess",
                hypothesis_id=hypothesis_id, plan_id=plan_id,
            )
        try:
            store.get_trial(trial_id)
        except KeyError:
            store.record_trial(
                plan_id, "validation", "RUNNING", external_run_id=run_id,
                artifact_path=cfg.output_root / run_id, trial_id=trial_id,
            )
        return idea_id, hypothesis_id, plan_id, trial_id

    @staticmethod
    def _write_artifacts(
        output: Path, cfg: EventStudyConfig, config_hash: str, card: ObservationCard,
        source_manifest: dict, label_version: str, e0: dict, summary: dict,
        matched: dict[str, pd.DataFrame], tables: dict[str, pd.DataFrame],
        mechanism_features: pd.DataFrame | None,
    ) -> None:
        if output.exists():
            raise FileExistsError(f"immutable event study already exists: {output}")
        output.mkdir(parents=True, exist_ok=False)
        created = utc_now()
        summary["created_at"] = created
        (output / "e0_audit.json").write_text(
            json.dumps(e0, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
        )
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
        )
        tables["progressive_controls"].to_csv(output / "e1_e3_progressive_controls.csv", index=False)
        tables["severity"].to_csv(output / "e2_severity_monotonicity.csv", index=False)
        tables["regime"].to_csv(output / "e4_regime_diagnostics.csv", index=False)
        if mechanism_features is not None:
            mechanism_features.to_csv(output / "mechanism_features.csv", index=False)
        matched_root = output / "matched_pairs"
        matched_root.mkdir()
        for stage, frame in matched.items():
            frame.to_parquet(matched_root / f"{stage}.parquet", index=False)
        paired_root = output / "paired_events"
        paired_root.mkdir()
        for name, frame in tables.items():
            if name.startswith("paired_"):
                frame.to_parquet(paired_root / f"{name.removeprefix('paired_')}.parquet", index=False)
        (output / "report.md").write_text(_report(summary, e0), encoding="utf-8")
        manifest = {
            "run_id": summary["run_id"], "runner_type": "radar_event_studies",
            "status": "COMPLETED", "started_at": created, "finished_at": created,
            "engine_version": EVENT_STUDY_ENGINE_VERSION,
            "observation_id": card.observation_id,
            "observation_card_sha256": source_manifest["card_sha256"],
            "label_data_version": label_version, "config_hash": config_hash,
            "primary_stage": cfg.inference.primary_stage,
            "primary_horizon": cfg.inference.primary_horizon,
            "gate": summary["gate"],
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _cached_result(output, run_id, config_hash, source_manifest, label_version):
        manifest_path, summary_path = output / "manifest.json", output / "summary.json"
        if not manifest_path.exists() or not summary_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            manifest.get("run_id") != run_id
            or manifest.get("engine_version") != EVENT_STUDY_ENGINE_VERSION
            or manifest.get("config_hash") != config_hash
            or manifest.get("observation_card_sha256") != source_manifest["card_sha256"]
            or manifest.get("label_data_version") != label_version
        ):
            raise ValueError(f"event study cache identity mismatch: {output}")
        return {
            "run_id": run_id, "artifact_path": str(output.resolve()),
            "observation_id": summary["observation_id"], "label_data_version": label_version,
            "primary_result": summary["primary_result"], "gate": summary["gate"],
            "_trial_id": summary["lineage"]["trial_id"], "cached": True,
        }


def _report(summary: dict, e0: dict) -> str:
    primary, gate = summary["primary_result"], summary["gate"]
    return (
        "# Matched Event Study\n\n"
        f"- Run: `{summary['run_id']}`\n"
        f"- Observation: `{summary['observation_id']}`\n"
        f"- Frozen audit: `{e0['status']}`\n"
        f"- Primary: `{summary['primary_metric']['stage']}` / `{summary['primary_metric']['horizon']}D`\n"
        f"- Mature matched events: `{primary.get('mature_matched_events')}`\n"
        f"- Daily equal-weight mean paired excess: `{primary.get('daily_equal_weight_mean_paired_excess')}`\n"
        f"- Event-weighted mean paired excess: `{primary.get('event_weighted_mean_paired_excess')}`\n"
        f"- Newey-West t: `{primary.get('nw_t_value')}`\n"
        f"- FDR q: `{primary.get('fdr_q')}`\n"
        f"- Gate: `{gate['status']}` -> `{gate['next_action']}`\n\n"
        "The source ObservationCard remains label-free and unchanged. Labels exist only in this artifact.\n"
    )


def _json_default(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp): return value.isoformat()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)
