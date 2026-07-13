from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from factor_forge.config import load_project
from factor_forge.data import DataVersionRepository
from factor_forge.data.freshness import FreshnessPolicy, MarketDataFreshnessService
from factor_forge.data.tushare_provider import TushareProvider
from factor_forge.research_control import ArtifactIndexer, ResearchControlStore

from .drift import RelationDriftRunner
from .drift_models import DriftCard
from .runner import RadarRunner
from .models import ObservationCard


BATCH_IMPLEMENTATION_VERSION = 4


class FreshnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto_sync: bool = True
    require_current: bool = True
    data_ready_after: str = Field(default="18:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    min_last_day_rows: int = Field(default=1_000, ge=1)
    min_last_day_tradeable: int = Field(default=500, ge=1)
    min_last_day_liquid: int = Field(default=500, ge=1)
    max_required_missing_rate: float = Field(default=0.05, ge=0, le=1)


class MarketScanConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    name: str
    project_config: Path = Path("configs/project.yaml")
    event_templates: list[Path] = Field(default_factory=list, min_length=1)
    drift_templates: list[Path] = Field(default_factory=list, min_length=1)
    event_template_globs: list[str] = Field(default_factory=list)
    drift_template_globs: list[str] = Field(default_factory=list)
    template_excludes: list[str] = Field(default_factory=list)
    event_output_root: Path = Path("artifacts/radar_observations")
    drift_output_root: Path = Path("artifacts/radar_drifts")
    output_root: Path = Path("artifacts/market_anomaly_scans")
    dedup_jaccard_threshold: float = Field(default=0.70, ge=0, le=1)
    max_highlights: int = Field(default=5, ge=1, le=10)
    freshness: FreshnessConfig = Field(default_factory=FreshnessConfig)


def load_market_scan_config(path: str | Path) -> MarketScanConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    workspace = next(
        (parent for parent in [config_path.parent, *config_path.parents] if (parent / "pyproject.toml").exists()),
        Path.cwd(),
    )
    excludes = set(payload.get("template_excludes", []))

    def discover(explicit_key: str, glob_key: str) -> list[Path]:
        candidates = [workspace / item for item in payload.get(explicit_key, [])]
        for pattern in payload.get(glob_key, []):
            candidates.extend(workspace.glob(pattern))
        unique = {}
        for candidate in candidates:
            resolved = candidate.resolve()
            relative = resolved.relative_to(workspace).as_posix() if resolved.is_relative_to(workspace) else resolved.as_posix()
            if candidate.name in excludes or relative in excludes:
                continue
            unique[resolved.as_posix()] = resolved
        return [unique[key] for key in sorted(unique)]

    payload["event_templates"] = discover("event_templates", "event_template_globs")
    payload["drift_templates"] = discover("drift_templates", "drift_template_globs")
    config = MarketScanConfig.model_validate(payload)
    _validate_discovered_templates(config)
    return config


def _validate_discovered_templates(config: MarketScanConfig) -> None:
    from .drift_templates import load_drift_template
    from .templates import load_radar_template

    groups = [
        ("event", [load_radar_template(path) for path in config.event_templates]),
        ("drift", [load_drift_template(path) for path in config.drift_templates]),
    ]
    for label, templates in groups:
        ids = [template.id for template in templates]
        hashes = [template.definition_hash() for template in templates]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate discovered {label} template id")
        if len(hashes) != len(set(hashes)):
            raise ValueError(f"duplicate discovered {label} template definition")


class MarketAnomalyScanRunner:
    def run(
        self,
        config_path: str | Path = "configs/radar/latest_market_scan_v1.yaml",
        *,
        data_version: str = "latest",
        as_of_date: str | None = None,
        sync: bool | None = None,
    ) -> dict:
        cfg = load_market_scan_config(config_path)
        project = load_project(cfg.project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        freshness = self._freshness(cfg, project, repository, data_version, as_of_date, sync)
        if cfg.freshness.require_current and freshness["status"] == "STALE_OR_INCOMPLETE":
            raise RuntimeError(
                "freshness gate blocked anomaly scan: " + ", ".join(freshness["failures"])
            )
        if freshness["status"] == "CURRENT":
            data_version = freshness["data_version"]
        resolved, manifest = repository.load_manifest(data_version)
        as_of = pd.Timestamp(as_of_date or manifest["end_date"]).strftime("%Y-%m-%d")
        config_hash = hashlib.sha256(
            json.dumps({
                "implementation_version": BATCH_IMPLEMENTATION_VERSION,
                "config": cfg.model_dump(mode="json"),
            }, sort_keys=True).encode("utf-8")
        ).hexdigest()
        scan_id = "market_scan_" + hashlib.sha256(
            f"{config_hash}|{resolved}|{as_of}".encode("utf-8")
        ).hexdigest()[:16]
        output = cfg.output_root / scan_id
        summary_path = output / "scan_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary["data_version"] != resolved or summary["as_of_date"] != as_of:
                raise ValueError(f"market scan cache identity mismatch: {output}")
            return {**summary, "artifact_path": str(output.resolve()), "cached": True}

        event_rows, current_sets = [], {}
        for path in cfg.event_templates:
            run = RadarRunner().run(
                path, project_config=cfg.project_config, data_version=resolved,
                as_of_date=as_of, output_root=cfg.event_output_root,
            )
            card = ObservationCard.model_validate_json(Path(run["card_path"]).read_text(encoding="utf-8"))
            events = pd.read_parquet(run["events_path"], columns=["trade_date", "ts_code", "severity"])
            current = events.loc[pd.to_datetime(events["trade_date"]).eq(pd.Timestamp(as_of))]
            current_sets[card.definition.id] = set(current["ts_code"].astype(str))
            zscore = card.evidence.rolling_event_rate_zscore
            ratio = card.evidence.event_rate_ratio
            ratio_score = abs(math.log(float(ratio))) if ratio is not None and ratio > 0 else 0.0
            score = max(abs(zscore) if zscore is not None else 0.0, ratio_score)
            event_rows.append({
                "template_id": card.definition.id,
                "kind": card.definition.kind,
                "description": card.definition.description,
                "card_path": run["card_path"],
                "events_path": run["events_path"],
                "quality_gate_passed": card.quality.quality_gate_passed,
                "quality_gate_failures": card.quality.quality_gate_failures,
                "scan_date_event_count": card.evidence.scan_date_event_count,
                "scan_date_event_rate": card.evidence.scan_date_event_rate,
                "rolling_event_rate_zscore": zscore,
                "recent_event_rate": card.evidence.recent_event_rate,
                "historical_event_rate": card.evidence.historical_event_rate,
                "event_rate_ratio": ratio,
                "unique_stocks": card.evidence.unique_entities,
                "unique_industries": card.evidence.unique_industries,
                "severity_p90": card.evidence.severity_p90,
                "priority_score": score,
                "duplicate_of": None,
            })
        self._deduplicate(event_rows, current_sets, cfg.dedup_jaccard_threshold)

        drift_rows = []
        for path in cfg.drift_templates:
            run = RelationDriftRunner().run(
                path, project_config=cfg.project_config, data_version=resolved,
                as_of_date=as_of, output_root=cfg.drift_output_root,
            )
            card_path = Path(run["artifact_path"]) / "drift_card.json"
            card = DriftCard.model_validate_json(card_path.read_text(encoding="utf-8"))
            drift_rows.append({
                "template_id": card.template_id,
                "card_path": str(card_path.resolve()),
                "quality_gate_passed": card.quality.quality_gate_passed,
                "quality_gate_failures": card.quality.quality_gate_failures,
                "drift_count": card.drift_count,
                "relations": [item.model_dump(mode="json") for item in card.relations],
            })

        ranked = sorted(
            [
                row for row in event_rows
                if row["quality_gate_passed"]
                and row["duplicate_of"] is None
                and row["scan_date_event_count"] > 0
            ],
            key=lambda row: (row["priority_score"], row["scan_date_event_count"]), reverse=True,
        )
        highlights = ranked[:cfg.max_highlights]
        summary = {
            "scan_id": scan_id, "implementation_version": BATCH_IMPLEMENTATION_VERSION,
            "data_version": resolved, "as_of_date": as_of,
            "freshness": freshness,
            "event_template_count": len(event_rows), "drift_template_count": len(drift_rows),
            "event_quality_pass_count": sum(row["quality_gate_passed"] for row in event_rows),
            "detected_drift_count": sum(row["drift_count"] for row in drift_rows),
            "highlights": highlights,
            "events": event_rows,
            "drifts": drift_rows,
            "interpretation_boundary": (
                "Event cards contain no future labels. Drift cards describe relationship change. "
                "Neither output is an Alpha or trading recommendation."
            ),
        }
        output.mkdir(parents=True, exist_ok=False)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "report.md").write_text(self._report(summary), encoding="utf-8")
        (output / "manifest.json").write_text(json.dumps({
            "run_id": scan_id, "runner_type": "market_anomaly_scans", "status": "COMPLETED",
            "data_version": resolved, "as_of_date": as_of,
            "event_templates": len(event_rows), "drift_templates": len(drift_rows),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        store = ResearchControlStore(project.paths.data_root / "research.sqlite3")
        store.initialize()
        ArtifactIndexer(store, cfg.output_root.parent).index()
        return {**summary, "artifact_path": str(output.resolve()), "cached": False}

    @staticmethod
    def _freshness(cfg, project, repository, data_version, as_of_date, sync):
        if as_of_date is not None or data_version != "latest":
            resolved, manifest = repository.load_manifest(data_version)
            return {
                "status": "PINNED", "expected_latest_trade_date": None,
                "data_end_date": manifest["end_date"], "data_version": resolved,
                "version_kind": manifest.get("version_kind", "legacy_complete"),
                "synchronized": False, "incremental_versions": [], "failures": [],
            }
        policy = FreshnessPolicy(
            data_ready_after=cfg.freshness.data_ready_after,
            min_last_day_rows=cfg.freshness.min_last_day_rows,
            min_last_day_tradeable=cfg.freshness.min_last_day_tradeable,
            min_last_day_liquid=cfg.freshness.min_last_day_liquid,
            max_required_missing_rate=cfg.freshness.max_required_missing_rate,
        )
        service = MarketDataFreshnessService(project, TushareProvider(), policy=policy)
        sync_enabled = cfg.freshness.auto_sync if sync is None else sync
        if sync_enabled:
            return service.ensure_current().to_dict()
        expected = service.expected_latest_trade_date()
        return service.audit(expected).to_dict()

    @staticmethod
    def _deduplicate(rows, current_sets, threshold):
        ordered = sorted(rows, key=lambda row: row["priority_score"], reverse=True)
        kept = []
        for row in ordered:
            current = current_sets[row["template_id"]]
            duplicate = None
            if current:
                for prior in kept:
                    other = current_sets[prior["template_id"]]
                    union = current | other
                    similarity = len(current & other) / len(union) if union else 0.0
                    if similarity >= threshold:
                        duplicate = prior["template_id"]
                        break
            row["duplicate_of"] = duplicate
            if duplicate is None:
                kept.append(row)

    @staticmethod
    def _report(summary):
        lines = [
            "# Latest Market Anomaly Scan", "",
            f"- Data version: `{summary['data_version']}`",
            f"- As of: `{summary['as_of_date']}`",
            f"- Freshness: `{summary['freshness']['status']}` "
            f"(expected={summary['freshness']['expected_latest_trade_date']}, "
            f"data_end={summary['freshness']['data_end_date']})",
            f"- Event templates: `{summary['event_template_count']}`",
            f"- Drift templates: `{summary['drift_template_count']}`",
            f"- Event quality gates passed: `{summary['event_quality_pass_count']}`",
            f"- Detected relation drifts: `{summary['detected_drift_count']}`", "",
            "## Highlights", "",
        ]
        if not summary["highlights"]:
            lines.append("No quality-passing current event anomaly was highlighted.")
        for row in summary["highlights"]:
            lines.append(
                f"- **{row['template_id']}**: current={row['scan_date_event_count']}, "
                f"rate_z={row['rolling_event_rate_zscore']}, ratio={row['event_rate_ratio']}"
            )
        lines.extend(["", "## Quality gate failures", ""])
        failed = [row for row in summary["events"] if not row["quality_gate_passed"]]
        if not failed:
            lines.append("All event templates passed their frozen quality gates.")
        for row in failed:
            lines.append(
                f"- **{row['template_id']}**: {', '.join(row['quality_gate_failures'])}"
            )
        lines.extend(["", "## Relation drift", ""])
        for group in summary["drifts"]:
            lines.append(f"- **{group['template_id']}**: drift_count={group['drift_count']}")
            for relation in group["relations"]:
                lines.append(
                    f"  - {relation['relation_id']}: drift={relation['is_drift']}, "
                    f"z={relation['robust_delta_zscore']}, cusum={relation['cusum_score']}, "
                    f"effective={relation['effective_as_of_date']}"
                )
        lines.extend(["", "> " + summary["interpretation_boundary"], ""])
        return "\n".join(lines)
