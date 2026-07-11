from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from factor_forge.config import load_project
from factor_forge.data import DataVersionRepository
from factor_forge.research_control import ArtifactIndexer, ResearchControlStore

from .models import ObservationCard
from .scanner import RelationAnomalyScanner, observation_id_for
from .templates import filter_required_fields, load_radar_template, required_trading_rows
from .writer import ObservationWriter


class RadarRunner:
    def run(
        self,
        template_path: str | Path,
        *,
        project_config: str | Path = "configs/project.yaml",
        data_version: str = "latest",
        as_of_date: str | None = None,
        output_root: str | Path = "artifacts/radar_observations",
        research_db: str | Path | None = None,
    ) -> dict:
        project = load_project(project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        template = load_radar_template(template_path)
        resolved_version, manifest = repository.load_manifest(data_version)
        cutoff = pd.Timestamp(as_of_date or manifest["end_date"])
        observation_id = observation_id_for(template, resolved_version, cutoff)
        cached_dir = Path(output_root) / observation_id
        cached_card_path = cached_dir / "observation_card.json"
        cached_manifest_path = cached_dir / "manifest.json"
        cached_events_path = cached_dir / "events.parquet"
        store = ResearchControlStore(research_db or project.paths.data_root / "research.sqlite3")
        store.initialize()
        if cached_card_path.exists() and cached_manifest_path.exists() and cached_events_path.exists():
            card = ObservationCard.model_validate_json(cached_card_path.read_text(encoding="utf-8"))
            cached_manifest = json.loads(cached_manifest_path.read_text(encoding="utf-8"))
            actual_card_sha = hashlib.sha256(cached_card_path.read_bytes()).hexdigest()
            actual_events_sha = hashlib.sha256(cached_events_path.read_bytes()).hexdigest()
            if (
                card.observation_id != observation_id
                or card.definition.definition_hash != template.definition_hash()
                or card.data_version != resolved_version
                or card.as_of_date != cutoff.strftime("%Y-%m-%d")
                or cached_manifest.get("contains_future_labels") is not False
                or cached_manifest.get("card_sha256") != actual_card_sha
                or cached_manifest.get("events_sha256") != actual_events_sha
            ):
                raise ValueError(f"cached observation failed identity or label-free validation: {cached_dir}")
            store.register_observation(
                observation_id=card.observation_id, template_id=card.definition.id,
                definition_hash=card.definition.definition_hash, data_version=card.data_version,
                as_of_date=card.as_of_date, artifact_path=cached_dir,
                card_sha256=cached_manifest["card_sha256"], discovered_at=card.discovered_at,
                status=card.status,
            )
            ArtifactIndexer(store, Path(output_root).parent).index()
            return {
                "observation_id": observation_id,
                "artifact_path": str(cached_dir.resolve()),
                "card_path": str(cached_card_path.resolve()),
                "events_path": str(cached_events_path.resolve()),
                "event_count": int(cached_manifest["event_count"]),
                "data_version": resolved_version,
                "as_of_date": card.as_of_date,
                "input_rows": card.quality.input_rows,
                "input_trading_dates": None,
                "temporal_audit_passed": card.quality.temporal_audit_passed,
                "cached": True,
            }
        required_rows = required_trading_rows(template)
        calendar_days = math.ceil(required_rows * 7 / 5) + 90
        start = cutoff - pd.Timedelta(days=calendar_days)
        panel_path = (
            project.paths.data_root / "versions" / resolved_version
            / "curated" / "stock_daily_panel.parquet"
        )
        columns = list(dict.fromkeys([
            *template.data.required_fields,
            *filter_required_fields(template),
            template.data.entity_field,
            template.data.date_field,
            template.data.industry_field,
            template.data.universe_field,
        ]))
        panel = pd.read_parquet(
            panel_path,
            columns=columns,
            filters=[
                (template.data.date_field, ">=", start),
                (template.data.date_field, "<=", cutoff),
            ],
        )
        available_dates = pd.to_datetime(panel[template.data.date_field]).nunique()
        if available_dates < required_rows:
            raise ValueError(
                f"radar slice has {available_dates} trading dates but template requires "
                f"at least {required_rows}; data start={manifest['start_date']} cutoff={cutoff.date()}"
            )
        result = RelationAnomalyScanner().scan(
            panel, template, data_version=resolved_version, as_of_date=cutoff
        )
        artifact = ObservationWriter(output_root).write(result)
        store.register_observation(
            observation_id=result.card.observation_id,
            template_id=result.card.definition.id,
            definition_hash=result.card.definition.definition_hash,
            data_version=result.card.data_version,
            as_of_date=result.card.as_of_date,
            artifact_path=artifact.artifact_path,
            card_sha256=artifact.card_sha256,
            discovered_at=result.card.discovered_at,
            status=result.card.status,
        )
        ArtifactIndexer(store, Path(output_root).parent).index()
        return {
            "observation_id": result.card.observation_id,
            "artifact_path": str(artifact.artifact_path.resolve()),
            "card_path": str(artifact.card_path.resolve()),
            "events_path": str(artifact.events_path.resolve()),
            "event_count": artifact.event_count,
            "data_version": resolved_version,
            "as_of_date": result.card.as_of_date,
            "input_rows": len(panel),
            "input_trading_dates": int(available_dates),
            "temporal_audit_passed": result.card.quality.temporal_audit_passed,
            "cached": False,
        }
