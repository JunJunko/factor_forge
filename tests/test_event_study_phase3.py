from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from factor_forge.data import DataVersionRepository
from factor_forge.event_study import EventStudyRunner, load_event_study_config
from factor_forge.event_study.labels import build_point_in_time_features_and_labels
from factor_forge.event_study.matching import mark_frozen_events, match_all_stages
from factor_forge.radar.models import (
    ObservationCard,
    ObservationDefinition,
    ObservationEvidence,
    ObservationQuality,
    RadarScanResult,
)
from factor_forge.radar.writer import ObservationWriter
from factor_forge.research_control import ResearchControlStore
from factor_forge.research_control.models import utc_now
from conftest import make_panel


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _config(project_path: Path, observation_dir: Path, version: str, output: Path, db: Path) -> dict:
    return {
        "version": 1,
        "name": "phase3_test",
        "project_config": str(project_path),
        "observation_dir": str(observation_dir),
        "label_data_version": version,
        "horizons": [3, 5, 10],
        "universe_field": "is_liquid",
        "matching": {
            "exact": ["trade_date", "industry_l1_code"],
            "neighbors": 2,
            "caliper": 10.0,
            "allow_control_reuse": True,
            "min_match_rate": 0.5,
            "stages": [
                {"id": "same_date_industry", "controls": []},
                {"id": "prior_return_control", "controls": ["prior_return_5d"]},
                {"id": "return_volatility_controls", "controls": ["prior_return_5d", "volatility_20d"]},
                {"id": "full_controls", "controls": [
                    "prior_return_5d", "volatility_20d", "log_avg_amount_20d", "log_total_mv"
                ]},
            ],
        },
        "inference": {
            "primary_horizon": 5,
            "primary_stage": "full_controls",
            "nw_lag_rule": "horizon_minus_one",
            "fdr_alpha": 0.10,
            "min_mature_events": 20,
            "min_regime_events": 20,
            "severity_groups": 3,
        },
        "gate": {
            "min_abs_nw_t": 2.0,
            "max_fdr_q": 0.10,
            "min_daily_direction_ratio": 0.52,
            "max_abs_smd": 0.50,
        },
        "output_root": str(output),
        "research_db": str(db),
    }


def test_labels_follow_t1_open_semantics_and_mark_unmature_tail():
    dates = pd.bdate_range("2026-01-02", periods=15)
    panel = pd.DataFrame({
        "trade_date": dates,
        "ts_code": ["A"] * len(dates),
        "adj_open": range(10, 25),
        "adj_close": range(10, 25),
        "amount_cny": [1e8] * len(dates),
        "log_total_mv": [20.0] * len(dates),
        "industry_l1_code": ["I1"] * len(dates),
        "is_liquid": [True] * len(dates),
    })
    result = build_point_in_time_features_and_labels(panel, [3, 5, 10])
    assert result.loc[0, "forward_return_3"] == (14 / 11 - 1)
    assert bool(result.loc[0, "label_mature_10"]) is True
    assert bool(result.loc[len(result) - 10, "label_mature_10"]) is False


def test_matching_controls_are_same_date_industry_and_not_events(tmp_path):
    panel = make_panel(days=50, stocks=12)
    features = build_point_in_time_features_and_labels(panel, [3, 5, 10])
    date = pd.Timestamp(panel["trade_date"].sort_values().unique()[30])
    events = pd.DataFrame({
        "trade_date": [date, date], "ts_code": ["000000.SZ", "000002.SZ"],
        "severity": [1.0, 2.0],
    })
    marked = mark_frozen_events(features, events)
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _config(
        tmp_path / "project.yaml", tmp_path / "obs", "v", tmp_path / "out", tmp_path / "r.db"
    ))
    cfg = load_event_study_config(config_path)
    matched = match_all_stages(marked, cfg.matching, cfg.horizons)["full_controls"]
    assert len(matched) > 0
    assert matched["trade_date"].eq(date).all()
    assert matched["industry_l1_code"].eq("I0").all()
    assert not set(matched["control_code"]) & set(events["ts_code"])


def test_phase3_runner_preserves_source_card_and_registers_validation_trial(tmp_path):
    panel = make_panel(days=120, stocks=20)
    data_root, metadata_db = tmp_path / "data", tmp_path / "metadata.sqlite3"
    version = DataVersionRepository(data_root, metadata_db).publish(panel, source="test")
    project_path = tmp_path / "project.yaml"
    _write_yaml(project_path, {
        "project_name": "phase3_test", "timezone": "Asia/Shanghai",
        "paths": {
            "data_root": str(data_root), "metadata_db": str(metadata_db),
            "artifacts_root": str(tmp_path / "factor_runs"),
        },
    })

    dates = sorted(pd.to_datetime(panel["trade_date"]).unique())
    event_rows = []
    for date_index in [35, 45, 55, 65, 75, 85]:
        for stock in [0, 2, 4, 6]:
            event_rows.append({
                "trade_date": dates[date_index], "ts_code": f"{stock:06d}.SZ",
                "industry_l1_code": "I0", "severity": 1.0 + stock / 10,
                "is_event": True, "template_id": "manual_relation_v1",
            })
    events = pd.DataFrame(event_rows)
    definition_hash = hashlib.sha256(b"manual_relation_v1").hexdigest()
    card = ObservationCard(
        observation_id="obs_manual_relation_phase3",
        definition=ObservationDefinition(
            id="manual_relation_v1", version=1, kind="manual_relation",
            description="test frozen events", definition_hash=definition_hash,
        ),
        discovered_at=utc_now(), data_version=version,
        as_of_date=pd.Timestamp(dates[-1]).strftime("%Y-%m-%d"),
        universe="is_liquid", discovery_window_days=60, recent_window_days=10,
        conditions={"test_rule": 1},
        evidence=ObservationEvidence(
            event_count=len(events), unique_entities=4, unique_industries=1,
        ),
        quality=ObservationQuality(
            input_rows=len(panel), eligible_rows=len(panel), duplicate_keys=0,
            measurement_missing_rates={"severity": 0.0}, temporal_audit_passed=True,
        ),
        event_fields=list(events.columns),
    )
    observation = ObservationWriter(tmp_path / "observations").write(RadarScanResult(card, events))
    original_card_hash = hashlib.sha256(observation.card_path.read_bytes()).hexdigest()
    research_db = tmp_path / "research.sqlite3"
    store = ResearchControlStore(research_db)
    store.initialize()
    store.register_observation(
        observation_id=card.observation_id, template_id=card.definition.id,
        definition_hash=definition_hash, data_version=version, as_of_date=card.as_of_date,
        artifact_path=observation.artifact_path, card_sha256=observation.card_sha256,
        discovered_at=card.discovered_at,
    )
    config_path = tmp_path / "event_study.yaml"
    _write_yaml(config_path, _config(
        project_path, observation.artifact_path, version,
        tmp_path / "event_studies", research_db,
    ))

    result = EventStudyRunner().run(config_path)
    cached = EventStudyRunner().run(config_path)
    output = Path(result["artifact_path"])
    assert result["cached"] is False
    assert cached["cached"] is True
    assert (output / "e0_audit.json").exists()
    assert (output / "e1_e3_progressive_controls.csv").exists()
    assert (output / "paired_events" / "full_controls.parquet").exists()
    assert hashlib.sha256(observation.card_path.read_bytes()).hexdigest() == original_card_hash
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["primary_metric"]["stage"] == "full_controls"
    assert summary["primary_metric"]["horizon"] == 5
    primary = summary["primary_result"]
    assert primary["matched_events"] >= primary["mature_matched_events"]
    assert primary["match_rate"] == primary["matched_events"] / summary["source_event_count"]
    assert "daily_equal_weight_mean_paired_excess" in primary
    assert "event_weighted_mean_paired_excess" in primary
    assert summary["lineage"]["trial_id"]
    with store.connect() as connection:
        trial = connection.execute(
            "SELECT data_role,status,validation_peek FROM trial_run WHERE id=?",
            (summary["lineage"]["trial_id"],),
        ).fetchone()
        decision_count = connection.execute("SELECT COUNT(*) FROM research_decision").fetchone()[0]
    assert dict(trial) == {"data_role": "validation", "status": "SUCCESS", "validation_peek": 1}
    assert decision_count == 1


def test_repository_phase3_configs_validate():
    first = load_event_study_config("configs/event_studies/price_drop_without_volume_phase3_v1.yaml")
    second = load_event_study_config("configs/event_studies/volume_surge_without_impact_phase3_v1.yaml")
    assert first.inference.primary_stage == second.inference.primary_stage == "full_controls"
