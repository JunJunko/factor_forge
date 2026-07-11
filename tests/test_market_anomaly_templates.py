from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from conftest import make_panel
from factor_forge.radar.models import ObservationCard
from factor_forge.radar.batch import MarketAnomalyScanRunner, load_market_scan_config
from factor_forge.radar.scanner import RelationAnomalyScanner
from factor_forge.radar.templates import load_radar_template


EVENT_TEMPLATE_PATHS = sorted(
    path for path in Path("configs/radar").glob("*_v1.yaml")
    if path.name != "latest_market_scan_v1.yaml"
)


def test_repository_contains_exactly_eight_frozen_event_templates():
    assert len(EVENT_TEMPLATE_PATHS) == 8
    templates = [load_radar_template(path) for path in EVENT_TEMPLATE_PATHS]
    assert len({template.id for template in templates}) == 8
    assert len({template.kind for template in templates}) == 8
    assert len({template.definition_hash() for template in templates}) == 8


def test_latest_bundle_contains_exactly_eight_events_and_two_drifts():
    config = load_market_scan_config("configs/radar/latest_market_scan_v1.yaml")
    assert len(config.event_templates) == 8
    assert len(config.drift_templates) == 2
    assert all(path.exists() for path in [*config.event_templates, *config.drift_templates])


def test_current_event_dedup_marks_only_high_jaccard_overlap():
    rows = [
        {"template_id": "a", "priority_score": 2.0, "duplicate_of": None},
        {"template_id": "b", "priority_score": 1.0, "duplicate_of": None},
        {"template_id": "c", "priority_score": 0.5, "duplicate_of": None},
    ]
    current = {"a": {"1", "2", "3"}, "b": {"1", "2", "3"}, "c": {"4"}}
    MarketAnomalyScanRunner._deduplicate(rows, current, 0.7)
    assert rows[1]["duplicate_of"] == "a"
    assert rows[2]["duplicate_of"] is None


def test_all_event_templates_emit_strictly_label_free_cards():
    panel = make_panel(days=390, stocks=30)
    cutoff = pd.Timestamp(panel["trade_date"].max())
    for path in EVENT_TEMPLATE_PATHS:
        template = load_radar_template(path)
        result = RelationAnomalyScanner().scan(
            panel, template, data_version="data_test", as_of_date=cutoff,
        )
        restored = ObservationCard.model_validate_json(result.card.model_dump_json())
        serialized = json.dumps(restored.model_dump(mode="json"), ensure_ascii=False).lower()
        assert restored.definition.definition_hash == template.definition_hash()
        assert restored.as_of_date == cutoff.strftime("%Y-%m-%d")
        assert restored.quality.future_label_fields_present is False
        assert restored.quality.strict_prior_history is True
        assert restored.quality.temporal_audit_passed is True
        assert "forward_return" not in serialized
        assert "rank_ic" not in serialized
        assert "sharpe" not in serialized
        assert set(result.events.columns) == set(restored.event_fields)
