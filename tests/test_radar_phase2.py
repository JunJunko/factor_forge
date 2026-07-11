from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from factor_forge.radar.models import ObservationCard
from factor_forge.radar.percentiles import pit_rolling_percentile, temporal_prefix_audit
from factor_forge.radar.scanner import RelationAnomalyScanner
from factor_forge.radar.templates import RADAR_TEMPLATE_ADAPTER, load_radar_template
from factor_forge.radar.writer import ObservationWriter
from factor_forge.research_control import ResearchControlStore


def _panel(days: int = 90, stocks: int = 4) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=days)
    rows = []
    for stock in range(stocks):
        prices = np.full(days, 10.0 + stock)
        prices += np.arange(days) * 0.002
        volumes = np.full(days, 1_000_000.0 + stock * 10_000)
        if stock == 0:
            prices[60:] -= 1.5  # extreme 3-day drop without a volume surge
        if stock == 1:
            volumes[70] = 9_000_000.0  # volume surge with nearly unchanged price
        for index, date in enumerate(dates):
            rows.append({
                "trade_date": date,
                "ts_code": f"{stock:06d}.SZ",
                "industry_l1_code": f"I{stock % 2}",
                "adj_close": prices[index],
                "volume_shares": volumes[index],
                "is_liquid": True,
            })
    return pd.DataFrame(rows)


def _template(kind: str):
    common = {
        "version": 1,
        "id": f"{kind}_test",
        "kind": kind,
        "description": "unit test",
        "observation_type": "relation_anomaly",
        "data": {
            "required_fields": ["adj_close", "volume_shares"],
            "universe_field": "is_liquid",
            "entity_field": "ts_code",
            "date_field": "trade_date",
            "industry_field": "industry_l1_code",
        },
        "scan": {"discovery_window_days": 60, "recent_window_days": 10},
    }
    if kind == "price_drop_without_volume_confirmation":
        common["parameters"] = {
            "return_horizon": 3,
            "return_history": {"window": 40, "min_periods": 10},
            "volume_history": {"window": 20, "min_periods": 10},
            "return_percentile_lte": 0.10,
            "volume_percentile_lte": 0.60,
        }
    else:
        common["parameters"] = {
            "return_horizon": 1,
            "abs_return_history": {"window": 40, "min_periods": 10},
            "volume_history": {"window": 20, "min_periods": 10},
            "volume_percentile_gte": 0.90,
            "abs_return_percentile_lte": 0.60,
        }
    return RADAR_TEMPLATE_ADAPTER.validate_python(common)


def test_pit_percentile_uses_strict_prior_rows_and_future_append_is_invariant():
    frame = pd.DataFrame({
        "trade_date": pd.bdate_range("2026-01-01", periods=4),
        "ts_code": ["A"] * 4,
        "value": [1.0, 2.0, 2.0, 0.0],
    })
    result = pit_rolling_percentile(frame, "value", window=3, min_periods=2)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(0.75)
    assert result.iloc[3] == pytest.approx(0.0)

    extended = pd.concat([
        frame,
        pd.DataFrame({"trade_date": [pd.Timestamp("2026-01-07")], "ts_code": ["A"], "value": [99.0]}),
    ], ignore_index=True)
    extended_result = pit_rolling_percentile(extended, "value", window=3, min_periods=2)
    np.testing.assert_allclose(result, extended_result.iloc[:4], equal_nan=True)
    assert temporal_prefix_audit(
        extended, "value", entity_column="ts_code", date_column="trade_date",
        window=3, min_periods=2,
    )


@pytest.mark.parametrize("kind", [
    "price_drop_without_volume_confirmation",
    "volume_surge_without_price_impact",
])
def test_frozen_relation_templates_emit_label_free_observation_cards(kind):
    result = RelationAnomalyScanner().scan(
        _panel(), _template(kind), data_version="data_test", as_of_date="2025-05-07"
    )
    payload = result.card.model_dump(mode="json")
    serialized = json.dumps(payload).lower()
    assert result.card.evidence.event_count > 0
    assert result.card.quality.temporal_audit_passed is True
    assert result.card.quality.future_label_fields_present is False
    assert "forward_return" not in serialized
    assert "rank_ic" not in serialized
    assert "sharpe" not in serialized
    assert set(result.events.columns) == set(result.card.event_fields)
    assert result.events["is_event"].all()


def test_observation_card_schema_rejects_future_label_in_open_condition_map():
    valid = RelationAnomalyScanner().scan(
        _panel(), _template("price_drop_without_volume_confirmation"),
        data_version="data_test",
    ).card.model_dump(mode="json")
    valid["conditions"]["forward_return"] = 0.02
    with pytest.raises(ValidationError, match="future-label field"):
        ObservationCard.model_validate(valid)
    valid["conditions"].pop("forward_return")
    valid["conditions"]["future_5d"] = 0.02
    with pytest.raises(ValidationError, match="future-label field"):
        ObservationCard.model_validate(valid)


def test_observation_card_json_round_trip_keeps_explicit_no_label_audit_field():
    card = RelationAnomalyScanner().scan(
        _panel(), _template("volume_surge_without_price_impact"), data_version="data_test"
    ).card
    restored = ObservationCard.model_validate_json(card.model_dump_json())
    assert restored.quality.future_label_fields_present is False


def test_observation_writer_and_registry_are_immutable_and_idempotent(tmp_path):
    result = RelationAnomalyScanner().scan(
        _panel(), _template("volume_surge_without_price_impact"),
        data_version="data_test",
    )
    writer = ObservationWriter(tmp_path / "observations")
    first = writer.write(result)
    second = writer.write(result)
    assert first.artifact_path == second.artifact_path
    assert first.card_sha256 == second.card_sha256
    assert first.events_path.exists()
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["contains_future_labels"] is False

    store = ResearchControlStore(tmp_path / "research.sqlite3")
    store.initialize()
    registered = store.register_observation(
        observation_id=result.card.observation_id,
        template_id=result.card.definition.id,
        definition_hash=result.card.definition.definition_hash,
        data_version=result.card.data_version,
        as_of_date=result.card.as_of_date,
        artifact_path=first.artifact_path,
        card_sha256=first.card_sha256,
        discovered_at=result.card.discovered_at,
    )
    repeated = store.register_observation(
        observation_id=result.card.observation_id,
        template_id=result.card.definition.id,
        definition_hash=result.card.definition.definition_hash,
        data_version=result.card.data_version,
        as_of_date=result.card.as_of_date,
        artifact_path=first.artifact_path,
        card_sha256=first.card_sha256,
        discovered_at=result.card.discovered_at,
    )
    assert registered["observation_id"] == repeated["observation_id"]


def test_repository_templates_validate():
    first = load_radar_template("configs/radar/price_drop_without_volume_confirmation_v1.yaml")
    second = load_radar_template("configs/radar/volume_surge_without_price_impact_v1.yaml")
    assert first.definition_hash() != second.definition_hash()
