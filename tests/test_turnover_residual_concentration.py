from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import make_panel
from factor_forge.radar.concentration_features import (
    TURNOVER_RESIDUAL_CONCENTRATION_FIELDS,
    build_turnover_residual_concentration_features,
    turnover_residual_concentration_prefix_audit,
)
from factor_forge.radar.models import ObservationCard
from factor_forge.radar.scanner import RelationAnomalyScanner
from factor_forge.radar.templates import load_radar_template


def _panel(days: int = 45, stocks: int = 12) -> pd.DataFrame:
    panel = make_panel(days=days, stocks=stocks)
    stock = panel["ts_code"].str[:6].astype(int) + 1
    day = panel.groupby("ts_code").cumcount() + 1
    cycle = 1 + 0.25 * np.sin(day / 3 + stock)
    panel["amount_cny"] = (2e7 + stock * 3e6) * cycle
    panel["listing_trade_days"] = 300 + day
    return panel


def _kwargs() -> dict:
    return {
        "liquidity_window": 5,
        "liquidity_min_periods": 3,
        "history_window": 10,
        "history_min_periods": 5,
        "persistence_window": 3,
        "contributor_percentile": 0.80,
        "concentration_percentile": 0.80,
        "min_cross_section": 5,
    }


def test_eight_residual_concentration_measurements_are_label_free_and_finite():
    result = build_turnover_residual_concentration_features(_panel(), **_kwargs())
    assert set(TURNOVER_RESIDUAL_CONCENTRATION_FIELDS) <= set(result.columns)
    assert len(TURNOVER_RESIDUAL_CONCENTRATION_FIELDS) == 8
    assert not any("forward" in column or "label" in column for column in result.columns)
    tail = result.loc[result["residual_concentration_history_percentile"].notna()]
    assert len(tail) > 0
    for field in TURNOVER_RESIDUAL_CONCENTRATION_FIELDS:
        assert pd.to_numeric(tail[field], errors="coerce").notna().any()


def test_residual_concentration_passes_strict_pit_prefix_audit():
    assert turnover_residual_concentration_prefix_audit(_panel(), **_kwargs()) is True


def test_future_mutation_cannot_change_prior_residual_features():
    panel = _panel()
    cutoff = pd.Timestamp(sorted(panel["trade_date"].unique())[29])
    original = build_turnover_residual_concentration_features(panel, **_kwargs())
    mutated = panel.copy()
    future = pd.to_datetime(mutated["trade_date"]).gt(cutoff)
    mutated.loc[future, "amount_cny"] *= np.linspace(1.5, 25.0, future.sum())
    mutated.loc[future, "adj_close"] *= np.linspace(0.7, 1.4, future.sum())
    changed = build_turnover_residual_concentration_features(mutated, **_kwargs())
    keys = ["trade_date", "ts_code"]
    fields = [*keys, *TURNOVER_RESIDUAL_CONCENTRATION_FIELDS]
    left = original.loc[original["trade_date"].le(cutoff), fields].sort_values(keys).reset_index(drop=True)
    right = changed.loc[changed["trade_date"].le(cutoff), fields].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_frozen_v2_template_emits_strictly_label_free_observation_card():
    panel = _panel(days=45, stocks=12)
    template = load_radar_template("configs/radar/turnover_residual_concentration_v2.yaml")
    template.scan.discovery_window_days = 20
    template.scan.recent_window_days = 5
    template.parameters.history.window = 10
    template.parameters.history.min_periods = 5
    template.parameters.long_window = 5
    template.parameters.liquidity_min_periods = 3
    template.parameters.upper_percentile = 0.80
    template.parameters.min_cross_section = 5
    template.quality_gate.min_events = 0
    result = RelationAnomalyScanner().scan(
        panel, template, data_version="data_test", as_of_date=panel["trade_date"].max()
    )
    restored = ObservationCard.model_validate_json(result.card.model_dump_json())
    assert restored.quality.future_label_fields_present is False
    assert restored.quality.temporal_audit_passed is True
    assert set(TURNOVER_RESIDUAL_CONCENTRATION_FIELDS) <= set(restored.event_fields)
