from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from conftest import make_panel
from factor_forge.event_study.mechanism_features import (
    TURNOVER_CONCENTRATION_AGGREGATE_FIELDS,
    build_turnover_concentration_aggregate_features,
    turnover_concentration_prefix_audit,
)
from factor_forge.event_study.runner import EventStudyRunner
from factor_forge.research_control import ResearchControlStore
from factor_forge.research_control.store import ResearchControlError


def _panel(days: int = 30) -> pd.DataFrame:
    panel = make_panel(days=days, stocks=4)
    stock_number = panel["ts_code"].str[:6].astype(int) + 1
    day_number = panel.groupby("ts_code").cumcount() + 1
    panel["amount_cny"] = stock_number * day_number * 1_000_000.0
    return panel


def _kwargs() -> dict:
    return {
        "history_window": 10,
        "history_min_periods": 5,
        "persistence_window": 3,
        "top_amount_fraction": 0.25,
        "top_size_fraction": 0.25,
        "concentration_threshold": 0.70,
    }


def test_aggregate_features_have_auditable_cross_section_semantics():
    result = build_turnover_concentration_aggregate_features(_panel(), **_kwargs())
    row = result.iloc[-1]
    assert row["top5_amount_share"] == pytest.approx(0.7)
    assert row["industry_amount_hhi"] == pytest.approx(0.52)
    assert row["top_size_decile_amount_share"] == pytest.approx(0.7)
    assert 0 <= row["contributor_return_sign_coherence"] <= 1
    assert 0 <= row["return_contribution_concentration"] <= 1
    assert row["industry_unknown_amount_share"] == 0
    assert set(TURNOVER_CONCENTRATION_AGGREGATE_FIELDS) <= set(result.columns)
    assert not any("forward" in column or "label" in column for column in result.columns)


def test_aggregate_features_pass_strict_pit_prefix_audit():
    panel = _panel()
    assert turnover_concentration_prefix_audit(panel, **_kwargs()) is True


def test_future_mutation_cannot_change_earlier_aggregate_features():
    panel = _panel()
    cutoff = pd.Timestamp(sorted(panel["trade_date"].unique())[19])
    original = build_turnover_concentration_aggregate_features(panel, **_kwargs())
    mutated = panel.copy()
    future = pd.to_datetime(mutated["trade_date"]).gt(cutoff)
    mutated.loc[future, "amount_cny"] *= np.linspace(2.0, 20.0, future.sum())
    mutated.loc[future, "adj_close"] *= np.linspace(0.5, 1.5, future.sum())
    changed = build_turnover_concentration_aggregate_features(mutated, **_kwargs())
    fields = ["trade_date", *TURNOVER_CONCENTRATION_AGGREGATE_FIELDS]
    left = original.loc[original["trade_date"].le(cutoff), fields].reset_index(drop=True)
    right = changed.loc[changed["trade_date"].le(cutoff), fields].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_aggregate_features_reject_duplicate_stock_dates():
    panel = _panel()
    duplicate = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique stock/date"):
        build_turnover_concentration_aggregate_features(duplicate, **_kwargs())


def test_event_study_binds_to_pre_registered_lineage(tmp_path):
    store = ResearchControlStore(tmp_path / "research.sqlite3")
    store.initialize()
    store.create_idea(
        "Concentration", "Competing mechanisms", "concentration",
        idea_id="idea_concentration", target_horizon=5,
    )
    store.set_idea_status("idea_concentration", "ACTIVE")
    store.add_hypothesis(
        "idea_concentration", "Information-driven concentration",
        hypothesis_id="hyp_concentration",
    )
    store.create_plan(
        "idea_concentration", "phase3", "full_controls_5d_daily_mean_paired_excess",
        hypothesis_id="hyp_concentration", plan_id="plan_concentration",
    )
    cfg = SimpleNamespace(
        output_root=tmp_path / "events",
        lineage=SimpleNamespace(
            idea_id="idea_concentration",
            hypothesis_id="hyp_concentration",
            plan_id="plan_concentration",
            trial_id="trial_concentration",
            primary_metric="full_controls_5d_daily_mean_paired_excess",
        ),
    )
    lineage = EventStudyRunner._start_lineage(store, None, "event_run", cfg)
    assert lineage == (
        "idea_concentration", "hyp_concentration", "plan_concentration", "trial_concentration"
    )
    assert store.get_trial("trial_concentration").data_role.value == "validation"
    with pytest.raises(ResearchControlError, match="already exists"):
        EventStudyRunner._start_lineage(store, None, "event_run", cfg)
