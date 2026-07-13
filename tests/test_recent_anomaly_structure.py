from __future__ import annotations

import numpy as np
import pandas as pd

from factor_forge.ml.recent_anomaly_config import load_recent_anomaly_structure_config
from factor_forge.ml.recent_anomaly_structure import (
    assert_fold_label_maturity,
    attach_label_available_date,
    build_pit_recent_structure_features,
    build_walk_forward_folds,
)


def test_repository_recent_structure_config_is_frozen():
    cfg = load_recent_anomaly_structure_config(
        "configs/ml/event_rankers/recent_anomaly_structure_v1.yaml"
    )
    assert cfg.primary_horizon == 5
    assert cfg.recent_structure.efficacy_windows == [20, 60, 120]
    assert cfg.walk_forward.step_days == cfg.walk_forward.test_days
    assert cfg.walk_forward.maximum_folds == 3
    assert len(cfg.event_templates) == 7
    assert cfg.training.random_seeds == [17]


def test_label_availability_is_horizon_plus_one_trading_days():
    dates = pd.bdate_range("2026-01-02", periods=12)
    episodes = pd.DataFrame({"trade_date": [dates[0], dates[5], dates[8]]})
    result = attach_label_available_date(episodes, dates, horizon=5)
    assert result.loc[0, "label_available_date"] == dates[6]
    assert result.loc[1, "label_available_date"] == dates[11]
    assert pd.isna(result.loc[2, "label_available_date"])


def test_recent_structure_uses_only_labels_mature_by_signal_date():
    dates = pd.bdate_range("2026-01-02", periods=30)
    episodes = pd.DataFrame({
        "episode_id": [f"E{i}" for i in range(6)],
        "trade_date": [dates[0], dates[1], dates[2], dates[3], dates[10], dates[20]],
        "template_id": ["A", "A", "A", "A", "A", "A"],
        "target": [1.0, -1.0, 3.0, 1000.0, 7.0, -9999.0],
        "severity": [1, 2, 3, 4, 5, 6],
    })
    episodes = attach_label_available_date(episodes, dates, horizon=2)
    result, names = build_pit_recent_structure_features(
        episodes, calendar=dates, windows=[20], factor_columns=["severity"],
        minimum_mature_events=2,
    )
    # At dates[10], the first four labels have matured. Its own target cannot leak in.
    row = result.loc[result["trade_date"].eq(dates[10])].iloc[0]
    assert row["template_mature_count_20"] == 4
    assert row["template_target_mean_20"] == (1.0 - 1.0 + 3.0 + 1000.0) / 4
    # The live-like last row's own extreme target is never part of its state.
    last = result.loc[result["trade_date"].eq(dates[20])].iloc[0]
    assert last["template_target_mean_20"] != -9999.0
    assert "factor_ic_severity_20" in names


def test_appending_unmatured_future_target_cannot_change_existing_state():
    dates = pd.bdate_range("2026-01-02", periods=40)
    base = pd.DataFrame({
        "trade_date": dates[:8], "template_id": "A",
        "target": np.arange(8, dtype=float), "severity": np.arange(8, dtype=float),
    })
    base = attach_label_available_date(base, dates, horizon=5)
    left, columns = build_pit_recent_structure_features(
        base, calendar=dates, windows=[20], factor_columns=["severity"],
        minimum_mature_events=2,
    )
    future = pd.DataFrame({
        "trade_date": [dates[30]], "template_id": ["A"],
        "target": [1e9], "severity": [1e9],
    })
    future = attach_label_available_date(future, dates, horizon=5)
    right, _ = build_pit_recent_structure_features(
        pd.concat([base, future], ignore_index=True), calendar=dates,
        windows=[20], factor_columns=["severity"], minimum_mature_events=2,
    )
    pd.testing.assert_frame_equal(left[columns], right.iloc[: len(left)][columns])


def test_walk_forward_has_horizon_embargo_and_mature_labels():
    dates = pd.bdate_range("2024-01-02", periods=260)
    folds = build_walk_forward_folds(
        dates, training_days=126, validation_days=40, test_days=20,
        step_days=20, horizon=5,
    )
    assert folds
    first = folds[0]
    assert dates.get_loc(first.valid_start) - dates.get_loc(first.train_end) == 7
    assert dates.get_loc(first.test_start) - dates.get_loc(first.valid_end) == 7
    episodes = pd.DataFrame({"trade_date": dates[:200]})
    episodes = attach_label_available_date(episodes, dates, horizon=5)
    assert_fold_label_maturity(episodes, first)
