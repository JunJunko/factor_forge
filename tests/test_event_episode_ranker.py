from __future__ import annotations

import numpy as np
import pandas as pd

from factor_forge.event_study.matching import match_episode_anchors
from factor_forge.ml.event_episode_config import load_event_episode_config
from factor_forge.ml.event_episode_dataset import deduplicate_event_episodes


def test_repository_event_episode_config_is_frozen_to_one_template():
    cfg = load_event_episode_config(
        "configs/ml/event_rankers/price_drop_without_volume_episode_v1.yaml"
    )
    assert cfg.horizons == [1, 3, 5, 10]
    assert cfg.primary_horizon == 5
    assert cfg.episode.dedup_trading_days == 5
    assert cfg.template.name == "price_drop_without_volume_confirmation_v1.yaml"


def test_episode_dedup_uses_anchored_five_trading_day_window():
    dates = pd.bdate_range("2026-01-02", periods=12)
    trigger_indices = [0, 1, 4, 5, 6, 10]
    rows = []
    for code in ["A", "B"]:
        for index, date in enumerate(dates):
            event = index in trigger_indices if code == "A" else index == 1
            rows.append({
                "trade_date": date, "ts_code": code,
                "template__event": float(event),
                "template__severity": float(index + 1) if event else 0.0,
            })
    _, anchors = deduplicate_event_episodes(
        pd.DataFrame(rows), template_id="template", definition_hash="hash",
        dedup_trading_days=5,
    )
    a = anchors.loc[anchors["ts_code"].eq("A")].reset_index(drop=True)
    assert list(a["trade_date"]) == [dates[0], dates[5], dates[10]]
    assert list(a["diagnostic_trigger_count"]) == [3, 2, 1]
    assert list(a["severity"]) == [1.0, 6.0, 11.0]
    assert list(a["diagnostic_max_severity"]) == [5.0, 7.0, 11.0]
    assert anchors.loc[anchors["ts_code"].eq("B"), "trade_date"].iloc[0] == dates[1]


def test_raw_non_anchor_events_are_never_episode_controls():
    cfg = load_event_episode_config(
        "configs/ml/event_rankers/price_drop_without_volume_episode_v1.yaml"
    )
    date = pd.Timestamp("2026-01-05")
    rows = []
    for index, code in enumerate(["A", "B", "C", "D", "E"]):
        row = {
            "trade_date": date, "ts_code": code, "industry_l1_code": "I1",
            "is_liquid": True, "is_raw_event": code in {"A", "B"},
            "is_episode_anchor": code == "A", "severity": 1.0 if code == "A" else np.nan,
            "prior_return_5d": index * 0.01, "volatility_20d": 0.02 + index * 0.001,
            "log_avg_amount_20d": 18 + index * 0.1, "log_total_mv": 20 + index * 0.1,
        }
        for horizon in cfg.horizons:
            row[f"forward_return_{horizon}"] = index * 0.001 + horizon * 0.0001
            row[f"label_mature_{horizon}"] = True
        rows.append(row)
    pairs = match_episode_anchors(pd.DataFrame(rows), cfg.matching, cfg.horizons)
    assert len(pairs) > 0
    assert set(pairs["event_code"]) == {"A"}
    assert "B" not in set(pairs["control_code"])
    assert set(pairs["control_code"]) <= {"C", "D", "E"}

