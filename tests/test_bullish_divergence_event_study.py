from __future__ import annotations

import numpy as np
import pandas as pd

from factor_forge.research.bullish_divergence_event_study import (
    BullishDivergenceEventStudyConfig,
    build_labels_and_controls,
    classify_touch_state,
    run_bullish_divergence_event_study,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    from conftest import make_panel

    panel = make_panel(days=130, stocks=12)
    date = pd.Timestamp(sorted(panel["trade_date"].unique())[75])
    daily = panel[["trade_date", "ts_code"]].copy()
    daily["div__event_candidate"] = False
    event_codes = ["000000.SZ", "000002.SZ"]
    daily.loc[
        daily["trade_date"].eq(date) & daily["ts_code"].isin(event_codes),
        "div__event_candidate",
    ] = True
    episodes = pd.DataFrame({
        "trade_date": [date, date],
        "ts_code": event_codes,
        "event_id": [f"{code}:{date:%Y%m%d}" for code in event_codes],
        "episode_id": [f"{code}:{date:%Y%m%d}" for code in event_codes],
        "div__score": [80.0, 60.0],
        "div__score_rank": [0.90, 0.70],
        "touch__occurred_10d": [0.0, 1.0],
        "touch__pre_b_count": [0.0, 0.0],
        "touch__post_b_count": [0.0, 1.0],
        "touch__last_close_reclaim_atr": [np.nan, 0.5],
        "touch__false_break_reclaim": [0.0, 0.0],
        "touch__acceptance_score": [0.0, 75.0],
        "touch__level_raw": [10.0, 12.0],
    })
    return panel, daily, episodes, date


def test_labels_use_next_open_and_industry_leave_one_out():
    panel, _, _, date = _inputs()
    config = BullishDivergenceEventStudyConfig(bootstrap_samples=20)
    labeled = build_labels_and_controls(panel, config)
    row = labeled.loc[
        labeled["trade_date"].eq(date) & labeled["ts_code"].eq("000000.SZ")
    ].iloc[0]
    stock = panel.loc[panel["ts_code"].eq("000000.SZ")].sort_values("trade_date")
    position = stock["trade_date"].tolist().index(date)
    expected = stock.iloc[position + 11]["adj_open"] / stock.iloc[position + 1]["adj_open"] - 1
    assert np.isclose(row["label__return_10d"], expected)
    assert np.isfinite(row["label__industry_excess_10d"])
    assert np.isclose(
        row["label__industry_excess_net_10d"],
        row["label__industry_excess_10d"] - 0.004,
    )
    assert row["label__mae_atr_10d"] <= row["label__mfe_atr_10d"]


def test_touch_state_classification_is_mutually_exclusive():
    frame = pd.DataFrame({
        "touch__occurred_10d": [0, 1, 1, 1],
        "touch__last_close_reclaim_atr": [np.nan, -0.1, 0.2, 0.3],
        "touch__false_break_reclaim": [0, 0, 0, 1],
    })
    assert classify_touch_state(frame).tolist() == [
        "U0_no_touch", "U1_touch_no_reclaim", "U2_touch_reclaim", "U3_false_break_reclaim",
    ]


def test_matched_event_study_uses_same_date_industry_and_excludes_raw_events():
    panel, daily, episodes, date = _inputs()
    config = BullishDivergenceEventStudyConfig(
        neighbors=2, caliper=10.0, bootstrap_samples=50, minimum_industry_size=3
    )
    result = run_bullish_divergence_event_study(panel, daily, episodes, config)
    pairs = result.matched_pairs
    assert len(pairs) == 4
    assert pairs["trade_date"].eq(date).all()
    assert pairs["industry_l1_code"].eq("I0").all()
    assert not set(pairs["control_code"]) & {"000000.SZ", "000002.SZ"}
    assert result.paired_events["event_id"].nunique() == 2
    assert set(result.touch_summary["touch_state"]) == {"U0_no_touch", "U2_touch_reclaim"}
    assert len(result.matching_balance) == 8
    assert result.summary["mature_episode_count"] == 2


def test_future_mutation_does_not_change_event_date_controls():
    panel, _, _, date = _inputs()
    config = BullishDivergenceEventStudyConfig(bootstrap_samples=20)
    before = build_labels_and_controls(panel, config)
    mutated = panel.copy()
    future = mutated["trade_date"].gt(date)
    # Labels should change, but matching controls frozen at T must not.
    mutated.loc[future, "adj_close"] *= 2.0
    after = build_labels_and_controls(mutated, config)
    left = before.loc[before["trade_date"].eq(date)].sort_values("ts_code")
    right = after.loc[after["trade_date"].eq(date)].sort_values("ts_code")
    assert np.allclose(left[[*result_control_fields()]], right[[*result_control_fields()]], equal_nan=True)


def result_control_fields() -> list[str]:
    from factor_forge.research.bullish_divergence_event_study import CONTROL_FIELDS

    return CONTROL_FIELDS

