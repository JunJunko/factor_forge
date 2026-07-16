import numpy as np
import pandas as pd

from factor_forge.research.narrative_rotation import attach_rotation_signals, narrative_stage


def test_narrative_stage_boundaries():
    assert narrative_stage(pd.Timestamp("2026-01-01")) == "spring_risk_on"
    assert narrative_stage(pd.Timestamp("2026-03-01")) == "shock_and_repair"
    assert narrative_stage(pd.Timestamp("2026-05-01")) == "technology_main_wave"
    assert narrative_stage(pd.Timestamp("2026-07-14")) == "broadening"
    assert narrative_stage(pd.Timestamp("2025-12-31")) is None


def test_rotation_signals_are_prefix_stable():
    dates = pd.bdate_range("2025-01-01", periods=45)
    rows = []
    for concept in range(35):
        for position, date in enumerate(dates):
            rows.append({
                "trade_date": date, "concept_code": f"C{concept:02d}",
                "concept_amount": 1e8 * (1 + position / 100 + concept / 1000),
                "rs_20d": concept / 100 + position / 1000,
                "rs_momentum_5d": concept / 1000 + position / 10000,
                "concept_return_5d": concept / 500 + position / 1000,
                "concept_return_20d": concept / 200 + position / 1000,
                "common_delta_rank": (concept + 1) / 35,
                "eligible_concept": True,
            })
    frame = pd.DataFrame(rows)
    short = attach_rotation_signals(frame.loc[frame["trade_date"].le(dates[34])])
    full = attach_rotation_signals(frame)
    columns = [
        "trade_date", "concept_code", "concept_amount_ratio", "rotation_leader_score",
        "rotation_successor_score", "rotation_momentum_score",
    ]
    pd.testing.assert_frame_equal(
        short[columns].reset_index(drop=True),
        full.loc[full["trade_date"].le(dates[34]), columns].reset_index(drop=True),
    )


def test_simple_momentum_score_is_only_positive_twenty_day_rank():
    date = pd.Timestamp("2026-01-05")
    frame = pd.DataFrame({
        "trade_date": [date] * 3, "concept_code": ["A", "B", "C"],
        "concept_amount": [1e8] * 3, "rs_20d": [0.1, 0.2, 0.3],
        "rs_momentum_5d": [0.1] * 3, "concept_return_5d": [0.1] * 3,
        "concept_return_20d": [-0.1, 0.1, 0.2], "common_delta_rank": [0.5] * 3,
        "eligible_concept": [True] * 3,
    })
    result = attach_rotation_signals(frame).set_index("concept_code")
    assert np.isnan(result.loc["A", "rotation_momentum_score"])
    assert result.loc["C", "rotation_momentum_score"] > result.loc["B", "rotation_momentum_score"]
