from __future__ import annotations

import numpy as np
import pandas as pd

from factor_forge.ml.bullish_divergence_conditional_ml import (
    ARM_BLOCKS,
    STRUCTURE_ARM_BLOCKS,
    TestPeriod,
    arm_feature_columns,
    attach_structure_features,
    compare_paired_placebo_arms,
    daily_equal_sample_weight,
    make_folds,
    shuffle_labels_within_date,
    summarize_paired_placebos,
)


def test_m0_m7_feature_blocks_are_nested_as_registered() -> None:
    frame = pd.DataFrame(
        {
            "control__size": [1.0],
            "div__price": [1.0],
            "touch__count": [1.0],
            "structure__double_divergence_trend_score": [1.0],
            "regime__breadth": [1.0],
            "industry_rotation__rs": [1.0],
            "concept__best_rs": [1.0],
            "momentum__ret_5d": [1.0],
            "interaction__div_x_regime": [1.0],
        }
    )
    assert set(ARM_BLOCKS) == {f"M{number}" for number in range(8)}
    assert arm_feature_columns(frame, "M0") == ["control__size"]
    assert set(arm_feature_columns(frame, "M1")) == {
        "control__size",
        "div__price",
        "touch__count",
    }
    assert "structure__double_divergence_trend_score" not in arm_feature_columns(frame, "M1")
    assert set(arm_feature_columns(frame, "H3")) == {
        "control__size",
        "div__price",
        "touch__count",
        "structure__double_divergence_trend_score",
    }
    assert STRUCTURE_ARM_BLOCKS["H5"] == ("X", "D", "T", "S", "R", "C", "M")
    assert set(arm_feature_columns(frame, "M6")) == {
        "control__size",
        "div__price",
        "touch__count",
        "regime__breadth",
        "industry_rotation__rs",
        "concept__best_rs",
        "momentum__ret_5d",
    }
    assert "interaction__div_x_regime" in arm_feature_columns(frame, "M7")


def test_walk_forward_fold_uses_label_availability_purge() -> None:
    events = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2021-12-01", "2021-12-20", "2022-01-03", "2022-02-01"]
            ),
            "label_available_date": pd.to_datetime(
                ["2021-12-16", "2022-01-05", "2022-01-18", "2022-02-16"]
            ),
        }
    )
    folds = make_folds(events, [TestPeriod(1, "2022-01-01", "2022-12-31")])
    assert folds.loc[0, "train_count"] == 1
    assert folds.loc[0, "test_count"] == 2
    assert folds.loc[0, "max_train_label_available_date"] < folds.loc[0, "test_start"]


def test_daily_equal_weights_give_each_date_equal_total_weight() -> None:
    dates = pd.Series(pd.to_datetime(["2025-01-02"] * 2 + ["2025-01-03"] * 5))
    weights = daily_equal_sample_weight(dates)
    totals = pd.Series(weights).groupby(dates.reset_index(drop=True)).sum()
    assert np.isclose(totals.iloc[0], totals.iloc[1])


def test_within_date_shuffle_preserves_each_dates_label_multiset() -> None:
    dates = pd.Series(pd.to_datetime(["2025-01-02"] * 4 + ["2025-01-03"] * 4))
    labels = pd.Series(np.arange(8, dtype=float))
    shuffled = shuffle_labels_within_date(labels, dates, seed=7)
    for date in dates.unique():
        mask = dates.eq(date)
        assert sorted(shuffled.loc[mask].tolist()) == sorted(labels.loc[mask].tolist())
    assert not shuffled.equals(labels)


def test_attach_structure_features_requires_one_to_one_event_match() -> None:
    events = pd.DataFrame({
        "event_id": ["a", "b"],
        "trade_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
        "ts_code": ["000001.SZ", "000002.SZ"],
        "label__industry_excess_10d": [0.01, -0.01],
    })
    episodes = events.loc[:, ["event_id", "trade_date", "ts_code"]].copy()
    episodes["structure__double_divergence_trend_score"] = [70.0, 30.0]
    episodes["structure__pivot_p_date"] = pd.to_datetime(["2024-11-01", "2024-11-04"])
    merged = attach_structure_features(events, episodes)
    assert merged["structure__double_divergence_trend_score"].tolist() == [70.0, 30.0]
    assert "structure__pivot_p_date" not in merged.columns


def test_custom_arm_blocks_select_structure_variant() -> None:
    frame = pd.DataFrame({
        "control__size": [1.0],
        "structure__double_divergence_present": [1.0],
        "structure__double_divergence_trend_score": [70.0],
    })
    blocks = {
        "X": ["control__size"],
        "S_FLAG": ["structure__double_divergence_present"],
    }
    arms = {"X_FLAG": ("X", "S_FLAG")}
    assert arm_feature_columns(
        frame,
        "X_FLAG",
        feature_blocks=blocks,
        arm_blocks=arms,
    ) == ["control__size", "structure__double_divergence_present"]
    reversed_arms = {"X_FLAG": ("S_FLAG", "X")}
    assert arm_feature_columns(
        frame,
        "X_FLAG",
        feature_blocks=blocks,
        arm_blocks=reversed_arms,
    ) == ["control__size", "structure__double_divergence_present"]


def test_paired_placebo_compares_same_repeat_increment() -> None:
    rows = []
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    for arm, values in {
        "BASE": [0.0, 0.0, 0.0],
        "PLUS": [0.2, 0.1, 0.3],
        "BASE_shuffle_00": [0.0, 0.1, 0.0],
        "PLUS_shuffle_00": [0.1, 0.0, 0.1],
    }.items():
        for date, value in zip(dates, values):
            rows.append({
                "scope": "test",
                "algorithm": "ridge",
                "arm": arm,
                "fold_id": 1,
                "trade_date": date,
                "top_n": 10,
                "cost_bps": 40,
                "rank_ic": value,
                "top_net": value,
                "top_minus_all": value,
                "top_minus_bottom": value,
            })
    comparisons = compare_paired_placebo_arms(
        pd.DataFrame(rows), [("PLUS", "BASE")]
    )
    actual = comparisons.loc[
        comparisons["metric"].eq("rank_ic") & ~comparisons["is_placebo"]
    ].iloc[0]
    placebo = comparisons.loc[
        comparisons["metric"].eq("rank_ic") & comparisons["is_placebo"]
    ].iloc[0]
    assert np.isclose(actual["mean_delta"], 0.2)
    assert np.isclose(placebo["mean_delta"], 1 / 30)
    summary = summarize_paired_placebos(comparisons)
    assert summary["placebo_count"].eq(1).all()
