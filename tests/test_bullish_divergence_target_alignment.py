from __future__ import annotations

import numpy as np
import pandas as pd

from factor_forge.ml.bullish_divergence_target_alignment import (
    TargetAlignmentConfig,
    balanced_daily_class_weight,
    build_target_aligned_daily_evaluation,
    within_date_relevance,
    within_date_top_fraction,
)


def test_within_date_targets_have_expected_counts_and_order() -> None:
    dates = pd.Series(pd.to_datetime(["2025-01-02"] * 10 + ["2025-01-03"] * 5))
    labels = pd.Series(np.r_[np.arange(10), np.arange(5)], dtype="float64")
    relevance = within_date_relevance(labels, dates, grades=10)
    assert relevance.iloc[:10].tolist() == list(range(10))
    assert relevance.iloc[10:].tolist() == [0, 2, 4, 6, 8]
    top = within_date_top_fraction(labels, dates, fraction=0.10)
    assert top.iloc[:10].sum() == 1
    assert top.iloc[10:].sum() == 1
    assert top.iloc[9] == 1
    assert top.iloc[14] == 1


def test_balanced_daily_class_weight_equalizes_dates_and_classes() -> None:
    dates = pd.Series(pd.to_datetime(["2025-01-02"] * 10 + ["2025-01-03"] * 5))
    target = pd.Series([0] * 9 + [1] + [0] * 4 + [1])
    weight = pd.Series(balanced_daily_class_weight(target, dates))
    for date in dates.unique():
        mask = dates.eq(date)
        positive = mask & target.eq(1)
        negative = mask & target.eq(0)
        assert np.isclose(weight.loc[positive].sum(), weight.loc[negative].sum())
    totals = weight.groupby(dates).sum()
    assert np.isclose(totals.iloc[0], totals.iloc[1])


def test_portfolio_evaluation_rewards_perfect_daily_ranking() -> None:
    predictions = pd.DataFrame({
        "scope": ["test"] * 20,
        "objective": ["lgb_lambdarank"] * 20,
        "arm": ["DT_BASE"] * 20,
        "is_placebo": [False] * 20,
        "placebo_repeat": [np.nan] * 20,
        "fold_id": [1] * 20,
        "trade_date": pd.to_datetime(["2025-01-02"] * 20),
        "score": np.arange(20, dtype="float64"),
        "label__industry_excess_10d": np.arange(20, dtype="float64") / 100,
    })
    daily = build_target_aligned_daily_evaluation(
        predictions,
        config=TargetAlignmentConfig(costs_bps=(40,)),
    )
    top5 = daily.loc[daily["portfolio"].eq("top_5")].iloc[0]
    rank_long = daily.loc[daily["portfolio"].eq("rank_weighted_long")].iloc[0]
    rank_ls = daily.loc[daily["portfolio"].eq("rank_weighted_ls")].iloc[0]
    assert top5["gross"] > top5["all_gross"]
    assert rank_long["gross"] > rank_long["all_gross"]
    assert rank_ls["gross"] > 0
