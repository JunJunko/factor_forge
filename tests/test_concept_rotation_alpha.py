import numpy as np
import pandas as pd
import pytest

from factor_forge.research.concept_rotation_alpha import (
    _jaccard,
    build_concept_dataset,
    latest_membership_backfill,
    paired_signal_differences,
    prepare_stock_panel,
    repair_partial_member_snapshots,
    select_deduplicated_concepts,
)


def test_stock_can_belong_to_multiple_concepts_without_duplicate_stock_panel():
    dates = pd.bdate_range("2025-01-01", periods=25)
    panel = pd.DataFrame({
        "trade_date": list(dates) * 2, "ts_code": ["A"] * 25 + ["B"] * 25,
        "adj_close": np.r_[np.arange(1, 26), np.arange(2, 27)],
        "adj_open": np.r_[np.arange(1, 26), np.arange(2, 27)],
        "circ_mv_cny": 1e9, "amount_cny": 1e8, "is_tradeable": True,
    })
    result = prepare_stock_panel(panel)
    assert len(result) == 50
    assert result.groupby("ts_code")["stock_return_20d"].count().tolist() == [5, 5]


def test_daily_jaccard_dedup_keeps_only_one_near_duplicate():
    date = pd.Timestamp("2025-01-02")
    features = pd.DataFrame({
        "trade_date": [date] * 3, "concept_code": ["C1", "C2", "C3"],
        "score": [3.0, 2.0, 1.0],
    })
    members = pd.DataFrame({
        "trade_date": [date] * 7,
        "concept_code": ["C1", "C1", "C2", "C2", "C3", "C3", "C3"],
        "ts_code": ["A", "B", "A", "B", "X", "Y", "Z"],
    })
    selected = select_deduplicated_concepts(features, members, "score", top_n=3)
    assert selected["concept_code"].tolist() == ["C1", "C3"]
    assert _jaccard(frozenset("AB"), frozenset("AB")) == 1


def test_latest_membership_backfill_uses_final_snapshot_on_every_date():
    dates = pd.to_datetime(["2025-06-30", "2025-07-01"])
    index = pd.DataFrame({
        "trade_date": dates, "concept_code": ["C1", "C1"],
        "concept_name": ["old", "new"],
    })
    members = pd.DataFrame({
        "trade_date": dates, "concept_code": ["C1", "C1"],
        "ts_code": ["A", "B"], "stock_name": ["A", "B"],
    })
    back_index, back_members = latest_membership_backfill(index, members)
    assert set(back_index["concept_name"]) == {"new"}
    assert set(back_members["ts_code"]) == {"B"}
    assert len(back_members) == 2


def test_build_dataset_computes_common_member_churn_without_full_history_join():
    dates = pd.bdate_range("2025-01-01", periods=80)
    codes = [f"S{i}" for i in range(10)]
    panel = pd.MultiIndex.from_product([codes, dates], names=["ts_code", "trade_date"]).to_frame(index=False)
    step = panel.groupby("ts_code").cumcount()
    panel["adj_close"] = 10 + step * 0.01
    panel["adj_open"] = panel["adj_close"]
    panel["circ_mv_cny"] = 1e9
    panel["amount_cny"] = 1e8
    panel["is_tradeable"] = True
    index = pd.MultiIndex.from_product([dates, ["C1", "C2"]], names=["trade_date", "concept_code"]).to_frame(index=False)
    index["concept_name"] = index["concept_code"]
    rows = []
    for date in dates:
        for concept, stock_codes in (("C1", codes[:8]), ("C2", codes[2:])):
            rows.extend((date, concept, code, code) for code in stock_codes)
    members = pd.DataFrame(rows, columns=["trade_date", "concept_code", "ts_code", "stock_name"])
    _, concepts, _ = build_concept_dataset(panel, index, members, minimum_age=10)
    mature = concepts.loc[concepts["concept_age_days"].ge(10), "membership_churn_5d"].dropna()
    assert not mature.empty
    assert mature.eq(0).all()


def test_partial_member_snapshot_is_replaced_with_prior_complete_snapshot():
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    rows = []
    for date, concepts in zip(dates, [["C1", "C2"], ["C1"], ["C1", "C2"]]):
        rows.extend((date, concept, "S") for concept in concepts)
    members = pd.DataFrame(rows, columns=["trade_date", "concept_code", "ts_code"])
    repaired, repaired_dates = repair_partial_member_snapshots(members)
    assert repaired_dates == ["2025-01-02"]
    assert repaired.loc[repaired["trade_date"].eq(dates[1]), "concept_code"].nunique() == 2


def test_paired_signal_difference_is_date_aligned():
    dates = pd.bdate_range("2025-01-01", periods=25)
    daily = pd.DataFrame({
        "trade_date": list(dates) * 2, "membership_mode": "point_in_time", "split": "holdout",
        "signal": ["combined"] * 25 + ["baseline"] * 25,
        "net_excess_20bps": [0.02] * 25 + [0.01] * 25,
    })
    result = paired_signal_differences(daily, comparisons=[("combined", "baseline")])
    assert result.iloc[0]["days"] == 25
    assert result.iloc[0]["incremental_net_excess"] == pytest.approx(0.01)
