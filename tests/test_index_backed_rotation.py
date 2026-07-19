import pandas as pd

from factor_forge.research.index_backed_rotation import (
    build_monthly_index_history_eligibility,
    build_dynamic_etf_signal_panel,
    build_monthly_pit_etf_mapping,
    classify_theme_cluster,
    expand_monthly_index_membership,
    filter_mapping_by_weight_coverage,
    official_theme_etf_candidates,
    recover_delisted_theme_etf_candidates,
    select_exact_index_etf_mapping,
)


def test_select_exact_mapping_uses_liquid_official_index_etf():
    basic = pd.DataFrame({
        "ts_code": ["A.SH", "B.SH", "C.SH"],
        "csname": ["芯片ETF甲", "芯片ETF乙", "海外芯片ETF"],
        "index_code": ["930001.CSI", "930001.CSI", "930002.CSI"],
        "index_name": ["中证芯片", "中证芯片", "海外芯片"],
        "list_date": ["20200101"] * 3,
        "list_status": ["L"] * 3,
        "etf_type": ["纯境内", "纯境内", "QDII"],
    })
    dates = pd.date_range("2026-01-01", periods=3)
    daily = pd.DataFrame([
        {"ts_code": code, "trade_date": date, "amount": amount, "close": 1.0}
        for code, amount in [("A.SH", 100_000), ("B.SH", 200_000), ("C.SH", 500_000)]
        for date in dates
    ])
    share = pd.DataFrame([
        {"ts_code": code, "trade_date": dates[-1], "fd_share": 200_000}
        for code in ["A.SH", "B.SH", "C.SH"]
    ])
    mapping, _ = select_exact_index_etf_mapping(
        basic, daily, share, selection_cutoff="2026-01-03",
        minimum_adv_cny=1, minimum_aum_cny=1, minimum_observations=3,
    )
    assert mapping["etf_code"].tolist() == ["B.SH"]
    assert mapping["match_type"].eq("exact_official_tracking_index").all()


def test_monthly_membership_is_lagged_one_session():
    weights = pd.DataFrame({
        "index_code": ["930001.CSI", "930001.CSI", "930001.CSI"],
        "trade_date": ["20230131", "20230131", "20230202"],
        "con_code": ["A.SH", "B.SH", "C.SH"],
        "weight": [50, 50, 100],
    })
    mapping = pd.DataFrame({
        "concept_code": ["930001.CSI"], "concept_name": ["中证芯片"],
    })
    dates = pd.to_datetime(["2023-01-31", "2023-02-01", "2023-02-02", "2023-02-03"])
    index, members = expand_monthly_index_membership(weights, mapping, dates, lag_sessions=1)
    assert index["trade_date"].min() == pd.Timestamp("2023-02-01")
    feb2 = set(members.loc[members["trade_date"].eq("2023-02-02"), "ts_code"])
    feb3 = set(members.loc[members["trade_date"].eq("2023-02-03"), "ts_code"])
    assert feb2 == {"A.SH", "B.SH"}
    assert feb3 == {"C.SH"}


def test_weight_coverage_gate_and_cluster():
    mapping = pd.DataFrame({
        "concept_code": ["A.CSI", "B.CSI"],
        "concept_name": ["芯片", "证券"],
    })
    rows = []
    for month in range(1, 4):
        for member in ["X", "Y"]:
            rows.append({
                "index_code": "A.CSI", "trade_date": f"20230{month}28", "con_code": member,
            })
    rows.append({"index_code": "B.CSI", "trade_date": "20230128", "con_code": "X"})
    selected, audit = filter_mapping_by_weight_coverage(
        mapping, pd.DataFrame(rows), minimum_weight_months=3, minimum_members=2,
    )
    assert selected["concept_code"].tolist() == ["A.CSI"]
    assert audit.set_index("concept_code").loc["B.CSI", "weight_coverage_pass"] == 0
    assert classify_theme_cluster("中证芯片产业") == "semiconductors"


def test_official_candidates_do_not_require_current_listing_status():
    basic = pd.DataFrame({
        "ts_code": ["A.SH", "B.SH"], "index_code": ["930001.CSI", "930002.CSI"],
        "index_name": ["中证芯片", "宽基指数"], "list_date": ["20200101", "20200101"],
        "list_status": ["D", "L"], "etf_type": ["纯境内", "纯境内"],
    })
    result = official_theme_etf_candidates(basic, as_of="2026-01-01")
    assert result["ts_code"].tolist() == ["A.SH"]


def test_monthly_pit_mapping_uses_prior_month_data_only():
    dates = pd.bdate_range("2023-01-02", "2023-04-05")
    etf_rows = []
    for code, amount in [("A.SH", 100_000), ("B.SH", 200_000)]:
        for position, date in enumerate(dates):
            etf_rows.append({
                "ts_code": code, "trade_date": date, "amount_cny": amount * 1000,
                "aum_cny": 2_000_000_000, "etf_return_1d": position / 10000,
            })
    etfs = pd.DataFrame(etf_rows)
    candidates = pd.DataFrame({
        "ts_code": ["A.SH", "B.SH"], "index_code": ["930001.CSI"] * 2,
        "index_name": ["中证芯片"] * 2, "csname": ["甲", "乙"],
        "list_date_parsed": pd.to_datetime(["20200101"] * 2),
        "cluster": ["semiconductors"] * 2,
    })
    concepts = pd.DataFrame({
        "trade_date": dates, "concept_code": "930001.CSI",
        "concept_return_1d": pd.Series(range(len(dates))) / 10000,
    })
    schedule, _ = build_monthly_pit_etf_mapping(
        etfs, candidates, concepts, dates,
        minimum_listing_sessions=20, liquidity_window=5,
        minimum_liquidity_observations=5, minimum_adv_cny=1, minimum_aum_cny=1,
        correlation_window=20, minimum_correlation_observations=10,
        minimum_mapping_correlation=0.5,
    )
    assert schedule["etf_code"].eq("B.SH").all()
    assert schedule["effective_start"].gt(schedule["selection_date"]).all()


def test_dynamic_panel_keeps_nonselected_prices_but_only_scores_selected_etf():
    date = pd.Timestamp("2024-02-01")
    etfs = pd.DataFrame({
        "trade_date": [date, date], "ts_code": ["A.SH", "B.SH"],
        "etf_momentum_20d": [0.1, 0.2], "etf_momentum_60d": [0.2, 0.3],
    })
    candidates = pd.DataFrame({
        "ts_code": ["A.SH", "B.SH"], "index_code": ["I1.CSI", "I2.CSI"],
        "index_name": ["芯片", "医药"], "csname": ["甲", "乙"],
        "cluster": ["semiconductors", "healthcare"],
    })
    schedule = pd.DataFrame({
        "effective_month": ["2024-02"], "concept_code": ["I1.CSI"],
        "etf_code": ["A.SH"], "selection_date": [pd.Timestamp("2024-01-31")],
        "effective_start": [date], "match_type": ["exact_official_tracking_index_pit"],
        "mapping_correlation_pit": [0.98], "correlation_observations": [120],
    })
    concepts = pd.DataFrame({
        "trade_date": [date, date], "concept_code": ["I1.CSI", "I2.CSI"],
        "concept_return_1d": [0.01, 0.02], "eligible_concept": [True, True],
        "breadth_float": [0.5, 0.5], "common_delta_rank": [0.5, 0.5],
        "rs_momentum_5d": [0.1, 0.1], "rrg_quadrant": ["leading", "leading"],
        "signal_rrg_only": [1.0, 1.0], "common_breadth_delta_smooth5": [0.1, 0.1],
    })
    panel = build_dynamic_etf_signal_panel(concepts, etfs, schedule, candidates)
    assert len(panel) == 2
    assert panel.set_index("ts_code").loc["A.SH", "mapping_pass"]
    assert not panel.set_index("ts_code").loc["B.SH", "mapping_pass"]
    assert pd.isna(panel.set_index("ts_code").loc["B.SH", "score_etf_momentum"])


def test_index_history_gate_is_evaluated_at_each_month_end():
    dates = pd.bdate_range("2023-01-02", "2023-05-03")
    weights = pd.DataFrame([
        {"index_code": "I.CSI", "trade_date": date, "con_code": member}
        for date in pd.to_datetime(["2023-01-31", "2023-02-28", "2023-03-31"])
        for member in ("A", "B")
    ])
    result = build_monthly_index_history_eligibility(
        weights, dates, minimum_weight_months=3, minimum_members=2,
        availability_lag_sessions=1,
    )
    march = result.loc[result["selection_date"].eq("2023-03-31")].iloc[0]
    april = result.loc[result["selection_date"].eq("2023-04-28")].iloc[0]
    assert march["available_weight_months"] == 2
    assert not march["index_history_pass"]
    assert april["available_weight_months"] == 3
    assert april["index_history_pass"]


def test_recover_delisted_candidate_requires_exact_unambiguous_theme_benchmark():
    etf_basic = pd.DataFrame({
        "ts_code": ["D.SH", "X.SH"], "csname": ["退市芯片ETF", "退市宽基ETF"],
        "list_status": ["D", "D"], "list_date": ["20200101", "20200101"],
    })
    fund_basic = pd.DataFrame({
        "ts_code": ["D.SH", "X.SH"],
        "benchmark": ["中证芯片产业指数收益率×100%", "宽基指数收益率×100%"],
        "delist_date": ["20250101", "20250101"],
    })
    indexes = pd.DataFrame({
        "ts_code": ["930001.CSI", "000001.SH"],
        "indx_name": ["中证芯片产业指数", "宽基指数"],
        "indx_csname": ["中证芯片", "宽基"],
    })
    recovered, audit = recover_delisted_theme_etf_candidates(
        etf_basic, fund_basic, indexes, as_of="2026-01-01",
    )
    assert recovered["ts_code"].tolist() == ["D.SH"]
    assert recovered.iloc[0]["index_code"] == "930001.CSI"
    assert audit.set_index("ts_code").loc["X.SH", "recovery_pass"] == 0
