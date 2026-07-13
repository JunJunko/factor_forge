from __future__ import annotations

import pandas as pd

from factor_forge.config import ProjectConfig
from factor_forge.data.panel import DailyPanelBuilder


def test_source_units_and_point_in_time_status_are_normalized():
    daily = pd.DataFrame({
        "trade_date": ["20240102", "20240103"], "ts_code": ["000001.SZ", "000001.SZ"],
        "open": [10.0, 10.5], "high": [10.2, 10.7], "low": [9.9, 10.4],
        "close": [10.1, 10.6], "pre_close": [9.8, 10.1], "vol": [123.0, 200.0],
        "amount": [456.0, 800.0], "pct_chg": [3.0, 5.0],
    })
    datasets = {
        "daily": daily,
        "adj_factor": pd.DataFrame({"trade_date": ["20240102", "20240103"],
                                    "ts_code": ["000001.SZ", "000001.SZ"], "adj_factor": [2.0, 2.0]}),
        "daily_basic": pd.DataFrame({"trade_date": ["20240102", "20240103"],
                                     "ts_code": ["000001.SZ", "000001.SZ"],
                                     "total_mv": [100.0, 110.0], "circ_mv": [80.0, 90.0],
                                     "turnover_rate": [1.0, 1.1]}),
        "stk_limit": pd.DataFrame(), "suspend": pd.DataFrame(),
        "stock_basic": pd.DataFrame({"ts_code": ["000001.SZ"], "name": ["测试"],
                                     "list_date": ["19910101"], "delist_date": [None]}),
        "industry_membership": pd.DataFrame({"ts_code": ["000001.SZ"], "industry_code": ["801010"],
                                             "industry_name": ["农林牧渔"], "in_date": ["20200101"],
                                             "out_date": [None]}),
        "st_status": pd.DataFrame({"trade_date": ["20240103"], "ts_code": ["000001.SZ"],
                                   "name": ["ST测试"]}),
        "st_status_coverage": pd.DataFrame({"trade_date": ["20240102", "20240103"]}),
    }
    project = ProjectConfig.model_validate({"data": {"listing_age_days": 1,
        "liquidity": {"window": 2, "min_avg_amount_cny": 1, "min_traded_days": 1}}})
    panel = DailyPanelBuilder(project).build(datasets)
    first, second = panel.iloc[0], panel.iloc[1]
    assert first.volume_shares == 12_300
    assert first.amount_cny == 456_000
    assert first.total_mv_cny == 1_000_000
    assert first.adj_close == 20.2
    assert first.industry_l1_code == "801010"
    assert not first.is_st and second.is_st
    assert panel.st_status_known.all()


def test_l2_industry_membership_uses_level_specific_columns():
    daily = pd.DataFrame({
        "trade_date": ["20240102"], "ts_code": ["000001.SZ"],
        "open": [10.0], "high": [10.2], "low": [9.9], "close": [10.1],
        "pre_close": [9.8], "vol": [123.0], "amount": [456.0], "pct_chg": [3.0],
    })
    datasets = {
        "daily": daily,
        "adj_factor": pd.DataFrame({"trade_date": ["20240102"], "ts_code": ["000001.SZ"], "adj_factor": [1.0]}),
        "daily_basic": pd.DataFrame(), "stk_limit": pd.DataFrame(), "suspend": pd.DataFrame(),
        "stock_basic": pd.DataFrame({"ts_code": ["000001.SZ"], "name": ["测试"], "list_date": ["19910101"], "delist_date": [None]}),
        "industry_membership": pd.DataFrame({"ts_code": ["000001.SZ"], "industry_code": ["801016.SI"],
                                             "industry_name": ["种植业"], "in_date": ["20200101"], "out_date": [None]}),
        "st_status": pd.DataFrame(), "st_status_coverage": pd.DataFrame({"trade_date": ["20240102"]}),
    }
    project = ProjectConfig.model_validate({"data": {"industry_level": "L2", "listing_age_days": 1,
        "liquidity": {"window": 1, "min_avg_amount_cny": 1, "min_traded_days": 1}}})
    panel = DailyPanelBuilder(project).build(datasets)
    assert panel.loc[0, "industry_l2_code"] == "801016.SI"
    assert panel.loc[0, "industry_l2_name"] == "种植业"
    assert "industry_l1_code" not in panel


def test_panel_filters_to_configured_security_master_and_converts_moneyflow_units():
    daily = pd.DataFrame({
        "trade_date": ["20240102", "20240102"],
        "ts_code": ["600000.SH", "300001.SZ"],
        "open": [10.0, 20.0], "high": [10.2, 20.2], "low": [9.9, 19.9],
        "close": [10.1, 20.1], "pre_close": [10.0, 20.0],
        "vol": [100.0, 100.0], "amount": [1000.0, 1000.0], "pct_chg": [1.0, 0.5],
    })
    datasets = {
        "daily": daily,
        "adj_factor": pd.DataFrame({
            "trade_date": ["20240102", "20240102"],
            "ts_code": ["600000.SH", "300001.SZ"], "adj_factor": [1.0, 1.0],
        }),
        "daily_basic": pd.DataFrame(), "stk_limit": pd.DataFrame(),
        "suspend": pd.DataFrame({"trade_date": ["20240102"], "ts_code": ["300002.SZ"]}),
        "stock_basic": pd.DataFrame({"ts_code": ["600000.SH"], "market": ["主板"]}),
        "moneyflow": pd.DataFrame({
            "trade_date": ["20240102", "20240102"],
            "ts_code": ["600000.SH", "300001.SZ"], "net_mf_amount": [12.5, 99.0],
            "buy_sm_amount": [30.0, 1.0], "sell_sm_amount": [20.0, 1.0],
            "buy_lg_amount": [40.0, 1.0], "sell_lg_amount": [10.0, 1.0],
            "buy_elg_amount": [50.0, 1.0], "sell_elg_amount": [5.0, 1.0],
        }),
        "industry_membership": pd.DataFrame(), "st_status": pd.DataFrame(),
        "st_status_coverage": pd.DataFrame({"trade_date": ["20240102"]}),
    }
    project = ProjectConfig.model_validate({"data": {"boards": ["main"], "listing_age_days": 1,
        "liquidity": {"window": 1, "min_avg_amount_cny": 1, "min_traded_days": 1}}})

    panel = DailyPanelBuilder(project).build(datasets)

    assert panel["ts_code"].tolist() == ["600000.SH"]
    assert panel.loc[0, "net_mf_amount_cny"] == 125_000
    assert panel.loc[0, "buy_sm_amount_cny"] == 300_000
    assert panel.loc[0, "sell_lg_amount_cny"] == 100_000
    assert panel.loc[0, "buy_elg_amount_cny"] == 500_000
