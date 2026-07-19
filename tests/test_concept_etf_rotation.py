import pandas as pd
import pytest

from factor_forge.research.concept_etf_rotation import _select_clusters, prepare_etf_panel


def test_prepare_etf_panel_uses_next_open_for_forward_return():
    dates = pd.bdate_range("2025-01-01", periods=7)
    daily = pd.DataFrame({
        "ts_code": "ETF", "trade_date": dates.strftime("%Y%m%d"),
        "open": range(10, 17), "high": range(10, 17), "low": range(10, 17),
        "close": range(10, 17), "pre_close": range(9, 16), "amount": 1000, "vol": 100,
    })
    share = pd.DataFrame({"ts_code": "ETF", "trade_date": dates.strftime("%Y%m%d"), "fd_share": 100})
    nav = pd.DataFrame({
        "ts_code": "ETF", "nav_date": dates.strftime("%Y%m%d"),
        "unit_nav": range(10, 17), "adj_nav": range(10, 17),
    })
    basic = pd.DataFrame({"ts_code": ["ETF"], "name": ["test"], "list_date": ["20200101"]})
    result = prepare_etf_panel(daily, share, nav, basic)
    assert result.iloc[0]["forward_open_5d"] == pytest.approx(16 / 11 - 1)
    assert result.iloc[0]["amount_cny"] == 1_000_000
    assert result.iloc[0]["aum_cny"] == 10_000_000


def test_cluster_selection_keeps_at_most_one_etf_per_cluster():
    candidates = pd.DataFrame({
        "ts_code": ["A", "B", "C", "D"], "cluster": ["ai", "ai", "health", "energy"],
        "score": [4, 3, 2, 1],
    })
    selected = _select_clusters(candidates, 3)
    assert selected["ts_code"].tolist() == ["A", "C", "D"]


def test_prepare_etf_panel_uses_reported_return_across_split_boundary():
    dates = pd.bdate_range("2025-01-01", periods=3)
    daily = pd.DataFrame({
        "ts_code": "ETF", "trade_date": dates.strftime("%Y%m%d"),
        "open": [2.0, 2.1, 1.0], "high": [2.0, 2.1, 1.0],
        "low": [2.0, 2.1, 1.0], "close": [2.0, 2.1, 1.0],
        "pre_close": [2.0, 2.0, 1.05], "pct_chg": [0.0, 5.0, -4.7619],
        "amount": 1000, "vol": 100,
    })
    share = pd.DataFrame({
        "ts_code": "ETF", "trade_date": dates.strftime("%Y%m%d"), "fd_share": 100,
    })
    nav = pd.DataFrame({
        "ts_code": "ETF", "nav_date": dates.strftime("%Y%m%d"),
        "unit_nav": [2.0, 1.05, 1.0], "adj_nav": [2.0, 2.1, 2.0],
    })
    basic = pd.DataFrame({"ts_code": ["ETF"], "name": ["test"], "list_date": ["20200101"]})

    result = prepare_etf_panel(daily, share, nav, basic)

    assert result["etf_return_1d"].iloc[1] == pytest.approx(0.05)
    assert result["etf_return_1d"].iloc[2] == pytest.approx(-0.047619, abs=1e-6)


def test_prepare_etf_panel_can_lag_share_availability_one_session():
    dates = pd.bdate_range("2025-01-01", periods=3)
    daily = pd.DataFrame({
        "ts_code": "ETF", "trade_date": dates.strftime("%Y%m%d"),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
        "pre_close": 1.0, "amount": 1000, "vol": 100,
    })
    share = pd.DataFrame({
        "ts_code": "ETF", "trade_date": dates.strftime("%Y%m%d"),
        "fd_share": [100, 200, 300],
    })
    basic = pd.DataFrame({"ts_code": ["ETF"], "name": ["test"], "list_date": ["20200101"]})
    result = prepare_etf_panel(
        daily, share, pd.DataFrame(), basic, share_availability_lag_sessions=1,
    )
    assert pd.isna(result.iloc[0]["fd_share"])
    assert result.iloc[1]["fd_share"] == 100
    assert result.iloc[2]["fd_share"] == 200
