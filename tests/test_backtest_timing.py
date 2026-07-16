from __future__ import annotations

import pandas as pd
import pytest

from factor_forge.backtest import BacktestEngine
from factor_forge.config import CostModel, ExecutionConstraints


def one_stock_panel(opens: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=len(opens))
    return pd.DataFrame({
        "trade_date": dates, "ts_code": "000001.SZ", "raw_open": opens,
        "adj_open": opens, "adj_close": opens, "is_tradeable": True, "is_liquid": True,
        "is_suspended": False, "is_limit_up_open": False, "is_limit_down_open": False,
        "is_st": False, "is_delisting_period": False, "listing_trade_days": 100,
    })


def run(panel):
    factor = panel[["trade_date", "ts_code"]].copy()
    factor["factor_value"] = 1.0
    return BacktestEngine().run(
        panel, factor, universe="liquid", top_n=1, holding_days=1,
        initial_cash=100_000, lot_size=100, constraints=ExecutionConstraints(min_listing_days=60),
        cost_model=CostModel(commission_bps_per_side=0, slippage_bps_per_side=0, stamp_duty_bps_sell=0),
        cost_scenario_bps=0,
    )


def test_signal_at_t_enters_t_plus_1_and_exits_next_open():
    result = run(one_stock_panel([8.0, 10.0, 11.0, 11.0]))
    first_buy = result.trades[result.trades.side == "BUY"].iloc[0]
    first_sell = result.trades[result.trades.side == "SELL"].iloc[0]
    assert first_buy.signal_date == pd.Timestamp("2024-01-02")
    assert first_buy.trade_date == pd.Timestamp("2024-01-03")
    assert first_buy.raw_open == 10.0
    assert first_sell.trade_date == pd.Timestamp("2024-01-04")
    assert result.daily.loc[result.daily.trade_date == pd.Timestamp("2024-01-04"), "nav"].iloc[0] == pytest.approx(110_000)


def test_limit_up_buy_is_not_replaced():
    panel = one_stock_panel([8.0, 10.0, 11.0])
    panel.loc[panel.trade_date == pd.Timestamp("2024-01-03"), "is_limit_up_open"] = True
    result = run(panel)
    blocked_date = pd.Timestamp("2024-01-03")
    assert not ((result.trades.side == "BUY") & (result.trades.trade_date == blocked_date)).any()
    # Cash stays idle for the blocked batch; a fresh signal may trade the next day.
    assert result.daily.loc[result.daily.trade_date == blocked_date, "nav"].iloc[0] == 100_000


def test_limit_down_sell_is_deferred():
    panel = one_stock_panel([8.0, 10.0, 11.0, 12.0])
    panel.loc[panel.trade_date == pd.Timestamp("2024-01-04"), "is_limit_down_open"] = True
    result = run(panel)
    first_sell = result.trades[result.trades.side == "SELL"].iloc[0]
    assert first_sell.trade_date == pd.Timestamp("2024-01-05")


def test_position_multiplier_only_scales_new_entries():
    panel = one_stock_panel([10.0, 10.0, 10.0, 10.0])
    factor = panel[["trade_date", "ts_code"]].copy()
    factor["factor_value"] = 1.0
    multiplier = pd.Series(
        [0.0, 0.5, 0.0, 0.0], index=pd.to_datetime(panel["trade_date"].unique())
    )
    result = BacktestEngine().run(
        panel, factor, universe="liquid", top_n=1, holding_days=2,
        initial_cash=100_000, lot_size=100,
        constraints=ExecutionConstraints(min_listing_days=60),
        cost_model=CostModel(commission_bps_per_side=0, slippage_bps_per_side=0,
                             stamp_duty_bps_sell=0),
        cost_scenario_bps=0, position_multiplier=multiplier,
    )
    buys = result.trades[result.trades.side == "BUY"]
    assert len(buys) == 1
    assert buys.iloc[0]["signal_date"] == pd.Timestamp("2024-01-02")
    assert buys.iloc[0]["trade_date"] == pd.Timestamp("2024-01-03")
    assert buys.iloc[0]["gross_value"] == pytest.approx(25_000)
    assert result.daily.loc[result.daily.trade_date == pd.Timestamp("2024-01-04"),
                            "gross_exposure"].iloc[0] == pytest.approx(25_000)


def test_condition_membership_filters_selection_and_becomes_primary_benchmark():
    dates = pd.bdate_range("2024-01-02", periods=4)
    rows = []
    for code, opens in {"A.SZ": [8.0, 10.0, 20.0, 20.0], "B.SZ": [8.0, 10.0, 5.0, 5.0]}.items():
        stock = one_stock_panel(opens)
        stock["ts_code"] = code
        rows.append(stock)
    panel = pd.concat(rows, ignore_index=True)
    factor = panel[["trade_date", "ts_code"]].copy()
    factor["factor_value"] = factor["ts_code"].map({"A.SZ": 2.0, "B.SZ": 1.0})
    membership = pd.DataFrame({
        "trade_date": dates,
        "ts_code": "B.SZ",
        "condition_quantile": 5,
        "selection_eligible": True,
    })
    result = BacktestEngine().run(
        panel, factor, universe="liquid", top_n=1, holding_days=1,
        initial_cash=100_000, lot_size=100,
        constraints=ExecutionConstraints(min_listing_days=60),
        cost_model=CostModel(commission_bps_per_side=0, slippage_bps_per_side=0,
                             stamp_duty_bps_sell=0),
        cost_scenario_bps=0, selection_membership=membership,
    )
    buys = result.trades[result.trades.side == "BUY"]
    assert set(buys["ts_code"]) == {"B.SZ"}
    assert set(buys["condition_quantile"]) == {5}
    day_three = result.daily.loc[result.daily.trade_date == dates[2]].iloc[0]
    assert day_three["benchmark_return"] == pytest.approx(-0.5)
    assert day_three["universe_benchmark_return"] == pytest.approx(0.25)
    assert result.metrics["benchmark_scope"] == "condition_equal_weight"


def test_fully_invest_selected_reallocates_when_candidates_are_below_top_n():
    rows = []
    for code in ["A.SZ", "B.SZ"]:
        stock = one_stock_panel([10.0, 10.0, 10.0])
        stock["ts_code"] = code
        rows.append(stock)
    panel = pd.concat(rows, ignore_index=True)
    factor = panel[["trade_date", "ts_code"]].copy()
    factor["factor_value"] = 1.0
    result = BacktestEngine().run(
        panel, factor, universe="liquid", top_n=4, holding_days=1,
        initial_cash=100_000, lot_size=100,
        constraints=ExecutionConstraints(min_listing_days=60),
        cost_model=CostModel(commission_bps_per_side=0, slippage_bps_per_side=0,
                             stamp_duty_bps_sell=0),
        cost_scenario_bps=0, fully_invest_selected=True,
    )
    buys = result.trades[result.trades.side == "BUY"]
    first_batch = buys.loc[buys.trade_date.eq(pd.Timestamp("2024-01-03"))]
    assert len(first_batch) == 2
    assert first_batch["gross_value"].sum() == pytest.approx(100_000)
