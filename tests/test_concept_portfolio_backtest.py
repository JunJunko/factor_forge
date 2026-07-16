import numpy as np
import pandas as pd
import pytest

from factor_forge.research.concept_portfolio_backtest import (
    ExecutionRules,
    PortfolioRules,
    TargetBuildResult,
    cap_and_redistribute,
    run_non_overlapping_ledger,
    transaction_cost,
    build_liquidity_neutral_targets,
    attach_continuous_breadth_signals,
    build_market_regimes,
    rescale_target_build,
)


def test_weight_cap_redistributes_without_changing_total():
    weights = pd.Series([0.8, 0.1, 0.1])
    result = cap_and_redistribute(weights, 0.5)
    assert result.sum() == pytest.approx(1.0)
    assert result.max() == pytest.approx(0.5)


def test_weight_cap_leaves_cash_when_too_few_stocks():
    result = cap_and_redistribute(pd.Series([0.8, 0.2]), 0.3)
    assert result.tolist() == pytest.approx([0.3, 0.2])
    assert result.sum() == pytest.approx(0.5)


def test_sell_cost_includes_stamp_and_minimum_commission():
    rules = ExecutionRules(
        commission_bps_per_side=2.5, minimum_commission_cny=5,
        transfer_fee_bps_per_side=0.1, stamp_duty_bps_sell=5,
        base_slippage_bps_per_side=0, impact_eta=0,
    )
    cost, parts = transaction_cost(10_000, "SELL", 0, rules)
    assert parts["commission_cost"] == 5
    assert parts["transfer_cost"] == pytest.approx(0.1)
    assert parts["stamp_cost"] == pytest.approx(5.0)
    assert cost == pytest.approx(10.1)


def test_blocked_entry_and_exit_are_retried_on_following_days():
    dates = pd.bdate_range("2025-01-01", periods=9)
    panel = pd.DataFrame({
        "trade_date": dates, "ts_code": "A", "raw_open": 10.0,
        "adj_open": np.linspace(10.0, 10.8, len(dates)),
        "adj_close": np.linspace(10.0, 10.8, len(dates)),
        "amount_cny": 1e8, "amount_ma20": 1e8, "volatility_20d": 0.02,
        "is_suspended": False, "is_st": False, "is_delisting_period": False,
        "listing_trade_days": 100, "is_limit_up_open": False,
        "is_limit_down_open": False,
    })
    panel.loc[panel["trade_date"].eq(dates[1]), "is_limit_up_open"] = True
    panel.loc[panel["trade_date"].eq(dates[6]), "is_limit_down_open"] = True
    targets = pd.DataFrame({
        "signal_date": [dates[0]], "entry_date": [dates[1]],
        "ts_code": ["A"], "target_weight": [0.5],
    })
    build = TargetBuildResult(targets, pd.DataFrame(), [dates[1], dates[6]], [dates[1]])
    result = run_non_overlapping_ledger(
        panel, build, start=dates[0], end=dates[-1],
        portfolio_rules=PortfolioRules(initial_cash=1_000_000),
        execution_rules=ExecutionRules(base_slippage_bps_per_side=0, impact_eta=0),
    )
    buys = result.trades.loc[result.trades["side"].eq("BUY")]
    sells = result.trades.loc[result.trades["side"].eq("SELL")]
    assert buys.iloc[0]["trade_date"] == dates[2]
    assert sells.iloc[0]["trade_date"] == dates[7]
    assert result.daily["blocked_buys"].sum() >= 1
    assert result.daily["blocked_sells"].sum() >= 1


def test_liquidity_targets_consolidate_multi_concept_stock_and_cap_weight():
    dates = pd.bdate_range("2025-01-01", periods=12)
    features = pd.DataFrame({
        "trade_date": [dates[0], dates[0]], "concept_code": ["C1", "C2"],
        "eligible_concept": True, "signal_rrg_only": [2.0, 1.0],
    })
    members = pd.DataFrame({
        "trade_date": [dates[0]] * 6,
        "concept_code": ["C1"] * 3 + ["C2"] * 3,
        "ts_code": ["A", "B", "C", "A", "D", "E"],
    })
    panel = pd.MultiIndex.from_product([dates, list("ABCDE")], names=["trade_date", "ts_code"]).to_frame(index=False)
    panel["amount_ma20"] = panel["ts_code"].map({"A": 5e8, "B": 4e8, "C": 1e8, "D": 3e8, "E": 1e8})
    panel["is_tradeable"] = True
    built = build_liquidity_neutral_targets(
        features, members, panel, signal_name="rrg_only", start=dates[0], end=dates[-1], offset=0,
        rules=PortfolioRules(concepts_per_rebalance=2, stocks_per_concept=2, maximum_stock_weight=0.4),
    )
    assert not built.targets.duplicated(["entry_date", "ts_code"]).any()
    assert built.targets["target_weight"].max() <= 0.4 + 1e-12
    assert built.targets.loc[built.targets["ts_code"].eq("A"), "supporting_concepts"].iloc[0] == 2


def test_continuous_breadth_residual_is_cross_sectionally_orthogonal_to_controls():
    rng = np.random.default_rng(7)
    n = 80
    rs = rng.normal(size=n)
    momentum = rng.normal(size=n)
    churn = rng.uniform(size=n)
    members = rng.integers(10, 200, size=n)
    noise = rng.normal(scale=0.2, size=n)
    features = pd.DataFrame({
        "trade_date": pd.Timestamp("2025-01-02"), "rs_z": rs,
        "rs_momentum_z": momentum, "membership_churn_5d": churn,
        "matched_member_count": members,
        "common_breadth_delta_smooth5": 0.5 * rs - 0.3 * momentum + 0.2 * churn + noise,
        "signal_rrg_only": rs + momentum,
    })
    result = attach_continuous_breadth_signals(features)
    residual = result["signal_common_breadth_residual"]
    assert abs(residual.corr(pd.Series(rs))) < 1e-10
    assert abs(residual.corr(pd.Series(momentum))) < 1e-10


def test_market_regime_features_are_prefix_stable():
    dates = pd.bdate_range("2025-01-01", periods=60)
    panel = pd.MultiIndex.from_product([list("ABC"), dates], names=["ts_code", "trade_date"]).to_frame(index=False)
    day = panel.groupby("ts_code").cumcount()
    panel["adj_close"] = 10 + day * 0.02 + panel["ts_code"].map({"A": 0, "B": 1, "C": 2})
    panel["is_tradeable"] = True
    panel["circ_mv_cny"] = 1e9
    prefix = build_market_regimes(panel.loc[panel["trade_date"].le(dates[39])])
    full = build_market_regimes(panel)
    columns = ["trade_date", "market_breadth", "breadth_delta_5d", "market_return_20d", "regime"]
    pd.testing.assert_frame_equal(
        prefix[columns].reset_index(drop=True), full.loc[full["trade_date"].le(dates[39]), columns].reset_index(drop=True)
    )


def test_rescale_target_build_matches_requested_exposure():
    targets = pd.DataFrame({
        "signal_date": pd.Timestamp("2025-01-01"), "entry_date": pd.Timestamp("2025-01-02"),
        "ts_code": list("ABCD"), "target_weight": [0.4, 0.3, 0.2, 0.1],
    })
    build = TargetBuildResult(targets, pd.DataFrame(), [pd.Timestamp("2025-01-02")], [pd.Timestamp("2025-01-02")])
    result = rescale_target_build(build, 0.6, maximum_stock_weight=0.2)
    assert result.targets["target_weight"].sum() == pytest.approx(0.6)
    assert result.targets["target_weight"].max() <= 0.2 + 1e-12
