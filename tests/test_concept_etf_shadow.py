import pandas as pd
import pytest

from factor_forge.research.concept_etf_shadow import (
    CASH,
    _turnover,
    monthly_performance,
    nonoverlap_sleeve_statistics,
    nonoverlapping_holding_periods,
    simulate_portfolio,
    simulate_staggered_sleeves,
    simulate_weekly_daily_nav,
    staggered_target_weights,
    target_weights,
)


def sample_day() -> pd.DataFrame:
    return pd.DataFrame({
        "ts_code": ["A", "B", "C", "D"],
        "mapping_pass": True, "eligible_concept": True,
        "match_type": ["exact", "proxy", "approximate", "exact"],
        "cluster": ["ai", "ai", "health", "energy"],
        "score_etf_momentum": [4.0, 3.0, 2.0, 1.0],
        "rrg_quadrant": ["leading", "leading", "lagging", "improving"],
        "common_breadth_delta_smooth5": [-0.1, 0.1, -0.1, 0.1],
        "rs_momentum_5d": [-0.2, 0.2, 0.2, 0.1],
    })


def test_shadow_targets_respect_clusters_and_overlay_cash():
    p1 = target_weights(sample_day(), "P1_etf_momentum", top_n=3)
    assert set(p1) == {"A", "C", "D", CASH}
    assert p1[CASH] == pytest.approx(0)
    p2 = target_weights(sample_day(), "P2_breadth_overlay", top_n=3)
    assert p2["A"] == pytest.approx(1 / 6)
    assert p2[CASH] == pytest.approx(1 / 6)


def test_full_switch_turnover_is_one():
    assert _turnover({"A": 1.0, CASH: 0.0}, {"B": 1.0, CASH: 0.0}) == pytest.approx(1.0)
    assert _turnover({CASH: 1.0}, {"A": 1.0, CASH: 0.0}) == pytest.approx(1.0)


def test_simulator_charges_initial_turnover_and_uses_forward_open_return():
    day = sample_day()
    day["trade_date"] = pd.Timestamp("2025-01-01")
    day["forward_open_5d"] = [0.10, 0.20, 0.30, 0.50]
    result = simulate_portfolio(
        day, "P1_etf_momentum", start="2025-01-01", end="2025-01-01",
        horizon=5, top_n=3, roundtrip_cost_bps=20,
    )
    assert result.iloc[0]["gross_return"] == pytest.approx((0.10 + 0.30 + 0.50) / 3)
    assert result.iloc[0]["turnover"] == pytest.approx(1.0)
    assert result.iloc[0]["net_return"] == pytest.approx(0.30 - 0.002)


def test_weekly_daily_nav_executes_after_friday_and_reports_month_drawdown():
    dates = pd.bdate_range("2025-01-06", periods=10)
    rows = []
    for code, score in (("A", 2.0), ("B", 1.0)):
        for index, date in enumerate(dates):
            rows.append({
                "trade_date": date, "ts_code": code, "adj_open": 100 + index * (1 if code == "A" else 0),
                "mapping_pass": True, "eligible_concept": True, "match_type": "exact",
                "cluster": code, "score_etf_momentum": score, "rrg_quadrant": "leading",
                "common_breadth_delta_smooth5": 0.1, "rs_momentum_5d": 0.1,
            })
    panel = pd.DataFrame(rows)
    daily = simulate_weekly_daily_nav(
        panel, "P1_etf_momentum", start="2025-01-06", end="2025-01-17",
        top_n=1, roundtrip_cost_bps=20,
    )
    assert daily.iloc[0]["holding_date"] == pd.Timestamp("2025-01-13")
    assert daily.iloc[0]["signal_date"] == pd.Timestamp("2025-01-10")
    assert daily.iloc[0]["turnover"] == pytest.approx(1.0)
    monthly = monthly_performance(daily)
    assert monthly.iloc[0]["month"] == "2025-01"
    assert monthly.iloc[0]["monthly_max_drawdown"] <= 0


def test_absolute_momentum_leaves_failed_slots_in_cash():
    day = sample_day()
    day["etf_momentum_60d"] = [0.10, 0.10, -0.05, 0.20]
    weights = staggered_target_weights(day, "R2_absolute_momentum")
    assert set(weights) == {"A", "D", CASH}
    assert weights[CASH] == pytest.approx(1 / 3)


def test_staggered_targets_support_proxy_and_explicit_exclusions():
    day = sample_day()
    day["etf_momentum_60d"] = 0.10
    no_proxy = staggered_target_weights(day, "R1_staggered_momentum", universe="no_proxy")
    assert "B" not in no_proxy
    excluded = staggered_target_weights(
        day, "R1_staggered_momentum", excluded_etfs={"A"},
    )
    assert "A" not in excluded


def test_inverse_volatility_variant_caps_single_etf_at_thirty_percent():
    day = sample_day()
    day["etf_momentum_60d"] = 0.10
    day["volatility_20d"] = [0.01, 0.02, 0.03, 0.04]
    weights = staggered_target_weights(day, "R3_inverse_volatility")
    assert max(weight for code, weight in weights.items() if code != CASH) <= 0.3000001
    assert sum(weights.values()) == pytest.approx(1.0)


def test_staggered_simulator_combines_five_independent_sleeves():
    dates = pd.bdate_range("2025-01-01", periods=30)
    rows = []
    for code_index, code in enumerate(["A", "B", "C", "D", "E"]):
        for date_index, date in enumerate(dates):
            rows.append({
                "trade_date": date, "ts_code": code,
                "adj_open": 100 + date_index * (code_index + 1) * 0.1,
                "etf_return_1d": 0.001 * (code_index + 1),
                "etf_momentum_60d": 0.10, "volatility_20d": 0.01 + code_index * 0.005,
                "mapping_pass": True, "eligible_concept": True, "match_type": "exact",
                "cluster": code, "score_etf_momentum": 5 - code_index,
                "etf_name": code, "concept_name": code,
            })
    aggregate, sleeves, attribution = simulate_staggered_sleeves(
        pd.DataFrame(rows), "R1_staggered_momentum",
        start=str(dates[0].date()), end=str(dates[-1].date()), roundtrip_cost_bps=20,
    )
    assert sleeves["sleeve"].nunique() == 5
    assert aggregate.iloc[-1]["net_nav"] > 1
    assert attribution["positive_profit_share"].sum() == pytest.approx(1.0)
    sleeves["roundtrip_cost_bps"] = 20
    periods = nonoverlapping_holding_periods(sleeves)
    assert periods["holding_days"].eq(5).all()
    stats, paired = nonoverlap_sleeve_statistics(periods, periods, bootstrap_samples=50)
    assert paired["net_excess"].eq(0).all()
    assert stats["mean_net_excess"].eq(0).all()


def test_r4_robustness_parameters_change_sleeves_and_rank_buffer():
    day = pd.DataFrame({
        "ts_code": list("ABCDEF"),
        "mapping_pass": True,
        "eligible_concept": True,
        "match_type": "exact",
        "cluster": list("abcdef"),
        "score_etf_momentum": [6, 5, 4, 3, 2, 1],
        "etf_momentum_60d": 0.10,
        "volatility_20d": 0.02,
    })
    weights = staggered_target_weights(
        day,
        "R4_rank_buffer",
        previous_holdings={"E"},
        r4_selection_count=3,
        r4_retention_rank=5,
        r4_maximum_etf_weight=0.40,
    )
    assert set(weights) == {"A", "B", "E", CASH}
    assert max(weight for code, weight in weights.items() if code != CASH) <= 0.4000001

    dates = pd.bdate_range("2025-01-01", periods=12)
    panel = pd.concat([
        day.assign(
            trade_date=date,
            adj_open=100 + date_index,
            etf_return_1d=0.01,
            etf_name=day["ts_code"],
            concept_name=day["ts_code"],
        )
        for date_index, date in enumerate(dates)
    ], ignore_index=True)
    _, sleeves, _ = simulate_staggered_sleeves(
        panel,
        "R4_rank_buffer",
        start=str(dates[0].date()),
        end=str(dates[-1].date()),
        holding_days=3,
        r4_selection_count=3,
        r4_retention_rank=5,
        r4_maximum_etf_weight=0.40,
    )
    assert sleeves["sleeve"].nunique() == 3

    _, delayed, _ = simulate_staggered_sleeves(
        panel,
        "R4_rank_buffer",
        start=str(dates[0].date()),
        end=str(dates[-1].date()),
        holding_days=3,
        execution_delay_days=2,
    )
    first_execution = delayed.loc[
        delayed["sleeve"].eq(0) & delayed["is_rebalance"]
    ].iloc[0]
    assert first_execution["signal_date"] == dates[0]
    assert first_execution["holding_date"] == dates[2]
