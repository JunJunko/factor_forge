from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_forge.breakout_process import (
    BreakoutConfig,
    BreakoutEventBuilder,
    BreakoutProcessEngine,
    FactorOperator,
    FactorStage,
    OperatorRegistry,
    default_operator_registry,
)
from factor_forge.breakout_process.research import BreakoutResearchRunner
from factor_forge.breakout_process.backtest import EventBacktestRunner
from factor_forge.breakout_process.scenario_backtest import ScenarioBacktestRunner
from factor_forge.breakout_process.strategy_backtest import FrozenBreakoutStrategyRunner
from factor_forge.breakout_process.validation import exposure_residual, frozen_score


def _config(**overrides) -> BreakoutConfig:
    values = {
        "box_lookback": 8,
        "atr_window": 3,
        "volatility_short_window": 2,
        "volatility_long_window": 4,
        "process_window": 4,
        "acceleration_window": 2,
        "volume_window": 3,
        "max_active_days": 5,
        "max_box_width_atr": 20.0,
        "max_abs_slope_atr": 20.0,
        "max_volatility_ratio": None,
    }
    values.update(overrides)
    return BreakoutConfig(**values)


def _panel(closes: list[float], *, highs: list[float] | None = None) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=len(closes))
    close = np.asarray(closes, dtype=float)
    high = np.asarray(highs, dtype=float) if highs is not None else close + 0.05
    return pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": "000001.SZ",
            "adj_open": close - 0.01,
            "adj_high": high,
            "adj_low": close - 0.05,
            "adj_close": close,
            "volume_shares": np.linspace(1000, 1200, len(close)),
        }
    )


def test_frozen_boundary_detects_breakout_that_a_rolling_boundary_would_hide():
    closes = [10.00, 10.02, 10.01, 10.03, 10.00, 10.04, 10.02, 10.05, 10.08, 10.12]
    highs = [value + 0.05 for value in closes]
    highs[8] = 10.20  # Intraday high after box creation must not move the frozen upper.
    result = BreakoutProcessEngine(_config()).run(_panel(closes, highs=highs))

    assert len(result.events) == 1
    event = result.events.iloc[0]
    assert event.upper == pytest.approx(10.10)
    assert event.event_time == pd.Timestamp("2025-01-15")
    assert event.pre_window_end == pd.Timestamp("2025-01-14")
    assert event.breakout_strength > 0
    assert result.boxes.iloc[0].state == "triggered"
    assert result.boxes.iloc[0].close_reason == "breakout"


def test_event_factor_snapshot_has_the_three_explicit_stages():
    closes = [10.00, 10.00, 10.01, 10.00, 10.01, 10.02, 10.04, 10.06, 10.08, 10.15]
    result = BreakoutProcessEngine(_config()).run(_panel(closes))
    event = result.events.iloc[0]

    expected = {
        "range_compactness",
        "volatility_contraction",
        "trend_flatness",
        "approach_velocity",
        "pre_acceleration",
        "direction_persistence",
        "consolidation_age",
        "breakout_strength",
        "breakout_velocity",
        "breakout_acceleration",
        "relative_volume",
        "gap_atr",
    }
    assert expected.issubset(result.events.columns)
    assert event.available_time == event.event_time
    assert event.pre_window_end < event.event_time


def test_future_rows_do_not_change_an_existing_event():
    closes = [10.00, 10.02, 10.01, 10.03, 10.00, 10.04, 10.02, 10.05, 10.08, 10.16]
    prefix = _panel(closes)
    future = _panel(closes + [12.0, 8.0, 11.0])
    engine = BreakoutProcessEngine(_config())

    prefix_event = engine.run(prefix).events.iloc[0]
    future_event = engine.run(future).events.iloc[0]
    comparable = [
        "box_id",
        "event_time",
        "upper",
        "lower",
        "frozen_atr",
        "pre_acceleration",
        "breakout_strength",
        "breakout_acceleration",
    ]
    pd.testing.assert_series_equal(
        prefix_event[comparable], future_event[comparable], check_names=False
    )


def test_box_can_close_without_emitting_a_breakout_event():
    closes = [10.00, 10.02, 10.01, 10.03, 10.00, 10.04, 10.02, 10.05, 9.80]
    result = BreakoutProcessEngine(_config()).run(_panel(closes))

    assert result.events.empty
    assert len(result.boxes) == 1
    assert result.boxes.iloc[0].state == "closed"
    assert result.boxes.iloc[0].close_reason == "downside_failure"


def test_operator_registry_is_injectable_without_using_the_existing_factor_engine():
    registry = default_operator_registry()
    registry.register(
        FactorOperator(
            "custom_box_midpoint",
            FactorStage.SETUP,
            lambda context: (context.box.upper + context.box.lower) / 2,
        )
    )
    closes = [10.00, 10.02, 10.01, 10.03, 10.00, 10.04, 10.02, 10.05, 10.08, 10.16]
    result = BreakoutProcessEngine(_config(), operators=registry).run(_panel(closes))

    assert "custom_box_midpoint" in result.events
    assert result.events.iloc[0].custom_box_midpoint == pytest.approx(
        (result.events.iloc[0].upper + result.events.iloc[0].lower) / 2
    )


def test_invalid_or_duplicate_input_is_rejected():
    panel = _panel([10.0] * 10)
    duplicate = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        BreakoutProcessEngine(_config()).run(duplicate)

    with pytest.raises(ValueError, match="missing columns"):
        BreakoutProcessEngine(_config()).run(panel.drop(columns="adj_high"))


def test_fast_event_builder_matches_extensible_engine_event_semantics():
    closes = [10.00, 10.02, 10.01, 10.03, 10.00, 10.04, 10.02, 10.05, 10.08, 10.16]
    panel = _panel(closes)
    expected = BreakoutProcessEngine(_config()).run(panel).events.iloc[0]
    actual = BreakoutEventBuilder(_config()).run(panel).events.iloc[0]

    columns = [
        "box_id",
        "event_time",
        "upper",
        "lower",
        "frozen_atr",
        "range_compactness",
        "volatility_contraction",
        "trend_flatness",
        "approach_velocity",
        "pre_acceleration",
        "direction_persistence",
        "breakout_strength",
        "breakout_velocity",
        "breakout_acceleration",
        "relative_volume",
        "gap_atr",
    ]
    pd.testing.assert_series_equal(expected[columns], actual[columns], check_names=False)


def test_research_runner_finds_a_stable_conditional_ic_in_synthetic_events():
    rng = np.random.default_rng(7)
    rows = []
    for date in pd.bdate_range("2024-01-02", periods=50):
        signal = np.linspace(-1, 1, 12) + rng.normal(0, 0.02, 12)
        for index, value in enumerate(signal):
            rows.append(
                {
                    "trade_date": date,
                    "ts_code": f"{index:06d}.SZ",
                    "range_compactness": value,
                    "volatility_contraction": value,
                    "trend_flatness": value,
                    "approach_velocity": value,
                    "pre_acceleration": value,
                    "direction_persistence": value,
                    "breakout_strength": value,
                    "breakout_velocity": value,
                    "breakout_acceleration": value,
                    "relative_volume": value,
                    "continuous_move": value,
                    "consolidation_age": 40 + index,
                    "gap_atr": 0.1,
                    "market_trend_20": 0.01,
                    "market_volatility_20": 0.02,
                    "market_volatility_reference": 0.01,
                    "forward_return_1": value + rng.normal(0, 0.05),
                }
            )
    events = pd.DataFrame(rows)
    runner = BreakoutResearchRunner()
    specs, scores = runner._build_scores(events, include_pairs=False)
    scored = pd.concat([events, scores], axis=1)
    results, daily = runner._evaluate(
        scored,
        specs,
        {"all": pd.Series(True, index=scored.index)},
        (1,),
        min_cross_section=8,
        min_ic_days=20,
    )

    row = results.loc[results.score == "single:pre_acceleration"].iloc[0]
    assert row.rank_ic_mean > 0.9
    assert row.promising
    assert daily.trade_date.nunique() == 50


def test_event_backtest_uses_next_open_and_defers_limit_down_exit():
    dates = list(pd.bdate_range("2024-01-02", periods=5))
    frames = {}
    for index, date in enumerate(dates):
        frames[date] = pd.DataFrame(
            {
                "ts_code": ["A.SZ"],
                "raw_open": [10.0 + index],
                "adj_open": [10.0 + index],
                "adj_close": [10.0 + index],
                "is_suspended": [False],
                "is_limit_up_open": [False],
                "is_limit_down_open": [index == 2],
                "is_st": [False],
                "is_delisting_period": [False],
                "listing_trade_days": [100],
            }
        ).set_index("ts_code")
    daily, trades, _ = EventBacktestRunner()._simulate(
        dates,
        frames,
        {dates[0]: ["A.SZ"]},
        holding_days=1,
        initial_cash=100_000,
        lot_size=100,
        min_listing_days=60,
        cost_bps=0,
        allocation_count=1,
    )

    buy = trades.loc[trades.side == "BUY"].iloc[0]
    sell = trades.loc[trades.side == "SELL"].iloc[0]
    assert buy.trade_date == dates[1]
    assert sell.trade_date == dates[3]
    assert daily.iloc[-1].nav > 100_000


def test_scenario_condition_deduplication_preserves_aliases():
    candidates = pd.DataFrame(
        {
            "score": ["A", "A", "B"],
            "condition": ["all", "contracting", "other"],
            "horizon": [10, 10, 10],
            "exploratory_score": [3.0, 2.0, 1.0],
        }
    )
    conditions = {
        "all": pd.Series([True, True, False]),
        "contracting": pd.Series([True, True, False]),
        "other": pd.Series([False, True, False]),
    }
    result, masks = ScenarioBacktestRunner._deduplicate_conditions(candidates, conditions)

    assert len(result) == 2
    first = result.loc[result.score == "A"].iloc[0]
    assert first.canonical_condition == "all"
    assert first.condition_aliases == "all+contracting"
    assert set(masks) == {"all", "other"}


def test_sma_signal_filter_uses_only_information_available_at_signal_close():
    dates = pd.bdate_range("2024-01-02", periods=6)
    panel = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": "000001.SZ",
            "adj_close": [10.0, 10.0, 10.0, 9.0, 11.0, 1.0],
        }
    )
    signals = pd.DataFrame(
        {
            "trade_date": [dates[3], dates[4]],
            "ts_code": "000001.SZ",
            "factor_value": [1.0, 2.0],
        }
    )
    result = FrozenBreakoutStrategyRunner._apply_signal_filter(
        signals, panel, {"type": "close_above_sma", "window": 3}
    )

    assert result.trade_date.tolist() == [dates[4]]


def test_frozen_score_is_exact_equal_weight_daily_rank_formula():
    frame = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2025-01-02")] * 3,
            "approach_velocity": [1.0, 2.0, 3.0],
            "gap_atr": [0.3, 0.2, 0.1],
        }
    )
    actual = frozen_score(frame)
    expected = pd.Series([1 / 3, 2 / 3, 1.0])
    pd.testing.assert_series_equal(actual.reset_index(drop=True), expected)


def test_exposure_residual_removes_daily_size_and_industry_linear_exposure():
    rows = []
    for index in range(20):
        industry = "A" if index < 10 else "B"
        size = float(index)
        rows.append(
            {
                "trade_date": pd.Timestamp("2025-01-02"),
                "industry_l2_code": industry,
                "log_total_mv": size,
                "value": 2.0 * size + (5.0 if industry == "B" else 0.0),
            }
        )
    frame = pd.DataFrame(rows)
    residual = exposure_residual(frame, "value")
    assert residual.abs().max() < 1e-10
