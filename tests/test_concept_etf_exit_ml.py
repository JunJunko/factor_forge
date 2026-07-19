import pandas as pd
import pytest

from factor_forge.research.concept_etf_exit_ml import (
    ExitMLRules,
    build_exit_folds,
    enrich_exit_features,
    fit_exit_walk_forward,
    simulate_r4_exit_sleeves,
)


def exit_panel(periods: int = 25) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=periods)
    rows = []
    for code_index, code in enumerate(["A", "B", "C", "D"]):
        price = 100.0
        for date_index, date in enumerate(dates):
            if date_index >= 4:
                price *= 0.98
            rows.append({
                "trade_date": date, "ts_code": code, "concept_code": code,
                "adj_open": price, "adj_close": price,
                "amount_cny": 100_000_000.0,
                "etf_return_1d": -0.02 if date_index >= 4 else 0.0,
                "etf_momentum_20d": 0.10, "etf_momentum_60d": 0.10,
                "volatility_20d": 0.02 + code_index * 0.001,
                "mapping_pass": True, "eligible_concept": True, "match_type": "exact",
                "cluster": code, "score_etf_momentum": 4 - code_index,
                "etf_name": code, "concept_name": code,
                "concept_return_1d": -0.01 if date_index >= 4 else 0.0,
                "breadth_float": 0.60,
                "common_breadth_delta_smooth5": -0.10 if date_index >= 2 else 0.10,
                "common_delta_rank": 0.50, "rs_momentum_5d": -0.01,
                "rrg_quadrant": "weakening",
            })
    return pd.DataFrame(rows)


def test_enriched_exit_features_only_use_current_and_past_rows():
    panel = exit_panel(10)
    base = enrich_exit_features(panel)
    changed = panel.copy()
    changed.loc[changed["trade_date"].eq(changed["trade_date"].max()), "adj_close"] *= 4
    mutated = enrich_exit_features(changed)
    cutoff = panel["trade_date"].sort_values().unique()[-2]
    columns = ["etf_return_3d", "etf_return_5d", "breadth_acceleration_5d"]
    pd.testing.assert_frame_equal(
        base.loc[base["trade_date"].le(cutoff), columns].reset_index(drop=True),
        mutated.loc[mutated["trade_date"].le(cutoff), columns].reset_index(drop=True),
    )


def test_exit_folds_apply_embargo_around_validation_and_test():
    dates = pd.bdate_range("2025-01-01", periods=45)
    rules = ExitMLRules(
        minimum_train_days=10, validation_days=5, test_days=5, embargo_days=3,
    )
    folds = build_exit_folds(dates, rules=rules)
    first = folds[0]
    calendar = pd.Index(dates)
    assert calendar.get_loc(first.valid_start) - calendar.get_loc(first.train_end) == 4
    assert calendar.get_loc(first.test_start) - calendar.get_loc(first.valid_end) == 4
    assert calendar.get_loc(folds[1].test_start) == calendar.get_loc(first.test_end) + 1


def test_walk_forward_training_and_validation_labels_are_mature():
    dates = pd.bdate_range("2025-01-01", periods=55)
    panel_rows, state_rows = [], []
    for date_index, date in enumerate(dates):
        for code_index, code in enumerate(["A", "B"]):
            value = (date_index + code_index) / 100
            panel_rows.append({"trade_date": date, "ts_code": code, "etf_return_1d": value})
            if date_index + 2 < len(dates):
                state_rows.append({
                    "state_date": date, "ts_code": code,
                    "holding_age": 1 + code_index, "days_remaining": 4 - code_index,
                    "etf_return_1d": value,
                    "label_available_date": dates[date_index + 2],
                    "exit_advantage": -value,
                })
    rules = ExitMLRules(
        minimum_train_days=10, validation_days=5, test_days=5, embargo_days=3,
        minimum_train_rows=20, minimum_validation_rows=5,
    )
    predictions, _, audit, _ = fit_exit_walk_forward(
        pd.DataFrame(state_rows), pd.DataFrame(panel_rows),
        start=str(dates[0].date()), end=str(dates[-1].date()),
        feature_columns=["etf_return_1d", "holding_age", "days_remaining"], rules=rules,
    )
    assert not predictions.empty
    assert (audit["train_label_available_max"] < audit["valid_start"]).all()
    assert (audit["valid_label_available_max"] < audit["test_start"]).all()


def test_ml_exit_requires_two_negative_closes_and_executes_next_open():
    panel = enrich_exit_features(exit_panel())
    ages = []
    for date in panel["trade_date"].unique():
        for code in panel["ts_code"].unique():
            for age in range(1, 5):
                ages.append({
                    "state_date": date, "ts_code": code, "holding_age": age,
                    "days_remaining": 5 - age,
                    "predicted_exit_advantage": 0.03,
                    "placebo_exit_advantage": -0.03,
                })
    predictions = pd.DataFrame(ages)
    fixed, _, _ = simulate_r4_exit_sleeves(
        panel, predictions, policy="X0_fixed", start="2025-01-01", end="2025-02-04",
        roundtrip_cost_bps=20,
    )
    exited, sleeves, events = simulate_r4_exit_sleeves(
        panel, predictions, policy="X2_ml", start="2025-01-01", end="2025-02-04",
        roundtrip_cost_bps=20,
    )
    assert not events.empty
    assert events["holding_age"].ge(2).all()
    assert (events["exit_date"] > events["exit_signal_date"]).all()
    assert sleeves["is_early_exit"].any()
    assert exited.iloc[-1]["net_nav"] > fixed.iloc[-1]["net_nav"]


def test_placebo_negative_score_does_not_trigger_exit():
    panel = enrich_exit_features(exit_panel())
    predictions = pd.DataFrame({
        "state_date": [panel["trade_date"].min()], "ts_code": ["A"],
        "holding_age": [1], "days_remaining": [4],
        "predicted_exit_advantage": [0.05], "placebo_exit_advantage": [-0.05],
    })
    _, sleeves, events = simulate_r4_exit_sleeves(
        panel, predictions, policy="X3_placebo", start="2025-01-01", end="2025-02-04",
        roundtrip_cost_bps=20,
    )
    assert events.empty
    assert not sleeves["is_early_exit"].any()
    assert sleeves["turnover"].ge(0).all()
    assert sleeves["cash_weight"].between(0, 1).all()
    assert sleeves["net_nav"].notna().all()
    assert sleeves["cost_drag"].ge(0).all()
    assert sleeves["net_return"].gt(-1).all()
    assert sleeves["gross_return"].gt(-1).all()
    assert sleeves["target_weights"].str.len().gt(0).all()
    assert sleeves["sleeve"].nunique() == 5
    assert sleeves["portfolio"].eq("X3_placebo").all()
    assert sleeves["variant"].eq("X3_placebo").all()
    assert sleeves.iloc[0]["net_nav"] == pytest.approx(1.0)
