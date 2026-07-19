import pandas as pd

from factor_forge.research.concept_etf_exit_ml import (
    ExitMLRules,
    build_fixed_r4_held_states,
    enrich_exit_features,
    simulate_r4_exit_sleeves,
)
from factor_forge.research.concept_etf_tail_exit_ml import (
    evaluate_tail_predictions,
    fit_tail_walk_forward,
)


def tail_panel(periods: int = 55) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=periods)
    rows = []
    for code_index, code in enumerate(["A", "B", "C", "D"]):
        price = 100.0
        for date_index, date in enumerate(dates):
            if date_index % 11 in {6, 7}:
                price *= 0.95
            else:
                price *= 1.003
            rows.append({
                "trade_date": date, "ts_code": code, "concept_code": code,
                "adj_open": price, "adj_close": price, "amount_cny": 100_000_000.0,
                "etf_return_1d": -0.05 if date_index % 11 in {6, 7} else 0.003,
                "etf_momentum_20d": 0.08, "etf_momentum_60d": 0.12,
                "volatility_20d": 0.02 + code_index * 0.001,
                "mapping_pass": True, "eligible_concept": True, "match_type": "exact",
                "cluster": code, "score_etf_momentum": 4 - code_index,
                "etf_name": code, "concept_name": code,
                "concept_return_1d": -0.04 if date_index % 11 in {6, 7} else 0.002,
                "breadth_float": 0.55,
                "common_breadth_delta_smooth5": -0.1 if date_index % 11 >= 5 else 0.1,
                "common_delta_rank": 0.4, "rs_momentum_5d": -0.01,
                "rrg_quadrant": "weakening",
            })
    return enrich_exit_features(pd.DataFrame(rows))


def test_tail_label_starts_at_next_open_and_matures_within_two_days():
    panel = tail_panel()
    rules = ExitMLRules(tail_horizon_days=2, tail_loss_threshold=-0.03)
    states, _ = build_fixed_r4_held_states(
        panel, start="2025-01-01", end="2025-03-18", rules=rules,
    )
    assert not states.empty
    assert (states["tail_label_available_date"] > states["next_open_date"]).all()
    assert (states["tail_label_available_date"] <= states["label_available_date"]).all()
    assert states["tail_event"].eq(states["tail_worst_open_return"].le(-0.03)).all()


def test_tail_walk_forward_uses_only_mature_labels():
    panel = tail_panel()
    rules = ExitMLRules(
        minimum_train_days=10, validation_days=5, test_days=5, embargo_days=3,
        minimum_train_rows=20, minimum_validation_rows=5,
    )
    states, _ = build_fixed_r4_held_states(
        panel, start="2025-01-01", end="2025-03-18", rules=rules,
    )
    features = ["etf_return_1d", "common_delta_rank", "holding_age", "days_remaining"]
    predictions, _, audit, _ = fit_tail_walk_forward(
        states, panel, start="2025-01-01", end="2025-03-18",
        feature_columns=features, rules=rules,
    )
    assert not predictions.empty
    assert (audit["train_label_available_max"] < audit["valid_start"]).all()
    assert (audit["valid_label_available_max"] < audit["test_start"]).all()


def test_tail_exit_acts_after_one_close_and_records_realized_tail_event():
    panel = tail_panel(25)
    predictions = []
    for date in panel["trade_date"].unique():
        for code in panel["ts_code"].unique():
            for age in range(1, 5):
                predictions.append({
                    "state_date": date, "ts_code": code, "holding_age": age,
                    "days_remaining": 5 - age,
                    "predicted_tail_return": -0.05,
                    "placebo_tail_return": 0.00,
                })
    rules = ExitMLRules(consecutive_negative_days=1, tail_loss_threshold=-0.03)
    _, sleeves, events = simulate_r4_exit_sleeves(
        panel, pd.DataFrame(predictions), policy="T1_tail_ml",
        start="2025-01-01", end="2025-02-04", roundtrip_cost_bps=20, rules=rules,
    )
    assert not events.empty
    assert events["holding_age"].ge(1).all()
    assert (events["exit_date"] > events["exit_signal_date"]).all()
    assert {"tail_worst_return_after_exit", "tail_event_after_exit"}.issubset(events)
    assert sleeves["is_early_exit"].any()


def test_tail_prediction_audit_computes_precision_and_recall():
    states = pd.DataFrame({
        "state_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        "ts_code": ["A", "A", "A"], "holding_age": [1, 1, 1],
        "tail_worst_open_return": [-0.05, -0.01, -0.04],
        "tail_event": [True, False, True],
    })
    predictions = pd.DataFrame({
        "state_date": states["state_date"], "ts_code": "A", "holding_age": 1,
        "days_remaining": 4,
        "predicted_tail_return": [-0.06, -0.04, -0.01],
        "placebo_tail_return": [0.0, 0.0, 0.0], "fold": 0,
    })
    audit, _ = evaluate_tail_predictions(states, predictions, loss_threshold=-0.03)
    assert audit["tail_precision"] == 0.5
    assert audit["tail_recall"] == 0.5
