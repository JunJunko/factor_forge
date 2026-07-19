import pandas as pd

from factor_forge.research.concept_etf_exit_state import (
    ExitStateRules,
    classify_exit_state,
)


def row(**overrides):
    values = {
        "price_weak_exit": False,
        "diffusion_weak_exit": False,
        "relative_weak_exit": False,
        "absolute_breakdown_exit": False,
        "score_rank_exit": 1.0,
    }
    values.update(overrides)
    return pd.Series(values)


def test_price_policy_requires_two_confirmations_before_reduction():
    first = classify_exit_state(
        row(price_weak_exit=True, score_rank_exit=8),
        policy="E1_price_confirmed",
        price_streak=0,
        dual_streak=0,
        severe_streak=0,
        already_reduced=False,
    )
    second = classify_exit_state(
        row(price_weak_exit=True, score_rank_exit=8),
        policy="E1_price_confirmed",
        price_streak=first["price_streak"],
        dual_streak=first["dual_streak"],
        severe_streak=first["severe_streak"],
        already_reduced=False,
    )
    assert first["status"] == "WATCH"
    assert second["status"] == "REDUCE"
    assert second["action"] == "reduce_half"


def test_diffusion_policy_needs_price_and_diffusion_for_reduction():
    rules = ExitStateRules(confirmation_days=2)
    first = classify_exit_state(
        row(price_weak_exit=True, diffusion_weak_exit=False, score_rank_exit=8),
        policy="E2_price_diffusion_state",
        price_streak=0,
        dual_streak=0,
        severe_streak=0,
        already_reduced=False,
        rules=rules,
    )
    second = classify_exit_state(
        row(price_weak_exit=True, diffusion_weak_exit=False, score_rank_exit=8),
        policy="E2_price_diffusion_state",
        price_streak=first["price_streak"],
        dual_streak=first["dual_streak"],
        severe_streak=first["severe_streak"],
        already_reduced=False,
        rules=rules,
    )
    assert second["status"] == "WATCH"
    assert second["action"] == "none"


def test_absolute_momentum_breakdown_is_immediate_next_open_sell():
    result = classify_exit_state(
        row(absolute_breakdown_exit=True),
        policy="E2_price_diffusion_state",
        price_streak=0,
        dual_streak=0,
        severe_streak=0,
        already_reduced=False,
    )
    assert result["status"] == "SELL"
    assert result["action"] == "sell_all"


def test_severe_three_family_confirmation_sells_after_two_days():
    first = classify_exit_state(
        row(
            price_weak_exit=True,
            diffusion_weak_exit=True,
            relative_weak_exit=True,
            score_rank_exit=11,
        ),
        policy="E2_price_diffusion_state",
        price_streak=0,
        dual_streak=0,
        severe_streak=0,
        already_reduced=False,
    )
    second = classify_exit_state(
        row(
            price_weak_exit=True,
            diffusion_weak_exit=True,
            relative_weak_exit=True,
            score_rank_exit=11,
        ),
        policy="E2_price_diffusion_state",
        price_streak=first["price_streak"],
        dual_streak=first["dual_streak"],
        severe_streak=first["severe_streak"],
        already_reduced=False,
    )
    assert second["status"] == "SELL"
    assert second["action"] == "sell_all"
