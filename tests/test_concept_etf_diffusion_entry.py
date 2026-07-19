import numpy as np
import pandas as pd
import pytest

from factor_forge.research.concept_etf_diffusion_entry import (
    DIFFUSION_FEATURES,
    DiffusionBlendRules,
    attach_learned_oof_score,
    attach_positive_diffusion_scores,
    constrained_blend_weights,
    fit_positive_diffusion_walk_forward,
)
from factor_forge.research.concept_etf_shadow import staggered_target_weights


def diffusion_panel(periods: int = 120) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=periods)
    rows = []
    for code_index, code in enumerate(["A", "B", "C", "D", "E"]):
        for date_index, date in enumerate(dates):
            wave = np.sin((date_index + code_index) / 8)
            rows.append({
                "trade_date": date, "ts_code": code,
                "score_etf_momentum": wave + code_index * 0.05,
                "common_delta_rank": (code_index + 1) / 5,
                "common_breadth_delta_smooth5": wave / 10,
                "forward_open_5d": 0.01 * wave + 0.002 * code_index,
            })
    return pd.DataFrame(rows)


def test_positive_diffusion_features_do_not_use_future_rows():
    panel = diffusion_panel(30)
    base = attach_positive_diffusion_scores(panel)
    mutated = panel.copy()
    final_date = mutated["trade_date"].max()
    mutated.loc[mutated["trade_date"].eq(final_date), "common_breadth_delta_smooth5"] = 99
    changed = attach_positive_diffusion_scores(mutated)
    cutoff = sorted(panel["trade_date"].unique())[-2]
    columns = ["positive_diffusion_score", "score_B1_fixed_diffusion"]
    pd.testing.assert_frame_equal(
        base.loc[base["trade_date"].le(cutoff), columns].reset_index(drop=True),
        changed.loc[changed["trade_date"].le(cutoff), columns].reset_index(drop=True),
    )


def test_constrained_weights_are_positive_and_cap_total_diffusion():
    weights = constrained_blend_weights(
        np.array([0.1, 0.5, 0.3, 0.2]), maximum_diffusion_weight=0.30,
    )
    assert set(weights) == set(DIFFUSION_FEATURES)
    assert all(weight >= 0 for weight in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)
    assert sum(weights[name] for name in DIFFUSION_FEATURES[1:]) == pytest.approx(0.30)


def test_positive_ridge_walk_forward_uses_only_mature_labels():
    panel = attach_positive_diffusion_scores(diffusion_panel())
    rules = DiffusionBlendRules(
        minimum_train_days=20, validation_days=5, test_days=10,
        embargo_days=6, minimum_train_rows=100,
    )
    predictions, weights, audit = fit_positive_diffusion_walk_forward(
        panel, start="2025-01-01", end="2025-06-17", rules=rules,
    )
    assert not predictions.empty
    assert not weights.empty
    assert (audit["train_label_available_max"] < audit["test_start"]).all()
    diffusion_weight = weights[DIFFUSION_FEATURES[1:]].sum(axis=1)
    assert diffusion_weight.between(0, 0.3000001).all()
    attached = attach_learned_oof_score(panel, predictions)
    assert attached["score_B2_learned_diffusion"].notna().all()


def test_r4_can_rank_on_positive_diffusion_score_column():
    day = pd.DataFrame({
        "ts_code": ["A", "B", "C", "D"],
        "mapping_pass": True, "eligible_concept": True, "match_type": "exact",
        "cluster": ["A", "B", "C", "D"],
        "score_etf_momentum": [4.0, 3.0, 2.0, 1.0],
        "score_diffusion": [1.0, 2.0, 3.0, 4.0],
        "etf_momentum_60d": 0.10,
        "volatility_20d": [0.01, 0.02, 0.03, 0.04],
    })
    weights = staggered_target_weights(
        day, "R4_rank_buffer", score_column="score_diffusion",
    )
    assert "D" in weights
    assert sum(weights.values()) == pytest.approx(1.0)
