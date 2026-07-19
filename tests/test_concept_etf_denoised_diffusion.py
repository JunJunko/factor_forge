import numpy as np
import pandas as pd
import pytest

from factor_forge.research.concept_etf_denoised_diffusion import (
    DENOISED_ML_FEATURES,
    DenoisedDiffusionRules,
    attach_denoised_diffusion_scores,
    attach_forward_open_returns,
    attach_learned_denoised_score,
    constrained_denoised_weights,
    diffusion_signal_diagnostics,
    fit_denoised_diffusion_walk_forward,
)


def sample_panel(periods: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2025-01-01", periods=periods)
    rows, concepts = [], []
    codes = ["A", "B", "C", "D", "E"]
    for code_index, code in enumerate(codes):
        concept_code = f"C{code_index}"
        price = 1.0
        for date_index, date in enumerate(dates):
            wave = np.sin((date_index + code_index) / 8)
            price *= 1 + 0.001 * wave + 0.0002 * code_index
            rows.append({
                "trade_date": date,
                "ts_code": code,
                "concept_code": concept_code,
                "score_etf_momentum": wave + code_index * 0.05,
                "common_breadth_delta_smooth5": wave / 10 + code_index / 100,
                "forward_open_5d": 0.01 * wave + 0.002 * code_index,
                "adj_open": price,
            })
            concepts.append({
                "trade_date": date,
                "concept_code": concept_code,
                "breadth_float": 0.5 + wave / 10,
                "concept_amount": 1e8 * (code_index + 1),
                "matched_member_count": 20 + code_index,
                "membership_churn_5d": 0.05 + code_index / 100,
            })
    return pd.DataFrame(rows), pd.DataFrame(concepts)


def test_denoised_features_are_causal_and_confirmation_only_removes_candidates():
    panel, concepts = sample_panel(35)
    base = attach_denoised_diffusion_scores(panel, concepts)
    mutated = panel.copy()
    final_date = mutated["trade_date"].max()
    mutated.loc[mutated["trade_date"].eq(final_date), "common_breadth_delta_smooth5"] = 99
    changed = attach_denoised_diffusion_scores(mutated, concepts)
    cutoff = sorted(panel["trade_date"].unique())[-2]
    columns = ["denoised_diffusion_score", "score_D1_confirmation", "score_D2_ten_percent_boost"]
    pd.testing.assert_frame_equal(
        base.loc[base["trade_date"].le(cutoff), columns].reset_index(drop=True),
        changed.loc[changed["trade_date"].le(cutoff), columns].reset_index(drop=True),
    )
    assert base["score_D1_confirmation"].isna().sum() > 0
    assert base.loc[base["score_D1_confirmation"].notna(), "score_D0_price"].equals(
        base.loc[base["score_D1_confirmation"].notna(), "score_D1_confirmation"]
    )


def test_denoised_weights_are_nonnegative_and_capped_at_ten_percent():
    weights = constrained_denoised_weights(
        np.array([0.1, 0.5, 0.3, 0.2]), maximum_diffusion_weight=0.10,
    )
    assert set(weights) == set(DENOISED_ML_FEATURES)
    assert all(weight >= 0 for weight in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)
    assert sum(weights[name] for name in DENOISED_ML_FEATURES[1:]) == pytest.approx(0.10)


def test_walk_forward_uses_mature_labels_and_stability_audit():
    panel, concepts = sample_panel()
    scored = attach_denoised_diffusion_scores(panel, concepts)
    rules = DenoisedDiffusionRules(
        minimum_train_days=20,
        validation_days=5,
        test_days=10,
        embargo_days=6,
        minimum_train_rows=100,
        stability_blocks=4,
    )
    predictions, weights, stability, audit = fit_denoised_diffusion_walk_forward(
        scored, start="2025-01-01", end="2025-06-17", rules=rules,
    )
    assert not predictions.empty
    assert not weights.empty
    assert not stability.empty
    assert (audit["train_label_available_max"] < audit["test_start"]).all()
    assert stability[DENOISED_ML_FEATURES].ge(0).all().all()
    assert stability[DENOISED_ML_FEATURES].le(1).all().all()
    assert weights[DENOISED_ML_FEATURES[1:]].sum(axis=1).le(0.1000001).all()
    attached = attach_learned_denoised_score(scored, predictions)
    assert attached["score_D3_learned"].notna().all()


def test_forward_horizons_and_diagnostics_are_available():
    panel, concepts = sample_panel(50)
    scored = attach_denoised_diffusion_scores(panel, concepts)
    labelled = attach_forward_open_returns(scored)
    assert {"forward_open_1d", "forward_open_3d", "forward_open_5d", "forward_open_10d"} <= set(labelled)
    ic, buckets = diffusion_signal_diagnostics(
        labelled, start="2025-01-01", end="2025-03-31",
    )
    assert set(ic["horizon"]) == {1, 3, 5, 10}
    assert set(buckets["bucket"]) == {1, 2, 3, 4, 5}
