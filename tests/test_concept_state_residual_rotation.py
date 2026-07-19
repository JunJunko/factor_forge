import numpy as np
import pandas as pd

from factor_forge.research.concept_first_rotation import build_concept_first_features
from factor_forge.research.concept_state_residual_rotation import (
    StateResidualRules,
    attach_state_residual_scores_to_etfs,
    fit_state_residual_walk_forward,
    state_residual_coefficient_stability,
    within_state_oof_diagnostics,
)


def residual_concepts(concept_count: int = 24, periods: int = 150) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=periods)
    market_1d = 0.0004 * np.sin(np.arange(periods) / 10)
    market_price = np.cumprod(1 + market_1d)
    rows = []
    for concept_index in range(concept_count):
        returns = market_1d + 0.001 * np.sin((np.arange(periods) + concept_index) / 8)
        prices = np.cumprod(1 + returns)
        for index, date in enumerate(dates):
            def trailing(series, horizon):
                return series[index] / series[index - horizon] - 1 if index >= horizon else np.nan

            def label(horizon):
                if index + horizon + 1 >= periods:
                    return np.nan
                return (
                    prices[index + horizon + 1] / prices[index + 1]
                    - market_price[index + horizon + 1] / market_price[index + 1]
                )

            diffusion = np.sin((index + concept_index) / 12) / 10
            rows.append({
                "trade_date": date,
                "concept_code": f"C{concept_index:02d}",
                "concept_return_1d": returns[index],
                "concept_return_5d": trailing(prices, 5),
                "concept_return_20d": trailing(prices, 20),
                "concept_return_60d": trailing(prices, 60),
                "market_return_5d": trailing(market_price, 5),
                "market_return_20d": trailing(market_price, 20),
                "market_return_60d": trailing(market_price, 60),
                "rs_momentum_5d": diffusion / 2,
                "common_breadth_delta_smooth5": diffusion,
                "breadth_equal_raw": 0.5 + diffusion,
                "breadth_float_raw": 0.5 + diffusion / 2,
                "concept_amount": 1e8 * (concept_index + 1),
                "membership_churn_5d": 0.05,
                "member_match_coverage": 0.95,
                "rrg_quadrant": ["lagging", "improving", "leading", "weakening"][
                    (index // 12 + concept_index) % 4
                ],
                "eligible_concept": True,
                "forward_excess_3d": label(3),
                "forward_excess_5d": label(5),
                "forward_excess_10d": label(10),
            })
    return pd.DataFrame(rows)


def test_state_residual_walk_forward_uses_mature_labels_and_placebo_preserves_state():
    concepts = build_concept_first_features(residual_concepts())
    rules = StateResidualRules(
        minimum_train_days=45,
        validation_days=10,
        test_days=15,
        embargo_days=11,
        minimum_train_rows=300,
        hgb_max_iter=5,
    )
    scores, coefficients, priors, audit = fit_state_residual_walk_forward(
        concepts, start="2024-01-02", end="2024-07-29", rules=rules,
    )
    assert not scores.empty
    assert not coefficients.empty
    assert not priors.empty
    for horizon in (3, 5, 10):
        assert (
            audit[f"train_label_available_max_{horizon}d"] < audit["test_start"]
        ).all()
    for _, group in scores.groupby(["trade_date", "rrg_quadrant"], observed=True):
        np.testing.assert_allclose(
            np.sort(group["score_R3_within_multihorizon"].to_numpy()),
            np.sort(group["score_R5_within_state_placebo"].to_numpy()),
        )
    assert set(state_residual_coefficient_stability(coefficients)["horizon"]) == {3, 5, 10}


def test_state_residual_overlay_is_fixed_eighty_twenty_and_has_diagnostics():
    concepts = build_concept_first_features(residual_concepts())
    rules = StateResidualRules(
        minimum_train_days=45,
        validation_days=10,
        test_days=15,
        embargo_days=11,
        minimum_train_rows=300,
        hgb_max_iter=5,
    )
    scores, _, _, _ = fit_state_residual_walk_forward(
        concepts, start="2024-01-02", end="2024-07-29", rules=rules,
    )
    codes = [f"C{index:02d}" for index in range(4)]
    etfs = concepts.loc[concepts["concept_code"].isin(codes), [
        "trade_date", "concept_code",
    ]].copy()
    etfs["ts_code"] = etfs["concept_code"]
    etfs["score_etf_momentum"] = etfs.groupby("trade_date").cumcount().astype(float)
    mapped = attach_state_residual_scores_to_etfs(etfs, scores, concept_overlay_weight=0.20)
    oof = mapped.dropna(subset=["fold"])
    expected = 0.8 * oof["price_momentum_z"] + 0.2 * oof["score_R1_within_linear_5d"]
    np.testing.assert_allclose(oof["score_S1_linear_overlay"], expected)
    ic, buckets = within_state_oof_diagnostics(concepts, scores)
    assert set(ic["horizon"]) == {3, 5, 10}
    assert set(buckets["bucket"]) == {1, 2, 3, 4, 5}
