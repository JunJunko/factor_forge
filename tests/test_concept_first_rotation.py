import numpy as np
import pandas as pd

from factor_forge.research.concept_first_rotation import (
    CONCEPT_FEATURES,
    ConceptFirstRules,
    attach_concept_scores_to_etfs,
    build_concept_first_features,
    coefficient_stability,
    concept_oof_diagnostics,
    fit_concept_first_walk_forward,
)


def concept_panel(concepts: int = 30, periods: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=periods)
    rows = []
    market_returns = 0.0005 * np.sin(np.arange(periods) / 11)
    market_prices = np.cumprod(1 + market_returns)
    for concept_index in range(concepts):
        concept_returns = market_returns + 0.001 * np.sin(
            (np.arange(periods) + concept_index) / 9
        )
        prices = np.cumprod(1 + concept_returns)
        for date_index, date in enumerate(dates):
            def trailing_return(values, horizon):
                if date_index < horizon:
                    return np.nan
                return values[date_index] / values[date_index - horizon] - 1

            def forward_excess(horizon):
                if date_index + horizon + 1 >= periods:
                    return np.nan
                concept_forward = prices[date_index + horizon + 1] / prices[date_index + 1] - 1
                market_forward = market_prices[date_index + horizon + 1] / market_prices[date_index + 1] - 1
                return concept_forward - market_forward

            diffusion = np.sin((date_index + concept_index) / 13) / 10
            rows.append({
                "trade_date": date,
                "concept_code": f"C{concept_index:03d}",
                "concept_return_1d": concept_returns[date_index],
                "concept_return_5d": trailing_return(prices, 5),
                "concept_return_20d": trailing_return(prices, 20),
                "concept_return_60d": trailing_return(prices, 60),
                "market_return_5d": trailing_return(market_prices, 5),
                "market_return_20d": trailing_return(market_prices, 20),
                "market_return_60d": trailing_return(market_prices, 60),
                "rs_momentum_5d": diffusion / 2,
                "common_breadth_delta_smooth5": diffusion,
                "breadth_equal_raw": 0.5 + diffusion,
                "breadth_float_raw": 0.5 + diffusion / 2,
                "concept_amount": 1e8 * (1 + concept_index),
                "membership_churn_5d": 0.05,
                "member_match_coverage": 0.95,
                "rrg_quadrant": ["lagging", "improving", "leading", "weakening"][
                    (date_index // 15 + concept_index) % 4
                ],
                "eligible_concept": True,
                "forward_excess_3d": forward_excess(3),
                "forward_excess_5d": forward_excess(5),
                "forward_excess_10d": forward_excess(10),
            })
    return pd.DataFrame(rows)


def test_concept_features_are_causal():
    panel = concept_panel(concepts=8, periods=80)
    base = build_concept_first_features(panel)
    mutated = panel.copy()
    final_date = mutated["trade_date"].max()
    mutated.loc[mutated["trade_date"].eq(final_date), "concept_amount"] *= 1000
    mutated.loc[
        mutated["trade_date"].eq(final_date), "common_breadth_delta_smooth5"
    ] = 99
    changed = build_concept_first_features(mutated)
    cutoff = sorted(panel["trade_date"].unique())[-2]
    pd.testing.assert_frame_equal(
        base.loc[base["trade_date"].le(cutoff), CONCEPT_FEATURES].reset_index(drop=True),
        changed.loc[changed["trade_date"].le(cutoff), CONCEPT_FEATURES].reset_index(drop=True),
    )


def test_concept_walk_forward_uses_mature_labels_and_preserves_placebo_state_distribution():
    features = build_concept_first_features(concept_panel())
    rules = ConceptFirstRules(
        minimum_train_days=60,
        validation_days=10,
        test_days=15,
        embargo_days=11,
        minimum_train_rows=1_000,
        hgb_max_iter=10,
    )
    scores, coefficients, audit = fit_concept_first_walk_forward(
        features,
        start="2024-01-02",
        end="2024-09-09",
        rules=rules,
    )
    assert not scores.empty
    assert not coefficients.empty
    assert not audit.empty
    for horizon in (3, 5, 10):
        assert (
            audit[f"train_label_available_max_{horizon}d"] < audit["test_start"]
        ).all()
    grouped = scores.groupby(["trade_date", "rrg_quadrant"], observed=True)
    for _, group in grouped:
        np.testing.assert_allclose(
            np.sort(group["score_C3_multihorizon"].to_numpy()),
            np.sort(group["score_C5_state_placebo"].to_numpy()),
        )
    stability = coefficient_stability(coefficients)
    assert set(stability["horizon"]) == {3, 5, 10}


def test_scores_map_to_etfs_and_concept_diagnostics_cover_all_horizons():
    concepts = build_concept_first_features(concept_panel())
    rules = ConceptFirstRules(
        minimum_train_days=60,
        validation_days=10,
        test_days=15,
        embargo_days=11,
        minimum_train_rows=1_000,
        hgb_max_iter=10,
    )
    scores, _, _ = fit_concept_first_walk_forward(
        concepts, start="2024-01-02", end="2024-09-09", rules=rules,
    )
    codes = [f"C{index:03d}" for index in range(5)]
    etfs = concepts.loc[concepts["concept_code"].isin(codes), [
        "trade_date", "concept_code", "concept_return_1d", "eligible_concept",
    ]].copy()
    etfs["ts_code"] = etfs["concept_code"].map({
        code: f"ETF{index}" for index, code in enumerate(codes)
    })
    etfs["etf_return_1d"] = etfs["concept_return_1d"] + 0.0001
    etfs["amount_cny"] = 1e8
    etfs["aum_cny"] = 2e9
    etfs["score_etf_momentum"] = 1.0
    mapped = attach_concept_scores_to_etfs(etfs, scores)
    oof = mapped.loc[mapped["fold"].notna()]
    assert not oof.empty
    assert oof["score_C4_mapping_quality"].notna().all()
    ic, buckets = concept_oof_diagnostics(
        concepts,
        scores,
        policies={
            "C1": "score_C1_linear_5d",
            "C3": "score_C3_multihorizon",
        },
    )
    assert set(ic["horizon"]) == {3, 5, 10}
    assert set(buckets["bucket"]) == {1, 2, 3, 4, 5}
