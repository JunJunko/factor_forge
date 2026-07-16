import numpy as np
import pandas as pd

from factor_forge.research.concept_rotation_ml import (
    ConceptMLRules,
    _label_available_dates,
    _relevance_labels,
    build_expanding_ml_folds,
    concept_candidate_mask,
)


def test_label_available_date_matches_t_plus_six_open_for_five_day_horizon():
    calendar = pd.bdate_range("2026-01-01", periods=12)
    signal = pd.Series([calendar[0], calendar[4], calendar[6]])
    result = _label_available_dates(signal, calendar, horizon=5)
    assert result.iloc[0] == calendar[6]
    assert result.iloc[1] == calendar[10]
    assert pd.isna(result.iloc[2])


def test_relevance_labels_are_cross_sectional_order_only():
    values = pd.Series([0.05, -0.02, 0.01, 0.03, 0.00], index=list("abcde"))
    labels = _relevance_labels(values)
    assert labels.loc["a"] == 4
    assert labels.loc["b"] == 0
    assert labels.nunique() == 5


def test_expanding_folds_have_horizon_embargo_and_contiguous_tests():
    dates = pd.bdate_range("2025-01-01", periods=140)
    rules = ConceptMLRules(
        minimum_train_days=40, validation_days=10, test_days=10, horizon=5,
    )
    folds = build_expanding_ml_folds(dates, rules=rules)
    assert len(folds) > 2
    for fold in folds:
        train_end = dates.get_loc(fold.train_end)
        valid_start = dates.get_loc(fold.valid_start)
        valid_end = dates.get_loc(fold.valid_end)
        test_start = dates.get_loc(fold.test_start)
        assert valid_start - train_end == rules.horizon + 2
        assert test_start - valid_end == rules.horizon + 2
        assert fold.train_start == dates[0]
    assert all(
        dates.get_loc(right.test_start) - dates.get_loc(left.test_end) == 1
        for left, right in zip(folds, folds[1:])
    )


def test_no_future_fields_are_model_features():
    from factor_forge.research.concept_rotation_ml import FEATURE_COLUMNS

    forbidden = ("forward", "label", "relevance", "future")
    assert not any(any(token in column.lower() for token in forbidden) for column in FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
    assert np.isfinite(0.8 + ConceptMLRules().blend_model_weight)


def test_leading_and_breadth_are_hard_candidate_gates():
    frame = pd.DataFrame({
        "eligible_concept": [True] * 5,
        "rotation_momentum_score": [0.9, 0.8, 0.7, 0.6, 0.5],
        "rrg_quadrant": ["leading", "weakening", "leading", "leading", "leading"],
        "rotation_rs_rank": [0.90, 0.95, 0.80, 0.90, 0.90],
        "breadth_float": [0.60, 0.70, 0.70, 0.40, 0.60],
        "common_delta_rank": [0.80, 0.90, 0.90, 0.90, 0.60],
    })
    assert concept_candidate_mask(frame, "momentum").tolist() == [True] * 5
    assert concept_candidate_mask(frame, "leading").tolist() == [True, False, False, True, True]
    assert concept_candidate_mask(frame, "leading_breadth").tolist() == [True, False, False, False, False]
