import numpy as np
import pandas as pd

from factor_forge.research.concept_first_rotation import CONCEPT_FEATURES
from factor_forge.research.concept_state_residual_rotation import (
    StateResidualRules,
    fit_frozen_s2_model,
    score_frozen_s2_model,
    s2_fold_train_test_diagnostics,
)


def _concepts():
    dates = pd.bdate_range("2024-01-01", periods=40)
    rows = []
    for date_position, date in enumerate(dates):
        for concept_position in range(12):
            row = {
                "trade_date": date,
                "concept_code": f"C{concept_position}",
                "rrg_quadrant": ["leading", "improving"][concept_position % 2],
                "eligible_concept": True,
                "forward_excess_5d": (concept_position - 3.5) / 100 + date_position / 10000,
            }
            row.update({feature: (concept_position + date_position) / 100 for feature in CONCEPT_FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)


def test_frozen_s2_uses_only_mature_labels_and_scores_within_state():
    concepts = _concepts()
    rules = StateResidualRules(minimum_train_rows=100, minimum_train_days=10)
    model, audit = fit_frozen_s2_model(
        concepts, training_cutoff=str(concepts["trade_date"].max().date()), rules=rules,
    )
    assert pd.Timestamp(audit["label_available_max"]) <= concepts["trade_date"].max()
    scores = score_frozen_s2_model(concepts.tail(16), model)
    grouped_mean = scores.groupby(["trade_date", "rrg_quadrant"])[
        "score_R2_within_nonlinear_5d"
    ].mean()
    assert np.allclose(grouped_mean, 0.0, atol=1e-10)


def test_s2_fold_diagnostics_separates_train_and_oof():
    concepts = _concepts()
    rules = StateResidualRules(
        minimum_train_rows=100, minimum_train_days=10,
        validation_days=5, test_days=5, embargo_days=2,
    )
    diagnostics = s2_fold_train_test_diagnostics(
        concepts, start=str(concepts["trade_date"].min().date()),
        end=str(concepts["trade_date"].max().date()), rules=rules,
    )
    assert set(diagnostics["sample"]) == {"train_in_sample", "test_oof"}
    assert diagnostics.loc[
        diagnostics["sample"].eq("train_in_sample"), "state_date_groups"
    ].gt(0).all()
    assert diagnostics.loc[diagnostics["sample"].eq("test_oof"), "rows"].gt(0).all()
