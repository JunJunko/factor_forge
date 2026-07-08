import numpy as np
import pandas as pd

from factor_forge.timing.regime import (
    _align_state_order,
    _hmm_filtered_probabilities,
    TimingRegimeConfig,
    TimingRegimeGridConfig,
)
from factor_forge.timing.stable_factors import StableFactorSelectionConfig


class _ToyHMM:
    n_components = 3
    startprob_ = np.array([0.5, 0.3, 0.2])
    transmat_ = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    means_ = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    _covars_ = np.ones((3, 2))
    covars_ = _covars_


def test_hmm_filtered_probabilities_are_causal_and_normalized():
    prefix = np.array([[-0.5, 0.0], [0.2, 0.1], [0.7, -0.1]])
    future = np.array([[-4.0, 3.0], [5.0, -2.0]])
    short = _hmm_filtered_probabilities(_ToyHMM(), prefix)
    long = _hmm_filtered_probabilities(_ToyHMM(), np.vstack([prefix, future]))

    np.testing.assert_allclose(short.sum(axis=1), 1.0)
    np.testing.assert_allclose(short, long[: len(prefix)])


def test_state_alignment_matches_reference_order():
    reference = np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    shuffled = np.array([[2.1, 0.0], [-2.1, 0.0], [0.1, 0.0]])
    order = _align_state_order(shuffled, reference)
    np.testing.assert_array_equal(order, np.array([1, 2, 0]))


def test_timing_regime_config_accepts_gmm_and_strict_split():
    cfg = TimingRegimeConfig.model_validate({
        "dataset_path": "dummy.parquet",
        "regime": {"method": "gmm"},
        "interaction_model": {"train_end": "2024-12-31", "test_start": "2025-01-01"},
    })

    assert cfg.regime.method == "gmm"


def test_timing_regime_grid_config_has_10d_label_and_grid():
    cfg = TimingRegimeGridConfig.model_validate({
        "dataset_path": "dummy.parquet",
        "methods": ["hmm", "gmm"],
        "n_components_grid": [2, 3],
        "history_days_grid": [126],
        "random_states": [1, 2],
    })

    assert cfg.label_column == "label_10d_excess_return"
    assert cfg.n_components_grid == [2, 3]


def test_stable_factor_selection_config_defaults_to_fixed_hmm_3state():
    cfg = StableFactorSelectionConfig.model_validate({
        "dataset_path": "dummy.parquet",
    })

    assert cfg.regime_method == "hmm"
    assert cfg.n_components == 3
    assert cfg.history_days == 252
    assert cfg.label_column == "label_10d_excess_return"
