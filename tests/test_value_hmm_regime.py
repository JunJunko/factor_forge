from __future__ import annotations

import numpy as np

from factor_forge.ml.value_hmm_regime import ValueHMMRegimeRunner


class _ToyHMM:
    n_components = 3
    startprob_ = np.array([0.5, 0.3, 0.2])
    transmat_ = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    means_ = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    _covars_ = np.ones((3, 2))
    covars_ = _covars_


def test_filtered_regime_probabilities_are_causal_and_normalized():
    prefix = np.array([[-0.5, 0.0], [0.2, 0.1], [0.7, -0.1]])
    future = np.array([[-4.0, 3.0], [5.0, -2.0]])
    short = ValueHMMRegimeRunner._filtered(_ToyHMM(), prefix)
    long = ValueHMMRegimeRunner._filtered(_ToyHMM(), np.vstack([prefix, future]))
    np.testing.assert_allclose(short.sum(axis=1), 1.0)
    np.testing.assert_allclose(short, long[: len(prefix)])
