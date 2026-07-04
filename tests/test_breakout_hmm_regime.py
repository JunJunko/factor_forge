from types import SimpleNamespace

import numpy as np

from factor_forge.ml.breakout_hmm_regime import BreakoutHMMRegimeRunner


def test_online_filter_is_normalized_and_does_not_use_future_observations():
    model = SimpleNamespace(
        n_components=2,
        startprob_=np.array([0.6, 0.4]),
        transmat_=np.array([[0.9, 0.1], [0.2, 0.8]]),
        means_=np.array([[0.0, 0.0], [2.0, 2.0]]),
        covars_=np.ones((2, 2)),
    )
    observations = np.array([[0.1, -0.1], [0.2, 0.0], [1.8, 2.1], [2.2, 1.9]])
    runner = BreakoutHMMRegimeRunner()
    prefix = runner._filtered_probabilities(model, observations[:3])
    full = runner._filtered_probabilities(model, observations)
    np.testing.assert_allclose(prefix, full[:3])
    np.testing.assert_allclose(full.sum(axis=1), 1.0)
