import numpy as np
import pandas as pd

from factor_forge.research.absorption_continuation import (
    AbsorptionFeatureConfig,
    _future_extreme,
    _slope,
)


def test_future_extreme_excludes_the_signal_day_and_uses_only_next_horizon():
    value = pd.Series([10.0, 99.0, 12.0, 13.0, 14.0])
    codes = pd.Series(["A"] * len(value))
    future_max = _future_extreme(value, codes, horizon=2, method="max")
    assert future_max.iloc[0] == 99.0
    assert future_max.iloc[1] == 13.0
    assert np.isnan(future_max.iloc[3])


def test_slope_is_ols_slope_over_the_full_window_not_a_one_day_change():
    value = pd.Series([1.0, 1.0, 4.0, 4.0, 4.0])
    codes = pd.Series(["A"] * len(value))
    slopes = _slope(value, codes, window=3)
    assert np.isnan(slopes.iloc[1])
    assert slopes.iloc[2] == 1.5
    assert slopes.iloc[4] == 0.0
