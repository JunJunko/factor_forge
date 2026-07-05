import numpy as np
import pandas as pd
import pytest

from factor_forge.ml.value_diagnostics import (
    ValueDiagnosticsConfig,
    ValueDiagnosticsRunner,
)


def test_diagnostics_grid_requires_unique_positive_values(tmp_path):
    base = {
        "experiment_config": tmp_path / "experiment.yaml",
        "full_run_dir": tmp_path / "run",
    }
    with pytest.raises(ValueError):
        ValueDiagnosticsConfig.model_validate({**base, "top_n": [5, 5]})
    with pytest.raises(ValueError):
        ValueDiagnosticsConfig.model_validate({**base, "holding_days": [0, 5]})


def test_decile_curve_uses_equal_weighted_daily_cross_sections():
    rows = []
    for date in pd.to_datetime(["2024-01-02", "2024-01-03"]):
        for stock in range(20):
            score = stock / 19
            rows.append({
                "trade_date": date, "ts_code": str(stock),
                "full_prediction_blend": score,
                "price_prediction_blend": score,
                "label_5d": score - 0.5,
            })
    common = pd.DataFrame(rows)
    result = ValueDiagnosticsRunner._decile_returns(
        common,
        {"full": "full_prediction_blend", "price": "price_prediction_blend"},
        [5],
        10,
    )
    full = result.loc[result["model"].eq("full")].set_index("decile")
    assert len(full) == 10
    assert full.loc[10, "mean_excess_return"] > full.loc[1, "mean_excess_return"]
    assert np.isfinite(full["standard_error"]).all()
