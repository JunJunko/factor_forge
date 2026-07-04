import numpy as np
import pandas as pd

from factor_forge.ml.config import FeatureConfig, LabelConfig, MLExperimentConfig
from factor_forge.ml.dataset import build_dataset, to_qlib_frame


def test_label_uses_next_open_to_horizon_open_and_qlib_shape():
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    panel = pd.DataFrame({"trade_date": dates, "ts_code": "000001.SZ", "adj_open": np.arange(10.0, 18.0), "adj_close": np.arange(10.0, 18.0), "amount_cny": 1e8, "log_total_mv": 10.0, "log_circ_mv": 9.0, "turnover_rate": 1.0})
    data, names = build_dataset(panel, FeatureConfig(windows=[2], winsor_quantile=0, cross_sectional_zscore=False), LabelConfig(horizon=2, excess_over_universe=False))
    assert data.iloc[0]["label"] == 13 / 11 - 1
    qlib = to_qlib_frame(data, names)
    assert qlib.index.names == ["datetime", "instrument"]
    assert ("label", "LABEL0") in qlib.columns


def test_segments_must_be_strictly_chronological():
    import pytest
    with pytest.raises(ValueError):
        MLExperimentConfig.model_validate({"name": "bad", "segments": {"train": {"start": "2020-01-01", "end": "2021-01-01"}, "valid": {"start": "2020-06-01", "end": "2021-06-01"}, "test": {"start": "2022-01-01", "end": "2022-12-31"}}})


def test_full_segment_coverage_is_required_by_default():
    cfg = MLExperimentConfig.model_validate({"name": "strict", "segments": {"train": {"start": "2016-01-01", "end": "2022-12-31"}, "valid": {"start": "2023-01-01", "end": "2023-12-31"}, "test": {"start": "2024-01-01", "end": "2026-06-30"}}})
    assert cfg.require_full_segment_coverage is True
