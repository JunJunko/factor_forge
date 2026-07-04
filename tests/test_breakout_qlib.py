from __future__ import annotations

import numpy as np
import pandas as pd

from factor_forge.ml.breakout_qlib import (
    BreakoutQlibConfig,
    BreakoutQlibRunner,
    DailyEqualWeight,
)


def test_daily_reweighter_gives_each_date_equal_total_weight():
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "A"),
        ],
        names=["datetime", "instrument"],
    )
    weights = DailyEqualWeight().reweight(pd.DataFrame(index=index))
    totals = weights.groupby(level="datetime").sum()
    assert totals.iloc[0] == totals.iloc[1]
    assert weights.mean() == 1.0


def test_event_frame_has_native_qlib_shape():
    events = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ts_code": ["A", "B"],
            "feature_a": [1.0, 2.0],
            "label_absolute": [0.01, -0.01],
        }
    )
    frame = BreakoutQlibRunner._to_qlib_frame(
        events, ["feature_a"], "label_absolute"
    )
    assert frame.index.names == ["datetime", "instrument"]
    assert list(frame.columns) == [("feature", "feature_a"), ("label", "LABEL0")]


def test_breakout_qlib_segments_are_strict_and_targets_unique():
    payload = {
        "research_run": "artifacts/example",
        "segments": {
            "train": {"start": "2021-01-01", "end": "2022-12-31"},
            "valid": {"start": "2023-01-01", "end": "2023-12-31"},
            "test": {"start": "2024-01-01", "end": "2024-12-31"},
        },
    }
    config = BreakoutQlibConfig.model_validate(payload)
    assert config.targets == ["absolute", "event_excess", "cost_positive"]
