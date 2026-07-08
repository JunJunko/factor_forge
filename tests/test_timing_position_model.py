import numpy as np
import pandas as pd

from factor_forge.timing.position_model import (
    map_predictions_to_positions,
    performance_metrics,
    smooth_positions,
)


def test_map_predictions_to_positions_uses_training_thresholds():
    prediction = pd.Series([-0.2, 0.0, 0.4, 0.8, 1.2])
    thresholds = {0.2: -0.1, 0.4: 0.2, 0.6: 0.6, 0.8: 1.0}
    positions = [0.0, 0.25, 0.5, 0.75, 1.0]

    mapped = map_predictions_to_positions(prediction, thresholds, positions)

    assert mapped.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_smooth_positions_limits_daily_change():
    raw = pd.Series([0.0, 1.0, 0.0, 0.75])

    smoothed = smooth_positions(raw, max_daily_change=0.25)

    assert smoothed.tolist() == [0.0, 0.25, 0.0, 0.25]
    assert smoothed.diff().abs().dropna().le(0.25 + 1e-12).all()


def test_performance_metrics_handles_benchmark_and_turnover():
    strategy = pd.Series([0.01, -0.005, 0.002])
    benchmark = pd.Series([0.02, -0.01, 0.001])
    turnover = pd.Series([0.0, 0.25, 0.0])
    position = pd.Series([0.0, 0.25, 0.25])

    metrics = performance_metrics(strategy, benchmark, turnover, position)

    assert metrics["strategy_total_return"] == np.prod(1 + strategy) - 1
    assert metrics["benchmark_total_return"] == np.prod(1 + benchmark) - 1
    assert metrics["average_position"] == position.mean()
    assert metrics["average_daily_turnover"] == turnover.mean()
