from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from collections import deque
from collections.abc import Sequence

import numpy as np
import pandas as pd


def pit_rolling_percentile(
    frame: pd.DataFrame,
    value_column: str,
    *,
    entity_column: str = "ts_code",
    date_column: str = "trade_date",
    window: int,
    min_periods: int,
    output_column: str | None = None,
) -> pd.Series:
    """Mid-rank percentile against strictly prior observations for each entity.

    The current value is compared with at most ``window`` prior rows and is only
    inserted after its percentile has been computed. Missing rows consume a row
    in the rolling window but do not count toward ``min_periods``.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    if not 1 <= min_periods <= window:
        raise ValueError("min_periods must satisfy 1 <= min_periods <= window")
    required = {value_column, entity_column, date_column}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"missing percentile columns: {sorted(missing)}")
    if frame.duplicated([entity_column, date_column]).any():
        raise ValueError("PIT percentile requires unique entity/date rows")

    ordered = frame[[entity_column, date_column, value_column]].copy()
    ordered["__original_index"] = np.arange(len(frame))
    ordered[date_column] = pd.to_datetime(ordered[date_column])
    ordered = ordered.sort_values([entity_column, date_column, "__original_index"], kind="mergesort")
    output = np.full(len(frame), np.nan, dtype=float)
    for _, group in ordered.groupby(entity_column, sort=False, observed=True):
        history: deque[float] = deque()
        sorted_valid: list[float] = []
        values = pd.to_numeric(group[value_column], errors="coerce").to_numpy(dtype=float)
        locations = group["__original_index"].to_numpy(dtype=int)
        for value, location in zip(values, locations, strict=True):
            if np.isfinite(value) and len(sorted_valid) >= min_periods:
                left = bisect_left(sorted_valid, value)
                right = bisect_right(sorted_valid, value)
                output[location] = (left + 0.5 * (right - left)) / len(sorted_valid)
            history.append(value)
            if np.isfinite(value):
                insort(sorted_valid, float(value))
            if len(history) > window:
                expired = history.popleft()
                if np.isfinite(expired):
                    index = bisect_left(sorted_valid, expired)
                    sorted_valid.pop(index)
    return pd.Series(output, index=frame.index, name=output_column or f"{value_column}_pit_pct")


def temporal_prefix_audit(
    frame: pd.DataFrame,
    value_column: str,
    *,
    entity_column: str,
    date_column: str,
    window: int,
    min_periods: int,
    checkpoints: Sequence[float] = (0.55, 0.8),
) -> bool:
    """Verify that appending future rows never changes earlier percentiles."""
    if frame.empty:
        return True
    dates = pd.Series(pd.to_datetime(frame[date_column]).dropna().unique()).sort_values().tolist()
    if len(dates) < 3:
        return True
    full = pit_rolling_percentile(
        frame, value_column, entity_column=entity_column, date_column=date_column,
        window=window, min_periods=min_periods,
    )
    frame_dates = pd.to_datetime(frame[date_column])
    for fraction in checkpoints:
        cutoff = dates[min(len(dates) - 2, max(1, int(len(dates) * fraction)))]
        mask = frame_dates <= cutoff
        prefix = frame.loc[mask]
        prefix_result = pit_rolling_percentile(
            prefix, value_column, entity_column=entity_column, date_column=date_column,
            window=window, min_periods=min_periods,
        )
        expected = full.loc[mask]
        if not np.allclose(prefix_result.to_numpy(), expected.to_numpy(), equal_nan=True):
            return False
    return True
