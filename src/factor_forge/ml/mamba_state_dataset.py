from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class SequenceStore:
    """Compact flat storage with on-demand [length, features] sequence slicing."""

    frame: pd.DataFrame
    feature_names: list[str]
    length: int
    samples: pd.DataFrame
    values: np.ndarray
    valid: np.ndarray

    def take(self, sample_positions: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(list(sample_positions), dtype=np.int64)
        x = np.empty((len(positions), self.length, len(self.feature_names)), dtype=np.float32)
        mask = np.empty_like(x, dtype=np.float32)
        for output_index, sample_position in enumerate(positions):
            sample = self.samples.iloc[int(sample_position)]
            start, end = int(sample["start_row"]), int(sample["end_row"]) + 1
            x[output_index] = self.values[start:end]
            mask[output_index] = self.valid[start:end]
        return x, mask

    def positions_between(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> np.ndarray:
        dates = pd.to_datetime(self.samples["datetime"])
        return np.flatnonzero(dates.between(pd.Timestamp(start), pd.Timestamp(end)).to_numpy())


def build_sequence_store(
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    length: int,
    min_valid_days: int,
    validity_feature_names: list[str] | None = None,
) -> SequenceStore:
    required = {"datetime", "instrument", *feature_names}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"state sequence frame missing columns: {sorted(missing)}")
    if min_valid_days > length:
        raise ValueError("min_valid_days cannot exceed length")
    data = frame.copy()
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["instrument"] = data["instrument"].astype(str)
    if data.duplicated(["datetime", "instrument"]).any():
        raise ValueError("state sequence frame has duplicate datetime/instrument rows")
    data = data.sort_values(["instrument", "datetime"], kind="mergesort").reset_index(drop=True)
    numeric = data[feature_names].apply(pd.to_numeric, errors="coerce")
    raw = numeric.to_numpy(dtype=np.float32, copy=True)
    valid = np.isfinite(raw).astype(np.float32)
    values = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    calendar = pd.Index(sorted(data["datetime"].unique()))
    ordinal_map = pd.Series(np.arange(len(calendar), dtype=np.int64), index=calendar)
    ordinals = data["datetime"].map(ordinal_map).to_numpy(dtype=np.int64)
    validity_names = validity_feature_names or feature_names
    unknown = set(validity_names) - set(feature_names)
    if unknown:
        raise KeyError(f"unknown validity features: {sorted(unknown)}")
    validity_positions = [feature_names.index(name) for name in validity_names]
    any_valid = valid[:, validity_positions].any(axis=1).astype(np.int16)
    rows: list[dict] = []
    for instrument, group in data.groupby("instrument", sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        if len(indices) < length:
            continue
        rolling_valid = pd.Series(any_valid[indices]).rolling(length).sum().to_numpy()
        for local_end in range(length - 1, len(indices)):
            start_row, end_row = indices[local_end - length + 1], indices[local_end]
            if ordinals[end_row] - ordinals[start_row] != length - 1:
                continue
            if rolling_valid[local_end] < min_valid_days:
                continue
            source = data.iloc[end_row]
            row = {
                "datetime": source["datetime"],
                "instrument": instrument,
                "start_row": int(start_row),
                "end_row": int(end_row),
                "valid_days": int(rolling_valid[local_end]),
            }
            for column in ("label", "is_tradeable", "is_liquid"):
                if column in data:
                    row[column] = source[column]
            rows.append(row)
    samples = pd.DataFrame(rows)
    if samples.empty:
        samples = pd.DataFrame(columns=[
            "datetime", "instrument", "start_row", "end_row", "valid_days",
            "label", "is_tradeable", "is_liquid",
        ])
    return SequenceStore(data, list(feature_names), length, samples, values, valid)


def sequence_prefix_audit(
    original: SequenceStore,
    extended: SequenceStore,
    *,
    atol: float = 1e-7,
) -> bool:
    """Verify that appending future rows cannot alter existing sequence samples."""
    keys = original.samples[["datetime", "instrument"]].copy()
    lookup = extended.samples.reset_index().set_index(["datetime", "instrument"])["index"]
    for original_position, key in enumerate(keys.itertuples(index=False)):
        compound = (pd.Timestamp(key.datetime), str(key.instrument))
        if compound not in lookup.index:
            return False
        extended_position = int(lookup.loc[compound])
        left_x, left_mask = original.take([original_position])
        right_x, right_mask = extended.take([extended_position])
        if not np.allclose(left_x, right_x, atol=atol, rtol=0):
            return False
        if not np.array_equal(left_mask, right_mask):
            return False
    return True
