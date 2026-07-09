from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import reliability_feature_columns


@dataclass(frozen=True)
class ReliabilitySplitConfig:
    train_start: str = "2024-01-02"
    train_end: str = "2025-06-30"
    valid_start: str = "2025-07-01"
    valid_end: str = "2025-12-31"
    test_start: str = "2026-01-01"
    test_end: str = "2026-06-30"


def load_reliability_dataset(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    frame = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["date", "factor_name"]).reset_index(drop=True)


def load_feature_list(path: str | Path | None, dataset: pd.DataFrame, max_features: int = 40) -> list[str]:
    if path is None:
        features = reliability_feature_columns(dataset, max_features=max_features)
    else:
        features = pd.read_csv(path)["feature"].dropna().astype(str).tolist()
        features = [col for col in features if col in dataset.columns][:max_features]
    blocked = ("future_", "validity_label_")
    leaked = [col for col in features if col.startswith(blocked)]
    if leaked:
        raise ValueError(f"feature list contains leaked target columns: {leaked}")
    return features


def target_column(horizon: int) -> str:
    return f"future_top_bottom_spread_{horizon}d"


def split_dataset(
    dataset: pd.DataFrame,
    *,
    target: str,
    features: list[str],
    config: ReliabilitySplitConfig | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = config or ReliabilitySplitConfig()
    required = ["date", "factor_name", target, *features]
    missing = [col for col in required if col not in dataset.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    data = dataset[required].replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()
    date = data["date"]
    splits = {
        "train": data.loc[date.between(pd.Timestamp(cfg.train_start), pd.Timestamp(cfg.train_end))].copy(),
        "valid": data.loc[date.between(pd.Timestamp(cfg.valid_start), pd.Timestamp(cfg.valid_end))].copy(),
        "test": data.loc[date.between(pd.Timestamp(cfg.test_start), pd.Timestamp(cfg.test_end))].copy(),
    }
    if splits["train"].empty or splits["valid"].empty or splits["test"].empty:
        return chronological_fallback_split(data)
    return splits


def chronological_fallback_split(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dates = pd.Series(data["date"].drop_duplicates().sort_values().to_numpy())
    if len(dates) < 60:
        raise ValueError("not enough reliability samples for chronological split")
    train_end = dates.iloc[int(len(dates) * 0.6) - 1]
    valid_end = dates.iloc[int(len(dates) * 0.8) - 1]
    return {
        "train": data.loc[data["date"].le(train_end)].copy(),
        "valid": data.loc[data["date"].gt(train_end) & data["date"].le(valid_end)].copy(),
        "test": data.loc[data["date"].gt(valid_end)].copy(),
    }
