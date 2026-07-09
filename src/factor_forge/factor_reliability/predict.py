from __future__ import annotations

import numpy as np
import pandas as pd


def build_reliability_scores(
    predictions: pd.DataFrame,
    *,
    model_name: str = "ridge",
    calibration_samples: tuple[str, ...] = ("train", "valid"),
) -> pd.DataFrame:
    frame = predictions.loc[predictions["model"].eq(model_name)].copy()
    if frame.empty:
        raise ValueError(f"no predictions for model={model_name}")
    output_parts = []
    for horizon, group in frame.groupby("horizon", sort=True):
        calibration = group.loc[group["sample"].isin(calibration_samples), "predicted_spread"].dropna().sort_values()
        if calibration.empty:
            calibration = group["predicted_spread"].dropna().sort_values()
        g = group.copy()
        g[f"predicted_spread_{int(horizon)}d"] = g["predicted_spread"]
        g[f"reliability_{int(horizon)}d"] = g["predicted_spread"].map(lambda value: _percentile(value, calibration))
        output_parts.append(
            g[
                [
                    "date",
                    "factor_name",
                    "sample",
                    f"predicted_spread_{int(horizon)}d",
                    f"reliability_{int(horizon)}d",
                ]
            ]
        )
    out = output_parts[0]
    for part in output_parts[1:]:
        out = out.merge(part, on=["date", "factor_name", "sample"], how="outer")
    ordered = [
        "date",
        "factor_name",
        "sample",
        "reliability_5d",
        "reliability_10d",
        "reliability_20d",
        "predicted_spread_5d",
        "predicted_spread_10d",
        "predicted_spread_20d",
    ]
    return out[[col for col in ordered if col in out.columns]].sort_values(["date", "factor_name"]).reset_index(drop=True)


def _percentile(value: float, calibration: pd.Series) -> float:
    if pd.isna(value) or calibration.empty:
        return np.nan
    return float(np.searchsorted(calibration.to_numpy(), value, side="right") / len(calibration))
