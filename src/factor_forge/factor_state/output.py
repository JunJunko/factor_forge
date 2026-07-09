from __future__ import annotations

from pathlib import Path

import pandas as pd

from .label import STATE_NAMES, FactorState


def build_factor_state_output(
    predictions: pd.DataFrame,
    *,
    model_name: str | None = None,
    sample: str | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build the daily state-probability interface for dynamic factor allocation."""
    frame = predictions.copy()
    if model_name is not None:
        frame = frame.loc[frame["model"].eq(model_name)].copy()
    if sample is not None:
        frame = frame.loc[frame["sample"].eq(sample)].copy()
    if frame.empty:
        raise ValueError("no predictions available for requested model/sample")
    prob_cols = [f"p_{state.name.lower()}" for state in FactorState]
    for col in prob_cols:
        if col not in frame.columns:
            frame[col] = 0.0
    out = frame[
        [
            "date",
            "factor_name",
            "p_healthy",
            "p_weakening",
            "p_broken",
            "p_recovery",
            "predicted_state_name",
        ]
    ].rename(
        columns={
            "factor_name": "factor",
            "p_healthy": "healthy_probability",
            "p_weakening": "weakening_probability",
            "p_broken": "broken_probability",
            "p_recovery": "recovery_probability",
            "predicted_state_name": "state",
        }
    )
    out["state_id"] = out["state"].map({name: state for state, name in STATE_NAMES.items()})
    out = out.sort_values(["date", "factor"]).reset_index(drop=True)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False, encoding="utf-8-sig")
    return out
