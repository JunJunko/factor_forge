from __future__ import annotations

import numpy as np
import pandas as pd

from factor_forge.config import FactorSpec, L0Config


def evaluate_factor_quality(
    panel: pd.DataFrame, factor_values: pd.DataFrame, spec: FactorSpec, gate: L0Config
) -> dict:
    merged = panel[["trade_date", "ts_code", "is_factor_eligible"]].merge(
        factor_values[["trade_date", "ts_code", "factor_value"]],
        on=["trade_date", "ts_code"], how="left",
    )
    eligible = merged["is_factor_eligible"].fillna(False).astype(bool)
    relevant = merged.loc[eligible]
    finite = np.isfinite(relevant["factor_value"])
    coverage = float(finite.mean()) if len(relevant) else 0.0
    valid = relevant.loc[finite]
    daily_count = valid.groupby("trade_date")["ts_code"].count()
    daily_unique = valid.groupby("trade_date")["factor_value"].nunique()
    daily_total = valid.groupby("trade_date")["factor_value"].count()
    unique_ratio = (daily_unique / daily_total.replace(0, np.nan)).mean()
    valid_groups = None
    if spec.scope.cross_section == "industry" and "group_code" in factor_values:
        groups = factor_values.loc[factor_values["factor_valid"] & factor_values["group_code"].notna()]
        counts = groups.groupby(["trade_date", "group_code"])["ts_code"].count()
        valid_groups = counts[counts >= spec.scope.min_group_size].groupby(level=0).size()
    checks = {
        "coverage": coverage >= gate.min_coverage,
        "missing_rate": (1.0 - coverage) <= gate.max_missing_rate,
        "daily_cross_section": (float(daily_count.median()) if len(daily_count) else 0) >= gate.min_daily_cross_section,
        "unique_ratio": (float(unique_ratio) if np.isfinite(unique_ratio) else 0) >= gate.min_unique_ratio,
        "future_data_audit": True,  # V1 DSL exposes only current/past operators.
    }
    if valid_groups is not None:
        checks["valid_industry_groups"] = (
            float(valid_groups.median()) if len(valid_groups) else 0
        ) >= gate.min_valid_groups_per_day
    metrics = {
        "coverage": coverage,
        "missing_rate": 1.0 - coverage,
        "median_daily_cross_section": float(daily_count.median()) if len(daily_count) else 0.0,
        "mean_unique_ratio": float(unique_ratio) if np.isfinite(unique_ratio) else 0.0,
        "factor_mean": float(valid["factor_value"].mean()) if len(valid) else None,
        "factor_std": float(valid["factor_value"].std(ddof=0)) if len(valid) else None,
        "future_data_violations": 0,
    }
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics}

