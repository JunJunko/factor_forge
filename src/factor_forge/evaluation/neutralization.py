from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize(
    frame: pd.DataFrame,
    include_industry: bool,
    include_size: bool = True,
) -> pd.Series:
    """Daily OLS residualization; missing exposures remain missing."""
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, indexes in frame.groupby("trade_date").groups.items():
        group = frame.loc[indexes]
        columns: list[pd.Series] = [pd.Series(1.0, index=group.index, name="intercept")]
        if include_size:
            columns.append(group["log_total_mv"].rename("size"))
        if include_industry:
            dummies = pd.get_dummies(group["industry_l1_code"], prefix="industry", dtype=float)
            if dummies.shape[1] > 1:
                columns.extend([dummies[col] for col in dummies.columns[1:]])
        design = pd.concat(columns, axis=1)
        valid = group["factor_value"].notna() & design.notna().all(axis=1)
        if valid.sum() <= design.shape[1]:
            continue
        x = design.loc[valid].to_numpy(dtype=float)
        y = group.loc[valid, "factor_value"].to_numpy(dtype=float)
        coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        output.loc[group.index[valid]] = y - x @ coefficients
    return output


def build_variants(merged: pd.DataFrame, cross_section: str) -> dict[str, pd.Series]:
    variants = {"raw": merged["factor_value"]}
    if merged["log_total_mv"].notna().any():
        if cross_section == "industry":
            variants["size_neutralized"] = neutralize(merged, include_industry=False)
        elif merged["industry_l1_code"].notna().any():
            variants["industry_size_neutralized"] = neutralize(merged, include_industry=True)
    return variants

