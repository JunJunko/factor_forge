from __future__ import annotations

import numpy as np
import pandas as pd


def build_condition_membership(
    panel: pd.DataFrame,
    conditioning_factor_values: pd.DataFrame,
    *,
    universe: str,
    quantile_groups: int,
    include_quantiles: list[int],
    min_cross_section: int,
) -> pd.DataFrame:
    """Build point-in-time daily condition buckets inside a backtest universe."""
    universe_column = f"is_{universe}"
    keys = ["trade_date", "ts_code"]
    frame = panel[keys + [universe_column]].merge(
        conditioning_factor_values[keys + ["factor_value"]].rename(
            columns={"factor_value": "condition_value"}
        ),
        on=keys,
        how="left",
    )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    eligible = frame[universe_column].fillna(False).astype(bool) & frame["condition_value"].notna()
    valid = frame.loc[eligible].copy()
    daily_count = valid.groupby("trade_date")["ts_code"].transform("size")
    percentile = valid.groupby("trade_date")["condition_value"].rank(method="average", pct=True)
    valid["condition_quantile"] = np.ceil(percentile * quantile_groups).clip(
        1, quantile_groups
    ).astype("int16")
    valid["selection_eligible"] = (
        (daily_count >= min_cross_section)
        & valid["condition_quantile"].isin(include_quantiles)
    )
    return valid[
        keys + ["condition_value", "condition_quantile", "selection_eligible"]
    ].reset_index(drop=True)
