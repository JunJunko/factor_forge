from __future__ import annotations

import pandas as pd

from .l1 import _newey_west_stats


def compare_daily_rank_ic(
    market_daily: pd.DataFrame,
    industry_daily: pd.DataFrame,
    *,
    horizon: int,
    min_overlap_days: int = 60,
    min_retention_ratio: float = 0.50,
    max_p_value: float = 0.10,
) -> dict:
    """Compare aligned daily IC series for the composition-effect discriminator."""
    required = {"trade_date", "rank_ic"}
    for name, frame in (("market", market_daily), ("industry", industry_daily)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} daily IC is missing columns: {sorted(missing)}")
        if frame["trade_date"].duplicated().any():
            raise ValueError(f"{name} daily IC contains duplicate trade_date rows")

    aligned = (
        market_daily[["trade_date", "rank_ic"]]
        .rename(columns={"rank_ic": "market_rank_ic"})
        .merge(
            industry_daily[["trade_date", "rank_ic"]].rename(
                columns={"rank_ic": "industry_rank_ic"}
            ),
            on="trade_date",
            how="inner",
            validate="one_to_one",
        )
        .dropna()
        .sort_values("trade_date")
    )
    overlap_days = int(len(aligned))
    market_mean = float(aligned["market_rank_ic"].mean()) if overlap_days else None
    industry_mean = float(aligned["industry_rank_ic"].mean()) if overlap_days else None
    retention_ratio = (
        industry_mean / market_mean
        if market_mean is not None and industry_mean is not None and market_mean > 0
        else None
    )
    delta = aligned["industry_rank_ic"] - aligned["market_rank_ic"]
    delta_mean = float(delta.mean()) if overlap_days else None
    inference = _newey_west_stats(delta, max_lags=max(horizon - 1, 0))
    p_value = inference["nw_p_value"]

    if overlap_days < min_overlap_days:
        classification = "insufficient_overlap"
    elif market_mean is None or market_mean <= 0:
        classification = "market_gate_not_supported"
    elif (
        retention_ratio is not None
        and retention_ratio < min_retention_ratio
        and delta_mean is not None
        and delta_mean < 0
        and p_value is not None
        and p_value <= max_p_value
    ):
        classification = "supports_cross_section_composition"
    elif retention_ratio is not None and retention_ratio >= min_retention_ratio:
        classification = "weakens_cross_section_composition"
    else:
        classification = "inconclusive"

    return {
        "classification": classification,
        "overlap_days": overlap_days,
        "market_mean_rank_ic": market_mean,
        "industry_mean_rank_ic": industry_mean,
        "retention_ratio": retention_ratio,
        "daily_delta_mean": delta_mean,
        "nw_lags": inference["nw_lags"],
        "nw_t_value": inference["nw_t_value"],
        "nw_p_value": p_value,
        "thresholds": {
            "min_overlap_days": min_overlap_days,
            "min_retention_ratio": min_retention_ratio,
            "max_p_value": max_p_value,
        },
    }
