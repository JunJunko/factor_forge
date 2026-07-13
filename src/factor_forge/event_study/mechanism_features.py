"""Point-in-time market aggregates for anomaly-mechanism discrimination."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from factor_forge.radar.percentiles import pit_rolling_percentile


TURNOVER_CONCENTRATION_AGGREGATE_FIELDS = (
    "industry_amount_hhi",
    "top_size_decile_amount_share",
    "contributor_return_sign_coherence",
    "concentration_persistence_5d",
    "return_contribution_concentration",
)


def build_turnover_concentration_aggregate_features(
    panel: pd.DataFrame,
    *,
    universe_field: str = "is_liquid",
    history_window: int = 252,
    history_min_periods: int = 120,
    persistence_window: int = 5,
    top_amount_fraction: float = 0.05,
    top_size_fraction: float = 0.10,
    concentration_threshold: float = 0.95,
) -> pd.DataFrame:
    """Build daily, label-free aggregates using only information known at close T."""
    if not 0 < top_amount_fraction <= 0.5 or not 0 < top_size_fraction <= 0.5:
        raise ValueError("top fractions must be in (0, 0.5]")
    if not 0 <= concentration_threshold <= 1:
        raise ValueError("concentration_threshold must be in [0, 1]")
    if persistence_window < 1:
        raise ValueError("persistence_window must be positive")
    required = {
        "trade_date", "ts_code", "adj_close", "amount_cny", "log_total_mv",
        "industry_l1_code", universe_field,
    }
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"turnover concentration features missing columns: {sorted(missing)}")
    if panel.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("turnover concentration features require unique stock/date rows")

    data = panel.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data.sort_values(["ts_code", "trade_date"], kind="mergesort")
    close = pd.to_numeric(data["adj_close"], errors="coerce")
    data["return_1d"] = close.groupby(data["ts_code"], sort=False).pct_change(
        1, fill_method=None
    )
    data["amount_value"] = pd.to_numeric(data["amount_cny"], errors="coerce")
    data["size_value"] = pd.to_numeric(data["log_total_mv"], errors="coerce")
    data["eligible"] = (
        data[universe_field].fillna(False).astype(bool)
        & data["amount_value"].gt(0)
        & close.notna()
    )
    eligible_amount = data["amount_value"].where(data["eligible"])
    data["amount_percentile"] = eligible_amount.groupby(data["trade_date"]).rank(
        pct=True, method="average"
    )
    data["is_contributor"] = (
        data["eligible"] & data["amount_percentile"].ge(1 - top_amount_fraction)
    )
    eligible_size = data["size_value"].where(data["eligible"])
    data["size_percentile"] = eligible_size.groupby(data["trade_date"]).rank(
        pct=True, method="average"
    )
    data["is_top_size"] = data["eligible"] & data["size_percentile"].ge(
        1 - top_size_fraction
    )
    data["absolute_return_contribution"] = (
        data["return_1d"].abs() * data["amount_value"]
    ).where(data["eligible"])
    data["signed_return_contribution"] = (
        data["return_1d"] * data["amount_value"]
    ).where(data["eligible"])

    records = []
    for trade_date, group in data.groupby("trade_date", sort=True):
        eligible = group.loc[group["eligible"]]
        total_amount = float(eligible["amount_value"].sum())
        contributors = group.loc[group["is_contributor"]]
        top_size = group.loc[group["is_top_size"]]
        known_industry = eligible.loc[eligible["industry_l1_code"].notna()]
        industry_amount = known_industry.groupby("industry_l1_code")["amount_value"].sum()
        industry_shares = industry_amount / total_amount if total_amount > 0 else industry_amount * np.nan
        unknown_amount = eligible.loc[eligible["industry_l1_code"].isna(), "amount_value"].sum()
        contributor_abs = float(contributors["absolute_return_contribution"].sum(min_count=1))
        contributor_signed = float(contributors["signed_return_contribution"].sum(min_count=1))
        market_abs = float(eligible["absolute_return_contribution"].sum(min_count=1))
        records.append({
            "trade_date": trade_date,
            "eligible_stock_count": int(len(eligible)),
            "contributor_stock_count": int(len(contributors)),
            "top5_amount_share": _safe_ratio(contributors["amount_value"].sum(), total_amount),
            "industry_amount_hhi": float((industry_shares ** 2).sum()) if total_amount > 0 else np.nan,
            "industry_unknown_amount_share": _safe_ratio(unknown_amount, total_amount),
            "top_size_decile_amount_share": _safe_ratio(top_size["amount_value"].sum(), total_amount),
            "contributor_return_sign_coherence": _safe_ratio(
                abs(contributor_signed), contributor_abs
            ),
            "return_contribution_concentration": _safe_ratio(contributor_abs, market_abs),
        })
    daily = pd.DataFrame.from_records(records).sort_values("trade_date").reset_index(drop=True)
    daily["__market_entity"] = "market"
    daily["concentration_history_percentile"] = pit_rolling_percentile(
        daily,
        "top5_amount_share",
        entity_column="__market_entity",
        date_column="trade_date",
        window=history_window,
        min_periods=history_min_periods,
    )
    extreme = daily["concentration_history_percentile"].ge(concentration_threshold).where(
        daily["concentration_history_percentile"].notna()
    )
    daily["concentration_persistence_5d"] = extreme.astype(float).rolling(
        persistence_window, min_periods=persistence_window
    ).sum()
    return daily.drop(columns="__market_entity")


def turnover_concentration_prefix_audit(
    panel: pd.DataFrame,
    *,
    fields: Sequence[str] = TURNOVER_CONCENTRATION_AGGREGATE_FIELDS,
    checkpoints: Sequence[float] = (0.55, 0.80),
    **builder_kwargs,
) -> bool:
    """Verify that appending future rows never changes earlier aggregate Features."""
    full = build_turnover_concentration_aggregate_features(panel, **builder_kwargs)
    dates = full["trade_date"].tolist()
    if len(dates) < 3:
        return True
    panel_dates = pd.to_datetime(panel["trade_date"])
    for fraction in checkpoints:
        cutoff = dates[min(len(dates) - 2, max(1, int(len(dates) * fraction)))]
        prefix = build_turnover_concentration_aggregate_features(
            panel.loc[panel_dates.le(cutoff)], **builder_kwargs
        )
        expected = full.loc[full["trade_date"].le(cutoff), ["trade_date", *fields]]
        actual = prefix[["trade_date", *fields]]
        if not expected["trade_date"].reset_index(drop=True).equals(
            actual["trade_date"].reset_index(drop=True)
        ):
            return False
        if not np.allclose(
            expected[list(fields)].to_numpy(float),
            actual[list(fields)].to_numpy(float),
            equal_nan=True,
        ):
            return False
    return True


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if np.isfinite(denominator) and denominator > 0 else np.nan
