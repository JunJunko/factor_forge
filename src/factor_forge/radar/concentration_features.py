"""Deterministic PIT Features for residual turnover-concentration observations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .percentiles import pit_rolling_percentile


TURNOVER_RESIDUAL_CONCENTRATION_FIELDS = (
    "top5_positive_amount_residual_mass_share",
    "residual_concentration_history_percentile",
    "amount_residual_cross_section_percentile",
    "industry_residual_mass_hhi",
    "top_size_decile_residual_mass_share",
    "contributor_return_sign_coherence",
    "contributor_volume_price_efficiency",
    "residual_concentration_persistence_5d",
)


def build_turnover_residual_concentration_features(
    panel: pd.DataFrame,
    *,
    universe_field: str = "is_liquid",
    liquidity_window: int = 20,
    liquidity_min_periods: int = 10,
    history_window: int = 252,
    history_min_periods: int = 120,
    persistence_window: int = 5,
    contributor_percentile: float = 0.95,
    concentration_percentile: float = 0.95,
    min_cross_section: int = 100,
) -> pd.DataFrame:
    """Return stock-date Features using current/strictly-prior information only."""
    if not 0.5 <= contributor_percentile <= 1 or not 0.5 <= concentration_percentile <= 1:
        raise ValueError("residual concentration percentiles must be in [0.5, 1]")
    if liquidity_window < 2 or not 1 <= liquidity_min_periods <= liquidity_window:
        raise ValueError("invalid normal-liquidity window")
    if persistence_window < 1:
        raise ValueError("persistence_window must be positive")
    required = {
        "trade_date", "ts_code", "adj_close", "amount_cny", "log_total_mv",
        "industry_l1_code", universe_field,
    }
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"turnover residual concentration missing columns: {sorted(missing)}")
    if panel.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("turnover residual concentration requires unique stock/date rows")

    result = panel.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    result = result.sort_values(["ts_code", "trade_date"], kind="mergesort")
    amount = pd.to_numeric(result["amount_cny"], errors="coerce")
    log_amount = np.log(amount.where(amount > 0))
    size = pd.to_numeric(result["log_total_mv"], errors="coerce")
    close = pd.to_numeric(result["adj_close"], errors="coerce")
    result["log_amount"] = log_amount
    result["log_avg_amount_20d_prior"] = log_amount.groupby(
        result["ts_code"], sort=False
    ).transform(
        lambda values: values.shift(1).rolling(
            liquidity_window, min_periods=liquidity_min_periods
        ).mean()
    )
    result["return_1d"] = close.groupby(result["ts_code"], sort=False).pct_change(
        1, fill_method=None
    )
    eligible = (
        result[universe_field].fillna(False).astype(bool)
        & log_amount.notna()
        & result["log_avg_amount_20d_prior"].notna()
        & size.notna()
    )
    result["residual_feature_eligible"] = eligible
    result["amount_conditional_residual"] = _daily_residual(
        result,
        target="log_amount",
        controls=["log_avg_amount_20d_prior", "log_total_mv"],
        eligible=eligible,
        min_cross_section=min_cross_section,
    )
    result["amount_residual_cross_section_percentile"] = result[
        "amount_conditional_residual"
    ].where(eligible).groupby(result["trade_date"]).rank(pct=True, method="average")
    result["is_residual_contributor"] = (
        eligible
        & result["amount_conditional_residual"].gt(0)
        & result["amount_residual_cross_section_percentile"].ge(contributor_percentile)
    )
    result["positive_amount_residual_mass"] = np.expm1(
        result["amount_conditional_residual"].clip(lower=0)
    ).where(eligible)
    result["size_percentile"] = size.where(eligible).groupby(result["trade_date"]).rank(
        pct=True, method="average"
    )
    result["contributor_volume_price_efficiency"] = _daily_residual(
        result.assign(abs_return_1d=result["return_1d"].abs()),
        target="abs_return_1d",
        controls=["log_amount"],
        eligible=eligible & result["return_1d"].notna(),
        min_cross_section=min_cross_section,
    )

    daily_records = []
    for trade_date, group in result.groupby("trade_date", sort=True):
        valid = group.loc[group["residual_feature_eligible"]]
        contributors = group.loc[group["is_residual_contributor"]]
        total_mass = float(valid["positive_amount_residual_mass"].sum(min_count=1))
        contributor_mass = float(
            contributors["positive_amount_residual_mass"].sum(min_count=1)
        )
        industry_mass = valid.loc[valid["industry_l1_code"].notna()].groupby(
            "industry_l1_code"
        )["positive_amount_residual_mass"].sum()
        industry_shares = industry_mass / total_mass if total_mass > 0 else industry_mass * np.nan
        top_size_mass = valid.loc[
            valid["size_percentile"].ge(0.90), "positive_amount_residual_mass"
        ].sum(min_count=1)
        weights = contributors["positive_amount_residual_mass"]
        signed = contributors["return_1d"] * weights
        absolute = contributors["return_1d"].abs() * weights
        daily_records.append({
            "trade_date": trade_date,
            "top5_positive_amount_residual_mass_share": _safe_ratio(
                contributor_mass, total_mass
            ),
            "industry_residual_mass_hhi": (
                float((industry_shares ** 2).sum()) if total_mass > 0 else np.nan
            ),
            "top_size_decile_residual_mass_share": _safe_ratio(top_size_mass, total_mass),
            "contributor_return_sign_coherence": _safe_ratio(
                abs(float(signed.sum(min_count=1))), float(absolute.sum(min_count=1))
            ),
        })
    daily = pd.DataFrame.from_records(daily_records).sort_values("trade_date")
    daily["__market_entity"] = "market"
    daily["residual_concentration_history_percentile"] = pit_rolling_percentile(
        daily,
        "top5_positive_amount_residual_mass_share",
        entity_column="__market_entity",
        date_column="trade_date",
        window=history_window,
        min_periods=history_min_periods,
    )
    extreme = daily["residual_concentration_history_percentile"].ge(
        concentration_percentile
    ).where(daily["residual_concentration_history_percentile"].notna())
    daily["residual_concentration_persistence_5d"] = extreme.astype(float).rolling(
        persistence_window, min_periods=persistence_window
    ).sum()
    daily = daily.drop(columns="__market_entity")
    return result.join(daily.set_index("trade_date"), on="trade_date")


def turnover_residual_concentration_prefix_audit(
    panel: pd.DataFrame,
    *,
    fields: Sequence[str] = TURNOVER_RESIDUAL_CONCENTRATION_FIELDS,
    checkpoints: Sequence[float] = (0.55, 0.80),
    **builder_kwargs,
) -> bool:
    """Verify that appending future rows cannot change earlier Feature values."""
    full = build_turnover_residual_concentration_features(panel, **builder_kwargs)
    dates = sorted(pd.to_datetime(full["trade_date"]).unique())
    if len(dates) < 3:
        return True
    panel_dates = pd.to_datetime(panel["trade_date"])
    for fraction in checkpoints:
        cutoff = dates[min(len(dates) - 2, max(1, int(len(dates) * fraction)))]
        prefix = build_turnover_residual_concentration_features(
            panel.loc[panel_dates.le(cutoff)], **builder_kwargs
        )
        keys = ["trade_date", "ts_code"]
        expected = full.loc[full["trade_date"].le(cutoff), [*keys, *fields]].sort_values(keys)
        actual = prefix[[*keys, *fields]].sort_values(keys)
        if not expected[keys].reset_index(drop=True).equals(actual[keys].reset_index(drop=True)):
            return False
        if not np.allclose(
            expected[list(fields)].to_numpy(float),
            actual[list(fields)].to_numpy(float),
            equal_nan=True,
        ):
            return False
    return True


def _daily_residual(frame, *, target, controls, eligible, min_cross_section):
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.loc[eligible].groupby("trade_date", sort=False):
        usable = group[[target, *controls]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(usable) < max(min_cross_section, len(controls) + 2):
            continue
        x = np.column_stack([np.ones(len(usable)), usable[controls].to_numpy(float)])
        y = usable[target].to_numpy(float)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        output.loc[usable.index] = y - x @ beta
    return output


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if np.isfinite(denominator) and denominator > 0 else np.nan
