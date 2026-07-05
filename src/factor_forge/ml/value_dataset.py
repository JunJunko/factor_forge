from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import pandas as pd


FUNDAMENTAL_FIELDS = [
    "revenue_ttm",
    "net_assets",
    "roe_ttm",
    "revenue_growth_yoy",
    "roe_change_yoy",
    "debt_to_assets",
    "net_profit_ttm",
]

VALUE_FEATURES = [
    "liquidity_adjusted_value_gap",
    "fundamental_revision_20d",
    "residual_price_dislocation_20_5",
    "industry_relative_strength_5d",
    "amihud_improvement_5_20",
    "abnormal_attention_5_20",
    "price_delay_improvement_20d",
    "residual_cross_sectional_momentum_120_20",
]


@dataclass(frozen=True)
class ValueFeatureParameters:
    fundamental_revision_window: int = 20
    momentum_formation: int = 100
    momentum_skip: int = 20
    pullback_window: int = 15
    pullback_skip: int = 5
    confirmation_window: int = 5
    liquidity_baseline: int = 20
    delay_window: int = 120
    delay_lags: int = 5
    delay_change: int = 20
    ridge_alpha: float = 5.0
    min_industry_size: int = 20
    mad_scale: float = 5.0


@dataclass(frozen=True)
class CrossSectionGroups:
    items: list[tuple[tuple, np.ndarray]]

    @classmethod
    def build(cls, dates: pd.Series, industries: pd.Series) -> "CrossSectionGroups":
        keys = pd.DataFrame({"date": dates.to_numpy(), "industry": industries.to_numpy()})
        items = [
            (key, np.asarray(positions, dtype=np.int64))
            for key, positions in keys.groupby(
                ["date", "industry"], sort=False, dropna=False
            ).indices.items()
        ]
        return cls(items)


def attach_point_in_time_fundamentals(
    panel: pd.DataFrame, fundamentals: pd.DataFrame
) -> pd.DataFrame:
    """Attach the latest financial snapshot whose usable date is not after trade_date.

    ``available_date`` is deliberately part of the input contract.  It must be the
    first trading date on which the snapshot was knowable (normally the session
    after an after-hours announcement), not the accounting report period.
    """
    required = {"ts_code", "available_date", *FUNDAMENTAL_FIELDS}
    missing = required - set(fundamentals.columns)
    if missing:
        raise ValueError(f"fundamentals PIT table is missing fields: {sorted(missing)}")
    if fundamentals.duplicated(["ts_code", "available_date"]).any():
        raise ValueError("fundamentals PIT key ts_code + available_date is not unique")
    left = panel.copy()
    left["trade_date"] = pd.to_datetime(left["trade_date"])
    right = fundamentals[["ts_code", "available_date", *FUNDAMENTAL_FIELDS]].copy()
    right["available_date"] = pd.to_datetime(right["available_date"])
    # merge_asof requires the `on` key to be globally monotonic even when `by`
    # is supplied, hence date is the leading sort key.
    left = left.sort_values(["trade_date", "ts_code"])
    right = right.sort_values(["available_date", "ts_code"])
    result = pd.merge_asof(
        left,
        right,
        left_on="trade_date",
        right_on="available_date",
        by="ts_code",
        direction="backward",
        allow_exact_matches=True,
    )
    if (result["available_date"] > result["trade_date"]).fillna(False).any():
        raise AssertionError("point-in-time fundamental join used a future snapshot")
    return result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _mad_winsorize(value: pd.Series, dates: pd.Series, scale: float) -> pd.Series:
    median = value.groupby(dates).transform("median")
    mad = (value - median).abs().groupby(dates).transform("median")
    usable = mad.gt(0) & np.isfinite(mad)
    lower, upper = median - scale * mad, median + scale * mad
    return value.where(~usable, value.clip(lower, upper))


def _zscore(value: pd.Series, dates: pd.Series) -> pd.Series:
    grouped = value.groupby(dates)
    mean = grouped.transform("mean")
    std = grouped.transform("std", ddof=0)
    return (value - mean) / std.replace(0, np.nan)


def _rolling_by_stock(
    value: pd.Series,
    stocks: pd.Series,
    window: int,
    method: str = "mean",
    *,
    shift: int = 0,
) -> pd.Series:
    source = value.groupby(stocks, sort=False).shift(shift) if shift else value
    rolling = source.groupby(stocks, sort=False).rolling(window, min_periods=window)
    result = getattr(rolling, method)().reset_index(level=0, drop=True)
    return result.reindex(value.index)


def _group_residualize(
    target: pd.Series,
    controls: list[pd.Series],
    dates: pd.Series,
    industries: pd.Series,
    *,
    min_size: int,
    ridge_alpha: float = 0.0,
    return_coefficients: bool = False,
    groups: CrossSectionGroups | None = None,
    stage: str = "cross_section_residualize",
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[pd.Series, pd.DataFrame | None]:
    """Daily within-industry residualization with finite-sample safeguards."""
    result = np.full(len(target), np.nan, dtype=float)
    coefficient_rows: list[dict] = []
    y_values = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
    x_values = (
        np.column_stack([
            pd.to_numeric(control, errors="coerce").to_numpy(dtype=float)
            for control in controls
        ])
        if controls else np.empty((len(target), 0), dtype=float)
    )
    x_columns = [f"x{position}" for position in range(len(controls))]
    groups = groups or CrossSectionGroups.build(dates, industries)
    total_groups = len(groups.items)
    report_every = max(total_groups // 20, 1)
    for completed, ((date, industry), positions) in enumerate(groups.items, start=1):
        if progress and (completed == 1 or completed % report_every == 0 or completed == total_groups):
            progress(stage, completed, total_groups)
        if pd.isna(industry):
            continue
        local_y = y_values[positions]
        local_x = x_values[positions]
        finite = np.isfinite(local_y)
        if local_x.shape[1]:
            finite &= np.isfinite(local_x).all(axis=1)
        sample_positions = positions[finite]
        minimum = max(min_size, len(x_columns) + 3)
        if len(sample_positions) < minimum:
            continue
        y = local_y[finite]
        if not x_columns:
            fitted = np.repeat(y.mean(), len(y))
            raw_beta = np.empty(0)
            intercept = float(y.mean())
        else:
            raw_x = local_x[finite]
            x_mean = raw_x.mean(axis=0)
            x_std = raw_x.std(axis=0)
            usable = x_std > 1e-12
            if not usable.any():
                fitted = np.repeat(y.mean(), len(y))
                raw_beta = np.zeros(len(x_columns))
                intercept = float(y.mean())
            else:
                x = (raw_x[:, usable] - x_mean[usable]) / x_std[usable]
                gram = x.T @ x
                if ridge_alpha:
                    gram = gram + ridge_alpha * np.eye(gram.shape[0])
                rhs = x.T @ (y - y.mean())
                try:
                    beta = np.linalg.solve(gram, rhs)
                except np.linalg.LinAlgError:
                    beta = np.linalg.lstsq(gram, rhs, rcond=None)[0]
                fitted = y.mean() + x @ beta
                raw_beta = np.zeros(len(x_columns))
                raw_beta[usable] = beta / x_std[usable]
                intercept = float(y.mean() - raw_beta @ x_mean)
        result[sample_positions] = y - fitted
        if return_coefficients:
            row = {"trade_date": date, "industry": industry, "intercept": intercept}
            row.update({f"beta_{column}": float(raw_beta[i]) for i, column in enumerate(x_columns)})
            coefficient_rows.append(row)
    coefficients = pd.DataFrame(coefficient_rows) if return_coefficients else None
    return pd.Series(result, index=target.index), coefficients


def _fundamental_value_components(
    data: pd.DataFrame,
    parameters: ValueFeatureParameters,
    groups: CrossSectionGroups,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[pd.Series, pd.Series]:
    log_revenue = np.log(pd.to_numeric(data["revenue_ttm"], errors="coerce").where(lambda x: x > 0))
    log_assets = np.log(pd.to_numeric(data["net_assets"], errors="coerce").where(lambda x: x > 0))
    profit = pd.to_numeric(data["net_profit_ttm"], errors="coerce")
    loss_flag = profit.le(0).astype(float).where(profit.notna())
    transformed = [
        log_revenue,
        log_assets,
        pd.to_numeric(data["roe_ttm"], errors="coerce"),
        pd.to_numeric(data["revenue_growth_yoy"], errors="coerce"),
        pd.to_numeric(data["roe_change_yoy"], errors="coerce"),
        pd.to_numeric(data["debt_to_assets"], errors="coerce"),
        loss_flag,
    ]
    y = pd.to_numeric(data["log_total_mv"], errors="coerce")
    residual, coefficients = _group_residualize(
        y,
        transformed,
        data["trade_date"],
        data["industry_l1_code"],
        min_size=parameters.min_industry_size,
        ridge_alpha=parameters.ridge_alpha,
        return_coefficients=True,
        groups=groups,
        stage="fundamental_value_regression",
        progress=progress,
    )
    gap = -residual
    revision = pd.Series(np.nan, index=data.index, dtype=float)
    if coefficients is None or coefficients.empty:
        return gap, revision

    coefficient_map = coefficients.set_index(["trade_date", "industry"])
    dates = pd.Index(data["trade_date"].drop_duplicates().sort_values())
    lag_date_map = pd.Series(dates, index=dates).shift(parameters.fundamental_revision_window)
    lag_dates = data["trade_date"].map(lag_date_map)
    lagged_x = [series.groupby(data["ts_code"], sort=False).shift(parameters.fundamental_revision_window) for series in transformed]
    same_industry = data["industry_l1_code"].eq(
        data["industry_l1_code"].groupby(data["ts_code"], sort=False).shift(parameters.fundamental_revision_window)
    )
    lookup = pd.MultiIndex.from_arrays([lag_dates, data["industry_l1_code"]])
    for position, (current, previous) in enumerate(zip(transformed, lagged_x)):
        beta_name = f"beta_x{position}"
        beta = coefficient_map[beta_name].reindex(lookup).set_axis(data.index)
        contribution = beta * (current - previous)
        revision = contribution if position == 0 else revision.add(contribution, fill_value=0)
    revision = revision.where(same_industry & lag_dates.notna())
    return gap, revision


def _rolling_corr(
    left: pd.Series, right: pd.Series, stocks: pd.Series, window: int
) -> pd.Series:
    left_mean = _rolling_by_stock(left, stocks, window)
    right_mean = _rolling_by_stock(right, stocks, window)
    product_mean = _rolling_by_stock(left * right, stocks, window)
    covariance = product_mean - left_mean * right_mean
    left_var = _rolling_by_stock(left * left, stocks, window) - left_mean * left_mean
    right_var = _rolling_by_stock(right * right, stocks, window) - right_mean * right_mean
    return covariance / np.sqrt(left_var.clip(lower=0) * right_var.clip(lower=0)).replace(0, np.nan)


def build_value_dataset(
    panel: pd.DataFrame,
    *,
    horizons: list[int] | tuple[int, ...] = (5, 10, 20),
    parameters: ValueFeatureParameters | None = None,
    excess_over_universe: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the eight economically ordered value-recovery features and labels."""
    parameters = parameters or ValueFeatureParameters()
    required = {
        "trade_date", "ts_code", "adj_open", "adj_close", "amount_cny",
        "turnover_rate", "log_total_mv", "log_circ_mv", "industry_l1_code",
        *FUNDAMENTAL_FIELDS,
    }
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing value-regression fields: {sorted(missing)}")
    data = panel.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    stocks = data["ts_code"]
    dates = data["trade_date"]
    industries = data["industry_l1_code"]
    cross_sections = CrossSectionGroups.build(dates, industries)
    if progress:
        progress("cross_section_index", len(cross_sections.items), len(cross_sections.items))

    close = pd.to_numeric(data["adj_close"], errors="coerce").where(lambda x: x > 0)
    log_return = np.log(close).groupby(stocks, sort=False).diff()
    group_keys = [dates, industries]
    count = log_return.notna().groupby(group_keys, dropna=False).transform("sum")
    total = log_return.fillna(0).groupby(group_keys, dropna=False).transform("sum")
    industry_loo_return = (total - log_return) / (count - 1).replace(0, np.nan)
    relative_return = log_return - industry_loo_return

    turnover = pd.to_numeric(data["turnover_rate"], errors="coerce").where(lambda x: x >= 0)
    amount = pd.to_numeric(data["amount_cny"], errors="coerce").where(lambda x: x > 0)
    amihud = log_return.abs() / (amount / 100_000_000.0)
    recent_5_amihud = _rolling_by_stock(amihud, stocks, parameters.confirmation_window)
    baseline_amihud = _rolling_by_stock(
        amihud, stocks, parameters.liquidity_baseline, shift=parameters.confirmation_window
    )
    raw_f5 = np.log(baseline_amihud.where(lambda x: x > 0)) - np.log(recent_5_amihud.where(lambda x: x > 0))
    recent_turnover = _rolling_by_stock(turnover, stocks, parameters.confirmation_window)
    baseline_turnover = _rolling_by_stock(
        turnover, stocks, parameters.liquidity_baseline, shift=parameters.confirmation_window
    )
    raw_f6 = np.log(recent_turnover.where(lambda x: x > 0)) - np.log(baseline_turnover.where(lambda x: x > 0))

    fundamental_gap, raw_f2 = _fundamental_value_components(
        data, parameters, cross_sections, progress
    )
    volatility_20 = _rolling_by_stock(log_return, stocks, 20, method="std")
    liquidity_controls = [
        np.log(_rolling_by_stock(amihud, stocks, 20).where(lambda x: x > 0)),
        np.log(_rolling_by_stock(turnover, stocks, 20).where(lambda x: x > 0)),
        volatility_20,
        pd.to_numeric(data["log_circ_mv"], errors="coerce"),
    ]
    f1, _ = _group_residualize(
        fundamental_gap, liquidity_controls, dates, industries,
        min_size=parameters.min_industry_size, ridge_alpha=parameters.ridge_alpha,
        groups=cross_sections, stage="liquidity_adjusted_value_gap", progress=progress,
    )
    f2, _ = _group_residualize(
        raw_f2, [f1], dates, industries,
        min_size=parameters.min_industry_size, ridge_alpha=0,
        groups=cross_sections, stage="fundamental_revision_20d", progress=progress,
    )

    raw_f8 = _rolling_by_stock(
        relative_return, stocks, parameters.momentum_formation,
        method="sum", shift=parameters.momentum_skip,
    )
    f8, _ = _group_residualize(
        raw_f8, [f1, f2], dates, industries,
        min_size=parameters.min_industry_size, ridge_alpha=0,
        groups=cross_sections, stage="residual_momentum_120_20", progress=progress,
    )
    raw_f3 = -_rolling_by_stock(
        relative_return, stocks, parameters.pullback_window,
        method="sum", shift=parameters.pullback_skip,
    )
    f3, _ = _group_residualize(
        raw_f3, [f1, f2, f8], dates, industries,
        min_size=parameters.min_industry_size, ridge_alpha=0,
        groups=cross_sections, stage="price_dislocation_20_5", progress=progress,
    )
    raw_f4 = _rolling_by_stock(
        relative_return, stocks, parameters.confirmation_window, method="sum"
    )
    one_day_reversal = -relative_return
    f4, _ = _group_residualize(
        raw_f4, [f3, f8, one_day_reversal, volatility_20], dates, industries,
        min_size=parameters.min_industry_size, ridge_alpha=0,
        groups=cross_sections, stage="industry_strength_5d", progress=progress,
    )

    recent_abs_return = _rolling_by_stock(log_return.abs(), stocks, 5, method="sum")
    f5, _ = _group_residualize(
        raw_f5, [recent_abs_return, volatility_20, f3, f4], dates, industries,
        min_size=parameters.min_industry_size, ridge_alpha=0,
        groups=cross_sections, stage="amihud_improvement_5_20", progress=progress,
    )
    f6, _ = _group_residualize(
        raw_f6,
        [
            recent_abs_return,
            volatility_20,
            f5,
            pd.to_numeric(data["log_circ_mv"], errors="coerce"),
            f3,
            f4,
        ],
        dates, industries, min_size=parameters.min_industry_size, ridge_alpha=0,
        groups=cross_sections, stage="abnormal_attention_5_20", progress=progress,
    )

    # Low-frequency price-delay proxy: the share of industry-response correlation
    # carried by lags 1..K rather than the contemporaneous industry return.
    current_corr = _rolling_corr(log_return, industry_loo_return, stocks, parameters.delay_window).abs()
    lagged_corr = pd.Series(0.0, index=data.index)
    for lag in range(1, parameters.delay_lags + 1):
        lagged_industry = industry_loo_return.groupby(stocks, sort=False).shift(lag)
        lagged_corr = lagged_corr.add(
            _rolling_corr(log_return, lagged_industry, stocks, parameters.delay_window).abs(),
            fill_value=0,
        )
    delay = lagged_corr / (current_corr + lagged_corr).replace(0, np.nan)
    raw_f7 = delay.groupby(stocks, sort=False).shift(parameters.delay_change) - delay
    f7, _ = _group_residualize(
        raw_f7, [f4, f5, f6], dates, industries,
        min_size=parameters.min_industry_size, ridge_alpha=0,
        groups=cross_sections, stage="price_delay_improvement_20d", progress=progress,
    )

    raw_features = [f1, f2, f3, f4, f5, f6, f7, f8]
    for name, value in zip(VALUE_FEATURES, raw_features):
        clipped = _mad_winsorize(value, dates, parameters.mad_scale)
        data[name] = _zscore(clipped, dates)
    if progress:
        progress("feature_postprocessing", len(VALUE_FEATURES), len(VALUE_FEATURES))

    grouped = data.groupby("ts_code", sort=False)
    label_names: list[str] = []
    eligible = data.get("is_tradeable", pd.Series(True, index=data.index)).eq(True)
    for horizon in sorted(set(int(item) for item in horizons)):
        name = f"label_{horizon}d"
        future = grouped["adj_open"].shift(-(horizon + 1)) / grouped["adj_open"].shift(-1) - 1
        if excess_over_universe:
            future = future - future.where(eligible).groupby(dates).transform("mean")
        data[name] = future
        label_names.append(name)
    if progress:
        progress("labels", len(label_names), len(label_names))
    data = data.replace([np.inf, -np.inf], np.nan)
    return data.sort_values(["trade_date", "ts_code"]).reset_index(drop=True), VALUE_FEATURES.copy(), label_names


def daily_feature_dependence(dataset: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Summarize daily pairwise Spearman dependence for the independence audit."""
    pairs = [
        (left, right)
        for position, left in enumerate(features)
        for right in features[position + 1:]
    ]
    values = {pair: [] for pair in pairs}
    # One date pass computes the full 8x8 matrix.  The previous pair-by-pair
    # implementation scanned the multi-million-row training set 28 times.
    for _, frame in dataset.groupby("trade_date", sort=False):
        correlation = frame[features].corr(method="spearman")
        for pair in pairs:
            value = correlation.at[pair[0], pair[1]]
            if np.isfinite(value):
                values[pair].append(abs(float(value)))
    rows: list[dict] = []
    for left, right in pairs:
        daily = np.asarray(values[(left, right)], dtype=float)
        rows.append({
            "feature_left": left,
            "feature_right": right,
            "median_abs_spearman": float(np.median(daily)) if len(daily) else np.nan,
            "p90_abs_spearman": float(np.quantile(daily, 0.9)) if len(daily) else np.nan,
            "observations": int(len(daily)),
        })
    return pd.DataFrame(rows)
