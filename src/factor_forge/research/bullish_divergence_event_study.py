"""Matched E1/E1.1 event study for causal bullish-divergence episodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factor_forge.ml import atr_reversion_features as af
from factor_forge.ml import supply_features as sf


CONTROL_FIELDS = [
    "control__ret_5d",
    "control__ret_20d",
    "control__drawdown_60d",
    "control__log_circ_mv",
    "control__volatility_20d",
    "control__log_adv20",
    "control__turnover_20d",
    "control__beta_60d",
]


@dataclass(frozen=True)
class BullishDivergenceEventStudyConfig:
    horizons: tuple[int, ...] = (5, 10, 20)
    primary_horizon: int = 10
    neighbors: int = 3
    caliper: float = 3.0
    roundtrip_cost_bps: float = 40.0
    block_length: int = 10
    bootstrap_samples: int = 2_000
    bootstrap_seed: int = 42
    minimum_industry_size: int = 4

    def __post_init__(self) -> None:
        if self.primary_horizon not in self.horizons:
            raise ValueError("primary_horizon must be one of horizons")
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("horizons must be positive")
        if self.neighbors < 1 or self.caliper <= 0:
            raise ValueError("neighbors and caliper must be positive")
        if self.block_length < 1 or self.bootstrap_samples < 1:
            raise ValueError("bootstrap settings must be positive")


@dataclass
class BullishDivergenceEventStudyResult:
    episodes: pd.DataFrame
    matched_pairs: pd.DataFrame
    paired_events: pd.DataFrame
    event_summary: pd.DataFrame
    score_monotonicity: pd.DataFrame
    touch_summary: pd.DataFrame
    matching_balance: pd.DataFrame
    bootstrap: pd.DataFrame
    summary: dict


def run_bullish_divergence_event_study(
    panel: pd.DataFrame,
    daily_features: pd.DataFrame,
    episodes: pd.DataFrame,
    config: BullishDivergenceEventStudyConfig = BullishDivergenceEventStudyConfig(),
) -> BullishDivergenceEventStudyResult:
    labeled = build_labels_and_controls(panel, config)
    raw_keys = daily_features[["trade_date", "ts_code", "div__event_candidate"]].copy()
    raw_keys["trade_date"] = pd.to_datetime(raw_keys["trade_date"])
    raw_keys = raw_keys.drop_duplicates(["trade_date", "ts_code"], keep="last")
    labeled = labeled.merge(raw_keys, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    labeled["div__event_candidate"] = labeled["div__event_candidate"].fillna(False).astype(bool)

    event_columns = [
        "trade_date", "ts_code", "event_id", "episode_id", "div__score", "div__score_rank",
        "touch__occurred_10d", "touch__pre_b_count", "touch__post_b_count",
        "touch__last_close_reclaim_atr", "touch__false_break_reclaim",
        "touch__acceptance_score", "touch__level_raw",
    ]
    missing = set(event_columns) - set(episodes.columns)
    if missing:
        raise ValueError(f"episodes are missing event-study fields: {sorted(missing)}")
    event_keys = episodes[event_columns].copy()
    event_keys["trade_date"] = pd.to_datetime(event_keys["trade_date"])
    event_keys["touch_state"] = classify_touch_state(event_keys)
    event_keys["touch_phase"] = classify_touch_phase(event_keys)
    event_keys["score_quintile"] = _score_quintile(event_keys["div__score_rank"])
    enriched_events = event_keys.merge(
        labeled, on=["trade_date", "ts_code"], how="left", validate="one_to_one",
    )
    if enriched_events[f"label__return_{config.primary_horizon}d"].isna().all():
        raise ValueError("no mature primary labels were found for divergence episodes")

    pairs = match_episode_controls(labeled, enriched_events, config)
    paired = aggregate_matched_pairs(pairs, enriched_events, config)
    event_summary = summarize_events(enriched_events, config)
    score = summarize_score_monotonicity(enriched_events, paired, config)
    touch = summarize_touch_states(enriched_events, paired, config)
    balance = matching_balance(pairs)
    bootstrap = bootstrap_inference(paired, config)
    summary = build_decision_summary(enriched_events, pairs, paired, score, touch, balance, bootstrap, config)
    return BullishDivergenceEventStudyResult(
        episodes=enriched_events,
        matched_pairs=pairs,
        paired_events=paired,
        event_summary=event_summary,
        score_monotonicity=score,
        touch_summary=touch,
        matching_balance=balance,
        bootstrap=bootstrap,
        summary=summary,
    )


def build_labels_and_controls(
    panel: pd.DataFrame,
    config: BullishDivergenceEventStudyConfig,
) -> pd.DataFrame:
    required = {
        "trade_date", "ts_code", "adj_open", "adj_high", "adj_low", "adj_close",
        "amount_cny", "turnover_rate", "circ_mv_cny", "industry_l1_code",
        "is_tradeable", "is_suspended", "is_st", "is_delisting_period",
    }
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing event-study fields: {sorted(missing)}")
    data = panel.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    if data.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("panel contains duplicate trade_date/ts_code rows")
    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    codes, dates = data["ts_code"], data["trade_date"]
    close = pd.to_numeric(data["adj_close"], errors="coerce")
    open_ = pd.to_numeric(data["adj_open"], errors="coerce")
    high = pd.to_numeric(data["adj_high"], errors="coerce")
    low = pd.to_numeric(data["adj_low"], errors="coerce")
    amount = pd.to_numeric(data["amount_cny"], errors="coerce")
    turnover = pd.to_numeric(data["turnover_rate"], errors="coerce")
    ret1 = close.groupby(codes, sort=False).pct_change(fill_method=None)
    data["control__ret_5d"] = close.groupby(codes, sort=False).pct_change(5, fill_method=None)
    data["control__ret_20d"] = close.groupby(codes, sort=False).pct_change(20, fill_method=None)
    rolling_high = sf._rolling(close, codes, 60, method="max", min_periods=30)
    data["control__drawdown_60d"] = close / rolling_high.where(rolling_high > 0) - 1.0
    data["control__log_circ_mv"] = np.log(
        pd.to_numeric(data["circ_mv_cny"], errors="coerce").where(lambda values: values > 0)
    )
    data["control__volatility_20d"] = sf._rolling(
        ret1, codes, 20, method="std", min_periods=10, ddof=0
    )
    adv20 = sf._rolling(amount, codes, 20, method="mean", min_periods=10)
    data["control__log_adv20"] = np.log(adv20.where(adv20 > 0))
    data["control__turnover_20d"] = sf._rolling(
        turnover, codes, 20, method="mean", min_periods=10
    )
    eligible = (
        data["is_tradeable"].fillna(False)
        & ~data["is_suspended"].fillna(True)
        & ~data["is_st"].fillna(False)
        & ~data["is_delisting_period"].fillna(False)
        & data["industry_l1_code"].notna()
    )
    market_return = ret1.where(eligible).groupby(dates, sort=False).mean()
    market_on_rows = dates.map(market_return)
    mean_x = sf._rolling(ret1, codes, 60, method="mean", min_periods=30)
    mean_y = sf._rolling(market_on_rows, codes, 60, method="mean", min_periods=30)
    mean_xy = sf._rolling(ret1 * market_on_rows, codes, 60, method="mean", min_periods=30)
    mean_y2 = sf._rolling(market_on_rows**2, codes, 60, method="mean", min_periods=30)
    variance = mean_y2 - mean_y**2
    data["control__beta_60d"] = (mean_xy - mean_x * mean_y) / variance.where(variance.abs() > 1e-12)
    data["match_eligible"] = eligible

    grouped_open = open_.groupby(codes, sort=False)
    entry = grouped_open.shift(-1)
    atr20 = af.atr(high, low, close, codes, 20)
    atr_fraction = atr20 / close.where(close > 0)
    for horizon in config.horizons:
        exit_open = grouped_open.shift(-(horizon + 1))
        future_return = exit_open / entry - 1.0
        data[f"label__return_{horizon}d"] = future_return
        data[f"label__mature_{horizon}d"] = entry.notna() & exit_open.notna()
        benchmark = _industry_loo(future_return.where(eligible), dates, data["industry_l1_code"])
        data[f"label__industry_excess_{horizon}d"] = future_return - benchmark
        data[f"label__industry_excess_net_{horizon}d"] = (
            data[f"label__industry_excess_{horizon}d"] - config.roundtrip_cost_bps / 10_000.0
        )
        if horizon == config.primary_horizon:
            future_high = _future_extreme(high, codes, horizon, "max")
            future_low = _future_extreme(low, codes, horizon, "min")
            data["label__mfe_atr_10d"] = (future_high / entry - 1.0) / atr_fraction
            data["label__mae_atr_10d"] = (future_low / entry - 1.0) / atr_fraction
            data["label__quality_atr_10d"] = (
                data["label__mfe_atr_10d"] - 1.5 * data["label__mae_atr_10d"].abs()
            )
    return data


def classify_touch_state(events: pd.DataFrame) -> pd.Series:
    occurred = events["touch__occurred_10d"].fillna(0).gt(0)
    reclaimed = events["touch__last_close_reclaim_atr"].gt(0)
    false_break = events["touch__false_break_reclaim"].fillna(0).gt(0)
    return pd.Series(np.select(
        [false_break, occurred & reclaimed, occurred],
        ["U3_false_break_reclaim", "U2_touch_reclaim", "U1_touch_no_reclaim"],
        default="U0_no_touch",
    ), index=events.index)


def classify_touch_phase(events: pd.DataFrame) -> pd.Series:
    post = events["touch__post_b_count"].fillna(0).gt(0)
    pre = events["touch__pre_b_count"].fillna(0).gt(0)
    return pd.Series(np.select(
        [post, pre], ["post_b_retest", "pre_b_only"], default="none"
    ), index=events.index)


def match_episode_controls(
    labeled: pd.DataFrame,
    events: pd.DataFrame,
    config: BullishDivergenceEventStudyConfig,
) -> pd.DataFrame:
    primary_label = f"label__return_{config.primary_horizon}d"
    event_lookup = events.set_index(["trade_date", "ts_code"], drop=False)
    event_dates = set(events["trade_date"])
    candidates = labeled.loc[
        labeled["trade_date"].isin(event_dates) & labeled["match_eligible"]
    ].copy()
    rows: list[dict] = []
    for (date, industry), group in candidates.groupby(
        ["trade_date", "industry_l1_code"], sort=True, dropna=False
    ):
        if pd.isna(industry) or len(group) < config.minimum_industry_size:
            continue
        keys = pd.MultiIndex.from_arrays([group["trade_date"], group["ts_code"]])
        is_anchor = keys.isin(event_lookup.index)
        event_rows = group.loc[is_anchor]
        controls = group.loc[~group["div__event_candidate"]].dropna(
            subset=[*CONTROL_FIELDS, primary_label]
        )
        if event_rows.empty or controls.empty:
            continue
        centers, scales = _robust_location_scale(group, CONTROL_FIELDS)
        control_matrix = controls[CONTROL_FIELDS].to_numpy(float)
        center = np.array([centers[name] for name in CONTROL_FIELDS])
        scale = np.array([scales[name] for name in CONTROL_FIELDS])
        standardized_controls = (control_matrix - center) / scale
        for event in event_rows.itertuples(index=False):
            item = event._asdict()
            event_key = (pd.Timestamp(date), str(item["ts_code"]))
            meta = event_lookup.loc[event_key]
            if isinstance(meta, pd.DataFrame):
                meta = meta.iloc[0]
            if any(pd.isna(item[name]) for name in CONTROL_FIELDS) or pd.isna(item[primary_label]):
                continue
            event_values = np.array([item[name] for name in CONTROL_FIELDS], dtype=float)
            distance = np.sqrt(np.mean(
                (standardized_controls - (event_values - center) / scale) ** 2, axis=1
            ))
            ranked = pd.DataFrame({
                "control_position": np.arange(len(controls)),
                "match_distance": distance,
                "control_code": controls["ts_code"].astype(str).to_numpy(),
            }).sort_values(["match_distance", "control_code"], kind="mergesort")
            ranked = ranked.loc[ranked["match_distance"].le(config.caliper)].head(config.neighbors)
            for selected in ranked.itertuples(index=False):
                control = controls.iloc[int(selected.control_position)]
                row = {
                    "event_id": meta["event_id"],
                    "trade_date": pd.Timestamp(date),
                    "industry_l1_code": industry,
                    "event_code": str(item["ts_code"]),
                    "control_code": str(control["ts_code"]),
                    "touch_state": meta["touch_state"],
                    "touch_phase": meta["touch_phase"],
                    "score_quintile": meta["score_quintile"],
                    "match_distance": float(selected.match_distance),
                }
                for field in CONTROL_FIELDS:
                    row[f"event_{field}"] = item[field]
                    row[f"control_{field}"] = control[field]
                for horizon in config.horizons:
                    for label in ("return", "industry_excess", "industry_excess_net"):
                        name = f"label__{label}_{horizon}d"
                        row[f"event_{name}"] = item[name]
                        row[f"control_{name}"] = control[name]
                rows.append(row)
    return pd.DataFrame(rows)


def aggregate_matched_pairs(
    pairs: pd.DataFrame,
    events: pd.DataFrame,
    config: BullishDivergenceEventStudyConfig,
) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    identity = ["event_id", "trade_date", "event_code", "touch_state", "touch_phase", "score_quintile"]
    grouped = pairs.groupby(identity, observed=True, dropna=False)
    result = grouped.size().rename("matched_controls").reset_index()
    for horizon in config.horizons:
        event_col = f"event_label__return_{horizon}d"
        control_col = f"control_label__return_{horizon}d"
        values = grouped.agg(
            event_return=(event_col, "first"), control_return=(control_col, "mean")
        ).reset_index()
        values[f"paired_excess_{horizon}d"] = values["event_return"] - values["control_return"]
        result = result.merge(
            values[identity + [f"paired_excess_{horizon}d"]], on=identity, how="left", validate="one_to_one"
        )
    event_extra = events[[
        "event_id", "div__score", "div__score_rank", "touch__acceptance_score",
        "touch__post_b_count", "label__mfe_atr_10d", "label__mae_atr_10d",
        "label__quality_atr_10d",
        *[f"label__industry_excess_net_{horizon}d" for horizon in config.horizons],
    ]]
    return result.merge(event_extra, on="event_id", how="left", validate="one_to_one")


def summarize_events(events: pd.DataFrame, config: BullishDivergenceEventStudyConfig) -> pd.DataFrame:
    rows = []
    for state, group in events.groupby("touch_state", observed=True):
        for horizon in config.horizons:
            value = group[f"label__industry_excess_net_{horizon}d"].dropna()
            rows.append({
                "touch_state": state, "horizon": horizon,
                "events": int(len(value)), "dates": int(group.loc[value.index, "trade_date"].nunique()),
                "mean_industry_excess_net": float(value.mean()) if len(value) else np.nan,
                "median_industry_excess_net": float(value.median()) if len(value) else np.nan,
                "positive_share": float(value.gt(0).mean()) if len(value) else np.nan,
                "mean_mfe_atr_10d": float(group["label__mfe_atr_10d"].mean()),
                "mean_mae_atr_10d": float(group["label__mae_atr_10d"].mean()),
                "mean_quality_atr_10d": float(group["label__quality_atr_10d"].mean()),
            })
    return pd.DataFrame(rows)


def summarize_score_monotonicity(
    events: pd.DataFrame,
    paired: pd.DataFrame,
    config: BullishDivergenceEventStudyConfig,
) -> pd.DataFrame:
    rows = []
    for quintile, group in events.groupby("score_quintile", observed=True):
        for horizon in config.horizons:
            value = group[f"label__industry_excess_net_{horizon}d"].dropna()
            matched = paired.loc[paired["score_quintile"].eq(quintile), f"paired_excess_{horizon}d"].dropna()
            rows.append({
                "score_quintile": quintile, "horizon": horizon, "events": len(value),
                "mean_industry_excess_net": float(value.mean()) if len(value) else np.nan,
                "positive_share": float(value.gt(0).mean()) if len(value) else np.nan,
                "matched_events": len(matched),
                "mean_paired_excess": float(matched.mean()) if len(matched) else np.nan,
            })
    return pd.DataFrame(rows)


def summarize_touch_states(
    events: pd.DataFrame,
    paired: pd.DataFrame,
    config: BullishDivergenceEventStudyConfig,
) -> pd.DataFrame:
    rows = []
    for state, group in events.groupby("touch_state", observed=True):
        for horizon in config.horizons:
            raw = group[f"label__industry_excess_net_{horizon}d"].dropna()
            matched = paired.loc[paired["touch_state"].eq(state), f"paired_excess_{horizon}d"].dropna()
            rows.append({
                "touch_state": state, "horizon": horizon, "events": len(raw),
                "mean_industry_excess_net": float(raw.mean()) if len(raw) else np.nan,
                "matched_events": len(matched),
                "mean_paired_excess": float(matched.mean()) if len(matched) else np.nan,
            })
    return pd.DataFrame(rows)


def matching_balance(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=["field", "event_mean", "control_mean", "smd"])
    rows = []
    for field in CONTROL_FIELDS:
        event = pd.to_numeric(pairs[f"event_{field}"], errors="coerce")
        control = pd.to_numeric(pairs[f"control_{field}"], errors="coerce")
        pooled = math_sqrt_mean_variance(event, control)
        rows.append({
            "field": field, "event_mean": float(event.mean()), "control_mean": float(control.mean()),
            "smd": float((event.mean() - control.mean()) / pooled) if pooled > 1e-12 else 0.0,
        })
    return pd.DataFrame(rows)


def bootstrap_inference(
    paired: pd.DataFrame,
    config: BullishDivergenceEventStudyConfig,
) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    horizon = config.primary_horizon
    value = f"paired_excess_{horizon}d"
    daily_touch = paired.pivot_table(index="trade_date", columns="touch_state", values=value, aggfunc="mean")
    daily_score = paired.pivot_table(index="trade_date", columns="score_quintile", values=value, aggfunc="mean")
    rows = []
    for name, daily, positive, base in (
        ("U2_minus_U0", daily_touch, "U2_touch_reclaim", "U0_no_touch"),
        ("U3_minus_U0", daily_touch, "U3_false_break_reclaim", "U0_no_touch"),
        ("Q5_minus_Q1", daily_score, "Q5", "Q1"),
    ):
        estimate, low, high, usable_dates = _bootstrap_column_difference(
            daily, positive, base, config
        )
        rows.append({
            "contrast": name, "horizon": horizon, "estimate": estimate,
            "bootstrap_90_low": low, "bootstrap_90_high": high,
            "usable_dates": usable_dates,
        })
    overall_daily = paired.groupby("trade_date", observed=True)[value].mean().to_frame("overall")
    estimate, low, high, usable_dates = _bootstrap_column_difference(
        overall_daily.assign(zero=0.0), "overall", "zero", config
    )
    rows.append({
        "contrast": "matched_overall_vs_zero", "horizon": horizon, "estimate": estimate,
        "bootstrap_90_low": low, "bootstrap_90_high": high, "usable_dates": usable_dates,
    })
    return pd.DataFrame(rows)


def build_decision_summary(
    events: pd.DataFrame,
    pairs: pd.DataFrame,
    paired: pd.DataFrame,
    score: pd.DataFrame,
    touch: pd.DataFrame,
    balance: pd.DataFrame,
    bootstrap: pd.DataFrame,
    config: BullishDivergenceEventStudyConfig,
) -> dict:
    mature = int(events[f"label__return_{config.primary_horizon}d"].notna().sum())
    matched = int(paired["event_id"].nunique()) if not paired.empty else 0
    boot = bootstrap.set_index("contrast") if not bootstrap.empty else pd.DataFrame()
    q5q1 = boot.loc["Q5_minus_Q1"] if "Q5_minus_Q1" in boot.index else None
    u2u0 = boot.loc["U2_minus_U0"] if "U2_minus_U0" in boot.index else None
    max_smd = float(balance["smd"].abs().max()) if len(balance) else np.nan
    checks = {
        "match_rate_at_least_70pct": mature > 0 and matched / mature >= 0.70,
        "max_abs_smd_at_most_0_20": np.isfinite(max_smd) and max_smd <= 0.20,
        "q5_minus_q1_positive": q5q1 is not None and q5q1["estimate"] > 0,
        "q5_minus_q1_ci_low_positive": q5q1 is not None and q5q1["bootstrap_90_low"] > 0,
        "touch_u2_minus_u0_positive": u2u0 is not None and u2u0["estimate"] > 0,
    }
    return {
        "primary_horizon": config.primary_horizon,
        "episode_count": int(len(events)), "mature_episode_count": mature,
        "matched_episode_count": matched,
        "match_rate": matched / mature if mature else 0.0,
        "matched_pair_count": int(len(pairs)),
        "maximum_absolute_smd": max_smd,
        "touch_state_counts": events["touch_state"].value_counts().to_dict(),
        "checks": {name: bool(value) for name, value in checks.items()},
        "decision": "PROCEED_TO_CONDITIONAL_MATRIX" if all(checks.values()) else "STOP_OR_REVISE_BEFORE_ML",
        "config": config.__dict__,
    }


def _industry_loo(values: pd.Series, dates: pd.Series, industries: pd.Series) -> pd.Series:
    grouped = values.groupby([dates, industries], sort=False)
    total = grouped.transform("sum")
    count = grouped.transform("count")
    return ((total - values) / (count - 1)).where(count > 1)


def _future_extreme(
    values: pd.Series, codes: pd.Series, horizon: int, method: str,
) -> pd.Series:
    def calculate(series: pd.Series) -> pd.Series:
        future = series.shift(-1)
        reverse = future.iloc[::-1]
        rolling = reverse.rolling(horizon, min_periods=horizon)
        result = rolling.max() if method == "max" else rolling.min()
        return result.iloc[::-1]
    return values.groupby(codes, sort=False, group_keys=False).apply(calculate).reindex(values.index)


def _score_quintile(rank: pd.Series) -> pd.Series:
    bins = pd.cut(rank, bins=[0, .2, .4, .6, .8, 1.0], labels=["Q1", "Q2", "Q3", "Q4", "Q5"], include_lowest=True)
    return bins.astype("string")


def _robust_location_scale(group: pd.DataFrame, fields: list[str]) -> tuple[dict, dict]:
    centers, scales = {}, {}
    for field in fields:
        values = pd.to_numeric(group[field], errors="coerce").dropna()
        center = float(values.median()) if len(values) else 0.0
        mad = float((values - center).abs().median()) if len(values) else 0.0
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(values.std(ddof=0)) if len(values) else 1.0
        centers[field] = center
        scales[field] = scale if np.isfinite(scale) and scale > 1e-12 else 1.0
    return centers, scales


def math_sqrt_mean_variance(left: pd.Series, right: pd.Series) -> float:
    return float(np.sqrt((left.var(ddof=0) + right.var(ddof=0)) / 2.0))


def _bootstrap_column_difference(
    daily: pd.DataFrame,
    positive: str,
    base: str,
    config: BullishDivergenceEventStudyConfig,
) -> tuple[float, float, float, int]:
    if positive not in daily or base not in daily:
        return np.nan, np.nan, np.nan, 0
    values = daily[[positive, base]].to_numpy(float)
    estimate = float(np.nanmean(values[:, 0]) - np.nanmean(values[:, 1]))
    n = len(values)
    if n == 0:
        return estimate, np.nan, np.nan, 0
    rng = np.random.default_rng(config.bootstrap_seed)
    block = min(config.block_length, n)
    starts = np.arange(max(n - block + 1, 1))
    samples = np.full(config.bootstrap_samples, np.nan)
    for sample in range(config.bootstrap_samples):
        indices: list[int] = []
        while len(indices) < n:
            start = int(rng.choice(starts))
            indices.extend(range(start, min(start + block, n)))
        selected = values[np.array(indices[:n])]
        if np.isfinite(selected[:, 0]).any() and np.isfinite(selected[:, 1]).any():
            samples[sample] = np.nanmean(selected[:, 0]) - np.nanmean(selected[:, 1])
    finite = samples[np.isfinite(samples)]
    if not len(finite):
        return estimate, np.nan, np.nan, n
    return estimate, float(np.quantile(finite, .05)), float(np.quantile(finite, .95)), n

