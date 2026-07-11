from __future__ import annotations

import math

import numpy as np
import pandas as pd

from factor_forge.evaluation.l1 import _newey_west_stats

from .config import EventStudyConfig, MATCHING_CONTROLS


def analyze_event_study(
    events: pd.DataFrame,
    matched: dict[str, pd.DataFrame],
    regimes: pd.DataFrame,
    config: EventStudyConfig,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    progressive_rows: list[dict] = []
    paired_by_stage: dict[str, pd.DataFrame] = {}
    total_events = int(len(events))
    for stage in config.matching.stages:
        pairs = matched[stage.id]
        paired_frames = []
        matched_events = int(len(pairs[["trade_date", "event_code"]].drop_duplicates())) if len(pairs) else 0
        match_rate = matched_events / total_events if total_events else 0.0
        balance = _balance(pairs)
        for horizon in config.horizons:
            paired = _aggregate_pairs(pairs, horizon)
            paired["stage"] = stage.id
            paired["horizon"] = horizon
            paired_frames.append(paired)
            stats = _paired_stats(paired, horizon)
            progressive_rows.append({
                "stage": stage.id,
                "controls": "+".join(stage.controls) if stage.controls else "exact_only",
                "horizon": horizon,
                "total_frozen_events": total_events,
                "matched_events": matched_events,
                "match_rate": match_rate,
                "mature_matched_events": int(len(paired)),
                "censored_or_unmatched_events": total_events - int(len(paired)),
                **stats,
                **{f"balance_{key}": value for key, value in balance.items()},
                "fdr_q": None,
            })
        paired_by_stage[stage.id] = pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame()

    _apply_fdr(progressive_rows)
    progressive = pd.DataFrame(progressive_rows)
    primary_rows = progressive.loc[
        progressive["stage"].eq(config.inference.primary_stage)
        & progressive["horizon"].eq(config.inference.primary_horizon)
    ]
    primary = primary_rows.iloc[0].to_dict() if len(primary_rows) else {}

    full = paired_by_stage.get(config.inference.primary_stage, pd.DataFrame())
    severity = _severity_analysis(full, config)
    regime = _regime_analysis(full, regimes, config)
    gate = _gate(primary, progressive, config)
    summary = {
        "primary_metric": {
            "stage": config.inference.primary_stage,
            "horizon": config.inference.primary_horizon,
            "metric": "daily_mean_paired_open_to_open_excess",
        },
        "primary_result": _clean_dict(primary),
        "gate": gate,
        "multiple_testing": {
            "method": "Benjamini-Hochberg",
            "tests": len(progressive_rows),
            "alpha": config.inference.fdr_alpha,
            "note": "Gate remains based only on the pre-registered full-control 5-day result.",
        },
    }
    return summary, {
        "progressive_controls": progressive,
        "severity": severity,
        "regime": regime,
        **{f"paired_{stage}": frame for stage, frame in paired_by_stage.items()},
    }


def _aggregate_pairs(pairs: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=[
            "trade_date", "event_code", "severity", "event_return", "control_return",
            "paired_excess",
        ])
    required = pairs.loc[
        pairs[f"event_label_mature_{horizon}"]
        & pairs[f"control_label_mature_{horizon}"]
    ].dropna(subset=[f"event_return_{horizon}", f"control_return_{horizon}"])
    if required.empty:
        return pd.DataFrame(columns=[
            "trade_date", "event_code", "severity", "event_return", "control_return",
            "paired_excess",
        ])
    grouped = required.groupby(["trade_date", "event_code"], as_index=False).agg(
        severity=("severity", "first"),
        event_return=(f"event_return_{horizon}", "first"),
        control_return=(f"control_return_{horizon}", "mean"),
        control_count=("control_code", "nunique"),
        mean_match_distance=("match_distance", "mean"),
    )
    grouped["paired_excess"] = grouped["event_return"] - grouped["control_return"]
    return grouped


def _paired_stats(paired: pd.DataFrame, horizon: int) -> dict:
    if paired.empty:
        return {
            "mean_event_return": None, "mean_control_return": None,
            "event_weighted_mean_paired_excess": None,
            "daily_equal_weight_mean_paired_excess": None,
            "median_paired_excess": None,
            "event_win_rate": None, "daily_direction_ratio": None,
            "nw_lags": horizon - 1, "nw_t_value": None, "nw_p_value": None,
        }
    daily = paired.groupby("trade_date")["paired_excess"].mean()
    nw = _newey_west_stats(daily, max_lags=horizon - 1)
    event_weighted_mean = float(paired["paired_excess"].mean())
    daily_mean = float(daily.mean())
    direction = 1 if daily_mean >= 0 else -1
    return {
        "mean_event_return": float(paired["event_return"].mean()),
        "mean_control_return": float(paired["control_return"].mean()),
        "event_weighted_mean_paired_excess": event_weighted_mean,
        "daily_equal_weight_mean_paired_excess": daily_mean,
        "median_paired_excess": float(paired["paired_excess"].median()),
        "event_win_rate": float((paired["paired_excess"] > 0).mean()),
        "daily_direction_ratio": float((daily * direction > 0).mean()),
        **nw,
    }


def _balance(pairs: pd.DataFrame) -> dict:
    output = {}
    if pairs.empty:
        return {f"abs_smd_{field}": None for field in MATCHING_CONTROLS}
    event_once = pairs.drop_duplicates(["trade_date", "event_code"])
    for field in MATCHING_CONTROLS:
        event = pd.to_numeric(event_once[f"event_{field}"], errors="coerce").dropna()
        control = pd.to_numeric(pairs[f"control_{field}"], errors="coerce").dropna()
        pooled = math.sqrt((event.var(ddof=0) + control.var(ddof=0)) / 2) if len(event) and len(control) else np.nan
        smd = abs(float(event.mean() - control.mean()) / pooled) if np.isfinite(pooled) and pooled > 0 else None
        output[f"abs_smd_{field}"] = smd
    return output


def _severity_analysis(paired: pd.DataFrame, config: EventStudyConfig) -> pd.DataFrame:
    rows = []
    for horizon in config.horizons:
        sample = paired.loc[paired.get("horizon", pd.Series(dtype=int)).eq(horizon)].copy()
        if len(sample) < config.inference.severity_groups * 10:
            continue
        ranks = sample["severity"].rank(method="first")
        sample["severity_group"] = pd.qcut(ranks, config.inference.severity_groups, labels=False) + 1
        for group, bucket in sample.groupby("severity_group"):
            rows.append({
                "horizon": horizon, "severity_group": int(group), "events": len(bucket),
                "severity_min": float(bucket["severity"].min()),
                "severity_max": float(bucket["severity"].max()),
                "mean_paired_excess": float(bucket["paired_excess"].mean()),
                "median_paired_excess": float(bucket["paired_excess"].median()),
            })
    result = pd.DataFrame(rows)
    if len(result):
        monotonic = {}
        for horizon, group in result.groupby("horizon"):
            values = group.sort_values("severity_group")["mean_paired_excess"]
            monotonic[int(horizon)] = bool(values.is_monotonic_increasing or values.is_monotonic_decreasing)
        result["horizon_monotonic"] = result["horizon"].map(monotonic)
    return result


def _regime_analysis(paired: pd.DataFrame, regimes: pd.DataFrame, config: EventStudyConfig) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    sample = paired.merge(regimes, on="trade_date", how="left")
    rows = []
    for horizon in config.horizons:
        horizon_sample = sample.loc[sample["horizon"].eq(horizon)]
        for keys, bucket in horizon_sample.groupby(["market_direction", "market_volatility"]):
            if len(bucket) < config.inference.min_regime_events:
                continue
            daily = bucket.groupby("trade_date")["paired_excess"].mean()
            nw = _newey_west_stats(daily, max_lags=horizon - 1)
            rows.append({
                "horizon": horizon, "market_direction": keys[0], "market_volatility": keys[1],
                "events": len(bucket), "mean_paired_excess": float(bucket["paired_excess"].mean()),
                **nw,
            })
    return pd.DataFrame(rows)


def _gate(primary: dict, progressive: pd.DataFrame, config: EventStudyConfig) -> dict:
    if not primary:
        return {"status": "INVALID", "next_action": "reject", "reason": "primary result missing"}
    if primary.get("match_rate", 0) < config.matching.min_match_rate:
        return {"status": "INSUFFICIENT_MATCH", "next_action": "revise_one_hypothesis",
                "reason": "full-control match rate below threshold"}
    failed_balance = [
        field for field in MATCHING_CONTROLS
        if primary.get(f"balance_abs_smd_{field}") is None
        or primary[f"balance_abs_smd_{field}"] > config.gate.max_abs_smd
    ]
    if failed_balance:
        return {
            "status": "INSUFFICIENT_BALANCE",
            "next_action": "revise_one_hypothesis",
            "reason": "full-control matched sample exceeds SMD threshold: " + ",".join(failed_balance),
        }
    if primary.get("mature_matched_events", 0) < config.inference.min_mature_events:
        return {"status": "INSUFFICIENT_MATURITY", "next_action": "observe_forward",
                "reason": "primary horizon has too few mature matched events"}
    significant = (
        primary.get("nw_t_value") is not None
        and abs(primary["nw_t_value"]) >= config.gate.min_abs_nw_t
        and primary.get("fdr_q") is not None
        and primary["fdr_q"] <= config.gate.max_fdr_q
        and primary.get("daily_direction_ratio", 0) >= config.gate.min_daily_direction_ratio
    )
    if significant:
        return {
            "status": "MATCHED_DIFFERENCE_DETECTED",
            "next_action": "observe_forward",
            "observed_direction": "positive" if primary["daily_equal_weight_mean_paired_excess"] > 0 else "negative",
            "reason": "pre-registered full-control 5-day paired effect passed deterministic gate",
        }
    exact = progressive.loc[
        progressive["stage"].eq(config.matching.stages[0].id)
        & progressive["horizon"].eq(config.inference.primary_horizon)
    ]
    exact_t = exact.iloc[0]["nw_t_value"] if len(exact) else None
    if exact_t is not None and abs(exact_t) >= config.gate.min_abs_nw_t:
        return {"status": "EXPLAINED_BY_CONTROLS", "next_action": "revise_one_hypothesis",
                "reason": "exact-match effect weakened after progressive controls"}
    return {"status": "NO_MATCHED_DIFFERENCE", "next_action": "reject",
            "reason": "pre-registered primary result did not pass"}


def _apply_fdr(rows: list[dict]) -> None:
    indexed = [(i, row.get("nw_p_value")) for i, row in enumerate(rows) if row.get("nw_p_value") is not None]
    if not indexed:
        return
    ordered = sorted(indexed, key=lambda item: item[1])
    total = len(ordered)
    adjusted = [1.0] * total
    running = 1.0
    for position in range(total - 1, -1, -1):
        _, p_value = ordered[position]
        running = min(running, p_value * total / (position + 1))
        adjusted[position] = running
    for (position, _), value in zip(ordered, adjusted, strict=True):
        rows[position]["fdr_q"] = float(min(value, 1.0))


def _clean_dict(value: dict) -> dict:
    return {
        key: (None if isinstance(item, float) and not np.isfinite(item) else item)
        for key, item in value.items()
    }
