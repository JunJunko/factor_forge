"""Corrected diagnostic E1 for the mechanism-aligned bullish-divergence v2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factor_forge.research.bullish_divergence_event_study import (
    CONTROL_FIELDS,
    BullishDivergenceEventStudyConfig,
    build_labels_and_controls,
    math_sqrt_mean_variance,
)


GEOMETRY_MATCH_FIELDS = [
    "div__price_lower_low_atr",
    "div__trough_gap_days",
    "div__intervening_rebound_atr",
    "div__descent_into_a_3d",
    "div__descent_into_b_3d",
]
V2_MATCH_FIELDS = [*CONTROL_FIELDS, *GEOMETRY_MATCH_FIELDS]


@dataclass(frozen=True)
class BullishDivergenceV2EventStudyConfig:
    primary_horizon: int = 10
    neighbors: int = 3
    caliper: float = 3.0
    roundtrip_cost_bps: float = 40.0
    block_length: int = 10
    bootstrap_samples: int = 2_000
    bootstrap_seed: int = 42
    minimum_industry_size: int = 4


@dataclass
class BullishDivergenceV2EventStudyResult:
    events: pd.DataFrame
    pairs: pd.DataFrame
    paired_events: pd.DataFrame
    score_summary: pd.DataFrame
    state_summary: pd.DataFrame
    balance: pd.DataFrame
    bootstrap: pd.DataFrame
    summary: dict


def run_v2_origin_event_study(
    panel: pd.DataFrame,
    daily_features: pd.DataFrame,
    episodes: pd.DataFrame,
    config: BullishDivergenceV2EventStudyConfig = BullishDivergenceV2EventStudyConfig(),
) -> BullishDivergenceV2EventStudyResult:
    feature_fields = [
        "trade_date", "ts_code", "div_v2__geometry_candidate",
        "div_v2__event_candidate", "div_v2__score", "div_v2__score_rank",
        "support_v2__pre_b_present", "support_v2__pre_b_count",
        *GEOMETRY_MATCH_FIELDS,
    ]
    labeled = _labeled_with_features(panel, daily_features, feature_fields, config)
    event_fields = [
        "trade_date", "ts_code", "event_id", "episode_id",
        "div_v2__score", "div_v2__score_rank",
        "support_v2__pre_b_present", "support_v2__pre_b_count",
    ]
    events = episodes[event_fields].merge(
        labeled, on=["trade_date", "ts_code"], how="left",
        suffixes=("", "_daily"), validate="one_to_one",
    )
    events["event_score_rank"] = events.groupby("trade_date", sort=False)[
        "div_v2__score"
    ].rank(pct=True)
    events["score_quintile"] = _score_quintile(events["event_score_rank"])
    events["event_state"] = np.where(
        events["support_v2__pre_b_present"].fillna(False),
        "S1_pre_b_support", "S0_no_pre_b_support",
    )
    pairs = match_v2_events(
        labeled, events, config,
        control_mask=(
            labeled["div_v2__geometry_candidate"].fillna(False)
            & ~labeled["div_v2__event_candidate"].fillna(False)
        ),
        match_fields=V2_MATCH_FIELDS,
    )
    return _assemble_result(events, pairs, config, study_kind="origin_geometry_placebo")


def run_v2_retest_event_study(
    panel: pd.DataFrame,
    daily_features: pd.DataFrame,
    retest_events: pd.DataFrame,
    config: BullishDivergenceV2EventStudyConfig = BullishDivergenceV2EventStudyConfig(),
) -> BullishDivergenceV2EventStudyResult:
    feature_fields = [
        "trade_date", "ts_code", "div_v2__geometry_candidate",
        "div_v2__event_candidate", *GEOMETRY_MATCH_FIELDS,
    ]
    labeled = _labeled_with_features(panel, daily_features, feature_fields, config)
    events = retest_events.merge(
        labeled, on=["trade_date", "ts_code"], how="left", validate="one_to_one"
    )
    events["event_score_rank"] = events.groupby("trade_date", sort=False)[
        "div_v2__score"
    ].rank(pct=True)
    events["score_quintile"] = _score_quintile(events["event_score_rank"])
    events["event_state"] = np.select(
        [
            events["retest_v2__false_break_reclaim"].fillna(False),
            events["retest_v2__reclaimed"].fillna(False),
        ],
        ["R2_false_break_reclaim", "R1_reclaim"],
        default="R0_no_reclaim",
    )
    trigger_keys = pd.MultiIndex.from_frame(events[["trade_date", "ts_code"]])
    labeled_keys = pd.MultiIndex.from_frame(labeled[["trade_date", "ts_code"]])
    controls = ~labeled_keys.isin(trigger_keys)
    pairs = match_v2_events(
        labeled, events, config,
        control_mask=pd.Series(controls, index=labeled.index) & labeled["match_eligible"],
        match_fields=CONTROL_FIELDS,
    )
    return _assemble_result(events, pairs, config, study_kind="post_signal_retest")


def _labeled_with_features(
    panel: pd.DataFrame,
    daily_features: pd.DataFrame,
    feature_fields: list[str],
    config: BullishDivergenceV2EventStudyConfig,
) -> pd.DataFrame:
    base_config = BullishDivergenceEventStudyConfig(
        horizons=(config.primary_horizon,),
        primary_horizon=config.primary_horizon,
        neighbors=config.neighbors,
        caliper=config.caliper,
        roundtrip_cost_bps=config.roundtrip_cost_bps,
        block_length=config.block_length,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
        minimum_industry_size=config.minimum_industry_size,
    )
    labeled = build_labels_and_controls(panel, base_config)
    features = daily_features[feature_fields].copy()
    features["trade_date"] = pd.to_datetime(features["trade_date"])
    features = features.drop_duplicates(["trade_date", "ts_code"], keep="last")
    return labeled.merge(
        features, on=["trade_date", "ts_code"], how="left", validate="one_to_one"
    )


def match_v2_events(
    labeled: pd.DataFrame,
    events: pd.DataFrame,
    config: BullishDivergenceV2EventStudyConfig,
    *,
    control_mask: pd.Series,
    match_fields: list[str],
) -> pd.DataFrame:
    label = f"label__return_{config.primary_horizon}d"
    event_lookup = events.set_index(["trade_date", "ts_code"], drop=False)
    event_dates = set(events["trade_date"])
    candidates = labeled.loc[
        labeled["trade_date"].isin(event_dates) & labeled["match_eligible"]
    ].copy()
    allowed_control_keys = pd.MultiIndex.from_frame(
        labeled.loc[control_mask.fillna(False), ["trade_date", "ts_code"]]
    )
    rows: list[dict] = []
    for (date, industry), group in candidates.groupby(
        ["trade_date", "industry_l1_code"], sort=True, dropna=False
    ):
        if pd.isna(industry) or len(group) < config.minimum_industry_size:
            continue
        keys = pd.MultiIndex.from_frame(group[["trade_date", "ts_code"]])
        event_rows = group.loc[keys.isin(event_lookup.index)]
        controls = group.loc[keys.isin(allowed_control_keys)].dropna(
            subset=[*match_fields, label]
        )
        event_rows = event_rows.dropna(subset=[*match_fields, label])
        if event_rows.empty or controls.empty:
            continue
        centers, scales = _robust_location_scale(group, match_fields)
        center = np.array([centers[name] for name in match_fields])
        scale = np.array([scales[name] for name in match_fields])
        standardized_controls = (
            controls[match_fields].to_numpy(float) - center
        ) / scale
        standardized_events = (
            event_rows[match_fields].to_numpy(float) - center
        ) / scale
        distances = np.sqrt(np.mean(
            (
                standardized_events[:, None, :]
                - standardized_controls[None, :, :]
            ) ** 2,
            axis=2,
        ))
        control_codes = controls["ts_code"].astype(str).to_numpy()
        control_records = controls.to_dict("records")
        for event_position, event in enumerate(event_rows.itertuples(index=False)):
            item = event._asdict()
            meta = event_lookup.loc[(pd.Timestamp(date), str(item["ts_code"]))]
            if isinstance(meta, pd.DataFrame):
                meta = meta.iloc[0]
            distance = distances[event_position]
            order = np.lexsort((control_codes, distance))
            selected_positions = [
                int(position) for position in order
                if distance[position] <= config.caliper
            ][:config.neighbors]
            for control_position in selected_positions:
                control = control_records[control_position]
                row = {
                    "event_id": meta["event_id"],
                    "trade_date": pd.Timestamp(date),
                    "industry_l1_code": industry,
                    "event_code": str(item["ts_code"]),
                    "control_code": str(control["ts_code"]),
                    "score_quintile": meta["score_quintile"],
                    "event_state": meta["event_state"],
                    "match_distance": float(distance[control_position]),
                    "event_return": item[label],
                    "control_return": control[label],
                }
                for field in match_fields:
                    row[f"event_{field}"] = item[field]
                    row[f"control_{field}"] = control[field]
                rows.append(row)
    return pd.DataFrame(rows)


def _assemble_result(
    events: pd.DataFrame,
    pairs: pd.DataFrame,
    config: BullishDivergenceV2EventStudyConfig,
    *,
    study_kind: str,
) -> BullishDivergenceV2EventStudyResult:
    paired = aggregate_v2_pairs(pairs, config)
    raw_label = f"label__industry_excess_net_{config.primary_horizon}d"
    score_summary = _group_summary(events, paired, "score_quintile", raw_label)
    state_summary = _group_summary(events, paired, "event_state", raw_label)
    balance = matching_balance_v2(pairs)
    bootstrap = bootstrap_v2(paired, config)
    mature = int(events[f"label__return_{config.primary_horizon}d"].notna().sum())
    matched = int(paired["event_id"].nunique()) if len(paired) else 0
    max_smd = float(balance["smd"].abs().max()) if len(balance) else np.nan
    boot = bootstrap.set_index("contrast") if len(bootstrap) else pd.DataFrame()
    q5 = boot.loc["Q5_minus_Q1"] if "Q5_minus_Q1" in boot.index else None
    overall = boot.loc["overall_vs_zero"] if "overall_vs_zero" in boot.index else None
    summary = {
        "study_kind": study_kind,
        "research_status": "DIAGNOSTIC_V2_INSPECTED_HISTORY",
        "event_count": int(len(events)),
        "mature_event_count": mature,
        "matched_event_count": matched,
        "match_rate": matched / mature if mature else 0.0,
        "matched_pair_count": int(len(pairs)),
        "maximum_absolute_smd": max_smd,
        "checks": {
            "match_rate_at_least_70pct": mature > 0 and matched / mature >= .70,
            "maximum_absolute_smd_at_most_0_20": np.isfinite(max_smd) and max_smd <= .20,
            "q5_minus_q1_positive": q5 is not None and q5["estimate"] > 0,
            "q5_minus_q1_ci_low_positive": q5 is not None and q5["bootstrap_90_low"] > 0,
            "overall_matched_effect_positive": overall is not None and overall["estimate"] > 0,
        },
    }
    summary["decision"] = (
        "ELIGIBLE_FOR_CONDITIONAL_MATRIX_DIAGNOSTIC"
        if all(summary["checks"].values())
        else "STOP_OR_REVISE_BEFORE_CONDITIONAL_MATRIX"
    )
    return BullishDivergenceV2EventStudyResult(
        events=events, pairs=pairs, paired_events=paired,
        score_summary=score_summary, state_summary=state_summary,
        balance=balance, bootstrap=bootstrap, summary=summary,
    )


def aggregate_v2_pairs(
    pairs: pd.DataFrame,
    config: BullishDivergenceV2EventStudyConfig,
) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=[
            "event_id", "trade_date", "score_quintile", "event_state",
            "matched_controls", "paired_excess_10d", "paired_excess_net_10d",
        ])
    identity = ["event_id", "trade_date", "event_code", "score_quintile", "event_state"]
    grouped = pairs.groupby(identity, observed=True, dropna=False)
    result = grouped.agg(
        matched_controls=("control_code", "size"),
        event_return=("event_return", "first"),
        control_return=("control_return", "mean"),
    ).reset_index()
    result["paired_excess_10d"] = result["event_return"] - result["control_return"]
    result["paired_excess_net_10d"] = (
        result["paired_excess_10d"] - config.roundtrip_cost_bps / 10_000.0
    )
    return result


def matching_balance_v2(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=["field", "event_mean", "control_mean", "smd"])
    fields = [
        name.removeprefix("event_")
        for name in pairs.columns
        if name.startswith("event_control__") or name.startswith("event_div__")
    ]
    rows = []
    for field in fields:
        event = pd.to_numeric(pairs[f"event_{field}"], errors="coerce")
        control = pd.to_numeric(pairs[f"control_{field}"], errors="coerce")
        pooled = math_sqrt_mean_variance(event, control)
        rows.append({
            "field": field,
            "event_mean": float(event.mean()),
            "control_mean": float(control.mean()),
            "smd": float((event.mean() - control.mean()) / pooled) if pooled > 1e-12 else 0.0,
        })
    return pd.DataFrame(rows)


def bootstrap_v2(
    paired: pd.DataFrame,
    config: BullishDivergenceV2EventStudyConfig,
) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    value = "paired_excess_10d"
    rows = []
    daily_score = paired.pivot_table(
        index="trade_date", columns="score_quintile", values=value, aggfunc="mean"
    )
    rows.append(_bootstrap_complete_case(
        daily_score, "Q5", "Q1", "Q5_minus_Q1", config
    ))
    daily_state = paired.pivot_table(
        index="trade_date", columns="event_state", values=value, aggfunc="mean"
    )
    states = set(paired["event_state"])
    for positive, base, name in (
        ("S1_pre_b_support", "S0_no_pre_b_support", "pre_b_support_minus_none"),
        ("R1_reclaim", "R0_no_reclaim", "reclaim_minus_no_reclaim"),
        ("R2_false_break_reclaim", "R0_no_reclaim", "false_break_minus_no_reclaim"),
    ):
        if positive in states and base in states:
            rows.append(_bootstrap_complete_case(
                daily_state, positive, base, name, config
            ))
    daily = paired.groupby("trade_date", observed=True)[value].mean().dropna()
    estimate, low, high = _moving_block_bootstrap(daily.to_numpy(), config)
    rows.append({
        "contrast": "overall_vs_zero", "estimate": estimate,
        "bootstrap_90_low": low, "bootstrap_90_high": high,
        "usable_dates": int(len(daily)),
    })
    return pd.DataFrame(rows)


def _bootstrap_complete_case(
    daily: pd.DataFrame,
    positive: str,
    base: str,
    name: str,
    config: BullishDivergenceV2EventStudyConfig,
) -> dict:
    if positive not in daily or base not in daily:
        return {
            "contrast": name, "estimate": np.nan,
            "bootstrap_90_low": np.nan, "bootstrap_90_high": np.nan,
            "usable_dates": 0,
        }
    difference = (daily[positive] - daily[base]).dropna()
    estimate, low, high = _moving_block_bootstrap(difference.to_numpy(), config)
    return {
        "contrast": name, "estimate": estimate,
        "bootstrap_90_low": low, "bootstrap_90_high": high,
        "usable_dates": int(len(difference)),
    }


def _moving_block_bootstrap(
    values: np.ndarray,
    config: BullishDivergenceV2EventStudyConfig,
) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    estimate = float(values.mean())
    n = len(values)
    block = min(config.block_length, n)
    starts = np.arange(max(n - block + 1, 1))
    rng = np.random.default_rng(config.bootstrap_seed)
    samples = np.empty(config.bootstrap_samples)
    for sample in range(config.bootstrap_samples):
        indices: list[int] = []
        while len(indices) < n:
            start = int(rng.choice(starts))
            indices.extend(range(start, min(start + block, n)))
        samples[sample] = values[np.asarray(indices[:n])].mean()
    return estimate, float(np.quantile(samples, .05)), float(np.quantile(samples, .95))


def _group_summary(
    events: pd.DataFrame,
    paired: pd.DataFrame,
    field: str,
    raw_label: str,
) -> pd.DataFrame:
    rows = []
    for value, group in events.groupby(field, observed=True, dropna=False):
        raw = group[raw_label].dropna()
        matched = paired.loc[paired[field].eq(value), "paired_excess_10d"].dropna()
        rows.append({
            field: value,
            "events": int(len(raw)),
            "dates": int(group.loc[raw.index, "trade_date"].nunique()),
            "mean_industry_excess_net": float(raw.mean()) if len(raw) else np.nan,
            "matched_events": int(len(matched)),
            "mean_paired_excess": float(matched.mean()) if len(matched) else np.nan,
        })
    return pd.DataFrame(rows)


def _score_quintile(rank: pd.Series) -> pd.Series:
    return pd.cut(
        rank, bins=[0, .2, .4, .6, .8, 1.0],
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"], include_lowest=True,
    ).astype("string")


def _robust_location_scale(
    group: pd.DataFrame,
    fields: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
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
