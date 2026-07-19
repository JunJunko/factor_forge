from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from concept_etf_latest_signal import load_incremental_stock_panel
from factor_forge.research.concept_etf_rotation import prepare_etf_panel
from factor_forge.research.concept_etf_shadow import (
    monthly_performance,
    nonoverlapping_holding_periods,
    simulate_staggered_sleeves,
)
from factor_forge.research.concept_rotation_alpha import build_concept_dataset
from factor_forge.research.index_backed_rotation import (
    build_dynamic_etf_signal_panel,
    build_monthly_index_history_eligibility,
    build_monthly_pit_etf_mapping,
    expand_monthly_index_membership,
)


VARIANTS = ("strict_survivor_only", "strict_with_delisted")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    snapshot = resolve_snapshot(Path(args.data_root))
    candidates = pd.read_parquet(snapshot / "theme_etf_candidates.parquet")
    weights = pd.read_parquet(snapshot / "index_weights.parquet")
    daily = pd.read_parquet(snapshot / "fund_daily.parquet")
    share = pd.read_parquet(snapshot / "fund_share.parquet")
    basic = pd.read_parquet(snapshot / "etf_basic_all_statuses.parquet")

    print("loading stock history and building lagged index memberships", flush=True)
    stocks = load_incremental_stock_panel(
        [Path(path) for path in config["stock_panels"]],
        config["history_start"],
        config["history_end"],
    )
    calendar = pd.DatetimeIndex(sorted(stocks["trade_date"].unique()))
    index_metadata = candidates.drop_duplicates("index_code").rename(columns={
        "index_code": "concept_code", "index_name": "concept_name",
    })[["concept_code", "concept_name", "cluster"]]
    concept_index, members = expand_monthly_index_membership(
        weights,
        index_metadata,
        calendar,
        lag_sessions=int(config["data"]["membership_lag_sessions"]),
    )
    print(
        f"concepts={concept_index['concept_code'].nunique()} "
        f"member_rows={len(members)}",
        flush=True,
    )
    _, concepts, feature_audit = build_concept_dataset(
        stocks,
        concept_index,
        members,
        breadth_weight_lag=1,
    )
    history_days = int(concepts["trade_date"].nunique())
    if history_days < int(config["evaluation"]["minimum_history_days"]):
        raise RuntimeError(f"history gate failed: {history_days}")

    print("building per-selection-date index history eligibility", flush=True)
    history_eligibility = build_monthly_index_history_eligibility(
        weights,
        calendar,
        minimum_weight_months=int(config["data"]["minimum_weight_months_at_selection"]),
        minimum_members=int(config["data"]["minimum_index_members"]),
        availability_lag_sessions=int(
            config["data"]["index_weight_availability_lag_sessions"]
        ),
    )
    etfs = prepare_etf_panel(
        daily,
        share,
        pd.DataFrame(),
        basic.rename(columns={"csname": "name"}),
        share_availability_lag_sessions=int(
            config["data"]["share_availability_lag_sessions"]
        ),
    )

    panels, schedules, audits = {}, {}, {}
    for variant in VARIANTS:
        variant_candidates = candidates.copy()
        if variant == "strict_survivor_only":
            variant_candidates = variant_candidates.loc[
                variant_candidates["list_status"].eq("L")
            ].copy()
        print(f"building mapping {variant}", flush=True)
        schedule, audit = build_mapping(
            etfs,
            variant_candidates,
            concepts,
            calendar,
            history_eligibility,
            config,
        )
        if schedule.empty:
            raise RuntimeError(f"empty mapping schedule for {variant}")
        panels[variant] = build_dynamic_etf_signal_panel(
            concepts,
            etfs,
            schedule,
            variant_candidates,
        )
        schedules[variant], audits[variant] = schedule, audit

    evaluation_start = pd.Timestamp(config["evaluation"]["aligned_start"])
    summary, nav, sleeves, attribution = run_backtests(
        panels,
        evaluation_start,
        config,
    )
    primary_nav = nav.loc[nav["variant"].eq("strict_with_delisted")].copy()
    monthly_parts = []
    for cost, frame in primary_nav.groupby("roundtrip_cost_bps", observed=True):
        month = monthly_performance(frame, benchmark_portfolio="strict_with_delisted")
        month["roundtrip_cost_bps"] = int(cost)
        monthly_parts.append(month)
    monthly = pd.concat(monthly_parts, ignore_index=True)
    comparison = compare_canonical(summary, Path(args.canonical_artifact))
    coverage = mapping_coverage(schedules, history_eligibility, candidates, evaluation_start)
    prefix_check = prefix_causality_check(
        etfs,
        candidates,
        concepts,
        calendar,
        history_eligibility,
        schedules["strict_with_delisted"],
        config,
    )
    decision = make_decision(summary, schedules, audits, prefix_check, config)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_root) / f"strict_r7_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "canonical_comparison.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_performance.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output / "mapping_coverage.csv", index=False, encoding="utf-8-sig")
    history_eligibility.to_parquet(output / "index_history_eligibility.parquet", index=False)
    nav.to_parquet(output / "scenario_daily_nav.parquet", index=False)
    sleeves.to_parquet(output / "scenario_sleeve_daily.parquet", index=False)
    attribution.to_csv(output / "profit_attribution.csv", index=False, encoding="utf-8-sig")
    for variant in VARIANTS:
        schedules[variant].to_csv(
            output / f"{variant}_mapping_schedule.csv", index=False, encoding="utf-8-sig",
        )
        audits[variant].to_parquet(output / f"{variant}_mapping_audit.parquet", index=False)
    (output / "prefix_causality_check.json").write_text(
        json.dumps(prefix_check, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    data_audit = {
        "source_snapshot": str(snapshot.resolve()),
        "history_days": history_days,
        "history_start": str(concepts["trade_date"].min().date()),
        "history_end": str(concepts["trade_date"].max().date()),
        "concepts": int(concepts["concept_code"].nunique()),
        "candidate_etfs": int(candidates["ts_code"].nunique()),
        "recovered_delisted_candidates": int(candidates["list_status"].eq("D").sum()),
        "feature_audit": feature_audit,
        "share_availability_lag_sessions": int(
            config["data"]["share_availability_lag_sessions"]
        ),
        "index_weight_availability_lag_sessions": int(
            config["data"]["index_weight_availability_lag_sessions"]
        ),
    }
    (output / "data_audit.json").write_text(
        json.dumps(data_audit, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "research_report.md").write_text(
        render_report(summary, comparison, coverage, monthly, prefix_check, decision, data_audit),
        encoding="utf-8",
    )
    print(json.dumps({
        "run_dir": str(output),
        "summary": summary.to_dict("records"),
        "comparison": comparison.to_dict("records"),
        "decision": decision,
    }, ensure_ascii=False, indent=2, default=str))


def build_mapping(etfs, candidates, concepts, calendar, eligibility, config):
    pit = config["pit_universe"]
    return build_monthly_pit_etf_mapping(
        etfs,
        candidates,
        concepts,
        calendar,
        minimum_listing_sessions=int(pit["minimum_listing_sessions"]),
        liquidity_window=int(pit["liquidity_window"]),
        minimum_liquidity_observations=int(pit["minimum_liquidity_observations"]),
        minimum_adv_cny=float(pit["minimum_adv_cny"]),
        minimum_aum_cny=float(pit["minimum_aum_cny"]),
        correlation_window=int(pit["correlation_window"]),
        minimum_correlation_observations=int(pit["minimum_correlation_observations"]),
        minimum_mapping_correlation=float(pit["minimum_mapping_correlation"]),
        index_history_eligibility=eligibility,
    )


def run_backtests(panels, evaluation_start, config):
    summaries, navs, sleeve_parts, attribution_parts = [], [], [], []
    for variant, panel in panels.items():
        for cost in config["execution"]["roundtrip_cost_bps"]:
            daily, sleeves, attribution = simulate_staggered_sleeves(
                panel,
                "R4_rank_buffer",
                start=str(evaluation_start.date()),
                end=config["history_end"],
                roundtrip_cost_bps=float(cost),
                score_column="score_etf_momentum",
            )
            daily = daily.loc[daily["return_date"].gt(evaluation_start)].copy()
            daily["net_nav"] = (1 + daily["net_return"]).cumprod()
            daily["portfolio"] = variant
            daily["variant"] = variant
            daily["roundtrip_cost_bps"] = int(cost)
            sleeves["variant"] = variant
            sleeves["roundtrip_cost_bps"] = int(cost)
            attribution["variant"] = variant
            attribution["roundtrip_cost_bps"] = int(cost)
            summaries.append(summarize_variant(
                variant,
                int(cost),
                daily,
                sleeves,
                attribution,
                evaluation_start,
            ))
            navs.append(daily)
            sleeve_parts.append(sleeves)
            attribution_parts.append(attribution)
    return (
        pd.DataFrame(summaries),
        pd.concat(navs, ignore_index=True),
        pd.concat(sleeve_parts, ignore_index=True),
        pd.concat(attribution_parts, ignore_index=True),
    )


def summarize_variant(variant, cost, daily, sleeves, attribution, evaluation_start):
    drawdown = daily["net_nav"] / daily["net_nav"].cummax().clip(lower=1.0) - 1
    periods = nonoverlapping_holding_periods(sleeves)
    periods = periods.loc[periods["signal_date"].ge(evaluation_start)]
    sleeve_total = periods.groupby("sleeve", observed=True)["net_return"].apply(
        lambda values: float(np.prod(1 + values) - 1),
    )
    return {
        "variant": variant,
        "roundtrip_cost_bps": cost,
        "evaluation_start": evaluation_start,
        "evaluation_end": daily["return_date"].max(),
        "total_return": float(daily["net_nav"].iloc[-1] - 1),
        "maximum_drawdown": float(drawdown.min()),
        "mean_daily_turnover": float(daily["turnover"].mean()),
        "mean_cash_weight": float(daily["cash_weight"].mean()),
        "positive_sleeves": int(sleeve_total.gt(0).sum()),
        "minimum_sleeve_total_return": float(sleeve_total.min()),
        "median_sleeve_total_return": float(sleeve_total.median()),
        "maximum_positive_profit_share": float(attribution["positive_profit_share"].max()),
        "profit_etfs": int(attribution["ts_code"].nunique()),
    }


def compare_canonical(summary, canonical_artifact):
    canonical = pd.read_parquet(canonical_artifact)
    canonical = canonical.loc[
        canonical["portfolio"].eq("S0_etf_r4")
    ].copy()
    rows = []
    for cost in sorted(summary["roundtrip_cost_bps"].unique()):
        old = canonical.loc[canonical["roundtrip_cost_bps"].eq(cost)].sort_values("return_date")
        old = old.copy()
        old["aligned_nav"] = (1 + old["net_return"]).cumprod()
        old_drawdown = old["aligned_nav"] / old["aligned_nav"].cummax().clip(lower=1.0) - 1
        old_return = float(old["aligned_nav"].iloc[-1] - 1)
        old_mdd = float(old_drawdown.min())
        for item in summary.loc[summary["roundtrip_cost_bps"].eq(cost)].itertuples(index=False):
            rows.append({
                "roundtrip_cost_bps": int(cost),
                "variant": item.variant,
                "canonical_total_return": old_return,
                "strict_total_return": item.total_return,
                "strict_minus_canonical_return": item.total_return - old_return,
                "canonical_maximum_drawdown": old_mdd,
                "strict_maximum_drawdown": item.maximum_drawdown,
                "strict_minus_canonical_drawdown": item.maximum_drawdown - old_mdd,
            })
    return pd.DataFrame(rows)


def mapping_coverage(schedules, eligibility, candidates, evaluation_start):
    rows = []
    candidate_status = candidates.set_index("ts_code")["list_status"].to_dict()
    for variant, schedule in schedules.items():
        selected = schedule.loc[schedule["effective_start"].ge(evaluation_start)].copy()
        selected["list_status"] = selected["etf_code"].map(candidate_status)
        for month, group in selected.groupby("effective_month", observed=True):
            rows.append({
                "variant": variant,
                "effective_month": month,
                "selected_indexes": int(group["concept_code"].nunique()),
                "selected_etfs": int(group["etf_code"].nunique()),
                "selected_delisted_etfs": int(group["list_status"].eq("D").sum()),
                "clusters": int(group["cluster"].nunique()),
                "minimum_available_weight_months": int(
                    group["available_weight_months"].min()
                ),
                "mean_mapping_correlation": float(group["mapping_correlation_pit"].mean()),
            })
    result = pd.DataFrame(rows)
    result.attrs["eligible_rows"] = int(eligibility["index_history_pass"].sum())
    return result


def prefix_causality_check(
    etfs,
    candidates,
    concepts,
    calendar,
    eligibility,
    full_schedule,
    config,
):
    cutoff = pd.Timestamp("2025-06-30")
    future = calendar[calendar > cutoff]
    if future.empty:
        return {"passed": False, "reason": "no next session after cutoff"}
    next_session = pd.Timestamp(future[0])
    schedule, _ = build_mapping(
        etfs.loc[etfs["trade_date"].le(cutoff)],
        candidates,
        concepts.loc[concepts["trade_date"].le(cutoff)],
        calendar[calendar <= next_session],
        eligibility.loc[eligibility["selection_date"].le(cutoff)],
        config,
    )
    left = schedule.loc[schedule["selection_date"].eq(cutoff)].sort_values(
        ["concept_code", "etf_code"],
    ).reset_index(drop=True)
    right = full_schedule.loc[full_schedule["selection_date"].eq(cutoff)].sort_values(
        ["concept_code", "etf_code"],
    ).reset_index(drop=True)
    key_match = left[["concept_code", "etf_code"]].equals(
        right[["concept_code", "etf_code"]]
    )
    numeric_differences = {}
    for column in (
        "adv_cny_pit", "aum_cny", "mapping_correlation_pit", "available_weight_months",
    ):
        numeric_differences[column] = (
            float(np.nanmax(np.abs(left[column].to_numpy(float) - right[column].to_numpy(float))))
            if len(left) == len(right) and len(left) else np.nan
        )
    passed = bool(
        key_match
        and all(not np.isfinite(value) or value < 1e-4 for value in numeric_differences.values())
    )
    return {
        "cutoff": str(cutoff.date()),
        "truncated_rows": len(left),
        "full_rows": len(right),
        "key_match": key_match,
        "maximum_numeric_differences": numeric_differences,
        "passed": passed,
    }


def make_decision(summary, schedules, audits, prefix_check, config):
    primary = summary.loc[summary["variant"].eq("strict_with_delisted")].set_index(
        "roundtrip_cost_bps"
    )
    selected = schedules["strict_with_delisted"]
    audit = audits["strict_with_delisted"]
    audit_keys = set(map(tuple, audit.loc[
        audit["eligible_pit"], ["selection_date", "index_code", "ts_code"],
    ].to_numpy()))
    selected_keys = set(map(tuple, selected.rename(columns={
        "concept_code": "index_code", "etf_code": "ts_code",
    })[["selection_date", "index_code", "ts_code"]].to_numpy()))
    repair_checks = {
        "selected_index_history_gate_all_pass": bool(
            selected["available_weight_months"].ge(
                int(config["data"]["minimum_weight_months_at_selection"])
            ).all()
        ),
        "mapping_effective_after_selection": bool(
            selected["effective_start"].gt(selected["selection_date"]).all()
        ),
        "selected_candidates_are_pit_eligible": bool(
            selected_keys.issubset(audit_keys)
        ),
        "prefix_recomputation_identical": bool(prefix_check["passed"]),
        "share_lag_at_least_one_session": int(
            config["data"]["share_availability_lag_sessions"]
        ) >= 1,
        "no_fuzzy_delisted_mapping": not bool(
            config["repair_policy"]["fuzzy_mapping_allowed"]
        ),
    }
    strategy_checks = {
        "positive_total_return_20bps": bool(primary.loc[20, "total_return"] > 0),
        "positive_total_return_40bps": bool(primary.loc[40, "total_return"] > 0),
        "all_five_sleeves_positive_20bps": bool(primary.loc[20, "positive_sleeves"] == 5),
        "all_five_sleeves_positive_40bps": bool(primary.loc[40, "positive_sleeves"] == 5),
    }
    repair_passed = all(repair_checks.values())
    strategy_survived = all(strategy_checks.values())
    if repair_passed and strategy_survived:
        verdict = "STRICT_PIT_REPAIR_PASSED_R4_SURVIVES"
    elif repair_passed:
        verdict = "STRICT_PIT_REPAIR_PASSED_R4_NOT_ROBUST"
    else:
        verdict = "STRICT_PIT_REPAIR_FAILED"
    return {
        "verdict": verdict,
        "repair_passed": repair_passed,
        "strategy_survived": strategy_survived,
        "repair_checks": repair_checks,
        "strategy_checks": strategy_checks,
        "original_r4_parameters_changed": False,
        "parameter_tuning_performed": False,
        "s2_used": False,
        "not_real_money_authorization": True,
    }


def render_report(summary, comparison, coverage, monthly, prefix, decision, audit):
    summary_display = format_percent(summary.copy(), [
        "total_return", "maximum_drawdown", "mean_daily_turnover", "mean_cash_weight",
        "minimum_sleeve_total_return", "median_sleeve_total_return",
        "maximum_positive_profit_share",
    ])
    comparison_display = format_percent(comparison.copy(), [
        "canonical_total_return", "strict_total_return", "strict_minus_canonical_return",
        "canonical_maximum_drawdown", "strict_maximum_drawdown",
        "strict_minus_canonical_drawdown",
    ])
    coverage_summary = coverage.groupby("variant", as_index=False).agg(
        months=("effective_month", "nunique"),
        mean_indexes=("selected_indexes", "mean"),
        minimum_indexes=("selected_indexes", "min"),
        maximum_indexes=("selected_indexes", "max"),
        selected_delisted_months=("selected_delisted_etfs", lambda values: int(values.gt(0).sum())),
        minimum_weight_months=("minimum_available_weight_months", "min"),
        mean_mapping_correlation=("mean_mapping_correlation", "mean"),
    )
    coverage_summary = format_percent(coverage_summary, ["mean_mapping_correlation"])
    month20 = monthly.loc[monthly["roundtrip_cost_bps"].eq(20)]
    return f"""# R7 严格PIT修复实验

## 结论

**{decision['verdict']}**。修复过程没有调整原始5日R4参数，也没有使用S2或根据结果选择参数。

## 回测结果

{summary_display.to_markdown(index=False)}

## 与原PIT版本对照

{comparison_display.to_markdown(index=False)}

## 历史宇宙覆盖

{coverage_summary.to_markdown(index=False)}

- 指数权重历史：{audit['history_start']} 至 {audit['history_end']} 的板块信号，指数门槛在每个选择日重新计算。
- 候选ETF：{audit['candidate_etfs']}，其中恢复退市ETF {audit['recovered_delisted_candidates']}。
- 基金份额可用性滞后：{audit['share_availability_lag_sessions']}个交易日。
- 截断重算检查：{prefix['passed']}，截至{prefix['cutoff']}映射行数 {prefix['truncated_rows']}。

## 月度诊断（严格PIT含退市ETF，20bp）

- 月份数：{len(month20)}；正收益月份占比：{month20['monthly_return'].gt(0).mean():.1%}。
- 最差月收益：{month20['monthly_return'].min():.2%}；最佳月收益：{month20['monthly_return'].max():.2%}。
- 最差月内回撤：{month20['monthly_max_drawdown'].min():.2%}。

## 使用边界

本结果修复了已确认的全样本36个月筛选和可恢复退市ETF偏差。无法唯一映射的退市ETF仍被保守排除，因此结果是更严格的开发回测，而不是真实资金授权。
"""


def format_percent(frame, columns):
    result = frame.copy()
    for column in columns:
        result[column] = result[column].map(lambda value: f"{value:.2%}")
    return result


def resolve_snapshot(root: Path) -> Path:
    if root.is_file():
        raise ValueError("data root must be a directory")
    candidates = list(root.glob("strict_pit_*/manifest.json"))
    if not candidates:
        raise FileNotFoundError(f"no strict PIT snapshot below {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R7 strict PIT R4 repair experiment")
    parser.add_argument(
        "--config", default="configs/research/index_backed_r4_strict_pit_v1.yaml",
    )
    parser.add_argument("--data-root", default="data/index_backed_r4_strict_pit")
    parser.add_argument("--output-root", default="artifacts/index_backed_r4_strict_pit")
    parser.add_argument(
        "--canonical-artifact",
        default=(
            "artifacts/index_backed_s2_pit/pit_r5_20260718T144518Z/"
            "scenario_daily_nav.parquet"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
