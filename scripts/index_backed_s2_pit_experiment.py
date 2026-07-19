from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from concept_etf_latest_signal import load_incremental_stock_panel
from concept_etf_positive_diffusion_entry_experiment import (
    evaluation_periods,
    rebase_evaluation_daily,
    summarize_policy,
)
from concept_first_rotation_experiment import profit_diversity
from factor_forge.research.concept_etf_rotation import prepare_etf_panel
from factor_forge.research.concept_etf_shadow import (
    monthly_performance,
    nonoverlap_sleeve_statistics,
    simulate_staggered_sleeves,
)
from factor_forge.research.concept_first_rotation import (
    CONCEPT_FEATURES,
    build_concept_first_features,
)
from factor_forge.research.concept_rotation_alpha import build_concept_dataset
from factor_forge.research.concept_state_residual_rotation import (
    StateResidualRules,
    attach_state_residual_scores_to_etfs,
    fit_state_residual_walk_forward,
    s2_fold_train_test_diagnostics,
    within_state_oof_diagnostics,
)
from factor_forge.research.index_backed_rotation import (
    build_dynamic_etf_signal_panel,
    build_monthly_pit_etf_mapping,
    expand_monthly_index_membership,
)


POLICIES = {
    "S0_etf_r4": "score_S0_etf_r4",
    "S2_nonlinear_overlay": "score_S2_nonlinear_overlay",
}


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    snapshot = resolve_snapshot(Path(args.data_root), config["history_end"])
    candidates = pd.read_parquet(snapshot / "theme_etf_candidates.parquet")
    weights = pd.read_parquet(snapshot / "index_weights.parquet")
    daily = pd.read_parquet(snapshot / "fund_daily.parquet")
    share = pd.read_parquet(snapshot / "fund_share.parquet")
    basic = pd.read_parquet(snapshot / "etf_basic_all_statuses.parquet")

    print("loading stock history and expanding causal memberships", flush=True)
    stocks = load_incremental_stock_panel(
        [Path(path) for path in config["stock_panels"]],
        config["history_start"], config["history_end"],
    )
    calendar = pd.DatetimeIndex(sorted(stocks["trade_date"].unique()))
    index_metadata = candidates.drop_duplicates("index_code").rename(columns={
        "index_code": "concept_code", "index_name": "concept_name",
    })[["concept_code", "concept_name", "cluster"]]
    concept_index, members = expand_monthly_index_membership(
        weights, index_metadata, calendar,
        lag_sessions=int(config["data"]["membership_lag_sessions"]),
    )
    print(
        f"PIT concepts={concept_index['concept_code'].nunique()} "
        f"dates={concept_index['trade_date'].nunique()} members={len(members)}",
        flush=True,
    )
    _, concept_raw, feature_audit = build_concept_dataset(
        stocks, concept_index, members, breadth_weight_lag=1,
    )
    concepts = build_concept_first_features(concept_raw)
    history_days = int(concepts["trade_date"].nunique())
    if history_days < int(config["diagnostics"]["minimum_history_days"]):
        raise RuntimeError("PIT history gate failed")

    print("building prior-month-end ETF mapping schedule", flush=True)
    etf_metadata = basic.rename(columns={"csname": "name"})
    etfs = prepare_etf_panel(daily, share, pd.DataFrame(), etf_metadata)
    pit = config["pit_universe"]
    schedule, mapping_audit = build_monthly_pit_etf_mapping(
        etfs, candidates, concepts, calendar,
        minimum_listing_sessions=int(pit["minimum_listing_sessions"]),
        liquidity_window=int(pit["liquidity_window"]),
        minimum_liquidity_observations=int(pit["minimum_liquidity_observations"]),
        minimum_adv_cny=float(pit["minimum_adv_cny"]),
        minimum_aum_cny=float(pit["minimum_aum_cny"]),
        correlation_window=int(pit["correlation_window"]),
        minimum_correlation_observations=int(pit["minimum_correlation_observations"]),
        minimum_mapping_correlation=float(pit["minimum_mapping_correlation"]),
    )
    if schedule.empty:
        raise RuntimeError("monthly PIT mapping schedule is empty")
    panel = build_dynamic_etf_signal_panel(concept_raw, etfs, schedule, candidates)

    rules = rules_from_config(config)
    print("fitting unchanged S2 walk-forward model", flush=True)
    scores, _, _, fold_audit = fit_state_residual_walk_forward(
        concepts, start=str(concepts["trade_date"].min().date()),
        end=config["history_end"], rules=rules,
    )
    if scores.empty:
        raise RuntimeError("S2 walk-forward produced no scores")
    panel = attach_state_residual_scores_to_etfs(
        panel, scores, concept_overlay_weight=rules.concept_overlay_weight,
    )
    evaluation_start = max(
        pd.Timestamp(scores["trade_date"].min()),
        pd.Timestamp(panel.loc[panel["mapping_pass"], "trade_date"].min()),
    )
    print(f"PIT evaluation starts {evaluation_start.date()}", flush=True)

    summary, monthly, sleeve_stats, paired, nav, sleeves, attribution = run_backtests(
        panel, evaluation_start, config,
    )
    within_ic, within_buckets = within_state_oof_diagnostics(
        concepts, scores,
        policies={"R2_within_nonlinear_5d": "score_R2_within_nonlinear_5d"},
    )
    print("running R5-B sign-reversal diagnostics", flush=True)
    train_test = s2_fold_train_test_diagnostics(
        concepts, start=str(concepts["trade_date"].min().date()),
        end=config["history_end"], rules=rules,
    )
    group_diagnostics = s2_group_diagnostics(concepts, scores, index_metadata)
    feature_diagnostics = s2_feature_diagnostics(concepts, scores)
    fixed_comparison = compare_with_fixed_universe(nav, Path(args.fixed_artifact))
    universe_audit = summarize_universe(schedule, mapping_audit, evaluation_start)
    decision = make_decision(summary, within_ic, train_test)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_root) / f"pit_r5_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    concepts.to_parquet(output / "pit_concept_features.parquet", index=False)
    scores.to_parquet(output / "pit_oof_scores.parquet", index=False)
    panel.to_parquet(output / "pit_signal_panel.parquet", index=False)
    schedule.to_csv(output / "monthly_pit_mapping_schedule.csv", index=False, encoding="utf-8-sig")
    mapping_audit.to_parquet(output / "monthly_pit_mapping_audit.parquet", index=False)
    universe_audit.to_csv(output / "monthly_universe_audit.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_performance.csv", index=False, encoding="utf-8-sig")
    sleeve_stats.to_csv(output / "nonoverlap_sleeve_statistics.csv", index=False, encoding="utf-8-sig")
    paired.to_parquet(output / "paired_nonoverlap_periods.parquet", index=False)
    nav.to_parquet(output / "scenario_daily_nav.parquet", index=False)
    sleeves.to_parquet(output / "scenario_sleeve_daily.parquet", index=False)
    attribution.to_csv(output / "profit_attribution.csv", index=False, encoding="utf-8-sig")
    fold_audit.to_csv(output / "fold_audit.csv", index=False, encoding="utf-8-sig")
    within_ic.to_csv(output / "within_state_ic.csv", index=False, encoding="utf-8-sig")
    within_buckets.to_csv(output / "within_state_buckets.csv", index=False, encoding="utf-8-sig")
    train_test.to_csv(output / "s2_train_test_sign.csv", index=False, encoding="utf-8-sig")
    group_diagnostics.to_csv(output / "s2_group_diagnostics.csv", index=False, encoding="utf-8-sig")
    feature_diagnostics.to_csv(output / "s2_feature_diagnostics.csv", index=False, encoding="utf-8-sig")
    fixed_comparison.to_csv(output / "fixed_vs_pit_comparison.csv", index=False, encoding="utf-8-sig")
    data_audit = {
        "history_days": history_days,
        "history_start": str(concepts["trade_date"].min().date()),
        "history_end": str(concepts["trade_date"].max().date()),
        "concepts": int(concepts["concept_code"].nunique()),
        "candidate_etfs": int(candidates["ts_code"].nunique()),
        "mapping_months": int(schedule["effective_month"].nunique()),
        "selected_etfs_over_history": int(schedule["etf_code"].nunique()),
        "selected_indexes_over_history": int(schedule["concept_code"].nunique()),
        "evaluation_start": str(evaluation_start.date()),
        "evaluation_end": config["history_end"],
        "folds": int(fold_audit["fold"].nunique()),
        "feature_audit": feature_audit,
    }
    (output / "data_audit.json").write_text(
        json.dumps(data_audit, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "research_report.md").write_text(
        render_report(
            summary, fixed_comparison, monthly, within_ic, train_test,
            group_diagnostics, universe_audit, data_audit, decision,
        ), encoding="utf-8",
    )
    print(json.dumps({
        "run_dir": str(output), "data_audit": data_audit,
        "summary": summary.to_dict("records"), "decision": decision,
    }, ensure_ascii=False, indent=2, default=str))


def rules_from_config(config: dict) -> StateResidualRules:
    model = config["model"]
    return StateResidualRules(
        ridge_alpha=float(model["ridge_alpha"]),
        hgb_learning_rate=float(model["hgb_learning_rate"]),
        hgb_max_iter=int(model["hgb_max_iter"]),
        hgb_max_depth=int(model["hgb_max_depth"]),
        hgb_l2_regularization=float(model["hgb_l2_regularization"]),
        minimum_train_days=int(model["minimum_train_days"]),
        validation_days=int(model["validation_days"]),
        test_days=int(model["test_days"]),
        embargo_days=int(model["embargo_days"]),
        minimum_train_rows=int(model["minimum_train_rows"]),
        state_prior_weight=float(model["state_prior_weight"]),
        concept_overlay_weight=float(model["concept_overlay_weight"]),
        seed=int(model["seed"]),
    )


def run_backtests(panel, evaluation_start, config):
    summary_rows, monthly_parts, stat_parts, paired_parts = [], [], [], []
    nav_parts, sleeve_parts, attribution_parts = [], [], []
    for cost in config["execution"]["roundtrip_cost_bps"]:
        simulations = {}
        for policy, score_column in POLICIES.items():
            daily, sleeves, attribution = simulate_staggered_sleeves(
                panel, "R4_rank_buffer", start=str(evaluation_start.date()),
                end=config["history_end"], roundtrip_cost_bps=float(cost),
                score_column=score_column,
            )
            daily = rebase_evaluation_daily(daily, evaluation_start)
            daily["portfolio"] = policy
            daily["roundtrip_cost_bps"] = int(cost)
            sleeves["portfolio"] = policy
            sleeves["roundtrip_cost_bps"] = int(cost)
            attribution["policy"] = policy
            attribution["roundtrip_cost_bps"] = int(cost)
            simulations[policy] = (daily, sleeves, attribution)
            nav_parts.append(daily)
            sleeve_parts.append(sleeves)
            attribution_parts.append(attribution)
        baseline_periods = evaluation_periods(simulations["S0_etf_r4"][1], evaluation_start)
        for policy, (daily, sleeves, attribution) in simulations.items():
            periods = evaluation_periods(sleeves, evaluation_start)
            stats, paired = nonoverlap_sleeve_statistics(periods, baseline_periods)
            stats["policy"] = policy
            stats["roundtrip_cost_bps"] = int(cost)
            paired["policy"] = policy
            paired["roundtrip_cost_bps"] = int(cost)
            stat_parts.append(stats)
            paired_parts.append(paired)
            row = summarize_policy(
                policy, int(cost), daily, stats, attribution, evaluation_start,
            )
            row.update(profit_diversity(attribution, panel))
            summary_rows.append(row)
        month = monthly_performance(
            pd.concat([simulations[key][0] for key in POLICIES], ignore_index=True),
            benchmark_portfolio="S0_etf_r4",
        )
        month["roundtrip_cost_bps"] = int(cost)
        monthly_parts.append(month)
    return (
        pd.DataFrame(summary_rows), pd.concat(monthly_parts, ignore_index=True),
        pd.concat(stat_parts, ignore_index=True), pd.concat(paired_parts, ignore_index=True),
        pd.concat(nav_parts, ignore_index=True), pd.concat(sleeve_parts, ignore_index=True),
        pd.concat(attribution_parts, ignore_index=True),
    )


def s2_group_diagnostics(concepts, scores, metadata) -> pd.DataFrame:
    sample = scores[[
        "trade_date", "concept_code", "rrg_quadrant", "fold",
        "score_R2_within_nonlinear_5d",
    ]].merge(
        concepts[["trade_date", "concept_code", "forward_excess_5d"]],
        on=["trade_date", "concept_code"], how="left", validate="one_to_one",
    ).merge(metadata[["concept_code", "cluster"]], on="concept_code", how="left")
    sample["within_state_label"] = sample["forward_excess_5d"] - sample.groupby(
        ["trade_date", "rrg_quadrant"], observed=True,
    )["forward_excess_5d"].transform("mean")
    rows = []
    group_ic = state_date_ic(sample, "score_R2_within_nonlinear_5d")
    group_ic = group_ic.reset_index(name="rank_ic")
    group_ic["year"] = group_ic["trade_date"].dt.year
    for dimension, column in [("year", "year"), ("rrg_state", "rrg_quadrant")]:
        for value, group in group_ic.groupby(column, observed=True):
            rows.append({
                "dimension": dimension, "value": str(value), "groups": len(group),
                "mean_rank_ic": float(group["rank_ic"].mean()),
                "positive_rate": float(group["rank_ic"].gt(0).mean()),
            })
    for cluster, group in sample.dropna(subset=["within_state_label"]).groupby(
        "cluster", observed=True,
    ):
        rows.append({
            "dimension": "cluster", "value": str(cluster), "groups": len(group),
            "mean_rank_ic": float(group["score_R2_within_nonlinear_5d"].corr(
                group["within_state_label"], method="spearman",
            )),
            "positive_rate": np.nan,
        })
    return pd.DataFrame(rows)


def s2_feature_diagnostics(concepts, scores) -> pd.DataFrame:
    columns = ["trade_date", "concept_code", "rrg_quadrant", *CONCEPT_FEATURES]
    sample = scores[["trade_date", "concept_code"]].merge(
        concepts[columns + ["forward_excess_5d"]],
        on=["trade_date", "concept_code"], how="left", validate="one_to_one",
    )
    sample["within_state_label"] = sample["forward_excess_5d"] - sample.groupby(
        ["trade_date", "rrg_quadrant"], observed=True,
    )["forward_excess_5d"].transform("mean")
    rows = []
    for feature in CONCEPT_FEATURES:
        ic = state_date_ic(sample, feature)
        rows.append({
            "feature": feature, "state_date_groups": len(ic),
            "mean_univariate_rank_ic": float(ic.mean()),
            "positive_group_rate": float(ic.gt(0).mean()),
        })
    return pd.DataFrame(rows).sort_values("mean_univariate_rank_ic")


def state_date_ic(sample, score_column) -> pd.Series:
    return sample.dropna(subset=[score_column, "within_state_label"]).groupby(
        ["trade_date", "rrg_quadrant"], observed=True,
    ).apply(
        lambda group: (
            group[score_column].corr(group["within_state_label"], method="spearman")
            if len(group) >= 5 else np.nan
        ), include_groups=False,
    ).dropna()


def summarize_universe(schedule, audit, evaluation_start):
    selected = schedule.loc[schedule["effective_start"].ge(evaluation_start)].groupby(
        "effective_month", observed=True,
    ).agg(
        selected_indexes=("concept_code", "nunique"),
        selected_etfs=("etf_code", "nunique"),
        clusters=("cluster", "nunique"),
        mean_adv_cny=("adv_cny_pit", "mean"),
        mean_aum_cny=("aum_cny", "mean"),
        mean_mapping_correlation=("mapping_correlation_pit", "mean"),
    ).reset_index()
    eligible = audit.loc[
        audit["effective_start"].ge(evaluation_start) & audit["eligible_pit"]
    ].groupby("effective_month").size().rename("eligible_etf_candidates").reset_index()
    return selected.merge(eligible, on="effective_month", how="left")


def compare_with_fixed_universe(nav, artifact: Path) -> pd.DataFrame:
    if not artifact.exists():
        return pd.DataFrame()
    fixed = pd.read_csv(artifact)
    rows = []
    for item in fixed.itertuples(index=False):
        start = pd.Timestamp(item.start)
        frame = nav.loc[
            nav["portfolio"].eq(item.policy)
            & nav["roundtrip_cost_bps"].eq(int(item.roundtrip_cost_bps))
            & nav["return_date"].gt(start)
        ].sort_values("return_date")
        aligned_nav = (1 + frame["net_return"]).cumprod()
        drawdown = aligned_nav / aligned_nav.cummax().clip(lower=1.0) - 1
        rows.append({
            "policy": item.policy, "roundtrip_cost_bps": int(item.roundtrip_cost_bps),
            "aligned_start": str(start.date()),
            "pit_total_return": float(aligned_nav.iloc[-1] - 1),
            "pit_maximum_drawdown": float(drawdown.min()),
            "fixed_total_return": float(item.total_return),
            "fixed_maximum_drawdown": float(item.maximum_drawdown),
        })
    result = pd.DataFrame(rows)
    result["pit_minus_fixed_return"] = result["pit_total_return"] - result["fixed_total_return"]
    result["pit_minus_fixed_drawdown"] = (
        result["pit_maximum_drawdown"] - result["fixed_maximum_drawdown"]
    )
    return result


def make_decision(summary, within_ic, train_test):
    indexed = summary.set_index(["policy", "roundtrip_cost_bps"])
    excess = {
        str(cost): float(
            indexed.loc[("S2_nonlinear_overlay", cost), "total_return"]
            - indexed.loc[("S0_etf_r4", cost), "total_return"]
        ) for cost in (20, 40)
    }
    negative_horizons = int(within_ic["mean_within_state_rank_ic"].lt(0).sum())
    split = train_test.pivot(index="fold", columns="sample", values="mean_within_state_rank_ic")
    sign_flip_fraction = float(
        (split["train_in_sample"].gt(0) & split["test_oof"].lt(0)).mean()
    )
    checks = {
        "s2_nonnegative_ic_at_least_two_horizons": negative_horizons <= 1,
        "s2_train_test_sign_flip_fraction_below_half": sign_flip_fraction < 0.5,
    }
    for cost in (20, 40):
        candidate = indexed.loc[("S2_nonlinear_overlay", cost)]
        baseline = indexed.loc[("S0_etf_r4", cost)]
        checks[f"s2_positive_excess_{cost}bps"] = bool(excess[str(cost)] > 0)
        checks[f"s2_positive_sleeves_{cost}bps"] = int(candidate["positive_sleeves"]) >= 4
        checks[f"s2_drawdown_not_worse_{cost}bps"] = bool(
            candidate["maximum_drawdown"] >= baseline["maximum_drawdown"]
        )
        checks[f"s2_turnover_below_1_2x_{cost}bps"] = bool(
            candidate["mean_daily_turnover"] <= 1.2 * baseline["mean_daily_turnover"]
        )
        checks[f"s2_profit_concentration_below_30pct_{cost}bps"] = bool(
            candidate["maximum_positive_profit_share"] <= 0.30
        )
    passed = all(checks.values())
    return {
        "verdict": "KEEP_R4_REJECT_S2" if not passed else "S2_REQUIRES_NEW_FORWARD_CLOCK",
        "s2_passed": passed, "checks": checks, "s2_total_return_excess": excess,
        "negative_ic_horizons": negative_horizons,
        "train_positive_test_negative_fraction": sign_flip_fraction,
        "frozen_forward_experiment_changed": False,
        "sign_inversion_traded": False,
    }


def render_report(summary, fixed, monthly, within_ic, train_test, groups, universe, audit, decision):
    summary_display = format_percent(summary, [
        "total_return", "maximum_drawdown", "mean_daily_turnover", "mean_cash_weight",
        "minimum_sleeve_excess", "median_sleeve_excess", "maximum_positive_profit_share",
    ])
    fixed_display = format_percent(fixed, [
        "pit_total_return", "pit_maximum_drawdown", "fixed_total_return",
        "fixed_maximum_drawdown", "pit_minus_fixed_return", "pit_minus_fixed_drawdown",
    ])
    ic_display = format_percent(within_ic, ["mean_within_state_rank_ic", "positive_group_rate"])
    sign_summary = train_test.groupby("sample", as_index=False).agg(
        mean_rank_ic=("mean_within_state_rank_ic", "mean"),
        positive_fold_rate=("mean_within_state_rank_ic", lambda values: values.gt(0).mean()),
        folds=("fold", "nunique"),
    )
    sign_summary = format_percent(sign_summary, ["mean_rank_ic", "positive_fold_rate"])
    monthly20 = monthly.loc[monthly["roundtrip_cost_bps"].eq(20), [
        "month", "portfolio", "monthly_return", "monthly_max_drawdown", "monthly_excess_vs_p0",
    ]].copy()
    monthly20 = format_percent(monthly20, [
        "monthly_return", "monthly_max_drawdown", "monthly_excess_vs_p0",
    ])
    return f"""# R5-A PIT动态ETF宇宙与R5-B符号反转诊断

## 结论

**{decision['verdict']}**。本实验没有修改冻结S2，也没有交易反向分数。

- 历史：{audit['history_start']} 至 {audit['history_end']}，{audit['history_days']}个交易日
- 动态映射月份：{audit['mapping_months']}；历史实际入选ETF：{audit['selected_etfs_over_history']}
- OOF评估：{audit['evaluation_start']} 至 {audit['evaluation_end']}

## PIT回测汇总

{summary_display.to_markdown(index=False)}

## 固定宇宙与PIT宇宙差异

{fixed_display.to_markdown(index=False)}

## 状态内IC

{ic_display.to_markdown(index=False)}

## 训练与测试符号

{sign_summary.to_markdown(index=False)}

## 月度收益与月内回撤（20bp）

{monthly20.to_markdown(index=False)}

## 动态宇宙覆盖

{universe.to_markdown(index=False)}

## 分组诊断文件

年份、RRG状态和聚类诊断共{len(groups)}行，详见 `s2_group_diagnostics.csv`。
"""


def format_percent(frame, columns):
    result = frame.copy()
    for column in columns:
        if column in result:
            result[column] = result[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.2%}"
            )
    return result


def resolve_snapshot(root: Path, history_end: str) -> Path:
    candidates = []
    for path in root.glob("pit_index_backed_*/manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("history_end") == history_end:
            candidates.append((path.stat().st_mtime, path.parent))
    if not candidates:
        raise FileNotFoundError(f"no PIT snapshot ending {history_end}")
    return max(candidates)[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R5 PIT universe and S2 sign diagnostics")
    parser.add_argument("--config", default="configs/research/index_backed_s2_pit_v1.yaml")
    parser.add_argument("--data-root", default="data/index_backed_s2_pit")
    parser.add_argument("--output-root", default="artifacts/index_backed_s2_pit")
    parser.add_argument(
        "--fixed-artifact",
        default="artifacts/index_backed_s2_forward/frozen_s2_20260718T140943Z/development_historical_summary.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
