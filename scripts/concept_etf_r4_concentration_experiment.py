from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from concept_etf_positive_diffusion_entry_experiment import (
    evaluation_periods,
    rebase_evaluation_daily,
    summarize_policy,
)
from factor_forge.research.concept_etf_coordinated_r4 import (
    CoordinatedR4Rules,
    simulate_coordinated_r4,
)
from factor_forge.research.concept_etf_shadow import (
    monthly_performance,
    nonoverlap_sleeve_statistics,
)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rules = rules_from_config(config)
    panel = pd.read_parquet(config["signal_panel"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    evaluation_start = pd.Timestamp(config["start"])

    summary_rows: list[dict] = []
    monthly_parts: list[pd.DataFrame] = []
    stats_parts: list[pd.DataFrame] = []
    paired_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    sleeve_parts: list[pd.DataFrame] = []
    attribution_parts: list[pd.DataFrame] = []
    constraint_parts: list[pd.DataFrame] = []
    simulations_by_cost = {}
    for cost in config["roundtrip_cost_bps"]:
        simulations = {}
        for variant in config["variants"]:
            print(f"simulating {variant} at {cost}bps", flush=True)
            aggregate, sleeves, attribution, constraint_audit = simulate_coordinated_r4(
                panel,
                variant,
                start=config["start"],
                end=config["end"],
                roundtrip_cost_bps=float(cost),
                rules=rules,
            )
            aggregate["portfolio"] = variant
            aggregate["roundtrip_cost_bps"] = int(cost)
            sleeves["portfolio"] = variant
            sleeves["roundtrip_cost_bps"] = int(cost)
            attribution["policy"] = variant
            attribution["roundtrip_cost_bps"] = int(cost)
            constraint_audit["roundtrip_cost_bps"] = int(cost)
            simulations[variant] = (aggregate, sleeves, attribution, constraint_audit)
            daily_parts.append(aggregate)
            sleeve_parts.append(sleeves)
            attribution_parts.append(attribution)
            constraint_parts.append(constraint_audit)
        simulations_by_cost[int(cost)] = simulations
        baseline_periods = evaluation_periods(
            simulations["R4_A_base"][1], evaluation_start,
        )
        evaluation_daily = []
        for variant, (aggregate, sleeves, attribution, _) in simulations.items():
            daily = rebase_evaluation_daily(aggregate, evaluation_start)
            evaluation_daily.append(daily)
            periods = evaluation_periods(sleeves, evaluation_start)
            stats, paired = nonoverlap_sleeve_statistics(
                periods, baseline_periods, bootstrap_samples=2_000,
            )
            stats["policy"] = variant
            stats["roundtrip_cost_bps"] = int(cost)
            paired["policy"] = variant
            paired["roundtrip_cost_bps"] = int(cost)
            stats_parts.append(stats)
            paired_parts.append(paired)
            summary_rows.append(summarize_policy(
                variant, int(cost), daily, stats, attribution, evaluation_start,
            ))
        monthly = monthly_performance(
            pd.concat(evaluation_daily, ignore_index=True),
            benchmark_portfolio="R4_A_base",
        )
        monthly["roundtrip_cost_bps"] = int(cost)
        monthly_parts.append(monthly)

    print("running predefined CPO exclusion robustness", flush=True)
    cpo_robustness = run_cpo_robustness(panel, config, rules, evaluation_start)
    summary = pd.DataFrame(summary_rows)
    monthly = pd.concat(monthly_parts, ignore_index=True)
    sleeve_stats = pd.concat(stats_parts, ignore_index=True)
    paired = pd.concat(paired_parts, ignore_index=True)
    daily = pd.concat(daily_parts, ignore_index=True)
    sleeves = pd.concat(sleeve_parts, ignore_index=True)
    attribution = pd.concat(attribution_parts, ignore_index=True)
    constraint_audit = pd.concat(constraint_parts, ignore_index=True)
    constraint_summary = summarize_constraints(constraint_audit)
    decision = make_decision(
        summary, constraint_summary, cpo_robustness, config["acceptance"], rules,
    )
    audit = {
        "panel_start": str(panel["trade_date"].min().date()),
        "panel_end": str(panel["trade_date"].max().date()),
        "evaluation_start": config["start"],
        "evaluation_end": config["end"],
        "etfs": int(panel["ts_code"].nunique()),
        "constraints": config["constraints"],
        "base_reproduced_by_unit_test": True,
        "history_already_inspected": True,
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_root) / f"concept_etf_r4_concentration_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_performance.csv", index=False, encoding="utf-8-sig")
    sleeve_stats.to_csv(output / "nonoverlap_sleeve_statistics.csv", index=False, encoding="utf-8-sig")
    paired.to_parquet(output / "paired_nonoverlap_periods.parquet", index=False)
    daily.to_parquet(output / "scenario_daily_nav.parquet", index=False)
    sleeves.to_parquet(output / "scenario_sleeve_daily.parquet", index=False)
    attribution.to_csv(output / "profit_attribution.csv", index=False, encoding="utf-8-sig")
    constraint_audit.to_csv(output / "constraint_audit.csv", index=False, encoding="utf-8-sig")
    constraint_summary.to_csv(output / "constraint_summary.csv", index=False, encoding="utf-8-sig")
    cpo_robustness.to_csv(output / "exclude_cpo_robustness.csv", index=False, encoding="utf-8-sig")
    (output / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": config["experiment"],
        "status": config["status"],
        "config": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "signal_panel": str(Path(config["signal_panel"]).resolve()),
        "important": "Touched-history discovery only; passing starts a new forward clock and is not alpha confirmation.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "research_report.md").write_text(
        render_report(
            summary, monthly, sleeve_stats, attribution, constraint_summary,
            cpo_robustness, decision, audit,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "run_dir": str(output), "audit": audit, "decision": decision,
    }, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R4-B/R4-C coordinated sleeve experiment")
    parser.add_argument(
        "--config", default="configs/research/concept_etf_r4_concentration_v1.yaml",
    )
    parser.add_argument(
        "--output-root", default="artifacts/concept_etf_r4_concentration",
    )
    return parser.parse_args()


def rules_from_config(config: dict) -> CoordinatedR4Rules:
    values = config["constraints"]
    return CoordinatedR4Rules(
        maximum_sleeves_per_etf=int(values["maximum_sleeves_per_etf"]),
        maximum_aggregate_etf_weight=float(values["maximum_aggregate_etf_weight"]),
        maximum_aggregate_cluster_weight=float(values["maximum_aggregate_cluster_weight"]),
        correlation_window=int(values["correlation_window"]),
        correlation_minimum_observations=int(values["correlation_minimum_observations"]),
        maximum_pairwise_correlation=float(values["maximum_pairwise_correlation"]),
    )


def run_cpo_robustness(panel, config, rules, evaluation_start) -> pd.DataFrame:
    excluded = {str(config["robustness"]["excluded_cpo_etf"])}
    rows = []
    for cost in config["roundtrip_cost_bps"]:
        returns = {}
        drawdowns = {}
        for variant in config["variants"]:
            aggregate, _, _, _ = simulate_coordinated_r4(
                panel,
                variant,
                start=config["start"],
                end=config["end"],
                roundtrip_cost_bps=float(cost),
                excluded_etfs=excluded,
                rules=rules,
            )
            daily = rebase_evaluation_daily(aggregate, evaluation_start)
            returns[variant] = float(daily["net_nav"].iloc[-1] - 1)
            drawdowns[variant] = float(
                (daily["net_nav"] / daily["net_nav"].cummax().clip(lower=1.0) - 1).min()
            )
        for variant in config["primary_variants"]:
            rows.append({
                "variant": variant,
                "excluded_etf": next(iter(excluded)),
                "roundtrip_cost_bps": int(cost),
                "total_return": returns[variant],
                "R4_A_base_return": returns["R4_A_base"],
                "excess_vs_R4_A": returns[variant] - returns["R4_A_base"],
                "maximum_drawdown": drawdowns[variant],
                "R4_A_base_drawdown": drawdowns["R4_A_base"],
            })
    return pd.DataFrame(rows)


def summarize_constraints(audit: pd.DataFrame) -> pd.DataFrame:
    return audit.groupby(["variant", "roundtrip_cost_bps"], as_index=False).agg(
        maximum_aggregate_etf_weight=("maximum_aggregate_etf_weight", "max"),
        maximum_rebalanced_etf_weight=("maximum_rebalanced_etf_weight", "max"),
        maximum_aggregate_cluster_weight=("maximum_aggregate_cluster_weight", "max"),
        maximum_sleeve_frequency=("maximum_sleeve_frequency", "max"),
        maximum_entry_pairwise_correlation=("maximum_entry_pairwise_correlation", "max"),
        mean_target_cash_weight=("target_cash_weight", "mean"),
        mean_excluded_by_constraints=("excluded_by_constraints", "mean"),
    )


def make_decision(summary, constraints, cpo, acceptance, rules) -> dict:
    indexed = summary.set_index(["policy", "roundtrip_cost_bps"])
    constraint_index = constraints.set_index(["variant", "roundtrip_cost_bps"])
    cpo_index = cpo.set_index(["variant", "roundtrip_cost_bps"])
    variants = {}
    for variant in ("R4_B_concentration", "R4_C_correlation"):
        checks = {}
        for cost in (20, 40):
            candidate = indexed.loc[(variant, cost)]
            baseline = indexed.loc[("R4_A_base", cost)]
            constraint = constraint_index.loc[(variant, cost)]
            checks[f"positive_total_excess_{cost}bps"] = bool(
                candidate["total_return"] > baseline["total_return"]
            )
            checks[f"positive_sleeves_{cost}bps"] = bool(
                candidate["positive_sleeves"]
                >= int(acceptance["minimum_positive_sleeves_vs_r4a"])
            )
            checks[f"drawdown_not_worse_{cost}bps"] = bool(
                candidate["maximum_drawdown"] >= baseline["maximum_drawdown"]
                - float(acceptance["maximum_drawdown_deterioration"])
            )
            checks[f"profit_concentration_{cost}bps"] = bool(
                candidate["maximum_positive_profit_share"]
                <= float(acceptance["maximum_positive_profit_share"])
            )
            checks[f"turnover_within_limit_{cost}bps"] = bool(
                candidate["mean_daily_turnover"]
                <= baseline["mean_daily_turnover"]
                * float(acceptance["maximum_turnover_multiple"])
            )
            checks[f"etf_cap_enforced_{cost}bps"] = bool(
                constraint["maximum_rebalanced_etf_weight"]
                <= rules.maximum_aggregate_etf_weight + 1e-8
            )
            checks[f"frequency_cap_enforced_{cost}bps"] = bool(
                constraint["maximum_sleeve_frequency"]
                <= rules.maximum_sleeves_per_etf
            )
            checks[f"exclude_cpo_positive_excess_{cost}bps"] = bool(
                cpo_index.loc[(variant, cost), "excess_vs_R4_A"] > 0
            )
            if variant == "R4_C_correlation":
                checks[f"cluster_cap_enforced_{cost}bps"] = bool(
                    constraint["maximum_aggregate_cluster_weight"]
                    <= rules.maximum_aggregate_cluster_weight + 1e-8
                )
                maximum_correlation = constraint["maximum_entry_pairwise_correlation"]
                checks[f"correlation_cap_enforced_{cost}bps"] = bool(
                    pd.isna(maximum_correlation)
                    or maximum_correlation <= rules.maximum_pairwise_correlation + 1e-8
                )
        variants[variant] = {"passed": all(checks.values()), "checks": checks}
    passed = [variant for variant, result in variants.items() if result["passed"]]
    return {
        "verdict": "START_NEW_FORWARD_SHADOW_CLOCK" if passed else "R4_B_R4_C_NOT_VALIDATED",
        "passed_variants": passed,
        "variant_checks": variants,
        "not_alpha_confirmation": True,
        "history_already_inspected": True,
    }


def render_report(
    summary, monthly, sleeve_stats, attribution, constraints, cpo, decision, audit,
) -> str:
    summary_display = summary.copy()
    for column in [
        "total_return", "maximum_drawdown", "mean_daily_turnover", "mean_cash_weight",
        "minimum_sleeve_excess", "median_sleeve_excess", "maximum_positive_profit_share",
    ]:
        summary_display[column] = summary_display[column].map(lambda value: f"{value:.2%}")
    monthly_display = monthly.loc[monthly["roundtrip_cost_bps"].eq(20)].pivot(
        index="month", columns="portfolio", values="monthly_return",
    ).map(lambda value: f"{value:.2%}")
    stats_display = sleeve_stats.loc[sleeve_stats["roundtrip_cost_bps"].eq(20)].copy()
    for column in [
        "mean_net_return", "mean_net_excess", "bootstrap_95_low", "bootstrap_95_high",
        "positive_period_rate", "total_net_return", "maximum_drawdown", "mean_turnover",
    ]:
        stats_display[column] = stats_display[column].map(lambda value: f"{value:.2%}")
    attribution_display = attribution.loc[
        attribution["roundtrip_cost_bps"].eq(20)
    ].copy()
    for column in ["capital_contribution", "positive_profit_share"]:
        attribution_display[column] = attribution_display[column].map(lambda value: f"{value:.2%}")
    cpo_display = cpo.copy()
    for column in [
        "total_return", "R4_A_base_return", "excess_vs_R4_A",
        "maximum_drawdown", "R4_A_base_drawdown",
    ]:
        cpo_display[column] = cpo_display[column].map(lambda value: f"{value:.2%}")
    return f"""# R4-B / R4-C 跨袖套集中度实验

## 结论

**{decision['verdict']}**

- 区间：{audit['evaluation_start']} 至 {audit['evaluation_end']}
- R4-B：同一ETF最多3/5袖套，聚合目标权重不超过20%。
- R4-C：在R4-B基础上增加主题簇30%上限和60日相关系数0.75的新建仓限制。
- 本区间已经被查看，任何通过结果都只能启动新的前瞻影子时钟。

## 场景汇总

{summary_display.to_markdown(index=False)}

## 20bps月度收益

{monthly_display.to_markdown()}

## 20bps非重叠袖套

{stats_display.to_markdown(index=False)}

## 约束执行审计

{constraints.to_markdown(index=False)}

## 剔除CPO稳健性

{cpo_display.to_markdown(index=False)}

## 20bps利润贡献

{attribution_display.to_markdown(index=False)}

## 决策闸门

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```
"""


if __name__ == "__main__":
    main()
