from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from factor_forge.research.concept_etf_shadow import (
    monthly_performance,
    nonoverlap_sleeve_statistics,
    nonoverlapping_holding_periods,
    simulate_staggered_sleeves,
)


START = "2025-07-01"
END = "2026-07-14"
SCENARIOS = {
    "R1_base": {"variant": "R1_staggered_momentum", "universe": "all", "excluded": set()},
    "R4_base": {"variant": "R4_rank_buffer", "universe": "all", "excluded": set()},
    "R4_exclude_cpo": {"variant": "R4_rank_buffer", "universe": "all", "excluded": {"515880.SH"}},
    "R4_no_proxy": {"variant": "R4_rank_buffer", "universe": "no_proxy", "excluded": set()},
}


def main() -> None:
    args = parse_args()
    panel_path = resolve_panel(Path(args.signal_panel))
    panel = pd.read_parquet(panel_path)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    summary_rows, sleeve_stats_parts, paired_parts = [], [], []
    daily_parts, period_parts, attribution_parts, monthly_parts = [], [], [], []
    for cost in (20, 40):
        for scenario, specification in SCENARIOS.items():
            print(f"{scenario} at {cost}bps", flush=True)
            strategy_daily, strategy_sleeves, attribution = simulate_staggered_sleeves(
                panel, specification["variant"], start=START, end=END,
                roundtrip_cost_bps=cost, universe=specification["universe"],
                excluded_etfs=specification["excluded"],
            )
            benchmark_daily, benchmark_sleeves, _ = simulate_staggered_sleeves(
                panel, "S0_equal_weight", start=START, end=END,
                roundtrip_cost_bps=cost, universe=specification["universe"],
                excluded_etfs=specification["excluded"],
            )
            strategy_daily["portfolio"] = scenario
            benchmark_name = f"{scenario}__equal_weight"
            benchmark_daily["portfolio"] = benchmark_name
            for frame in (strategy_sleeves, benchmark_sleeves):
                frame["roundtrip_cost_bps"] = cost
            strategy_sleeves["scenario"] = scenario
            benchmark_sleeves["scenario"] = benchmark_name
            strategy_periods = nonoverlapping_holding_periods(strategy_sleeves)
            benchmark_periods = nonoverlapping_holding_periods(benchmark_sleeves)
            sleeve_stats, paired = nonoverlap_sleeve_statistics(strategy_periods, benchmark_periods)
            sleeve_stats["scenario"] = scenario
            sleeve_stats["roundtrip_cost_bps"] = cost
            paired["scenario"] = scenario
            paired["roundtrip_cost_bps"] = cost
            attribution["scenario"] = scenario
            attribution["roundtrip_cost_bps"] = cost
            combined_daily = pd.concat([strategy_daily, benchmark_daily], ignore_index=True)
            scenario_monthly = monthly_performance(combined_daily, benchmark_portfolio=benchmark_name)
            scenario_monthly["scenario"] = scenario
            scenario_monthly["roundtrip_cost_bps"] = cost
            summary_rows.append(summarize(
                scenario, cost, strategy_daily, benchmark_daily, scenario_monthly,
                sleeve_stats, attribution,
            ))
            daily_parts.append(combined_daily.assign(scenario=scenario, roundtrip_cost_bps=cost))
            period_parts.extend([strategy_periods, benchmark_periods])
            sleeve_stats_parts.append(sleeve_stats)
            paired_parts.append(paired)
            attribution_parts.append(attribution)
            monthly_parts.append(scenario_monthly)

    summary = pd.DataFrame(summary_rows)
    sleeve_stats = pd.concat(sleeve_stats_parts, ignore_index=True)
    paired = pd.concat(paired_parts, ignore_index=True)
    periods = pd.concat(period_parts, ignore_index=True)
    attribution = pd.concat(attribution_parts, ignore_index=True)
    monthly = pd.concat(monthly_parts, ignore_index=True)
    daily = pd.concat(daily_parts, ignore_index=True)
    decision = make_decision(summary)

    run_id = datetime.now(timezone.utc).strftime("concept_etf_nonoverlap_%Y%m%dT%H%M%SZ")
    output = Path(args.output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    sleeve_stats.to_csv(output / "nonoverlap_sleeve_statistics.csv", index=False, encoding="utf-8-sig")
    paired.to_parquet(output / "paired_nonoverlap_periods.parquet", index=False)
    periods.to_parquet(output / "all_nonoverlap_periods.parquet", index=False)
    daily.to_parquet(output / "scenario_daily_nav.parquet", index=False)
    monthly.to_csv(output / "monthly_performance.csv", index=False, encoding="utf-8-sig")
    attribution.to_csv(output / "profit_attribution.csv", index=False, encoding="utf-8-sig")
    (output / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "research_report.md").write_text(
        render_report(summary, sleeve_stats, monthly, decision), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(output), "decision": decision}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Non-overlapping five-sleeve ETF validation")
    parser.add_argument("--signal-panel", default="artifacts/concept_etf_rotation")
    parser.add_argument("--output-root", default="artifacts/concept_etf_nonoverlap")
    return parser.parse_args()


def resolve_panel(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = list(path.glob("concept_etf_rotation_*/etf_signal_panel.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no ETF signal panel below {path}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def summarize(scenario, cost, strategy, benchmark, monthly, sleeve_stats, attribution) -> dict:
    strategy = strategy.sort_values("return_date")
    benchmark = benchmark.sort_values("return_date")
    drawdown = strategy["net_nav"] / strategy["net_nav"].cummax().clip(lower=1.0) - 1
    strategy_monthly = monthly.loc[monthly["portfolio"].eq(scenario)]
    return {
        "scenario": scenario, "roundtrip_cost_bps": cost,
        "total_return": float(strategy["net_nav"].iloc[-1] - 1),
        "benchmark_total_return": float(benchmark["net_nav"].iloc[-1] - 1),
        "total_return_excess": float(strategy["net_nav"].iloc[-1] - benchmark["net_nav"].iloc[-1]),
        "maximum_drawdown": float(drawdown.min()),
        "worst_month": float(strategy_monthly["monthly_return"].min()),
        "mean_monthly_excess": float(strategy_monthly["monthly_excess_vs_p0"].mean()),
        "positive_sleeves": int(sleeve_stats["mean_net_excess"].gt(0).sum()),
        "minimum_sleeve_mean_excess": float(sleeve_stats["mean_net_excess"].min()),
        "median_sleeve_mean_excess": float(sleeve_stats["mean_net_excess"].median()),
        "minimum_sleeve_t": float(sleeve_stats["net_excess_t"].min()),
        "median_sleeve_t": float(sleeve_stats["net_excess_t"].median()),
        "sleeves_with_positive_bootstrap_low": int(sleeve_stats["bootstrap_95_low"].gt(0).sum()),
        "maximum_positive_profit_share": float(attribution["positive_profit_share"].max()),
        "mean_turnover": float(sleeve_stats["mean_turnover"].mean()),
    }


def make_decision(summary: pd.DataFrame) -> dict:
    twenty = summary.loc[summary["roundtrip_cost_bps"].eq(20)].set_index("scenario")
    rows = []
    for scenario in SCENARIOS:
        row = twenty.loc[scenario]
        checks = {
            "aggregate_excess_positive": bool(row["total_return_excess"] > 0),
            "all_sleeves_positive": bool(row["positive_sleeves"] == 5),
            "maximum_drawdown_within_20pct": bool(row["maximum_drawdown"] >= -0.20),
            "maximum_profit_share_within_30pct": bool(row["maximum_positive_profit_share"] <= 0.30),
            "at_least_one_sleeve_ci_above_zero": bool(row["sleeves_with_positive_bootstrap_low"] >= 1),
        }
        rows.append({"scenario": scenario, "passed": all(checks.values()), "checks": checks})
    passed = [row["scenario"] for row in rows if row["passed"]]
    return {
        "verdict": "NONOVERLAP_VALIDATION_PASSED" if passed else "NONOVERLAP_VALIDATION_FAILED",
        "passed_scenarios": passed, "scenario_checks": rows,
        "statistical_rule": "Each sleeve is tested separately; observations are not pooled across offsets.",
        "not_forward_confirmation": True,
    }


def render_report(summary, sleeve_stats, monthly, decision) -> str:
    display = summary.loc[summary["roundtrip_cost_bps"].eq(20)].copy()
    for column in (
        "total_return", "benchmark_total_return", "total_return_excess", "maximum_drawdown",
        "worst_month", "mean_monthly_excess", "minimum_sleeve_mean_excess",
        "median_sleeve_mean_excess", "maximum_positive_profit_share", "mean_turnover",
    ):
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    sleeves = sleeve_stats.loc[sleeve_stats["roundtrip_cost_bps"].eq(20)].copy()
    for column in (
        "mean_net_return", "mean_net_excess", "bootstrap_95_low", "bootstrap_95_high",
        "positive_period_rate", "total_net_return", "maximum_drawdown", "mean_turnover",
    ):
        sleeves[column] = sleeves[column].map(lambda value: f"{value:.2%}")
    returns = monthly.loc[
        monthly["roundtrip_cost_bps"].eq(20)
        & monthly["portfolio"].isin(SCENARIOS)
    ].pivot(index="month", columns="portfolio", values="monthly_return").map(
        lambda value: f"{value:.2%}"
    )
    return f"""# 五袖非重叠持仓与集中度反证

## 结论

**{decision['verdict']}**。每个袖子独立使用T+1开盘进入、持有5个交易日、下一周期再平衡；袖间结果不合并为一个扩大样本的t检验。

## 场景汇总（20bps）

{display.to_markdown(index=False)}

## 每袖独立统计（20bps）

{sleeves.to_markdown(index=False)}

## 月度收益（组合展示，不用于扩大非重叠样本量）

{returns.to_markdown()}

集中度反证只做预先指定的三种R4场景：原版、剔除通信ETF、剔除全部proxy映射；不在结果出来后继续搜索更多剔除组合。
"""


if __name__ == "__main__":
    main()
