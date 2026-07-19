from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from factor_forge.research.concept_etf_shadow import (
    STAGGERED_VARIANTS,
    monthly_performance,
    simulate_staggered_sleeves,
    simulate_weekly_daily_nav,
)


START = "2025-07-01"
END = "2026-07-14"


def main() -> None:
    args = parse_args()
    panel_path = resolve_panel(Path(args.signal_panel))
    panel = pd.read_parquet(panel_path)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    aggregate_parts, sleeve_parts, attribution_parts, monthly_parts, summary_rows = [], [], [], [], []

    for cost in (20, 40):
        print(f"running staggered R1-R4 at {cost}bps", flush=True)
        cost_aggregates, cost_sleeves, cost_attributions = [], [], []
        for variant in STAGGERED_VARIANTS:
            aggregate, sleeves, attribution = simulate_staggered_sleeves(
                panel, variant, start=START, end=END, roundtrip_cost_bps=cost,
            )
            aggregate["roundtrip_cost_bps"] = cost
            sleeves["roundtrip_cost_bps"] = cost
            attribution["roundtrip_cost_bps"] = cost
            cost_aggregates.append(aggregate)
            cost_sleeves.append(sleeves)
            cost_attributions.append(attribution)
        cost_daily = pd.concat(cost_aggregates, ignore_index=True)
        cost_sleeve_daily = pd.concat(cost_sleeves, ignore_index=True)
        cost_attribution = pd.concat(cost_attributions, ignore_index=True)
        cost_monthly = monthly_performance(cost_daily, benchmark_portfolio="S0_equal_weight")
        cost_monthly["roundtrip_cost_bps"] = cost
        sleeve_result = sleeve_comparison(cost_sleeve_daily)
        for variant in STAGGERED_VARIANTS:
            summary_rows.append(summarize_variant(
                cost_daily, cost_monthly, cost_attribution, sleeve_result, variant, cost,
            ))
        aggregate_parts.append(cost_daily)
        sleeve_parts.append(cost_sleeve_daily)
        attribution_parts.append(cost_attribution)
        monthly_parts.append(cost_monthly)

    fixed_rows = []
    for cost in (20, 40):
        for portfolio, output_name in (
            ("P0_equal_weight", "R0_fixed_equal_weight"),
            ("P1_etf_momentum", "R0_fixed_weekly_momentum"),
        ):
            daily = simulate_weekly_daily_nav(
                panel, portfolio, start=START, end=END, top_n=3, roundtrip_cost_bps=cost,
            )
            daily["portfolio"] = output_name
            daily["roundtrip_cost_bps"] = cost
            fixed_rows.append(daily)
    fixed_daily = pd.concat(fixed_rows, ignore_index=True)
    fixed_monthly_parts = []
    for cost, group in fixed_daily.groupby("roundtrip_cost_bps"):
        monthly = monthly_performance(group, benchmark_portfolio="R0_fixed_equal_weight")
        monthly["roundtrip_cost_bps"] = cost
        fixed_monthly_parts.append(monthly)
    fixed_monthly = pd.concat(fixed_monthly_parts, ignore_index=True)

    aggregate = pd.concat(aggregate_parts, ignore_index=True)
    sleeves = pd.concat(sleeve_parts, ignore_index=True)
    attribution = pd.concat(attribution_parts, ignore_index=True)
    monthly = pd.concat(monthly_parts, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    decision = make_decision(summary)

    run_id = datetime.now(timezone.utc).strftime("concept_etf_staggered_%Y%m%dT%H%M%SZ")
    output = Path(args.output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    aggregate.to_parquet(output / "staggered_daily_nav.parquet", index=False)
    sleeves.to_parquet(output / "sleeve_daily_nav.parquet", index=False)
    monthly.to_csv(output / "monthly_performance.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    attribution.to_csv(output / "profit_attribution.csv", index=False, encoding="utf-8-sig")
    fixed_daily.to_parquet(output / "r0_fixed_weekly_daily_nav.parquet", index=False)
    fixed_monthly.to_csv(output / "r0_fixed_weekly_monthly.csv", index=False, encoding="utf-8-sig")
    write_wide_tables(monthly, output)
    (output / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "research_report.md").write_text(
        render_report(summary, monthly, fixed_monthly, decision), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(output), "decision": decision}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R1-R4 staggered ETF momentum optimization")
    parser.add_argument("--signal-panel", default="artifacts/concept_etf_rotation")
    parser.add_argument("--output-root", default="artifacts/concept_etf_staggered")
    return parser.parse_args()


def resolve_panel(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = list(path.glob("concept_etf_rotation_*/etf_signal_panel.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no ETF signal panel below {path}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def sleeve_comparison(sleeves: pd.DataFrame) -> pd.DataFrame:
    final = sleeves.sort_values("return_date").groupby(["variant", "sleeve"], as_index=False).tail(1)
    final["sleeve_total_return"] = final["net_nav"] - 1
    benchmark = final.loc[final["variant"].eq("S0_equal_weight"), ["sleeve", "sleeve_total_return"]].rename(
        columns={"sleeve_total_return": "benchmark_sleeve_return"}
    )
    final = final.merge(benchmark, on="sleeve", how="left", validate="many_to_one")
    final["sleeve_excess"] = final["sleeve_total_return"] - final["benchmark_sleeve_return"]
    return final


def summarize_variant(daily, monthly, attribution, sleeve_result, variant, cost) -> dict:
    frame = daily.loc[daily["portfolio"].eq(variant)].sort_values("return_date")
    months = monthly.loc[monthly["portfolio"].eq(variant)]
    drawdown = frame["net_nav"] / frame["net_nav"].cummax().clip(lower=1.0) - 1
    contributions = attribution.loc[attribution["variant"].eq(variant)]
    variant_sleeves = sleeve_result.loc[sleeve_result["variant"].eq(variant)]
    return {
        "variant": variant, "roundtrip_cost_bps": cost,
        "total_return": float(frame["net_nav"].iloc[-1] - 1),
        "maximum_drawdown": float(drawdown.min()),
        "worst_month": float(months["monthly_return"].min()),
        "mean_monthly_return": float(months["monthly_return"].mean()),
        "mean_monthly_excess_vs_s0": float(months["monthly_excess_vs_p0"].mean()),
        "positive_months": int(months["monthly_return"].gt(0).sum()), "months": len(months),
        "total_turnover": float(frame["turnover"].sum()),
        "mean_cash_weight": float(frame["cash_weight"].mean()),
        "positive_sleeves": int(variant_sleeves["sleeve_excess"].gt(0).sum()),
        "minimum_sleeve_excess": float(variant_sleeves["sleeve_excess"].min()),
        "maximum_positive_profit_share": float(contributions["positive_profit_share"].max()),
    }


def make_decision(summary: pd.DataFrame) -> dict:
    twenty = summary.loc[summary["roundtrip_cost_bps"].eq(20)].set_index("variant")
    forty = summary.loc[summary["roundtrip_cost_bps"].eq(40)].set_index("variant")
    r1_turnover = twenty.loc["R1_staggered_momentum", "total_turnover"]
    rows = []
    for variant in STAGGERED_VARIANTS[1:]:
        row = twenty.loc[variant]
        passes = {
            "positive_monthly_excess": bool(row["mean_monthly_excess_vs_s0"] > 0),
            "maximum_drawdown_within_20pct": bool(row["maximum_drawdown"] >= -0.20),
            "worst_month_within_15pct": bool(row["worst_month"] >= -0.15),
            "positive_40bps_excess": bool(forty.loc[variant, "mean_monthly_excess_vs_s0"] > 0),
            "all_five_sleeves_positive": bool(row["positive_sleeves"] == 5),
            "profit_contribution_within_30pct": bool(row["maximum_positive_profit_share"] <= 0.30),
            "turnover_not_above_r1": bool(row["total_turnover"] <= r1_turnover + 1e-12),
        }
        rows.append({"variant": variant, "passed": all(passes.values()), "checks": passes})
    passed = [row["variant"] for row in rows if row["passed"]]
    return {
        "verdict": "HISTORICAL_RISK_GATE_PASSED" if passed else "NO_VARIANT_PASSES_HISTORICAL_RISK_GATE",
        "passed_variants": passed, "checks": rows,
        "not_forward_confirmation": True,
        "important": "All variants were evaluated on an already inspected interval.",
    }


def write_wide_tables(monthly: pd.DataFrame, output: Path) -> None:
    sample = monthly.loc[monthly["roundtrip_cost_bps"].eq(20)]
    for column, name in (
        ("monthly_return", "monthly_returns_20bps.csv"),
        ("monthly_max_drawdown", "monthly_drawdowns_20bps.csv"),
        ("monthly_excess_vs_p0", "monthly_excess_20bps.csv"),
    ):
        sample.pivot(index="month", columns="portfolio", values=column).to_csv(
            output / name, encoding="utf-8-sig"
        )


def render_report(summary, monthly, fixed_monthly, decision) -> str:
    display = summary.loc[summary["roundtrip_cost_bps"].eq(20)].copy()
    for column in (
        "total_return", "maximum_drawdown", "worst_month", "mean_monthly_return",
        "mean_monthly_excess_vs_s0", "mean_cash_weight", "minimum_sleeve_excess",
        "maximum_positive_profit_share",
    ):
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    returns = monthly.loc[monthly["roundtrip_cost_bps"].eq(20)].pivot(
        index="month", columns="portfolio", values="monthly_return"
    ).map(lambda value: f"{value:.2%}")
    drawdowns = monthly.loc[monthly["roundtrip_cost_bps"].eq(20)].pivot(
        index="month", columns="portfolio", values="monthly_max_drawdown"
    ).map(lambda value: f"{value:.2%}")
    fixed = fixed_monthly.loc[
        (fixed_monthly["roundtrip_cost_bps"].eq(20))
        & fixed_monthly["portfolio"].eq("R0_fixed_weekly_momentum")
    ]
    return f"""# 五袖错峰动量 R1-R4 消融实验

## 结论

**{decision['verdict']}**。本实验使用已经查看过的历史区间，只能作为风险排雷，不能作为前瞻确认。

R0固定周度动量的最差月为 {fixed['monthly_return'].min():.2%}，以下R1-R4逐项增加五袖错峰、绝对动量、波动率权重和排名缓冲。

## 20bps汇总

{display.to_markdown(index=False)}

## 每月收益

{returns.to_markdown()}

## 每月最大回撤

{drawdowns.to_markdown()}

准入要求同时包括：月均超额为正、最大回撤不超过20%、最差月不低于-15%、40bps仍有正超额、五袖全部为正、单ETF正贡献不超过30%、换手不高于R1。
"""


if __name__ == "__main__":
    main()
