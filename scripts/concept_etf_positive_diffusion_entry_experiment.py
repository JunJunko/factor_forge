from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from factor_forge.research.concept_etf_diffusion_entry import (
    DIFFUSION_FEATURES,
    DiffusionBlendRules,
    attach_learned_oof_score,
    attach_positive_diffusion_scores,
    fit_positive_diffusion_walk_forward,
)
from factor_forge.research.concept_etf_exit_ml import parse_weights
from factor_forge.research.concept_etf_shadow import (
    CASH,
    monthly_performance,
    nonoverlap_sleeve_statistics,
    nonoverlapping_holding_periods,
    simulate_staggered_sleeves,
)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    blend_rules = rules_from_config(config)
    panel = pd.read_parquet(config["signal_panel"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    print("building fixed positive-diffusion entry scores", flush=True)
    panel = attach_positive_diffusion_scores(
        panel, fixed_diffusion_weight=float(config["diffusion"]["fixed_weight"]),
    )
    print("fitting non-negative walk-forward diffusion weights", flush=True)
    predictions, learned_weights, fold_audit = fit_positive_diffusion_walk_forward(
        panel, start=config["start"], end=config["end"], rules=blend_rules,
    )
    if predictions.empty:
        raise RuntimeError("positive diffusion walk-forward produced no OOF predictions")
    panel = attach_learned_oof_score(panel, predictions)
    evaluation_start = pd.Timestamp(predictions["trade_date"].min())
    rank_ic = signal_rank_ic(panel, config["policies"], evaluation_start, config["end"])
    print(f"OOF evaluation starts {evaluation_start.date()}", flush=True)

    summary_rows: list[dict] = []
    monthly_parts: list[pd.DataFrame] = []
    stats_parts: list[pd.DataFrame] = []
    paired_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    sleeve_parts: list[pd.DataFrame] = []
    attribution_parts: list[pd.DataFrame] = []
    for cost in config["roundtrip_cost_bps"]:
        simulations = {}
        for policy, score_column in config["policies"].items():
            print(f"simulating {policy} at {cost}bps", flush=True)
            aggregate, sleeves, _ = simulate_staggered_sleeves(
                panel, "R4_rank_buffer", start=config["start"], end=config["end"],
                roundtrip_cost_bps=float(cost), score_column=score_column,
            )
            aggregate["portfolio"] = policy
            aggregate["roundtrip_cost_bps"] = int(cost)
            sleeves["portfolio"] = policy
            sleeves["variant"] = policy
            sleeves["roundtrip_cost_bps"] = int(cost)
            attribution = evaluation_attribution(panel, sleeves, evaluation_start)
            attribution["policy"] = policy
            attribution["roundtrip_cost_bps"] = int(cost)
            simulations[policy] = (aggregate, sleeves, attribution)
            daily_parts.append(aggregate)
            sleeve_parts.append(sleeves)
            attribution_parts.append(attribution)
        baseline_periods = evaluation_periods(simulations["B0_price"][1], evaluation_start)
        evaluation_daily = []
        for policy, (aggregate, sleeves, attribution) in simulations.items():
            daily = rebase_evaluation_daily(aggregate, evaluation_start)
            evaluation_daily.append(daily)
            periods = evaluation_periods(sleeves, evaluation_start)
            stats, paired = nonoverlap_sleeve_statistics(
                periods, baseline_periods, bootstrap_samples=2_000,
            )
            stats["policy"] = policy
            stats["roundtrip_cost_bps"] = int(cost)
            paired["policy"] = policy
            paired["roundtrip_cost_bps"] = int(cost)
            stats_parts.append(stats)
            paired_parts.append(paired)
            summary_rows.append(summarize_policy(
                policy, int(cost), daily, stats, attribution, evaluation_start,
            ))
        combined = pd.concat(evaluation_daily, ignore_index=True)
        monthly = monthly_performance(combined, benchmark_portfolio="B0_price")
        monthly["roundtrip_cost_bps"] = int(cost)
        monthly_parts.append(monthly)

    summary = pd.DataFrame(summary_rows)
    monthly = pd.concat(monthly_parts, ignore_index=True)
    sleeve_stats = pd.concat(stats_parts, ignore_index=True)
    paired = pd.concat(paired_parts, ignore_index=True)
    daily = pd.concat(daily_parts, ignore_index=True)
    sleeves = pd.concat(sleeve_parts, ignore_index=True)
    attribution = pd.concat(attribution_parts, ignore_index=True)
    weight_audit = audit_learned_weights(learned_weights, blend_rules)
    decision = make_decision(summary, weight_audit, config["acceptance"])
    audit = {
        "panel_start": str(panel["trade_date"].min().date()),
        "panel_end": str(panel["trade_date"].max().date()),
        "etfs": int(panel["ts_code"].nunique()),
        "oof_start": str(evaluation_start.date()),
        "oof_end": str(predictions["trade_date"].max().date()),
        "oof_prediction_rows": len(predictions),
        "folds_fitted": len(fold_audit),
        "fixed_diffusion_weight": float(config["diffusion"]["fixed_weight"]),
        "learned_weight_audit": weight_audit,
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_root) / f"concept_etf_positive_diffusion_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(output / "oof_learned_scores.parquet", index=False)
    learned_weights.to_csv(output / "learned_blend_weights.csv", index=False, encoding="utf-8-sig")
    fold_audit.to_csv(output / "fold_audit.csv", index=False, encoding="utf-8-sig")
    rank_ic.to_csv(output / "signal_rank_ic.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_performance.csv", index=False, encoding="utf-8-sig")
    sleeve_stats.to_csv(output / "nonoverlap_sleeve_statistics.csv", index=False, encoding="utf-8-sig")
    paired.to_parquet(output / "paired_nonoverlap_periods.parquet", index=False)
    daily.to_parquet(output / "scenario_daily_nav.parquet", index=False)
    sleeves.to_parquet(output / "scenario_sleeve_daily.parquet", index=False)
    attribution.to_csv(output / "profit_attribution.csv", index=False, encoding="utf-8-sig")
    (output / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": config["experiment"], "status": config["status"],
        "config": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "signal_panel": str(Path(config["signal_panel"]).resolve()),
        "evaluation_start": str(evaluation_start.date()), "evaluation_end": config["end"],
        "primary_policy": config["primary_policy"],
        "important": "Touched-history discovery only; no early exit and no live-trading authorization.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "research_report.md").write_text(
        render_report(summary, monthly, sleeve_stats, rank_ic, learned_weights, audit, decision, manifest),
        encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(output), "audit": audit, "decision": decision}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Positive diffusion entry blend for concept ETFs")
    parser.add_argument("--config", default="configs/research/concept_etf_positive_diffusion_entry_v1.yaml")
    parser.add_argument("--output-root", default="artifacts/concept_etf_positive_diffusion")
    return parser.parse_args()


def rules_from_config(config: dict) -> DiffusionBlendRules:
    learned = config["learned_blend"]
    return DiffusionBlendRules(
        fixed_diffusion_weight=float(config["diffusion"]["fixed_weight"]),
        maximum_learned_diffusion_weight=float(learned["maximum_diffusion_weight"]),
        ridge_alpha=float(learned["ridge_alpha"]),
        minimum_train_days=int(learned["minimum_train_days"]),
        validation_days=int(learned["validation_days"]),
        test_days=int(learned["test_days"]),
        embargo_days=int(learned["embargo_days"]),
        minimum_train_rows=int(learned["minimum_train_rows"]),
    )


def rebase_evaluation_daily(daily: pd.DataFrame, evaluation_start: pd.Timestamp) -> pd.DataFrame:
    result = daily.loc[daily["return_date"].gt(evaluation_start)].copy()
    result["net_nav"] = (1 + result["net_return"]).cumprod()
    return result


def evaluation_periods(sleeves: pd.DataFrame, evaluation_start: pd.Timestamp) -> pd.DataFrame:
    periods = nonoverlapping_holding_periods(sleeves)
    return periods.loc[periods["signal_date"].ge(evaluation_start)].reset_index(drop=True)


def evaluation_attribution(
    panel: pd.DataFrame,
    sleeves: pd.DataFrame,
    evaluation_start: pd.Timestamp,
) -> pd.DataFrame:
    prices = panel.pivot(index="trade_date", columns="ts_code", values="adj_open").sort_index()
    rows = []
    for sleeve_id, sleeve in sleeves.groupby("sleeve", observed=True):
        evaluation = sleeve.loc[sleeve["return_date"].gt(evaluation_start)].sort_values("holding_date")
        if evaluation.empty:
            continue
        base_nav = float(evaluation.iloc[0]["nav_before"])
        for item in evaluation.itertuples(index=False):
            weights = parse_weights(item.target_weights)
            normalized_nav = float(item.nav_before) / base_nav
            returns = prices.loc[item.return_date] / prices.loc[item.holding_date] - 1
            for code, weight in weights.items():
                if code == CASH or weight <= 0:
                    continue
                rows.append({
                    "ts_code": code,
                    "capital_contribution": normalized_nav * weight * float(returns[code]) / 5,
                    "sleeve": int(sleeve_id),
                })
    result = pd.DataFrame(rows).groupby("ts_code", as_index=False).agg(
        capital_contribution=("capital_contribution", "sum"),
        contribution_days=("capital_contribution", "size"),
    )
    positive_total = result["capital_contribution"].clip(lower=0).sum()
    result["positive_profit_share"] = (
        result["capital_contribution"].clip(lower=0) / positive_total if positive_total > 0 else 0.0
    )
    return result.sort_values("positive_profit_share", ascending=False)


def summarize_policy(policy, cost, daily, stats, attribution, evaluation_start) -> dict:
    drawdown = daily["net_nav"] / daily["net_nav"].cummax().clip(lower=1.0) - 1
    return {
        "policy": policy, "roundtrip_cost_bps": cost,
        "evaluation_start": evaluation_start, "evaluation_end": daily["return_date"].max(),
        "total_return": float(daily["net_nav"].iloc[-1] - 1),
        "maximum_drawdown": float(drawdown.min()),
        "mean_daily_turnover": float(daily["turnover"].mean()),
        "mean_cash_weight": float(daily["cash_weight"].mean()),
        "positive_sleeves": int(stats["mean_net_excess"].gt(0).sum()),
        "minimum_sleeve_excess": float(stats["mean_net_excess"].min()),
        "median_sleeve_excess": float(stats["mean_net_excess"].median()),
        "maximum_positive_profit_share": float(attribution["positive_profit_share"].max()),
    }


def signal_rank_ic(panel, policies, start, end) -> pd.DataFrame:
    frame = panel.loc[
        panel["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))
        & panel["forward_open_5d"].notna()
    ].copy()
    frame["forward_excess_5d"] = frame["forward_open_5d"] - frame.groupby(
        "trade_date", sort=False
    )["forward_open_5d"].transform("mean")
    rows = []
    for policy, column in policies.items():
        daily = frame.groupby("trade_date", observed=True).apply(
            lambda day: day[column].corr(day["forward_excess_5d"], method="spearman"),
            include_groups=False,
        ).dropna()
        rows.append({
            "policy": policy, "days": len(daily),
            "mean_rank_ic": float(daily.mean()),
            "positive_rank_ic_rate": float(daily.gt(0).mean()),
        })
    return pd.DataFrame(rows)


def audit_learned_weights(weights: pd.DataFrame, rules: DiffusionBlendRules) -> dict:
    diffusion = weights[DIFFUSION_FEATURES[1:]].sum(axis=1)
    boundary = diffusion.le(1e-8) | diffusion.ge(rules.maximum_learned_diffusion_weight - 1e-8)
    return {
        "folds": len(weights),
        "mean_diffusion_weight": float(diffusion.mean()),
        "median_diffusion_weight": float(diffusion.median()),
        "minimum_diffusion_weight": float(diffusion.min()),
        "maximum_diffusion_weight": float(diffusion.max()),
        "boundary_fraction": float(boundary.mean()),
    }


def make_decision(summary: pd.DataFrame, weight_audit: dict, acceptance: dict) -> dict:
    indexed = summary.set_index(["policy", "roundtrip_cost_bps"])
    variants = {}
    for policy in ("B1_fixed_diffusion", "B2_learned_diffusion"):
        checks = {}
        for cost in (20, 40):
            candidate = indexed.loc[(policy, cost)]
            baseline = indexed.loc[("B0_price", cost)]
            placebo = indexed.loc[("B3_placebo", cost)]
            checks[f"positive_total_excess_{cost}bps"] = bool(
                candidate["total_return"] > baseline["total_return"]
            )
            checks[f"positive_sleeves_{cost}bps"] = bool(
                candidate["positive_sleeves"] >= int(acceptance["minimum_positive_sleeves"])
            )
            checks[f"drawdown_not_worse_{cost}bps"] = bool(
                candidate["maximum_drawdown"] >= baseline["maximum_drawdown"]
                - float(acceptance["maximum_drawdown_deterioration"])
            )
            checks[f"profit_concentration_{cost}bps"] = bool(
                candidate["maximum_positive_profit_share"]
                <= float(acceptance["maximum_positive_profit_share"])
            )
            checks[f"beats_placebo_{cost}bps"] = bool(
                candidate["total_return"] > placebo["total_return"]
            )
        if policy == "B2_learned_diffusion":
            checks["learned_weight_not_boundary"] = bool(
                weight_audit["boundary_fraction"]
                <= float(acceptance["maximum_learned_boundary_fraction"])
            )
        variants[policy] = {"passed": all(checks.values()), "checks": checks}
    primary_passed = variants["B1_fixed_diffusion"]["passed"]
    return {
        "verdict": "KEEP_POSITIVE_DIFFUSION_FOR_FORWARD_SHADOW" if primary_passed else "POSITIVE_DIFFUSION_NOT_VALIDATED",
        "primary_policy": "B1_fixed_diffusion",
        "primary_passed": primary_passed,
        "variant_checks": variants,
        "not_forward_confirmation": True,
    }


def render_report(summary, monthly, sleeve_stats, rank_ic, weights, audit, decision, manifest) -> str:
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
    rank_display = rank_ic.copy()
    for column in ["mean_rank_ic", "positive_rank_ic_rate"]:
        rank_display[column] = rank_display[column].map(lambda value: f"{value:.2%}")
    return f"""# ETF价格动量加入正向扩散：非重叠走步实验

## 结论

**{decision['verdict']}**。B1固定使用80%价格动量＋20%正向扩散；B2仅使用非负约束并将扩散总权重限制在30%以内。所有方案固定持有5日，没有负向退出。

- OOF区间：{manifest['evaluation_start']} 至 {manifest['evaluation_end']}
- 固定扩散权重：{audit['fixed_diffusion_weight']:.2%}
- B2平均/中位扩散权重：{audit['learned_weight_audit']['mean_diffusion_weight']:.2%} / {audit['learned_weight_audit']['median_diffusion_weight']:.2%}
- B2权重触及边界比例：{audit['learned_weight_audit']['boundary_fraction']:.2%}

## 场景汇总

{summary_display.to_markdown(index=False)}

## 5日截面Rank IC

{rank_display.to_markdown(index=False)}

## 20bps月收益

{monthly_display.to_markdown()}

## 20bps每袖非重叠结果

{stats_display.to_markdown(index=False)}

## B2每折学习权重

{weights.to_markdown(index=False)}

## 决策闸门

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```
"""


if __name__ == "__main__":
    main()
