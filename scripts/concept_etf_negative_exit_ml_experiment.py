from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from factor_forge.research.concept_etf_exit_ml import (
    EXIT_FEATURES,
    ExitMLRules,
    build_fixed_r4_held_states,
    enrich_exit_features,
    fit_exit_walk_forward,
    simulate_r4_exit_sleeves,
)
from factor_forge.research.concept_etf_shadow import (
    monthly_performance,
    nonoverlap_sleeve_statistics,
    nonoverlapping_holding_periods,
)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rules = rules_from_config(config)
    panel = pd.read_parquet(config["signal_panel"])
    concepts = pd.read_parquet(config["concept_features"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    print("enriching ETF and constituent-diffusion exit features", flush=True)
    enriched = enrich_exit_features(panel, concepts)
    features = list(config.get("features", EXIT_FEATURES))
    print("building fixed-R4 held-state labels", flush=True)
    states, _ = build_fixed_r4_held_states(
        enriched, start=config["start"], end=config["end"], rules=rules,
    )
    print("fitting purged expanding walk-forward exit models", flush=True)
    predictions, importance, fold_audit, folds = fit_exit_walk_forward(
        states, enriched, start=config["start"], end=config["end"],
        feature_columns=features, rules=rules,
    )
    if predictions.empty:
        raise RuntimeError("exit walk-forward produced no OOF predictions")
    evaluation_start = pd.Timestamp(predictions["state_date"].min())
    model_audit = evaluate_oof_predictions(states, predictions)
    print(f"OOF evaluation starts {evaluation_start.date()}", flush=True)

    daily_parts: list[pd.DataFrame] = []
    sleeve_parts: list[pd.DataFrame] = []
    exit_parts: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    stats_parts: list[pd.DataFrame] = []
    paired_parts: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []
    for cost in config["roundtrip_cost_bps"]:
        simulations = {}
        for policy in config["policies"]:
            print(f"simulating {policy} at {cost}bps", flush=True)
            aggregate, sleeves, exits = simulate_r4_exit_sleeves(
                enriched, predictions, policy=policy,
                start=config["start"], end=config["end"],
                roundtrip_cost_bps=float(cost), rules=rules,
            )
            aggregate["roundtrip_cost_bps"] = int(cost)
            sleeves["roundtrip_cost_bps"] = int(cost)
            if not exits.empty:
                exits["roundtrip_cost_bps"] = int(cost)
                exit_parts.append(exits)
            simulations[policy] = (aggregate, sleeves, exits)
            daily_parts.append(aggregate)
            sleeve_parts.append(sleeves)
        baseline_periods = evaluation_periods(
            simulations["X0_fixed"][1], evaluation_start,
        )
        evaluation_daily = []
        for policy, (aggregate, sleeves, exits) in simulations.items():
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
                policy, int(cost), daily, stats, exits, evaluation_start,
            ))
        combined = pd.concat(evaluation_daily, ignore_index=True)
        monthly = monthly_performance(combined, benchmark_portfolio="X0_fixed")
        monthly["roundtrip_cost_bps"] = int(cost)
        monthly_parts.append(monthly)

    summary = pd.DataFrame(summary_rows)
    sleeve_stats = pd.concat(stats_parts, ignore_index=True)
    paired = pd.concat(paired_parts, ignore_index=True)
    monthly = pd.concat(monthly_parts, ignore_index=True)
    daily = pd.concat(daily_parts, ignore_index=True)
    sleeves = pd.concat(sleeve_parts, ignore_index=True)
    exits = pd.concat(exit_parts, ignore_index=True) if exit_parts else pd.DataFrame()
    decision = make_decision(summary, config["acceptance"])
    feature_importance = summarize_importance(importance)
    audit = {
        "panel_start": str(enriched["trade_date"].min().date()),
        "panel_end": str(enriched["trade_date"].max().date()),
        "etfs": int(enriched["ts_code"].nunique()),
        "held_state_rows": len(states),
        "exit_positive_label_rate": float(states["exit_advantage"].gt(0).mean()),
        "folds_defined": len(folds),
        "folds_fitted": len(fold_audit),
        "oof_start": str(predictions["state_date"].min().date()),
        "oof_end": str(predictions["state_date"].max().date()),
        "oof_grid_rows": len(predictions),
        "features": features,
        **model_audit,
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_root) / f"concept_etf_negative_exit_ml_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    states.to_parquet(output / "held_state_dataset.parquet", index=False)
    predictions.to_parquet(output / "oof_exit_predictions.parquet", index=False)
    importance.to_csv(output / "fold_feature_importance.csv", index=False, encoding="utf-8-sig")
    feature_importance.to_csv(output / "feature_importance.csv", index=False, encoding="utf-8-sig")
    fold_audit.to_csv(output / "fold_audit.csv", index=False, encoding="utf-8-sig")
    daily.to_parquet(output / "scenario_daily_nav.parquet", index=False)
    sleeves.to_parquet(output / "scenario_sleeve_daily.parquet", index=False)
    exits.to_csv(output / "early_exit_events.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    sleeve_stats.to_csv(output / "nonoverlap_sleeve_statistics.csv", index=False, encoding="utf-8-sig")
    paired.to_parquet(output / "paired_nonoverlap_periods.parquet", index=False)
    monthly.to_csv(output / "monthly_performance.csv", index=False, encoding="utf-8-sig")
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
        "concept_features": str(Path(config["concept_features"]).resolve()),
        "evaluation_start": str(evaluation_start.date()),
        "evaluation_end": config["end"],
        "important": "Touched-history discovery only; not authorization for live trading.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "research_report.md").write_text(
        render_report(summary, sleeve_stats, monthly, feature_importance, audit, decision, manifest),
        encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(output), "audit": audit, "decision": decision}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Purged walk-forward negative-factor ETF exit experiment")
    parser.add_argument("--config", default="configs/research/concept_etf_negative_exit_ml_v1.yaml")
    parser.add_argument("--output-root", default="artifacts/concept_etf_negative_exit_ml")
    return parser.parse_args()


def rules_from_config(config: dict) -> ExitMLRules:
    walk = config["walk_forward"]
    exit_config = config["exit"]
    return ExitMLRules(
        holding_days=int(exit_config["holding_days"]),
        minimum_train_days=int(walk["minimum_train_days"]),
        validation_days=int(walk["validation_days"]),
        test_days=int(walk["test_days"]),
        embargo_days=int(walk["embargo_days"]),
        minimum_train_rows=int(walk["minimum_train_rows"]),
        minimum_validation_rows=int(walk["minimum_validation_rows"]),
        consecutive_negative_days=int(exit_config["consecutive_negative_days"]),
        early_exit_cost_bps=float(exit_config["early_exit_cost_bps"]),
    )


def evaluate_oof_predictions(states: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    actual = states[["state_date", "ts_code", "holding_age", "exit_advantage"]]
    joined = actual.merge(
        predictions, on=["state_date", "ts_code", "holding_age"],
        how="inner", validate="many_to_one",
    )
    return {
        "oof_held_state_rows": len(joined),
        "oof_prediction_correlation": float(joined["predicted_exit_advantage"].corr(joined["exit_advantage"])),
        "placebo_prediction_correlation": float(joined["placebo_exit_advantage"].corr(joined["exit_advantage"])),
        "oof_negative_sign_accuracy": float(
            joined["predicted_exit_advantage"].gt(0).eq(joined["exit_advantage"].gt(0)).mean()
        ),
    }


def rebase_evaluation_daily(daily: pd.DataFrame, evaluation_start: pd.Timestamp) -> pd.DataFrame:
    result = daily.loc[daily["return_date"].gt(evaluation_start)].copy()
    result["net_nav"] = (1 + result["net_return"]).cumprod()
    return result


def evaluation_periods(sleeves: pd.DataFrame, evaluation_start: pd.Timestamp) -> pd.DataFrame:
    periods = nonoverlapping_holding_periods(sleeves)
    return periods.loc[periods["signal_date"].ge(evaluation_start)].reset_index(drop=True)


def summarize_policy(
    policy: str,
    cost: int,
    daily: pd.DataFrame,
    stats: pd.DataFrame,
    exits: pd.DataFrame,
    evaluation_start: pd.Timestamp,
) -> dict:
    drawdown = daily["net_nav"] / daily["net_nav"].cummax().clip(lower=1.0) - 1
    relevant_exits = exits.loc[exits["exit_date"].gt(evaluation_start)] if not exits.empty else exits
    false_exit_rate = float(relevant_exits["false_exit"].mean()) if not relevant_exits.empty else 0.0
    mean_avoided = (
        float(-relevant_exits["remaining_return_after_exit"].mean())
        if not relevant_exits.empty else 0.0
    )
    return {
        "policy": policy,
        "roundtrip_cost_bps": cost,
        "evaluation_start": evaluation_start,
        "evaluation_end": daily["return_date"].max(),
        "total_return": float(daily["net_nav"].iloc[-1] - 1),
        "maximum_drawdown": float(drawdown.min()),
        "mean_daily_turnover": float(daily["turnover"].mean()),
        "mean_cash_weight": float(daily["cash_weight"].mean()),
        "positive_sleeves": int(stats["mean_net_excess"].gt(0).sum()),
        "minimum_sleeve_excess": float(stats["mean_net_excess"].min()),
        "median_sleeve_excess": float(stats["mean_net_excess"].median()),
        "exit_events": len(relevant_exits),
        "false_exit_rate": false_exit_rate,
        "mean_avoided_remaining_return": mean_avoided,
    }


def summarize_importance(importance: pd.DataFrame) -> pd.DataFrame:
    result = importance.groupby("feature", as_index=False).agg(
        mean_gain=("gain", "mean"), median_gain=("gain", "median"),
        positive_gain_folds=("gain", lambda values: int(values.gt(0).sum())),
        folds=("fold", "nunique"),
    )
    total = result["mean_gain"].sum()
    result["gain_share"] = result["mean_gain"] / total if total > 0 else 0.0
    return result.sort_values("mean_gain", ascending=False).reset_index(drop=True)


def make_decision(summary: pd.DataFrame, acceptance: dict) -> dict:
    indexed = summary.set_index(["policy", "roundtrip_cost_bps"])
    checks: dict[str, bool] = {}
    for cost in (20, 40):
        ml = indexed.loc[("X2_ml", cost)]
        fixed = indexed.loc[("X0_fixed", cost)]
        placebo = indexed.loc[("X3_placebo", cost)]
        checks[f"positive_total_excess_{cost}bps"] = bool(ml["total_return"] > fixed["total_return"])
        checks[f"positive_sleeves_{cost}bps"] = bool(
            ml["positive_sleeves"] >= int(acceptance["minimum_positive_sleeves"])
        )
        checks[f"drawdown_not_worse_{cost}bps"] = bool(
            ml["maximum_drawdown"] >= fixed["maximum_drawdown"] + float(acceptance["minimum_drawdown_improvement"])
        )
        checks[f"beats_placebo_{cost}bps"] = bool(ml["total_return"] > placebo["total_return"])
    ml20 = indexed.loc[("X2_ml", 20)]
    checks["false_exit_rate"] = bool(
        ml20["false_exit_rate"] <= float(acceptance["maximum_false_exit_rate"])
    )
    return {
        "verdict": "KEEP_FOR_FORWARD_SHADOW" if all(checks.values()) else "NEGATIVE_EXIT_ML_NOT_VALIDATED",
        "checks": checks,
        "all_passed": all(checks.values()),
        "not_forward_confirmation": True,
    }


def render_report(summary, sleeve_stats, monthly, importance, audit, decision, manifest) -> str:
    summary_display = summary.copy()
    for column in (
        "total_return", "maximum_drawdown", "mean_daily_turnover", "mean_cash_weight",
        "minimum_sleeve_excess", "median_sleeve_excess", "false_exit_rate",
        "mean_avoided_remaining_return",
    ):
        summary_display[column] = summary_display[column].map(lambda value: f"{value:.2%}")
    stats_display = sleeve_stats.loc[sleeve_stats["roundtrip_cost_bps"].eq(20)].copy()
    for column in (
        "mean_net_return", "mean_net_excess", "bootstrap_95_low", "bootstrap_95_high",
        "positive_period_rate", "total_net_return", "maximum_drawdown", "mean_turnover",
    ):
        stats_display[column] = stats_display[column].map(lambda value: f"{value:.2%}")
    monthly_display = monthly.loc[
        monthly["roundtrip_cost_bps"].eq(20)
        & monthly["portfolio"].isin(["X0_fixed", "X1_rule", "X2_ml", "X3_placebo"])
    ].pivot(index="month", columns="portfolio", values="monthly_return")
    monthly_display = monthly_display.map(lambda value: f"{value:.2%}")
    importance_display = importance.head(12).copy()
    importance_display["gain_share"] = importance_display["gain_share"].map(lambda value: f"{value:.2%}")
    return f"""# ETF负向因子提前退出：走步验证

## 结论

**{decision['verdict']}**。本实验只使用样本外模型预测，但历史区间已经被查看，因此仍属于探索证据，不构成实盘授权。

- OOF区间：{manifest['evaluation_start']} 至 {manifest['evaluation_end']}
- 持仓状态样本：{audit['held_state_rows']}；拟合折数：{audit['folds_fitted']}
- OOF预测相关性：{audit['oof_prediction_correlation']:.3f}；安慰剂：{audit['placebo_prediction_correlation']:.3f}
- 模型信号方向准确率：{audit['oof_negative_sign_accuracy']:.2%}

## 场景汇总

{summary_display.to_markdown(index=False)}

## 20bps每袖非重叠结果

{stats_display.to_markdown(index=False)}

## 20bps月收益

{monthly_display.to_markdown()}

## 模型特征重要性

{importance_display.to_markdown(index=False)}

## 决策闸门

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```
"""


if __name__ == "__main__":
    main()
