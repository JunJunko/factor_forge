from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from concept_etf_positive_diffusion_entry_experiment import (
    evaluation_attribution,
    evaluation_periods,
    rebase_evaluation_daily,
    signal_rank_ic,
    summarize_policy,
)
from factor_forge.research.concept_etf_denoised_diffusion import (
    DENOISED_ML_FEATURES,
    DenoisedDiffusionRules,
    attach_denoised_diffusion_scores,
    attach_forward_open_returns,
    attach_learned_denoised_score,
    diffusion_signal_diagnostics,
    fit_denoised_diffusion_walk_forward,
)
from factor_forge.research.concept_etf_shadow import (
    monthly_performance,
    nonoverlap_sleeve_statistics,
    simulate_staggered_sleeves,
)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rules = rules_from_config(config)
    panel = pd.read_parquet(config["signal_panel"])
    concepts = pd.read_parquet(config["concept_features"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    print("building causal denoised diffusion scores", flush=True)
    panel = attach_denoised_diffusion_scores(
        panel,
        concepts,
        maximum_diffusion_weight=rules.maximum_diffusion_weight,
        control_ridge_alpha=rules.control_ridge_alpha,
        confirmation_percentile=float(config["denoising"]["confirmation_percentile"]),
        seed=rules.seed,
    )
    panel = attach_forward_open_returns(
        panel, tuple(int(value) for value in config["diagnostics"]["horizons"]),
    )
    print("fitting stability-filtered positive Ridge", flush=True)
    predictions, learned_weights, stability, fold_audit = fit_denoised_diffusion_walk_forward(
        panel, start=config["start"], end=config["end"], rules=rules,
    )
    if predictions.empty:
        raise RuntimeError("denoised diffusion walk-forward produced no OOF predictions")
    panel = attach_learned_denoised_score(panel, predictions)
    evaluation_start = pd.Timestamp(predictions["trade_date"].min())
    print(f"OOF evaluation starts {evaluation_start.date()}", flush=True)
    rank_ic = signal_rank_ic(panel, config["policies"], evaluation_start, config["end"])
    diagnostic_ic, diagnostic_buckets = diffusion_signal_diagnostics(
        panel,
        start=str(evaluation_start.date()),
        end=config["end"],
        horizons=tuple(int(value) for value in config["diagnostics"]["horizons"]),
    )

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
                panel,
                "R4_rank_buffer",
                start=config["start"],
                end=config["end"],
                roundtrip_cost_bps=float(cost),
                score_column=score_column,
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
        baseline_periods = evaluation_periods(simulations["D0_price"][1], evaluation_start)
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
        monthly = monthly_performance(
            pd.concat(evaluation_daily, ignore_index=True),
            benchmark_portfolio="D0_price",
        )
        monthly["roundtrip_cost_bps"] = int(cost)
        monthly_parts.append(monthly)

    summary = pd.DataFrame(summary_rows)
    monthly = pd.concat(monthly_parts, ignore_index=True)
    sleeve_stats = pd.concat(stats_parts, ignore_index=True)
    paired = pd.concat(paired_parts, ignore_index=True)
    daily = pd.concat(daily_parts, ignore_index=True)
    sleeves = pd.concat(sleeve_parts, ignore_index=True)
    attribution = pd.concat(attribution_parts, ignore_index=True)
    print("running leave-one-ETF-out concentration diagnostic", flush=True)
    leave_one_out = leave_one_etf_out_robustness(
        panel, config, evaluation_start,
    )
    weight_audit = audit_learned_weights(learned_weights)
    diagnostic_audit = audit_diagnostics(diagnostic_ic, diagnostic_buckets, config["diagnostics"])
    concentration_audit = audit_leave_one_out(leave_one_out)
    decision = make_decision(summary, diagnostic_audit, config["acceptance"])
    audit = {
        "panel_start": str(panel["trade_date"].min().date()),
        "panel_end": str(panel["trade_date"].max().date()),
        "etfs": int(panel["ts_code"].nunique()),
        "oof_start": str(evaluation_start.date()),
        "oof_end": str(predictions["trade_date"].max().date()),
        "oof_prediction_rows": len(predictions),
        "folds_fitted": len(fold_audit),
        "confirmation_gate_pass_rate": float(panel["diffusion_confirmation_gate"].mean()),
        "learned_weight_audit": weight_audit,
        "diagnostic_audit": diagnostic_audit,
        "leave_one_etf_out_audit": concentration_audit,
        "active_member_filter": "deferred_to_separate_versioned_feature_rebuild",
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_root) / f"concept_etf_denoised_diffusion_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(output / "oof_learned_scores.parquet", index=False)
    learned_weights.to_csv(output / "learned_weights.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(output / "feature_stability.csv", index=False, encoding="utf-8-sig")
    fold_audit.to_csv(output / "fold_audit.csv", index=False, encoding="utf-8-sig")
    rank_ic.to_csv(output / "strategy_rank_ic.csv", index=False, encoding="utf-8-sig")
    diagnostic_ic.to_csv(output / "diffusion_horizon_ic.csv", index=False, encoding="utf-8-sig")
    diagnostic_buckets.to_csv(output / "diffusion_bucket_returns.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_performance.csv", index=False, encoding="utf-8-sig")
    sleeve_stats.to_csv(output / "nonoverlap_sleeve_statistics.csv", index=False, encoding="utf-8-sig")
    paired.to_parquet(output / "paired_nonoverlap_periods.parquet", index=False)
    daily.to_parquet(output / "scenario_daily_nav.parquet", index=False)
    sleeves.to_parquet(output / "scenario_sleeve_daily.parquet", index=False)
    attribution.to_csv(output / "profit_attribution.csv", index=False, encoding="utf-8-sig")
    leave_one_out.to_csv(output / "leave_one_etf_out.csv", index=False, encoding="utf-8-sig")
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
        "primary_policy": config["primary_policy"],
        "important": "Touched-history discovery only; no early exit and no live-trading authorization.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "research_report.md").write_text(
        render_report(
            summary, monthly, sleeve_stats, rank_ic, diagnostic_ic,
            diagnostic_buckets, learned_weights, stability, leave_one_out,
            audit, decision,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "run_dir": str(output),
        "audit": audit,
        "decision": decision,
    }, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Denoised diffusion confirmation experiment")
    parser.add_argument(
        "--config", default="configs/research/concept_etf_denoised_diffusion_v1.yaml",
    )
    parser.add_argument(
        "--output-root", default="artifacts/concept_etf_denoised_diffusion",
    )
    return parser.parse_args()


def rules_from_config(config: dict) -> DenoisedDiffusionRules:
    learned = config["learned_blend"]
    return DenoisedDiffusionRules(
        maximum_diffusion_weight=float(learned["maximum_diffusion_weight"]),
        ridge_alpha=float(learned["ridge_alpha"]),
        control_ridge_alpha=float(config["denoising"]["control_ridge_alpha"]),
        minimum_train_days=int(learned["minimum_train_days"]),
        validation_days=int(learned["validation_days"]),
        test_days=int(learned["test_days"]),
        embargo_days=int(learned["embargo_days"]),
        minimum_train_rows=int(learned["minimum_train_rows"]),
        stability_blocks=int(learned["stability_blocks"]),
        minimum_stability_fraction=float(learned["minimum_stability_fraction"]),
    )


def audit_learned_weights(weights: pd.DataFrame) -> dict:
    diffusion = weights[DENOISED_ML_FEATURES[1:]].sum(axis=1)
    return {
        "folds": len(weights),
        "mean_diffusion_weight": float(diffusion.mean()),
        "median_diffusion_weight": float(diffusion.median()),
        "minimum_diffusion_weight": float(diffusion.min()),
        "maximum_diffusion_weight": float(diffusion.max()),
        "zero_weight_fraction": float(diffusion.le(1e-10).mean()),
    }


def audit_diagnostics(ic: pd.DataFrame, buckets: pd.DataFrame, config: dict) -> dict:
    pivot = buckets.pivot(index="horizon", columns="bucket", values="mean_forward_excess")
    spread = pivot.get(5, pd.Series(dtype=float)) - pivot.get(1, pd.Series(dtype=float))
    positive_ic = int(ic["mean_rank_ic"].gt(0).sum())
    positive_spread = int(spread.gt(0).sum())
    return {
        "positive_mean_rank_ic_horizons": positive_ic,
        "positive_top_bottom_horizons": positive_spread,
        "top_bottom_spread": {str(int(key)): float(value) for key, value in spread.items()},
        "rank_ic_passed": positive_ic >= int(config["require_positive_mean_rank_ic_horizons"]),
        "bucket_spread_passed": positive_spread >= int(config["require_positive_top_bottom_horizons"]),
    }


def leave_one_etf_out_robustness(
    panel: pd.DataFrame,
    config: dict,
    evaluation_start: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    codes = sorted(panel.loc[
        panel["trade_date"].between(evaluation_start, pd.Timestamp(config["end"])),
        "ts_code",
    ].dropna().unique())
    policies = {
        "D0_price": config["policies"]["D0_price"],
        "D1_confirmation": config["policies"]["D1_confirmation"],
    }
    for code in codes:
        for cost in config["roundtrip_cost_bps"]:
            returns = {}
            for policy, score_column in policies.items():
                aggregate, _, _ = simulate_staggered_sleeves(
                    panel,
                    "R4_rank_buffer",
                    start=config["start"],
                    end=config["end"],
                    roundtrip_cost_bps=float(cost),
                    excluded_etfs={str(code)},
                    score_column=score_column,
                )
                daily = rebase_evaluation_daily(aggregate, evaluation_start)
                returns[policy] = float(daily["net_nav"].iloc[-1] - 1)
            rows.append({
                "excluded_etf": code,
                "roundtrip_cost_bps": int(cost),
                "D0_price_return": returns["D0_price"],
                "D1_confirmation_return": returns["D1_confirmation"],
                "D1_excess": returns["D1_confirmation"] - returns["D0_price"],
            })
    return pd.DataFrame(rows)


def audit_leave_one_out(leave_one_out: pd.DataFrame) -> dict:
    rows = {}
    for cost, frame in leave_one_out.groupby("roundtrip_cost_bps", observed=True):
        rows[str(int(cost))] = {
            "positive_exclusions": int(frame["D1_excess"].gt(0).sum()),
            "total_exclusions": len(frame),
            "minimum_excess": float(frame["D1_excess"].min()),
            "median_excess": float(frame["D1_excess"].median()),
            "worst_exclusion": str(frame.loc[frame["D1_excess"].idxmin(), "excluded_etf"]),
        }
    return rows


def make_decision(summary: pd.DataFrame, diagnostics: dict, acceptance: dict) -> dict:
    indexed = summary.set_index(["policy", "roundtrip_cost_bps"])
    variants = {}
    for policy in ("D1_confirmation", "D2_ten_percent_boost", "D3_learned"):
        checks = {}
        for cost in (20, 40):
            candidate = indexed.loc[(policy, cost)]
            baseline = indexed.loc[("D0_price", cost)]
            placebo = indexed.loc[("D4_placebo", cost)]
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
        variants[policy] = {"passed": all(checks.values()), "checks": checks}
    diagnostic_checks = {
        "rank_ic_horizons": bool(diagnostics["rank_ic_passed"]),
        "top_bottom_horizons": bool(diagnostics["bucket_spread_passed"]),
    }
    primary_passed = variants["D1_confirmation"]["passed"] and all(diagnostic_checks.values())
    return {
        "verdict": (
            "DENOISED_CONFIRMATION_VALIDATED_FOR_FORWARD_SHADOW"
            if primary_passed else "DENOISED_DIFFUSION_NOT_VALIDATED"
        ),
        "primary_policy": "D1_confirmation",
        "primary_passed": primary_passed,
        "diagnostic_checks": diagnostic_checks,
        "variant_checks": variants,
        "not_forward_confirmation": True,
    }


def render_report(
    summary, monthly, sleeve_stats, rank_ic, diagnostic_ic, diagnostic_buckets,
    weights, stability, leave_one_out, audit, decision,
) -> str:
    summary_display = summary.copy()
    percent_columns = [
        "total_return", "maximum_drawdown", "mean_daily_turnover", "mean_cash_weight",
        "minimum_sleeve_excess", "median_sleeve_excess", "maximum_positive_profit_share",
    ]
    for column in percent_columns:
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
    diagnostic_display = diagnostic_ic.copy()
    for column in ["mean_rank_ic", "positive_rank_ic_rate"]:
        diagnostic_display[column] = diagnostic_display[column].map(lambda value: f"{value:.2%}")
    bucket_display = diagnostic_buckets.copy()
    bucket_display["mean_forward_excess"] = bucket_display["mean_forward_excess"].map(
        lambda value: f"{value:.2%}"
    )
    leave_one_out_display = leave_one_out.loc[
        leave_one_out["roundtrip_cost_bps"].eq(20)
    ].copy()
    for column in ["D0_price_return", "D1_confirmation_return", "D1_excess"]:
        leave_one_out_display[column] = leave_one_out_display[column].map(
            lambda value: f"{value:.2%}"
        )
    return f"""# 去噪扩散确认实验

## 结论

**{decision['verdict']}**

- OOF区间：{audit['oof_start']} 至 {audit['oof_end']}
- ETF数量：{audit['etfs']}
- 确认门槛通过率：{audit['confirmation_gate_pass_rate']:.2%}
- D3平均/中位扩散权重：{audit['learned_weight_audit']['mean_diffusion_weight']:.2%} / {audit['learned_weight_audit']['median_diffusion_weight']:.2%}
- 成分股活跃过滤：本轮未混入，留待独立版本化重建。

## 场景汇总

{summary_display.to_markdown(index=False)}

## 策略5日Rank IC

{rank_display.to_markdown(index=False)}

## 去噪扩散衰减诊断

{diagnostic_display.to_markdown(index=False)}

## 去噪扩散五分组

{bucket_display.to_markdown(index=False)}

## 20bps月度收益

{monthly_display.to_markdown()}

## 20bps非重叠袖套

{stats_display.to_markdown(index=False)}

## D3学习权重

{weights.to_markdown(index=False)}

## D3训练内稳定性

{stability.to_markdown(index=False)}

## 20bps逐只ETF剔除

{leave_one_out_display.to_markdown(index=False)}

## 决策闸门

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```
"""


if __name__ == "__main__":
    main()
