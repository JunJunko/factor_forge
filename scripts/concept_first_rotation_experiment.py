from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from concept_etf_positive_diffusion_entry_experiment import (
    evaluation_periods,
    rebase_evaluation_daily,
    signal_rank_ic,
    summarize_policy,
)
from factor_forge.research.concept_etf_shadow import (
    monthly_performance,
    nonoverlap_sleeve_statistics,
    simulate_staggered_sleeves,
)
from factor_forge.research.concept_first_rotation import (
    ConceptFirstRules,
    attach_concept_scores_to_etfs,
    build_concept_first_features,
    coefficient_stability,
    concept_oof_diagnostics,
    fit_concept_first_walk_forward,
)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rules = rules_from_config(config)
    concepts_raw = pd.read_parquet(config["concept_features"])
    panel = pd.read_parquet(config["signal_panel"])
    concepts_raw["trade_date"] = pd.to_datetime(concepts_raw["trade_date"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])

    print("building concept-first causal features", flush=True)
    concepts = build_concept_first_features(concepts_raw)
    print("fitting concept-level 3/5/10d walk-forward models", flush=True)
    scores, coefficients, fold_audit = fit_concept_first_walk_forward(
        concepts,
        start=config["model_start"],
        end=config["end"],
        rules=rules,
    )
    if scores.empty:
        raise RuntimeError("concept-first walk-forward produced no OOF scores")
    evaluation_start = pd.Timestamp(scores["trade_date"].min())
    panel = attach_concept_scores_to_etfs(panel, scores)
    concept_policies = {
        policy: column
        for policy, column in config["policies"].items()
        if policy in {
            "C1_concept_linear_5d", "C2_concept_nonlinear_5d",
            "C3_multihorizon", "C5_state_placebo",
        }
    }
    concept_ic, concept_buckets = concept_oof_diagnostics(
        concepts, scores, policies=concept_policies,
    )
    etf_ic = signal_rank_ic(panel, config["policies"], evaluation_start, config["end"])
    stability = coefficient_stability(coefficients)
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
            aggregate, sleeves, attribution = simulate_staggered_sleeves(
                panel,
                "R4_rank_buffer",
                start=str(evaluation_start.date()),
                end=config["end"],
                roundtrip_cost_bps=float(cost),
                score_column=score_column,
            )
            aggregate["portfolio"] = policy
            aggregate["roundtrip_cost_bps"] = int(cost)
            sleeves["portfolio"] = policy
            sleeves["variant"] = policy
            sleeves["roundtrip_cost_bps"] = int(cost)
            attribution["policy"] = policy
            attribution["roundtrip_cost_bps"] = int(cost)
            simulations[policy] = (aggregate, sleeves, attribution)
            daily_parts.append(aggregate)
            sleeve_parts.append(sleeves)
            attribution_parts.append(attribution)
        baseline_periods = evaluation_periods(
            simulations["C0_etf_r4"][1], evaluation_start,
        )
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
            row = summarize_policy(
                policy, int(cost), daily, stats, attribution, evaluation_start,
            )
            row.update(profit_diversity(attribution, panel))
            summary_rows.append(row)
        monthly = monthly_performance(
            pd.concat(evaluation_daily, ignore_index=True),
            benchmark_portfolio="C0_etf_r4",
        )
        monthly["roundtrip_cost_bps"] = int(cost)
        monthly_parts.append(monthly)

    print("running leave-one-cluster-out and mapping-universe diagnostics", flush=True)
    cluster_robustness = leave_one_cluster_out(
        panel, config, evaluation_start,
    )
    mapping_robustness = mapping_universe_robustness(
        panel, config, evaluation_start,
    )
    summary = pd.DataFrame(summary_rows)
    monthly = pd.concat(monthly_parts, ignore_index=True)
    sleeve_stats = pd.concat(stats_parts, ignore_index=True)
    paired = pd.concat(paired_parts, ignore_index=True)
    daily = pd.concat(daily_parts, ignore_index=True)
    sleeves = pd.concat(sleeve_parts, ignore_index=True)
    attribution = pd.concat(attribution_parts, ignore_index=True)
    diagnostic_audit = audit_concept_diagnostics(
        concept_ic, concept_buckets, config["diagnostics"],
    )
    history_days = int(concepts["trade_date"].nunique())
    data_gate = {
        "concept_rows": len(concepts),
        "concepts": int(concepts["concept_code"].nunique()),
        "history_days": history_days,
        "minimum_history_days": int(config["diagnostics"]["minimum_history_days"]),
        "history_gate_passed": history_days >= int(config["diagnostics"]["minimum_history_days"]),
        "history_start": str(concepts["trade_date"].min().date()),
        "history_end": str(concepts["trade_date"].max().date()),
        "mapped_concepts": int(panel["concept_code"].nunique()),
        "mapped_etfs": int(panel["ts_code"].nunique()),
        "oof_start": str(evaluation_start.date()),
        "oof_end": str(scores["trade_date"].max().date()),
        "oof_concept_rows": len(scores),
        "folds": int(fold_audit["fold"].nunique()),
    }
    cluster_audit = audit_cluster_robustness(cluster_robustness)
    decision = make_decision(
        summary,
        diagnostic_audit,
        cluster_audit,
        data_gate,
        config["acceptance"],
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_root) / f"concept_first_rotation_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    scores.to_parquet(output / "concept_oof_scores.parquet", index=False)
    coefficients.to_csv(output / "linear_coefficients.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(output / "coefficient_stability.csv", index=False, encoding="utf-8-sig")
    fold_audit.to_csv(output / "fold_audit.csv", index=False, encoding="utf-8-sig")
    concept_ic.to_csv(output / "concept_rank_ic.csv", index=False, encoding="utf-8-sig")
    concept_buckets.to_csv(output / "concept_bucket_returns.csv", index=False, encoding="utf-8-sig")
    etf_ic.to_csv(output / "etf_rank_ic.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_performance.csv", index=False, encoding="utf-8-sig")
    sleeve_stats.to_csv(output / "nonoverlap_sleeve_statistics.csv", index=False, encoding="utf-8-sig")
    paired.to_parquet(output / "paired_nonoverlap_periods.parquet", index=False)
    daily.to_parquet(output / "scenario_daily_nav.parquet", index=False)
    sleeves.to_parquet(output / "scenario_sleeve_daily.parquet", index=False)
    attribution.to_csv(output / "profit_attribution.csv", index=False, encoding="utf-8-sig")
    cluster_robustness.to_csv(output / "leave_one_cluster_out.csv", index=False, encoding="utf-8-sig")
    mapping_robustness.to_csv(output / "mapping_universe_robustness.csv", index=False, encoding="utf-8-sig")
    (output / "data_audit.json").write_text(
        json.dumps(data_gate, ensure_ascii=False, indent=2), encoding="utf-8",
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
        "concept_features": str(Path(config["concept_features"]).resolve()),
        "signal_panel": str(Path(config["signal_panel"]).resolve()),
        "evaluation_start": str(evaluation_start.date()),
        "evaluation_end": config["end"],
        "important": "Short-history discovery only; no live-trading authorization.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "research_report.md").write_text(
        render_report(
            summary, monthly, sleeve_stats, concept_ic, concept_buckets,
            stability, cluster_robustness, mapping_robustness,
            data_gate, diagnostic_audit, decision,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "run_dir": str(output),
        "data_gate": data_gate,
        "diagnostic_audit": diagnostic_audit,
        "cluster_audit": cluster_audit,
        "decision": decision,
    }, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concept-first ETF rotation alpha experiment")
    parser.add_argument(
        "--config", default="configs/research/concept_first_rotation_v1.yaml",
    )
    parser.add_argument(
        "--output-root", default="artifacts/concept_first_rotation",
    )
    return parser.parse_args()


def rules_from_config(config: dict) -> ConceptFirstRules:
    model = config["model"]
    return ConceptFirstRules(
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
        seed=int(model["seed"]),
    )


def profit_diversity(attribution: pd.DataFrame, panel: pd.DataFrame) -> dict:
    positive = attribution.loc[attribution["capital_contribution"].gt(0)].copy()
    shares = positive["positive_profit_share"]
    effective = float(1 / np.square(shares).sum()) if len(shares) and np.square(shares).sum() > 0 else 0.0
    cluster_map = panel[["ts_code", "cluster"]].drop_duplicates("ts_code")
    positive = positive.merge(cluster_map, on="ts_code", how="left")
    return {
        "effective_profit_contributors": effective,
        "positive_profit_etfs": int(positive["ts_code"].nunique()),
        "positive_profit_clusters": int(positive["cluster"].nunique()),
    }


def leave_one_cluster_out(panel, config, evaluation_start) -> pd.DataFrame:
    metadata = panel[["ts_code", "cluster"]].drop_duplicates()
    rows = []
    policies = {
        "C0_etf_r4": config["policies"]["C0_etf_r4"],
        "C1_concept_linear_5d": config["policies"]["C1_concept_linear_5d"],
    }
    for cluster, members in metadata.groupby("cluster", observed=True):
        excluded = set(members["ts_code"].astype(str))
        for cost in config["roundtrip_cost_bps"]:
            returns = {}
            for policy, score_column in policies.items():
                aggregate, _, _ = simulate_staggered_sleeves(
                    panel,
                    "R4_rank_buffer",
                    start=str(evaluation_start.date()),
                    end=config["end"],
                    roundtrip_cost_bps=float(cost),
                    excluded_etfs=excluded,
                    score_column=score_column,
                )
                daily = rebase_evaluation_daily(aggregate, evaluation_start)
                returns[policy] = float(daily["net_nav"].iloc[-1] - 1)
            rows.append({
                "excluded_cluster": cluster,
                "excluded_etfs": ",".join(sorted(excluded)),
                "roundtrip_cost_bps": int(cost),
                "C0_return": returns["C0_etf_r4"],
                "C1_return": returns["C1_concept_linear_5d"],
                "C1_excess": returns["C1_concept_linear_5d"] - returns["C0_etf_r4"],
            })
    return pd.DataFrame(rows)


def mapping_universe_robustness(panel, config, evaluation_start) -> pd.DataFrame:
    rows = []
    for universe in ("no_proxy", "exact"):
        for cost in config["roundtrip_cost_bps"]:
            for policy in ("C0_etf_r4", "C1_concept_linear_5d"):
                aggregate, _, _ = simulate_staggered_sleeves(
                    panel,
                    "R4_rank_buffer",
                    start=str(evaluation_start.date()),
                    end=config["end"],
                    roundtrip_cost_bps=float(cost),
                    universe=universe,
                    score_column=config["policies"][policy],
                )
                daily = rebase_evaluation_daily(aggregate, evaluation_start)
                drawdown = daily["net_nav"] / daily["net_nav"].cummax().clip(lower=1.0) - 1
                rows.append({
                    "universe": universe,
                    "roundtrip_cost_bps": int(cost),
                    "policy": policy,
                    "total_return": float(daily["net_nav"].iloc[-1] - 1),
                    "maximum_drawdown": float(drawdown.min()),
                    "mean_cash_weight": float(daily["cash_weight"].mean()),
                })
    return pd.DataFrame(rows)


def audit_concept_diagnostics(ic, buckets, config) -> dict:
    result = {}
    for policy in ic["policy"].unique():
        policy_ic = ic.loc[ic["policy"].eq(policy)]
        policy_buckets = buckets.loc[buckets["policy"].eq(policy)]
        pivot = policy_buckets.pivot(
            index="horizon", columns="bucket", values="mean_forward_excess",
        )
        spread = pivot.get(5, pd.Series(dtype=float)) - pivot.get(1, pd.Series(dtype=float))
        positive_ic = int(policy_ic["mean_rank_ic"].gt(0).sum())
        positive_spread = int(spread.gt(0).sum())
        result[policy] = {
            "positive_rank_ic_horizons": positive_ic,
            "positive_top_bottom_horizons": positive_spread,
            "rank_ic_passed": positive_ic >= int(config["minimum_positive_rank_ic_horizons"]),
            "bucket_spread_passed": positive_spread >= int(config["minimum_positive_top_bottom_horizons"]),
            "top_bottom_spread": {str(int(key)): float(value) for key, value in spread.items()},
        }
    return result


def audit_cluster_robustness(frame: pd.DataFrame) -> dict:
    result = {}
    for cost, sample in frame.groupby("roundtrip_cost_bps", observed=True):
        result[str(int(cost))] = {
            "positive_fraction": float(sample["C1_excess"].gt(0).mean()),
            "minimum_excess": float(sample["C1_excess"].min()),
            "median_excess": float(sample["C1_excess"].median()),
            "worst_cluster": str(sample.loc[sample["C1_excess"].idxmin(), "excluded_cluster"]),
        }
    return result


def make_decision(summary, diagnostics, cluster_audit, data_gate, acceptance) -> dict:
    indexed = summary.set_index(["policy", "roundtrip_cost_bps"])
    variants = {}
    for policy in (
        "C1_concept_linear_5d", "C2_concept_nonlinear_5d",
        "C3_multihorizon", "C4_mapping_quality",
    ):
        checks = {}
        for cost in (20, 40):
            candidate = indexed.loc[(policy, cost)]
            baseline = indexed.loc[("C0_etf_r4", cost)]
            placebo = indexed.loc[("C5_state_placebo", cost)]
            checks[f"positive_total_excess_{cost}bps"] = bool(
                candidate["total_return"] > baseline["total_return"]
            )
            checks[f"positive_sleeves_{cost}bps"] = bool(
                candidate["positive_sleeves"]
                >= int(acceptance["minimum_positive_sleeves_vs_c0"])
            )
            checks[f"drawdown_not_worse_{cost}bps"] = bool(
                candidate["maximum_drawdown"] >= baseline["maximum_drawdown"]
                - float(acceptance["maximum_drawdown_deterioration"])
            )
            checks[f"profit_concentration_{cost}bps"] = bool(
                candidate["maximum_positive_profit_share"]
                <= float(acceptance["maximum_positive_profit_share"])
            )
            checks[f"effective_contributors_{cost}bps"] = bool(
                candidate["effective_profit_contributors"]
                >= float(acceptance["minimum_effective_profit_contributors"])
            )
            checks[f"turnover_within_limit_{cost}bps"] = bool(
                candidate["mean_daily_turnover"]
                <= baseline["mean_daily_turnover"]
                * float(acceptance["maximum_turnover_multiple"])
            )
            checks[f"beats_placebo_{cost}bps"] = bool(
                candidate["total_return"] > placebo["total_return"]
            )
        if policy in diagnostics:
            checks["concept_rank_ic"] = bool(diagnostics[policy]["rank_ic_passed"])
            checks["concept_bucket_spread"] = bool(diagnostics[policy]["bucket_spread_passed"])
        if policy == "C1_concept_linear_5d":
            checks["leave_cluster_out_20bps"] = bool(
                cluster_audit["20"]["positive_fraction"]
                >= float(acceptance["minimum_positive_cluster_exclusion_fraction"])
            )
            checks["leave_cluster_out_40bps"] = bool(
                cluster_audit["40"]["positive_fraction"]
                >= float(acceptance["minimum_positive_cluster_exclusion_fraction"])
            )
        variants[policy] = {"passed": all(checks.values()), "checks": checks}
    primary_model_passed = variants["C1_concept_linear_5d"]["passed"]
    primary_passed = primary_model_passed and bool(data_gate["history_gate_passed"])
    return {
        "verdict": (
            "START_CONCEPT_FIRST_FORWARD_SHADOW"
            if primary_passed else "CONCEPT_FIRST_NOT_VALIDATED"
        ),
        "primary_policy": "C1_concept_linear_5d",
        "primary_model_passed": primary_model_passed,
        "history_gate_passed": bool(data_gate["history_gate_passed"]),
        "primary_passed": primary_passed,
        "variant_checks": variants,
        "not_alpha_confirmation": True,
    }


def render_report(
    summary, monthly, sleeve_stats, concept_ic, concept_buckets, stability,
    cluster_robustness, mapping_robustness, data_gate, diagnostics, decision,
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
    ic_display = concept_ic.copy()
    for column in ["mean_rank_ic", "positive_rank_ic_rate"]:
        ic_display[column] = ic_display[column].map(lambda value: f"{value:.2%}")
    bucket_display = concept_buckets.copy()
    bucket_display["mean_forward_excess"] = bucket_display["mean_forward_excess"].map(
        lambda value: f"{value:.2%}"
    )
    mapping_display = mapping_robustness.copy()
    for column in ["total_return", "maximum_drawdown", "mean_cash_weight"]:
        mapping_display[column] = mapping_display[column].map(lambda value: f"{value:.2%}")
    return f"""# 概念优先的板块轮动Alpha实验

## 结论

**{decision['verdict']}**

- 概念数：{data_gate['concepts']}
- 概念历史：{data_gate['history_days']}个交易日，最低要求{data_gate['minimum_history_days']}日
- OOF区间：{data_gate['oof_start']} 至 {data_gate['oof_end']}
- 映射到ETF：{data_gate['mapped_concepts']}个概念 / {data_gate['mapped_etfs']}只ETF
- 历史长度闸门：{'通过' if data_gate['history_gate_passed'] else '未通过'}

## ETF策略汇总

{summary_display.to_markdown(index=False)}

## 概念层Rank IC

{ic_display.to_markdown(index=False)}

## 概念五分组收益

{bucket_display.to_markdown(index=False)}

## 20bps月度收益

{monthly_display.to_markdown()}

## 20bps非重叠袖套

{stats_display.to_markdown(index=False)}

## 线性特征稳定性

{stability.to_markdown(index=False)}

## 逐板块簇剔除

{cluster_robustness.to_markdown(index=False)}

## 映射口径稳健性

{mapping_display.to_markdown(index=False)}

## 诊断审计

```json
{json.dumps(diagnostics, ensure_ascii=False, indent=2)}
```

## 决策闸门

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```
"""


if __name__ == "__main__":
    main()
