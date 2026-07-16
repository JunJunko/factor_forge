from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from factor_forge.research.concept_lifecycle_backtest import (
    LifecycleRules,
    attach_enhanced_stock_features,
    attach_lifecycle_fields,
    build_document_market_regimes,
    build_lifecycle_targets,
)
from factor_forge.research.concept_portfolio_backtest import (
    ExecutionRules,
    PortfolioRules,
    paired_portfolio_comparison,
    run_non_overlapping_ledger,
)
from factor_forge.research.concept_rotation_alpha import build_concept_dataset, load_dc_snapshot


SPLITS = {
    "discovery": ("2025-10-01", "2025-12-31"),
    "validation": ("2026-01-01", "2026-03-31"),
    "final_reused_diagnostic": ("2026-04-01", "2026-06-30"),
}
VARIANTS = ["A_rrg_baseline", "B_breadth_filter", "C_full_lifecycle", "D_breadth_placebo"]
COMPARISONS = {
    "breadth_increment_B_minus_A": ("B_breadth_filter", "A_rrg_baseline"),
    "lifecycle_increment_C_minus_B": ("C_full_lifecycle", "B_breadth_filter"),
    "real_breadth_C_minus_placebo_D": ("C_full_lifecycle", "D_breadth_placebo"),
    "full_strategy_C_minus_A": ("C_full_lifecycle", "A_rrg_baseline"),
}
PANEL_COLUMNS = [
    "trade_date", "ts_code", "raw_open", "raw_close", "adj_open", "adj_close", "amount_cny",
    "circ_mv_cny", "turnover_rate", "is_suspended", "is_limit_up_open", "is_limit_down_open",
    "is_st", "is_delisting_period", "listing_trade_days", "is_tradeable",
]


def main() -> None:
    args = parse_args()
    output = Path(args.output_root) / datetime.now(timezone.utc).strftime("concept_lifecycle_%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    panel = load_stock_panel(Path(args.base_panel), Path(args.increment_panel))
    enhanced = pd.read_parquet(args.daily_basic)
    panel = attach_enhanced_stock_features(panel, enhanced)
    concept_index, members = load_dc_snapshot(args.snapshot_root, trade_dates=panel["trade_date"].unique())
    if args.features_cache:
        features = pd.read_parquet(args.features_cache)
        feature_audit = {"source": "cached_free_float_features", "path": str(args.features_cache)}
    else:
        _, features, feature_audit = build_concept_dataset(panel, concept_index, members)
    features = attach_lifecycle_fields(features)
    regimes = build_document_market_regimes(panel)
    rules = LifecycleRules()
    portfolio_rules = PortfolioRules(
        initial_cash=args.initial_cash,
        concepts_per_rebalance=rules.concepts_per_rebalance,
        maximum_stock_weight=rules.maximum_stock_weight,
    )
    execution = ExecutionRules(
        commission_bps_per_side=args.commission_bps,
        base_slippage_bps_per_side=args.base_slippage_bps,
        maximum_adv_participation=args.adv_participation,
    )
    scenarios = {
        "base": execution,
        "cost_2x": replace(execution, cost_multiplier=2.0),
        "extra_20bps_roundtrip": replace(execution, extra_slippage_bps_per_side=10.0),
    }

    metric_rows: list[dict] = []
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    target_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    daily_lookup: dict[tuple[str, str, int, str], pd.DataFrame] = {}
    total = len(SPLITS) * rules.decision_interval * len(VARIANTS)
    progress = 0
    for split, (start, end) in SPLITS.items():
        for offset in range(rules.decision_interval):
            for variant in VARIANTS:
                progress += 1
                targets = build_lifecycle_targets(
                    features, members, panel, regimes,
                    variant=variant, start=start, end=end, offset=offset, rules=rules,
                )
                if not targets.targets.empty:
                    item = targets.targets.copy()
                    item.insert(0, "split", split); item.insert(1, "offset", offset)
                    target_frames.append(item)
                if not targets.selections.empty:
                    item = targets.selections.copy()
                    item.insert(0, "split", split); item.insert(1, "offset", offset)
                    selection_frames.append(item)
                for scenario, execution_rules in scenarios.items():
                    result = run_non_overlapping_ledger(
                        panel, targets, start=start, end=end,
                        portfolio_rules=portfolio_rules, execution_rules=execution_rules,
                    )
                    metric_rows.append({
                        "variant": variant, "split": split, "offset": offset,
                        "scenario": scenario, **result.metrics,
                    })
                    if not result.daily.empty:
                        item = result.daily.copy()
                        item.insert(0, "variant", variant); item.insert(1, "split", split)
                        item.insert(2, "offset", offset); item.insert(3, "scenario", scenario)
                        daily_frames.append(item)
                        daily_lookup[(variant, split, offset, scenario)] = result.daily
                    if scenario == "base" and not result.trades.empty:
                        item = result.trades.copy()
                        item.insert(0, "variant", variant); item.insert(1, "split", split)
                        item.insert(2, "offset", offset)
                        trade_frames.append(item)
                print(f"lifecycle paths {progress}/{total} split={split} offset={offset} variant={variant}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    targets = pd.concat(target_frames, ignore_index=True) if target_frames else pd.DataFrame()
    selections = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    comparisons = comparison_table(daily_lookup)
    metric_summary = summarize_metrics(metrics)
    comparison_summary = summarize_comparisons(comparisons)
    mechanism = mechanism_diagnostics(features)
    audit = experiment_audit(panel, regimes, targets, daily, trades, feature_audit)
    decision = lifecycle_decision(comparison_summary)

    features.to_parquet(output / "concept_features_free_float.parquet", index=False)
    regimes.to_csv(output / "market_regimes.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output / "variant_metrics.csv", index=False, encoding="utf-8-sig")
    metric_summary.to_csv(output / "variant_summary.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(output / "paired_comparisons.csv", index=False, encoding="utf-8-sig")
    comparison_summary.to_csv(output / "comparison_summary.csv", index=False, encoding="utf-8-sig")
    mechanism.to_csv(output / "mechanism_diagnostics.csv", index=False, encoding="utf-8-sig")
    daily.to_parquet(output / "variant_daily.parquet", index=False)
    trades.to_parquet(output / "variant_trades.parquet", index=False)
    targets.to_parquet(output / "variant_targets.parquet", index=False)
    selections.to_parquet(output / "concept_selections.parquet", index=False)
    (output / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "splits": SPLITS, "variants": VARIANTS, "comparisons": COMPARISONS,
        "lifecycle_rules": rules.__dict__, "portfolio_rules": portfolio_rules.__dict__,
        "execution_rules": execution.__dict__,
        "warning": "Every historical segment has already been inspected; results are diagnostic only.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "research_report.md").write_text(
        render_report(decision, comparison_summary, metric_summary, mechanism, audit, manifest), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(output), "decision": decision}, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Document-faithful concept rotation lifecycle experiment")
    parser.add_argument("--snapshot-root", default="data/concept_rotation/dc_20250630_20260714")
    parser.add_argument("--daily-basic", default="data/concept_rotation/daily_basic_20250501_20260714/daily_basic_enhanced.parquet")
    parser.add_argument("--base-panel", default="data/versions/data_v1_20260710T115133Z_208e9fc8/curated/stock_daily_panel.parquet")
    parser.add_argument("--increment-panel", default="data/versions/data_v1_20260715T053637Z_77138057/curated/stock_daily_panel.parquet")
    parser.add_argument("--output-root", default="artifacts/concept_rotation_lifecycle")
    parser.add_argument("--features-cache")
    parser.add_argument("--initial-cash", type=float, default=10_000_000)
    parser.add_argument("--commission-bps", type=float, default=2.5)
    parser.add_argument("--base-slippage-bps", type=float, default=3.0)
    parser.add_argument("--adv-participation", type=float, default=0.05)
    return parser.parse_args()


def load_stock_panel(base_path: Path, increment_path: Path) -> pd.DataFrame:
    base = pd.read_parquet(base_path, columns=PANEL_COLUMNS, filters=[("trade_date", ">=", pd.Timestamp("2025-05-01"))])
    increment = pd.read_parquet(increment_path, columns=PANEL_COLUMNS)
    for frame in (base, increment):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    last_listing = base.groupby("ts_code", observed=True)["listing_trade_days"].max()
    increment = increment.sort_values(["ts_code", "trade_date"])
    increment["listing_trade_days"] = increment.groupby("ts_code", observed=True).cumcount() + 1 + increment["ts_code"].map(last_listing).fillna(0)
    increment["is_tradeable"] = (
        increment["listing_trade_days"].ge(60) & ~increment["is_suspended"].fillna(True)
        & ~increment["is_st"].fillna(False) & ~increment["is_delisting_period"].fillna(False)
        & increment["adj_open"].notna() & increment["adj_close"].notna()
    )
    return pd.concat([base, increment], ignore_index=True).sort_values(["trade_date", "ts_code"]).drop_duplicates(
        ["trade_date", "ts_code"], keep="last"
    ).reset_index(drop=True)


def comparison_table(daily_lookup: dict) -> pd.DataFrame:
    frames = []
    for name, (primary, baseline) in COMPARISONS.items():
        frame = paired_portfolio_comparison(daily_lookup, primary=primary, baseline=baseline)
        if not frame.empty:
            frame.insert(0, "comparison", name)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby(["variant", "split", "scenario"], observed=True).agg(
        offsets=("offset", "nunique"), positive_offsets=("total_return", lambda x: int(x.gt(0).sum())),
        mean_total_return=("total_return", "mean"), median_total_return=("total_return", "median"),
        worst_total_return=("total_return", "min"), mean_max_drawdown=("max_drawdown", "mean"),
        mean_turnover=("annualized_turnover", "mean"), mean_cash_ratio=("average_cash_ratio", "mean"),
        mean_cost_drag=("cost_drag", "mean"),
    ).reset_index()


def summarize_comparisons(comparisons: pd.DataFrame) -> pd.DataFrame:
    if comparisons.empty:
        return comparisons
    return comparisons.groupby(["comparison", "split", "scenario"], observed=True).agg(
        offsets=("offset", "nunique"), positive_offsets=("relative_total_return", lambda x: int(x.gt(0).sum())),
        mean_relative_return=("relative_total_return", "mean"),
        median_relative_return=("relative_total_return", "median"),
        worst_relative_return=("relative_total_return", "min"),
        median_nw_t=("incremental_nw_t", "median"),
    ).reset_index()


def mechanism_diagnostics(features: pd.DataFrame) -> pd.DataFrame:
    sample = features.loc[features["eligible_concept"].fillna(False)].copy()
    sample["breadth_bucket"] = sample.groupby("trade_date")["common_delta_rank"].transform(
        lambda values: pd.cut(values, [0, 0.3, 0.7, 1.0], labels=["low", "middle", "high"], include_lowest=True)
    )
    return sample.groupby(["rrg_quadrant", "breadth_bucket", "lifecycle"], observed=True).agg(
        observations=("forward_excess_5d", "count"),
        mean_forward_excess_5d=("forward_excess_5d", "mean"),
        mean_forward_excess_10d=("forward_excess_10d", "mean"),
    ).reset_index()


def experiment_audit(panel: pd.DataFrame, regimes: pd.DataFrame, targets: pd.DataFrame,
                     daily: pd.DataFrame, trades: pd.DataFrame, feature_audit: dict) -> dict:
    target_sum = targets.groupby(["variant", "split", "offset", "entry_date"])["target_weight"].sum() if not targets.empty else pd.Series(dtype=float)
    return {
        "feature_audit": feature_audit,
        "panel_start": str(panel["trade_date"].min().date()), "panel_end": str(panel["trade_date"].max().date()),
        "panel_rows": int(len(panel)), "missing_free_share": int(panel["free_share"].isna().sum()),
        "regime_counts": {str(k): int(v) for k, v in regimes["regime"].value_counts().items()},
        "target_rows": int(len(targets)), "maximum_target_exposure": float(target_sum.max()) if len(target_sum) else None,
        "maximum_target_weight": float(targets["target_weight"].max()) if not targets.empty else None,
        "daily_rows": int(len(daily)), "nan_nav": int(daily["nav"].isna().sum()) if not daily.empty else 0,
        "trade_rows": int(len(trades)), "maximum_participation": float(trades["participation"].max()) if not trades.empty else None,
    }


def lifecycle_decision(summary: pd.DataFrame) -> dict:
    checks = {}
    for comparison in COMPARISONS:
        frame = summary.loc[summary["comparison"].eq(comparison)]
        base = frame.loc[frame["scenario"].eq("base")].set_index("split")
        validation = base.loc["validation"] if "validation" in base.index else None
        final = base.loc["final_reused_diagnostic"] if "final_reused_diagnostic" in base.index else None
        stress = frame.loc[
            frame["split"].eq("final_reused_diagnostic")
            & frame["scenario"].isin(["cost_2x", "extra_20bps_roundtrip"])
        ]
        checks[comparison] = {
            "validation_3_of_5_positive": bool(validation is not None and validation["positive_offsets"] >= 3 and validation["mean_relative_return"] > 0),
            "final_4_of_5_positive": bool(final is not None and final["positive_offsets"] >= 4 and final["mean_relative_return"] > 0),
            "cost_stress_positive": bool(len(stress) == 2 and stress["mean_relative_return"].gt(0).all()),
        }
    required = ["breadth_increment_B_minus_A", "real_breadth_C_minus_placebo_D", "full_strategy_C_minus_A"]
    passed = all(all(checks[name].values()) for name in required)
    return {
        "decision": "FUTURE_SHADOW_CANDIDATE" if passed else "NO_FULL_STRATEGY_PASS",
        "historical_screen_passed": passed, "checks": checks,
        "warning": "No result is confirmatory because all historical dates were previously inspected.",
    }


def render_report(decision: dict, comparisons: pd.DataFrame, metrics: pd.DataFrame,
                  mechanism: pd.DataFrame, audit: dict, manifest: dict) -> str:
    focus = comparisons.loc[
        comparisons["scenario"].eq("base")
        & comparisons["split"].isin(["validation", "final_reused_diagnostic"])
    ]
    return f"""# Document-faithful concept rotation lifecycle experiment

## Decision

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```

The experiment implements the chain: market regime -> concept breadth/RRG lifecycle ->
core or catch-up stock selection -> state-driven exits. A/B/C/D share execution rules and
the same ex-ante regime exposure schedule. Realized exposure can still differ when a variant
has too few eligible concepts or stocks under the position cap, and is reported as part of attribution.

## Paired comparison

```text
{focus.to_string(index=False)}
```

## Portfolio summary

```text
{metrics.loc[metrics['scenario'].eq('base')].to_string(index=False)}
```

## Mechanism diagnostics

```text
{mechanism.to_string(index=False)}
```

## Audit

```json
{json.dumps(audit, ensure_ascii=False, indent=2)}
```

## Manifest

```json
{json.dumps(manifest, ensure_ascii=False, indent=2)}
```
"""


if __name__ == "__main__":
    main()
