from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from concept_rotation_lightgbm_experiment import (
    PANEL_COLUMNS,
    ensemble_results,
    load_daily_basic,
    load_stock_panel,
)
from factor_forge.research.concept_lifecycle_backtest import (
    attach_enhanced_stock_features,
    attach_lifecycle_fields,
    build_document_market_regimes,
)
from factor_forge.research.concept_portfolio_backtest import (
    ExecutionRules,
    PortfolioRules,
    run_non_overlapping_ledger,
)
from factor_forge.research.concept_rotation_alpha import build_concept_dataset
from factor_forge.research.concept_rotation_ml import (
    FEATURE_COLUMNS,
    ConceptMLRules,
    build_concept_ml_dataset,
    build_ml_rotation_targets,
    fit_walk_forward_rankers,
    load_dc_snapshot_roots,
    prediction_diagnostics,
)


VARIANT_POOL = {
    "M0_momentum": "momentum",
    "M1_leading_momentum": "leading",
    "M2_leading_lgbm": "leading",
    "M3_leading_breadth_momentum": "leading_breadth",
    "M4_leading_breadth_lgbm": "leading_breadth",
}


def main() -> None:
    args = parse_args()
    output = Path(args.output_root) / datetime.now(timezone.utc).strftime("leading_lgbm_%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    panel = attach_enhanced_stock_features(
        load_stock_panel(Path(args.base_panel), Path(args.increment_panel), pd.Timestamp(args.panel_start)),
        load_daily_basic(args.daily_basic),
    )
    regimes = build_document_market_regimes(panel)
    concept_index, members = load_dc_snapshot_roots(
        args.snapshot_root, trade_dates=panel["trade_date"].unique()
    )
    _, features, feature_audit = build_concept_dataset(panel, concept_index, members)
    features = attach_lifecycle_fields(features)
    rules = ConceptMLRules()

    datasets = {
        mode: build_concept_ml_dataset(
            features, members, panel, regimes, candidate_mode=mode, rules=rules,
        )
        for mode in ("momentum", "leading", "leading_breadth")
    }
    leading_predictions, leading_importance, leading_folds = fit_walk_forward_rankers(
        datasets["leading"], rules=rules, seed=42,
    )
    breadth_predictions, breadth_importance, breadth_folds = fit_walk_forward_rankers(
        datasets["leading_breadth"], rules=rules, seed=314159,
    )
    if leading_predictions.empty or breadth_predictions.empty:
        raise RuntimeError("a filtered candidate pool produced no OOF predictions")
    leading_predictions["candidate_mode"] = "leading"
    breadth_predictions["candidate_mode"] = "leading_breadth"
    common_start = max(
        pd.Timestamp(leading_predictions["trade_date"].min()),
        pd.Timestamp(breadth_predictions["trade_date"].min()),
    )
    common_end = min(
        pd.Timestamp(leading_predictions["trade_date"].max()),
        pd.Timestamp(breadth_predictions["trade_date"].max()),
    )
    predictions = {
        "momentum": momentum_scores(datasets["momentum"], common_start, common_end),
        "leading": leading_predictions.loc[leading_predictions["trade_date"].between(common_start, common_end)].copy(),
        "leading_breadth": breadth_predictions.loc[breadth_predictions["trade_date"].between(common_start, common_end)].copy(),
    }
    importance = pd.concat([
        leading_importance.assign(candidate_mode="leading"),
        breadth_importance.assign(candidate_mode="leading_breadth"),
    ], ignore_index=True)
    diagnostics = pd.concat([
        prediction_diagnostics(predictions["leading"]).assign(candidate_mode="leading"),
        prediction_diagnostics(predictions["leading_breadth"]).assign(candidate_mode="leading_breadth"),
    ], ignore_index=True)

    portfolio_rules = PortfolioRules(
        initial_cash=args.initial_cash, holding_days=rules.horizon,
        concepts_per_rebalance=rules.selected_concepts,
        stocks_per_concept=rules.stocks_per_concept,
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
    metrics_rows, daily_frames, trade_frames, target_frames, selection_frames = [], [], [], [], []
    total = len(VARIANT_POOL) * rules.horizon
    progress = 0
    for offset in range(rules.horizon):
        for variant, pool in VARIANT_POOL.items():
            progress += 1
            targets = build_ml_rotation_targets(
                predictions[pool], members, regimes, panel, variant=variant,
                start=common_start, end=common_end, offset=offset, rules=rules,
            )
            if not targets.targets.empty:
                item = targets.targets.copy(); item.insert(0, "offset", offset); target_frames.append(item)
            if not targets.selections.empty:
                item = targets.selections.copy(); item.insert(0, "offset", offset); selection_frames.append(item)
            for scenario, execution_rules in scenarios.items():
                result = run_non_overlapping_ledger(
                    panel, targets, start=common_start, end=common_end,
                    portfolio_rules=portfolio_rules, execution_rules=execution_rules,
                )
                metrics_rows.append({
                    "variant": variant, "offset": offset, "scenario": scenario, **result.metrics,
                })
                if not result.daily.empty:
                    item = result.daily.copy(); item.insert(0, "variant", variant)
                    item.insert(1, "offset", offset); item.insert(2, "scenario", scenario)
                    daily_frames.append(item)
                if scenario == "base" and not result.trades.empty:
                    item = result.trades.copy(); item.insert(0, "variant", variant)
                    item.insert(1, "offset", offset); trade_frames.append(item)
            print(f"leading LightGBM {progress}/{total} offset={offset} variant={variant}", flush=True)

    metrics = pd.DataFrame(metrics_rows)
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    targets = pd.concat(target_frames, ignore_index=True) if target_frames else pd.DataFrame()
    selections = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    ensemble, monthly = ensemble_results(daily, args.initial_cash)
    comparisons = planned_comparisons(ensemble, monthly)
    coverage = candidate_coverage(datasets, common_start, common_end)
    decision = ablation_decision(comparisons, ensemble)
    audit = experiment_audit(
        datasets, predictions, leading_folds, breadth_folds, targets, daily, trades,
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variants": VARIANT_POOL, "features": FEATURE_COLUMNS, "rules": asdict(rules),
        "portfolio_rules": asdict(portfolio_rules), "execution_rules": asdict(execution),
        "common_oof_start": common_start.strftime("%Y-%m-%d"),
        "common_oof_end": common_end.strftime("%Y-%m-%d"),
        "feature_audit": feature_audit,
        "warning": "All dates through 2026-07-14 were previously inspected; diagnostic only.",
    }

    for mode, frame in datasets.items():
        frame.to_parquet(output / f"dataset_{mode}.parquet", index=False)
    for mode, frame in predictions.items():
        frame.to_parquet(output / f"predictions_{mode}.parquet", index=False)
    importance.to_csv(output / "feature_importance.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(output / "prediction_diagnostics.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output / "candidate_coverage.csv", index=False, encoding="utf-8-sig")
    pd.concat([
        pd.DataFrame([asdict(fold) for fold in leading_folds]).assign(candidate_mode="leading"),
        pd.DataFrame([asdict(fold) for fold in breadth_folds]).assign(candidate_mode="leading_breadth"),
    ], ignore_index=True).to_csv(output / "walk_forward_folds.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output / "variant_metrics.csv", index=False, encoding="utf-8-sig")
    ensemble.to_csv(output / "ensemble_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "ensemble_monthly_returns.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(output / "planned_comparisons.csv", index=False, encoding="utf-8-sig")
    daily.to_parquet(output / "variant_daily.parquet", index=False)
    trades.to_parquet(output / "variant_trades.parquet", index=False)
    targets.to_parquet(output / "variant_targets.parquet", index=False)
    selections.to_parquet(output / "concept_selections.parquet", index=False)
    for name, payload in (("decision.json", decision), ("audit.json", audit), ("manifest.json", manifest)):
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "research_report.md").write_text(
        render_report(decision, ensemble, comparisons, monthly, diagnostics, coverage, importance, audit, manifest),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "decision": decision}, ensure_ascii=False, indent=2), flush=True)


def momentum_scores(dataset: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    output = dataset.loc[dataset["trade_date"].between(start, end), [
        "trade_date", "concept_code", "concept_name", "basket_stocks", "basket_stock_count",
        "rotation_momentum_score", "basket_forward_gross_5d", "basket_forward_net_5d",
    ]].copy()
    output["momentum_rank"] = output.groupby("trade_date", observed=True)["rotation_momentum_score"].rank(pct=True)
    output["model_rank"] = output["momentum_rank"]
    output["blend_rank"] = output["momentum_rank"]
    output["placebo_rank"] = output["momentum_rank"]
    output["fold"] = -1
    return output


def candidate_coverage(datasets: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(datasets["momentum"].loc[
        datasets["momentum"]["trade_date"].between(start, end), "trade_date"
    ].unique()))
    rows = []
    for mode, frame in datasets.items():
        counts = frame.loc[frame["trade_date"].between(start, end)].groupby("trade_date", observed=True).size().reindex(dates, fill_value=0)
        rows.append({
            "candidate_mode": mode, "calendar_dates": int(len(dates)),
            "active_dates": int(counts.gt(0).sum()), "active_date_share": float(counts.gt(0).mean()),
            "median_candidates": float(counts.median()), "p10_candidates": float(counts.quantile(0.10)),
        })
    return pd.DataFrame(rows)


def planned_comparisons(ensemble: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    pairs = {
        "leading_filter_M1_minus_M0": ("M1_leading_momentum", "M0_momentum"),
        "lgbm_in_leading_M2_minus_M1": ("M2_leading_lgbm", "M1_leading_momentum"),
        "breadth_increment_M3_minus_M1": ("M3_leading_breadth_momentum", "M1_leading_momentum"),
        "full_filter_M3_minus_M0": ("M3_leading_breadth_momentum", "M0_momentum"),
        "lgbm_in_breadth_M4_minus_M3": ("M4_leading_breadth_lgbm", "M3_leading_breadth_momentum"),
    }
    rows = []
    for name, (left, right) in pairs.items():
        for scenario in ensemble["scenario"].unique():
            lhs = ensemble.loc[ensemble["variant"].eq(left) & ensemble["scenario"].eq(scenario)]
            rhs = ensemble.loc[ensemble["variant"].eq(right) & ensemble["scenario"].eq(scenario)]
            if len(lhs) != 1 or len(rhs) != 1:
                continue
            left_month = monthly.loc[monthly["variant"].eq(left) & monthly["scenario"].eq(scenario), ["month", "return"]]
            right_month = monthly.loc[monthly["variant"].eq(right) & monthly["scenario"].eq(scenario), ["month", "return"]]
            joined = left_month.merge(right_month, on="month", suffixes=("_left", "_right"))
            rows.append({
                "comparison": name, "primary": left, "baseline": right, "scenario": scenario,
                "incremental_total_return": float((1 + lhs.iloc[0]["total_return"]) / (1 + rhs.iloc[0]["total_return"]) - 1),
                "drawdown_change": float(lhs.iloc[0]["max_drawdown"] - rhs.iloc[0]["max_drawdown"]),
                "monthly_win_share": float(joined["return_left"].gt(joined["return_right"]).mean()),
            })
    return pd.DataFrame(rows)


def ablation_decision(comparisons: pd.DataFrame, ensemble: pd.DataFrame) -> dict:
    base = comparisons.loc[comparisons["scenario"].eq("base")].set_index("comparison")
    gates = {
        "leading_filter_adds_return": bool(base.loc["leading_filter_M1_minus_M0", "incremental_total_return"] > 0),
        "leading_lgbm_adds_return": bool(base.loc["lgbm_in_leading_M2_minus_M1", "incremental_total_return"] > 0),
        "breadth_filter_adds_return": bool(base.loc["breadth_increment_M3_minus_M1", "incremental_total_return"] > 0),
        "breadth_lgbm_adds_return": bool(base.loc["lgbm_in_breadth_M4_minus_M3", "incremental_total_return"] > 0),
    }
    positive = ensemble.loc[ensemble["scenario"].eq("base")].sort_values("total_return", ascending=False).iloc[0]
    return {
        "best_variant": str(positive["variant"]), "best_total_return": float(positive["total_return"]),
        "gates": gates,
        "decision": "KEEP_ONLY_COMPONENTS_WITH_POSITIVE_INCREMENT",
        "confirmation_status": "UNCONFIRMED_REQUIRES_POST_2026_07_15_FORWARD_TEST",
    }


def experiment_audit(datasets, predictions, leading_folds, breadth_folds, targets, daily, trades) -> dict:
    return {
        "dataset_rows": {mode: int(len(frame)) for mode, frame in datasets.items()},
        "dataset_dates": {mode: int(frame["trade_date"].nunique()) for mode, frame in datasets.items()},
        "prediction_dates": {mode: int(frame["trade_date"].nunique()) for mode, frame in predictions.items()},
        "leading_folds": len(leading_folds), "leading_breadth_folds": len(breadth_folds),
        "duplicate_target_keys": int(targets.duplicated(["variant", "offset", "entry_date", "ts_code"]).sum()),
        "maximum_target_weight": float(targets["target_weight"].max()),
        "maximum_target_exposure": float(targets.groupby(["variant", "offset", "entry_date"])["target_weight"].sum().max()),
        "nan_nav": int(daily["nav"].isna().sum()),
        "maximum_participation": float(trades["participation"].max()),
    }


def render_report(decision, ensemble, comparisons, monthly, diagnostics, coverage, importance, audit, manifest) -> str:
    base_comparisons = comparisons.loc[comparisons["scenario"].eq("base")]
    top_importance = importance.groupby(["candidate_mode", "feature"], observed=True)["gain"].mean().reset_index().sort_values(
        ["candidate_mode", "gain"], ascending=[True, False]
    ).groupby("candidate_mode", observed=True).head(12)
    return f"""# Leading-zone hard filter and LightGBM ablation

## Decision

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```

The leading condition is a pre-model hard gate: RRG leading, relative-strength rank at least 85%,
positive 20-day momentum, and the existing eligibility rules. The breadth gate additionally requires
free-float breadth above 50% and common-membership breadth delta rank at least 70%. No weakening concept
is used as fallback; fewer than three candidates means cash for that sleeve.

## Five-sleeve account results

```text
{ensemble.to_string(index=False)}
```

## Planned incremental comparisons

```text
{base_comparisons.to_string(index=False)}
```

## Candidate coverage

```text
{coverage.to_string(index=False)}
```

## Prediction and placebo diagnostics

```text
{diagnostics.to_string(index=False)}
```

## Monthly base-scenario returns

```text
{monthly.loc[monthly['scenario'].eq('base')].to_string(index=False)}
```

## Mean feature gain

```text
{top_importance.to_string(index=False)}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leading-zone LightGBM candidate-pool ablation")
    parser.add_argument("--snapshot-root", action="append", default=[
        "data/concept_rotation/dc_20241230_20250627_by_date",
        "data/concept_rotation/dc_20250630_20260714",
    ])
    parser.add_argument("--daily-basic", action="append", default=[
        "data/concept_rotation/daily_basic_20241101_20250430/daily_basic_enhanced.parquet",
        "data/concept_rotation/daily_basic_20250501_20260714/daily_basic_enhanced.parquet",
    ])
    parser.add_argument("--base-panel", default="data/versions/data_v1_20260710T115133Z_208e9fc8/curated/stock_daily_panel.parquet")
    parser.add_argument("--increment-panel", default="data/versions/data_v1_20260715T053637Z_77138057/curated/stock_daily_panel.parquet")
    parser.add_argument("--panel-start", default="2024-11-01")
    parser.add_argument("--output-root", default="artifacts/concept_rotation_lightgbm_leading")
    parser.add_argument("--initial-cash", type=float, default=10_000_000)
    parser.add_argument("--commission-bps", type=float, default=2.5)
    parser.add_argument("--base-slippage-bps", type=float, default=3.0)
    parser.add_argument("--adv-participation", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    main()
