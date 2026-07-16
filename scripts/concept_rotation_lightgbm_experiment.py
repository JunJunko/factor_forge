from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

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


PANEL_COLUMNS = [
    "trade_date", "ts_code", "raw_open", "raw_close", "adj_open", "adj_close", "amount_cny",
    "circ_mv_cny", "turnover_rate", "is_suspended", "is_limit_up_open", "is_limit_down_open",
    "is_st", "is_delisting_period", "listing_trade_days", "is_tradeable",
]
VARIANTS = ["momentum_baseline", "lgbm_direct", "lgbm_blend_20", "label_shuffle_placebo"]


def main() -> None:
    args = parse_args()
    output = Path(args.output_root) / datetime.now(timezone.utc).strftime("concept_lgbm_%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    panel = attach_enhanced_stock_features(
        load_stock_panel(Path(args.base_panel), Path(args.increment_panel), pd.Timestamp(args.panel_start)),
        load_daily_basic(args.daily_basic),
    )
    regimes = build_document_market_regimes(panel)
    snapshot_roots = args.snapshot_root or ["data/concept_rotation/dc_20250630_20260714"]
    concept_index, members = load_dc_snapshot_roots(
        snapshot_roots, trade_dates=panel["trade_date"].unique()
    )
    if args.rebuild_features:
        _, features, feature_audit = build_concept_dataset(panel, concept_index, members)
        features = attach_lifecycle_fields(features)
    else:
        features = pd.read_parquet(args.features)
        feature_audit = {"source": "cached", "path": args.features}
    rules = ConceptMLRules()
    dataset = build_concept_ml_dataset(features, members, panel, regimes, rules=rules)
    predictions, importance, folds = fit_walk_forward_rankers(dataset, rules=rules)
    if predictions.empty:
        raise RuntimeError("walk-forward produced no OOF predictions")
    diagnostics = prediction_diagnostics(predictions)
    start = pd.Timestamp(predictions["trade_date"].min())
    end = pd.Timestamp(predictions["trade_date"].max())

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
    metrics_rows: list[dict] = []
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    target_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    total = len(VARIANTS) * rules.horizon
    progress = 0
    for offset in range(rules.horizon):
        for variant in VARIANTS:
            progress += 1
            targets = build_ml_rotation_targets(
                predictions, members, regimes, panel, variant=variant,
                start=start, end=end, offset=offset, rules=rules,
            )
            if not targets.targets.empty:
                item = targets.targets.copy(); item.insert(0, "offset", offset); target_frames.append(item)
            if not targets.selections.empty:
                item = targets.selections.copy(); item.insert(0, "offset", offset); selection_frames.append(item)
            for scenario, execution_rules in scenarios.items():
                result = run_non_overlapping_ledger(
                    panel, targets, start=start, end=end,
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
            print(f"LightGBM backtest {progress}/{total} offset={offset} variant={variant}", flush=True)

    metrics = pd.DataFrame(metrics_rows)
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    targets = pd.concat(target_frames, ignore_index=True) if target_frames else pd.DataFrame()
    selections = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    ensemble, monthly = ensemble_results(daily, args.initial_cash)
    comparisons = compare_to_baseline(ensemble, monthly)
    decision = experiment_decision(ensemble, monthly)
    audit = experiment_audit(dataset, predictions, folds, targets, daily, trades)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURE_COLUMNS,
        "rules": asdict(rules),
        "portfolio_rules": asdict(portfolio_rules),
        "execution_rules": asdict(execution),
        "oof_start": start.strftime("%Y-%m-%d"), "oof_end": end.strftime("%Y-%m-%d"),
        "research_status": "DIAGNOSTIC_ALL_AVAILABLE_HISTORY_PREVIOUSLY_INSPECTED",
        "snapshot_roots": snapshot_roots, "feature_audit": feature_audit,
    }

    dataset.to_parquet(output / "concept_ml_dataset.parquet", index=False)
    predictions.to_parquet(output / "oof_predictions.parquet", index=False)
    importance.to_csv(output / "feature_importance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([asdict(fold) for fold in folds]).to_csv(output / "walk_forward_folds.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(output / "prediction_diagnostics.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output / "variant_metrics.csv", index=False, encoding="utf-8-sig")
    ensemble.to_csv(output / "ensemble_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "ensemble_monthly_returns.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(output / "comparison_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_parquet(output / "variant_daily.parquet", index=False)
    trades.to_parquet(output / "variant_trades.parquet", index=False)
    targets.to_parquet(output / "variant_targets.parquet", index=False)
    selections.to_parquet(output / "concept_selections.parquet", index=False)
    (output / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "research_report.md").write_text(
        render_report(decision, diagnostics, ensemble, monthly, comparisons, importance, audit, manifest),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "decision": decision}, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward LightGBM concept momentum reranker")
    parser.add_argument("--features", default="artifacts/concept_rotation_lifecycle/concept_lifecycle_20260715T123602Z/concept_features_free_float.parquet")
    parser.add_argument("--snapshot-root", action="append", default=[])
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--daily-basic", action="append", default=[])
    parser.add_argument("--base-panel", default="data/versions/data_v1_20260710T115133Z_208e9fc8/curated/stock_daily_panel.parquet")
    parser.add_argument("--increment-panel", default="data/versions/data_v1_20260715T053637Z_77138057/curated/stock_daily_panel.parquet")
    parser.add_argument("--panel-start", default="2025-05-01")
    parser.add_argument("--output-root", default="artifacts/concept_rotation_lightgbm")
    parser.add_argument("--initial-cash", type=float, default=10_000_000)
    parser.add_argument("--commission-bps", type=float, default=2.5)
    parser.add_argument("--base-slippage-bps", type=float, default=3.0)
    parser.add_argument("--adv-participation", type=float, default=0.05)
    return parser.parse_args()


def load_daily_basic(paths: list[str]) -> pd.DataFrame:
    sources = paths or ["data/concept_rotation/daily_basic_20250501_20260714/daily_basic_enhanced.parquet"]
    frames = [pd.read_parquet(path) for path in sources]
    return pd.concat(frames, ignore_index=True).sort_values(["trade_date", "ts_code"]).drop_duplicates(
        ["trade_date", "ts_code"], keep="last"
    )


def load_stock_panel(base_path: Path, increment_path: Path, start: pd.Timestamp) -> pd.DataFrame:
    base = pd.read_parquet(base_path, columns=PANEL_COLUMNS, filters=[("trade_date", ">=", start)])
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


def ensemble_results(daily: pd.DataFrame, initial_cash: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows, monthly_rows = [], []
    for (variant, scenario), group in daily.groupby(["variant", "scenario"], observed=True):
        pivot = group.pivot(index="trade_date", columns="offset", values="nav").sort_index().ffill().fillna(initial_cash)
        nav = pivot.sum(axis=1) / pivot.shape[1]
        returns = nav.pct_change().fillna(nav.iloc[0] / initial_cash - 1)
        total = float(nav.iloc[-1] / initial_cash - 1)
        drawdown = float((nav / nav.cummax() - 1).min())
        summary_rows.append({
            "variant": variant, "scenario": scenario, "days": len(nav),
            "total_return": total, "max_drawdown": drawdown,
            "annualized_volatility": float(returns.std(ddof=1) * (252**0.5)),
        })
        monthly = (1 + returns).groupby(pd.to_datetime(returns.index).to_period("M")).prod() - 1
        for month, value in monthly.items():
            monthly_rows.append({"variant": variant, "scenario": scenario, "month": str(month), "return": float(value)})
    return pd.DataFrame(summary_rows), pd.DataFrame(monthly_rows)


def compare_to_baseline(ensemble: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, scenario), item in ensemble.loc[~ensemble["variant"].eq("momentum_baseline")].groupby(["variant", "scenario"], observed=True):
        base = ensemble.loc[ensemble["variant"].eq("momentum_baseline") & ensemble["scenario"].eq(scenario)]
        if len(item) != 1 or len(base) != 1:
            continue
        left = monthly.loc[monthly["variant"].eq(variant) & monthly["scenario"].eq(scenario), ["month", "return"]]
        right = monthly.loc[monthly["variant"].eq("momentum_baseline") & monthly["scenario"].eq(scenario), ["month", "return"]]
        joined = left.merge(right, on="month", suffixes=("_model", "_baseline"))
        rows.append({
            "variant": variant, "scenario": scenario,
            "incremental_total_return": float((1 + item.iloc[0]["total_return"]) / (1 + base.iloc[0]["total_return"]) - 1),
            "drawdown_change": float(item.iloc[0]["max_drawdown"] - base.iloc[0]["max_drawdown"]),
            "monthly_win_share": float(joined["return_model"].gt(joined["return_baseline"]).mean()),
        })
    return pd.DataFrame(rows)


def experiment_decision(ensemble: pd.DataFrame, monthly: pd.DataFrame) -> dict:
    base = ensemble.loc[ensemble["variant"].eq("momentum_baseline") & ensemble["scenario"].eq("base")].iloc[0]
    blend = ensemble.loc[ensemble["variant"].eq("lgbm_blend_20") & ensemble["scenario"].eq("base")].iloc[0]
    stress = ensemble.loc[ensemble["variant"].eq("lgbm_blend_20") & ensemble["scenario"].isin(["cost_2x", "extra_20bps_roundtrip"])]
    model_monthly = monthly.loc[monthly["variant"].eq("lgbm_blend_20") & monthly["scenario"].eq("base"), ["month", "return"]]
    base_monthly = monthly.loc[monthly["variant"].eq("momentum_baseline") & monthly["scenario"].eq("base"), ["month", "return"]]
    joined = model_monthly.merge(base_monthly, on="month", suffixes=("_model", "_base"))
    incremental = float((1 + blend["total_return"]) / (1 + base["total_return"]) - 1)
    gates = {
        "positive_incremental_return": bool(incremental > 0),
        "incremental_at_least_3pp": bool(incremental >= 0.03),
        "drawdown_not_worse": bool(blend["max_drawdown"] >= base["max_drawdown"]),
        "stress_returns_positive": len(stress) == 2 and bool(stress["total_return"].gt(0).all()),
        "monthly_win_share_at_least_60pct": len(joined) > 0 and float(joined["return_model"].gt(joined["return_base"]).mean()) >= 0.60,
    }
    return {
        "decision": "OOF_ENHANCEMENT_CANDIDATE" if all(gates.values()) else "DO_NOT_REPLACE_MOMENTUM_BASELINE",
        "incremental_total_return": incremental,
        "gates": gates,
        "confirmation_status": "UNCONFIRMED_REQUIRES_POST_2026_07_15_FORWARD_TEST",
    }


def experiment_audit(dataset, predictions, folds, targets, daily, trades) -> dict:
    fold_frame = pd.DataFrame([asdict(fold) for fold in folds])
    return {
        "dataset_rows": int(len(dataset)), "dataset_dates": int(dataset["trade_date"].nunique()),
        "oof_rows": int(len(predictions)), "oof_dates": int(predictions["trade_date"].nunique()),
        "folds": int(len(folds)),
        "fold_boundaries_non_overlapping": bool((fold_frame["train_end"] < fold_frame["valid_start"]).all() and (fold_frame["valid_end"] < fold_frame["test_start"]).all()),
        "duplicate_prediction_keys": int(predictions.duplicated(["trade_date", "concept_code"]).sum()),
        "nan_model_scores": int(predictions["lgb_score"].isna().sum()),
        "maximum_target_weight": float(targets["target_weight"].max()) if not targets.empty else None,
        "maximum_target_exposure": float(targets.groupby(["variant", "offset", "entry_date"])["target_weight"].sum().max()) if not targets.empty else None,
        "nan_nav": int(daily["nav"].isna().sum()) if not daily.empty else 0,
        "maximum_participation": float(trades["participation"].max()) if not trades.empty else None,
    }


def render_report(decision, diagnostics, ensemble, monthly, comparisons, importance, audit, manifest) -> str:
    top_importance = importance.groupby("feature", observed=True)["gain"].mean().sort_values(ascending=False).head(15)
    return f"""# Concept momentum LightGBM walk-forward reranker

## Decision

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```

LightGBM only reranks the top-20 positive trailing-20-day momentum concepts. Portfolio construction,
three liquid core stocks per concept, five non-overlapping sleeves, market-state exposure, T+1 open
execution, position caps, blocked orders and impact costs remain frozen. The primary variant blends
80% momentum rank with 20% out-of-fold model rank.

All available history has already been inspected during prior research. These are purged walk-forward
diagnostics, not a clean confirmation; confirmation begins after 2026-07-15.

## Prediction diagnostics

```text
{diagnostics.to_string(index=False)}
```

## Five-sleeve account results

```text
{ensemble.to_string(index=False)}
```

## Increment versus momentum baseline

```text
{comparisons.to_string(index=False)}
```

## Monthly returns

```text
{monthly.loc[monthly['scenario'].eq('base')].to_string(index=False)}
```

## Mean feature gain

```text
{top_importance.to_string()}
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
