from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from factor_forge.research.concept_lifecycle_backtest import (
    attach_enhanced_stock_features,
    build_document_market_regimes,
)
from factor_forge.research.concept_portfolio_backtest import (
    ExecutionRules, PortfolioRules, paired_portfolio_comparison, run_non_overlapping_ledger,
)
from factor_forge.research.concept_rotation_alpha import load_dc_snapshot
from factor_forge.research.narrative_rotation import (
    NarrativeRules, attach_rotation_signals, build_narrative_rotation_targets,
)


PANEL_COLUMNS = [
    "trade_date", "ts_code", "raw_open", "raw_close", "adj_open", "adj_close", "amount_cny",
    "circ_mv_cny", "turnover_rate", "is_suspended", "is_limit_up_open", "is_limit_down_open",
    "is_st", "is_delisting_period", "listing_trade_days", "is_tradeable",
]


SPLITS = {
    "pre_narrative_2025Q4": ("2025-10-01", "2025-12-31"),
    "narrative_replay_2026": ("2026-01-01", "2026-07-14"),
}
VARIANTS = ["momentum_baseline", "causal_leader", "causal_leader_successor", "narrative_assisted"]
COMPARISONS = {
    "leader_vs_momentum": ("causal_leader", "momentum_baseline"),
    "leader_successor_vs_momentum": ("causal_leader_successor", "momentum_baseline"),
    "successor_increment": ("causal_leader_successor", "causal_leader"),
    "narrative_assisted_vs_momentum": ("narrative_assisted", "momentum_baseline"),
}


def main() -> None:
    args = parse_args()
    output = Path(args.output_root) / datetime.now(timezone.utc).strftime("narrative_replay_%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    panel = attach_enhanced_stock_features(
        load_stock_panel(Path(args.base_panel), Path(args.increment_panel)),
        pd.read_parquet(args.daily_basic),
    )
    features = attach_rotation_signals(pd.read_parquet(args.features))
    _, members = load_dc_snapshot(args.snapshot_root, trade_dates=panel["trade_date"].unique())
    regimes = build_document_market_regimes(panel)
    rules = NarrativeRules()
    portfolio_rules = PortfolioRules(
        initial_cash=args.initial_cash, concepts_per_rebalance=rules.concepts,
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
    daily_lookup = {}
    total = len(SPLITS) * rules.holding_days * len(VARIANTS)
    progress = 0
    for split, (start, end) in SPLITS.items():
        for offset in range(rules.holding_days):
            for variant in VARIANTS:
                progress += 1
                targets = build_narrative_rotation_targets(
                    features, members, panel, regimes, variant=variant,
                    start=start, end=end, offset=offset, rules=rules,
                )
                if not targets.targets.empty:
                    item = targets.targets.copy(); item.insert(0, "split", split); item.insert(1, "offset", offset)
                    target_frames.append(item)
                if not targets.selections.empty:
                    item = targets.selections.copy(); item.insert(0, "split", split); item.insert(1, "offset", offset)
                    selection_frames.append(item)
                for scenario, execution_rules in scenarios.items():
                    result = run_non_overlapping_ledger(
                        panel, targets, start=start, end=end,
                        portfolio_rules=portfolio_rules, execution_rules=execution_rules,
                    )
                    metrics_rows.append({
                        "variant": variant, "split": split, "offset": offset,
                        "scenario": scenario, **result.metrics,
                    })
                    if not result.daily.empty:
                        item = result.daily.copy(); item.insert(0, "variant", variant); item.insert(1, "split", split)
                        item.insert(2, "offset", offset); item.insert(3, "scenario", scenario)
                        daily_frames.append(item)
                        daily_lookup[(variant, split, offset, scenario)] = result.daily
                    if scenario == "base" and not result.trades.empty:
                        item = result.trades.copy(); item.insert(0, "variant", variant); item.insert(1, "split", split)
                        item.insert(2, "offset", offset); trade_frames.append(item)
                print(f"narrative replay {progress}/{total} split={split} offset={offset} variant={variant}", flush=True)

    metrics = pd.DataFrame(metrics_rows)
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    targets = pd.concat(target_frames, ignore_index=True) if target_frames else pd.DataFrame()
    selections = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    comparisons = comparison_table(daily_lookup)
    metric_summary = summarize_metrics(metrics)
    comparison_summary = summarize_comparisons(comparisons)
    capture = theme_capture(selections)
    audit = build_audit(targets, daily, trades, selections)
    ensemble = ensemble_summary(daily, args.initial_cash)
    monthly = ensemble_monthly_returns(daily, args.initial_cash)
    decision = replay_decision(metric_summary, comparison_summary, ensemble)

    metrics.to_csv(output / "variant_metrics.csv", index=False, encoding="utf-8-sig")
    metric_summary.to_csv(output / "variant_summary.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(output / "paired_comparisons.csv", index=False, encoding="utf-8-sig")
    comparison_summary.to_csv(output / "comparison_summary.csv", index=False, encoding="utf-8-sig")
    ensemble.to_csv(output / "ensemble_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "ensemble_monthly_returns.csv", index=False, encoding="utf-8-sig")
    capture.to_csv(output / "theme_capture.csv", index=False, encoding="utf-8-sig")
    regimes.to_csv(output / "market_regimes.csv", index=False, encoding="utf-8-sig")
    daily.to_parquet(output / "variant_daily.parquet", index=False)
    trades.to_parquet(output / "variant_trades.parquet", index=False)
    targets.to_parquet(output / "variant_targets.parquet", index=False)
    selections.to_parquet(output / "concept_selections.parquet", index=False)
    (output / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "splits": SPLITS,
        "variants": VARIANTS, "comparisons": COMPARISONS, "rules": rules.__dict__,
        "warning": "The 2026 narrative and all replay dates were selected after observation. Profit is hypothesis-generating, not confirmed alpha.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "research_report.md").write_text(
        render_report(decision, metric_summary, comparison_summary, ensemble, monthly, capture, audit, manifest), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(output), "decision": decision}, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reverse replay of the 2026 concept-rotation narrative")
    parser.add_argument("--features", default="artifacts/concept_rotation_lifecycle/concept_lifecycle_20260715T123602Z/concept_features_free_float.parquet")
    parser.add_argument("--snapshot-root", default="data/concept_rotation/dc_20250630_20260714")
    parser.add_argument("--daily-basic", default="data/concept_rotation/daily_basic_20250501_20260714/daily_basic_enhanced.parquet")
    parser.add_argument("--base-panel", default="data/versions/data_v1_20260710T115133Z_208e9fc8/curated/stock_daily_panel.parquet")
    parser.add_argument("--increment-panel", default="data/versions/data_v1_20260715T053637Z_77138057/curated/stock_daily_panel.parquet")
    parser.add_argument("--output-root", default="artifacts/concept_rotation_narrative_replay")
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
            frame.insert(0, "comparison", name); frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    clean = metrics.dropna(subset=["total_return"])
    return clean.groupby(["variant", "split", "scenario"], observed=True).agg(
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
        worst_relative_return=("relative_total_return", "min"), median_nw_t=("incremental_nw_t", "median"),
    ).reset_index()


def theme_capture(selections: pd.DataFrame) -> pd.DataFrame:
    if selections.empty:
        return pd.DataFrame()
    item = selections.copy()
    item["month"] = pd.to_datetime(item["signal_date"]).dt.to_period("M").astype(str)
    return item.groupby(["variant", "month", "concept_code", "concept_name", "role"], observed=True).agg(
        selections=("signal_date", "count"), mean_score=("score", "mean"),
    ).reset_index().sort_values(["variant", "month", "selections"], ascending=[True, True, False]).groupby(
        ["variant", "month"], observed=True
    ).head(10)


def build_audit(targets: pd.DataFrame, daily: pd.DataFrame, trades: pd.DataFrame,
                selections: pd.DataFrame) -> dict:
    return {
        "target_rows": int(len(targets)),
        "duplicate_targets": int(targets.duplicated(["variant", "split", "offset", "entry_date", "ts_code"]).sum()) if not targets.empty else 0,
        "maximum_target_weight": float(targets["target_weight"].max()) if not targets.empty else None,
        "maximum_target_exposure": float(targets.groupby(["variant", "split", "offset", "entry_date"])["target_weight"].sum().max()) if not targets.empty else None,
        "nan_nav": int(daily["nav"].isna().sum()) if not daily.empty else 0,
        "trade_rows": int(len(trades)),
        "maximum_participation": float(trades["participation"].max()) if not trades.empty else None,
        "selected_concepts": int(selections["concept_code"].nunique()) if not selections.empty else 0,
    }


def ensemble_summary(daily: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    rows = []
    for (variant, split, scenario), group in daily.groupby(["variant", "split", "scenario"], observed=True):
        pivot = group.pivot(index="trade_date", columns="offset", values="nav").sort_index()
        pivot = pivot.ffill().fillna(initial_cash)
        nav = pivot.sum(axis=1) / 5
        returns = nav.pct_change().fillna(0.0)
        rows.append({
            "variant": variant, "split": split, "scenario": scenario,
            "total_return": float(nav.iloc[-1] / initial_cash - 1),
            "max_drawdown": float((nav / nav.cummax() - 1).min()),
            "annualized_volatility": float(returns.std(ddof=1) * (252 ** 0.5)),
            "days": int(len(nav)),
        })
    return pd.DataFrame(rows)


def ensemble_monthly_returns(daily: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    rows = []
    for (variant, split, scenario), group in daily.groupby(["variant", "split", "scenario"], observed=True):
        pivot = group.pivot(index="trade_date", columns="offset", values="nav").sort_index().ffill().fillna(initial_cash)
        nav = pivot.sum(axis=1) / 5
        returns = nav.pct_change().fillna(nav.iloc[0] / initial_cash - 1)
        monthly = (1 + returns).groupby(pd.to_datetime(returns.index).to_period("M")).prod() - 1
        for month, value in monthly.items():
            rows.append({"variant": variant, "split": split, "scenario": scenario, "month": str(month), "return": float(value)})
    return pd.DataFrame(rows)


def replay_decision(metrics: pd.DataFrame, comparisons: pd.DataFrame, ensemble: pd.DataFrame) -> dict:
    candidates = []
    for variant in ["momentum_baseline", "causal_leader", "causal_leader_successor"]:
        rows = ensemble.loc[(ensemble["variant"].eq(variant)) & (ensemble["split"].eq("narrative_replay_2026"))]
        base = rows.loc[rows["scenario"].eq("base")]
        stress = rows.loc[rows["scenario"].isin(["cost_2x", "extra_20bps_roundtrip"])]
        if len(base) == 1 and base.iloc[0]["total_return"] > 0 and len(stress) == 2 and stress["total_return"].gt(0).all():
            candidates.append(variant)
    return {
        "profitable_replay_candidates": candidates,
        "decision": "DATA_MINED_REPLAY_CANDIDATE" if candidates else "NO_PROFITABLE_CAUSAL_REPLAY",
        "confirmation_status": "UNCONFIRMED_REQUIRES_FORWARD_TEST",
    }


def render_report(decision: dict, metrics: pd.DataFrame, comparisons: pd.DataFrame,
                  ensemble: pd.DataFrame,
                  monthly: pd.DataFrame,
                  capture: pd.DataFrame, audit: dict, manifest: dict) -> str:
    focus_metrics = metrics.loc[(metrics["scenario"].eq("base"))]
    focus_comparisons = comparisons.loc[(comparisons["scenario"].eq("base"))]
    return f"""# 2026 concept-rotation narrative reverse replay

## Decision

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```

The narrative-assisted strategy knows the ex-post stage dates and theme vocabulary, so it is
an attainable upper-bound diagnostic rather than a tradable alpha test. The causal strategies
receive no dates or theme names: they hold concentrated core stocks in persistent relative
leaders and optionally reserve one slot for an accelerating successor concept.

## Profitable replay candidate: simple concentrated momentum

- Split capital equally into five sleeves; one sleeve rebalances each trading day and holds five days.
- At each sleeve rebalance, rank eligible concepts only by trailing 20-day concept return.
- Select the top three concepts after 0.80 Jaccard de-duplication.
- In each concept, hold three liquid core stocks ranked by 20-day average traded amount and trend.
- Target 90% exposure normally, 60% in overheat, and 30% in broad-market retreat.
- Trade at T+1 open with 10% single-stock cap, lot size, blocked-order and impact-cost simulation.

This is intentionally simpler than the failed leader/successor composite. It is a profitable
historical replay candidate, not yet a deployable claim, because 2026 and the narrative were inspected.

## Portfolio results

```text
{focus_metrics.to_string(index=False)}
```

## Five-sleeve ensemble (primary implementation)

```text
{ensemble.to_string(index=False)}
```

## Ensemble monthly returns

```text
{monthly.loc[(monthly['scenario'].eq('base'))].to_string(index=False)}
```

## Paired results

```text
{focus_comparisons.to_string(index=False)}
```

## Theme capture

```text
{capture.to_string(index=False)}
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
