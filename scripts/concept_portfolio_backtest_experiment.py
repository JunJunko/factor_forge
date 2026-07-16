from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from factor_forge.research.concept_portfolio_backtest import (
    ExecutionRules,
    PortfolioRules,
    SIGNAL_COLUMNS,
    build_liquidity_neutral_targets,
    paired_portfolio_comparison,
    prepare_execution_panel,
    run_non_overlapping_ledger,
)
from factor_forge.research.concept_rotation_alpha import load_dc_snapshot


PANEL_COLUMNS = [
    "trade_date", "ts_code", "raw_open", "adj_open", "adj_close", "amount_cny",
    "is_suspended", "is_limit_up_open", "is_limit_down_open", "is_st",
    "is_delisting_period", "listing_trade_days", "is_tradeable",
]
SPLITS = {
    "discovery": ("2025-10-01", "2025-12-31"),
    "validation": ("2026-01-01", "2026-03-31"),
    "final_holdout_reused_diagnostic": ("2026-04-01", "2026-06-30"),
}
PRIMARY = "common_membership_breadth_rrg"
BASELINE = "rrg_only"


def main() -> None:
    args = parse_args()
    portfolio_rules = PortfolioRules(
        initial_cash=args.initial_cash,
        concepts_per_rebalance=args.concepts,
        stocks_per_concept=args.stocks_per_concept,
        maximum_stock_weight=args.maximum_stock_weight,
        minimum_adv20_cny=args.minimum_adv20,
    )
    base_execution = ExecutionRules(
        commission_bps_per_side=args.commission_bps,
        minimum_commission_cny=args.minimum_commission,
        base_slippage_bps_per_side=args.base_slippage_bps,
        maximum_adv_participation=args.adv_participation,
    )
    scenarios = execution_scenarios(base_execution)
    panel = prepare_execution_panel(load_stock_panel(Path(args.base_panel), Path(args.increment_panel)))
    feature_path = resolve_feature_path(Path(args.features))
    features = pd.read_parquet(feature_path)
    features["trade_date"] = pd.to_datetime(features["trade_date"])
    _, members = load_dc_snapshot(
        args.snapshot_root, trade_dates=panel["trade_date"].unique()
    )

    run_id = datetime.now(timezone.utc).strftime("concept_portfolio_%Y%m%dT%H%M%SZ")
    output = Path(args.output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    metric_rows, daily_frames, trade_frames, target_frames, selection_frames = [], [], [], [], []
    daily_lookup: dict[tuple[str, str, int, str], pd.DataFrame] = {}

    total_builds = len(SPLITS) * portfolio_rules.holding_days * len(SIGNAL_COLUMNS)
    build_number = 0
    for split, (start, end) in SPLITS.items():
        for offset in range(portfolio_rules.holding_days):
            for signal in SIGNAL_COLUMNS:
                build_number += 1
                targets = build_liquidity_neutral_targets(
                    features, members, panel, signal_name=signal,
                    start=start, end=end, offset=offset, rules=portfolio_rules,
                )
                if not targets.targets.empty:
                    target_frame = targets.targets.copy()
                    target_frame.insert(0, "signal", signal)
                    target_frame.insert(1, "split", split)
                    target_frame.insert(2, "offset", offset)
                    target_frames.append(target_frame)
                if not targets.selections.empty:
                    selected_frame = targets.selections.copy()
                    selected_frame.insert(0, "signal", signal)
                    selected_frame.insert(1, "split", split)
                    selected_frame.insert(2, "offset", offset)
                    selection_frames.append(selected_frame)
                names = list(scenarios) if signal in {PRIMARY, BASELINE} else ["base"]
                for scenario in names:
                    result = run_non_overlapping_ledger(
                        panel, targets, start=start, end=end,
                        portfolio_rules=portfolio_rules,
                        execution_rules=scenarios[scenario],
                    )
                    metric_rows.append({
                        "signal": signal, "split": split, "offset": offset,
                        "scenario": scenario, **result.metrics,
                    })
                    if not result.daily.empty:
                        daily = result.daily.copy()
                        daily.insert(0, "signal", signal)
                        daily.insert(1, "split", split)
                        daily.insert(2, "offset", offset)
                        daily.insert(3, "scenario", scenario)
                        daily_frames.append(daily)
                        daily_lookup[(signal, split, offset, scenario)] = result.daily
                    if not result.trades.empty:
                        trades = result.trades.copy()
                        trades.insert(0, "signal", signal)
                        trades.insert(1, "split", split)
                        trades.insert(2, "offset", offset)
                        trades.insert(3, "scenario", scenario)
                        trade_frames.append(trades)
                if build_number == 1 or build_number % 10 == 0 or build_number == total_builds:
                    print(f"portfolio targets {build_number}/{total_builds} split={split} offset={offset} signal={signal}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    aggregate = aggregate_metrics(metrics)
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    targets_all = pd.concat(target_frames, ignore_index=True) if target_frames else pd.DataFrame()
    selections = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    exposure_audit = target_exposure_audit(targets_all, selections)
    comparisons = [
        paired_portfolio_comparison(daily_lookup, primary=PRIMARY, baseline=BASELINE),
        paired_portfolio_comparison(daily_lookup, primary="current_membership_breadth_rrg", baseline=BASELINE),
        paired_portfolio_comparison(daily_lookup, primary=PRIMARY, baseline="momentum_20d"),
    ]
    paired = pd.concat(comparisons, ignore_index=True)
    decision = retention_decision(metrics, paired)

    metrics.to_csv(output / "portfolio_metrics.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(output / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(output / "paired_signal_comparison.csv", index=False, encoding="utf-8-sig")
    exposure_audit.to_csv(output / "target_exposure_audit.csv", index=False, encoding="utf-8-sig")
    daily.to_parquet(output / "portfolio_daily.parquet", index=False)
    trades.to_parquet(output / "portfolio_trades.parquet", index=False)
    targets_all.to_parquet(output / "stock_targets.parquet", index=False)
    selections.to_parquet(output / "concept_selections.parquet", index=False)
    (output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_path": str(feature_path), "snapshot_root": str(args.snapshot_root),
        "portfolio_rules": portfolio_rules.__dict__,
        "base_execution_rules": base_execution.__dict__,
        "scenarios": {name: value.__dict__ for name, value in scenarios.items()},
        "splits": SPLITS,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "research_report.md").write_text(
        render_report(decision, aggregate, paired, exposure_audit, manifest), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(output), "decision": decision}, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Non-overlapping Tushare concept stock portfolio backtest")
    parser.add_argument("--features", default="artifacts/concept_rotation_alpha")
    parser.add_argument("--snapshot-root", default="data/concept_rotation/dc_20250630_20260714")
    parser.add_argument(
        "--base-panel",
        default="data/versions/data_v1_20260710T115133Z_208e9fc8/curated/stock_daily_panel.parquet",
    )
    parser.add_argument(
        "--increment-panel",
        default="data/versions/data_v1_20260715T053637Z_77138057/curated/stock_daily_panel.parquet",
    )
    parser.add_argument("--output-root", default="artifacts/concept_portfolio_backtest")
    parser.add_argument("--initial-cash", type=float, default=10_000_000)
    parser.add_argument("--concepts", type=int, default=10)
    parser.add_argument("--stocks-per-concept", type=int, default=3)
    parser.add_argument("--maximum-stock-weight", type=float, default=0.05)
    parser.add_argument("--minimum-adv20", type=float, default=20_000_000)
    parser.add_argument("--commission-bps", type=float, default=2.5)
    parser.add_argument("--minimum-commission", type=float, default=5.0)
    parser.add_argument("--base-slippage-bps", type=float, default=3.0)
    parser.add_argument("--adv-participation", type=float, default=0.05)
    return parser.parse_args()


def load_stock_panel(base_path: Path, increment_path: Path) -> pd.DataFrame:
    base = pd.read_parquet(
        base_path, columns=PANEL_COLUMNS,
        filters=[("trade_date", ">=", pd.Timestamp("2025-05-01"))],
    )
    increment = pd.read_parquet(increment_path, columns=PANEL_COLUMNS)
    for frame in (base, increment):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    last_listing = base.groupby("ts_code", observed=True)["listing_trade_days"].max()
    increment = increment.sort_values(["ts_code", "trade_date"])
    increment["listing_trade_days"] = (
        increment.groupby("ts_code", observed=True).cumcount() + 1
        + increment["ts_code"].map(last_listing).fillna(0)
    )
    increment["is_tradeable"] = (
        increment["listing_trade_days"].ge(60)
        & ~increment["is_suspended"].fillna(True)
        & ~increment["is_st"].fillna(False)
        & ~increment["is_delisting_period"].fillna(False)
        & increment["adj_open"].notna() & increment["adj_close"].notna()
    )
    return pd.concat([base, increment], ignore_index=True).sort_values(
        ["trade_date", "ts_code"]
    ).drop_duplicates(["trade_date", "ts_code"], keep="last").reset_index(drop=True)


def resolve_feature_path(path: Path) -> Path:
    if path.is_file():
        return path
    runs = sorted(path.glob("concept_rotation_*/concept_daily_features_pit.parquet"))
    if not runs:
        raise FileNotFoundError(f"no PIT concept feature artifact under {path}")
    return runs[-1]


def execution_scenarios(base: ExecutionRules) -> dict[str, ExecutionRules]:
    return {
        "base": base,
        "cost_1_5x": replace(base, cost_multiplier=1.5),
        "cost_2x": replace(base, cost_multiplier=2.0),
        "extra_20bps_roundtrip": replace(base, extra_slippage_bps_per_side=10.0),
        "capacity_1pct_adv": replace(base, maximum_adv_participation=0.01),
        "capacity_2pct_adv": replace(base, maximum_adv_participation=0.02),
        "capacity_10pct_adv": replace(base, maximum_adv_participation=0.10),
    }


def retention_decision(metrics: pd.DataFrame, paired: pd.DataFrame) -> dict:
    final_name = "final_holdout_reused_diagnostic"
    def summary(split: str, scenario: str, baseline: str) -> dict:
        frame = paired.loc[
            paired["split"].eq(split) & paired["scenario"].eq(scenario)
            & paired["primary"].eq(PRIMARY) & paired["baseline"].eq(baseline)
        ]
        return {
            "offsets": int(len(frame)),
            "positive_offsets": int(frame["relative_total_return"].gt(0).sum()),
            "mean_relative_total_return": float(frame["relative_total_return"].mean()) if len(frame) else None,
            "median_relative_total_return": float(frame["relative_total_return"].median()) if len(frame) else None,
            "worst_relative_total_return": float(frame["relative_total_return"].min()) if len(frame) else None,
            "median_incremental_nw_t": float(frame["incremental_nw_t"].median()) if len(frame) else None,
        }
    final_base = summary(final_name, "base", BASELINE)
    validation = summary("validation", "base", BASELINE)
    doubled = summary(final_name, "cost_2x", BASELINE)
    extra = summary(final_name, "extra_20bps_roundtrip", BASELINE)
    capacity = {
        name: summary(final_name, name, BASELINE)
        for name in ("capacity_1pct_adv", "capacity_2pct_adv", "capacity_10pct_adv")
    }
    criteria = {
        "final_positive_4_of_5_offsets": final_base["positive_offsets"] >= 4,
        "validation_positive_3_of_5_offsets": validation["positive_offsets"] >= 3,
        "final_mean_increment_positive": (final_base["mean_relative_total_return"] or 0) > 0,
        "double_cost_mean_increment_positive": (doubled["mean_relative_total_return"] or 0) > 0,
        "extra_20bps_mean_increment_positive": (extra["mean_relative_total_return"] or 0) > 0,
    }
    keep = all(criteria.values())
    risk_rows = []
    for split in ("validation", final_name):
        frame = metrics.loc[metrics["split"].eq(split) & metrics["scenario"].eq("base")]
        primary_metrics = frame.loc[frame["signal"].eq(PRIMARY)].mean(numeric_only=True)
        baseline_metrics = frame.loc[frame["signal"].eq(BASELINE)].mean(numeric_only=True)
        risk_rows.append({
            "split": split,
            "volatility_delta": float(primary_metrics["annualized_volatility"] - baseline_metrics["annualized_volatility"]),
            "max_drawdown_delta": float(primary_metrics["max_drawdown"] - baseline_metrics["max_drawdown"]),
            "turnover_delta": float(primary_metrics["annualized_turnover"] - baseline_metrics["annualized_turnover"]),
            "cash_ratio_delta": float(primary_metrics["average_cash_ratio"] - baseline_metrics["average_cash_ratio"]),
        })
    lower_volatility_both = all(row["volatility_delta"] < 0 for row in risk_rows)
    drawdown_improved_both = all(row["max_drawdown_delta"] > 0 for row in risk_rows)
    turnover_not_worse_both = all(row["turnover_delta"] <= 0 for row in risk_rows)
    risk_filter_candidate = (
        not keep and lower_volatility_both and drawdown_improved_both and turnover_not_worse_both
    )
    return {
        "decision": "KEEP_AS_ALPHA" if keep else "DO_NOT_KEEP_AS_ALPHA",
        "risk_filter_status": "RETEST_AS_RISK_FILTER" if risk_filter_candidate else "REJECT_UNDER_CURRENT_RULE",
        "risk_filter_evidence": risk_rows,
        "criteria": criteria,
        "primary_vs_rrg": {
            "validation": validation, "final_reused_diagnostic": final_base,
            "double_cost": doubled, "extra_20bps_roundtrip": extra,
            "capacity": capacity,
        },
        "warning": "The former final holdout was already inspected; this run is diagnostic, not new confirmatory evidence.",
    }


def target_exposure_audit(targets: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    if targets.empty:
        return pd.DataFrame()
    daily = targets.groupby(["signal", "split", "offset", "entry_date"], observed=True).agg(
        stocks=("ts_code", "nunique"), target_exposure=("target_weight", "sum"),
        maximum_target_weight=("target_weight", "max"),
    ).reset_index()
    if not selections.empty:
        concept_counts = selections.groupby(
            ["signal", "split", "offset", "signal_date"], observed=True
        )["concept_code"].nunique().reset_index(name="concepts")
        concept_counts = concept_counts.rename(columns={"signal_date": "entry_signal_date"})
        daily = daily.sort_values(["signal", "split", "offset", "entry_date"])
        concepts = concept_counts.groupby(["signal", "split", "offset"], observed=True)["concepts"].mean()
    else:
        concepts = pd.Series(dtype=float)
    audit = daily.groupby(["signal", "split"], observed=True).agg(
        rebalance_count=("entry_date", "size"), average_stocks=("stocks", "mean"),
        average_target_exposure=("target_exposure", "mean"),
        minimum_target_exposure=("target_exposure", "min"),
        maximum_target_weight=("maximum_target_weight", "max"),
    ).reset_index()
    offset_concepts = concepts.groupby(level=[0, 1]).mean().rename("average_concepts").reset_index()
    return audit.merge(offset_concepts, on=["signal", "split"], how="left")


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby(["signal", "split", "scenario"], observed=True).agg(
        offsets=("offset", "nunique"), positive_offsets=("total_return", lambda values: int(values.gt(0).sum())),
        mean_total_return=("total_return", "mean"), median_total_return=("total_return", "median"),
        worst_total_return=("total_return", "min"), mean_max_drawdown=("max_drawdown", "mean"),
        mean_annualized_volatility=("annualized_volatility", "mean"),
        mean_annualized_turnover=("annualized_turnover", "mean"),
        mean_cash_ratio=("average_cash_ratio", "mean"), mean_cost_drag=("cost_drag", "mean"),
        maximum_actual_weight=("maximum_position_weight", "max"),
        maximum_participation=("p95_participation", "max"),
    ).reset_index()


def render_report(
    decision: dict, aggregate: pd.DataFrame, paired: pd.DataFrame,
    exposure_audit: pd.DataFrame, manifest: dict,
) -> str:
    final_name = "final_holdout_reused_diagnostic"
    focus_metrics = aggregate.loc[
        aggregate["split"].isin(["validation", final_name])
        & aggregate["scenario"].eq("base")
    ]
    focus_paired = paired.loc[
        paired["primary"].eq(PRIMARY)
        & paired["baseline"].eq(BASELINE)
        & paired["split"].isin(["validation", final_name])
    ]
    return f"""# Concept rotation non-overlapping stock portfolio backtest

## Decision

`{decision['decision']}`

This is a five-trading-day, non-overlapping, five-offset account-level simulation. The primary
question is whether common-member breadth + RRG improves the same liquidity-neutral stock
portfolio over RRG-only after actual turnover costs. The former final holdout has already been
seen, so its result is diagnostic and a future untouched sample is still required.

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```

## Frozen implementation

- Signal at T close; target orders begin at T+1 open.
- Ten Jaccard-deduplicated concepts; three highest-ADV20 stocks per concept.
- Duplicate stocks are consolidated, individual weights capped at 5%, and residual cash retained.
- Suspended/limit-up buys and suspended/limit-down sells remain pending and retry on later days.
- 100-share lots, minimum commission, sell-side stamp duty, transfer fee, volatility/ADV impact,
  and 1%/2%/5%/10% ADV capacity are modeled.

## Validation and reused-final portfolio metrics

```text
{focus_metrics.to_string(index=False)}
```

## Paired increment over RRG-only

```text
{focus_paired.to_string(index=False)}
```

## Target exposure audit

The breadth + RRG rule can select fewer than ten concepts. Residual weight remains cash because
the 5% stock cap is not relaxed. This is part of the executable rule, but means its return gap
contains both concept selection and exposure timing.

```text
{exposure_audit.to_string(index=False)}
```

## Manifest

```json
{json.dumps(manifest, ensure_ascii=False, indent=2)}
```
"""


if __name__ == "__main__":
    main()
