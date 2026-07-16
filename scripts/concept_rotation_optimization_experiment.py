from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.research.concept_portfolio_backtest import (
    ExecutionRules,
    PortfolioRules,
    TargetBuildResult,
    attach_continuous_breadth_signals,
    blend_target_builds,
    build_liquidity_neutral_targets,
    build_market_regimes,
    paired_portfolio_comparison,
    prepare_execution_panel,
    rescale_target_build,
    run_non_overlapping_ledger,
)
from factor_forge.research.concept_rotation_alpha import load_dc_snapshot


PANEL_COLUMNS = [
    "trade_date", "ts_code", "raw_open", "adj_open", "adj_close", "amount_cny", "circ_mv_cny",
    "is_suspended", "is_limit_up_open", "is_limit_down_open", "is_st",
    "is_delisting_period", "listing_trade_days", "is_tradeable",
]
SPLITS = {
    "discovery": ("2025-10-01", "2025-12-31"),
    "validation": ("2026-01-01", "2026-03-31"),
    "final_reused_diagnostic": ("2026-04-01", "2026-06-30"),
}
RRR = "rrg_original"
def main() -> None:
    args = parse_args()
    portfolio_rules = PortfolioRules(initial_cash=args.initial_cash)
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
    panel = prepare_execution_panel(load_stock_panel(Path(args.base_panel), Path(args.increment_panel)))
    features = attach_continuous_breadth_signals(pd.read_parquet(resolve_feature_path(Path(args.features))))
    features["trade_date"] = pd.to_datetime(features["trade_date"])
    regimes = build_market_regimes(panel)
    regime_map = regimes.set_index("trade_date")["regime"].to_dict()
    _, members = load_dc_snapshot(args.snapshot_root, trade_dates=panel["trade_date"].unique())

    output = Path(args.output_root) / datetime.now(timezone.utc).strftime("concept_optimization_%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    metrics_rows, daily_frames, trade_frames, target_frames = [], [], [], []
    daily_lookup: dict[tuple[str, str, int, str], pd.DataFrame] = {}
    total = len(SPLITS) * portfolio_rules.holding_days
    position = 0
    for split, (start, end) in SPLITS.items():
        for offset in range(portfolio_rules.holding_days):
            position += 1
            primary = build_liquidity_neutral_targets(
                features, members, panel, signal_name="common_membership_breadth_rrg",
                start=start, end=end, offset=offset, rules=portfolio_rules,
            )
            rrg = build_liquidity_neutral_targets(
                features, members, panel, signal_name="rrg_only",
                start=start, end=end, offset=offset, rules=portfolio_rules,
            )
            residual = build_liquidity_neutral_targets(
                features, members, panel, signal_name="common_breadth_residual",
                start=start, end=end, offset=offset, rules=portfolio_rules,
            )
            rrg_residual = build_liquidity_neutral_targets(
                features, members, panel, signal_name="rrg_plus_common_breadth_residual",
                start=start, end=end, offset=offset, rules=portfolio_rules,
            )
            placebo = build_liquidity_neutral_targets(
                features, members, panel, signal_name="common_breadth_residual_placebo",
                start=start, end=end, offset=offset, rules=portfolio_rules,
            )
            rrg_placebo = build_liquidity_neutral_targets(
                features, members, panel, signal_name="rrg_plus_common_breadth_residual_placebo",
                start=start, end=end, offset=offset, rules=portfolio_rules,
            )
            primary_exposure = target_exposure(primary)
            rrg_cash_matched = rescale_target_build(
                rrg, primary_exposure, maximum_stock_weight=portfolio_rules.maximum_stock_weight
            )
            primary_full = rescale_target_build(primary, 1.0, maximum_stock_weight=0.10)
            rrg_vol_matched = volatility_match_target(rrg, primary, panel)
            regime_switch = blend_target_builds(primary, rrg, regime_map, active_regime="repair")
            variants = {
                "primary_original": primary,
                RRR: rrg,
                "rrg_cash_matched_to_primary": rrg_cash_matched,
                "rrg_volatility_matched_to_primary": rrg_vol_matched,
                "primary_fully_invested_10pct": primary_full,
                "breadth_residual_only": residual,
                "rrg_plus_breadth_residual": rrg_residual,
                "breadth_residual_placebo": placebo,
                "rrg_plus_breadth_residual_placebo": rrg_placebo,
                "regime_switch_rrg_to_primary": regime_switch,
            }
            for variant, targets in variants.items():
                for scenario in scenarios:
                    result = run_non_overlapping_ledger(
                        panel, targets, start=start, end=end,
                        portfolio_rules=portfolio_rules, execution_rules=scenarios[scenario],
                    )
                    metrics_rows.append({
                        "variant": variant, "split": split, "offset": offset,
                        "scenario": scenario, **result.metrics,
                    })
                    if not result.daily.empty:
                        daily = result.daily.copy()
                        daily.insert(0, "variant", variant)
                        daily.insert(1, "split", split)
                        daily.insert(2, "offset", offset)
                        daily.insert(3, "scenario", scenario)
                        daily_frames.append(daily)
                        daily_lookup[(variant, split, offset, scenario)] = result.daily
                    if not result.trades.empty and scenario == "base":
                        trades = result.trades.copy()
                        trades.insert(0, "variant", variant)
                        trades.insert(1, "split", split)
                        trades.insert(2, "offset", offset)
                        trade_frames.append(trades)
                if not targets.targets.empty:
                    target = targets.targets.copy()
                    target.insert(0, "variant", variant)
                    target.insert(1, "split", split)
                    target.insert(2, "offset", offset)
                    target_frames.append(target)
            print(f"optimization paths {position}/{total} split={split} offset={offset}", flush=True)

    metrics = pd.DataFrame(metrics_rows)
    daily = pd.concat(daily_frames, ignore_index=True)
    trades = pd.concat(trade_frames, ignore_index=True)
    targets = pd.concat(target_frames, ignore_index=True)
    comparisons = comparison_table(daily_lookup)
    summary = summarize_metrics(metrics)
    comparison_summary = summarize_comparisons(comparisons)
    decision = optimization_decision(comparison_summary)
    exposure_daily = targets.groupby(
        ["variant", "split", "offset", "entry_date"], observed=True
    )["target_weight"].sum().reset_index(name="target_exposure")
    exposure = exposure_daily.groupby(["variant", "split"], observed=True).agg(
        average_target_exposure=("target_exposure", "mean"),
        minimum_target_exposure=("target_exposure", "min"),
    ).reset_index()

    metrics.to_csv(output / "variant_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "variant_summary.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(output / "paired_comparisons.csv", index=False, encoding="utf-8-sig")
    comparison_summary.to_csv(output / "comparison_summary.csv", index=False, encoding="utf-8-sig")
    exposure.to_csv(output / "exposure_summary.csv", index=False, encoding="utf-8-sig")
    regimes.to_csv(output / "market_regimes.csv", index=False, encoding="utf-8-sig")
    daily.to_parquet(output / "variant_daily.parquet", index=False)
    trades.to_parquet(output / "variant_trades.parquet", index=False)
    targets.to_parquet(output / "variant_targets.parquet", index=False)
    (output / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "splits": SPLITS,
        "portfolio_rules": portfolio_rules.__dict__, "execution_rules": execution.__dict__,
        "regime_definition": {
            "repair": "market_return_20d > 0, breadth_delta_5d > 0, 0.30 <= market_breadth < 0.70",
            "overheat": "market_return_20d > 0 and market_breadth >= 0.70",
            "retreat": "otherwise",
        },
        "warning": "All available historical segments have been inspected; no variant can be confirmed before a future untouched sample.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "research_report.md").write_text(
        render_report(decision, summary, comparison_summary, exposure, regimes, manifest), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(output), "decision": decision}, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exposure/residual/regime optimization of concept rotation")
    parser.add_argument("--features", default="artifacts/concept_rotation_alpha")
    parser.add_argument("--snapshot-root", default="data/concept_rotation/dc_20250630_20260714")
    parser.add_argument("--base-panel", default="data/versions/data_v1_20260710T115133Z_208e9fc8/curated/stock_daily_panel.parquet")
    parser.add_argument("--increment-panel", default="data/versions/data_v1_20260715T053637Z_77138057/curated/stock_daily_panel.parquet")
    parser.add_argument("--output-root", default="artifacts/concept_rotation_optimization")
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


def resolve_feature_path(path: Path) -> Path:
    if path.is_file():
        return path
    paths = sorted(path.glob("concept_rotation_*/concept_daily_features_pit.parquet"))
    if not paths:
        raise FileNotFoundError(path)
    return paths[-1]


def target_exposure(build: TargetBuildResult) -> dict[pd.Timestamp, float]:
    if build.targets.empty:
        return {}
    return build.targets.groupby("entry_date")["target_weight"].sum().to_dict()


def volatility_match_target(
    baseline: TargetBuildResult, primary: TargetBuildResult, panel: pd.DataFrame,
) -> TargetBuildResult:
    vol = panel[["trade_date", "ts_code", "volatility_20d"]].rename(columns={"trade_date": "signal_date"})
    def diagonal(build: TargetBuildResult) -> pd.Series:
        joined = build.targets.merge(vol, on=["signal_date", "ts_code"], how="left")
        joined["variance"] = joined["target_weight"].pow(2) * joined["volatility_20d"].fillna(0.02).pow(2)
        return joined.groupby("entry_date")["variance"].sum().pow(0.5)
    primary_vol, baseline_vol = diagonal(primary), diagonal(baseline)
    baseline_exposure = baseline.targets.groupby("entry_date")["target_weight"].sum()
    desired = {}
    for date, exposure in baseline_exposure.items():
        ratio = primary_vol.get(date, np.nan) / baseline_vol.get(date, np.nan)
        desired[pd.Timestamp(date)] = float(np.clip(exposure * ratio, 0, 1)) if np.isfinite(ratio) else float(exposure)
    return rescale_target_build(baseline, desired, maximum_stock_weight=0.05)


def comparison_table(daily_lookup: dict) -> pd.DataFrame:
    pairs = {
        "original_primary_vs_rrg": ("primary_original", RRR),
        "primary_vs_cash_matched_rrg": ("primary_original", "rrg_cash_matched_to_primary"),
        "primary_vs_vol_matched_rrg": ("primary_original", "rrg_volatility_matched_to_primary"),
        "fully_invested_primary_vs_rrg": ("primary_fully_invested_10pct", RRR),
        "rrg_plus_residual_vs_rrg": ("rrg_plus_breadth_residual", RRR),
        "residual_only_vs_rrg": ("breadth_residual_only", RRR),
        "placebo_vs_rrg": ("breadth_residual_placebo", RRR),
        "residual_vs_placebo": ("breadth_residual_only", "breadth_residual_placebo"),
        "rrg_plus_residual_vs_placebo": (
            "rrg_plus_breadth_residual", "rrg_plus_breadth_residual_placebo"
        ),
        "regime_switch_vs_rrg": ("regime_switch_rrg_to_primary", RRR),
    }
    frames = []
    for name, (primary, baseline) in pairs.items():
        frame = paired_portfolio_comparison(daily_lookup, primary=primary, baseline=baseline)
        if not frame.empty:
            frame.insert(0, "comparison", name)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby(["variant", "split", "scenario"], observed=True).agg(
        offsets=("offset", "nunique"), positive_offsets=("total_return", lambda values: int(values.gt(0).sum())),
        mean_total_return=("total_return", "mean"), median_total_return=("total_return", "median"),
        worst_total_return=("total_return", "min"), mean_max_drawdown=("max_drawdown", "mean"),
        mean_volatility=("annualized_volatility", "mean"), mean_turnover=("annualized_turnover", "mean"),
        mean_cash_ratio=("average_cash_ratio", "mean"), mean_cost_drag=("cost_drag", "mean"),
    ).reset_index()


def summarize_comparisons(comparisons: pd.DataFrame) -> pd.DataFrame:
    return comparisons.groupby(["comparison", "split", "scenario"], observed=True).agg(
        offsets=("offset", "nunique"), positive_offsets=("relative_total_return", lambda values: int(values.gt(0).sum())),
        mean_relative_return=("relative_total_return", "mean"),
        median_relative_return=("relative_total_return", "median"),
        worst_relative_return=("relative_total_return", "min"),
        median_nw_t=("incremental_nw_t", "median"),
    ).reset_index()


def optimization_decision(summary: pd.DataFrame) -> dict:
    rows = []
    for comparison, frame in summary.loc[~summary["comparison"].isin(["placebo_vs_rrg"])].groupby("comparison"):
        base = frame.loc[frame["scenario"].eq("base")].set_index("split")
        checks = {}
        for split, required in (("discovery", 3), ("validation", 3), ("final_reused_diagnostic", 4)):
            row = base.loc[split] if split in base.index else None
            checks[f"{split}_consistent"] = bool(
                row is not None and row["positive_offsets"] >= required and row["mean_relative_return"] > 0
            )
        final_stress = frame.loc[
            frame["split"].eq("final_reused_diagnostic")
            & frame["scenario"].isin(["cost_2x", "extra_20bps_roundtrip"])
        ]
        checks["cost_stress_positive"] = bool(
            len(final_stress) == 2 and final_stress["mean_relative_return"].gt(0).all()
        )
        rows.append({
            "comparison": comparison, "passes_historical_screen": all(checks.values()),
            "checks": checks,
        })
    candidates = [row["comparison"] for row in rows if row["passes_historical_screen"]]
    return {
        "historical_screen_candidates": candidates,
        "direction_results": rows,
        "decision": "FUTURE_SHADOW_REQUIRED" if candidates else "NO_OPTIMIZATION_PASSED",
        "warning": "No candidate is confirmed because every available date has already been inspected.",
    }


def render_report(decision: dict, metrics: pd.DataFrame, comparisons: pd.DataFrame,
                  exposure: pd.DataFrame, regimes: pd.DataFrame, manifest: dict) -> str:
    focus = comparisons.loc[
        comparisons["scenario"].eq("base")
        & comparisons["split"].isin(["validation", "final_reused_diagnostic"])
    ]
    regime_counts = regimes.loc[regimes["trade_date"].between("2025-10-01", "2026-06-30"), "regime"].value_counts()
    return f"""# Concept rotation optimization experiment: exposure, residual breadth, and regimes

## Decision

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```

This round does not overwrite the frozen negative result. It asks three narrower questions:
whether cash/risk exposure explains the gap, whether breadth residualized against RRG/momentum/churn
adds information, and whether breadth helps only during a pre-defined market repair regime.

## Paired results versus matched baselines

```text
{focus.to_string(index=False)}
```

## Variant portfolio summary

```text
{metrics.loc[metrics['scenario'].eq('base')].to_string(index=False)}
```

## Exposure summary

```text
{exposure.to_string(index=False)}
```

## Market regime counts

```text
{regime_counts.to_string()}
```

## Manifest

```json
{json.dumps(manifest, ensure_ascii=False, indent=2)}
```
"""


if __name__ == "__main__":
    main()
