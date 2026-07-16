from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from factor_forge.research.concept_rotation_alpha import (
    build_concept_dataset,
    evaluate_concept_signals,
    latest_membership_backfill,
    load_dc_snapshot,
    paired_signal_differences,
)


PANEL_COLUMNS = [
    "trade_date", "ts_code", "adj_open", "adj_close", "amount_cny", "circ_mv_cny",
    "is_suspended", "is_st", "is_delisting_period", "listing_trade_days", "is_tradeable",
]
SPLITS = {
    "discovery": ("2025-10-01", "2025-12-31"),
    "validation": ("2026-01-01", "2026-03-31"),
    "final_holdout": ("2026-04-01", "2026-06-30"),
    "post_holdout_shadow": ("2026-07-01", "2026-07-14"),
}


def main() -> None:
    args = parse_args()
    snapshot_root = Path(args.snapshot_root)
    require_complete_snapshot(snapshot_root)
    panel = load_stock_panel(Path(args.base_panel), Path(args.increment_panel))
    concept_index, members = load_dc_snapshot(
        snapshot_root, trade_dates=panel["trade_date"].unique()
    )
    raw_audit = raw_snapshot_audit(concept_index, members, panel)
    failures = raw_gate_failures(raw_audit)
    if failures:
        raise RuntimeError("concept snapshot data gate failed: " + "; ".join(failures))

    run_id = datetime.now(timezone.utc).strftime("concept_rotation_%Y%m%dT%H%M%SZ")
    output = Path(args.output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw_data_audit.json").write_text(
        json.dumps(raw_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("building point-in-time concept features", flush=True)
    relations, features, pit_audit = build_concept_dataset(panel, concept_index, members)
    derived_failures = derived_gate_failures(pit_audit)
    if derived_failures:
        raise RuntimeError("derived concept data gate failed: " + "; ".join(derived_failures))
    features.to_parquet(output / "concept_daily_features_pit.parquet", index=False)
    pit_summary, pit_daily = evaluate_concept_signals(features, relations, splits=SPLITS)
    pit_summary.insert(0, "membership_mode", "point_in_time")
    pit_daily.insert(0, "membership_mode", "point_in_time")
    del relations, features
    gc.collect()

    summaries = [pit_summary]
    daily_results = [pit_daily]
    backfill_audit = None
    if args.latest_backfill_diagnostic:
        print("building latest-membership backfill diagnostic", flush=True)
        back_index, back_members = latest_membership_backfill(concept_index, members)
        relations, features, backfill_audit = build_concept_dataset(panel, back_index, back_members)
        features.to_parquet(output / "concept_daily_features_latest_backfill.parquet", index=False)
        back_summary, back_daily = evaluate_concept_signals(features, relations, splits=SPLITS)
        back_summary.insert(0, "membership_mode", "latest_backfill_lookahead")
        back_daily.insert(0, "membership_mode", "latest_backfill_lookahead")
        summaries.append(back_summary)
        daily_results.append(back_daily)
        del relations, features, back_index, back_members
        gc.collect()

    summary = pd.concat(summaries, ignore_index=True)
    daily = pd.concat(daily_results, ignore_index=True)
    summary.to_csv(output / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output / "daily_signal_payoffs.csv", index=False, encoding="utf-8-sig")
    incremental = paired_signal_differences(daily)
    incremental.to_csv(output / "incremental_vs_baselines.csv", index=False, encoding="utf-8-sig")
    audit = {"raw": raw_audit, "point_in_time": pit_audit, "latest_backfill": backfill_audit}
    (output / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = render_report(snapshot_root, audit, summary, incremental)
    (output / "research_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"run_dir": str(output), "audit": audit}, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PIT Tushare concept breadth + RRG alpha experiment")
    parser.add_argument("--snapshot-root", default="data/concept_rotation/dc_20250630_20260714")
    parser.add_argument(
        "--base-panel",
        default="data/versions/data_v1_20260710T115133Z_208e9fc8/curated/stock_daily_panel.parquet",
    )
    parser.add_argument(
        "--increment-panel",
        default="data/versions/data_v1_20260715T053637Z_77138057/curated/stock_daily_panel.parquet",
    )
    parser.add_argument("--output-root", default="artifacts/concept_rotation_alpha")
    parser.add_argument(
        "--latest-backfill-diagnostic", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def require_complete_snapshot(root: Path) -> None:
    if not (root / "manifest.json").exists():
        raise FileNotFoundError(f"incomplete concept snapshot (manifest missing): {root}")
    if (root / "failures.json").exists():
        raise RuntimeError(f"incomplete concept snapshot (failure checkpoint exists): {root}")


def load_stock_panel(base_path: Path, increment_path: Path) -> pd.DataFrame:
    base = pd.read_parquet(
        base_path, columns=PANEL_COLUMNS,
        filters=[("trade_date", ">=", pd.Timestamp("2025-05-01"))],
    )
    increment = pd.read_parquet(increment_path, columns=PANEL_COLUMNS)
    base["trade_date"] = pd.to_datetime(base["trade_date"])
    increment["trade_date"] = pd.to_datetime(increment["trade_date"])
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
    panel = pd.concat([base, increment], ignore_index=True).sort_values(["trade_date", "ts_code"])
    panel = panel.drop_duplicates(["trade_date", "ts_code"], keep="last")
    return panel.loc[
        panel["trade_date"].between(pd.Timestamp("2025-05-01"), pd.Timestamp("2026-07-14"))
    ].reset_index(drop=True)


def raw_snapshot_audit(index: pd.DataFrame, members: pd.DataFrame, panel: pd.DataFrame) -> dict:
    index_dates = set(index["trade_date"].unique())
    member_dates = set(members["trade_date"].unique())
    expected_dates = set(panel.loc[
        panel["trade_date"].between(index["trade_date"].min(), index["trade_date"].max()), "trade_date"
    ].unique())
    counts = index.groupby("trade_date")["concept_code"].nunique()
    memberships = members.groupby(["trade_date", "ts_code"], observed=True).size()
    return {
        "start_date": index["trade_date"].min().strftime("%Y-%m-%d"),
        "end_date": index["trade_date"].max().strftime("%Y-%m-%d"),
        "index_rows": int(len(index)), "member_rows": int(len(members)),
        "concepts": int(index["concept_code"].nunique()), "stocks": int(members["ts_code"].nunique()),
        "snapshot_dates": int(len(index_dates)),
        "missing_index_trade_dates": int(len(expected_dates - index_dates)),
        "missing_member_snapshot_dates": int(len(index_dates - member_dates)),
        "duplicate_index_keys": int(index.duplicated(["trade_date", "concept_code"]).sum()),
        "duplicate_member_keys": int(members.duplicated(["trade_date", "concept_code", "ts_code"]).sum()),
        "repaired_partial_member_dates": list(members.attrs.get("repaired_member_dates", [])),
        "min_daily_concepts": int(counts.min()), "median_daily_concepts": float(counts.median()),
        "max_daily_concepts": int(counts.max()),
        "median_concepts_per_stock": float(memberships.median()),
        "p95_concepts_per_stock": float(memberships.quantile(0.95)),
    }


def raw_gate_failures(audit: dict) -> list[str]:
    failures = []
    for key in ("missing_index_trade_dates", "missing_member_snapshot_dates", "duplicate_index_keys", "duplicate_member_keys"):
        if audit[key]:
            failures.append(f"{key}={audit[key]}")
    if audit["min_daily_concepts"] < 350:
        failures.append("daily concept count below 350")
    if audit["median_concepts_per_stock"] <= 1:
        failures.append("many-to-many membership not observed")
    return failures


def derived_gate_failures(audit: dict) -> list[str]:
    failures = []
    if audit["member_support_coverage"] < 0.95:
        failures.append("member-to-price support below 95%")
    if audit["eligible_concept_days"] == 0:
        failures.append("no eligible concept days")
    return failures


def render_report(
    snapshot: Path, audit: dict, summary: pd.DataFrame, incremental: pd.DataFrame,
) -> str:
    focus = summary.loc[
        summary["split"].isin(["validation", "final_holdout"])
        & summary["signal"].isin([
            "hot_1d", "momentum_20d", "current_membership_breadth_rrg",
            "common_membership_breadth_rrg", "membership_churn_placebo",
        ])
    ]
    paired = incremental.loc[
        incremental["split"].isin(["validation", "final_holdout"])
        & incremental["signal"].eq("common_membership_breadth_rrg")
    ]
    def row(signal: str, split: str = "final_holdout") -> pd.Series:
        return summary.loc[
            summary["membership_mode"].eq("point_in_time")
            & summary["split"].eq(split) & summary["signal"].eq(signal)
        ].iloc[0]

    primary_final = row("common_membership_breadth_rrg")
    primary_validation = row("common_membership_breadth_rrg", "validation")
    rrg_final = row("rrg_only")
    momentum_final = row("momentum_20d")
    incremental_rrg = incremental.loc[
        incremental["membership_mode"].eq("point_in_time")
        & incremental["split"].eq("final_holdout")
        & incremental["signal"].eq("common_membership_breadth_rrg")
        & incremental["benchmark_signal"].eq("rrg_only")
    ].iloc[0]
    return f"""# Tushare concept rotation alpha experiment

## Verdict

The primary common-member breadth + RRG rule produced **{primary_final['top_net_excess_20bps']:.2%}**
net 5-day market excess in the point-in-time final holdout (Newey-West t =
{primary_final['top_net_nw_t']:.2f}). However, it was **{primary_validation['top_net_excess_20bps']:.2%}**
in validation (t = {primary_validation['top_net_nw_t']:.2f}), and its final-holdout increment over
RRG-only was **{incremental_rrg['incremental_net_excess']:.2%}** (t =
{incremental_rrg['incremental_nw_t']:.2f}). RRG-only returned {rrg_final['top_net_excess_20bps']:.2%}
(t = {rrg_final['top_net_nw_t']:.2f}) and 20-day momentum returned
{momentum_final['top_net_excess_20bps']:.2%} (t = {momentum_final['top_net_nw_t']:.2f}).

**Conclusion:** this one-year sample contains positive final-holdout excess, but it does not establish
incremental breadth alpha. The evidence is more consistent with a momentum/RRG regime effect and is
not yet robust enough for live deployment.

## Design

- Point-in-time daily `dc_index` + `dc_member` snapshots are the primary specification.
- Final-date membership is expanded backwards only as a look-ahead-bias diagnostic.
- A stock may belong to multiple concepts; daily Jaccard similarity above 0.80 removes near-duplicate selected concepts.
- Breadth uses free-float-cap weights and a 20-day smoother. The common-member version measures the same stocks at T and T-5 to isolate membership churn.
- Labels freeze T membership and use T+1 open through T+h+1 open. The preregistered horizon is 5 trading days.
- Reported inference uses Newey-West t-statistics, 20-day block bootstrap, and a 20 bps round-trip cost.

## Data gate

Snapshot: `{snapshot}`

```json
{json.dumps(audit, ensure_ascii=False, indent=2)}
```

## Validation and final holdout

```text
{focus.to_string(index=False)}
```

## Incremental result versus simple baselines

```text
{paired.to_string(index=False)}
```

Interpretation rule: only the point-in-time final holdout counts as confirmatory evidence. A stronger latest-backfill result is evidence of taxonomy/membership look-ahead bias, not alpha.
"""


if __name__ == "__main__":
    main()
