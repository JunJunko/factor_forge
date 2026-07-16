from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.research.concept_lifecycle_backtest import attach_enhanced_stock_features
from factor_forge.research.concept_portfolio_backtest import (
    ExecutionRules,
    PortfolioRules,
    TargetBuildResult,
    run_non_overlapping_ledger,
)
from factor_forge.research.concept_rotation_alpha import load_dc_snapshot


PROXY_CODES = ["BK1136.DC", "BK1128.DC"]  # optical-communication modules and CPO
EXPOST_FIBER_CODE = "BK1660.DC"
PANEL_COLUMNS = [
    "trade_date", "ts_code", "raw_open", "raw_close", "adj_open", "adj_close", "amount_cny",
    "circ_mv_cny", "turnover_rate", "is_suspended", "is_limit_up_open", "is_limit_down_open",
    "is_st", "is_delisting_period", "listing_trade_days", "is_tradeable",
]


def main() -> None:
    args = parse_args()
    output = Path(args.output_root) / datetime.now(timezone.utc).strftime("optical_event_%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    panel = attach_enhanced_stock_features(
        load_stock_panel(Path(args.base_panel), Path(args.increment_panel)),
        pd.read_parquet(args.daily_basic),
    )
    features = pd.read_parquet(args.features)
    features["trade_date"] = pd.to_datetime(features["trade_date"])
    regimes = pd.read_csv(args.regimes, parse_dates=["trade_date"])
    _, members = load_dc_snapshot(args.snapshot_root, trade_dates=panel["trade_date"].unique())
    proxy_features = features.loc[
        features["concept_code"].isin(PROXY_CODES)
        & features["trade_date"].between("2025-11-17", "2026-02-13")
    ].copy()
    timeline = proxy_features.merge(
        regimes[["trade_date", "regime", "target_exposure"]], on="trade_date", how="left"
    )

    onset_signal = first_signal(proxy_features, strict_level=False, start="2025-11-24", end="2025-12-10")
    strict_signal = first_signal(proxy_features, strict_level=True, start="2025-11-24", end="2025-12-15")
    first_exit = first_weakening(proxy_features, after=onset_signal, end="2026-01-15")
    second_signal = first_signal(proxy_features, strict_level=True, start="2026-01-20", end="2026-02-05")
    second_exit = first_weakening(proxy_features, after=second_signal, end="2026-02-13")
    calendar = pd.Index(sorted(panel["trade_date"].unique()))

    first_proxy_members = union_members(members, PROXY_CODES, onset_signal)
    strict_proxy_members = union_members(members, PROXY_CODES, strict_signal)
    second_proxy_members = union_members(members, PROXY_CODES, second_signal)
    fiber_file = Path(args.snapshot_root) / "members_by_concept" / f"{EXPOST_FIBER_CODE}.parquet"
    fiber = pd.read_parquet(fiber_file)
    first_fiber_date = fiber["trade_date"].min()
    expost_fiber_members = set(
        fiber.loc[fiber["trade_date"].eq(first_fiber_date), "con_code"].astype(str)
    )

    events = {
        "current_gate_rrg_replay": {
            "signal": onset_signal, "exit_signal": pd.Timestamp("2025-12-03"),
            "members": first_proxy_members, "selection": "core10", "lookahead": False, "exposure": 0.35,
            "note": "RRG proxy entered while neutral, then global retreat gate forced exit.",
        },
        "proxy_onset_core_no_gate": {
            "signal": onset_signal, "exit_signal": first_exit,
            "members": first_proxy_members, "selection": "core10", "lookahead": False, "exposure": 0.35,
            "note": "Remove 50% breadth-level threshold and global retreat veto.",
        },
        "proxy_strict_core_no_gate": {
            "signal": strict_signal, "exit_signal": first_exit,
            "members": strict_proxy_members, "selection": "core10", "lookahead": False, "exposure": 0.35,
            "note": "Keep the 50% breadth level, remove only the global retreat veto.",
        },
        "proxy_onset_equal_members_no_gate": {
            "signal": onset_signal, "exit_signal": first_exit,
            "members": first_proxy_members, "selection": "equal_members", "lookahead": False, "exposure": 0.35,
            "note": "Point-in-time proxy concept basket rather than core-stock selection.",
        },
        "expost_fiber_core_oracle": {
            "signal": onset_signal, "exit_signal": first_exit,
            "members": expost_fiber_members, "selection": "core10", "lookahead": True, "exposure": 0.35,
            "note": f"Uses {EXPOST_FIBER_CODE} membership first published on {first_fiber_date}; upper-bound diagnostic only.",
        },
        "proxy_second_wave_strict": {
            "signal": second_signal, "exit_signal": second_exit,
            "members": second_proxy_members, "selection": "core10", "lookahead": False, "exposure": 0.35,
            "note": "Late-January second signal, retained to test false re-entry risk.",
        },
    }
    execution = ExecutionRules(maximum_adv_participation=0.05)
    scenarios = {"base": execution, "cost_2x": replace(execution, cost_multiplier=2.0)}
    metric_rows, daily_frames, trade_frames, target_frames, event_rows = [], [], [], [], []
    for name, event in events.items():
        signal = pd.Timestamp(event["signal"])
        exit_signal = pd.Timestamp(event["exit_signal"])
        picks = select_event_stocks(panel, event["members"], signal, event["selection"])
        target_build = event_target_build(calendar, signal, exit_signal, picks, exposure=float(event["exposure"]))
        if not target_build.targets.empty:
            item = target_build.targets.copy(); item.insert(0, "event", name); target_frames.append(item)
        event_rows.append({
            "event": name, "signal_date": signal, "entry_date": next_date(calendar, signal),
            "exit_signal_date": exit_signal, "exit_date": next_date(calendar, exit_signal),
            "source_members": len(event["members"]), "selected_stocks": len(picks),
            "selection": event["selection"], "exposure": event["exposure"],
            "lookahead": event["lookahead"], "note": event["note"],
        })
        for scenario, rules in scenarios.items():
            result = run_non_overlapping_ledger(
                panel, target_build, start=next_date(calendar, signal), end=next_date(calendar, exit_signal),
                portfolio_rules=PortfolioRules(maximum_stock_weight=0.10), execution_rules=rules,
            )
            metric_rows.append({"event": name, "scenario": scenario, **result.metrics})
            if not result.daily.empty:
                item = result.daily.copy(); item.insert(0, "event", name); item.insert(1, "scenario", scenario)
                daily_frames.append(item)
            if scenario == "base" and not result.trades.empty:
                item = result.trades.copy(); item.insert(0, "event", name); trade_frames.append(item)

    metrics = pd.DataFrame(metric_rows)
    event_table = pd.DataFrame(event_rows)
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    targets = pd.concat(target_frames, ignore_index=True) if target_frames else pd.DataFrame()
    overlap = membership_overlap(members, expost_fiber_members, onset_signal, features)
    signature_events, signature_summary = signature_scan(features, regimes)
    audit = {
        "proxy_codes": PROXY_CODES, "expost_fiber_code": EXPOST_FIBER_CODE,
        "expost_fiber_first_member_date": str(first_fiber_date),
        "expost_fiber_members": len(expost_fiber_members),
        "target_duplicates": int(targets.duplicated(["event", "entry_date", "ts_code"]).sum()) if not targets.empty else 0,
        "nan_nav": int(daily["nav"].isna().sum()) if not daily.empty else 0,
        "maximum_target_weight": float(targets["target_weight"].max()) if not targets.empty else None,
        "maximum_participation": float(trades["participation"].max()) if not trades.empty else None,
    }
    timeline.to_csv(output / "proxy_signal_timeline.csv", index=False, encoding="utf-8-sig")
    event_table.to_csv(output / "event_definitions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output / "event_metrics.csv", index=False, encoding="utf-8-sig")
    overlap.to_csv(output / "expost_fiber_proxy_overlap.csv", index=False, encoding="utf-8-sig")
    signature_events.to_csv(output / "signature_events.csv", index=False, encoding="utf-8-sig")
    signature_summary.to_csv(output / "signature_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_parquet(output / "event_daily.parquet", index=False)
    trades.to_parquet(output / "event_trades.parquet", index=False)
    targets.to_parquet(output / "event_targets.parquet", index=False)
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "research_report.md").write_text(
        render_report(event_table, metrics, overlap, timeline, signature_summary, audit), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(output), "events": event_table.to_dict("records"), "audit": audit}, ensure_ascii=False, indent=2, default=str), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrospective optical-fiber rotation event study")
    parser.add_argument("--snapshot-root", default="data/concept_rotation/dc_20250630_20260714")
    parser.add_argument("--features", default="artifacts/concept_rotation_lifecycle/concept_lifecycle_20260715T123602Z/concept_features_free_float.parquet")
    parser.add_argument("--regimes", default="artifacts/concept_rotation_lifecycle/concept_lifecycle_20260715T123602Z/market_regimes.csv")
    parser.add_argument("--daily-basic", default="data/concept_rotation/daily_basic_20250501_20260714/daily_basic_enhanced.parquet")
    parser.add_argument("--base-panel", default="data/versions/data_v1_20260710T115133Z_208e9fc8/curated/stock_daily_panel.parquet")
    parser.add_argument("--increment-panel", default="data/versions/data_v1_20260715T053637Z_77138057/curated/stock_daily_panel.parquet")
    parser.add_argument("--output-root", default="artifacts/concept_rotation_optical_event")
    return parser.parse_args()


def first_signal(features: pd.DataFrame, *, strict_level: bool, start: str, end: str) -> pd.Timestamp:
    sample = features.loc[features["trade_date"].between(start, end)].copy()
    condition = (
        sample["rrg_quadrant"].isin(["leading", "improving"])
        & sample["common_delta_rank"].ge(0.70)
        & sample["common_breadth_delta_smooth5"].gt(0)
    )
    if strict_level:
        condition &= sample["breadth_float"].gt(0.50)
    counts = sample.loc[condition].groupby("trade_date")["concept_code"].nunique()
    valid = counts.loc[counts.ge(1)]
    if valid.empty:
        raise RuntimeError(f"no optical proxy signal in {start}:{end}")
    return pd.Timestamp(valid.index.min())


def first_weakening(features: pd.DataFrame, *, after: pd.Timestamp, end: str) -> pd.Timestamp:
    sample = features.loc[
        features["trade_date"].between(after + pd.Timedelta(days=1), pd.Timestamp(end))
    ]
    counts = sample.loc[sample["rrg_quadrant"].isin(["weakening", "lagging"])].groupby("trade_date")["concept_code"].nunique()
    valid = counts.loc[counts.ge(len(PROXY_CODES))]
    if valid.empty:
        raise RuntimeError(f"no common weakening after {after}")
    return pd.Timestamp(valid.index.min())


def union_members(members: pd.DataFrame, codes: list[str], date: pd.Timestamp) -> set[str]:
    return set(members.loc[
        members["trade_date"].eq(pd.Timestamp(date)) & members["concept_code"].isin(codes), "ts_code"
    ].astype(str))


def select_event_stocks(panel: pd.DataFrame, members: set[str], signal: pd.Timestamp, selection: str) -> list[str]:
    day = panel.loc[panel["trade_date"].eq(signal) & panel["ts_code"].isin(members)].copy()
    day = day.loc[day["is_tradeable"].fillna(False) & day["amount_ma20"].ge(20_000_000)]
    if selection == "equal_members":
        return day.sort_values("amount_ma20", ascending=False)["ts_code"].astype(str).tolist()
    day = day.loc[day["stock_return_20d"].gt(0)].copy()
    day["score"] = (
        np.log(day["amount_ma20"].clip(lower=1)).rank(pct=True) * 0.35
        + np.log(day["free_float_mv_cny"].clip(lower=1)).rank(pct=True) * 0.25
        + day["stock_return_20d"].rank(pct=True) * 0.40
    )
    return day.sort_values(["score", "amount_ma20"], ascending=False).head(10)["ts_code"].astype(str).tolist()


def event_target_build(calendar: pd.Index, signal: pd.Timestamp, exit_signal: pd.Timestamp,
                       picks: list[str], *, exposure: float) -> TargetBuildResult:
    entry, exit_date = next_date(calendar, signal), next_date(calendar, exit_signal)
    if not picks:
        return TargetBuildResult(pd.DataFrame(), pd.DataFrame(), [entry, exit_date], [entry])
    weights = np.repeat(exposure / len(picks), len(picks))
    targets = pd.DataFrame({
        "signal_date": signal, "entry_date": entry, "ts_code": picks, "target_weight": weights,
    })
    return TargetBuildResult(targets, pd.DataFrame(), [entry, exit_date], [entry])


def next_date(calendar: pd.Index, date: pd.Timestamp) -> pd.Timestamp:
    position = calendar.get_loc(pd.Timestamp(date))
    return pd.Timestamp(calendar[position + 1])


def membership_overlap(members: pd.DataFrame, cohort: set[str], date: pd.Timestamp,
                       features: pd.DataFrame) -> pd.DataFrame:
    day = members.loc[members["trade_date"].eq(date)]
    sets = day.groupby("concept_code", observed=True)["ts_code"].agg(lambda x: set(x.astype(str)))
    names = features[["concept_code", "concept_name"]].drop_duplicates("concept_code").set_index("concept_code")["concept_name"].to_dict()
    rows = []
    for code, values in sets.items():
        overlap = len(cohort & values)
        union = len(cohort | values)
        rows.append({
            "concept_code": code, "concept_name": names.get(code), "overlap": overlap,
            "cohort_recall": overlap / len(cohort), "jaccard": overlap / union if union else 0,
            "concept_members": len(values),
        })
    return pd.DataFrame(rows).sort_values(["overlap", "jaccard"], ascending=False)


def signature_scan(features: pd.DataFrame, regimes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = features.merge(regimes[["trade_date", "regime"]], on="trade_date", how="left")
    sample = sample.sort_values(["concept_code", "trade_date"]).copy()
    base = (
        sample["eligible_concept"].fillna(False)
        & sample["rrg_quadrant"].eq("leading") & sample["rs_20d"].gt(0)
        & sample["regime"].isin(["retreat", "neutral"])
    )
    high = base & sample["common_delta_rank"].ge(0.90) & sample["common_breadth_delta_smooth5"].gt(0)
    control = base & sample["common_delta_rank"].between(0.40, 0.60)
    sample["high_event"] = high & ~high.groupby(sample["concept_code"]).shift(1, fill_value=False)
    sample["control_event"] = control & ~control.groupby(sample["concept_code"]).shift(1, fill_value=False)
    sample["split"] = np.select(
        [
            sample["trade_date"].between("2025-10-01", "2025-12-31"),
            sample["trade_date"].between("2026-01-01", "2026-03-31"),
            sample["trade_date"].between("2026-04-01", "2026-06-30"),
        ], ["discovery", "validation", "final_reused_diagnostic"], default="outside",
    )
    rows = []
    for event_type, column in (("high_diffusion_signature", "high_event"), ("mid_diffusion_control", "control_event")):
        item = sample.loc[sample[column] & ~sample["split"].eq("outside")].copy()
        item["event_type"] = event_type
        rows.append(item)
    events = pd.concat(rows, ignore_index=True)
    events["is_optical_proxy"] = events["concept_code"].isin(PROXY_CODES)
    summary_rows = []
    for split in ["discovery", "validation", "final_reused_diagnostic"]:
        split_events = events.loc[events["split"].eq(split)]
        for event_type, group in split_events.groupby("event_type", observed=True):
            summary_rows.append({
                "split": split, "statistic": event_type,
                "events": int(len(group)), "event_dates": int(group["trade_date"].nunique()),
                "concepts": int(group["concept_code"].nunique()),
                "mean_forward_excess_5d": float(group["forward_excess_5d"].mean()),
                "median_forward_excess_5d": float(group["forward_excess_5d"].median()),
                "positive_5d": float(group["forward_excess_5d"].gt(0).mean()),
                "mean_forward_excess_10d": float(group["forward_excess_10d"].mean()),
            })
        high_daily = split_events.loc[split_events["event_type"].eq("high_diffusion_signature")].groupby("trade_date")["forward_excess_5d"].mean()
        control_daily = split_events.loc[split_events["event_type"].eq("mid_diffusion_control")].groupby("trade_date")["forward_excess_5d"].mean()
        matched = pd.concat([high_daily.rename("high"), control_daily.rename("control")], axis=1).dropna()
        summary_rows.append({
            "split": split, "statistic": "daily_high_minus_control",
            "events": int(len(matched)), "event_dates": int(len(matched)), "concepts": None,
            "mean_forward_excess_5d": float((matched["high"] - matched["control"]).mean()) if len(matched) else None,
            "median_forward_excess_5d": float((matched["high"] - matched["control"]).median()) if len(matched) else None,
            "positive_5d": float(matched["high"].gt(matched["control"]).mean()) if len(matched) else None,
            "mean_forward_excess_10d": None,
        })
    return events[[
        "trade_date", "split", "event_type", "concept_code", "concept_name", "regime",
        "common_delta_rank", "common_breadth_delta_smooth5", "rs_20d",
        "forward_excess_5d", "forward_excess_10d", "is_optical_proxy",
    ]], pd.DataFrame(summary_rows)


def load_stock_panel(base_path: Path, increment_path: Path) -> pd.DataFrame:
    base = pd.read_parquet(base_path, columns=PANEL_COLUMNS, filters=[("trade_date", ">=", pd.Timestamp("2025-05-01"))])
    increment = pd.read_parquet(increment_path, columns=PANEL_COLUMNS)
    for frame in (base, increment): frame["trade_date"] = pd.to_datetime(frame["trade_date"])
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


def render_report(events: pd.DataFrame, metrics: pd.DataFrame, overlap: pd.DataFrame,
                  timeline: pd.DataFrame, signature_summary: pd.DataFrame, audit: dict) -> str:
    base = metrics.loc[metrics["scenario"].eq("base"), [
        "event", "total_return", "max_drawdown", "annualized_turnover", "cost_drag",
        "blocked_buys", "blocked_sells",
    ]]
    focus = timeline.loc[timeline["trade_date"].between("2025-11-24", "2025-12-18"), [
        "trade_date", "concept_code", "breadth_float", "common_breadth_delta_smooth5",
        "common_delta_rank", "rrg_quadrant", "lifecycle", "regime", "target_exposure",
    ]]
    return f"""# Optical-fiber rotation retrospective event study

## Scope warning

The December 2025 optical-fiber episode was selected after observing its strength. This study
is a mechanism replay and cannot establish alpha. `{EXPOST_FIBER_CODE}` did not publish member
history until April 2026, so its backfilled cohort is explicitly labeled look-ahead/oracle.

## Event definitions

```text
{events.to_string(index=False)}
```

## Account-level event results

```text
{base.to_string(index=False)}
```

## Signal timeline around the first wave

```text
{focus.to_string(index=False)}
```

## Best point-in-time proxies for the later fiber cohort

```text
{overlap.head(20).to_string(index=False)}
```

## Cross-concept scan of the event-derived signature

The signature (leading RRG, positive RS, top-decile rising diffusion while the market is
neutral/retreat) was derived after observing the optical episode. Its cross-concept scan is
therefore diagnostic. `daily_high_minus_control` compares event-date averages with otherwise
similar leading concepts whose diffusion rank was between 40% and 60%.

```text
{signature_summary.to_string(index=False)}
```

## Audit

```json
{json.dumps(audit, ensure_ascii=False, indent=2)}
```
"""


if __name__ == "__main__":
    main()
