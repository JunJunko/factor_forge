from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from factor_forge.research.bullish_divergence_event_study import (
    BullishDivergenceEventStudyConfig,
    run_bullish_divergence_event_study,
)


PANEL_COLUMNS = [
    "trade_date", "ts_code", "adj_open", "adj_high", "adj_low", "adj_close",
    "amount_cny", "turnover_rate", "circ_mv_cny", "industry_l1_code",
    "is_tradeable", "is_suspended", "is_st", "is_delisting_period",
]


def main() -> None:
    args = parse_args()
    features = pd.read_parquet(args.features)
    episodes = pd.read_parquet(args.episodes)
    if features.empty or episodes.empty:
        raise ValueError("features and episodes must both be non-empty")
    for frame in (features, episodes):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    start = features["trade_date"].min() - pd.Timedelta(days=args.warmup_calendar_days)
    end = features["trade_date"].max() + pd.Timedelta(days=args.label_tail_calendar_days)
    schema = set(pq.ParquetFile(args.panel).schema_arrow.names)
    missing = set(PANEL_COLUMNS) - schema
    if missing:
        raise ValueError(f"panel is missing event-study columns: {sorted(missing)}")
    panel = pd.read_parquet(
        args.panel, columns=PANEL_COLUMNS,
        filters=[("trade_date", ">=", start), ("trade_date", "<=", end)],
    )
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    coverage = feature_universe_coverage(panel, features)
    if coverage["median_stock_coverage"] < 0.95 and not args.allow_partial_universe:
        raise ValueError(
            "daily features cover less than 95% of the panel universe; "
            "use a full-market feature artifact or --allow-partial-universe for diagnostics"
        )

    config = BullishDivergenceEventStudyConfig(
        neighbors=args.neighbors,
        caliper=args.caliper,
        roundtrip_cost_bps=args.cost_bps,
        block_length=args.block_length,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed,
    )
    result = run_bullish_divergence_event_study(panel, features, episodes, config)
    created = datetime.now(timezone.utc)
    output = Path(args.output_root) / created.strftime("divergence_e1_%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    result.episodes.to_parquet(output / "episodes_labeled.parquet", index=False)
    result.matched_pairs.to_parquet(output / "matched_pairs.parquet", index=False)
    result.paired_events.to_parquet(output / "paired_events.parquet", index=False)
    result.event_summary.to_csv(output / "event_summary.csv", index=False, encoding="utf-8-sig")
    result.score_monotonicity.to_csv(output / "score_monotonicity.csv", index=False, encoding="utf-8-sig")
    result.touch_summary.to_csv(output / "touch_state_summary.csv", index=False, encoding="utf-8-sig")
    result.matching_balance.to_csv(output / "matching_balance.csv", index=False, encoding="utf-8-sig")
    result.bootstrap.to_csv(output / "block_bootstrap.csv", index=False, encoding="utf-8-sig")
    summary = {
        **result.summary, "created_at": created.isoformat(), "coverage": coverage,
        "panel": str(Path(args.panel)), "features": str(Path(args.features)),
        "episodes": str(Path(args.episodes)), "config": asdict(config),
        "research_status": "DIAGNOSTIC_PARTIAL_UNIVERSE" if coverage["median_stock_coverage"] < .95 else "E1_E1_1",
    }
    if coverage["median_stock_coverage"] < .95:
        summary["decision"] = "DIAGNOSTIC_ONLY_PARTIAL_UNIVERSE"
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
    )
    (output / "report.md").write_text(render_report(summary, result), encoding="utf-8")
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2, default=json_default))


def feature_universe_coverage(panel: pd.DataFrame, features: pd.DataFrame) -> dict:
    dates = set(features["trade_date"].unique())
    panel_counts = panel.loc[panel["trade_date"].isin(dates)].groupby("trade_date")["ts_code"].nunique()
    feature_counts = features.groupby("trade_date")["ts_code"].nunique()
    aligned = pd.concat([panel_counts.rename("panel"), feature_counts.rename("features")], axis=1).dropna()
    ratio = aligned["features"] / aligned["panel"]
    return {
        "dates": int(len(aligned)),
        "median_stock_coverage": float(ratio.median()) if len(ratio) else 0.0,
        "minimum_stock_coverage": float(ratio.min()) if len(ratio) else 0.0,
    }


def render_report(summary, result) -> str:
    return "\n".join([
        "# Bullish Divergence E1/E1.1",
        "",
        f"- Research status: `{summary['research_status']}`",
        f"- Episodes / mature / matched: `{summary['episode_count']}` / `{summary['mature_episode_count']}` / `{summary['matched_episode_count']}`",
        f"- Match rate: `{summary['match_rate']:.2%}`",
        f"- Maximum absolute SMD: `{summary['maximum_absolute_smd']}`",
        f"- Decision: `{summary['decision']}`",
        "",
        "## Checks",
        "",
        *[f"- {name}: `{value}`" for name, value in summary["checks"].items()],
        "",
        "## Block bootstrap",
        "",
        result.bootstrap.to_markdown(index=False) if len(result.bootstrap) else "No matched bootstrap results.",
        "",
        "A partial-universe run is diagnostic only and cannot pass the research Gate.",
    ]) + "\n"


def json_default(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp): return value.isoformat()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bullish-divergence E1/E1.1 matched event study")
    parser.add_argument("--panel", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--caliper", type=float, default=3.0)
    parser.add_argument("--cost-bps", type=float, default=40.0)
    parser.add_argument("--block-length", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-calendar-days", type=int, default=180)
    parser.add_argument("--label-tail-calendar-days", type=int, default=45)
    parser.add_argument("--allow-partial-universe", action="store_true")
    parser.add_argument("--output-root", default="artifacts/bullish_divergence_event_studies")
    return parser.parse_args()


if __name__ == "__main__":
    main()
