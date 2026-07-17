from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from factor_forge.research.bullish_divergence_v2_event_study import (
    BullishDivergenceV2EventStudyConfig,
    run_v2_origin_event_study,
    run_v2_retest_event_study,
)


PANEL_COLUMNS = [
    "trade_date", "ts_code", "adj_open", "adj_high", "adj_low", "adj_close",
    "amount_cny", "turnover_rate", "circ_mv_cny", "industry_l1_code",
    "is_tradeable", "is_suspended", "is_st", "is_delisting_period",
]


def main() -> None:
    args = parse_args()
    daily = pd.read_parquet(args.features)
    episodes = pd.read_parquet(args.episodes)
    retests = pd.read_parquet(args.retests)
    for frame in (daily, episodes, retests):
        if "trade_date" in frame:
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    start = daily["trade_date"].min() - pd.Timedelta(days=args.warmup_calendar_days)
    end = daily["trade_date"].max() + pd.Timedelta(days=args.label_tail_calendar_days)
    schema = set(pq.ParquetFile(args.panel).schema_arrow.names)
    if missing := set(PANEL_COLUMNS) - schema:
        raise ValueError(f"panel is missing event-study columns: {sorted(missing)}")
    panel = pd.read_parquet(
        args.panel, columns=PANEL_COLUMNS,
        filters=[("trade_date", ">=", start), ("trade_date", "<=", end)],
    )
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    coverage = feature_universe_coverage(panel, daily)
    if coverage["median_stock_coverage"] < .95:
        raise ValueError("v2 E1 requires at least 95% median full-universe feature coverage")

    config = BullishDivergenceV2EventStudyConfig(
        neighbors=args.neighbors,
        caliper=args.caliper,
        roundtrip_cost_bps=args.cost_bps,
        block_length=args.block_length,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed,
    )
    origin = run_v2_origin_event_study(panel, daily, episodes, config)
    retest = run_v2_retest_event_study(panel, daily, retests, config)
    created = datetime.now(timezone.utc)
    output = Path(args.output_root) / created.strftime(
        "divergence_v2_e1_%Y%m%dT%H%M%SZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    write_result(output / "origin", origin)
    write_result(output / "retest", retest)
    summary = {
        "created_at": created.isoformat(),
        "research_status": "DIAGNOSTIC_V2_INSPECTED_HISTORY",
        "coverage": coverage,
        "panel": str(Path(args.panel)),
        "features": str(Path(args.features)),
        "episodes": str(Path(args.episodes)),
        "retests": str(Path(args.retests)),
        "config": asdict(config),
        "origin": origin.summary,
        "retest": retest.summary,
        "decision": (
            "ELIGIBLE_FOR_CONDITIONAL_MATRIX_DIAGNOSTIC"
            if origin.summary["decision"] == "ELIGIBLE_FOR_CONDITIONAL_MATRIX_DIAGNOSTIC"
            else "STOP_OR_REVISE_BEFORE_CONDITIONAL_MATRIX"
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    (output / "report.md").write_text(render_report(summary, origin, retest), encoding="utf-8")
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))


def write_result(path: Path, result) -> None:
    path.mkdir(parents=True, exist_ok=False)
    result.events.to_parquet(path / "events_labeled.parquet", index=False)
    result.pairs.to_parquet(path / "matched_pairs.parquet", index=False)
    result.paired_events.to_parquet(path / "paired_events.parquet", index=False)
    result.score_summary.to_csv(path / "score_summary.csv", index=False, encoding="utf-8-sig")
    result.state_summary.to_csv(path / "state_summary.csv", index=False, encoding="utf-8-sig")
    result.balance.to_csv(path / "matching_balance.csv", index=False, encoding="utf-8-sig")
    result.bootstrap.to_csv(path / "block_bootstrap.csv", index=False, encoding="utf-8-sig")
    (path / "summary.json").write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def feature_universe_coverage(panel: pd.DataFrame, features: pd.DataFrame) -> dict:
    dates = set(features["trade_date"].unique())
    panel_counts = panel.loc[panel["trade_date"].isin(dates)].groupby("trade_date")[
        "ts_code"
    ].nunique()
    feature_counts = features.groupby("trade_date")["ts_code"].nunique()
    aligned = pd.concat(
        [panel_counts.rename("panel"), feature_counts.rename("features")], axis=1
    ).dropna()
    ratio = aligned["features"] / aligned["panel"]
    return {
        "dates": int(len(aligned)),
        "median_stock_coverage": float(ratio.median()) if len(ratio) else 0.0,
        "minimum_stock_coverage": float(ratio.min()) if len(ratio) else 0.0,
    }


def render_report(summary, origin, retest) -> str:
    return "\n".join([
        "# Bullish Divergence v2 Corrected E1",
        "",
        "This is a diagnostic revision on already inspected history, not an unseen validation.",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Full-universe median coverage: `{summary['coverage']['median_stock_coverage']:.2%}`",
        "",
        "## Origin event versus price-geometry placebo",
        "",
        f"- Events / mature / matched: `{origin.summary['event_count']}` / "
        f"`{origin.summary['mature_event_count']}` / `{origin.summary['matched_event_count']}`",
        f"- Match rate: `{origin.summary['match_rate']:.2%}`",
        f"- Maximum absolute SMD: `{origin.summary['maximum_absolute_smd']}`",
        "",
        origin.bootstrap.to_markdown(index=False) if len(origin.bootstrap) else "No origin bootstrap results.",
        "",
        "## First post-signal retest",
        "",
        f"- Events / mature / matched: `{retest.summary['event_count']}` / "
        f"`{retest.summary['mature_event_count']}` / `{retest.summary['matched_event_count']}`",
        f"- Match rate: `{retest.summary['match_rate']:.2%}`",
        f"- Maximum absolute SMD: `{retest.summary['maximum_absolute_smd']}`",
        "",
        retest.bootstrap.to_markdown(index=False) if len(retest.bootstrap) else "No retest bootstrap results.",
    ]) + "\n"


def json_default(value):
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run corrected bullish-divergence v2 E1")
    parser.add_argument("--panel", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--retests", required=True)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--caliper", type=float, default=3.0)
    parser.add_argument("--cost-bps", type=float, default=40.0)
    parser.add_argument("--block-length", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-calendar-days", type=int, default=180)
    parser.add_argument("--label-tail-calendar-days", type=int, default=45)
    parser.add_argument(
        "--output-root", default="artifacts/bullish_divergence_v2_event_studies"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
