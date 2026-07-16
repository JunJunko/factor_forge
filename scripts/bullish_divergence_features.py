from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from factor_forge.ml.bullish_divergence_config import BullishDivergenceFeatureConfig
from factor_forge.ml.bullish_divergence_dataset import (
    REQUIRED_PANEL_COLUMNS,
    build_bullish_divergence_features,
    build_divergence_episodes,
    bullish_divergence_feature_manifest,
)


OPTIONAL_COLUMNS = [
    "industry_l1_code", "industry_l2_code", "log_circ_mv", "circ_mv_cny",
    "main_sell_ratio", "sell_main_ratio", "sell_pressure_ratio",
    "sell_lg_amount", "sell_elg_amount", "sell_lg_amount_cny", "sell_elg_amount_cny",
]


def main() -> None:
    args = parse_args()
    panel_path = Path(args.panel)
    schema = set(pq.ParquetFile(panel_path).schema_arrow.names)
    missing = REQUIRED_PANEL_COLUMNS - schema
    if missing:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")
    columns = sorted(REQUIRED_PANEL_COLUMNS | (schema & set(OPTIONAL_COLUMNS)))
    filters: list[tuple] = []
    signal_start = pd.Timestamp(args.start_date) if args.start_date else None
    if signal_start is not None:
        read_start = signal_start - pd.Timedelta(days=args.warmup_calendar_days)
        filters.append(("trade_date", ">=", read_start))
    if args.end_date:
        filters.append(("trade_date", "<=", pd.Timestamp(args.end_date)))
    if args.stock_code:
        filters.append(("ts_code", "in", args.stock_code))
    panel = pd.read_parquet(panel_path, columns=columns, filters=filters or None)
    if args.max_stocks:
        codes = sorted(panel["ts_code"].dropna().astype(str).unique())[:args.max_stocks]
        panel = panel.loc[panel["ts_code"].astype(str).isin(codes)].copy()
    if panel.empty:
        raise ValueError("panel filters produced no rows")

    config = BullishDivergenceFeatureConfig()
    daily, model_features = build_bullish_divergence_features(panel, config)
    if signal_start is not None:
        daily = daily.loc[daily["trade_date"].ge(signal_start)].copy()
    episodes = build_divergence_episodes(daily, config)
    candidates = daily.loc[daily["div__event_candidate"].fillna(False)]
    post_b_observable = candidates["touch__post_b_observable"].eq(1)

    created = datetime.now(timezone.utc)
    output = Path(args.output_root) / created.strftime("bullish_divergence_%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    daily.to_parquet(output / "stock_day_features.parquet", index=False)
    episodes.to_parquet(output / "divergence_episodes.parquet", index=False)
    (output / "feature_manifest.json").write_text(
        json.dumps(bullish_divergence_feature_manifest(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "created_at": created.isoformat(),
        "panel_path": str(panel_path),
        "signal_start": str(signal_start.date()) if signal_start is not None else None,
        "signal_end": str(pd.Timestamp(daily["trade_date"].max()).date()),
        "input_rows_with_warmup": int(len(panel)),
        "output_rows": int(len(daily)),
        "stocks": int(daily["ts_code"].nunique()),
        "model_feature_count": int(len(model_features)),
        "event_candidate_rows": int(daily["div__event_candidate"].fillna(False).sum()),
        "episode_count": int(len(episodes)),
        "touch_observation_rate": float(daily["touch__occurred_10d"].mean()),
        "candidate_touch_rate": (
            float(candidates["touch__occurred_10d"].mean()) if len(candidates) else None
        ),
        "candidate_post_b_observable_rate": (
            float(post_b_observable.mean()) if len(candidates) else None
        ),
        "candidate_post_b_touch_rate_when_observable": (
            float(candidates.loc[post_b_observable, "touch__post_b_count"].gt(0).mean())
            if post_b_observable.any() else None
        ),
        "config": asdict(config),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build causal bullish-divergence and support-touch features"
    )
    parser.add_argument("--panel", required=True)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--stock-code", action="append", default=[])
    parser.add_argument("--max-stocks", type=int)
    parser.add_argument("--warmup-calendar-days", type=int, default=180)
    parser.add_argument("--output-root", default="artifacts/bullish_divergence_runs")
    return parser.parse_args()


if __name__ == "__main__":
    main()
