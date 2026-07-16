from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from factor_forge.ml.bullish_divergence_dataset import (
    REQUIRED_PANEL_COLUMNS,
    bullish_divergence_feature_manifest,
)
from factor_forge.ml.bullish_divergence_v2 import (
    V2_FEATURES,
    BullishDivergenceV2Config,
    build_bullish_divergence_v2_features,
    build_v2_divergence_episodes,
    build_v2_post_signal_retest_events,
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
    if missing := REQUIRED_PANEL_COLUMNS - schema:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")
    columns = sorted(REQUIRED_PANEL_COLUMNS | (schema & set(OPTIONAL_COLUMNS)))
    signal_start = pd.Timestamp(args.start_date) if args.start_date else None
    filters: list[tuple] = []
    if signal_start is not None:
        filters.append((
            "trade_date", ">=",
            signal_start - pd.Timedelta(days=args.warmup_calendar_days),
        ))
    if args.end_date:
        filters.append(("trade_date", "<=", pd.Timestamp(args.end_date)))
    panel = pd.read_parquet(panel_path, columns=columns, filters=filters or None)
    if panel.empty:
        raise ValueError("panel filters produced no rows")

    config = BullishDivergenceV2Config()
    daily, model_features = build_bullish_divergence_v2_features(panel, config)
    if signal_start is not None:
        daily = daily.loc[daily["trade_date"].ge(signal_start)].copy()
    episodes = build_v2_divergence_episodes(daily, config)
    retests = build_v2_post_signal_retest_events(panel, episodes, config)
    geometry = daily.loc[daily["div_v2__geometry_candidate"].fillna(False)]
    candidates = daily.loc[daily["div_v2__event_candidate"].fillna(False)]

    created = datetime.now(timezone.utc)
    output = Path(args.output_root) / created.strftime("bullish_divergence_v2_%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    daily.to_parquet(output / "stock_day_features.parquet", index=False)
    episodes.to_parquet(output / "divergence_episodes.parquet", index=False)
    retests.to_parquet(output / "post_signal_retest_events.parquet", index=False)
    manifest = bullish_divergence_feature_manifest() + [
        {
            "name": name,
            "group": "D_divergence_v2" if name.startswith("div_v2__") else "T_support_v2",
            "role": "predictor" if not name.endswith("candidate") else "filter",
            "clock": "signal_close_T",
            "missing_policy": "preserve_nan",
        }
        for name in V2_FEATURES
    ]
    (output / "feature_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "created_at": created.isoformat(),
        "research_status": "DIAGNOSTIC_V2_INSPECTED_HISTORY",
        "panel_path": str(panel_path),
        "signal_start": str(signal_start.date()) if signal_start is not None else None,
        "signal_end": str(pd.Timestamp(daily["trade_date"].max()).date()),
        "output_rows": int(len(daily)),
        "stocks": int(daily["ts_code"].nunique()),
        "model_feature_count": int(len(model_features)),
        "geometry_candidate_rows": int(len(geometry)),
        "divergence_candidate_rows": int(len(candidates)),
        "episode_count": int(len(episodes)),
        "post_signal_retest_event_count": int(len(retests)),
        "geometry_without_oscillator_rows": int(
            (geometry["div_v2__event_candidate"].fillna(False) == False).sum()  # noqa: E712
        ),
        "pre_b_support_episode_rate": (
            float(episodes["support_v2__pre_b_present"].mean()) if len(episodes) else None
        ),
        "post_signal_retest_rate": float(len(retests) / len(episodes)) if len(episodes) else None,
        "post_signal_reclaim_rate": (
            float(retests["retest_v2__reclaimed"].mean()) if len(retests) else None
        ),
        "config": asdict(config),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build mechanism-aligned bullish-divergence v2 features"
    )
    parser.add_argument("--panel", required=True)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--warmup-calendar-days", type=int, default=180)
    parser.add_argument(
        "--output-root", default="artifacts/bullish_divergence_v2_runs"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

