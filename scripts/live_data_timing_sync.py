from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from factor_forge.config import load_project
from factor_forge.data.ingestion import TushareIngestor
from factor_forge.data.repository import DataVersionRepository, is_complete_manifest
from factor_forge.data.tushare_provider import TushareProvider
from factor_forge.timing.position_model import TimingPositionModelRunner
from factor_forge.timing.runner import TimingDatasetBuildRunner


DEFAULT_PROJECT = Path("configs/project.yaml")
DEFAULT_TIMING_BUILD_CONFIG = Path("configs/ml/timing_factor_library_v1.yaml")
DEFAULT_TIMING_POSITION_CONFIG = Path("configs/ml/timing_position_model_v1.yaml")
DEFAULT_TIMING_DIR = Path("data/timing")
TEMP_POSITION_CONFIG = Path("artifacts/timing_position_models/timing_position_model_v1_latest_web_config.yaml")


def main() -> None:
    args = parse_args()
    provider = TushareProvider()
    target_open_dates = trade_dates_with_lag(provider, args.start, args.end, include_lag_day=False)
    open_dates = trade_dates_with_lag(provider, args.start, args.end, include_lag_day=not args.no_lag_day)
    fetch_start = min(open_dates) if open_dates else args.start
    fetch_end = max(open_dates) if open_dates else args.end
    log(
        f"requested start={args.start} end={args.end}; "
        f"target_open_dates={target_open_dates}; fetch_start={fetch_start} fetch_end={fetch_end}"
    )

    summary: dict[str, Any] = {
        "requested_start": args.start,
        "requested_end": args.end,
        "fetch_start": fetch_start,
        "fetch_end": fetch_end,
        "target_open_dates": target_open_dates,
        "open_dates": open_dates,
    }
    timing_required_date = required_timing_date(target_open_dates, open_dates)
    timing_input_dates = [date for date in open_dates if timing_required_date is None or date <= timing_required_date]
    summary["timing_required_date"] = timing_required_date
    summary["timing_input_dates"] = timing_input_dates
    panel_gap = inspect_main_panel_dates(args.project, open_dates)
    summary["panel_gap_before"] = panel_gap
    missing_main_dates = panel_gap["missing_dates"]
    if missing_main_dates:
        log(f"main panel missing dates: {missing_main_dates}")
    else:
        log("main panel required dates already complete; skip main ingest")
    data_status = sync_main_panel_dates(missing_main_dates, open_dates, provider, args.project, args.merge_full_history)
    summary["data"] = data_status

    margin_status = sync_margin(provider, timing_input_dates, args.timing_dir)
    summary["margin"] = margin_status
    stock_status = ensure_timing_stock_daily(args.project, args.timing_dir, timing_input_dates)
    summary["stock_daily"] = stock_status

    if args.skip_timing:
        log("skip timing feature/model rebuild")
    elif not should_rebuild_timing(
        target_open_dates=target_open_dates,
        open_dates=open_dates,
        stock_status=stock_status,
        margin_status=margin_status,
        force=args.force_timing,
    ):
        timing_status = latest_timing_position_status()
        summary["timing_position"] = {"skipped": True, "reason": "latest timing position already covers required entry dates", **timing_status}
        log(
            "timing position already current; skip rebuild "
            f"latest={timing_status.get('latest', {}).get('trade_date')}"
        )
    else:
        timing_features = TimingDatasetBuildRunner().run(args.timing_build_config)
        log(f"timing features rebuilt: {timing_features['run_dir']}")
        position_config = write_position_config(
            base_config=args.timing_position_config,
            timing_dataset=Path(timing_features["dataset_path"]),
            feature_names=Path(timing_features["run_dir"]) / "feature_names.json",
        )
        timing_position = TimingPositionModelRunner().run(position_config)
        timing_daily = Path(timing_position["run_dir"]) / "timing_position_daily.csv"
        latest_position = latest_timing_position(timing_daily)
        summary["timing_features"] = timing_features
        summary["timing_position"] = {
            **timing_position,
            "timing_daily": str(timing_daily),
            "latest": latest_position,
        }
        log(
            "timing position rebuilt: "
            f"{timing_position['run_dir']} latest={latest_position.get('trade_date')} "
            f"target={latest_position.get('target_position')}"
        )

    output = Path(args.output_json) if args.output_json else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
        log(f"summary written: {output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync main data, timing raw inputs, timing features, and timing position model.")
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--timing-dir", type=Path, default=DEFAULT_TIMING_DIR)
    parser.add_argument("--timing-build-config", type=Path, default=DEFAULT_TIMING_BUILD_CONFIG)
    parser.add_argument("--timing-position-config", type=Path, default=DEFAULT_TIMING_POSITION_CONFIG)
    parser.add_argument("--no-merge-full-history", dest="merge_full_history", action="store_false")
    parser.add_argument("--no-lag-day", action="store_true", help="Do not automatically include the previous open day.")
    parser.add_argument("--skip-timing", action="store_true", help="Only sync main panel and timing raw tables.")
    parser.add_argument("--force-timing", action="store_true", help="Rebuild timing features/model even if coverage is already sufficient.")
    parser.add_argument("--output-json", default="")
    parser.set_defaults(merge_full_history=True)
    return parser.parse_args()


def trade_dates_with_lag(provider: TushareProvider, start: str, end: str, *, include_lag_day: bool) -> list[str]:
    start_dt = datetime.strptime(start, "%Y%m%d")
    cal_start = (start_dt - timedelta(days=14)).strftime("%Y%m%d") if include_lag_day else start
    calendar = provider.query("trade_cal", exchange="SSE", start_date=cal_start, end_date=end)
    dates = (
        calendar.loc[pd.to_numeric(calendar["is_open"], errors="coerce").eq(1), "cal_date"]
        .dropna()
        .astype(str)
        .sort_values()
        .tolist()
    )
    wanted = [d for d in dates if start <= d <= end]
    if include_lag_day:
        previous = [d for d in dates if d < start]
        if previous:
            wanted = [previous[-1], *wanted]
    return sorted(set(wanted))


def sync_main_panel_dates(
    missing_dates: list[str],
    calendar_dates: list[str],
    provider: TushareProvider,
    project_path: Path,
    merge_full_history: bool,
) -> dict[str, Any]:
    project = load_project(project_path)
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    if not missing_dates:
        resolved, manifest = repo.load_manifest("latest")
        return {"version": resolved, "skipped": True, "missing_dates": [], **manifest}

    version = ""
    for start, end in contiguous_date_ranges(missing_dates, calendar_dates):
        version = sync_main_panel(start, end, provider, project_path, merge_full_history)
    resolved, manifest = repo.load_manifest("latest")
    return {"version": resolved, "skipped": False, "missing_dates": missing_dates, **manifest}


def sync_main_panel(
    start: str,
    end: str,
    provider: TushareProvider,
    project_path: Path,
    merge_full_history: bool,
) -> str:
    project = load_project(project_path)

    def progress(done: int, total: int, date: str) -> None:
        log(f"main ingest {done}/{total}: {date}")

    version = TushareIngestor(project, provider, progress=progress).ingest(
        start, end, version_kind="incremental"
    )
    log(f"main ingest published increment={version}")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    if merge_full_history:
        resolved, manifest = repo.load_manifest(version)
        if not is_complete_manifest(manifest):
            version = merge_increment_to_complete(repo, version)
            log(f"merged complete data_version={version}")
    return version


def merge_increment_to_complete(repo: DataVersionRepository, increment_version: str) -> str:
    base_version = previous_complete_version(repo, exclude=increment_version)
    if not base_version:
        raise RuntimeError("No previous complete data version found to merge with increment.")
    _, base = repo.load_panel(base_version)
    _, inc = repo.load_panel(increment_version)
    panel = pd.concat([base, inc], ignore_index=True)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = (
        panel.sort_values(["trade_date", "ts_code"])
        .drop_duplicates(["trade_date", "ts_code"], keep="last")
        .sort_values(["ts_code", "trade_date"])
        .reset_index(drop=True)
    )
    recompute_panel_flags(panel)
    return repo.publish(panel, raw_datasets=None, source="tushare_append_live_sync")


def previous_complete_version(repo: DataVersionRepository, *, exclude: str) -> str | None:
    with repo.metadata.connect() as conn:
        rows = conn.execute(
            "SELECT data_version FROM meta_data_version WHERE quality_status='PASSED' ORDER BY created_at DESC"
        ).fetchall()
    for row in rows:
        version = row["data_version"]
        if version == exclude:
            continue
        try:
            _, manifest = repo.load_manifest(version)
        except FileNotFoundError:
            continue
        if is_complete_manifest(manifest):
            return version
    return None


def inspect_main_panel_dates(project_path: Path, required_dates: list[str]) -> dict[str, Any]:
    project = load_project(project_path)
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, panel = repo.load_panel("latest")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    required = [pd.Timestamp(d).normalize() for d in required_dates]
    rows = []
    missing: list[str] = []
    for date in required:
        day = panel.loc[panel["trade_date"].eq(date)]
        tradeable_rows = int(day["is_tradeable"].fillna(False).astype(bool).sum()) if "is_tradeable" in day else 0
        ok = len(day) > 0 and tradeable_rows > 0
        if not ok:
            missing.append(date.strftime("%Y%m%d"))
        rows.append(
            {
                "trade_date": date.strftime("%Y-%m-%d"),
                "rows": int(len(day)),
                "tradeable_rows": tradeable_rows,
                "status": "ok" if ok else ("missing" if day.empty else "no_tradeable_rows"),
            }
        )
    return {
        "version": version,
        "panel_start": panel["trade_date"].min() if len(panel) else None,
        "panel_end": panel["trade_date"].max() if len(panel) else None,
        "required_dates": [d.strftime("%Y%m%d") for d in required],
        "missing_dates": missing,
        "date_rows": rows,
    }


def contiguous_date_ranges(missing_dates: list[str], calendar_dates: list[str]) -> list[tuple[str, str]]:
    missing = [d for d in calendar_dates if d in set(missing_dates)]
    if not missing:
        return []
    ranges: list[tuple[str, str]] = []
    start = prev = missing[0]
    positions = {date: idx for idx, date in enumerate(calendar_dates)}
    for date in missing[1:]:
        if positions[date] == positions[prev] + 1:
            prev = date
            continue
        ranges.append((start, prev))
        start = prev = date
    ranges.append((start, prev))
    return ranges


def recompute_panel_flags(panel: pd.DataFrame) -> None:
    grouped = panel.groupby("ts_code", sort=False)
    panel["listing_trade_days"] = grouped.cumcount() + 1
    panel["is_tradeable"] = (
        panel["raw_open"].notna()
        & panel["adj_open"].notna()
        & ~panel["is_suspended"].fillna(True).astype(bool)
    )
    panel["is_factor_eligible"] = (
        panel["is_tradeable"].fillna(False).astype(bool)
        & ~panel["is_st"].fillna(False).astype(bool)
        & ~panel["is_delisting_period"].fillna(False).astype(bool)
        & panel["listing_trade_days"].ge(60)
    )
    rolling_amount = (
        panel["amount_cny"].where(panel["amount_cny"] > 0)
        .groupby(panel["ts_code"], sort=False)
        .rolling(20, min_periods=18)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(panel.index)
    )
    rank = rolling_amount.where(panel["is_tradeable"]).groupby(panel["trade_date"], sort=False).rank(
        method="first", ascending=False
    )
    panel["is_liquid"] = rank.le(1000).fillna(False)


def sync_margin(provider: TushareProvider, open_dates: list[str], timing_dir: Path) -> dict[str, Any]:
    path = timing_dir / "margin.parquet"
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    existing_dates = set()
    if not existing.empty and "trade_date" in existing:
        existing_dates = set(existing["trade_date"].astype(str))
    missing_dates = [date for date in open_dates if date not in existing_dates]
    if not missing_dates:
        log("margin required dates already complete; skip margin fetch")
        return {
            "path": str(path),
            "rows": int(len(existing)),
            "date_start": str(pd.to_datetime(existing["trade_date"]).min().date()) if "trade_date" in existing and not existing.empty else None,
            "date_end": str(pd.to_datetime(existing["trade_date"]).max().date()) if "trade_date" in existing and not existing.empty else None,
            "missing_dates": [],
            "fetched_dates": [],
            "skipped": True,
            "changed": False,
        }
    frames = []
    fetched_dates = []
    for date in missing_dates:
        try:
            frame = provider.query("margin", trade_date=date)
        except Exception as exc:
            log(f"margin {date} failed: {exc}")
            frame = pd.DataFrame()
        if frame is not None and not frame.empty:
            frames.append(frame)
            fetched_dates.append(date)
            log(f"margin {date}: rows={len(frame)}")
        else:
            log(f"margin {date}: rows=0")
    if frames:
        combined = pd.concat([existing, *frames], ignore_index=True)
    else:
        combined = existing.copy()
    if not combined.empty:
        combined["trade_date"] = combined["trade_date"].astype(str)
        key_cols = [c for c in ["trade_date", "exchange_id"] if c in combined.columns]
        combined = combined.sort_values(key_cols).drop_duplicates(key_cols, keep="last") if key_cols else combined
    timing_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    return {
        "path": str(path),
        "rows": int(len(combined)),
        "date_start": str(pd.to_datetime(combined["trade_date"]).min().date()) if "trade_date" in combined and not combined.empty else None,
        "date_end": str(pd.to_datetime(combined["trade_date"]).max().date()) if "trade_date" in combined and not combined.empty else None,
        "missing_dates": missing_dates,
        "fetched_dates": fetched_dates,
        "skipped": False,
        "changed": bool(fetched_dates),
    }


def ensure_timing_stock_daily(project_path: Path, timing_dir: Path, required_dates: list[str]) -> dict[str, Any]:
    path = timing_dir / "stock_daily.parquet"
    if path.exists():
        try:
            existing = pd.read_parquet(path, columns=["trade_date", "ts_code", "is_tradeable"])
        except Exception:
            existing = pd.DataFrame()
        if not existing.empty:
            existing["trade_date"] = pd.to_datetime(existing["trade_date"])
            missing = []
            for date in [pd.Timestamp(d).normalize() for d in required_dates]:
                day = existing.loc[existing["trade_date"].eq(date)]
                tradeable_rows = int(day["is_tradeable"].fillna(False).astype(bool).sum()) if "is_tradeable" in day else int(len(day))
                if day.empty or tradeable_rows <= 0:
                    missing.append(date.strftime("%Y%m%d"))
            if not missing:
                log("timing stock_daily required dates already complete; skip stock_daily rebuild")
                return {
                    "path": str(path),
                    "rows": int(len(existing)),
                    "date_start": str(existing["trade_date"].min().date()),
                    "date_end": str(existing["trade_date"].max().date()),
                    "missing_dates": [],
                    "skipped": True,
                    "changed": False,
                }
            log(f"timing stock_daily missing dates: {missing}")
    status = update_timing_stock_daily(project_path, timing_dir)
    return {**status, "skipped": False, "changed": True}


def update_timing_stock_daily(project_path: Path, timing_dir: Path) -> dict[str, Any]:
    project = load_project(project_path)
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, panel = repo.load_panel("latest")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    columns = [
        c
        for c in [
            "trade_date",
            "ts_code",
            "adj_close",
            "close",
            "pre_close",
            "pct_chg",
            "amount",
            "amount_cny",
            "is_tradeable",
            "is_liquid",
        ]
        if c in panel.columns
    ]
    stock_daily = panel[columns].copy()
    if "pct_chg" not in stock_daily and "adj_close" in stock_daily:
        stock_daily = stock_daily.sort_values(["ts_code", "trade_date"])
        stock_daily["pct_chg"] = stock_daily.groupby("ts_code")["adj_close"].pct_change(fill_method=None)
    if "amount" not in stock_daily and "amount_cny" in stock_daily:
        stock_daily["amount"] = stock_daily["amount_cny"]
    stock_daily = stock_daily.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    path = timing_dir / "stock_daily.parquet"
    timing_dir.mkdir(parents=True, exist_ok=True)
    stock_daily.to_parquet(path, index=False)
    log(f"timing stock_daily updated from {version}: rows={len(stock_daily):,}")
    return {
        "path": str(path),
        "data_version": version,
        "rows": int(len(stock_daily)),
        "date_start": str(stock_daily["trade_date"].min().date()) if len(stock_daily) else None,
        "date_end": str(stock_daily["trade_date"].max().date()) if len(stock_daily) else None,
    }


def should_rebuild_timing(
    *,
    target_open_dates: list[str],
    open_dates: list[str],
    stock_status: dict[str, Any],
    margin_status: dict[str, Any],
    force: bool,
) -> bool:
    if force:
        return True
    timing_status = latest_timing_position_status()
    required = required_timing_date(target_open_dates, open_dates)
    latest = timing_status.get("latest", {}).get("trade_date")
    if stock_status.get("changed") or margin_status.get("changed"):
        return True
    if required is None:
        return False
    if not latest:
        return True
    return pd.Timestamp(latest).normalize() < pd.Timestamp(required).normalize()


def required_timing_date(target_open_dates: list[str], open_dates: list[str]) -> str | None:
    if not target_open_dates:
        return None
    target = max(target_open_dates)
    previous = [date for date in open_dates if date < target]
    return previous[-1] if previous else target


def latest_timing_position_status() -> dict[str, Any]:
    root = Path("artifacts/timing_position_models")
    candidates = [p for p in root.glob("timing_position_model_v1_*/timing_position_daily.csv") if p.is_file()]
    if not candidates:
        return {"latest": {}, "path": None}
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    return {"path": str(path), "latest": latest_timing_position(path)}


def write_position_config(base_config: Path, timing_dataset: Path, feature_names: Path) -> Path:
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg["dataset_path"] = str(timing_dataset)
    cfg["feature_names_path"] = str(feature_names)
    TEMP_POSITION_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    TEMP_POSITION_CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return TEMP_POSITION_CONFIG


def latest_timing_position(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    daily = pd.read_csv(path, parse_dates=["trade_date"])
    if daily.empty:
        return {}
    row = daily.sort_values("trade_date").iloc[-1].replace({pd.NA: None})
    return {
        "trade_date": row.get("trade_date"),
        "prediction": row.get("prediction"),
        "raw_position": row.get("raw_position"),
        "target_position": row.get("target_position"),
        "sample": row.get("sample"),
    }


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


if __name__ == "__main__":
    main()
