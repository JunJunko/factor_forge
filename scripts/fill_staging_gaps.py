from __future__ import annotations

import argparse
from pathlib import Path

from factor_forge.config import load_project
from factor_forge.data.tushare_provider import TushareProvider


ENDPOINTS = {
    "daily": "daily",
    "adj_factor": "adj_factor",
    "daily_basic": "daily_basic",
    "stk_limit": "stk_limit",
    "suspend": "suspend_d",
    "st_status": "stock_st",
}
NONEMPTY_DATASETS = {"daily", "adj_factor", "daily_basic", "stk_limit"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill missing date-partitioned Tushare staging files")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()

    project = load_project(args.config)
    provider = TushareProvider()
    calendar = provider.query(
        "trade_cal", exchange="SSE", start_date=args.start, end_date=args.end
    )
    dates = sorted(
        calendar.loc[calendar["is_open"].astype(int).eq(1), "cal_date"].astype(str).tolist()
    )
    staging = project.paths.data_root / "staging" / f"tushare_{args.start}_{args.end}"
    missing = {
        name: [
            date for date in dates
            if not (staging / name / f"trade_date={date}.parquet").exists()
        ]
        for name in ENDPOINTS
    }
    print("expected_trade_days=", len(dates), sep="")
    for name, gaps in missing.items():
        print(f"{name}: missing={len(gaps)} first={gaps[:1]} last={gaps[-1:]}")

    total = sum(map(len, missing.values()))
    completed = 0
    for name, endpoint in ENDPOINTS.items():
        for date in missing[name]:
            frame = provider.query(endpoint, trade_date=date)
            if name in NONEMPTY_DATASETS and frame.empty:
                print(f"unavailable dataset={name} trade_date={date} rows=0")
                continue
            if "trade_date" not in frame:
                frame["trade_date"] = date
            path = staging / name / f"trade_date={date}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".parquet.tmp")
            frame.to_parquet(temporary, index=False)
            temporary.replace(path)
            completed += 1
            print(f"fetch {completed}/{total} dataset={name} trade_date={date} rows={len(frame)}")

    remaining = []
    for name in ENDPOINTS:
        for date in dates:
            path = staging / name / f"trade_date={date}.parquet"
            if not path.exists():
                remaining.append(str(path))
            elif name in NONEMPTY_DATASETS:
                import pandas as pd

                if pd.read_parquet(path).empty:
                    path.unlink()
                    remaining.append(str(path))
    if remaining:
        raise RuntimeError(f"Staging still has {len(remaining)} missing partitions")
    print(f"COMPLETE staging={staging} trade_days={len(dates)} partitions={len(dates) * len(ENDPOINTS)}")


if __name__ == "__main__":
    main()
