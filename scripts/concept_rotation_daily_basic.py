from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import threading

import pandas as pd

from factor_forge.data.tushare_provider import TushareProvider


FIELDS = "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,free_share,circ_mv"
_LOCAL = threading.local()


def provider() -> TushareProvider:
    instance = getattr(_LOCAL, "provider", None)
    if instance is None:
        instance = TushareProvider()
        _LOCAL.provider = instance
    return instance


def fetch_date(trade_date: str, output: Path) -> tuple[str, int]:
    path = output / f"trade_date={trade_date}.parquet"
    if path.exists():
        return trade_date, len(pd.read_parquet(path, columns=["ts_code"]))
    frame = provider().query("daily_basic", trade_date=trade_date, fields=FIELDS)
    if frame.empty:
        raise RuntimeError(f"daily_basic returned no rows for {trade_date}")
    frame = frame.drop_duplicates(["trade_date", "ts_code"], keep="last")
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    return trade_date, len(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch enhanced daily_basic fields for concept rotation")
    parser.add_argument("--start", default="20250501")
    parser.add_argument("--end", default="20260714")
    parser.add_argument("--output", default="data/concept_rotation/daily_basic_20250501_20260714")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    output = Path(args.output)
    partitions = output / "partitions"
    partitions.mkdir(parents=True, exist_ok=True)
    calendar = provider().query(
        "trade_cal", exchange="SSE", start_date=args.start, end_date=args.end,
        fields="cal_date,is_open",
    )
    dates = sorted(calendar.loc[calendar["is_open"].astype(int).eq(1), "cal_date"].astype(str))
    rows = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_date, date, partitions): date for date in dates}
        for position, future in enumerate(as_completed(futures), start=1):
            date, count = future.result()
            rows += count
            if position == 1 or position % 20 == 0 or position == len(futures):
                print(f"daily_basic {position}/{len(futures)} date={date} rows={count}", flush=True)

    files = sorted(partitions.glob("trade_date=*.parquet"))
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.sort_values(["trade_date", "ts_code"]).drop_duplicates(
        ["trade_date", "ts_code"], keep="last"
    )
    final = output / "daily_basic_enhanced.parquet"
    frame.to_parquet(final, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start_date": args.start,
        "end_date": args.end,
        "trade_dates": int(frame["trade_date"].nunique()),
        "rows": int(len(frame)),
        "stocks": int(frame["ts_code"].nunique()),
        "duplicate_keys": int(frame.duplicated(["trade_date", "ts_code"]).sum()),
        "missing_free_share": int(frame["free_share"].isna().sum()),
        "missing_turnover_rate_f": int(frame["turnover_rate_f"].isna().sum()),
        "fields": FIELDS.split(","),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"path": str(final), **manifest}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
