from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from factor_forge.data.tushare_provider import TushareProvider
from concept_rotation_data import fetch_index


PAGE_SIZE = 8_000
_LOCAL = threading.local()


def provider() -> TushareProvider:
    instance = getattr(_LOCAL, "provider", None)
    if instance is None:
        instance = TushareProvider()
        _LOCAL.provider = instance
    return instance


def query_with_retry(endpoint: str, **kwargs) -> pd.DataFrame:
    for attempt in range(6):
        try:
            return provider().query(endpoint, **kwargs)
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 16))
    raise RuntimeError("unreachable")


def fetch_date(date: str, concepts: frozenset[str], output: Path) -> tuple[str, int, int]:
    path = output / f"trade_date={date}.parquet"
    if path.exists():
        return date, len(pd.read_parquet(path, columns=["ts_code"])), 0
    pages: list[pd.DataFrame] = []
    page_count = 0
    for offset in range(0, 200_000, PAGE_SIZE):
        raw_page = query_with_retry(
            "dc_member", trade_date=date, limit=PAGE_SIZE, offset=offset,
        )
        page_count += 1
        if not raw_page.empty:
            page = raw_page.loc[raw_page["ts_code"].astype(str).isin(concepts)]
            if not page.empty:
                pages.append(page)
        if len(raw_page) < PAGE_SIZE:
            break
        if offset + PAGE_SIZE >= 200_000:
            raise RuntimeError(f"member pagination did not terminate for {date}")
    frame = pd.concat(pages, ignore_index=True) if pages else pd.DataFrame(
        columns=["trade_date", "ts_code", "con_code", "name"]
    )
    frame = frame.drop_duplicates(["trade_date", "ts_code", "con_code"], keep="last")
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    return date, len(frame), page_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill PIT Eastmoney concept members by trade date")
    parser.add_argument("--start", default="20241230")
    parser.add_argument("--end", default="20250627")
    parser.add_argument("--output-root", default="data/concept_rotation")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    output = Path(args.output_root) / f"dc_{args.start}_{args.end}_by_date"
    index_dir = output / "index_monthly"
    member_dir = output / "members_by_concept"
    index_dir.mkdir(parents=True, exist_ok=True)
    member_dir.mkdir(parents=True, exist_ok=True)
    index = fetch_index(args.start, args.end, index_dir)
    # The endpoint's Chinese labels can be mojibaked by upstream transports.
    # Concept boards are the type with the largest distinct board universe.
    concept_type = index.groupby("idx_type", observed=True)["ts_code"].nunique().idxmax()
    concepts = frozenset(index.loc[index["idx_type"].eq(concept_type), "ts_code"].dropna().astype(str))
    calendar = provider().query(
        "trade_cal", exchange="SSE", start_date=args.start, end_date=args.end,
        fields="cal_date,is_open",
    )
    dates = sorted(calendar.loc[calendar["is_open"].astype(int).eq(1), "cal_date"].astype(str))
    total_rows = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_date, date, concepts, member_dir): date for date in dates}
        for position, future in enumerate(as_completed(futures), start=1):
            date, rows, pages = future.result()
            total_rows += rows
            if position == 1 or position % 10 == 0 or position == len(futures):
                print(f"members {position}/{len(futures)} date={date} rows={rows} pages={pages}", flush=True)

    digest = hashlib.sha256()
    for path in sorted(output.rglob("*.parquet")):
        digest.update(path.relative_to(output).as_posix().encode())
        digest.update(str(path.stat().st_size).encode())
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "tushare_dc_by_trade_date",
        "start_date": args.start,
        "end_date": args.end,
        "index_rows": int(len(index)),
        "concepts": int(len(concepts)),
        "member_rows": int(total_rows),
        "member_dates": int(len(dates)),
        "files": int(len(list(member_dir.glob("*.parquet")))),
        "size_digest": digest.hexdigest(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **manifest}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
