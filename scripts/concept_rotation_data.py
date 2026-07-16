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


INDEX_PAGE_SIZE = 5_000
MEMBER_PAGE_SIZE = 8_000
_LOCAL = threading.local()


def main() -> None:
    args = parse_args()
    output = Path(args.output_root) / f"dc_{args.start}_{args.end}"
    index_dir = output / "index_monthly"
    member_dir = output / "members_by_concept"
    index_dir.mkdir(parents=True, exist_ok=True)
    member_dir.mkdir(parents=True, exist_ok=True)
    index = fetch_index(args.start, args.end, index_dir)
    concepts = index.loc[index["idx_type"].eq("概念板块"), ["ts_code", "name"]].drop_duplicates("ts_code")
    pending = [row for row in concepts.itertuples(index=False) if not (member_dir / f"{row.ts_code}.parquet").exists()]
    print(f"concept index rows={len(index)} concepts={len(concepts)} pending={len(pending)}", flush=True)
    failures: list[dict] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_one_concept, row.ts_code, args.start, args.end, member_dir): row.ts_code
            for row in pending
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                rows, pages = future.result()
                completed += 1
                if completed == 1 or completed % 10 == 0 or completed == len(pending):
                    print(f"members {completed}/{len(pending)} code={code} rows={rows} pages={pages}", flush=True)
            except Exception as exc:
                failures.append({"concept_code": code, "error": str(exc)})
                print(f"FAILED code={code}: {exc}", flush=True)
    if failures:
        (output / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"concept member fetch failures={len(failures)}")
    (output / "failures.json").unlink(missing_ok=True)
    manifest = build_manifest(output, index, concepts, member_dir, args.start, args.end)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **manifest}, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch PIT Eastmoney concept index/member snapshots")
    parser.add_argument("--start", default="20250630")
    parser.add_argument("--end", default="20260714")
    parser.add_argument("--output-root", default="data/concept_rotation")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


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
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError("unreachable")


def fetch_index(start: str, end: str, output: Path) -> pd.DataFrame:
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    frames = []
    for month in months:
        path = output / f"{month}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
            continue
        left = max(pd.Timestamp(start), month.start_time).strftime("%Y%m%d")
        right = min(pd.Timestamp(end), month.end_time).strftime("%Y%m%d")
        pages = []
        for offset in range(0, 10_000_000, INDEX_PAGE_SIZE):
            page = query_with_retry(
                "dc_index", start_date=left, end_date=right,
                limit=INDEX_PAGE_SIZE, offset=offset,
            )
            if not page.empty:
                pages.append(page)
            if len(page) < INDEX_PAGE_SIZE:
                break
        frame = pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()
        frame.to_parquet(path, index=False)
        frames.append(frame)
        print(f"index month={month} rows={len(frame)}", flush=True)
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["trade_date", "ts_code"], keep="last"
    )


def fetch_one_concept(code: str, start: str, end: str, output: Path) -> tuple[int, int]:
    pages = []
    page_count = 0
    # dc_member refuses very large offsets. Very broad concepts can exceed 100k
    # even in one month, so paginate bounded seven-calendar-day slices.
    for left_date in pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="7D"):
        right_date = min(left_date + pd.Timedelta(days=6), pd.Timestamp(end))
        left, right = left_date.strftime("%Y%m%d"), right_date.strftime("%Y%m%d")
        for offset in range(0, 100_000, MEMBER_PAGE_SIZE):
            page = query_with_retry(
                "dc_member", ts_code=code, start_date=left, end_date=right,
                limit=MEMBER_PAGE_SIZE, offset=offset,
            )
            page_count += 1
            if not page.empty:
                pages.append(page)
            if len(page) < MEMBER_PAGE_SIZE:
                break
        else:
            raise RuntimeError(f"weekly pagination did not terminate for {code} {left}-{right}")
    frame = pd.concat(pages, ignore_index=True) if pages else pd.DataFrame(
        columns=["trade_date", "ts_code", "con_code", "name"]
    )
    if not frame.empty:
        frame = frame.drop_duplicates(["trade_date", "ts_code", "con_code"], keep="last")
        if set(frame["ts_code"].dropna().astype(str)) != {code}:
            raise ValueError(f"unexpected concept codes returned for {code}")
    temporary = output / f".{code}.tmp.parquet"
    final = output / f"{code}.parquet"
    frame.to_parquet(temporary, index=False)
    temporary.replace(final)
    return len(frame), page_count


def build_manifest(output: Path, index: pd.DataFrame, concepts: pd.DataFrame,
                   member_dir: Path, start: str, end: str) -> dict:
    files = sorted(member_dir.glob("*.parquet"))
    rows = 0
    dates: set[str] = set()
    stock_codes: set[str] = set()
    for path in files:
        frame = pd.read_parquet(path, columns=["trade_date", "con_code"])
        rows += len(frame)
        dates.update(frame["trade_date"].dropna().astype(str).unique())
        stock_codes.update(frame["con_code"].dropna().astype(str).unique())
    digest = hashlib.sha256()
    for path in sorted(output.rglob("*.parquet")):
        digest.update(path.relative_to(output).as_posix().encode())
        digest.update(str(path.stat().st_size).encode())
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "tushare_dc", "start_date": start, "end_date": end,
        "index_rows": int(len(index)), "concepts": int(len(concepts)),
        "member_rows": int(rows), "member_dates": int(len(dates)),
        "stocks": int(len(stock_codes)), "files": int(len(files)),
        "size_digest": digest.hexdigest(),
    }


if __name__ == "__main__":
    main()
