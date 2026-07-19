from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from factor_forge.data.tushare_provider import TushareProvider
from factor_forge.research.index_backed_rotation import (
    recover_delisted_theme_etf_candidates,
)


def query_with_retry(provider: TushareProvider, endpoint: str, **kwargs) -> pd.DataFrame:
    for attempt in range(7):
        try:
            return provider.query(endpoint, **kwargs)
        except Exception:
            if attempt == 6:
                raise
            time.sleep(min(2**attempt, 20))
    raise RuntimeError("unreachable")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = Path(config["source_snapshot"])
    provider = TushareProvider()

    print("loading current and delisted ETF metadata", flush=True)
    etf_basic = pd.concat([
        query_with_retry(provider, "etf_basic", list_status=status)
        for status in ("L", "D")
    ], ignore_index=True).drop_duplicates("ts_code", keep="first")
    fund_basic_delisted = query_with_retry(provider, "fund_basic", market="E", status="D")
    etf_indexes = query_with_retry(provider, "etf_index")
    recovered, recovery_audit = recover_delisted_theme_etf_candidates(
        etf_basic,
        fund_basic_delisted,
        etf_indexes,
        as_of=config["history_end"],
    )

    base_candidates = pd.read_parquet(source / "theme_etf_candidates.parquet").copy()
    base_candidates["mapping_source"] = "current_etf_basic_exact"
    candidate_columns = sorted(set(base_candidates.columns) | set(recovered.columns))
    candidates = pd.concat([
        base_candidates.reindex(columns=candidate_columns),
        recovered.reindex(columns=candidate_columns),
    ], ignore_index=True).drop_duplicates("ts_code", keep="last")
    print(
        f"candidates={candidates['ts_code'].nunique()} "
        f"recovered_delisted={recovered['ts_code'].nunique()} "
        f"indexes={candidates['index_code'].nunique()}",
        flush=True,
    )

    base_weights = pd.read_parquet(source / "index_weights.parquet")
    base_weights["trade_date"] = base_weights["trade_date"].astype(str)
    base_codes = set(base_weights["index_code"].astype(str))
    codes = sorted(candidates["index_code"].dropna().astype(str).unique())
    weight_parts = [base_weights]
    for position, code in enumerate(codes, start=1):
        historical = query_index_weights_complete(
            provider,
            code,
            config["weight_history_start"],
            "2022-12-31",
        )
        if not historical.empty:
            weight_parts.append(historical)
        if code not in base_codes:
            recent = query_index_weights_complete(
                provider,
                code,
                config["history_start"],
                config["history_end"],
            )
            if not recent.empty:
                weight_parts.append(recent)
        if position == 1 or position % 20 == 0 or position == len(codes):
            print(f"index weights {position}/{len(codes)} code={code}", flush=True)
    weights = pd.concat(weight_parts, ignore_index=True).drop_duplicates(
        ["index_code", "trade_date", "con_code"], keep="last",
    )

    fund_daily = pd.read_parquet(source / "fund_daily.parquet")
    fund_share = pd.read_parquet(source / "fund_share.parquet")
    daily_parts, share_parts = [fund_daily], [fund_share]
    for position, item in enumerate(recovered.itertuples(index=False), start=1):
        end = min(pd.Timestamp(config["history_end"]), pd.Timestamp(item.delist_date))
        daily_parts.append(query_with_retry(
            provider,
            "fund_daily",
            ts_code=item.ts_code,
            start_date=config["history_start"].replace("-", ""),
            end_date=end.strftime("%Y%m%d"),
        ))
        share_parts.append(query_with_retry(
            provider,
            "fund_share",
            ts_code=item.ts_code,
            start_date=config["history_start"].replace("-", ""),
            end_date=end.strftime("%Y%m%d"),
        ))
        print(f"delisted ETF histories {position}/{len(recovered)} {item.ts_code}", flush=True)
    fund_daily = pd.concat(daily_parts, ignore_index=True).drop_duplicates(
        ["ts_code", "trade_date"], keep="last",
    )
    fund_share = pd.concat(share_parts, ignore_index=True).drop_duplicates(
        ["ts_code", "trade_date"], keep="last",
    )

    digest = hashlib.sha256(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    output = Path(args.output_root) / (
        f"strict_pit_{config['weight_history_start'].replace('-', '')}_"
        f"{config['history_end'].replace('-', '')}_{digest}"
    )
    if output.exists():
        raise FileExistsError(f"immutable R7 snapshot already exists: {output}")
    output.mkdir(parents=True)
    etf_basic.to_parquet(output / "etf_basic_all_statuses.parquet", index=False)
    fund_basic_delisted.to_parquet(output / "fund_basic_delisted.parquet", index=False)
    etf_indexes.to_parquet(output / "etf_indexes.parquet", index=False)
    candidates.to_parquet(output / "theme_etf_candidates.parquet", index=False)
    recovery_audit.to_csv(output / "delisted_recovery_audit.csv", index=False, encoding="utf-8-sig")
    weights.to_parquet(output / "index_weights.parquet", index=False)
    fund_daily.to_parquet(output / "fund_daily.parquet", index=False)
    fund_share.to_parquet(output / "fund_share.parquet", index=False)
    weight_dates = pd.to_datetime(weights["trade_date"].astype(str))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "IMMUTABLE_R7_STRICT_PIT_SNAPSHOT",
        "config": str(config_path.resolve()),
        "source_snapshot": str(source.resolve()),
        "weight_history_start": str(weight_dates.min().date()),
        "weight_history_end": str(weight_dates.max().date()),
        "candidate_etfs": int(candidates["ts_code"].nunique()),
        "candidate_indexes": int(candidates["index_code"].nunique()),
        "recovered_delisted_theme_etfs": int(recovered["ts_code"].nunique()),
        "unresolved_delisted_etfs": int((~recovery_audit["recovery_pass"]).sum()),
        "index_weight_rows": int(len(weights)),
        "fund_daily_rows": int(len(fund_daily)),
        "fund_share_rows": int(len(fund_share)),
        "fuzzy_mapping_allowed": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({"snapshot_root": str(output), **manifest}, ensure_ascii=False, indent=2))


def query_index_weights_complete(
    provider: TushareProvider,
    index_code: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    frame = query_with_retry(
        provider,
        "index_weight",
        index_code=index_code,
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
    )
    if len(frame) < 5_000:
        return frame
    parts = []
    for year in range(start_date.year, end_date.year + 1):
        left = max(start_date, pd.Timestamp(f"{year}-01-01"))
        right = min(end_date, pd.Timestamp(f"{year}-12-31"))
        part = query_with_retry(
            provider,
            "index_weight",
            index_code=index_code,
            start_date=left.strftime("%Y%m%d"),
            end_date=right.strftime("%Y%m%d"),
        )
        if len(part) >= 5_000:
            raise RuntimeError(f"index_weight result still truncated for {index_code} {year}")
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the R7 strict PIT ETF rotation snapshot")
    parser.add_argument(
        "--config", default="configs/research/index_backed_r4_strict_pit_v1.yaml",
    )
    parser.add_argument("--output-root", default="data/index_backed_r4_strict_pit")
    return parser.parse_args()


if __name__ == "__main__":
    main()
