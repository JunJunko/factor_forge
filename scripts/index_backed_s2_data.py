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
    filter_mapping_by_weight_coverage,
    select_exact_index_etf_mapping,
)


def query_with_retry(provider: TushareProvider, endpoint: str, **kwargs) -> pd.DataFrame:
    for attempt in range(6):
        try:
            return provider.query(endpoint, **kwargs)
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 16))
    raise RuntimeError("unreachable")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    provider = TushareProvider()
    calendar = query_with_retry(
        provider, "trade_cal", exchange="SSE", start_date=config["history_start"].replace("-", ""),
        end_date=config["history_end"].replace("-", ""), fields="cal_date,is_open",
    )
    dates = sorted(calendar.loc[calendar["is_open"].astype(int).eq(1), "cal_date"].astype(str))
    recent_dates = dates[-int(data_cfg["recent_selection_sessions"]):]

    print("fetching official ETF metadata and recent liquidity window", flush=True)
    etf_basic = query_with_retry(provider, "etf_basic")
    recent_daily = pd.concat([
        query_with_retry(provider, "fund_daily", trade_date=date) for date in recent_dates
    ], ignore_index=True).drop_duplicates(["ts_code", "trade_date"], keep="last")
    recent_share = pd.concat([
        query_with_retry(provider, "fund_share", trade_date=date) for date in recent_dates
    ], ignore_index=True).drop_duplicates(["ts_code", "trade_date"], keep="last")
    mapping, candidate_audit = select_exact_index_etf_mapping(
        etf_basic, recent_daily, recent_share,
        selection_cutoff=config["selection_cutoff"],
        minimum_adv_cny=float(data_cfg["minimum_adv_cny"]),
        minimum_aum_cny=float(data_cfg["minimum_aum_cny"]),
        minimum_observations=max(1, len(recent_dates) - 2),
        allowed_index_suffixes=data_cfg["allowed_index_suffixes"],
    )
    print(f"liquid exact-index candidates={len(mapping)}", flush=True)

    weight_parts = []
    index_codes = sorted(mapping["concept_code"].unique())
    years = range(pd.Timestamp(config["history_start"]).year, pd.Timestamp(config["history_end"]).year + 1)
    for position, code in enumerate(index_codes, start=1):
        parts = []
        for year in years:
            start = max(pd.Timestamp(config["history_start"]), pd.Timestamp(f"{year}-01-01"))
            end = min(pd.Timestamp(config["history_end"]), pd.Timestamp(f"{year}-12-31"))
            if start > end:
                continue
            part = query_with_retry(
                provider, "index_weight", index_code=code,
                start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"),
            )
            if not part.empty:
                parts.append(part)
        if parts:
            weight_parts.append(pd.concat(parts, ignore_index=True))
        if position == 1 or position % 10 == 0 or position == len(index_codes):
            print(f"index weights {position}/{len(index_codes)} code={code}", flush=True)
    index_weights = pd.concat(weight_parts, ignore_index=True) if weight_parts else pd.DataFrame(
        columns=["index_code", "con_code", "trade_date", "weight"]
    )
    index_weights = index_weights.drop_duplicates(
        ["index_code", "trade_date", "con_code"], keep="last",
    )
    mapping, weight_audit = filter_mapping_by_weight_coverage(
        mapping, index_weights,
        minimum_weight_months=int(data_cfg["minimum_weight_months"]),
        minimum_members=int(data_cfg["minimum_members"]),
    )
    index_weights = index_weights.loc[
        index_weights["index_code"].isin(mapping["concept_code"])
    ].copy()
    if mapping.empty:
        raise RuntimeError("no exact ETF-index mapping passed the historical weight gate")
    print(f"weight-qualified exact mappings={len(mapping)}", flush=True)

    daily_parts, share_parts = [], []
    for position, code in enumerate(sorted(mapping["etf_code"].unique()), start=1):
        daily_parts.append(query_with_retry(
            provider, "fund_daily", ts_code=code,
            start_date=config["history_start"].replace("-", ""),
            end_date=config["history_end"].replace("-", ""),
        ))
        share_parts.append(query_with_retry(
            provider, "fund_share", ts_code=code,
            start_date=config["history_start"].replace("-", ""),
            end_date=config["history_end"].replace("-", ""),
        ))
        if position == 1 or position % 10 == 0 or position == len(mapping):
            print(f"ETF histories {position}/{len(mapping)} code={code}", flush=True)
    fund_daily = pd.concat(daily_parts, ignore_index=True).drop_duplicates(
        ["ts_code", "trade_date"], keep="last",
    )
    fund_share = pd.concat(share_parts, ignore_index=True).drop_duplicates(
        ["ts_code", "trade_date"], keep="last",
    )

    digest_source = yaml.safe_dump(config, allow_unicode=True, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(digest_source).hexdigest()[:10]
    output = Path(args.output_root) / (
        f"index_backed_{config['history_start'].replace('-', '')}_"
        f"{config['history_end'].replace('-', '')}_{digest}"
    )
    if output.exists():
        raise FileExistsError(f"immutable snapshot already exists: {output}")
    output.mkdir(parents=True)
    etf_basic.to_parquet(output / "etf_basic.parquet", index=False)
    recent_daily.to_parquet(output / "recent_fund_daily.parquet", index=False)
    recent_share.to_parquet(output / "recent_fund_share.parquet", index=False)
    candidate_audit.to_parquet(output / "candidate_audit.parquet", index=False)
    weight_audit.to_parquet(output / "weight_coverage_audit.parquet", index=False)
    mapping.to_parquet(output / "exact_index_etf_mapping.parquet", index=False)
    mapping.to_csv(output / "exact_index_etf_mapping.csv", index=False, encoding="utf-8-sig")
    index_weights.to_parquet(output / "index_weights.parquet", index=False)
    fund_daily.to_parquet(output / "fund_daily.parquet", index=False)
    fund_share.to_parquet(output / "fund_share.parquet", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "IMMUTABLE_DATA_SNAPSHOT",
        "source": "Tushare etf_basic/fund_daily/fund_share/index_weight",
        "config": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "history_start": config["history_start"],
        "history_end": config["history_end"],
        "selection_cutoff": config["selection_cutoff"],
        "trading_days": len(dates),
        "liquid_exact_index_candidates": int(len(weight_audit)),
        "selected_exact_mappings": int(len(mapping)),
        "clusters": int(mapping["cluster"].nunique()),
        "weight_months_min": int(mapping["weight_months"].min()),
        "weight_months_max": int(mapping["weight_months"].max()),
        "rows": {
            "index_weights": int(len(index_weights)),
            "fund_daily": int(len(fund_daily)),
            "fund_share": int(len(fund_share)),
        },
        "causality": "monthly index membership is consumed with a one-session lag downstream",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({"snapshot_root": str(output), **manifest}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build exact ETF-index data for frozen S2")
    parser.add_argument("--config", default="configs/research/index_backed_s2_forward_v1.yaml")
    parser.add_argument("--output-root", default="data/index_backed_s2")
    return parser.parse_args()


if __name__ == "__main__":
    main()
