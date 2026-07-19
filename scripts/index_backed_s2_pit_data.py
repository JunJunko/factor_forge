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
    official_theme_etf_candidates,
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
    provider = TushareProvider()
    print("fetching listed and delisted official ETF metadata", flush=True)
    basic_parts = [
        query_with_retry(provider, "etf_basic", list_status=status) for status in ("L", "D")
    ]
    etf_basic = pd.concat(basic_parts, ignore_index=True).drop_duplicates("ts_code", keep="first")
    candidates = official_theme_etf_candidates(
        etf_basic, as_of=config["history_end"],
        allowed_index_suffixes=config["data"]["allowed_index_suffixes"],
    )
    print(
        f"unconditional theme candidates ETFs={len(candidates)} "
        f"indexes={candidates['index_code'].nunique()}", flush=True,
    )

    seed_weights = load_seed_weights(Path(args.seed_snapshot))
    weight_parts = []
    years = range(pd.Timestamp(config["history_start"]).year, pd.Timestamp(config["history_end"]).year + 1)
    index_codes = sorted(candidates["index_code"].unique())
    for position, code in enumerate(index_codes, start=1):
        seeded = seed_weights.loc[seed_weights["index_code"].eq(code)]
        if not seeded.empty:
            weight_parts.append(seeded)
        else:
            parts = []
            for year in years:
                start = max(pd.Timestamp(config["history_start"]), pd.Timestamp(f"{year}-01-01"))
                end = min(pd.Timestamp(config["history_end"]), pd.Timestamp(f"{year}-12-31"))
                part = query_with_retry(
                    provider, "index_weight", index_code=code,
                    start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"),
                )
                if not part.empty:
                    parts.append(part)
            if parts:
                weight_parts.append(pd.concat(parts, ignore_index=True))
        if position == 1 or position % 20 == 0 or position == len(index_codes):
            print(f"index weights {position}/{len(index_codes)} code={code}", flush=True)
    weights = pd.concat(weight_parts, ignore_index=True) if weight_parts else pd.DataFrame(
        columns=["index_code", "con_code", "trade_date", "weight"]
    )
    weights = weights.drop_duplicates(["index_code", "trade_date", "con_code"], keep="last")
    index_mapping = candidates.drop_duplicates("index_code").rename(columns={
        "index_code": "concept_code", "index_name": "concept_name",
    })[["concept_code", "concept_name", "cluster"]]
    qualified, weight_audit = filter_mapping_by_weight_coverage(
        index_mapping, weights,
        minimum_weight_months=int(config["data"]["minimum_weight_months"]),
        minimum_members=int(config["data"]["minimum_members"]),
    )
    candidates = candidates.loc[candidates["index_code"].isin(qualified["concept_code"])].copy()
    weights = weights.loc[weights["index_code"].isin(qualified["concept_code"])].copy()
    if candidates.empty:
        raise RuntimeError("no PIT ETF candidates passed index-weight history gates")
    print(
        f"weight-qualified ETFs={len(candidates)} indexes={candidates['index_code'].nunique()}",
        flush=True,
    )

    daily_parts, share_parts = [], []
    codes = sorted(candidates["ts_code"].unique())
    for position, code in enumerate(codes, start=1):
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
        if position == 1 or position % 20 == 0 or position == len(codes):
            print(f"ETF histories {position}/{len(codes)} code={code}", flush=True)
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
        f"pit_index_backed_{config['history_start'].replace('-', '')}_"
        f"{config['history_end'].replace('-', '')}_{digest}"
    )
    if output.exists():
        raise FileExistsError(f"immutable PIT snapshot already exists: {output}")
    output.mkdir(parents=True)
    etf_basic.to_parquet(output / "etf_basic_all_statuses.parquet", index=False)
    candidates.to_parquet(output / "theme_etf_candidates.parquet", index=False)
    candidates.to_csv(output / "theme_etf_candidates.csv", index=False, encoding="utf-8-sig")
    weight_audit.to_parquet(output / "index_weight_coverage_audit.parquet", index=False)
    weights.to_parquet(output / "index_weights.parquet", index=False)
    fund_daily.to_parquet(output / "fund_daily.parquet", index=False)
    fund_share.to_parquet(output / "fund_share.parquet", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "IMMUTABLE_PIT_CANDIDATE_SNAPSHOT",
        "config": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "history_start": config["history_start"], "history_end": config["history_end"],
        "candidate_etfs": int(candidates["ts_code"].nunique()),
        "candidate_indexes": int(candidates["index_code"].nunique()),
        "listed_candidates": int(candidates["list_status"].eq("L").sum()),
        "delisted_candidates": int(candidates["list_status"].eq("D").sum()),
        "index_weight_rows": int(len(weights)),
        "fund_daily_rows": int(len(fund_daily)),
        "fund_share_rows": int(len(fund_share)),
        "important": "Index eligibility is independent of 2026 liquidity; ETF choice is deferred to each prior month end.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({"snapshot_root": str(output), **manifest}, ensure_ascii=False, indent=2))


def load_seed_weights(snapshot: Path) -> pd.DataFrame:
    path = snapshot / "index_weights.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(
        columns=["index_code", "con_code", "trade_date", "weight"]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build all-history ETF candidates for PIT rotation")
    parser.add_argument("--config", default="configs/research/index_backed_s2_pit_v1.yaml")
    parser.add_argument(
        "--seed-snapshot",
        default="data/index_backed_s2/index_backed_20230101_20260717_0537643008",
    )
    parser.add_argument("--output-root", default="data/index_backed_s2_pit")
    return parser.parse_args()


if __name__ == "__main__":
    main()
