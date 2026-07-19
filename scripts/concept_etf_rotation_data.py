from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from factor_forge.data.tushare_provider import TushareProvider


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.history_end:
        config["history_end"] = args.history_end
    effective_config = yaml.safe_dump(config, allow_unicode=True, sort_keys=True).encode("utf-8")
    provider = TushareProvider()
    basic = provider.query("fund_basic", market="E", status="L")
    candidate_codes = sorted({code for theme in config["themes"] for code in theme["candidates"]})
    daily_parts, share_parts, nav_parts = [], [], []
    for position, code in enumerate(candidate_codes, start=1):
        print(f"[{position}/{len(candidate_codes)}] {code}", flush=True)
        daily_parts.append(provider.query(
            "fund_daily", ts_code=code, start_date=config["history_start"], end_date=config["history_end"]
        ))
        share_parts.append(provider.query(
            "fund_share", ts_code=code, start_date=config["history_start"], end_date=config["history_end"]
        ))
        nav_parts.append(provider.query(
            "fund_nav", ts_code=code, start_date=config["history_start"], end_date=config["history_end"]
        ))
    daily = pd.concat(daily_parts, ignore_index=True).drop_duplicates(["ts_code", "trade_date"])
    share = pd.concat(share_parts, ignore_index=True).drop_duplicates(["ts_code", "trade_date"])
    nav = pd.concat(nav_parts, ignore_index=True).drop_duplicates(["ts_code", "nav_date"])
    selected, candidate_audit, audit = select_mapping(config, basic, daily, share)

    digest = hashlib.sha256(effective_config).hexdigest()[:8]
    root = Path(args.output_root) / f"tushare_{config['history_start']}_{config['history_end']}_{digest}"
    if root.exists():
        raise FileExistsError(f"immutable data snapshot already exists: {root}")
    root.mkdir(parents=True)
    basic.loc[basic["ts_code"].isin(candidate_codes)].to_parquet(root / "fund_basic.parquet", index=False)
    daily.to_parquet(root / "fund_daily.parquet", index=False)
    share.to_parquet(root / "fund_share.parquet", index=False)
    nav.to_parquet(root / "fund_nav.parquet", index=False)
    selected.to_parquet(root / "concept_etf_mapping.parquet", index=False)
    selected.to_csv(root / "concept_etf_mapping.csv", index=False, encoding="utf-8-sig")
    candidate_audit.to_parquet(root / "candidate_selection_audit.parquet", index=False)
    candidate_audit.to_csv(root / "candidate_selection_audit.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.resolve()),
        "config_file_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "effective_config_sha256": hashlib.sha256(effective_config).hexdigest(),
        "source": "Tushare Pro", "point_in_time_selection_cutoff": config["selection_cutoff"],
        "validation_start": config["validation_start"], "history_end": config["history_end"],
        "candidate_etfs": len(candidate_codes), "selected_etfs": int(selected["selected"].sum()),
        "endpoints": ["fund_basic", "fund_daily", "fund_share", "fund_nav"],
        "row_counts": {
            "fund_basic": int(basic["ts_code"].isin(candidate_codes).sum()),
            "fund_daily": len(daily), "fund_share": len(share), "fund_nav": len(nav),
            "candidate_selection_audit": len(candidate_audit),
        },
        "selection_audit": audit, "excluded_themes": config.get("excluded_themes", []),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"snapshot_root": str(root), "selected": selected.loc[selected["selected"]].to_dict("records")}, ensure_ascii=False, indent=2))


def select_mapping(config: dict, basic: pd.DataFrame, daily: pd.DataFrame, share: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    daily = daily.copy()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"].astype(str))
    share = share.copy()
    share["trade_date"] = pd.to_datetime(share["trade_date"].astype(str))
    cutoff = pd.Timestamp(config["selection_cutoff"])
    pre = daily.loc[daily["trade_date"].le(cutoff)].sort_values(["ts_code", "trade_date"])
    pre = pre.groupby("ts_code", observed=True).tail(int(config["selection_window_days"]))
    stats = pre.groupby("ts_code", observed=True).agg(
        observations=("trade_date", "nunique"), adv_cny=("amount", lambda x: pd.to_numeric(x).mean() * 1000),
        last_close=("close", "last"), last_trade_date=("trade_date", "max"),
    ).reset_index()
    latest_share = share.loc[share["trade_date"].le(cutoff)].sort_values("trade_date").groupby("ts_code", observed=True).tail(1)
    latest_share = latest_share[["ts_code", "fd_share"]]
    stats = stats.merge(latest_share, on="ts_code", how="left")
    stats["aum_cny"] = pd.to_numeric(stats["last_close"]) * pd.to_numeric(stats["fd_share"]) * 10000
    fields = [c for c in ["ts_code", "name", "list_date", "benchmark", "m_fee", "c_fee"] if c in basic]
    stats = stats.merge(basic[fields].drop_duplicates("ts_code"), on="ts_code", how="left")
    rows, candidate_rows = [], []
    for theme in config["themes"]:
        candidates = stats.loc[stats["ts_code"].isin(theme["candidates"])].copy()
        candidates["eligible"] = (
            candidates["observations"].ge(int(config["selection_window_days"]) * 0.9)
            & candidates["adv_cny"].ge(float(config["minimum_adv_cny"]))
            & candidates["aum_cny"].ge(float(config["minimum_aum_cny"]))
            & pd.to_datetime(candidates["list_date"], errors="coerce").le(cutoff)
        )
        for candidate in candidates.itertuples():
            reasons = []
            if candidate.observations < int(config["selection_window_days"]) * 0.9:
                reasons.append("insufficient observations")
            if candidate.adv_cny < float(config["minimum_adv_cny"]):
                reasons.append("ADV below gate")
            if not pd.notna(candidate.aum_cny) or candidate.aum_cny < float(config["minimum_aum_cny"]):
                reasons.append("AUM below gate")
            if pd.to_datetime(candidate.list_date, errors="coerce") > cutoff:
                reasons.append("listed after cutoff")
            candidate_rows.append({
                **{k: theme[k] for k in ("concept_code", "concept_name", "cluster", "match_type")},
                "etf_code": candidate.ts_code, "etf_name": candidate.name,
                "benchmark": candidate.benchmark, "observations": candidate.observations,
                "adv_cny": candidate.adv_cny, "aum_cny": candidate.aum_cny,
                "last_trade_date": candidate.last_trade_date, "eligible": bool(candidate.eligible),
                "gate_reason": "PASS" if candidate.eligible else "; ".join(reasons),
            })
        candidates = candidates.sort_values(["eligible", "adv_cny"], ascending=[False, False])
        chosen = candidates.iloc[0] if not candidates.empty and bool(candidates.iloc[0]["eligible"]) else None
        rows.append({
            **{k: theme[k] for k in ("concept_code", "concept_name", "cluster", "match_type")},
            "etf_code": None if chosen is None else chosen["ts_code"],
            "etf_name": None if chosen is None else chosen.get("name"),
            "benchmark": None if chosen is None else chosen.get("benchmark"),
            "adv_cny": None if chosen is None else float(chosen["adv_cny"]),
            "aum_cny": None if chosen is None else float(chosen["aum_cny"]),
            "selected": chosen is not None,
            "selection_reason": "highest pre-cutoff ADV among size/liquidity eligible candidates" if chosen is not None else "no candidate passed pre-cutoff gates",
        })
    result = pd.DataFrame(rows)
    audit = {
        "minimum_adv_cny": float(config["minimum_adv_cny"]), "minimum_aum_cny": float(config["minimum_aum_cny"]),
        "selection_window_days": int(config["selection_window_days"]),
        "themes_requested": len(config["themes"]), "themes_selected": int(result["selected"].sum()),
    }
    return result, pd.DataFrame(candidate_rows), audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze point-in-time ETF data for concept rotation")
    parser.add_argument("--config", default="configs/research/concept_etf_rotation_v1.yaml")
    parser.add_argument("--history-end", help="override history_end without modifying the frozen selection config")
    parser.add_argument("--output-root", default="data/concept_etf_rotation")
    return parser.parse_args()


if __name__ == "__main__":
    main()
