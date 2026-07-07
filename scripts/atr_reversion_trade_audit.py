"""Audit ATR backtest trade files for common leakage / execution issues."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository


def main(run_dir: str) -> None:
    run = Path(run_dir)
    version, panel = _load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    trade_files = sorted(run.rglob("trades_*.parquet"))
    daily_files = sorted(run.rglob("daily_*.parquet"))
    rows = []
    for path in trade_files:
        trades = pd.read_parquet(path)
        if trades.empty:
            continue
        trades["trade_date"] = pd.to_datetime(trades["trade_date"])
        joined = trades.merge(
            panel[[
                "trade_date", "ts_code", "raw_open", "is_suspended", "is_limit_up_open",
                "is_limit_down_open", "is_st", "is_delisting_period", "listing_trade_days",
                "is_tradeable",
            ]],
            on=["trade_date", "ts_code"],
            how="left",
        )
        buys = joined["side"].eq("BUY")
        sells = joined["side"].isin(["SELL", "EARLY_SELL", "RISK_SELL"])
        rows.append({
            "file": str(path.relative_to(run)),
            "trades": int(len(joined)),
            "buys": int(buys.sum()),
            "sells": int(sells.sum()),
            "buy_missing_open": int((buys & joined["raw_open"].isna()).sum()),
            "buy_suspended": int((buys & joined["is_suspended"].fillna(True)).sum()),
            "buy_limit_up_open": int((buys & joined["is_limit_up_open"].fillna(False)).sum()),
            "buy_st": int((buys & joined["is_st"].fillna(True)).sum()),
            "buy_delisting": int((buys & joined["is_delisting_period"].fillna(True)).sum()),
            "buy_listing_lt60": int((buys & joined["listing_trade_days"].lt(60).fillna(True)).sum()),
            "sell_missing_open": int((sells & joined["raw_open"].isna()).sum()),
            "sell_suspended": int((sells & joined["is_suspended"].fillna(True)).sum()),
            "sell_limit_down_open": int((sells & joined["is_limit_down_open"].fillna(False)).sum()),
            "sell_st": int((sells & joined["is_st"].fillna(False)).sum()),
            "sell_delisting": int((sells & joined["is_delisting_period"].fillna(False)).sum()),
            "total_cost": float(joined.get("cost", pd.Series(0, index=joined.index)).sum()),
            "total_value": float(joined.get("value", pd.Series(0, index=joined.index)).sum()),
        })
    audit = pd.DataFrame(rows)
    if not audit.empty:
        audit["effective_cost_bps"] = audit["total_cost"] / audit["total_value"].replace(0, pd.NA) * 10_000
    audit.to_csv(run / "trade_execution_audit.csv", index=False, encoding="utf-8-sig")

    daily_rows = []
    for path in daily_files:
        daily = pd.read_parquet(path)
        if daily.empty:
            continue
        daily_rows.append({
            "file": str(path.relative_to(run)),
            "days": int(len(daily)),
            "nav_missing": int(daily["nav"].isna().sum()) if "nav" in daily else None,
            "holding_when_cash_all": int(((daily.get("holding_count", 0) > 0) & (daily.get("cash_ratio", 0) >= 0.999)).sum()),
            "negative_nav": int((daily["nav"] <= 0).sum()) if "nav" in daily else None,
            "max_turnover": float(daily.get("turnover", pd.Series(dtype=float)).max()),
            "avg_turnover": float(daily.get("turnover", pd.Series(dtype=float)).mean()),
        })
    daily_audit = pd.DataFrame(daily_rows)
    daily_audit.to_csv(run / "daily_nav_audit.csv", index=False, encoding="utf-8-sig")

    summary = {
        "data_version": version,
        "run_dir": str(run),
        "trade_files": len(trade_files),
        "daily_files": len(daily_files),
        "blocking_trade_issues": int(audit[[
            "buy_missing_open", "buy_suspended", "buy_limit_up_open", "buy_st",
            "buy_delisting", "buy_listing_lt60", "sell_missing_open", "sell_suspended",
            "sell_limit_down_open",
        ]].sum().sum()) if not audit.empty else 0,
    }
    (run / "trade_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _load_panel():
    project = load_project("configs/project.yaml")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest("latest")
    _, panel = repo.load_panel(version)
    return version, panel


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/atr_reversion_trade_audit.py RUN_DIR")
    main(sys.argv[1])
