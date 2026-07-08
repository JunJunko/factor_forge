from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sell_impact_sorting_repair as base


SOURCE_RUN = Path("artifacts/strategy_reviews/sell_impact_ranker_timing_compare_20260708T120449Z")
OUTPUT_ROOT = Path("artifacts/strategy_reviews")
SELECTION = "ranker_direct_top"
TIMING = "timing_target_position"
TOP_N = 5


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_ranker_trade_audit_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    version, panel = base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    trades = pd.read_parquet(SOURCE_RUN / "ranker_timing_trades.parquet")
    daily = pd.read_parquet(SOURCE_RUN / "ranker_timing_daily.parquet")
    metrics = pd.read_csv(SOURCE_RUN / "ranker_timing_backtest_metrics.csv")

    trades = trades.loc[
        trades["selection"].eq(SELECTION)
        & trades["timing"].eq(TIMING)
        & trades["top_n"].eq(TOP_N)
    ].copy()
    daily = daily.loc[
        daily["selection"].eq(SELECTION)
        & daily["timing"].eq(TIMING)
        & daily["top_n"].eq(TOP_N)
    ].copy()
    metric_rows = metrics.loc[
        metrics["selection"].eq(SELECTION)
        & metrics["timing"].eq(TIMING)
        & metrics["top_n"].eq(TOP_N)
    ].copy()
    if trades.empty or daily.empty:
        raise ValueError("selected audit target has no trades or daily rows")

    audit = build_trade_audit(panel, trades)
    yearly = yearly_audit(daily, trades, metric_rows)
    concentration = concentration_audit(daily, trades)
    summary = summary_payload(audit, yearly, concentration, metric_rows, version)

    audit.to_csv(output / "trade_execution_audit.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_trade_audit.csv", index=False, encoding="utf-8-sig")
    concentration.to_csv(output / "concentration_audit.csv", index=False, encoding="utf-8-sig")
    metric_rows.to_csv(output / "audited_backtest_metrics.csv", index=False, encoding="utf-8-sig")
    (output / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(output, summary, yearly, concentration)
    print(f"done -> {output}")


def build_trade_audit(panel: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    flags = panel[
        [
            "trade_date",
            "ts_code",
            "raw_open",
            "is_suspended",
            "is_limit_up_open",
            "is_limit_down_open",
            "is_st",
            "is_delisting_period",
            "listing_trade_days",
        ]
    ].copy()
    t = trades.copy()
    for col in ["trade_date", "signal_date", "entry_date", "due_date"]:
        if col in t:
            t[col] = pd.to_datetime(t[col])
    t = t.merge(flags, on=["trade_date", "ts_code"], how="left", suffixes=("", "_panel"))
    dates = pd.Index(pd.to_datetime(panel["trade_date"].drop_duplicates()).sort_values())
    date_rank = {date: i for i, date in enumerate(dates)}
    t["signal_to_trade_days"] = [
        date_rank.get(pd.Timestamp(trade_date), np.nan) - date_rank.get(pd.Timestamp(signal_date), np.nan)
        if pd.notna(signal_date)
        else np.nan
        for trade_date, signal_date in zip(t["trade_date"], t["signal_date"])
    ]
    t["holding_trade_days_at_sell"] = [
        date_rank.get(pd.Timestamp(trade_date), np.nan) - date_rank.get(pd.Timestamp(entry_date), np.nan)
        if side == "SELL" and pd.notna(entry_date)
        else np.nan
        for side, trade_date, entry_date in zip(t["side"], t["trade_date"], t["entry_date"])
    ]
    is_buy = t["side"].eq("BUY")
    is_sell = t["side"].eq("SELL")
    t["audit_issue"] = ""
    add_issue(t, is_buy & t["signal_to_trade_days"].ne(1), "buy_not_t_plus_1")
    add_issue(t, is_buy & t["is_suspended"].fillna(True).astype(bool), "buy_suspended")
    add_issue(t, is_buy & t["is_limit_up_open"].fillna(False).astype(bool), "buy_limit_up_open")
    add_issue(t, is_buy & t["is_st"].fillna(False).astype(bool), "buy_st")
    add_issue(t, is_buy & t["is_delisting_period"].fillna(False).astype(bool), "buy_delisting_period")
    add_issue(t, is_buy & t["listing_trade_days"].lt(60), "buy_listing_lt_60")
    add_issue(t, is_buy & ~np.isfinite(pd.to_numeric(t["raw_open"], errors="coerce")), "buy_missing_open")
    add_issue(t, is_sell & t["is_suspended"].fillna(True).astype(bool), "sell_suspended")
    add_issue(t, is_sell & t["is_limit_down_open"].fillna(False).astype(bool), "sell_limit_down_open")
    add_issue(t, is_sell & t["holding_trade_days_at_sell"].lt(base.HOLDING_DAYS), "sell_before_holding_days")
    t["audit_pass"] = t["audit_issue"].eq("")
    return t.sort_values(["fold", "trade_date", "side", "ts_code"]).reset_index(drop=True)


def add_issue(frame: pd.DataFrame, mask: pd.Series, issue: str) -> None:
    idx = mask.fillna(False)
    frame.loc[idx, "audit_issue"] = frame.loc[idx, "audit_issue"].apply(
        lambda old: issue if not old else f"{old};{issue}"
    )


def yearly_audit(daily: pd.DataFrame, trades: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, d in daily.groupby("fold"):
        t = trades.loc[trades["fold"].eq(fold)]
        buys = t.loc[t["side"].eq("BUY")]
        sells = t.loc[t["side"].eq("SELL")]
        m = metrics.loc[metrics["fold"].eq(fold)].iloc[0].to_dict() if not metrics.loc[metrics["fold"].eq(fold)].empty else {}
        rows.append(
            {
                "fold": fold,
                "start": pd.to_datetime(d["trade_date"]).min(),
                "end": pd.to_datetime(d["trade_date"]).max(),
                "annualized_return": m.get("annualized_return"),
                "csi1000_annualized_return": m.get("csi1000_annualized_return"),
                "annualized_excess_return_vs_csi1000": m.get("annualized_excess_return_vs_csi1000"),
                "max_drawdown": m.get("max_drawdown"),
                "sharpe": m.get("sharpe"),
                "generated_signals": m.get("generated_signals"),
                "executed_buys": m.get("executed_buys"),
                "execution_rate": m.get("execution_rate"),
                "buy_trades": int(len(buys)),
                "sell_trades": int(len(sells)),
                "unique_buy_stocks": int(buys["ts_code"].nunique()),
                "avg_daily_cash_ratio": float(d["cash_ratio"].mean()),
                "avg_daily_holding_count": float(d["holding_count"].mean()),
                "avg_daily_unique_holding_count": float(d["unique_holding_count"].mean()),
                "duplicate_signal_rate": safe_div(float(d["duplicate_signals"].sum()), float(d["new_signals"].sum())),
                "avg_daily_turnover": float(d["portfolio_turnover"].mean()),
            }
        )
    return pd.DataFrame(rows)


def concentration_audit(daily: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, t in trades.groupby("fold"):
        buys = t.loc[t["side"].eq("BUY")]
        d = daily.loc[daily["fold"].eq(fold)].copy()
        by_stock = buys.groupby("ts_code")["gross_value"].sum().sort_values(ascending=False)
        d["month"] = pd.to_datetime(d["trade_date"]).dt.to_period("M").astype(str)
        monthly = d.groupby("month")["return"].apply(lambda s: float((1.0 + s).prod() - 1.0)).sort_values(ascending=False)
        positive = monthly[monthly > 0]
        rows.append(
            {
                "fold": fold,
                "top1_stock_buy_share": safe_div(float(by_stock.head(1).sum()), float(by_stock.sum())),
                "top5_stock_buy_share": safe_div(float(by_stock.head(5).sum()), float(by_stock.sum())),
                "top10_stock_buy_share": safe_div(float(by_stock.head(10).sum()), float(by_stock.sum())),
                "largest_month": monthly.index[0] if len(monthly) else None,
                "largest_month_return": float(monthly.iloc[0]) if len(monthly) else np.nan,
                "top_month_positive_return_share": (
                    safe_div(float(positive.iloc[0]), float(positive.sum())) if len(positive) else np.nan
                ),
                "positive_months": int((monthly > 0).sum()) if len(monthly) else 0,
                "negative_months": int((monthly < 0).sum()) if len(monthly) else 0,
                "top_stock_list": ",".join(by_stock.head(5).index.astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows)


def summary_payload(
    audit: pd.DataFrame,
    yearly: pd.DataFrame,
    concentration: pd.DataFrame,
    metrics: pd.DataFrame,
    data_version: str,
) -> dict:
    issue_rows = audit.loc[~audit["audit_pass"]]
    buy_issues = issue_rows.loc[issue_rows["side"].eq("BUY")]
    sell_issues = issue_rows.loc[issue_rows["side"].eq("SELL")]
    return {
        "data_version": data_version,
        "source_run": str(SOURCE_RUN),
        "selection": SELECTION,
        "timing": TIMING,
        "top_n": TOP_N,
        "trade_rows": int(len(audit)),
        "buy_rows": int(audit["side"].eq("BUY").sum()),
        "sell_rows": int(audit["side"].eq("SELL").sum()),
        "audit_pass_rows": int(audit["audit_pass"].sum()),
        "audit_issue_rows": int(len(issue_rows)),
        "buy_issue_rows": int(len(buy_issues)),
        "sell_issue_rows": int(len(sell_issues)),
        "issue_counts": issue_counts(issue_rows),
        "mean_annualized_return": float(metrics["annualized_return"].mean()),
        "mean_excess_vs_csi1000": float(metrics["annualized_excess_return_vs_csi1000"].mean()),
        "worst_max_drawdown": float(metrics["max_drawdown"].min()),
        "yearly": yearly.replace({np.nan: None}).to_dict("records"),
        "concentration": concentration.replace({np.nan: None}).to_dict("records"),
    }


def issue_counts(issue_rows: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in issue_rows.get("audit_issue", pd.Series(dtype=str)).dropna():
        for part in str(item).split(";"):
            if part:
                counts[part] = counts.get(part, 0) + 1
    return counts


def write_report(output: Path, summary: dict, yearly: pd.DataFrame, concentration: pd.DataFrame) -> None:
    lines = [
        "# Sell Impact Ranker Trade Audit",
        "",
        "## Scope",
        f"- Source run: `{SOURCE_RUN}`",
        f"- Strategy: `{SELECTION} + {TIMING} + Top{TOP_N} + hold10 + 20bps`",
        "",
        "## Executive Summary",
        f"- Trade rows: `{summary['trade_rows']}`; buys: `{summary['buy_rows']}`; sells: `{summary['sell_rows']}`",
        f"- Audit issue rows: `{summary['audit_issue_rows']}`; buy issues: `{summary['buy_issue_rows']}`; sell issues: `{summary['sell_issue_rows']}`",
        f"- Mean annualized return: `{summary['mean_annualized_return']:.2%}`",
        f"- Mean excess vs CSI1000: `{summary['mean_excess_vs_csi1000']:.2%}`",
        f"- Worst MDD: `{summary['worst_max_drawdown']:.2%}`",
        f"- Issue counts: `{summary['issue_counts']}`",
        "",
        "## Yearly Execution",
        yearly.round(6).to_markdown(index=False),
        "",
        "## Concentration",
        concentration.round(6).to_markdown(index=False),
        "",
        "## Interpretation",
        "- `audit_issue_rows=0` is the desired result for the mechanical execution checks.",
        "- Concentration is still a research risk even when execution audit passes; watch top-month contribution and top-stock buy share.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else np.nan


def json_default(obj):
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
