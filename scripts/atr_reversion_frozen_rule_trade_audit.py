"""Concentration audit for the frozen fit-quality flip rule."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from atr_reversion_small_portfolio_backtest import _json_default


DEFAULT_RUN = (
    "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/"
    "fit_quality_sensitivity_20260707T031019Z"
)
LOOKBACK = 40
MIN_OBS = 15
POLICY = "fit_quality_flip_only"
TOP_N = 5
COST = 20


def main(run_dir: str = DEFAULT_RUN) -> None:
    run = Path(run_dir)
    out = run / f"frozen_rule_audit_lookback{LOOKBACK}_minobs{MIN_OBS}_cost{COST}"
    out.mkdir(parents=True, exist_ok=True)
    trades = _load_trades(run)
    daily = _load_daily(run)
    roundtrips = _roundtrips(trades)
    cycle = _cycle_returns(daily)
    stock = _stock_summary(roundtrips)
    month = _month_summary(roundtrips, daily)
    year = _year_summary(roundtrips, daily, cycle)
    concentration = _concentration(roundtrips)

    trades.to_csv(out / "frozen_rule_trades.csv", index=False, encoding="utf-8-sig")
    roundtrips.to_csv(out / "frozen_rule_roundtrips.csv", index=False, encoding="utf-8-sig")
    cycle.to_csv(out / "frozen_rule_cycles.csv", index=False, encoding="utf-8-sig")
    stock.to_csv(out / "frozen_rule_stock_concentration.csv", index=False, encoding="utf-8-sig")
    month.to_csv(out / "frozen_rule_month_summary.csv", index=False, encoding="utf-8-sig")
    year.to_csv(out / "frozen_rule_year_summary.csv", index=False, encoding="utf-8-sig")
    concentration.to_csv(out / "frozen_rule_concentration.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(run),
                "audit_dir": str(out),
                "lookback": LOOKBACK,
                "min_obs": MIN_OBS,
                "policy": POLICY,
                "top_n": TOP_N,
                "cost_bps": COST,
                "year_summary": year.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (out / "report.md").write_text(_report(year, month, stock, concentration, cycle), encoding="utf-8")
    print(f"audit_dir={out}")


def _load_trades(run: Path) -> pd.DataFrame:
    pattern = f"trades_lookback{LOOKBACK}_minobs{MIN_OBS}_*_fit_quality_flip_only_top{TOP_N}_cost{COST}.parquet"
    rows = []
    for path in sorted(run.glob(pattern)):
        fold = _fold_from_name(path.name)
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["fold"] = fold
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _load_daily(run: Path) -> pd.DataFrame:
    pattern = f"daily_lookback{LOOKBACK}_minobs{MIN_OBS}_*_fit_quality_flip_only_top{TOP_N}_cost{COST}.parquet"
    rows = []
    for path in sorted(run.glob(pattern)):
        fold = _fold_from_name(path.name)
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["fold"] = fold
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _fold_from_name(name: str) -> str:
    match = re.search(r"_(test_\d{4}(?:h1)?)_", name)
    if not match:
        raise ValueError(f"cannot parse fold from {name}")
    return match.group(1)


def _roundtrips(trades: pd.DataFrame) -> pd.DataFrame:
    buys: dict[tuple[str, str], list[dict]] = {}
    rows = []
    for row in trades.sort_values(["fold", "trade_date", "side"]).to_dict("records"):
        key = (row["fold"], row["ts_code"])
        if row["side"] == "BUY":
            buys.setdefault(key, []).append(row)
        elif row["side"] == "SELL" and buys.get(key):
            buy = buys[key].pop(0)
            buy_cash = float(buy["value"]) + float(buy.get("cost", 0.0))
            sell_cash = float(row["value"]) - float(row.get("cost", 0.0))
            pnl = sell_cash - buy_cash
            rows.append(
                {
                    "fold": row["fold"],
                    "ts_code": row["ts_code"],
                    "buy_date": buy["trade_date"],
                    "sell_date": row["trade_date"],
                    "year": int(pd.Timestamp(row["trade_date"]).year),
                    "month": pd.Timestamp(row["trade_date"]).strftime("%Y-%m"),
                    "buy_value": float(buy["value"]),
                    "sell_value": float(row["value"]),
                    "pnl": pnl,
                    "return": pnl / buy_cash if buy_cash else np.nan,
                    "buy_exposure": float(buy.get("exposure", np.nan)),
                    "sell_exposure": float(row.get("exposure", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def _cycle_returns(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, d in daily.sort_values(["fold", "trade_date"]).groupby("fold"):
        dates = d["trade_date"].tolist()
        for idx in range(0, len(dates) - 1, 10):
            segment = d.iloc[idx + 1 : min(idx + 11, len(d))].copy()
            if segment.empty:
                continue
            ret = float((1.0 + segment["return"].fillna(0.0)).prod() - 1.0)
            bench = float((1.0 + segment["benchmark_return"].fillna(0.0)).prod() - 1.0)
            rows.append(
                {
                    "fold": fold,
                    "year": int(segment["trade_date"].iloc[0].year),
                    "month": segment["trade_date"].iloc[0].strftime("%Y-%m"),
                    "start_date": segment["trade_date"].iloc[0],
                    "end_date": segment["trade_date"].iloc[-1],
                    "cycle_return": ret,
                    "cycle_benchmark_return": bench,
                    "cycle_excess_return": ret - bench,
                    "avg_exposure": float(segment["exposure"].mean()),
                    "active": bool(segment["exposure"].mean() > 0.0),
                }
            )
    return pd.DataFrame(rows)


def _stock_summary(roundtrips: pd.DataFrame) -> pd.DataFrame:
    if roundtrips.empty:
        return pd.DataFrame()
    return (
        roundtrips.groupby(["year", "ts_code"])
        .agg(
            trades=("pnl", "size"),
            pnl=("pnl", "sum"),
            avg_return=("return", "mean"),
            win_rate=("return", lambda s: float((s > 0.0).mean())),
        )
        .reset_index()
        .sort_values(["year", "pnl"], ascending=[True, False])
    )


def _month_summary(roundtrips: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    nav_month = []
    for (year, month), g in daily.groupby([daily["trade_date"].dt.year, daily["trade_date"].dt.strftime("%Y-%m")]):
        nav_month.append(
            {
                "year": int(year),
                "month": month,
                "return": float(g["nav"].iloc[-1] / g["nav"].iloc[0] - 1.0),
                "benchmark_return": float((1.0 + g["benchmark_return"]).prod() - 1.0),
                "max_drawdown": float((g["nav"] / g["nav"].cummax() - 1.0).min()),
                "avg_exposure": float(g["exposure"].mean()),
            }
        )
    out = pd.DataFrame(nav_month)
    if not roundtrips.empty:
        rt = roundtrips.groupby(["year", "month"]).agg(
            trade_count=("pnl", "size"),
            pnl=("pnl", "sum"),
            trade_win_rate=("return", lambda s: float((s > 0.0).mean())),
            avg_trade_return=("return", "mean"),
        ).reset_index()
        out = out.merge(rt, on=["year", "month"], how="left")
    return out.sort_values(["year", "month"])


def _year_summary(roundtrips: pd.DataFrame, daily: pd.DataFrame, cycle: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, d in daily.groupby(daily["trade_date"].dt.year):
        rt = roundtrips[roundtrips["year"].eq(year)] if not roundtrips.empty else pd.DataFrame()
        cy = cycle[cycle["year"].eq(year)] if not cycle.empty else pd.DataFrame()
        rows.append(
            {
                "year": int(year),
                "return": float(d["nav"].iloc[-1] / d["nav"].iloc[0] - 1.0),
                "benchmark_return": float((1.0 + d["benchmark_return"]).prod() - 1.0),
                "excess_return": float(d["nav"].iloc[-1] / d["nav"].iloc[0] - 1.0 - (1.0 + d["benchmark_return"]).prod() + 1.0),
                "max_drawdown": float((d["nav"] / d["nav"].cummax() - 1.0).min()),
                "avg_exposure": float(d["exposure"].mean()),
                "roundtrip_trades": int(len(rt)),
                "trade_win_rate": float((rt["return"] > 0.0).mean()) if len(rt) else np.nan,
                "avg_trade_return": float(rt["return"].mean()) if len(rt) else np.nan,
                "best_trade": float(rt["return"].max()) if len(rt) else np.nan,
                "worst_trade": float(rt["return"].min()) if len(rt) else np.nan,
                "active_cycles": int(cy["active"].sum()) if len(cy) else 0,
                "cycle_win_rate": float((cy.loc[cy["active"], "cycle_return"] > 0.0).mean()) if len(cy) and cy["active"].any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _concentration(roundtrips: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if roundtrips.empty:
        return pd.DataFrame()
    for year, g in roundtrips.groupby("year"):
        total = float(g["pnl"].sum())
        positive = float(g.loc[g["pnl"] > 0.0, "pnl"].sum())
        by_stock = g.groupby("ts_code")["pnl"].sum().sort_values(ascending=False)
        by_trade = g["pnl"].sort_values(ascending=False)
        rows.append(
            {
                "year": int(year),
                "total_pnl": total,
                "positive_pnl": positive,
                "top1_stock_pnl": float(by_stock.head(1).sum()),
                "top3_stock_pnl": float(by_stock.head(3).sum()),
                "top5_stock_pnl": float(by_stock.head(5).sum()),
                "top1_trade_pnl": float(by_trade.head(1).sum()),
                "top3_trade_pnl": float(by_trade.head(3).sum()),
                "top5_trade_pnl": float(by_trade.head(5).sum()),
                "top1_stock_positive_share": float(by_stock.head(1).sum() / positive) if positive else np.nan,
                "top3_stock_positive_share": float(by_stock.head(3).sum() / positive) if positive else np.nan,
                "top5_stock_positive_share": float(by_stock.head(5).sum() / positive) if positive else np.nan,
                "top5_trade_positive_share": float(by_trade.head(5).sum() / positive) if positive else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _fmt_pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    return out


def _report(year: pd.DataFrame, month: pd.DataFrame, stock: pd.DataFrame, concentration: pd.DataFrame, cycle: pd.DataFrame) -> str:
    y = _fmt_pct(year, ["return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure", "trade_win_rate", "avg_trade_return", "best_trade", "worst_trade", "cycle_win_rate"])
    m = _fmt_pct(month, ["return", "benchmark_return", "max_drawdown", "avg_exposure", "trade_win_rate", "avg_trade_return"])
    c = concentration.copy()
    for col in ["top1_stock_positive_share", "top3_stock_positive_share", "top5_stock_positive_share", "top5_trade_positive_share"]:
        if col in c:
            c[col] = c[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    top_stock = stock.groupby("year", group_keys=False).head(10).copy()
    top_stock = _fmt_pct(top_stock, ["avg_return", "win_rate"])
    cy = _fmt_pct(cycle, ["cycle_return", "cycle_benchmark_return", "cycle_excess_return", "avg_exposure"])
    return "\n".join(
        [
            "# Frozen Rule Trade Audit",
            "",
            f"Rule: lookback={LOOKBACK}, min_obs={MIN_OBS}, flip when rolling RankIC < 0 and decile_spread < 0.",
            "",
            "## Year Summary",
            "",
            y.to_markdown(index=False),
            "",
            "## PnL Concentration",
            "",
            c.to_markdown(index=False),
            "",
            "## Monthly",
            "",
            m.to_markdown(index=False),
            "",
            "## Top Stocks By Year",
            "",
            top_stock.to_markdown(index=False),
            "",
            "## Cycles",
            "",
            cy.to_markdown(index=False),
            "",
        ]
    )


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN
    main(run)
