"""Small-account ATR reversion backtest with periodic full rebalance.

This is intentionally different from the overlapping-sleeve research backtest:
it holds only Top5/Top10 names and rebalances the whole portfolio every 5/10/15
trading days, closer to what an individual investor can operate.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository


TOP_NS = [5, 10]
REBALANCE_DAYS = [5, 10, 15]
COST_BPS = [0, 10, 20, 30]
INITIAL_CASH = 1_000_000.0
LOT_SIZE = 100


def main(
    predictions_path: str = "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_backtest_20260706T080140Z/predictions_all_features.parquet",
) -> None:
    pred_path = Path(predictions_path)
    out_dir = pred_path.parent / f"small_portfolio_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out_dir.mkdir(parents=True, exist_ok=False)
    log_path = out_dir / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"loading predictions {pred_path}")
    pred = pd.read_parquet(pred_path)
    pred["trade_date"] = pd.to_datetime(pred["trade_date"])
    log(f"predictions rows={len(pred):,}")

    version, panel = _load_latest_panel()
    log(f"loaded panel rows={len(panel):,} version={version}")
    keep = panel.groupby("ts_code")["amount_cny"].median().nlargest(1000).index
    panel = panel[panel["ts_code"].isin(keep)].copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    test_panel = panel[
        panel["trade_date"].between(pred["trade_date"].min(), pred["trade_date"].max())
    ].copy()
    log(f"test top1000 panel rows={len(test_panel):,}")

    rows = []
    for rebalance_days in REBALANCE_DAYS:
        for top_n in TOP_NS:
            for cost_bps in COST_BPS:
                log(f"backtest top_n={top_n} rebalance_days={rebalance_days} cost_bps={cost_bps}")
                daily, trades = _run_periodic_backtest(
                    test_panel,
                    pred,
                    top_n=top_n,
                    rebalance_days=rebalance_days,
                    cost_bps=cost_bps,
                )
                tag = f"top{top_n}_rebalance{rebalance_days}_cost{cost_bps}"
                daily.to_parquet(out_dir / f"daily_{tag}.parquet", index=False)
                trades.to_parquet(out_dir / f"trades_{tag}.parquet", index=False)
                metrics = _metrics(daily, trades)
                metrics.update({
                    "top_n": top_n,
                    "rebalance_days": rebalance_days,
                    "cost_bps": cost_bps,
                    "avg_holding_count": float(daily["holding_count"].mean()),
                    "avg_cash_ratio": float(daily["cash_ratio"].mean()),
                    "avg_daily_turnover": float(daily["turnover"].mean()),
                })
                rows.append(metrics)
                log(
                    "done "
                    f"ann={metrics['annualized_return']:.2%} "
                    f"excess={metrics['annualized_excess_return']:.2%} "
                    f"sharpe={metrics['sharpe']:.2f} "
                    f"maxdd={metrics['max_drawdown']:.2%}"
                )

    metrics_df = pd.DataFrame(rows).sort_values(
        ["annualized_excess_return", "sharpe"], ascending=False
    )
    metrics_df.to_csv(out_dir / "small_portfolio_metrics.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "data_version": version,
                "prediction_path": str(pred_path),
                "run_dir": str(out_dir),
                "best": metrics_df.head(10).to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(_report(metrics_df, version), encoding="utf-8")
    log("wrote small_portfolio_metrics.csv, summary.json, report.md")
    log("done")
    print(f"run_dir={out_dir}")


def _run_periodic_backtest(
    panel: pd.DataFrame,
    pred: pd.DataFrame,
    *,
    top_n: int,
    rebalance_days: int,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = panel.merge(pred, on=["trade_date", "ts_code"], how="left")
    data = data.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    dates = list(pd.Index(data["trade_date"].unique()).sort_values())
    by_date = {d: g.set_index("ts_code") for d, g in data.groupby("trade_date")}
    pred_by_date = {d: g.set_index("ts_code") for d, g in pred.groupby("trade_date")}

    cash = INITIAL_CASH
    positions: dict[str, dict] = {}
    daily_rows = []
    trade_rows = []
    half_cost = cost_bps / 2.0 / 10_000.0

    for i, date in enumerate(dates):
        today = by_date[date]
        turnover = 0.0
        cost = 0.0
        buys = sells = 0

        if i > 0 and (i - 1) % rebalance_days == 0:
            # Signal at dates[i-1], execute at today's open.
            for code, pos in list(positions.items()):
                if code not in today.index or not _can_sell_at_open(today.loc[code]):
                    continue
                value = _position_value(pos, today.loc[code])
                sell_cost = value * half_cost
                cash += value - sell_cost
                turnover += value
                cost += sell_cost
                sells += 1
                trade_rows.append({"trade_date": date, "ts_code": code, "side": "SELL", "value": value, "cost": sell_cost})
                del positions[code]

            signal_date = dates[i - 1]
            signals = pred_by_date.get(signal_date)
            if signals is not None and cash > 0:
                eligible = signals.join(today[[
                    "raw_open", "adj_open", "is_tradeable", "is_suspended",
                    "is_st", "is_delisting_period", "listing_trade_days", "is_limit_up_open",
                ]], how="inner")
                mask = (
                    eligible["factor_value"].notna()
                    & eligible["is_tradeable"].fillna(False).astype(bool)
                    & ~eligible["is_suspended"].fillna(True).astype(bool)
                    & ~eligible["is_st"].fillna(True).astype(bool)
                    & ~eligible["is_delisting_period"].fillna(True).astype(bool)
                    & eligible["listing_trade_days"].ge(60)
                    & ~eligible["is_limit_up_open"].fillna(False).astype(bool)
                )
                picks = eligible[mask].sort_values("factor_value", ascending=False).head(top_n)
                target_cash = cash / max(len(picks), 1)
                for code, row in picks.iterrows():
                    price = float(row["raw_open"])
                    shares = int(target_cash // (price * LOT_SIZE)) * LOT_SIZE
                    if shares <= 0:
                        continue
                    gross = shares * price
                    buy_cost = gross * half_cost
                    if gross + buy_cost > cash:
                        continue
                    cash -= gross + buy_cost
                    turnover += gross
                    cost += buy_cost
                    buys += 1
                    positions[code] = {
                        "shares": shares,
                        "entry_raw_open": price,
                        "entry_adj_open": float(row["adj_open"]),
                    }
                    trade_rows.append({"trade_date": date, "ts_code": code, "side": "BUY", "value": gross, "cost": buy_cost})

        pos_value = 0.0
        for code, pos in positions.items():
            if code in today.index:
                pos_value += _position_value(pos, today.loc[code])
            else:
                pos_value += pos["shares"] * pos["entry_raw_open"]
        nav = cash + pos_value
        bench = _benchmark_return(data, dates, i)
        daily_rows.append({
            "trade_date": date,
            "nav": nav,
            "cash": cash,
            "cash_ratio": cash / nav if nav > 0 else np.nan,
            "holding_count": len(positions),
            "turnover": turnover / (daily_rows[-1]["nav"] if daily_rows else INITIAL_CASH),
            "transaction_cost": cost,
            "executed_buys": buys,
            "executed_sells": sells,
            "benchmark_return": bench,
        })

    daily = pd.DataFrame(daily_rows)
    daily["return"] = daily["nav"].pct_change().fillna(0.0)
    daily["excess_return"] = daily["return"] - daily["benchmark_return"]
    return daily, pd.DataFrame(trade_rows)


def _position_value(pos: dict, row: pd.Series) -> float:
    mark = row.get("adj_open", np.nan)
    if not np.isfinite(mark):
        mark = pos["entry_adj_open"]
    return pos["shares"] * pos["entry_raw_open"] * float(mark) / pos["entry_adj_open"]


def _can_sell_at_open(row: pd.Series) -> bool:
    return bool(
        np.isfinite(row.get("raw_open", np.nan))
        and not row.get("is_suspended", True)
        and not row.get("is_limit_down_open", False)
    )


def _benchmark_return(data: pd.DataFrame, dates: list[pd.Timestamp], i: int) -> float:
    if i == 0:
        return 0.0
    today = data[data["trade_date"].eq(dates[i]) & data["is_tradeable"].fillna(False)]
    yesterday = data[data["trade_date"].eq(dates[i - 1]) & data["is_tradeable"].fillna(False)]
    joined = yesterday[["ts_code", "adj_open"]].merge(
        today[["ts_code", "adj_open"]],
        on="ts_code",
        suffixes=("_prev", "_cur"),
    )
    ret = joined["adj_open_cur"] / joined["adj_open_prev"] - 1.0
    return float(ret.replace([np.inf, -np.inf], np.nan).dropna().mean() or 0.0)


def _metrics(daily: pd.DataFrame, trades: pd.DataFrame) -> dict:
    count = max(len(daily) - 1, 1)
    total = daily["nav"].iloc[-1] / daily["nav"].iloc[0] - 1
    annual = (1 + total) ** (252 / count) - 1 if total > -1 else -1.0
    bench_total = float((1 + daily["benchmark_return"]).prod() - 1)
    bench_annual = (1 + bench_total) ** (252 / count) - 1 if bench_total > -1 else -1.0
    vol = float(daily["return"].std(ddof=1) * np.sqrt(252))
    dd = daily["nav"] / daily["nav"].cummax() - 1
    return {
        "total_return": float(total),
        "annualized_return": float(annual),
        "benchmark_annualized_return": float(bench_annual),
        "annualized_excess_return": float(annual - bench_annual),
        "annualized_volatility": vol,
        "sharpe": float(annual / vol) if vol > 0 else np.nan,
        "max_drawdown": float(dd.min()),
        "trade_count": int(len(trades)),
    }


def _load_latest_panel() -> tuple[str, pd.DataFrame]:
    project = load_project("configs/project.yaml")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest("latest")
    _, panel = repo.load_panel(version)
    return version, panel


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def _report(metrics: pd.DataFrame, version: str) -> str:
    table = metrics[[
        "top_n", "rebalance_days", "cost_bps", "annualized_return",
        "benchmark_annualized_return", "annualized_excess_return", "sharpe",
        "max_drawdown", "avg_daily_turnover", "avg_holding_count",
    ]].head(20).copy()
    pct_cols = [
        "annualized_return", "benchmark_annualized_return", "annualized_excess_return",
        "max_drawdown", "avg_daily_turnover",
    ]
    for col in pct_cols:
        table[col] = table[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    table["sharpe"] = table["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    return "\n".join([
        "# ATR Reversion Small Portfolio Backtest",
        "",
        f"- data version: `{version}`",
        "- portfolio: Top5/Top10 only",
        "- rebalance: full rebalance every 5/10/15 trading days",
        "- execution: signal at close, trade next open",
        "",
        table.to_markdown(index=False),
        "",
    ])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_backtest_20260706T080140Z/predictions_all_features.parquet")
