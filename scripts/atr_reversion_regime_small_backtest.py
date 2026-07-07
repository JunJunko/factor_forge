"""ATR small-portfolio backtest gated by existing HMM regime states."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from atr_reversion_benchmark import csi1000_open_to_open_returns
from atr_reversion_small_portfolio_backtest import (
    _can_sell_at_open,
    _json_default,
    _load_latest_panel,
    _metrics,
    _position_value,
)


TOP_NS = [5, 10]
REBALANCE_DAYS = [10]
COST_BPS = [10, 20]
INITIAL_CASH = 1_000_000.0
LOT_SIZE = 100


def main(
    predictions_path: str = "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_backtest_20260706T080140Z/predictions_all_features.parquet",
    states_path: str = "artifacts/value_hmm_regime_validations/value_hmm_regime_v1_20260704T164011Z_4ef4866a/hmm_daily_states.csv",
) -> None:
    pred_path = Path(predictions_path)
    states_path = Path(states_path)
    out_dir = pred_path.parent / f"regime_small_portfolio_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out_dir.mkdir(parents=True, exist_ok=False)
    log_path = out_dir / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    pred = pd.read_parquet(pred_path)
    pred["trade_date"] = pd.to_datetime(pred["trade_date"])
    states = pd.read_csv(states_path)
    states["trade_date"] = pd.to_datetime(states["trade_date"])
    log(f"loaded predictions rows={len(pred):,}")
    log(f"loaded HMM states rows={len(states):,} path={states_path}")

    version, panel = _load_latest_panel()
    keep = panel.groupby("ts_code")["amount_cny"].median().nlargest(1000).index
    panel = panel[panel["ts_code"].isin(keep)].copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    test_panel = panel[
        panel["trade_date"].between(pred["trade_date"].min(), pred["trade_date"].max())
    ].copy()
    log(f"test top1000 panel rows={len(test_panel):,} version={version}")

    policies = {
        "ungated": lambda row: 1.0,
        "hmm_position": lambda row: float(row.get("position_multiplier", 1.0)),
        "hmm_hard_ge_0p6": lambda row: 1.0 if float(row.get("position_multiplier", 1.0)) >= 0.6 else 0.0,
        "hmm_hard_full_only": lambda row: 1.0 if float(row.get("position_multiplier", 1.0)) >= 0.95 else 0.0,
    }
    rows = []
    for top_n in TOP_NS:
        for rebalance_days in REBALANCE_DAYS:
            for cost in COST_BPS:
                for policy_name, policy in policies.items():
                    log(f"backtest policy={policy_name} top_n={top_n} rebalance={rebalance_days} cost={cost}")
                    daily, trades = _run_regime_backtest(
                        test_panel,
                        pred,
                        states,
                        top_n=top_n,
                        rebalance_days=rebalance_days,
                        cost_bps=cost,
                        policy=policy,
                    )
                    tag = f"{policy_name}_top{top_n}_rebalance{rebalance_days}_cost{cost}"
                    daily.to_parquet(out_dir / f"daily_{tag}.parquet", index=False)
                    trades.to_parquet(out_dir / f"trades_{tag}.parquet", index=False)
                    metrics = _metrics(daily, trades)
                    metrics.update({
                        "policy": policy_name,
                        "top_n": top_n,
                        "rebalance_days": rebalance_days,
                        "cost_bps": cost,
                        "avg_holding_count": float(daily["holding_count"].mean()),
                        "avg_cash_ratio": float(daily["cash_ratio"].mean()),
                        "avg_daily_turnover": float(daily["turnover"].mean()),
                        "avg_exposure": float(daily["exposure"].mean()),
                        "active_rebalance_ratio": float((daily["exposure"] > 0).mean()),
                    })
                    rows.append(metrics)
                    log(
                        f"done ann={metrics['annualized_return']:.2%} "
                        f"excess={metrics['annualized_excess_return']:.2%} "
                        f"sharpe={metrics['sharpe']:.2f} maxdd={metrics['max_drawdown']:.2%}"
                    )

    metrics_df = pd.DataFrame(rows).sort_values(
        ["annualized_excess_return", "sharpe"], ascending=False
    )
    metrics_df.to_csv(out_dir / "regime_small_portfolio_metrics.csv", index=False, encoding="utf-8-sig")
    yearly = _yearly_tables(out_dir)
    yearly.to_csv(out_dir / "regime_yearly_metrics.csv", index=False, encoding="utf-8-sig")
    state_exposure = _state_summary(states, pred)
    state_exposure.to_csv(out_dir / "regime_state_monthly.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "data_version": version,
                "prediction_path": str(pred_path),
                "states_path": str(states_path),
                "run_dir": str(out_dir),
                "best": metrics_df.head(12).to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(_report(metrics_df, yearly, state_exposure), encoding="utf-8")
    log("wrote metrics/yearly/state/report")
    log("done")
    print(f"run_dir={out_dir}")


def _run_regime_backtest(panel, pred, states, *, top_n, rebalance_days, cost_bps, policy):
    data = panel.merge(pred, on=["trade_date", "ts_code"], how="left")
    data = data.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    dates = list(pd.Index(data["trade_date"].unique()).sort_values())
    by_date = {d: g.set_index("ts_code") for d, g in data.groupby("trade_date")}
    pred_by_date = {d: g.set_index("ts_code") for d, g in pred.groupby("trade_date")}
    states_by_date = states.set_index("trade_date")

    cash = INITIAL_CASH
    positions: dict[str, dict] = {}
    daily_rows, trade_rows = [], []
    half_cost = cost_bps / 2.0 / 10_000.0

    for i, date in enumerate(dates):
        today = by_date[date]
        turnover = cost = 0.0
        buys = sells = 0
        exposure = daily_rows[-1]["exposure"] if daily_rows else 0.0
        if i > 0 and (i - 1) % rebalance_days == 0:
            signal_date = dates[i - 1]
            state_row = states_by_date.loc[signal_date] if signal_date in states_by_date.index else pd.Series(dtype=float)
            exposure = float(np.clip(policy(state_row), 0.0, 1.0))
            # Full rebalance: sell what can be sold, then deploy exposure * cash.
            for code, pos in list(positions.items()):
                if code not in today.index or not _can_sell_at_open(today.loc[code]):
                    continue
                value = _position_value(pos, today.loc[code])
                sell_cost = value * half_cost
                cash += value - sell_cost
                turnover += value
                cost += sell_cost
                sells += 1
                trade_rows.append({"trade_date": date, "ts_code": code, "side": "SELL", "value": value, "cost": sell_cost, "exposure": exposure})
                del positions[code]

            signals = pred_by_date.get(signal_date)
            deploy_cash = cash * exposure
            if signals is not None and deploy_cash > 0:
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
                target_cash = deploy_cash / max(len(picks), 1)
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
                    trade_rows.append({"trade_date": date, "ts_code": code, "side": "BUY", "value": gross, "cost": buy_cost, "exposure": exposure})

        pos_value = 0.0
        for code, pos in positions.items():
            pos_value += _position_value(pos, today.loc[code]) if code in today.index else pos["shares"] * pos["entry_raw_open"]
        nav = cash + pos_value
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
            "exposure": exposure,
        })

    daily = pd.DataFrame(daily_rows)
    daily["benchmark_return"] = csi1000_open_to_open_returns(dates)
    daily["return"] = daily["nav"].pct_change().fillna(0.0)
    daily["excess_return"] = daily["return"] - daily["benchmark_return"]
    return daily, pd.DataFrame(trade_rows)


def _benchmark_return(data, dates, i):
    if i == 0:
        return 0.0
    today = data[data["trade_date"].eq(dates[i]) & data["is_tradeable"].fillna(False)]
    yesterday = data[data["trade_date"].eq(dates[i - 1]) & data["is_tradeable"].fillna(False)]
    joined = yesterday[["ts_code", "adj_open"]].merge(today[["ts_code", "adj_open"]], on="ts_code", suffixes=("_prev", "_cur"))
    ret = joined["adj_open_cur"] / joined["adj_open_prev"] - 1.0
    return float(ret.replace([np.inf, -np.inf], np.nan).dropna().mean() or 0.0)


def _yearly_tables(out_dir: Path) -> pd.DataFrame:
    rows = []
    for path in out_dir.glob("daily_*.parquet"):
        tag = path.stem.removeprefix("daily_")
        parts = tag.split("_")
        policy = "_".join(parts[:-3])
        top_n = int(parts[-3].removeprefix("top"))
        rebalance = int(parts[-2].removeprefix("rebalance"))
        cost = int(parts[-1].removeprefix("cost"))
        df = pd.read_parquet(path)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        for year, g in df.groupby(df["trade_date"].dt.year):
            if len(g) < 2:
                continue
            total = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1
            bench = (1 + g["benchmark_return"]).prod() - 1
            dd = g["nav"] / g["nav"].cummax() - 1
            rows.append({
                "policy": policy,
                "top_n": top_n,
                "rebalance_days": rebalance,
                "cost_bps": cost,
                "year": int(year),
                "return": float(total),
                "benchmark_return": float(bench),
                "excess_return": float(total - bench),
                "max_drawdown": float(dd.min()),
                "avg_exposure": float(g["exposure"].mean()),
            })
    return pd.DataFrame(rows)


def _state_summary(states, pred):
    s = states.copy()
    s["month"] = s["trade_date"].dt.to_period("M").astype(str)
    pred_days = pd.DataFrame({"trade_date": sorted(pd.to_datetime(pred["trade_date"].unique()))})
    s = pred_days.merge(s, on="trade_date", how="left")
    return s.groupby(["month", "predicted_state", "state_name"], dropna=False).agg(
        days=("trade_date", "size"),
        avg_position_multiplier=("position_multiplier", "mean"),
    ).reset_index()


def _report(metrics, yearly, state_summary):
    show = metrics[[
        "policy", "top_n", "cost_bps", "annualized_return", "benchmark_annualized_return",
        "annualized_excess_return", "sharpe", "max_drawdown", "avg_exposure", "avg_daily_turnover",
    ]].head(20).copy()
    for col in ["annualized_return", "benchmark_annualized_return", "annualized_excess_return", "max_drawdown", "avg_exposure", "avg_daily_turnover"]:
        show[col] = show[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    show["sharpe"] = show["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    y = yearly[(yearly.top_n.eq(5)) & (yearly.cost_bps.eq(10))].copy()
    for col in ["return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"]:
        y[col] = y[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    return "\n".join([
        "# ATR Reversion Small Portfolio With Existing HMM Regime",
        "",
        "## Overall",
        "",
        show.to_markdown(index=False),
        "",
        "## Yearly Top5 Cost10",
        "",
        y.to_markdown(index=False),
        "",
        "## State Monthly Summary Tail",
        "",
        state_summary.tail(20).to_markdown(index=False),
        "",
    ])


if __name__ == "__main__":
    pred = sys.argv[1] if len(sys.argv) > 1 else "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_backtest_20260706T080140Z/predictions_all_features.parquet"
    states = sys.argv[2] if len(sys.argv) > 2 else "artifacts/value_hmm_regime_validations/value_hmm_regime_v1_20260704T164011Z_4ef4866a/hmm_daily_states.csv"
    main(pred, states)
