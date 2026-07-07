"""Test sell-impact hazard rules on ATR Top5 10-day strategy.

Rules are evaluated as an execution overlay:
- no-buy: select the original Top5, then skip hazardous names and keep cash.
- early-exit: if a held stock is hazardous at T close, sell at T+1 open
  before the 10-day scheduled rebalance, subject to sellability.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository

from atr_reversion_pit_hmm_calibrated_backtest import _tiered_weight
from atr_reversion_small_portfolio_backtest import (
    INITIAL_CASH,
    LOT_SIZE,
    _can_sell_at_open,
    _json_default,
    _metrics,
    _position_value,
)


PIT_RUN = Path("artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z")
TRAIN_RUN = PIT_RUN / "training_window_experiment_20260706T130710Z" / "rolling_2y"
GATE_RUN = PIT_RUN / "training_window_defensive_gate_rolling_2y_20260706T133808Z"
SELL_IMPACT_RUN = Path("artifacts/runs/sell_impact_efficiency_v1__20260702T032703Z__be62d97d")
OUTPUT_ROOT = Path("artifacts/atr_reversion_sell_impact_exit")
TOP_N = 5
REBALANCE_DAYS = 10
FOLDS = [
    {"name": "test_2025", "start": "2025-01-01", "end": "2025-12-31"},
    {"name": "test_2026h1", "start": "2026-01-01", "end": "2026-06-30"},
]


def _load_panel() -> tuple[str, pd.DataFrame]:
    project = load_project("configs/project.yaml")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest("latest")
    _, panel = repo.load_panel(version)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    return version, panel


def _load_hazard() -> pd.DataFrame:
    eff = pd.read_parquet(SELL_IMPACT_RUN / "factor_values.parquet")
    dev = pd.read_parquet(SELL_IMPACT_RUN / "conditioning_factor_values.parquet")
    eff["trade_date"] = pd.to_datetime(eff["trade_date"])
    dev["trade_date"] = pd.to_datetime(dev["trade_date"])
    eff = eff.rename(columns={"factor_value": "sell_impact_efficiency"})
    dev = dev.rename(columns={"factor_value": "sell_impact_deviation_60d"})
    data = eff[["trade_date", "ts_code", "sell_impact_efficiency", "factor_valid"]].rename(
        columns={"factor_valid": "eff_valid"}
    ).merge(
        dev[["trade_date", "ts_code", "sell_impact_deviation_60d", "factor_valid"]].rename(
            columns={"factor_valid": "dev_valid"}
        ),
        on=["trade_date", "ts_code"],
        how="inner",
    )
    valid_eff = data["eff_valid"].fillna(False) & data["sell_impact_efficiency"].notna()
    valid_dev = data["dev_valid"].fillna(False) & data["sell_impact_deviation_60d"].notna()
    data["eff_q80"] = data["sell_impact_efficiency"] >= data["sell_impact_efficiency"].where(valid_eff).groupby(data["trade_date"]).transform(lambda s: s.quantile(0.8))
    data["dev_q80"] = data["sell_impact_deviation_60d"] >= data["sell_impact_deviation_60d"].where(valid_dev).groupby(data["trade_date"]).transform(lambda s: s.quantile(0.8))
    data["hazard_dev_q5"] = valid_dev & data["dev_q80"].fillna(False)
    data["hazard_dev_q5_eff_q5"] = data["hazard_dev_q5"] & valid_eff & data["eff_q80"].fillna(False)
    return data[[
        "trade_date",
        "ts_code",
        "sell_impact_efficiency",
        "sell_impact_deviation_60d",
        "hazard_dev_q5",
        "hazard_dev_q5_eff_q5",
    ]]


def _hmm_ranks(path: Path) -> dict[str, int]:
    perf = pd.read_csv(path)
    ordered = perf.sort_values("mean_excess", ascending=False)["predicted_state"].astype(int).tolist()
    return {"best": ordered[0], "neutral": ordered[1], "worst": ordered[2]}


def _benchmark_return(data: pd.DataFrame, dates: list[pd.Timestamp], i: int) -> float:
    if i == 0:
        return 0.0
    signal_date = dates[i - 1]
    today = data[data["trade_date"].eq(dates[i])]
    signal = data[
        data["trade_date"].eq(signal_date)
        & data["pit_top1000"].fillna(False)
        & data["is_tradeable"].fillna(False)
    ]
    joined = signal[["ts_code", "adj_open"]].merge(
        today[["ts_code", "adj_open", "is_tradeable"]],
        on="ts_code",
        suffixes=("_prev", "_cur"),
    )
    ret = joined.loc[joined["is_tradeable"].fillna(False), "adj_open_cur"] / joined.loc[
        joined["is_tradeable"].fillna(False), "adj_open_prev"
    ] - 1.0
    return float(ret.replace([np.inf, -np.inf], np.nan).dropna().mean() or 0.0)


def _run_backtest(
    panel: pd.DataFrame,
    pred: pd.DataFrame,
    states: pd.DataFrame,
    hazard: pd.DataFrame,
    *,
    cost_bps: int,
    hazard_col: str,
    no_buy: bool,
    early_exit: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = panel.merge(pred, on=["trade_date", "ts_code"], how="left")
    data = data.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    dates = list(pd.Index(data["trade_date"].unique()).sort_values())
    by_date = {d: g.set_index("ts_code") for d, g in data.groupby("trade_date")}
    pred_by_date = {d: g.set_index("ts_code") for d, g in pred.groupby("trade_date")}
    states_by_date = states.set_index("trade_date")
    hazard_by_date = {
        d: g.set_index("ts_code")
        for d, g in hazard[hazard["trade_date"].isin(dates)].groupby("trade_date")
    }
    cash = INITIAL_CASH
    positions: dict[str, dict] = {}
    daily_rows: list[dict] = []
    trade_rows: list[dict] = []
    half_cost = cost_bps / 2.0 / 10_000.0
    current_exposure = 0.0

    for i, date in enumerate(dates):
        today = by_date[date]
        turnover = cost = 0.0
        buys = sells = early_sells = skipped_buys = 0
        scheduled_rebalance = i > 0 and (i - 1) % REBALANCE_DAYS == 0

        if early_exit and i > 0 and not scheduled_rebalance:
            signal_date = dates[i - 1]
            hazards = hazard_by_date.get(signal_date)
            if hazards is not None:
                for code, pos in list(positions.items()):
                    if i <= int(pos.get("entry_i", -1)):
                        continue
                    if code not in hazards.index or not bool(hazards.loc[code].get(hazard_col, False)):
                        continue
                    if code not in today.index or not _can_sell_at_open(today.loc[code]):
                        continue
                    value = _position_value(pos, today.loc[code])
                    sell_cost = value * half_cost
                    cash += value - sell_cost
                    turnover += value
                    cost += sell_cost
                    sells += 1
                    early_sells += 1
                    hrow = hazards.loc[code]
                    trade_rows.append({
                        "trade_date": date,
                        "signal_date": signal_date,
                        "ts_code": code,
                        "side": "EARLY_SELL",
                        "value": value,
                        "cost": sell_cost,
                        "exposure": current_exposure,
                        "hazard_col": hazard_col,
                        "sell_impact_efficiency": float(hrow.get("sell_impact_efficiency", np.nan)),
                        "sell_impact_deviation_60d": float(hrow.get("sell_impact_deviation_60d", np.nan)),
                    })
                    del positions[code]

        if scheduled_rebalance:
            signal_date = dates[i - 1]
            state_row = states_by_date.loc[signal_date] if signal_date in states_by_date.index else pd.Series(dtype=float)
            current_exposure = float(np.clip(state_row.get("final_exposure", state_row.get("exposure", 0.0)), 0.0, 1.0))
            for code, pos in list(positions.items()):
                if code not in today.index or not _can_sell_at_open(today.loc[code]):
                    continue
                value = _position_value(pos, today.loc[code])
                sell_cost = value * half_cost
                cash += value - sell_cost
                turnover += value
                cost += sell_cost
                sells += 1
                trade_rows.append({
                    "trade_date": date,
                    "signal_date": signal_date,
                    "ts_code": code,
                    "side": "SELL",
                    "value": value,
                    "cost": sell_cost,
                    "exposure": current_exposure,
                })
                del positions[code]

            signals = pred_by_date.get(signal_date)
            deploy_cash = cash * current_exposure
            if signals is not None and deploy_cash > 0:
                eligible = signals.join(today[[
                    "raw_open",
                    "adj_open",
                    "is_tradeable",
                    "is_suspended",
                    "is_st",
                    "is_delisting_period",
                    "listing_trade_days",
                    "is_limit_up_open",
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
                picks = eligible[mask].sort_values("factor_value", ascending=False).head(TOP_N).copy()
                hazards = hazard_by_date.get(signal_date)
                if no_buy and hazards is not None and not picks.empty:
                    hazard_flags = picks.index.to_series().map(
                        lambda code: bool(hazards.loc[code].get(hazard_col, False)) if code in hazards.index else False
                    )
                    skipped_buys = int(hazard_flags.sum())
                    picks = picks.loc[~hazard_flags.to_numpy()]
                target_cash = deploy_cash / TOP_N
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
                        "entry_i": i,
                    }
                    trade_rows.append({
                        "trade_date": date,
                        "signal_date": signal_date,
                        "ts_code": code,
                        "side": "BUY",
                        "value": gross,
                        "cost": buy_cost,
                        "exposure": current_exposure,
                    })

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
            "early_sells": early_sells,
            "skipped_buys": skipped_buys,
            "benchmark_return": _benchmark_return(data, dates, i),
            "exposure": current_exposure,
        })
    daily = pd.DataFrame(daily_rows)
    daily["return"] = daily["nav"].pct_change().fillna(0.0)
    daily["excess_return"] = daily["return"] - daily["benchmark_return"]
    return daily, pd.DataFrame(trade_rows)


def _yearly(daily: pd.DataFrame, policy: str, fold: str, cost: int, hazard_col: str) -> list[dict]:
    rows = []
    for year, g in daily.groupby(daily["trade_date"].dt.year):
        if len(g) < 2:
            continue
        total = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1.0
        bench = (1.0 + g["benchmark_return"]).prod() - 1.0
        dd = g["nav"] / g["nav"].cummax() - 1.0
        rows.append({
            "policy": policy,
            "fold": fold,
            "year": int(year),
            "cost_bps": cost,
            "hazard_col": hazard_col,
            "return": float(total),
            "benchmark_return": float(bench),
            "excess_return": float(total - bench),
            "max_drawdown": float(dd.min()),
            "avg_exposure": float(g["exposure"].mean()),
            "avg_holding_count": float(g["holding_count"].mean()),
            "early_sells": int(g["early_sells"].sum()) if "early_sells" in g else 0,
            "skipped_buys": int(g["skipped_buys"].sum()) if "skipped_buys" in g else 0,
        })
    return rows


def _report(comparison: pd.DataFrame, yearly: pd.DataFrame) -> str:
    c = comparison.copy()
    for col in [x for x in c.columns if x.endswith(("ann", "excess", "maxdd", "exposure", "delta"))]:
        c[col] = c[col].map(lambda v: f"{v:.2%}" if pd.notna(v) else "")
    for col in ["base_sharpe", "overlay_sharpe"]:
        c[col] = c[col].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
    y = yearly.copy()
    for col in ["return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"]:
        y[col] = y[col].map(lambda v: f"{v:.2%}" if pd.notna(v) else "")
    return "\n".join([
        "# ATR Sell-Impact Hazard Exit Overlay",
        "",
        "## OOS Comparison",
        "",
        c.to_markdown(index=False),
        "",
        "## Yearly",
        "",
        y.to_markdown(index=False),
        "",
    ])


def _comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    base = metrics[metrics["policy"].eq("baseline")].copy()
    overlays = metrics[~metrics["policy"].eq("baseline")].copy()
    keep = ["fold", "cost_bps", "annualized_return", "annualized_excess_return", "sharpe", "max_drawdown", "avg_exposure"]
    left = base[keep].rename(columns={
        "annualized_return": "base_ann",
        "annualized_excess_return": "base_excess",
        "sharpe": "base_sharpe",
        "max_drawdown": "base_maxdd",
        "avg_exposure": "base_exposure",
    })
    right = overlays[["policy", "hazard_col", *keep]].rename(columns={
        "annualized_return": "overlay_ann",
        "annualized_excess_return": "overlay_excess",
        "sharpe": "overlay_sharpe",
        "max_drawdown": "overlay_maxdd",
        "avg_exposure": "overlay_exposure",
    })
    out = left.merge(right, on=["fold", "cost_bps"], how="inner")
    out["ann_delta"] = out["overlay_ann"] - out["base_ann"]
    out["excess_delta"] = out["overlay_excess"] - out["base_excess"]
    out["maxdd_delta"] = out["overlay_maxdd"] - out["base_maxdd"]
    out["exposure_delta"] = out["overlay_exposure"] - out["base_exposure"]
    return out


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_exit_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    version, panel = _load_panel()
    pit = pd.read_parquet(PIT_RUN / "pit_liquidity_flags.parquet")
    pit["trade_date"] = pd.to_datetime(pit["trade_date"])
    hazard = _load_hazard()
    log(f"loaded panel version={version} hazard_dates={hazard.trade_date.min().date()}..{hazard.trade_date.max().date()}")
    rows = []
    yearly_rows = []
    policies: dict[str, tuple[bool, bool]] = {
        "baseline": (False, False),
        "no_buy": (True, False),
        "early_exit": (False, True),
        "no_buy_and_early_exit": (True, True),
    }
    hazard_cols = ["hazard_dev_q5", "hazard_dev_q5_eff_q5"]

    for fold in FOLDS:
        fold_name = fold["name"]
        pred = pd.read_parquet(TRAIN_RUN / fold_name / "predictions_valid_test.parquet")
        pred["trade_date"] = pd.to_datetime(pred["trade_date"])
        pred = pred[pred["trade_date"].between(pd.Timestamp(fold["start"]), pd.Timestamp(fold["end"]))]
        states = pd.read_csv(PIT_RUN / "walk_forward_20260706T102017Z" / fold_name / "hmm_daily_states.csv")
        states["trade_date"] = pd.to_datetime(states["trade_date"])
        panel_bt = panel[panel["trade_date"].between(pd.Timestamp(fold["start"]), pd.Timestamp(fold["end"]))].merge(
            pit, on=["trade_date", "ts_code"], how="left"
        )
        panel_bt["pit_top1000"] = panel_bt["pit_top1000"].fillna(False).astype(bool)
        for cost in [10, 20]:
            ranks = _hmm_ranks(TRAIN_RUN / fold_name / f"state_validation_perf_cost{cost}.csv")
            scores = pd.read_csv(GATE_RUN / f"gate_scores_rolling_2y_{fold_name}_risk_kill_only_cost{cost}.csv")
            scores["trade_date"] = pd.to_datetime(scores["trade_date"])
            states_ext = states.merge(scores[["trade_date", "strategy_gate"]], on="trade_date", how="left")
            states_ext["strategy_gate"] = states_ext["strategy_gate"].fillna(1.0)
            states_ext["hmm_exposure"] = states_ext.apply(lambda r: _tiered_weight(r, ranks), axis=1)
            states_ext["final_exposure"] = states_ext["hmm_exposure"] * states_ext["strategy_gate"]
            for hazard_col in hazard_cols:
                for policy_name, (no_buy, early_exit) in policies.items():
                    if policy_name == "baseline" and hazard_col != hazard_cols[0]:
                        continue
                    log(f"run fold={fold_name} cost={cost} policy={policy_name} hazard={hazard_col}")
                    daily, trades = _run_backtest(
                        panel_bt,
                        pred,
                        states_ext,
                        hazard,
                        cost_bps=cost,
                        hazard_col=hazard_col,
                        no_buy=no_buy,
                        early_exit=early_exit,
                    )
                    effective_hazard = "none" if policy_name == "baseline" else hazard_col
                    tag = f"{fold_name}_{policy_name}_{effective_hazard}_top{TOP_N}_cost{cost}"
                    daily.to_parquet(output / f"daily_{tag}.parquet", index=False)
                    trades.to_parquet(output / f"trades_{tag}.parquet", index=False)
                    metrics = _metrics(daily, trades)
                    metrics.update({
                        "policy": policy_name,
                        "fold": fold_name,
                        "cost_bps": cost,
                        "hazard_col": effective_hazard,
                        "top_n": TOP_N,
                        "rebalance_days": REBALANCE_DAYS,
                        "avg_exposure": float(daily["exposure"].mean()),
                        "avg_holding_count": float(daily["holding_count"].mean()),
                        "avg_daily_turnover": float(daily["turnover"].mean()),
                        "early_sell_count": int((trades["side"].eq("EARLY_SELL")).sum()) if not trades.empty and "side" in trades else 0,
                        "skipped_buy_count": int(daily["skipped_buys"].sum()) if "skipped_buys" in daily else 0,
                    })
                    rows.append(metrics)
                    yearly_rows.extend(_yearly(daily, policy_name, fold_name, cost, effective_hazard))
                    log(
                        f"done ann={metrics['annualized_return']:.2%} "
                        f"maxdd={metrics['max_drawdown']:.2%} "
                        f"early={metrics['early_sell_count']} skip={metrics['skipped_buy_count']}"
                    )

    metrics = pd.DataFrame(rows)
    yearly = pd.DataFrame(yearly_rows)
    comparison = _comparison(metrics)
    metrics.to_csv(output / "sell_impact_exit_metrics.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "sell_impact_exit_yearly.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "sell_impact_exit_comparison.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "data_version": version,
                "pit_run": str(PIT_RUN),
                "training_run": str(TRAIN_RUN),
                "gate_run": str(GATE_RUN),
                "sell_impact_run": str(SELL_IMPACT_RUN),
                "run_dir": str(output),
                "metrics": metrics.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(comparison, yearly), encoding="utf-8")
    log(f"done output={output}")
    print(f"run_dir={output}")


if __name__ == "__main__":
    main()
