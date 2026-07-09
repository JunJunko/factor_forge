from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sell_impact_ranker_timing_compare as timing_compare
import sell_impact_score_band_walkforward as wf
import sell_impact_sorting_repair as base
from factor_forge.config import CostModel, ExecutionConstraints


DEFAULT_SOURCE_RUN = Path("artifacts/strategy_reviews/sell_impact_low_vol_regime_20260708T155220Z")
DEFAULT_PREDICTIONS_FILE = "predictions.parquet"
DEFAULT_VARIANT = "B_cluster_plus_low_vol"
DEFAULT_OUTPUT_ROOT = Path("artifacts/strategy_reviews")
DEFAULT_INITIAL_CASH = 110_000.0
DEFAULT_TOP_N = 5
DEFAULT_HOLDING_DAYS = 10
DEFAULT_COST_BPS = 20.0
DEFAULT_LOT_SIZE = 100


@dataclass
class AccountPosition:
    ts_code: str
    shares: int
    entry_date: pd.Timestamp
    due_date: pd.Timestamp | None
    entry_raw_open: float
    entry_adj_open: float
    signal_date: pd.Timestamp
    condition_quantile: int | None = None


def main() -> None:
    args = parse_args()
    output = args.output_root / (
        f"sell_impact_account_target_exposure_"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"source_run={args.source_run}")
    log(
        "account target exposure backtest "
        f"variant={args.variant} initial_cash={args.initial_cash:,.2f} "
        f"top_n_per_day={args.top_n} holding_days={args.holding_days} "
        f"cost_bps={args.cost_bps:g} lot_size={args.lot_size}"
    )
    version, panel = base.load_panel()
    market_benchmark = timing_compare.load_market_benchmark(version)
    predictions = load_predictions(args.source_run, args.predictions_file, args.variant)
    timing = timing_compare.load_position_multiplier(args.timing_daily)
    log(
        f"loaded panel_rows={len(panel):,} prediction_rows={len(predictions):,} "
        f"timing_dates={len(timing):,} data_version={version}"
    )

    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []

    for fold in wf.FOLDS:
        fold_name = fold["fold"]
        pred = predictions.loc[
            predictions["sample"].eq("test")
            & predictions["fold"].eq(fold_name)
            & predictions["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))
        ].copy()
        panel_slice = panel.loc[
            panel["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))
        ].copy()
        log(f"{fold_name}: panel_rows={len(panel_slice):,} pred_rows={len(pred):,}")
        daily, trades, positions, metrics = run_account_backtest(
            panel=panel_slice,
            predictions=pred,
            timing=timing,
            market_benchmark=market_benchmark,
            initial_cash=args.initial_cash,
            top_n=args.top_n,
            holding_days=args.holding_days,
            lot_size=args.lot_size,
            cost_bps=args.cost_bps,
            constraints=ExecutionConstraints(
                exclude_suspended=True,
                cannot_buy_limit_up=True,
                cannot_sell_limit_down=True,
                exclude_st=True,
                exclude_delisting_period=True,
                min_listing_days=60,
            ),
            cost_model=CostModel(
                commission_bps_per_side=3,
                slippage_bps_per_side=5,
                stamp_duty_bps_sell=5,
            ),
        )
        csi1000_annual = float(metrics.get("market_index_annualized_return", np.nan))
        row = {
            "fold": fold_name,
            "variant": args.variant,
            "initial_cash": args.initial_cash,
            "top_n_per_day": args.top_n,
            "holding_days": args.holding_days,
            "cost_bps": args.cost_bps,
            **metrics,
            "annualized_excess_return_vs_csi1000": float(metrics["annualized_return"] - csi1000_annual),
        }
        fold_rows.append(row)
        daily["fold"] = fold_name
        trades["fold"] = fold_name
        positions["fold"] = fold_name
        daily_frames.append(daily)
        trade_frames.append(trades)
        position_frames.append(positions)
        log(
            f"{fold_name}: ann={metrics['annualized_return']:.2%} "
            f"excess_csi1000={row['annualized_excess_return_vs_csi1000']:.2%} "
            f"mdd={metrics['max_drawdown']:.2%} buys={metrics['executed_buys']} "
            f"exec_rate={metrics['execution_rate']:.2%} "
            f"avg_exposure={metrics['avg_gross_exposure_ratio']:.2%}"
        )

    daily_all = pd.concat(daily_frames, ignore_index=True)
    trades_all = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    positions_all = pd.concat(position_frames, ignore_index=True) if position_frames else pd.DataFrame()
    fold_metrics = pd.DataFrame(fold_rows)
    yearly = period_breakdown(daily_all, trades_all, "Y")
    monthly = period_breakdown(daily_all, trades_all, "M")

    daily_all.to_parquet(output / "account_target_exposure_daily.parquet", index=False)
    trades_all.to_parquet(output / "account_target_exposure_trades.parquet", index=False)
    positions_all.to_parquet(output / "account_target_exposure_positions.parquet", index=False)
    fold_metrics.to_csv(output / "account_target_exposure_fold_metrics.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "account_target_exposure_yearly.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "account_target_exposure_monthly.csv", index=False, encoding="utf-8-sig")
    write_report(output, fold_metrics, yearly, monthly)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "source_run": str(args.source_run),
                "predictions_file": args.predictions_file,
                "variant": args.variant,
                "data_version": version,
                "timing_daily": str(args.timing_daily),
                "initial_cash": args.initial_cash,
                "top_n_per_day": args.top_n,
                "holding_days": args.holding_days,
                "cost_bps": args.cost_bps,
                "lot_size": args.lot_size,
                "portfolio_rule": (
                    "Daily top-N new candidates, fixed holding period lots, no max total holding-count cap; "
                    "new-buy budget is capped by target_position * NAV - current gross exposure."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest sell-impact ranker with account-level timing target exposure."
    )
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--predictions-file", default=DEFAULT_PREDICTIONS_FILE)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--timing-daily", type=Path, default=timing_compare.TIMING_DAILY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--holding-days", type=int, default=DEFAULT_HOLDING_DAYS)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--lot-size", type=int, default=DEFAULT_LOT_SIZE)
    return parser.parse_args()


def load_predictions(source_run: Path, predictions_file: str, variant: str) -> pd.DataFrame:
    path = source_run / predictions_file
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {"trade_date", "ts_code", "score", "sample", "fold", "variant"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    frame = frame.loc[frame["variant"].eq(variant)].copy()
    if frame.empty:
        raise ValueError(f"variant not found in predictions: {variant}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def run_account_backtest(
    *,
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    timing: pd.Series,
    market_benchmark: pd.DataFrame,
    initial_cash: float,
    top_n: int,
    holding_days: int,
    lot_size: int,
    cost_bps: float,
    constraints: ExecutionConstraints,
    cost_model: CostModel,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data = panel.merge(
        predictions[["trade_date", "ts_code", "score"]].rename(columns={"score": "factor_value"}),
        on=["trade_date", "ts_code"],
        how="left",
    )
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    dates = list(pd.Index(data["trade_date"].unique()).sort_values())
    by_date = {date: frame.set_index("ts_code") for date, frame in data.groupby("trade_date")}
    mark_prices = mark_price_lookup(data)
    cash = float(initial_cash)
    positions: list[AccountPosition] = []
    trades: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    generated = executed = 0

    for date_index, date in enumerate(dates):
        today = by_date[date]
        generated_today = buys_today = sells_today = blocked_cash_today = blocked_trade_today = 0
        duplicate_signals_today = 0
        turnover_today = cost_today = 0.0

        remaining: list[AccountPosition] = []
        for position in positions:
            if position.due_date is None or date < position.due_date or not can_sell(today, position.ts_code, constraints):
                remaining.append(position)
                continue
            value = position_value(position, date, mark_prices)
            sell_cost = cost(value, "SELL", cost_model, cost_bps)
            cash += value - sell_cost
            executed += 1
            sells_today += 1
            turnover_today += value
            cost_today += sell_cost
            trades.append(trade_row(position, date, "SELL", value, sell_cost, today))
        positions = remaining

        nav_before_buy, gross_before_buy, position_values_before = account_value(cash, positions, date, mark_prices)
        target_position = float(np.clip(timing.get(date, 1.0), 0.0, 1.0))
        target_gross = nav_before_buy * target_position
        buy_budget = max(0.0, min(cash, target_gross - gross_before_buy))

        if date_index >= 1 and buy_budget > 0 and top_n > 0:
            signal_date = dates[date_index - 1]
            signals = by_date[signal_date]
            candidate_mask = (
                signals["is_liquid"].fillna(False).astype(bool)
                & signals["factor_value"].notna()
            )
            candidates = signals[candidate_mask].sort_values("factor_value", ascending=False).head(top_n)
            generated += len(candidates)
            generated_today = len(candidates)
            held_codes = {position.ts_code for position in positions}
            duplicate_signals_today = int(candidates.index.isin(held_codes).sum())
            per_name_budget = buy_budget / len(candidates) if len(candidates) else 0.0
            for ts_code in candidates.index:
                if not can_buy(today, ts_code, constraints):
                    blocked_trade_today += 1
                    continue
                row = today.loc[ts_code]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                price = float(row["raw_open"])
                shares = int(per_name_budget // (price * lot_size)) * lot_size
                if shares <= 0:
                    blocked_cash_today += 1
                    continue
                gross = shares * price
                buy_cost = cost(gross, "BUY", cost_model, cost_bps)
                if gross + buy_cost > cash:
                    blocked_cash_today += 1
                    continue
                cash -= gross + buy_cost
                due_index = date_index + holding_days
                position = AccountPosition(
                    ts_code=ts_code,
                    shares=shares,
                    entry_date=date,
                    due_date=dates[due_index] if due_index < len(dates) else None,
                    entry_raw_open=price,
                    entry_adj_open=float(row["adj_open"]),
                    signal_date=signal_date,
                    condition_quantile=None,
                )
                positions.append(position)
                executed += 1
                buys_today += 1
                turnover_today += gross
                cost_today += buy_cost
                trades.append(trade_row(position, date, "BUY", gross, buy_cost, today))

        nav, gross_exposure, position_values = account_value(cash, positions, date, mark_prices)
        aggregate_values: dict[str, float] = {}
        for code, value in position_values:
            aggregate_values[code] = aggregate_values.get(code, 0.0) + value
        unique_codes = set(aggregate_values)
        previous_nav = daily_rows[-1]["nav"] if daily_rows else initial_cash
        daily_rows.append(
            {
                "trade_date": date,
                "nav": nav,
                "gross_exposure": gross_exposure,
                "cash": cash,
                "target_position": target_position,
                "target_gross": target_position * nav,
                "pre_buy_gross_exposure": gross_before_buy,
                "pre_buy_buy_budget": buy_budget,
                "new_signals": generated_today,
                "duplicate_signals": duplicate_signals_today,
                "executed_buys": buys_today,
                "executed_sells": sells_today,
                "cash_blocked_candidates": blocked_cash_today,
                "trade_blocked_candidates": blocked_trade_today,
                "holding_count": len(positions),
                "unique_holding_count": len(unique_codes),
                "portfolio_turnover": turnover_today / previous_nav if previous_nav > 0 else 0.0,
                "cash_ratio": cash / nav if nav > 0 else 0.0,
                "gross_exposure_ratio": gross_exposure / nav if nav > 0 else 0.0,
                "largest_position_weight": (
                    max(aggregate_values.values()) / nav if aggregate_values and nav > 0 else 0.0
                ),
                "transaction_cost": cost_today,
            }
        )
        for position in positions:
            value = position_value(position, date, mark_prices)
            position_rows.append(
                {
                    "trade_date": date,
                    "ts_code": position.ts_code,
                    "shares": position.shares,
                    "market_value": value,
                    "entry_date": position.entry_date,
                    "due_date": position.due_date,
                    "signal_date": position.signal_date,
                }
            )

    daily = pd.DataFrame(daily_rows)
    daily["return"] = daily["nav"].pct_change().fillna(0.0)
    daily["market_index_return"] = market_index_returns(market_benchmark, dates)
    trades_frame = pd.DataFrame(trades)
    positions_frame = pd.DataFrame(position_rows)
    metrics = metrics_from_daily(daily, trades_frame, generated, executed)
    return daily, trades_frame, positions_frame, metrics


def mark_price_lookup(data: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], float]:
    ordered = data[["ts_code", "trade_date", "adj_open", "adj_close"]].sort_values(["ts_code", "trade_date"])
    base_price = ordered["adj_open"].fillna(ordered["adj_close"])
    ordered["mark"] = base_price.groupby(ordered["ts_code"]).ffill()
    return {(row.trade_date, row.ts_code): row.mark for row in ordered.itertuples()}


def position_value(position: AccountPosition, date: pd.Timestamp, prices: dict) -> float:
    mark = prices.get((date, position.ts_code), np.nan)
    if not np.isfinite(mark):
        mark = position.entry_adj_open
    return position.shares * position.entry_raw_open * mark / position.entry_adj_open


def account_value(
    cash: float,
    positions: list[AccountPosition],
    date: pd.Timestamp,
    prices: dict,
) -> tuple[float, float, list[tuple[str, float]]]:
    values = [(position.ts_code, position_value(position, date, prices)) for position in positions]
    gross = float(sum(value for _, value in values))
    return float(cash + gross), gross, values


def can_buy(today: pd.DataFrame, ts_code: str, constraints: ExecutionConstraints) -> bool:
    if ts_code not in today.index:
        return False
    row = today.loc[ts_code]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return bool(
        np.isfinite(row.get("raw_open", np.nan))
        and (not constraints.exclude_suspended or not row.get("is_suspended", True))
        and (not constraints.cannot_buy_limit_up or not row.get("is_limit_up_open", False))
        and (not constraints.exclude_st or not row.get("is_st", False))
        and (not constraints.exclude_delisting_period or not row.get("is_delisting_period", False))
        and row.get("listing_trade_days", 0) >= constraints.min_listing_days
    )


def can_sell(today: pd.DataFrame, ts_code: str, constraints: ExecutionConstraints) -> bool:
    if ts_code not in today.index:
        return False
    row = today.loc[ts_code]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return bool(
        np.isfinite(row.get("raw_open", np.nan))
        and (not constraints.exclude_suspended or not row.get("is_suspended", True))
        and (not constraints.cannot_sell_limit_down or not row.get("is_limit_down_open", False))
    )


def cost(value: float, side: str, model: CostModel, scenario_bps: float | None) -> float:
    if scenario_bps is not None:
        return value * (scenario_bps / 2.0) / 10_000.0
    bps = model.commission_bps_per_side + model.slippage_bps_per_side
    if side == "SELL":
        bps += model.stamp_duty_bps_sell
    return value * bps / 10_000.0


def trade_row(position: AccountPosition, date: pd.Timestamp, side: str, gross: float, trade_cost: float, today: pd.DataFrame) -> dict:
    row = today.loc[position.ts_code]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return {
        "trade_date": date,
        "signal_date": position.signal_date,
        "entry_date": position.entry_date,
        "due_date": position.due_date,
        "ts_code": position.ts_code,
        "side": side,
        "shares": position.shares,
        "raw_open": row.get("raw_open"),
        "gross_value": gross,
        "cost": trade_cost,
        "condition_quantile": position.condition_quantile,
    }


def market_index_returns(benchmark: pd.DataFrame, dates: list[pd.Timestamp]) -> list[float]:
    frame = benchmark.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.sort_values("trade_date")
    frame["return"] = frame["open"].pct_change()
    values = frame.set_index("trade_date")["return"]
    return [float(values.get(date, 0.0) or 0.0) for date in dates]


def metrics_from_daily(daily: pd.DataFrame, trades: pd.DataFrame, generated: int, executed: int) -> dict[str, Any]:
    count = max(len(daily) - 1, 1)
    total = daily["nav"].iloc[-1] / daily["nav"].iloc[0] - 1 if len(daily) else 0.0
    annual = (1 + total) ** (252 / count) - 1 if total > -1 else -1.0
    returns = daily["return"]
    volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
    drawdown = daily["nav"] / daily["nav"].cummax() - 1
    index_total = float((1 + daily["market_index_return"]).prod() - 1)
    buy_count = int((trades["side"] == "BUY").sum()) if not trades.empty else 0
    sell_count = int((trades["side"] == "SELL").sum()) if not trades.empty else 0
    return {
        "total_return": float(total),
        "annualized_return": float(annual),
        "annualized_volatility": volatility,
        "sharpe": float(annual / volatility) if volatility > 0 else np.nan,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "calmar": float(annual / abs(drawdown.min())) if len(drawdown) and drawdown.min() < 0 else np.nan,
        "market_index_annualized_return": (
            (1 + index_total) ** (252 / count) - 1 if index_total > -1 else -1.0
        ),
        "generated_signals": int(generated),
        "executed_buys": buy_count,
        "executed_sells": sell_count,
        "execution_rate": float(buy_count / generated) if generated else 0.0,
        "turnover_notional": float(trades["gross_value"].sum()) if not trades.empty else 0.0,
        "annualized_turnover": float(daily["portfolio_turnover"].mean() * 252),
        "avg_target_position": float(daily["target_position"].mean()),
        "avg_gross_exposure_ratio": float(daily["gross_exposure_ratio"].mean()),
        "avg_cash_ratio": float(daily["cash_ratio"].mean()),
        "avg_holding_count": float(daily["holding_count"].mean()),
        "max_holding_count": int(daily["holding_count"].max()) if len(daily) else 0,
        "avg_unique_holding_count": float(daily["unique_holding_count"].mean()),
        "max_unique_holding_count": int(daily["unique_holding_count"].max()) if len(daily) else 0,
        "avg_largest_position_weight": float(daily["largest_position_weight"].mean()),
        "max_largest_position_weight": float(daily["largest_position_weight"].max()),
        "cash_blocked_candidates": int(daily["cash_blocked_candidates"].sum()),
        "trade_blocked_candidates": int(daily["trade_blocked_candidates"].sum()),
    }


def period_breakdown(daily: pd.DataFrame, trades: pd.DataFrame, freq: str) -> pd.DataFrame:
    frame = daily.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["period"] = frame["trade_date"].dt.to_period(freq).astype(str)
    trade_frame = trades.copy()
    if not trade_frame.empty:
        trade_frame["trade_date"] = pd.to_datetime(trade_frame["trade_date"])
        trade_frame["period"] = trade_frame["trade_date"].dt.to_period(freq).astype(str)
    rows = []
    for (fold, period), group in frame.groupby(["fold", "period"], sort=True):
        group = group.sort_values("trade_date")
        start_nav = float(group["nav"].iloc[0])
        end_nav = float(group["nav"].iloc[-1])
        period_return = end_nav / start_nav - 1 if start_nav > 0 else 0.0
        count = max(len(group) - 1, 1)
        annualized = (1 + period_return) ** (252 / count) - 1 if period_return > -1 else -1.0
        interval_curve = group["nav"] / start_nav
        interval_drawdown = interval_curve / interval_curve.cummax() - 1
        period_trades = (
            trade_frame.loc[(trade_frame["fold"].eq(fold)) & (trade_frame["period"].eq(period))]
            if not trade_frame.empty
            else pd.DataFrame()
        )
        rows.append(
            {
                "fold": fold,
                "period": period,
                "start_date": group["trade_date"].iloc[0].date().isoformat(),
                "end_date": group["trade_date"].iloc[-1].date().isoformat(),
                "trading_days": len(group),
                "period_return": float(period_return),
                "annualized_return": float(annualized),
                "interval_max_drawdown": float(interval_drawdown.min()) if len(interval_drawdown) else 0.0,
                "trade_count": int(len(period_trades)),
                "buy_count": int((period_trades["side"] == "BUY").sum()) if not period_trades.empty else 0,
                "sell_count": int((period_trades["side"] == "SELL").sum()) if not period_trades.empty else 0,
                "generated_signals": int(group["new_signals"].sum()),
                "execution_rate": (
                    float(group["executed_buys"].sum() / group["new_signals"].sum())
                    if group["new_signals"].sum() > 0
                    else 0.0
                ),
                "avg_target_position": float(group["target_position"].mean()),
                "avg_gross_exposure_ratio": float(group["gross_exposure_ratio"].mean()),
                "avg_holding_count": float(group["holding_count"].mean()),
                "max_holding_count": int(group["holding_count"].max()),
                "avg_unique_holding_count": float(group["unique_holding_count"].mean()),
                "max_unique_holding_count": int(group["unique_holding_count"].max()),
            }
        )
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(output: Path, fold_metrics: pd.DataFrame, yearly: pd.DataFrame, monthly: pd.DataFrame) -> None:
    lines = [
        "# Account Target Exposure Backtest",
        "",
        "## Portfolio Rule",
        "- Signal timing: T close signal, T+1 open execution.",
        "- New buys: daily top-N candidates only.",
        "- Holding: fixed 10 trading-day lots by default.",
        "- Total holdings: no Top5 account cap.",
        "- Timing: `target_position` caps new-buy budget at account level; existing positions are not force-sold.",
        "",
        "## Fold Metrics",
        md_table(fold_metrics, 20),
        "",
        "## Yearly Breakdown",
        md_table(yearly, 40),
        "",
        "## Monthly Breakdown",
        md_table(monthly, 80),
        "",
        "## Files",
        "- `account_target_exposure_daily.parquet`",
        "- `account_target_exposure_trades.parquet`",
        "- `account_target_exposure_positions.parquet`",
        "- `account_target_exposure_fold_metrics.csv`",
        "- `account_target_exposure_yearly.csv`",
        "- `account_target_exposure_monthly.csv`",
    ]
    (output / "account_target_exposure_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
