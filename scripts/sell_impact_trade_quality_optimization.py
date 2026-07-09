from __future__ import annotations

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

import sell_impact_account_target_exposure_backtest as acct
import sell_impact_ranker_timing_compare as timing_compare
import sell_impact_recent_halfyear_tactical as tactical
import sell_impact_sorting_repair as base
from factor_forge.config import CostModel, ExecutionConstraints


SOURCE_RUN = Path("artifacts/strategy_reviews/sell_impact_recent_halfyear_tactical_20260709T100107Z")
OUTPUT_ROOT = Path("artifacts/strategy_reviews")
MODEL = "K_recent_2024_2025q3"
TEST_START = "20260101"
TEST_END = "20260623"
INITIAL_CASH = 1_000_000.0
LOT_SIZE = 100
COST_BPS = 20.0
BAND_TARGET = 0.95

VARIANTS = [
    {
        "variant": "B_only_fill_fixed10_max5",
        "description": "max 5 account holdings; no active replacement; sell after 10 trading days; fill empty slots only",
        "max_positions": 5,
        "entry_pool": "top_band",
        "entry_band_rank_min": 0.0,
        "entry_raw_rank_min": 0.0,
        "min_hold_days": 1,
        "max_hold_days": 10,
        "sell_rule": "fixed",
    },
    {
        "variant": "A_hold_continue_max5",
        "description": "max 5 account holdings; continue if band rank still top 20% or raw score above median",
        "max_positions": 5,
        "entry_pool": "top_band",
        "entry_band_rank_min": 0.0,
        "entry_raw_rank_min": 0.0,
        "min_hold_days": 3,
        "max_hold_days": 20,
        "sell_rule": "continue",
        "continue_band_rank_min": 0.80,
        "continue_raw_rank_min": 0.50,
    },
    {
        "variant": "C_high_threshold_low_turnover_max5",
        "description": "max 5 account holdings; only buy top 5% band-quality candidates with simple risk filters; allow cash",
        "max_positions": 5,
        "entry_pool": "threshold",
        "entry_band_rank_min": 0.95,
        "entry_raw_rank_min": 0.65,
        "min_hold_days": 3,
        "max_hold_days": 20,
        "sell_rule": "continue",
        "continue_band_rank_min": 0.85,
        "continue_raw_rank_min": 0.55,
        "max_microcap_score": 1.0,
        "min_liquidity": -1.0,
        "min_price_reversal": -0.5,
    },
    {
        "variant": "A_hold_continue_max8",
        "description": "max 8 account holdings; continue if band rank still top 20% or raw score above median",
        "max_positions": 8,
        "entry_pool": "top_band",
        "entry_band_rank_min": 0.0,
        "entry_raw_rank_min": 0.0,
        "min_hold_days": 3,
        "max_hold_days": 20,
        "sell_rule": "continue",
        "continue_band_rank_min": 0.80,
        "continue_raw_rank_min": 0.50,
    },
    {
        "variant": "C_high_threshold_low_turnover_max8",
        "description": "max 8 account holdings; only buy top 5% band-quality candidates with simple risk filters; allow cash",
        "max_positions": 8,
        "entry_pool": "threshold",
        "entry_band_rank_min": 0.95,
        "entry_raw_rank_min": 0.65,
        "min_hold_days": 3,
        "max_hold_days": 20,
        "sell_rule": "continue",
        "continue_band_rank_min": 0.85,
        "continue_raw_rank_min": 0.55,
        "max_microcap_score": 1.0,
        "min_liquidity": -1.0,
        "min_price_reversal": -0.5,
    },
]


@dataclass
class Position:
    ts_code: str
    shares: int
    entry_date: pd.Timestamp
    entry_index: int
    entry_raw_open: float
    entry_adj_open: float
    signal_date: pd.Timestamp
    entry_band_rank_pct: float
    entry_raw_rank_pct: float


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_trade_quality_optimization_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"source_run={SOURCE_RUN}")
    signals = load_signals()
    signals.to_parquet(output / "trade_quality_signals.parquet", index=False)
    log(f"signals rows={len(signals):,} dates={signals['trade_date'].nunique():,}")

    version, panel = base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel.loc[panel["trade_date"].between(pd.Timestamp(TEST_START), pd.Timestamp(TEST_END))].copy()
    timing = timing_compare.load_position_multiplier(timing_compare.TIMING_DAILY)
    market_benchmark = timing_compare.load_market_benchmark(version)
    constraints = ExecutionConstraints(
        exclude_suspended=True,
        cannot_buy_limit_up=True,
        cannot_sell_limit_down=True,
        exclude_st=True,
        exclude_delisting_period=True,
        min_listing_days=60,
    )
    cost_model = CostModel(commission_bps_per_side=3, slippage_bps_per_side=5, stamp_duty_bps_sell=5)

    metric_rows: list[dict[str, Any]] = []
    monthly_rows: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []

    for cfg in VARIANTS:
        log(f"running {cfg['variant']}: {cfg['description']}")
        daily, trades, positions, metrics = run_trade_quality_backtest(
            panel=panel,
            signals=signals,
            timing=timing,
            market_benchmark=market_benchmark,
            constraints=constraints,
            cost_model=cost_model,
            cfg=cfg,
        )
        csi1000 = float(metrics.get("market_index_annualized_return", np.nan))
        row = {
            "variant": cfg["variant"],
            "description": cfg["description"],
            **metrics,
            "annualized_excess_return_vs_csi1000": float(metrics["annualized_return"] - csi1000),
        }
        metric_rows.append(row)
        daily["variant"] = cfg["variant"]
        trades["variant"] = cfg["variant"]
        positions["variant"] = cfg["variant"]
        daily_frames.append(daily)
        trade_frames.append(trades)
        position_frames.append(positions)
        monthly = period_breakdown(daily, trades, cfg["variant"], "M")
        monthly_rows.append(monthly)
        log(
            f"{cfg['variant']}: ann={metrics['annualized_return']:.2%} "
            f"excess_csi1000={row['annualized_excess_return_vs_csi1000']:.2%} "
            f"mdd={metrics['max_drawdown']:.2%} buys={metrics['executed_buys']} "
            f"trades={metrics['trade_count']} avg_holdings={metrics['avg_unique_holding_count']:.2f}"
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values("annualized_return", ascending=False)
    monthly_df = pd.concat(monthly_rows, ignore_index=True)
    daily_all = pd.concat(daily_frames, ignore_index=True)
    trades_all = pd.concat(trade_frames, ignore_index=True)
    positions_all = pd.concat(position_frames, ignore_index=True)
    trade_quality = trade_quality_summary(trades_all, panel)

    metrics_df.to_csv(output / "trade_quality_variant_metrics.csv", index=False, encoding="utf-8-sig")
    monthly_df.to_csv(output / "trade_quality_monthly.csv", index=False, encoding="utf-8-sig")
    trade_quality.to_csv(output / "trade_quality_single_trade_summary.csv", index=False, encoding="utf-8-sig")
    daily_all.to_parquet(output / "trade_quality_daily.parquet", index=False)
    trades_all.to_parquet(output / "trade_quality_trades.parquet", index=False)
    positions_all.to_parquet(output / "trade_quality_positions.parquet", index=False)
    write_report(output, metrics_df, monthly_df, trade_quality)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "source_run": str(SOURCE_RUN),
                "model": MODEL,
                "band_target": BAND_TARGET,
                "test_window": [TEST_START, TEST_END],
                "initial_cash": INITIAL_CASH,
                "cost_bps": COST_BPS,
                "lot_size": LOT_SIZE,
                "timing_daily": str(timing_compare.TIMING_DAILY),
                "data_version": version,
                "variants": VARIANTS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def load_signals() -> pd.DataFrame:
    predictions = pd.read_parquet(SOURCE_RUN / "test_2026h1_predictions.parquet")
    predictions["trade_date"] = pd.to_datetime(predictions["trade_date"])
    predictions = predictions.loc[predictions["model"].eq(MODEL)].copy()
    predictions = predictions.rename(columns={"score": "raw_score"})
    predictions["raw_rank_pct"] = predictions.groupby("trade_date")["raw_score"].rank(pct=True, method="first")
    predictions["band_score"] = -(predictions["raw_rank_pct"] - BAND_TARGET).abs()
    predictions["band_rank_pct"] = predictions.groupby("trade_date")["band_score"].rank(pct=True, method="first")
    dataset = pd.read_parquet(SOURCE_RUN / "recent_halfyear_dataset.parquet")
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    cols = [
        "trade_date",
        "ts_code",
        "condition_quantile",
        "stock_state_small_size",
        "cluster_liquidity",
        "cluster_price_reversal",
        "stock_state_low_vol",
        "log_amount_20_z",
        "amount_chg_5_20_z",
    ]
    cols = [col for col in cols if col in dataset.columns]
    signals = predictions.merge(dataset[cols], on=["trade_date", "ts_code"], how="left")
    return signals.replace([np.inf, -np.inf], np.nan)


def run_trade_quality_backtest(
    *,
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    timing: pd.Series,
    market_benchmark: pd.DataFrame,
    constraints: ExecutionConstraints,
    cost_model: CostModel,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data = panel.merge(signals, on=["trade_date", "ts_code"], how="left")
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    dates = list(pd.Index(data["trade_date"].unique()).sort_values())
    by_date = {date: frame.set_index("ts_code") for date, frame in data.groupby("trade_date")}
    mark_prices = acct.mark_price_lookup(data)

    cash = INITIAL_CASH
    positions: list[Position] = []
    trades: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    generated = 0

    for date_index, date in enumerate(dates):
        today = by_date[date]
        signal_date = dates[date_index - 1] if date_index >= 1 else None
        signal_frame = by_date[signal_date] if signal_date is not None else None
        sells_today = buys_today = generated_today = blocked_trade_today = blocked_cash_today = 0
        turnover_today = cost_today = 0.0

        remaining: list[Position] = []
        for position in positions:
            sell_reason = sell_decision(position, date_index, signal_frame, cfg)
            if sell_reason is None or not acct.can_sell(today, position.ts_code, constraints):
                remaining.append(position)
                continue
            value = acct.position_value(to_account_position(position), date, mark_prices)
            sell_cost = acct.cost(value, "SELL", cost_model, COST_BPS)
            cash += value - sell_cost
            sells_today += 1
            turnover_today += value
            cost_today += sell_cost
            trades.append(trade_row(position, date, "SELL", value, sell_cost, today, sell_reason))
        positions = remaining

        nav_before, gross_before, _ = account_value(cash, positions, date, mark_prices)
        target_position = float(np.clip(timing.get(date, 1.0), 0.0, 1.0))
        target_gross = nav_before * target_position
        slot_count = int(cfg["max_positions"])
        vacancy = max(0, slot_count - len(positions))

        if signal_frame is not None and vacancy > 0:
            candidates = entry_candidates(signal_frame, positions, cfg)
            generated_today = int(len(candidates))
            generated += generated_today
            if not candidates.empty:
                buy_budget = max(0.0, min(cash, target_gross - gross_before))
                target_slot_value = target_gross / slot_count if slot_count else 0.0
                per_buy_budget = min(target_slot_value, buy_budget / vacancy) if vacancy else 0.0
                selected = candidates.head(vacancy)
                for ts_code, signal_row in selected.iterrows():
                    if not acct.can_buy(today, ts_code, constraints):
                        blocked_trade_today += 1
                        continue
                    row = today.loc[ts_code]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    price = float(row["raw_open"])
                    shares = int(per_buy_budget // (price * LOT_SIZE)) * LOT_SIZE
                    if shares <= 0:
                        blocked_cash_today += 1
                        continue
                    gross = shares * price
                    buy_cost = acct.cost(gross, "BUY", cost_model, COST_BPS)
                    if gross + buy_cost > cash:
                        blocked_cash_today += 1
                        continue
                    position = Position(
                        ts_code=str(ts_code),
                        shares=shares,
                        entry_date=date,
                        entry_index=date_index,
                        entry_raw_open=price,
                        entry_adj_open=float(row["adj_open"]),
                        signal_date=signal_date,
                        entry_band_rank_pct=float(signal_row.get("band_rank_pct", np.nan)),
                        entry_raw_rank_pct=float(signal_row.get("raw_rank_pct", np.nan)),
                    )
                    cash -= gross + buy_cost
                    buys_today += 1
                    turnover_today += gross
                    cost_today += buy_cost
                    positions.append(position)
                    trades.append(trade_row(position, date, "BUY", gross, buy_cost, today, "entry"))

        nav, gross_exposure, position_values = account_value(cash, positions, date, mark_prices)
        aggregate_values = {code: value for code, value in position_values}
        previous_nav = daily_rows[-1]["nav"] if daily_rows else INITIAL_CASH
        daily_rows.append(
            {
                "trade_date": date,
                "nav": nav,
                "gross_exposure": gross_exposure,
                "cash": cash,
                "target_position": target_position,
                "new_signals": generated_today,
                "executed_buys": buys_today,
                "executed_sells": sells_today,
                "cash_blocked_candidates": blocked_cash_today,
                "trade_blocked_candidates": blocked_trade_today,
                "holding_count": len(positions),
                "unique_holding_count": len({position.ts_code for position in positions}),
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
            value = acct.position_value(to_account_position(position), date, mark_prices)
            position_rows.append(
                {
                    "trade_date": date,
                    "ts_code": position.ts_code,
                    "shares": position.shares,
                    "market_value": value,
                    "entry_date": position.entry_date,
                    "signal_date": position.signal_date,
                    "holding_trade_days": date_index - position.entry_index,
                    "entry_band_rank_pct": position.entry_band_rank_pct,
                    "entry_raw_rank_pct": position.entry_raw_rank_pct,
                }
            )

    daily = pd.DataFrame(daily_rows)
    daily["return"] = daily["nav"].pct_change().fillna(0.0)
    daily["market_index_return"] = acct.market_index_returns(market_benchmark, dates)
    trades_frame = pd.DataFrame(trades)
    positions_frame = pd.DataFrame(position_rows)
    metrics = metrics_from_daily(daily, trades_frame, generated)
    return daily, trades_frame, positions_frame, metrics


def sell_decision(position: Position, date_index: int, signal_frame: pd.DataFrame | None, cfg: dict[str, Any]) -> str | None:
    holding_days = date_index - position.entry_index
    if holding_days >= int(cfg["max_hold_days"]):
        return "max_hold"
    if str(cfg["sell_rule"]) == "fixed":
        return None
    if holding_days < int(cfg["min_hold_days"]):
        return None
    if signal_frame is None or position.ts_code not in signal_frame.index:
        return None
    row = signal_frame.loc[position.ts_code]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    band_rank = float(row.get("band_rank_pct", np.nan))
    raw_rank = float(row.get("raw_rank_pct", np.nan))
    keep = (
        np.isfinite(band_rank)
        and band_rank >= float(cfg.get("continue_band_rank_min", 0.80))
    ) or (
        np.isfinite(raw_rank)
        and raw_rank >= float(cfg.get("continue_raw_rank_min", 0.50))
    )
    return None if keep else "signal_deteriorated"


def entry_candidates(signal_frame: pd.DataFrame, positions: list[Position], cfg: dict[str, Any]) -> pd.DataFrame:
    held = {position.ts_code for position in positions}
    frame = signal_frame.loc[signal_frame["band_score"].notna()].copy()
    if held:
        frame = frame.loc[~frame.index.isin(held)].copy()
    frame = frame.loc[
        frame["band_rank_pct"].ge(float(cfg.get("entry_band_rank_min", 0.0)))
        & frame["raw_rank_pct"].ge(float(cfg.get("entry_raw_rank_min", 0.0)))
    ].copy()
    if cfg.get("entry_pool") == "threshold":
        if "stock_state_small_size" in frame:
            frame = frame.loc[frame["stock_state_small_size"].le(float(cfg.get("max_microcap_score", np.inf)))]
        if "cluster_liquidity" in frame:
            frame = frame.loc[frame["cluster_liquidity"].ge(float(cfg.get("min_liquidity", -np.inf)))]
        if "cluster_price_reversal" in frame:
            frame = frame.loc[frame["cluster_price_reversal"].ge(float(cfg.get("min_price_reversal", -np.inf)))]
    return frame.sort_values(["band_score", "raw_score"], ascending=False)


def to_account_position(position: Position) -> acct.AccountPosition:
    return acct.AccountPosition(
        ts_code=position.ts_code,
        shares=position.shares,
        entry_date=position.entry_date,
        due_date=None,
        entry_raw_open=position.entry_raw_open,
        entry_adj_open=position.entry_adj_open,
        signal_date=position.signal_date,
    )


def account_value(
    cash: float,
    positions: list[Position],
    date: pd.Timestamp,
    prices: dict[tuple[pd.Timestamp, str], float],
) -> tuple[float, float, list[tuple[str, float]]]:
    values = [(position.ts_code, acct.position_value(to_account_position(position), date, prices)) for position in positions]
    gross = float(sum(value for _, value in values))
    return float(cash + gross), gross, values


def trade_row(
    position: Position,
    date: pd.Timestamp,
    side: str,
    gross: float,
    trade_cost: float,
    today: pd.DataFrame,
    reason: str,
) -> dict[str, Any]:
    row = today.loc[position.ts_code]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return {
        "trade_date": date,
        "signal_date": position.signal_date,
        "entry_date": position.entry_date,
        "ts_code": position.ts_code,
        "side": side,
        "shares": position.shares,
        "raw_open": row.get("raw_open"),
        "gross_value": gross,
        "cost": trade_cost,
        "reason": reason,
        "holding_trade_days": np.nan if side == "BUY" else None,
        "entry_band_rank_pct": position.entry_band_rank_pct,
        "entry_raw_rank_pct": position.entry_raw_rank_pct,
    }


def metrics_from_daily(daily: pd.DataFrame, trades: pd.DataFrame, generated: int) -> dict[str, Any]:
    count = max(len(daily) - 1, 1)
    total = daily["nav"].iloc[-1] / daily["nav"].iloc[0] - 1 if len(daily) else 0.0
    annual = (1.0 + total) ** (252 / count) - 1.0 if total > -1 else -1.0
    returns = daily["return"]
    volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
    drawdown = daily["nav"] / daily["nav"].cummax() - 1
    index_total = float((1.0 + daily["market_index_return"]).prod() - 1.0)
    buy_count = int(trades["side"].eq("BUY").sum()) if not trades.empty else 0
    sell_count = int(trades["side"].eq("SELL").sum()) if not trades.empty else 0
    return {
        "total_return": float(total),
        "annualized_return": float(annual),
        "annualized_volatility": volatility,
        "sharpe": float(annual / volatility) if volatility > 0 else np.nan,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "calmar": float(annual / abs(drawdown.min())) if len(drawdown) and drawdown.min() < 0 else np.nan,
        "market_index_annualized_return": (
            (1.0 + index_total) ** (252 / count) - 1.0 if index_total > -1 else -1.0
        ),
        "generated_signals": int(generated),
        "executed_buys": buy_count,
        "executed_sells": sell_count,
        "trade_count": int(len(trades)),
        "execution_rate": float(buy_count / generated) if generated else 0.0,
        "turnover_notional": float(trades["gross_value"].sum()) if not trades.empty else 0.0,
        "annualized_turnover": float(daily["portfolio_turnover"].mean() * 252),
        "avg_target_position": float(daily["target_position"].mean()),
        "avg_gross_exposure_ratio": float(daily["gross_exposure_ratio"].mean()),
        "avg_cash_ratio": float(daily["cash_ratio"].mean()),
        "avg_unique_holding_count": float(daily["unique_holding_count"].mean()),
        "max_unique_holding_count": int(daily["unique_holding_count"].max()) if len(daily) else 0,
        "avg_largest_position_weight": float(daily["largest_position_weight"].mean()),
        "max_largest_position_weight": float(daily["largest_position_weight"].max()),
        "top_stock_buy_share": timing_compare.stock_buy_share(trades),
        "top_month_return_share": timing_compare.month_return_share(daily),
    }


def period_breakdown(daily: pd.DataFrame, trades: pd.DataFrame, variant: str, freq: str) -> pd.DataFrame:
    frame = daily.copy()
    frame["period"] = pd.to_datetime(frame["trade_date"]).dt.to_period(freq).astype(str)
    trade_frame = trades.copy()
    if not trade_frame.empty:
        trade_frame["period"] = pd.to_datetime(trade_frame["trade_date"]).dt.to_period(freq).astype(str)
    rows = []
    for period, group in frame.groupby("period", sort=True):
        group = group.sort_values("trade_date")
        curve = (1.0 + group["return"]).cumprod()
        period_trades = trade_frame.loc[trade_frame["period"].eq(period)] if not trade_frame.empty else pd.DataFrame()
        rows.append(
            {
                "variant": variant,
                "period": period,
                "trading_days": int(len(group)),
                "period_return": float((1.0 + group["return"]).prod() - 1.0),
                "interval_max_drawdown": float((curve / curve.cummax() - 1.0).min()),
                "trade_count": int(len(period_trades)),
                "buy_count": int(period_trades["side"].eq("BUY").sum()) if not period_trades.empty else 0,
                "sell_count": int(period_trades["side"].eq("SELL").sum()) if not period_trades.empty else 0,
                "avg_gross_exposure_ratio": float(group["gross_exposure_ratio"].mean()),
                "avg_unique_holding_count": float(group["unique_holding_count"].mean()),
            }
        )
    return pd.DataFrame(rows)


def trade_quality_summary(trades: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    buys = trades.loc[trades["side"].eq("BUY")].copy()
    sells = trades.loc[trades["side"].eq("SELL")].copy()
    if buys.empty or sells.empty:
        return pd.DataFrame()
    pairs = buys.merge(
        sells,
        on=["variant", "ts_code", "entry_date"],
        suffixes=("_buy", "_sell"),
        how="inner",
    )
    if pairs.empty:
        return pd.DataFrame()
    pairs["trade_return_net"] = (
        (pairs["gross_value_sell"] - pairs["cost_sell"]) / (pairs["gross_value_buy"] + pairs["cost_buy"]) - 1.0
    )
    pairs["holding_days"] = (
        pd.to_datetime(pairs["trade_date_sell"]) - pd.to_datetime(pairs["trade_date_buy"])
    ).dt.days
    rows = []
    for variant, group in pairs.groupby("variant"):
        wins = group.loc[group["trade_return_net"].gt(0), "trade_return_net"]
        losses = group.loc[group["trade_return_net"].le(0), "trade_return_net"]
        rows.append(
            {
                "variant": variant,
                "round_trips": int(len(group)),
                "mean_trade_return": float(group["trade_return_net"].mean()),
                "median_trade_return": float(group["trade_return_net"].median()),
                "win_rate": float(group["trade_return_net"].gt(0).mean()),
                "avg_win": float(wins.mean()) if len(wins) else np.nan,
                "avg_loss": float(losses.mean()) if len(losses) else np.nan,
                "payoff_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() < 0 else np.nan,
                "avg_holding_calendar_days": float(group["holding_days"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_trade_return", ascending=False)


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(output: Path, metrics: pd.DataFrame, monthly: pd.DataFrame, trade_quality: pd.DataFrame) -> None:
    lines = [
        "# Sell-impact Trade Quality Optimization",
        "",
        "## Scope",
        f"- Source tactical run: `{SOURCE_RUN}`",
        f"- Model: `{MODEL}`, score-band target `{BAND_TARGET}`.",
        "- Account-level execution: max total holdings, T close signal, T+1 open execution.",
        "- Existing holdings are not force-rebalanced; variants differ only in sell/entry quality rules.",
        "- This is a turnover and trade-quality experiment, not a new alpha training run.",
        "",
        "## Variant Metrics",
        md_table(metrics, 20),
        "",
        "## Monthly Breakdown",
        md_table(monthly, 120),
        "",
        "## Single Trade Quality",
        md_table(trade_quality, 40),
        "",
        "## Files",
        "- `trade_quality_variant_metrics.csv`",
        "- `trade_quality_monthly.csv`",
        "- `trade_quality_single_trade_summary.csv`",
        "- `trade_quality_daily.parquet`",
        "- `trade_quality_trades.parquet`",
        "- `trade_quality_positions.parquet`",
    ]
    (output / "trade_quality_optimization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
