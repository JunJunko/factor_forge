from __future__ import annotations

"""Experiment 15: train-free validation of clean PIT rank competition.

The experiment intentionally imports only the execution-safe portion of
Experiment 14 Clean Stage 1.  No reliability, holding-health, residual-alpha
or forward-label feature enters a decision.  Every parameter is pre-registered
in this file and all decisions use an Alpha rank available at signal close.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sell_impact_position_competition_clean as clean
import sell_impact_trade_quality_optimization as tq
from factor_forge.config import CostModel, ExecutionConstraints


OUTPUT_ROOT = Path("artifacts/strategy_reviews/experiment15_clean_competition")
MARGINS = (0.05, 0.10, 0.15)
COST_BPS = (20.0, 50.0, 100.0)
DELAYS = (0, 1, 2)
HOLD_DAYS = (5, 10, 15, 20)
BASELINE = {"rule": "baseline", "margin": 0.0, "name": "baseline_fixed_top5_hold10"}
RANK_RULE = "raw_rank"
PLACEBO_RULE = "random_placebo"


def main() -> None:
    output = OUTPUT_ROOT / f"clean_competition_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    log_path = output / "run.log"

    def log(message: str) -> None:
        line = f"[{time.time() - started:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("loading frozen PIT raw Alpha ranks; Reliability and Holding Health are disabled")
    signals = clean.load_pit_signals()
    version, full_panel = tq.base.load_panel()
    full_panel["trade_date"] = pd.to_datetime(full_panel["trade_date"])
    panel = full_panel.loc[
        full_panel["trade_date"].between(clean.TEST_START, clean.TEST_END)
        & full_panel["ts_code"].isin(signals["ts_code"].unique())
    ].copy()
    del full_panel
    constraints = ExecutionConstraints(
        exclude_suspended=True,
        cannot_buy_limit_up=True,
        cannot_sell_limit_down=True,
        exclude_st=True,
        exclude_delisting_period=True,
        min_listing_days=60,
    )
    log(f"data_version={version}; PIT score rows={len(signals):,}; price rows={len(panel):,}")

    cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def execute(name: str, rule: str, margin: float, cost_bps: float = 20.0, delay: int = 0, hold_days: int = 10, forced_dates: set[pd.Timestamp] | None = None) -> dict[str, Any]:
        key = (rule, margin, cost_bps, delay, hold_days, tuple(sorted(forced_dates)) if forced_dates else None)
        if key in cache:
            cached = cache[key].copy()
            cached["name"] = name
            return cached
        log(f"run {name}: rule={rule}, margin={margin:.2f}, cost={cost_bps:.0f}, delay={delay}, hold={hold_days}")
        cost_model = CostModel(commission_bps_per_side=3, slippage_bps_per_side=5, stamp_duty_bps_sell=5)
        daily, trades, states, metrics = clean.run_backtest(
            panel, signals, constraints, cost_model, rule, margin,
            cost_bps=cost_bps, execution_delay=delay, max_hold_days=hold_days,
            forced_replacement_dates=forced_dates,
        )
        record = {"name": name, "rule": rule, "margin": margin, "cost_bps": cost_bps, "delay": delay, "hold_days": hold_days, "daily": daily, "trades": trades, "states": states, "metrics": metrics}
        cache[key] = record
        return record.copy()

    # Core frozen ablation.  Margin values are all reported; none is selected.
    core = [execute(BASELINE["name"], "baseline", 0.0)]
    for margin in MARGINS:
        core.append(execute(f"rank_competition_m{int(margin * 100):02d}", RANK_RULE, margin))
    reference = next(item for item in core if item["name"] == "rank_competition_m10")
    real_dates = replacement_dates(reference["trades"])
    placebo = execute("random_replacement_matched_m10", PLACEBO_RULE, 0.10, forced_dates=real_dates)
    core.append(placebo)
    log(f"matched placebo replacement schedule contains {len(real_dates)} dates")

    # A: Cost stress.  The fixed 0.10 margin is the pre-declared reference only,
    # while all three margins remain visible at the base 20bps core test.
    cost_runs = []
    for cost in COST_BPS:
        cost_runs.append(execute(f"cost{int(cost)}_baseline", "baseline", 0.0, cost_bps=cost))
        cost_runs.append(execute(f"cost{int(cost)}_rank_m10", RANK_RULE, 0.10, cost_bps=cost))

    # B: Signal-to-fill delay.  Each run uses the score observed before its fill.
    delay_runs = []
    for delay in DELAYS:
        delay_runs.append(execute(f"delay{delay}_baseline", "baseline", 0.0, delay=delay))
        delay_runs.append(execute(f"delay{delay}_rank_m10", RANK_RULE, 0.10, delay=delay))

    # C: Holding-period sensitivity.  No period is promoted by this experiment.
    hold_runs = []
    for hold in HOLD_DAYS:
        hold_runs.append(execute(f"hold{hold}_baseline", "baseline", 0.0, hold_days=hold))
        hold_runs.append(execute(f"hold{hold}_rank_m10", RANK_RULE, 0.10, hold_days=hold))

    all_runs = core + cost_runs + delay_runs + hold_runs
    core_summary = summary_frame(core)
    cost_summary = summary_frame(cost_runs)
    delay_summary = summary_frame(delay_runs)
    hold_summary = summary_frame(hold_runs)
    placebo_summary = summary_frame([core[0], reference, placebo])
    period = pd.concat([period_results(item) for item in all_runs], ignore_index=True)
    quality = pd.concat([replacement_quality(item, panel, signals) for item in all_runs], ignore_index=True)
    nav = pd.concat([tag_frame(item["daily"], item["name"]) for item in all_runs], ignore_index=True)
    trades = pd.concat([tag_frame(item["trades"], item["name"]) for item in all_runs], ignore_index=True)
    for filename, frame in {
        "portfolio_results.csv": core_summary,
        "cost_stress.csv": cost_summary,
        "delay_test.csv": delay_summary,
        "holding_period_test.csv": hold_summary,
        "replacement_quality.csv": quality,
        "placebo_results.csv": placebo_summary,
        "period_results.csv": period,
        "portfolio_nav.csv": nav,
        "trades.csv": trades,
    }.items():
        frame.to_csv(output / filename, index=False, encoding="utf-8-sig")
    write_report(output, core_summary, cost_summary, delay_summary, hold_summary, placebo_summary, quality, period)
    (output / "summary.json").write_text(json.dumps({
        "experiment": "Experiment 15: Clean Position Competition Validation",
        "data_version": version,
        "alpha_source": str(clean.PIT_PREDICTIONS),
        "alpha_variant": "raw_model; sample=test; 2024-2026H1",
        "frozen_rules": {"candidate_pool_top_pct": 10, "max_positions": 5, "baseline_hold_days": 10, "cost_bps": 20, "margins": list(MARGINS)},
        "stress_tests": {"cost_bps": list(COST_BPS), "execution_delay_days": list(DELAYS), "hold_days": list(HOLD_DAYS)},
        "prohibited_inputs": ["Signal Reliability", "Holding Health", "Alpha Residual", "future labels", "retrained Alpha"],
        "execution": "T close rank -> T+1/2/3 raw_open depending on delay; price limits/ST/suspension/listing constraints enabled",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"done -> {output}")


def tag_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    out = frame.copy(); out.insert(0, "variant", name); return out


def summary_frame(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in runs:
        trades = item["trades"]
        buys = trades.loc[trades["side"].eq("BUY")] if not trades.empty else pd.DataFrame()
        sells = trades.loc[trades["side"].eq("SELL")] if not trades.empty else pd.DataFrame()
        metrics = item["metrics"]
        rows.append({
            "variant": item["name"], "rule": item["rule"], "margin": item["margin"], "cost_bps": item["cost_bps"], "delay": item["delay"], "hold_days": item["hold_days"],
            "total_return": metrics["total_return"], "annualized_return": metrics["annualized_return"], "sharpe": metrics["sharpe"], "max_drawdown": metrics["max_drawdown"], "calmar": metrics["calmar"],
            "trade_count": metrics["trade_count"], "buy_count": int(len(buys)), "sell_count": int(len(sells)), "annualized_turnover": metrics["annualized_turnover"],
            "average_holding_days": average_holding_days(trades), "replacement_count": int(sells["reason"].astype(str).str.contains("replace_exit", na=False).sum()) if not sells.empty else 0,
        })
    return pd.DataFrame(rows)


def average_holding_days(trades: pd.DataFrame) -> float:
    if trades.empty:
        return float("nan")
    sells = trades.loc[trades["side"].eq("SELL")]
    return float(pd.to_numeric(sells.get("holding_trade_days"), errors="coerce").mean())


def replacement_dates(trades: pd.DataFrame) -> set[pd.Timestamp]:
    return set(pd.to_datetime(trades.loc[trades["reason"].astype(str).str.contains("replace_exit", na=False), "trade_date"]))


def replacement_quality(run: dict[str, Any], panel: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    trades = run["trades"].copy()
    exits = trades.loc[trades["reason"].astype(str).str.contains("replace_exit", na=False)].copy()
    entries = trades.loc[trades["reason"].eq("competition_entry")].copy()
    if exits.empty or entries.empty:
        return pd.DataFrame(columns=["variant", "trade_date", "old_stock", "new_stock", "old_rank", "new_rank", "old_future_return_5d", "new_future_return_5d", "return_improvement", "replacement_success"])
    pairs = exits.merge(entries, on="trade_date", suffixes=("_old", "_new"), how="inner")
    rank = signals.rename(columns={"trade_date": "signal_date", "ts_code": "stock", "alpha_rank_pct": "rank"})[["signal_date", "stock", "rank"]]
    decision_date = prior_signal_dates(pairs["trade_date"], panel, run["delay"])
    pairs["decision_date"] = decision_date.to_numpy()
    pairs = pairs.merge(rank.rename(columns={"stock": "ts_code_old", "rank": "old_rank"}), left_on=["decision_date", "ts_code_old"], right_on=["signal_date", "ts_code_old"], how="left").drop(columns="signal_date")
    pairs = pairs.merge(rank.rename(columns={"stock": "ts_code_new", "rank": "new_rank"}), left_on=["decision_date", "ts_code_new"], right_on=["signal_date", "ts_code_new"], how="left").drop(columns="signal_date")
    # An absent held stock is treated as rank 0 by the execution rule: it has
    # fallen outside the PIT scoring universe on the decision date.
    pairs["old_rank"] = pairs["old_rank"].fillna(0.0)
    prices = panel[["trade_date", "ts_code", "adj_open"]].sort_values(["ts_code", "trade_date"]).copy()
    prices["future_open_5d"] = prices.groupby("ts_code")["adj_open"].shift(-5)
    old = prices.rename(columns={"ts_code": "ts_code_old", "adj_open": "old_open", "future_open_5d": "old_future_open"})
    new = prices.rename(columns={"ts_code": "ts_code_new", "adj_open": "new_open", "future_open_5d": "new_future_open"})
    pairs = pairs.merge(old[["trade_date", "ts_code_old", "old_open", "old_future_open"]], on=["trade_date", "ts_code_old"], how="left")
    pairs = pairs.merge(new[["trade_date", "ts_code_new", "new_open", "new_future_open"]], on=["trade_date", "ts_code_new"], how="left")
    pairs["old_future_return_5d"] = pairs["old_future_open"] / pairs["old_open"] - 1.0
    pairs["new_future_return_5d"] = pairs["new_future_open"] / pairs["new_open"] - 1.0
    pairs["return_improvement"] = pairs["new_future_return_5d"] - pairs["old_future_return_5d"]
    pairs["replacement_success"] = np.where(pairs["return_improvement"].notna(), (pairs["return_improvement"] > 0).astype(int), np.nan)
    pairs.insert(0, "variant", run["name"])
    return pairs.rename(columns={"ts_code_old": "old_stock", "ts_code_new": "new_stock"})[["variant", "trade_date", "decision_date", "old_stock", "new_stock", "old_rank", "new_rank", "old_future_return_5d", "new_future_return_5d", "return_improvement", "replacement_success"]]


def prior_signal_dates(trade_dates: pd.Series, panel: pd.DataFrame, delay: int) -> pd.Series:
    dates = pd.Index(panel["trade_date"].drop_duplicates().sort_values())
    loc = {date: index for index, date in enumerate(dates)}
    return pd.Series([dates[loc[pd.Timestamp(date)] - 1 - delay] if loc[pd.Timestamp(date)] >= 1 + delay else pd.NaT for date in trade_dates], index=trade_dates.index)


def period_results(run: dict[str, Any]) -> pd.DataFrame:
    daily = run["daily"].copy(); daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    trades = run["trades"].copy(); trades["trade_date"] = pd.to_datetime(trades["trade_date"])
    rows = []
    for year, group in daily.groupby(daily["trade_date"].dt.year):
        curve = (1 + group["return"]).cumprod()
        tx = trades.loc[trades["trade_date"].dt.year.eq(year)]
        rows.append({"variant": run["name"], "year": int(year), "period_return": float(curve.iloc[-1] - 1), "period_max_drawdown": float((curve / curve.cummax() - 1).min()), "buy_count": int(tx["side"].eq("BUY").sum()), "sell_count": int(tx["side"].eq("SELL").sum()), "annualized_turnover": float(group["portfolio_turnover"].mean() * 252)})
    return pd.DataFrame(rows)


def compact(frame: pd.DataFrame, columns: list[str]) -> str:
    return "_empty_" if frame.empty else frame.loc[:, [column for column in columns if column in frame]].round(6).to_markdown(index=False)


def write_report(output: Path, core: pd.DataFrame, costs: pd.DataFrame, delays: pd.DataFrame, holds: pd.DataFrame, placebo: pd.DataFrame, quality: pd.DataFrame, period: pd.DataFrame) -> None:
    quality_summary = quality.groupby("variant", as_index=False).agg(replacement_count=("replacement_success", "size"), replacement_success_rate=("replacement_success", "mean"), average_return_improvement=("return_improvement", "mean")) if not quality.empty else pd.DataFrame()
    report = [
        "# Experiment 15: Clean Position Competition Validation", "",
        "## Frozen Scope", "- Alpha input: existing PIT `raw_model` output, `sample=test`, 2024 through 2026H1.", "- Decisions use only current PIT Alpha cross-sectional rank. Signal Reliability, Holding Health, Alpha Residual, future-label features, Alpha retraining and parameter search are excluded.", "- Baseline: Top 10% candidate pool, maximum 5 positions, 10 trading-day hold, 20bps all-in cost. Signal is observed at T close and filled at the configured future open.", "",
        "## Core Ablation", compact(core, ["variant", "margin", "annualized_return", "sharpe", "max_drawdown", "calmar", "trade_count", "annualized_turnover", "average_holding_days", "replacement_count"]), "",
        "## Cost Stress", compact(costs, ["variant", "cost_bps", "annualized_return", "sharpe", "max_drawdown", "annualized_turnover"]), "",
        "## Execution Delay", compact(delays, ["variant", "delay", "annualized_return", "sharpe", "max_drawdown", "annualized_turnover"]), "",
        "## Holding Period", compact(holds, ["variant", "hold_days", "annualized_return", "sharpe", "max_drawdown", "annualized_turnover"]), "",
        "## Random Replacement Placebo", compact(placebo, ["variant", "annualized_return", "sharpe", "max_drawdown", "replacement_count", "annualized_turnover"]), "- The placebo attempts the real model's replacement-date schedule. It never bypasses suspension/price-limit/lot constraints; therefore an unavailable scheduled exit can leave its realized replacement count below the target schedule.", "",
        "## Replacement Quality", compact(quality_summary, ["variant", "replacement_count", "replacement_success_rate", "average_return_improvement"]), "",
        "## Train-free Period Results", compact(period, ["variant", "year", "period_return", "period_max_drawdown", "buy_count", "sell_count", "annualized_turnover"]), "",
        "## Interpretation Rules", "- Rank Competition is independently supported only if its replacement quality is positive, it exceeds the schedule-matched random placebo, and the direction survives reasonable cost, delay and holding-period stress.", "- Identical results across pre-registered margins mean the threshold itself has not demonstrated incremental value. Do not choose one based on this output.", "- This is a train-free chronological evaluation rather than a model-fitting walk-forward. Each annual block is reported separately; no parameter was learned from any block.", "",
        "## Files", "- `portfolio_results.csv`, `cost_stress.csv`, `delay_test.csv`, `holding_period_test.csv`", "- `replacement_quality.csv`, `placebo_results.csv`, `period_results.csv`, `portfolio_nav.csv`, `trades.csv`",
    ]
    (output / "clean_competition_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
