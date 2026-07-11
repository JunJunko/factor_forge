from __future__ import annotations

"""Experiment 14: Position Competition Model.

Research-only alpha-capacity experiment.  It uses a fixed, interpretable rule
to compare an occupied portfolio slot with a new param_068 candidate.  No
Alpha, Signal Reliability, timing, entry parameter, or ML model is tuned.
"""

import argparse
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

import sell_impact_holding_health as hh
import sell_impact_ranker_timing_compare as timing_compare
import sell_impact_trade_quality_optimization as tq
from factor_forge.config import CostModel, ExecutionConstraints


OUTPUT_ROOT = Path("artifacts/strategy_reviews/experiment14_position_competition")
RANDOM_SEED = 20260710
HOLDING_ALPHA_WEIGHT = 0.50
HOLDING_HEALTH_WEIGHT = 0.30
HOLDING_RELIABILITY_WEIGHT = 0.20
CANDIDATE_ALPHA_WEIGHT = 0.70
CANDIDATE_RELIABILITY_WEIGHT = 0.30
MARGINS = (0.05, 0.10, 0.15)
PLACEBO_MARGIN = 0.10
HEALTH_HORIZON = 5
HEALTH_MODEL = "logistic"  # Selected in Experiment 12 by validation ROC-AUC.


def main() -> None:
    args = parse_args()
    output = OUTPUT_ROOT / f"position_competition_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    log_path = output / "run.log"

    def log(message: str) -> None:
        line = f"[{time.time() - started:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("loading frozen Alpha scores, Signal Reliability, and frozen param_068 configuration")
    cfg = hh.load_fixed10_config()
    signals, alpha_validation = hh.load_historical_signals(args.signal_dataset, args.predictions, cfg, log)
    alpha_validation.to_csv(output / "alpha_score_reconstruction.csv", index=False, encoding="utf-8-sig")
    version, full_panel = hh.base.load_panel()
    full_panel["trade_date"] = pd.to_datetime(full_panel["trade_date"])
    panel = full_panel.loc[
        full_panel["ts_code"].isin(signals["ts_code"].unique())
        & full_panel["trade_date"].between(pd.Timestamp("2023-01-01"), hh.TEST_END)
    ].copy()
    del full_panel
    simulation_panel = panel.loc[panel["trade_date"].between(signals["trade_date"].min(), hh.TEST_END)].copy()
    timing = timing_compare.load_position_multiplier(timing_compare.TIMING_DAILY)
    market_benchmark = timing_compare.load_market_benchmark(version)
    constraints = ExecutionConstraints(
        exclude_suspended=True, cannot_buy_limit_up=True, cannot_sell_limit_down=True,
        exclude_st=True, exclude_delisting_period=True, min_listing_days=60,
    )
    cost_model = CostModel(commission_bps_per_side=3, slippage_bps_per_side=5, stamp_duty_bps_sell=5)
    log(f"signals={len(signals):,}; panel={len(panel):,}; data_version={version}")

    log("replaying fixed-10-day baseline to obtain actual holding-day observations")
    baseline_daily, baseline_trades, baseline_positions, baseline_metrics = tq.run_trade_quality_backtest(
        panel=simulation_panel, signals=signals, timing=timing, market_benchmark=market_benchmark,
        constraints=constraints, cost_model=cost_model, cfg=cfg,
    )
    baseline_daily["variant"] = "baseline_fixed10"
    baseline_trades["variant"] = "baseline_fixed10"
    log(f"baseline entries={baseline_metrics['executed_buys']:,} ann={baseline_metrics['annualized_return']:.2%}")

    log("reconstructing frozen Holding Health probabilities for every potential param_068 holding")
    potential = hh.build_potential_holding_panel(signals, panel, cfg, log)
    health_training = hh.actual_holding_dataset(potential, baseline_positions)
    health_training = hh.add_labels(health_training, panel, signals, cfg, log)
    health_training["sample"] = hh.sample_labels(health_training["observation_date"])
    health_training = health_training.loc[health_training["sample"].isin(["train", "valid", "test"])].copy()
    health_features = hh.feature_columns(health_training)
    health_models = hh.fit_holding_models(health_training, health_features, log)["models"]
    health_predictions = hh.predict_potential_health(potential, health_features, health_models)
    health_col = f"holding_health_{HEALTH_HORIZON}d_{HEALTH_MODEL}"
    potential = potential.merge(health_predictions[["observation_date", "entry_date", "ts_code", health_col]], on=["observation_date", "entry_date", "ts_code"], how="left")
    potential[health_col] = potential[health_col].fillna(0.5).clip(0.0, 1.0)
    potential["holding_score"] = (
        HOLDING_ALPHA_WEIGHT * pd.to_numeric(potential["current_rank_pct"], errors="coerce").fillna(0.0)
        + HOLDING_HEALTH_WEIGHT * potential[health_col]
        + HOLDING_RELIABILITY_WEIGHT * pd.to_numeric(potential["current_reliability"], errors="coerce").fillna(0.5)
    )
    signals = signals.copy()
    signals["candidate_score"] = (
        CANDIDATE_ALPHA_WEIGHT * pd.to_numeric(signals["raw_rank_pct"], errors="coerce").fillna(0.0)
        + CANDIDATE_RELIABILITY_WEIGHT * pd.to_numeric(signals["signal_probability"], errors="coerce").fillna(0.5)
    )
    log(f"holding health source={HEALTH_MODEL} {HEALTH_HORIZON}d; potential states={len(potential):,}")

    actual_states = hh.actual_holding_dataset(potential, baseline_positions)
    position_scores = build_position_scores(actual_states, signals, health_col)
    competition = build_competition_dataset(actual_states, signals, panel, cfg, health_col)
    position_scores.to_csv(output / "position_scores.csv", index=False, encoding="utf-8-sig")
    competition.to_csv(output / "competition_dataset.csv", index=False, encoding="utf-8-sig")
    log(f"position score rows={len(position_scores):,}; candidate-vs-holder comparisons={len(competition):,}")

    score_lookup = {
        (pd.Timestamp(row.observation_date), pd.Timestamp(row.entry_date), str(row.ts_code)): float(row.holding_score)
        for row in potential[["observation_date", "entry_date", "ts_code", "holding_score"]].itertuples()
        if np.isfinite(row.holding_score)
    }
    variants = [("baseline_fixed10", "baseline", None)]
    variants.extend((f"competition_margin_{int(m * 100):02d}", "competition", m) for m in MARGINS)
    variants.append(("random_competition_placebo_m10", "random_placebo", PLACEBO_MARGIN))
    daily_frames, trade_frames = [baseline_daily], [baseline_trades]
    results = [{"variant": "baseline_fixed10", "rule": "baseline", "margin": np.nan, **baseline_metrics}]
    for variant, rule, margin in variants[1:]:
        log(f"backtesting {variant}")
        daily, trades, metrics = run_competition_backtest(
            panel=simulation_panel, signals=signals, timing=timing, market_benchmark=market_benchmark,
            constraints=constraints, cost_model=cost_model, cfg=cfg, holding_score_lookup=score_lookup,
            rule=rule, margin=float(margin), seed=RANDOM_SEED,
        )
        daily["variant"] = variant; trades["variant"] = variant
        daily_frames.append(daily); trade_frames.append(trades)
        results.append({"variant": variant, "rule": rule, "margin": margin, **metrics})
        log(f"{variant}: ann={metrics['annualized_return']:.2%} sharpe={metrics['sharpe']:.2f} mdd={metrics['max_drawdown']:.2%}")
    portfolio_nav = pd.concat(daily_frames, ignore_index=True)
    trades = pd.concat(trade_frames, ignore_index=True)
    portfolio_results = pd.DataFrame(results)
    portfolio_nav.to_csv(output / "portfolio_nav.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(output / "trades.csv", index=False, encoding="utf-8-sig")
    portfolio_results.to_csv(output / "portfolio_results.csv", index=False, encoding="utf-8-sig")

    replacement = replacement_analysis(trades, panel)
    holding_quality = holding_quality_analysis(trades)
    placebo = portfolio_results.loc[portfolio_results["rule"].isin(["competition", "random_placebo"])].copy()
    yearly = yearly_results(portfolio_nav, trades)
    replacement.to_csv(output / "replacement_analysis.csv", index=False, encoding="utf-8-sig")
    holding_quality.to_csv(output / "holding_quality.csv", index=False, encoding="utf-8-sig")
    placebo.to_csv(output / "placebo_results.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_results.csv", index=False, encoding="utf-8-sig")
    write_report(output, position_scores, competition, portfolio_results, yearly, replacement, holding_quality, placebo)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "experiment": "Experiment 14: Position Competition Model",
                "run_dir": str(output), "data_version": version,
                "weights": {"holding": [HOLDING_ALPHA_WEIGHT, HOLDING_HEALTH_WEIGHT, HOLDING_RELIABILITY_WEIGHT], "candidate": [CANDIDATE_ALPHA_WEIGHT, CANDIDATE_RELIABILITY_WEIGHT]},
                "sign_audit": "Health and reliability are success probabilities, so both receive positive weights. Margin requires candidate_score > holding_score + margin.",
                "margins": list(MARGINS), "placebo_margin": PLACEBO_MARGIN,
                "no_alpha_or_reliability_retraining": True, "no_production_code_modified": True,
            }, ensure_ascii=False, indent=2,
        ), encoding="utf-8")
    log(f"done -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 14: Position Competition Model.")
    parser.add_argument("--signal-dataset", type=Path, default=hh.SIGNAL_DATASET)
    parser.add_argument("--predictions", type=Path, default=hh.SIGNAL_PREDICTIONS)
    return parser.parse_args()


def build_position_scores(actual_states: pd.DataFrame, signals: pd.DataFrame, health_col: str) -> pd.DataFrame:
    holding = actual_states[["observation_date", "entry_date", "ts_code", "holding_days", "current_rank_pct", "current_reliability", health_col, "holding_score"]].copy()
    holding["role"] = "holding"; holding = holding.rename(columns={"ts_code": "stock", "current_rank_pct": "alpha_rank", "current_reliability": "reliability", health_col: "health", "holding_score": "position_score"})
    candidate = signals.loc[signals["candidate_pool_flag"].eq(1), ["trade_date", "ts_code", "raw_rank_pct", "signal_probability", "candidate_score"]].copy()
    candidate["role"] = "candidate"; candidate["entry_date"] = pd.NaT; candidate["holding_days"] = np.nan; candidate["health"] = np.nan
    candidate = candidate.rename(columns={"trade_date": "observation_date", "ts_code": "stock", "raw_rank_pct": "alpha_rank", "signal_probability": "reliability", "candidate_score": "position_score"})
    cols = ["observation_date", "entry_date", "stock", "holding_days", "role", "alpha_rank", "reliability", "health", "position_score"]
    return pd.concat([holding[cols], candidate[cols]], ignore_index=True).sort_values(["observation_date", "role", "position_score"], ascending=[True, True, False])


def build_competition_dataset(actual_states: pd.DataFrame, signals: pd.DataFrame, panel: pd.DataFrame, cfg: dict[str, Any], health_col: str) -> pd.DataFrame:
    states = actual_states[["observation_date", "entry_date", "ts_code", "holding_days", "holding_score", health_col, "current_rank_pct", "current_reliability"]].copy()
    candidates = signals.loc[signals["candidate_pool_flag"].eq(1), ["trade_date", "ts_code", "candidate_score", "raw_rank_pct", "signal_probability"]].rename(columns={"trade_date": "observation_date"})
    dates = pd.Index(panel["trade_date"].drop_duplicates().sort_values())
    index = {date: i for i, date in enumerate(dates)}
    prices = panel[["trade_date", "ts_code", "adj_open"]].sort_values(["ts_code", "trade_date"]).copy()
    prices["entry_open"] = prices.groupby("ts_code")["adj_open"].shift(-1)
    prices["exit_open_5d"] = prices.groupby("ts_code")["adj_open"].shift(-6)
    returns = prices[["trade_date", "ts_code", "entry_open", "exit_open_5d"]].copy()
    returns["future_return_5d"] = returns.exit_open_5d / returns.entry_open - 1.0
    rows = []
    for date, held in states.groupby("observation_date", sort=False):
        pool = candidates.loc[candidates["observation_date"].eq(date)]
        if pool.empty: continue
        held_codes = set(held["ts_code"])
        pool = pool.loc[~pool["ts_code"].isin(held_codes)]
        if pool.empty: continue
        left = held.assign(key=1); right = pool.assign(key=1)
        pair = left.merge(right, on="key", suffixes=("_holding", "_candidate")).drop(columns="key")
        pair["date"] = date
        rows.append(pair)
    if not rows: return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    hold_ret = returns.rename(columns={"trade_date": "date", "ts_code": "holding_stock", "future_return_5d": "holding_future_return_5d"})[["date", "holding_stock", "holding_future_return_5d"]]
    cand_ret = returns.rename(columns={"trade_date": "date", "ts_code": "candidate_stock", "future_return_5d": "candidate_future_return_5d"})[["date", "candidate_stock", "candidate_future_return_5d"]]
    out = out.rename(columns={"ts_code_holding": "holding_stock", "ts_code_candidate": "candidate_stock", "holding_score": "holding_score", "candidate_score": "candidate_score", health_col: "holding_health", "signal_probability": "entry_reliability", "current_rank_pct": "current_alpha_rank"})
    out = out.merge(hold_ret, on=["date", "holding_stock"], how="left").merge(cand_ret, on=["date", "candidate_stock"], how="left")
    out["return_difference"] = out.candidate_future_return_5d - out.holding_future_return_5d
    out["candidate_wins"] = np.where(out.return_difference.notna(), (out.return_difference > 0).astype(int), np.nan)
    wanted = ["date", "entry_date", "holding_days", "holding_stock", "candidate_stock", "holding_score", "candidate_score", "holding_health", "entry_reliability", "current_alpha_rank", "raw_rank_pct", "holding_future_return_5d", "candidate_future_return_5d", "return_difference", "candidate_wins"]
    return out[[c for c in wanted if c in out.columns]].sort_values(["date", "holding_stock", "candidate_score"], ascending=[True, True, False])


def run_competition_backtest(*, panel: pd.DataFrame, signals: pd.DataFrame, timing: pd.Series, market_benchmark: pd.DataFrame, constraints: ExecutionConstraints, cost_model: CostModel, cfg: dict[str, Any], holding_score_lookup: dict, rule: str, margin: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data = panel.merge(signals, on=["trade_date", "ts_code"], how="left").sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    dates = list(pd.Index(data["trade_date"].unique()).sort_values())
    by_date = {date: group.set_index("ts_code") for date, group in data.groupby("trade_date")}
    marks = tq.acct.mark_price_lookup(data); rng = np.random.default_rng(seed)
    cash, positions, trades, daily_rows, generated = tq.INITIAL_CASH, [], [], [], 0
    for date_index, date in enumerate(dates):
        today = by_date[date]; signal_date = dates[date_index - 1] if date_index else None
        signal_frame = by_date[signal_date] if signal_date is not None else None
        sells = buys = generated_today = blocked_trade = blocked_cash = 0; turnover = costs = 0.0; chosen_candidate = None
        remaining = []
        for position in positions:
            if date_index - position.entry_index >= int(cfg["max_hold_days"]) and tq.acct.can_sell(today, position.ts_code, constraints):
                value = tq.acct.position_value(tq.to_account_position(position), date, marks); cost = tq.acct.cost(value, "SELL", cost_model, tq.COST_BPS)
                cash += value - cost; sells += 1; turnover += value; costs += cost
                trade = tq.trade_row(position, date, "SELL", value, cost, today, "max_hold"); trade["holding_trade_days"] = date_index - position.entry_index; trades.append(trade)
            else: remaining.append(position)
        positions = remaining
        if signal_frame is not None and len(positions) >= int(cfg["max_positions"]):
            candidates = tq.entry_candidates(signal_frame, positions, cfg).copy()
            if not candidates.empty:
                candidates["competition_score"] = CANDIDATE_ALPHA_WEIGHT * candidates["raw_rank_pct"].fillna(0.0) + CANDIDATE_RELIABILITY_WEIGHT * candidates["signal_probability"].fillna(0.5)
                eligible = []
                for position in positions:
                    if date_index - position.entry_index < int(cfg["min_hold_days"]): continue
                    score = holding_score_lookup.get((pd.Timestamp(signal_date), pd.Timestamp(position.entry_date), str(position.ts_code)))
                    if score is not None: eligible.append((float(score), position))
                if eligible:
                    holding_score, victim = min(eligible, key=lambda item: (item[0], item[1].ts_code))
                    top = candidates.sort_values(["competition_score", "raw_score"], ascending=False).iloc[0]
                    if float(top.competition_score) > holding_score + margin and tq.acct.can_sell(today, victim.ts_code, constraints):
                        if rule == "random_placebo":
                            chosen_candidate = str(rng.choice(candidates.index.to_numpy()))
                        else:
                            chosen_candidate = str(top.name)
                        value = tq.acct.position_value(tq.to_account_position(victim), date, marks); cost = tq.acct.cost(value, "SELL", cost_model, tq.COST_BPS)
                        cash += value - cost; sells += 1; turnover += value; costs += cost; positions.remove(victim)
                        trade = tq.trade_row(victim, date, "SELL", value, cost, today, f"{rule}_replace_exit"); trade["holding_trade_days"] = date_index - victim.entry_index; trades.append(trade)
        nav_before, gross_before, _ = tq.account_value(cash, positions, date, marks)
        target = float(np.clip(timing.get(date, 1.0), 0.0, 1.0)); vacancies = max(0, int(cfg["max_positions"]) - len(positions))
        if signal_frame is not None and vacancies:
            candidates = tq.entry_candidates(signal_frame, positions, cfg).copy(); generated_today = len(candidates); generated += generated_today
            if not candidates.empty:
                if chosen_candidate in candidates.index:
                    candidates = pd.concat([candidates.loc[[chosen_candidate]], candidates.drop(index=chosen_candidate)])
                budget = max(0.0, min(cash, nav_before * target - gross_before)); slot = nav_before * target / int(cfg["max_positions"]); per_buy = min(slot, budget / vacancies)
                for ts_code, signal_row in candidates.head(vacancies).iterrows():
                    if not tq.acct.can_buy(today, ts_code, constraints): blocked_trade += 1; continue
                    row = today.loc[ts_code]; row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
                    shares = int(per_buy // (float(row.raw_open) * tq.LOT_SIZE)) * tq.LOT_SIZE
                    if shares <= 0: blocked_cash += 1; continue
                    gross = shares * float(row.raw_open); cost = tq.acct.cost(gross, "BUY", cost_model, tq.COST_BPS)
                    if gross + cost > cash: blocked_cash += 1; continue
                    position = tq.Position(str(ts_code), shares, date, date_index, float(row.raw_open), float(row.adj_open), signal_date, float(signal_row.get("band_rank_pct", np.nan)), float(signal_row.get("raw_rank_pct", np.nan)))
                    cash -= gross + cost; buys += 1; turnover += gross; costs += cost; positions.append(position)
                    reason = "competition_entry" if str(ts_code) == chosen_candidate else "entry"; trades.append(tq.trade_row(position, date, "BUY", gross, cost, today, reason))
        nav, gross, values = tq.account_value(cash, positions, date, marks); previous = daily_rows[-1]["nav"] if daily_rows else tq.INITIAL_CASH
        daily_rows.append({"trade_date": date, "nav": nav, "gross_exposure": gross, "cash": cash, "target_position": target, "new_signals": generated_today, "executed_buys": buys, "executed_sells": sells, "cash_blocked_candidates": blocked_cash, "trade_blocked_candidates": blocked_trade, "holding_count": len(positions), "unique_holding_count": len({p.ts_code for p in positions}), "portfolio_turnover": turnover / previous if previous else 0.0, "cash_ratio": cash / nav if nav else 0.0, "gross_exposure_ratio": gross / nav if nav else 0.0, "largest_position_weight": max((v for _, v in values), default=0.0) / nav if nav else 0.0, "transaction_cost": costs})
    daily = pd.DataFrame(daily_rows); daily["return"] = daily.nav.pct_change().fillna(0.0); daily["market_index_return"] = tq.acct.market_index_returns(market_benchmark, dates)
    trade_frame = pd.DataFrame(trades); return daily, trade_frame, tq.metrics_from_daily(daily, trade_frame, generated)


def replacement_analysis(trades: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    exits = trades.loc[trades.reason.astype(str).str.contains("replace_exit", na=False)].copy()
    if exits.empty: return pd.DataFrame()
    prices = panel[["trade_date", "ts_code", "adj_open"]].sort_values(["ts_code", "trade_date"]).copy(); prices["future_open_5d"] = prices.groupby("ts_code")["adj_open"].shift(-5)
    exits = exits.merge(prices, on=["trade_date", "ts_code"], how="left"); exits["replaced_future_5d_return"] = exits.future_open_5d / exits.adj_open - 1.0
    entries = trades.loc[trades.reason.eq("competition_entry"), ["variant", "trade_date", "ts_code"]].merge(prices, on=["trade_date", "ts_code"], how="left")
    entries["new_future_5d_return"] = entries.future_open_5d / entries.adj_open - 1.0
    return exits.groupby("variant", as_index=False).agg(replacement_count=("ts_code", "size"), replaced_future_5d_return=("replaced_future_5d_return", "mean")).merge(entries.groupby("variant", as_index=False).agg(new_entry_count=("ts_code", "size"), new_entry_future_5d_return=("new_future_5d_return", "mean")), on="variant", how="left")


def holding_quality_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    buys = trades.loc[trades.side.eq("BUY")]; sells = trades.loc[trades.side.eq("SELL")]
    pairs = buys.merge(sells, on=["variant", "ts_code", "entry_date"], suffixes=("_buy", "_sell"))
    if pairs.empty: return pd.DataFrame()
    pairs["trade_return"] = (pairs.gross_value_sell - pairs.cost_sell) / (pairs.gross_value_buy + pairs.cost_buy) - 1.0; pairs["holding_days"] = (pd.to_datetime(pairs.trade_date_sell) - pd.to_datetime(pairs.trade_date_buy)).dt.days
    return pairs.groupby("variant", as_index=False).agg(round_trips=("ts_code", "size"), mean_holding_return=("trade_return", "mean"), loss_trade_ratio=("trade_return", lambda s: float((s <= 0).mean())), avg_holding_calendar_days=("holding_days", "mean"))


def yearly_results(nav: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = nav.copy(); data["trade_date"] = pd.to_datetime(data["trade_date"])
    trade_data = trades.copy(); trade_data["trade_date"] = pd.to_datetime(trade_data["trade_date"])
    for (variant, year), group in data.groupby(["variant", data["trade_date"].dt.year], sort=True):
        curve = (1.0 + group["return"]).cumprod()
        period_trades = trade_data.loc[(trade_data["variant"].eq(variant)) & (trade_data["trade_date"].dt.year.eq(year))]
        rows.append({"variant": variant, "year": int(year), "period_return": float(curve.iloc[-1] - 1.0), "period_max_drawdown": float((curve / curve.cummax() - 1.0).min()), "buy_count": int(period_trades["side"].eq("BUY").sum()), "sell_count": int(period_trades["side"].eq("SELL").sum()), "annualized_turnover": float(group["portfolio_turnover"].mean() * 252)})
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, limit: int = 30) -> str:
    return "_empty_" if frame is None or frame.empty else frame.head(limit).round(6).to_markdown(index=False)


def write_report(output: Path, scores: pd.DataFrame, competition: pd.DataFrame, portfolio: pd.DataFrame, yearly: pd.DataFrame, replacement: pd.DataFrame, holding: pd.DataFrame, placebo: pd.DataFrame) -> None:
    winner = competition.groupby("candidate_wins", dropna=True).agg(pairs=("return_difference", "size"), mean_return_difference=("return_difference", "mean"), mean_score_gap=("candidate_score", "mean")).reset_index() if not competition.empty else pd.DataFrame()
    lines = [
        "# Experiment 14: Position Competition Model", "",
        "## Scope", "- A rule-based slot-selection layer compares an occupied slot with the strongest new candidate. It does not retrain Alpha or Signal Reliability, change entry/timing, or fit a competition ML model.", "- Holding Health and Signal Reliability are success probabilities. Their signs are positive in the score; the written negative-sign formula would reward weaker holdings, so it is not economically coherent.", "- Scores use rank/probability units: `holding=0.5*current_alpha_rank + 0.3*holding_health + 0.2*current_reliability`; `candidate=0.7*entry_alpha_rank + 0.3*entry_reliability`.", "- A replacement requires `candidate_score > weakest_holding_score + margin`; decisions use T-close information and execute at T+1 open.", "",
        "## Dataset", f"- Position-score rows: {len(scores):,}; candidate-vs-holder pairs: {len(competition):,}.", "- Competition labels are analytical only: candidate wins when its next 5-session open-to-open return exceeds the holder's return over the same period.", "", "## Competition Label Audit", md_table(winner), "", "## Portfolio Results", md_table(portfolio.sort_values("annualized_return", ascending=False)), "", "## Yearly / OOS Split", md_table(yearly.sort_values(["year", "variant"]), 30), "", "## Replacement Quality", md_table(replacement), "", "## Holding Quality", md_table(holding), "", "## Placebo", md_table(placebo), "",
        "## Interpretation", "- All margins (0.05, 0.10, 0.15) are pre-registered and reported; none is selected by backtest outcome.", "- The random placebo uses the same margin-0.10 trigger as real competition but randomly chooses the replacement candidate. It tests whether any benefit is due to score selection rather than simply increasing turnover.", "- The reconstructed static Alpha scorer trained through 2025Q3, so 2024-2025 performance contains Alpha in-sample exposure. Treat 2026H1 as the cleanest available OOS evidence.", "- Do not promote a version if replacement entrants do not outperform the stocks removed, even if the aggregate NAV happened to improve.", "",
        "## Files", "- `position_scores.csv`, `competition_dataset.csv`", "- `portfolio_results.csv`, `yearly_results.csv`, `replacement_analysis.csv`, `holding_quality.csv`, `placebo_results.csv`", "- `trades.csv`, `portfolio_nav.csv`",
    ]
    (output / "position_competition_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
