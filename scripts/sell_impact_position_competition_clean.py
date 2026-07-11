from __future__ import annotations

"""Leakage-cleaned Stage 1 of Experiment 14.

This deliberately quarantines the previous Signal Reliability and Holding Health
artifacts: both were fed by a feature derived from a forward label.  The script
uses only the existing PIT ``raw_model`` Alpha scores and a deterministic
rank-retention health proxy.  It is a research-only, fixed-rule audit; it does
not touch the production trading system.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sell_impact_trade_quality_optimization as tq
from factor_forge.config import CostModel, ExecutionConstraints


PIT_PREDICTIONS = Path(
    "artifacts/strategy_reviews/sell_impact_pit_regime_blend_20260709T082238Z/"
    "pit_regime_blend_predictions.parquet"
)
OUTPUT_ROOT = Path("artifacts/strategy_reviews/experiment14_position_competition_clean")
TEST_START = pd.Timestamp("2024-01-02")
TEST_END = pd.Timestamp("2026-06-12")
POOL_RANK_MIN = 0.90
MAX_POSITIONS = 5
MAX_HOLD_DAYS = 10
MIN_HOLD_DAYS = 1
MARGINS = (0.05, 0.10, 0.15)
PLACEBO_MARGIN = 0.10
RANDOM_SEED = 20260710


def main() -> None:
    output = OUTPUT_ROOT / f"position_competition_clean_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    log_path = output / "run.log"

    def log(message: str) -> None:
        line = f"[{time.time() - started:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("loading PIT raw Alpha test predictions; excluding all legacy reliability artifacts")
    signals = load_pit_signals()
    version, panel = tq.base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel.loc[
        panel["trade_date"].between(TEST_START, TEST_END)
        & panel["ts_code"].isin(signals["ts_code"].unique())
    ].copy()
    constraints = ExecutionConstraints(
        exclude_suspended=True,
        cannot_buy_limit_up=True,
        cannot_sell_limit_down=True,
        exclude_st=True,
        exclude_delisting_period=True,
        min_listing_days=60,
    )
    cost_model = CostModel(commission_bps_per_side=3, slippage_bps_per_side=5, stamp_duty_bps_sell=5)
    log(f"data_version={version}; PIT signals={len(signals):,}; panel rows={len(panel):,}")

    variants = [("baseline_fixed_top5_hold10", "baseline", np.nan)]
    variants += [(f"competition_clean_m{int(m * 100):02d}", "competition", m) for m in MARGINS]
    variants += [("raw_rank_only_m10", "raw_rank", PLACEBO_MARGIN)]
    variants += [("random_competition_placebo_m10", "random_placebo", PLACEBO_MARGIN)]
    nav_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    state_frames: list[pd.DataFrame] = []
    results: list[dict[str, object]] = []
    for variant, rule, margin in variants:
        log(f"running {variant}")
        daily, trades, states, metrics = run_backtest(
            panel, signals, constraints, cost_model, rule, float(margin) if pd.notna(margin) else 0.0
        )
        daily["variant"] = variant
        trades["variant"] = variant
        states["variant"] = variant
        nav_frames.append(daily)
        trade_frames.append(trades)
        state_frames.append(states)
        results.append({"variant": variant, "rule": rule, "margin": margin, **metrics})
        log(f"{variant}: cagr={metrics['annualized_return']:.2%}; sharpe={metrics['sharpe']:.2f}; mdd={metrics['max_drawdown']:.2%}")

    nav = pd.concat(nav_frames, ignore_index=True)
    trades = pd.concat(trade_frames, ignore_index=True)
    states = pd.concat(state_frames, ignore_index=True)
    summary = pd.DataFrame(results)
    yearly = yearly_results(nav, trades)
    replacements = replacement_analysis(trades, panel)
    competition = competition_dataset(states, signals, panel)
    for name, frame in {
        "portfolio_nav.csv": nav,
        "trades.csv": trades,
        "position_scores.csv": states,
        "portfolio_results.csv": summary,
        "yearly_results.csv": yearly,
        "replacement_analysis.csv": replacements,
        "competition_dataset.csv": competition,
        "signals_pit_raw_model.csv": signals,
    }.items():
        frame.to_csv(output / name, index=False, encoding="utf-8-sig")
    write_report(output, summary, yearly, replacements, competition)
    (output / "summary.json").write_text(json.dumps({
        "experiment": "Experiment 14 clean stage 1",
        "data_version": version,
        "alpha_source": str(PIT_PREDICTIONS),
        "alpha_variant": "raw_model / sample=test only",
        "fixed_rules": {"pool_rank_min": POOL_RANK_MIN, "max_positions": MAX_POSITIONS, "max_hold_days": MAX_HOLD_DAYS, "margins": list(MARGINS)},
        "quarantined_artifacts": [
            "stock Signal Reliability (historical_signal_success_rate future-label leakage)",
            "Holding Health model that consumed Signal Reliability",
            "static K_recent Alpha reconstruction",
            "param_068 selection and timing position control",
        ],
        "execution": "T-close PIT signal; T+1 raw_open execution; 20bps all-in cost; A-share constraints enabled",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"done -> {output}")


def load_pit_signals() -> pd.DataFrame:
    frame = pd.read_parquet(PIT_PREDICTIONS, columns=["trade_date", "ts_code", "sample", "variant", "score"])
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.loc[
        frame["sample"].eq("test")
        & frame["variant"].eq("raw_model")
        & frame["trade_date"].between(TEST_START, TEST_END)
    ].copy()
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("PIT test predictions unexpectedly contain duplicate date-stock keys")
    frame["alpha_rank_pct"] = frame.groupby("trade_date")["score"].rank(pct=True, method="average")
    frame["candidate_pool_flag"] = (frame["alpha_rank_pct"] >= POOL_RANK_MIN).astype(int)
    return frame[["trade_date", "ts_code", "score", "alpha_rank_pct", "candidate_pool_flag"]].sort_values(["trade_date", "score"], ascending=[True, False])


def run_backtest(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    constraints: ExecutionConstraints,
    cost_model: CostModel,
    rule: str,
    margin: float,
    *,
    cost_bps: float = tq.COST_BPS,
    execution_delay: int = 0,
    max_hold_days: int = MAX_HOLD_DAYS,
    forced_replacement_dates: set[pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    data = panel.merge(signals, on=["trade_date", "ts_code"], how="inner").sort_values(["trade_date", "ts_code"])
    dates = list(pd.Index(data["trade_date"].unique()).sort_values())
    by_date = {date: group.set_index("ts_code", drop=False) for date, group in data.groupby("trade_date")}
    marks = tq.acct.mark_price_lookup(data)
    rng = np.random.default_rng(RANDOM_SEED)
    cash, positions, trades, daily_rows, state_rows, generated = tq.INITIAL_CASH, [], [], [], [], 0
    for index, date in enumerate(dates):
        today = by_date[date]
        signal_index = index - 1 - execution_delay
        signal_date = dates[signal_index] if signal_index >= 0 else None
        signal_frame = by_date[signal_date] if signal_date is not None else None
        buys = sells = blocked_trade = blocked_cash = generated_today = 0
        turnover = costs = 0.0
        # Fixed holding rule: exits are always processed before any replacement decision.
        retained = []
        for position in positions:
            if index - position.entry_index >= max_hold_days and tq.acct.can_sell(today, position.ts_code, constraints):
                value = tq.acct.position_value(tq.to_account_position(position), date, marks)
                cost = tq.acct.cost(value, "SELL", cost_model, cost_bps)
                cash += value - cost; sells += 1; turnover += value; costs += cost
                row = tq.trade_row(position, date, "SELL", value, cost, today, "max_hold")
                row["holding_trade_days"] = index - position.entry_index
                trades.append(row)
            else:
                retained.append(position)
        positions = retained
        chosen_candidate = None
        forced_today = forced_replacement_dates is not None and date in forced_replacement_dates
        if signal_frame is not None and rule != "baseline" and (len(positions) >= MAX_POSITIONS or forced_today):
            candidates = candidates_for(signal_frame, positions)
            # The real rule compares the literal weakest eligible holding.  The
            # schedule-matched random placebo may choose the weakest *sellable*
            # holding so that its externally supplied event dates remain fixed.
            eligible = (
                [p for p in positions if tq.acct.can_sell(today, p.ts_code, constraints)]
                if forced_today
                else [p for p in positions if index - p.entry_index >= MIN_HOLD_DAYS]
            )
            if not candidates.empty and eligible:
                def holding_score(p: tq.Position) -> float:
                    current_rank = float(signal_frame.loc[p.ts_code, "alpha_rank_pct"]) if p.ts_code in signal_frame.index else 0.0
                    # Current/entry rank retention is observable at signal close.  It is not a model prediction.
                    health = float(np.clip(0.5 + current_rank - p.entry_raw_rank_pct, 0.0, 1.0))
                    if rule == "raw_rank":
                        return current_rank
                    return 0.5 * current_rank + 0.3 * health + 0.2 * 0.5
                victim = min(eligible, key=lambda p: (holding_score(p), p.ts_code))
                victim_value = tq.acct.position_value(tq.to_account_position(victim), date, marks)
                victim_cost = tq.acct.cost(victim_value, "SELL", cost_model, cost_bps)
                projected_cash = cash + victim_value - victim_cost
                current_nav, _, _ = tq.account_value(cash, positions, date, marks)
                slot_budget = current_nav / MAX_POSITIONS

                def can_complete_replacement(ts_code: str) -> bool:
                    if not tq.acct.can_buy(today, ts_code, constraints):
                        return False
                    row = today.loc[ts_code]
                    price = float(row.raw_open)
                    shares = int(slot_budget // (price * tq.LOT_SIZE)) * tq.LOT_SIZE
                    gross = shares * price
                    return shares > 0 and gross + tq.acct.cost(gross, "BUY", cost_model, cost_bps) <= projected_cash

                # A replacement is atomic in intent: do not release an occupied
                # slot unless an eligible candidate also fits the cash/lot budget.
                candidates = candidates.loc[[can_complete_replacement(str(code)) for code in candidates.index]]
                if not candidates.empty:
                    top = candidates.iloc[0]
                    candidate_score = float(top.alpha_rank_pct) if rule == "raw_rank" else 0.7 * float(top.alpha_rank_pct) + 0.3 * 0.5
                    allow_replacement = forced_today if forced_replacement_dates is not None else candidate_score > holding_score(victim) + margin
                    if allow_replacement and tq.acct.can_sell(today, victim.ts_code, constraints):
                        if rule == "random_placebo":
                            chosen_candidate = str(rng.choice(candidates.index.to_numpy()))
                        else:
                            chosen_candidate = str(top.name)
                        cash += victim_value - victim_cost; sells += 1; turnover += victim_value; costs += victim_cost; positions.remove(victim)
                        row = tq.trade_row(victim, date, "SELL", victim_value, victim_cost, today, f"{rule}_replace_exit")
                        row["holding_trade_days"] = index - victim.entry_index
                        trades.append(row)
        nav_before, gross_before, _ = tq.account_value(cash, positions, date, marks)
        vacancies = MAX_POSITIONS - len(positions)
        if signal_frame is not None and vacancies:
            candidates = candidates_for(signal_frame, positions)
            generated_today = len(candidates); generated += generated_today
            if chosen_candidate in candidates.index:
                candidates = pd.concat([candidates.loc[[chosen_candidate]], candidates.drop(index=chosen_candidate)])
            slot_budget = nav_before / MAX_POSITIONS
            for ts_code, signal_row in candidates.head(vacancies).iterrows():
                if not tq.acct.can_buy(today, ts_code, constraints):
                    blocked_trade += 1; continue
                raw = today.loc[ts_code]
                price = float(raw.raw_open)
                shares = int(slot_budget // (price * tq.LOT_SIZE)) * tq.LOT_SIZE
                gross = shares * price
                cost = tq.acct.cost(gross, "BUY", cost_model, cost_bps)
                if shares <= 0 or gross + cost > cash:
                    blocked_cash += 1; continue
                position = tq.Position(str(ts_code), shares, date, index, price, float(raw.adj_open), signal_date, np.nan, float(signal_row.alpha_rank_pct))
                cash -= gross + cost; buys += 1; turnover += gross; costs += cost; positions.append(position)
                reason = "competition_entry" if str(ts_code) == chosen_candidate else "entry"
                trades.append(tq.trade_row(position, date, "BUY", gross, cost, today, reason))
        nav, gross, values = tq.account_value(cash, positions, date, marks)
        for position in positions:
            rank = float(signal_frame.loc[position.ts_code, "alpha_rank_pct"]) if signal_frame is not None and position.ts_code in signal_frame.index else np.nan
            health = float(np.clip(0.5 + (0.0 if not np.isfinite(rank) else rank) - position.entry_raw_rank_pct, 0.0, 1.0))
            state_rows.append({"observation_date": date, "signal_date": signal_date, "entry_date": position.entry_date, "ts_code": position.ts_code, "holding_days": index - position.entry_index, "current_alpha_rank": rank, "entry_alpha_rank": position.entry_raw_rank_pct, "holding_rank_health": health})
        previous = daily_rows[-1]["nav"] if daily_rows else tq.INITIAL_CASH
        daily_rows.append({"trade_date": date, "nav": nav, "cash": cash, "gross_exposure": gross, "holding_count": len(positions), "unique_holding_count": len({p.ts_code for p in positions}), "executed_buys": buys, "executed_sells": sells, "new_signals": generated_today, "target_position": 1.0, "gross_exposure_ratio": gross / nav if nav else 0.0, "cash_ratio": cash / nav if nav else 0.0, "largest_position_weight": max((value for _, value in values), default=0.0) / nav if nav else 0.0, "portfolio_turnover": turnover / previous if previous else 0.0, "transaction_cost": costs, "trade_blocked_candidates": blocked_trade, "cash_blocked_candidates": blocked_cash})
    daily = pd.DataFrame(daily_rows)
    daily["return"] = daily["nav"].pct_change().fillna(0.0)
    trade_frame = pd.DataFrame(trades)
    metrics = tq.metrics_from_daily(daily.assign(market_index_return=0.0), trade_frame, generated)
    return daily, trade_frame, pd.DataFrame(state_rows), metrics


def candidates_for(signal_frame: pd.DataFrame, positions: list[tq.Position]) -> pd.DataFrame:
    held = {p.ts_code for p in positions}
    return signal_frame.loc[
        signal_frame["candidate_pool_flag"].eq(1) & ~signal_frame.index.isin(held)
    ].sort_values(["alpha_rank_pct", "score"], ascending=False)


def replacement_analysis(trades: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    exits = trades.loc[trades["reason"].astype(str).str.contains("replace_exit", na=False)].copy()
    entries = trades.loc[trades["reason"].eq("competition_entry")].copy()
    if exits.empty:
        return pd.DataFrame()
    prices = panel[["trade_date", "ts_code", "adj_open"]].sort_values(["ts_code", "trade_date"]).copy()
    prices["future_open_5d"] = prices.groupby("ts_code")["adj_open"].shift(-5)
    def future(frame: pd.DataFrame) -> pd.DataFrame:
        x = frame.merge(prices, on=["trade_date", "ts_code"], how="left")
        x["future_5d_return"] = x["future_open_5d"] / x["adj_open"] - 1.0
        return x
    ex, en = future(exits), future(entries)
    return ex.groupby("variant", as_index=False).agg(replacement_count=("ts_code", "size"), replaced_future_5d_return=("future_5d_return", "mean")).merge(en.groupby("variant", as_index=False).agg(new_entry_count=("ts_code", "size"), new_entry_future_5d_return=("future_5d_return", "mean")), on="variant", how="left")


def competition_dataset(states: pd.DataFrame, signals: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    base = states.loc[states["variant"].eq("baseline_fixed_top5_hold10")].copy()
    cand = signals.loc[signals["candidate_pool_flag"].eq(1), ["trade_date", "ts_code", "alpha_rank_pct"]].rename(columns={"trade_date": "signal_date", "ts_code": "candidate_stock", "alpha_rank_pct": "candidate_alpha_rank"})
    held = base.rename(columns={"ts_code": "holding_stock"})
    pairs = held.merge(cand, on="signal_date", how="inner")
    pairs = pairs.loc[pairs["holding_stock"].ne(pairs["candidate_stock"])].copy()
    return pairs[["signal_date", "holding_stock", "candidate_stock", "holding_days", "current_alpha_rank", "holding_rank_health", "candidate_alpha_rank"]]


def yearly_results(nav: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    nav = nav.copy(); nav["trade_date"] = pd.to_datetime(nav["trade_date"])
    trades = trades.copy(); trades["trade_date"] = pd.to_datetime(trades["trade_date"])
    for (variant, year), group in nav.groupby(["variant", nav["trade_date"].dt.year]):
        curve = (1.0 + group["return"]).cumprod()
        tx = trades.loc[(trades["variant"].eq(variant)) & (trades["trade_date"].dt.year.eq(year))]
        rows.append({"variant": variant, "year": int(year), "period_return": curve.iloc[-1] - 1.0, "period_max_drawdown": (curve / curve.cummax() - 1.0).min(), "buy_count": int(tx["side"].eq("BUY").sum()), "sell_count": int(tx["side"].eq("SELL").sum()), "annualized_turnover": group["portfolio_turnover"].mean() * 252})
    return pd.DataFrame(rows)


def as_table(frame: pd.DataFrame) -> str:
    return "_empty_" if frame.empty else frame.round(6).to_markdown(index=False)


def write_report(output: Path, summary: pd.DataFrame, yearly: pd.DataFrame, replacements: pd.DataFrame, competition: pd.DataFrame) -> None:
    lines = [
        "# Experiment 14: Position Competition, Clean Stage 1", "",
        "## Scope", "- This is an audit reset, not a performance claim. It uses only out-of-sample rows from the pre-existing PIT `raw_model` Alpha output.", "- Legacy Signal Reliability and Holding Health are deliberately quarantined because the former used a forward-label-derived feature. Their values are replaced by the neutral constant 0.5.", "- Holding quality is an observable rank-retention proxy: `clip(0.5 + current PIT Alpha rank - entry PIT Alpha rank, 0, 1)`.", "- Baseline is fixed before execution: top 10% PIT Alpha candidate pool, maximum 5 positions, 10 trading-day holding period, no timing, 20bps all-in costs.", "",
        "## Timing and Execution Audit", "- Signal is read at T close from the PIT output for T; all fills execute at T+1 `raw_open`.", "- Existing suspension, price-limit, ST, delisting-period and listing-age checks are enforced through the shared execution constraints.", "- `param_068` is intentionally not used: its prior selection used 2026H1 research results.", "",
        "## Results", as_table(summary.sort_values("variant")), "", "## Yearly Results", as_table(yearly.sort_values(["year", "variant"])), "", "## Replacement Forward Check", as_table(replacements), "",
        "## Controls", "- `raw_rank_only_m10` asks whether rank-retention health contributes beyond current raw PIT rank.", "- `random_competition_placebo_m10` preserves the same replacement trigger but randomly selects the entrant. It tests whether any result comes simply from turnover.", "",
        "## Decision Rule", "- Do not compare these results to the earlier 120% annualized output: that output is invalidated by a future-label leak and in-sample Alpha reconstruction.", "- If clean competition and `raw_rank_only_m10` are identical, the rank-retention proxy adds no incremental decision information; treat the observed result as raw Alpha re-ranking only.", "- If all pre-registered margins give identical paths, this rule has no demonstrated margin robustness. Do not select one by its aggregate return.", "- A future cleaned Signal Reliability model must be built and frozen separately before it can re-enter this experiment.", "",
        "## Files", "- `portfolio_nav.csv`, `trades.csv`, `position_scores.csv`", "- `portfolio_results.csv`, `yearly_results.csv`, `replacement_analysis.csv`, `competition_dataset.csv`",
    ]
    (output / "position_competition_clean_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
