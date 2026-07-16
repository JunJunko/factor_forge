from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from factor_forge.research.concept_rotation_alpha import (
    newey_west_mean,
    select_deduplicated_concepts,
)


SIGNAL_COLUMNS = {
    "common_membership_breadth_rrg": "signal_common_breadth_rrg",
    "current_membership_breadth_rrg": "signal_current_breadth_rrg",
    "rrg_only": "signal_rrg_only",
    "momentum_20d": "signal_momentum_20d",
    "common_breadth_delta": "signal_common_breadth_delta",
    "common_breadth_residual": "signal_common_breadth_residual",
    "rrg_plus_common_breadth_residual": "signal_rrg_plus_common_breadth_residual",
    "common_breadth_residual_placebo": "signal_common_breadth_residual_placebo",
    "rrg_plus_common_breadth_residual_placebo": "signal_rrg_plus_common_breadth_residual_placebo",
}


@dataclass(frozen=True)
class PortfolioRules:
    holding_days: int = 5
    concepts_per_rebalance: int = 10
    stocks_per_concept: int = 3
    concept_preselect: int = 30
    jaccard_limit: float = 0.80
    minimum_adv20_cny: float = 20_000_000.0
    maximum_stock_weight: float = 0.05
    lot_size: int = 100
    initial_cash: float = 10_000_000.0


@dataclass(frozen=True)
class ExecutionRules:
    # Brokerage commission is treated as the all-in quoted commission. Exchange
    # handling/regulatory charges must not be added again when already included.
    commission_bps_per_side: float = 2.5
    minimum_commission_cny: float = 5.0
    transfer_fee_bps_per_side: float = 0.10
    stamp_duty_bps_sell: float = 5.0
    base_slippage_bps_per_side: float = 3.0
    impact_eta: float = 0.15
    maximum_impact_bps: float = 100.0
    maximum_adv_participation: float = 0.05
    cost_multiplier: float = 1.0
    extra_slippage_bps_per_side: float = 0.0


@dataclass
class TargetBuildResult:
    targets: pd.DataFrame
    selections: pd.DataFrame
    rebalance_dates: list[pd.Timestamp]
    entry_dates: list[pd.Timestamp]


@dataclass
class LedgerResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    metrics: dict[str, Any]


def prepare_execution_panel(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.sort_values(["ts_code", "trade_date"]).copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    grouped = result.groupby("ts_code", sort=False)
    close_return = grouped["adj_close"].pct_change(fill_method=None)
    result["amount_ma20"] = grouped["amount_cny"].transform(
        lambda values: values.rolling(20, min_periods=18).mean()
    )
    result["volatility_20d"] = close_return.groupby(result["ts_code"]).transform(
        lambda values: values.rolling(20, min_periods=18).std()
    ).groupby(result["ts_code"]).shift(1)
    return result


def attach_continuous_breadth_signals(features: pd.DataFrame) -> pd.DataFrame:
    """Residualize breadth cross-sectionally without using any future returns."""
    result = features.copy()
    result["signal_common_breadth_residual"] = np.nan
    result["signal_common_breadth_residual_placebo"] = np.nan
    controls = ["rs_z", "rs_momentum_z", "membership_churn_5d", "matched_member_count"]
    for date, indices in result.groupby("trade_date", sort=True).groups.items():
        day = result.loc[indices]
        y = pd.to_numeric(day["common_breadth_delta_smooth5"], errors="coerce")
        x = day[controls].apply(pd.to_numeric, errors="coerce").copy()
        x["matched_member_count"] = np.log1p(x["matched_member_count"].clip(lower=0))
        valid = y.notna() & x.notna().all(axis=1)
        if valid.sum() < 30:
            continue
        xv = x.loc[valid]
        std = xv.std(ddof=0).replace(0, np.nan)
        xv = ((xv - xv.mean()) / std).fillna(0.0)
        design = np.column_stack([np.ones(len(xv)), xv.to_numpy(float)])
        beta = np.linalg.lstsq(design, y.loc[valid].to_numpy(float), rcond=None)[0]
        residual = y.loc[valid].to_numpy(float) - design @ beta
        residual_std = residual.std(ddof=0)
        residual_z = residual / residual_std if residual_std > 0 else np.zeros_like(residual)
        valid_indices = day.index[valid]
        result.loc[valid_indices, "signal_common_breadth_residual"] = residual_z
        rng = np.random.default_rng(int(pd.Timestamp(date).strftime("%Y%m%d")))
        result.loc[valid_indices, "signal_common_breadth_residual_placebo"] = rng.permutation(residual_z)
    result["signal_rrg_plus_common_breadth_residual"] = (
        result["signal_rrg_only"] + result["signal_common_breadth_residual"]
    )
    result["signal_rrg_plus_common_breadth_residual_placebo"] = (
        result["signal_rrg_only"] + result["signal_common_breadth_residual_placebo"]
    )
    return result


def build_market_regimes(stock_panel: pd.DataFrame) -> pd.DataFrame:
    """Create fixed, ex-ante market breadth regimes from information known at T close."""
    stocks = stock_panel.sort_values(["ts_code", "trade_date"]).copy()
    grouped = stocks.groupby("ts_code", sort=False)
    stocks["return_1d"] = grouped["adj_close"].pct_change(fill_method=None)
    stocks["return_20d"] = grouped["adj_close"].pct_change(20, fill_method=None)
    stocks["cap_lag1"] = grouped["circ_mv_cny"].shift(1) if "circ_mv_cny" in stocks else 1.0
    valid = stocks["is_tradeable"].fillna(False) & stocks["return_20d"].notna()
    breadth = stocks.loc[valid].groupby("trade_date")["return_20d"].agg(
        market_breadth=lambda values: float(values.gt(0).mean())
    )
    market = stocks.loc[
        stocks["is_tradeable"].fillna(False) & stocks["return_1d"].notna()
        & pd.to_numeric(stocks["cap_lag1"], errors="coerce").gt(0)
    ].copy()
    market["weighted_return"] = market["return_1d"] * market["cap_lag1"]
    market_return = market.groupby("trade_date").apply(
        lambda day: day["weighted_return"].sum() / day["cap_lag1"].sum(),
        include_groups=False,
    ).rename("market_return_1d")
    regime = breadth.join(market_return, how="outer").sort_index().reset_index()
    regime["breadth_delta_5d"] = regime["market_breadth"] - regime["market_breadth"].shift(5)
    regime["market_return_20d"] = np.expm1(
        np.log1p(regime["market_return_1d"].clip(lower=-0.999999)).rolling(20, min_periods=20).sum()
    )
    regime["regime"] = np.select(
        [
            regime["market_return_20d"].gt(0)
            & regime["breadth_delta_5d"].gt(0)
            & regime["market_breadth"].between(0.30, 0.70, inclusive="left"),
            regime["market_return_20d"].gt(0) & regime["market_breadth"].ge(0.70),
        ],
        ["repair", "overheat"], default="retreat",
    )
    return regime


def build_liquidity_neutral_targets(
    features: pd.DataFrame,
    members: pd.DataFrame,
    stock_panel: pd.DataFrame,
    *,
    signal_name: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    offset: int,
    rules: PortfolioRules = PortfolioRules(),
) -> TargetBuildResult:
    if signal_name not in SIGNAL_COLUMNS:
        raise KeyError(f"unknown concept signal: {signal_name}")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    calendar = pd.Index(sorted(
        stock_panel.loc[stock_panel["trade_date"].between(start, end), "trade_date"].unique()
    ))
    if not 0 <= offset < rules.holding_days:
        raise ValueError("offset must be between zero and holding_days - 1")
    signal_dates = []
    entry_dates = []
    exit_dates = []
    for signal_position in range(offset, len(calendar), rules.holding_days):
        entry_position = signal_position + 1
        exit_position = entry_position + rules.holding_days
        if exit_position >= len(calendar):
            break
        signal_dates.append(pd.Timestamp(calendar[signal_position]))
        entry_dates.append(pd.Timestamp(calendar[entry_position]))
        exit_dates.append(pd.Timestamp(calendar[exit_position]))
    if not signal_dates:
        return TargetBuildResult(
            pd.DataFrame(columns=["signal_date", "entry_date", "ts_code", "target_weight"]),
            pd.DataFrame(), [], [],
        )

    signal_column = SIGNAL_COLUMNS[signal_name]
    period = features.loc[
        features["trade_date"].isin(signal_dates) & features["eligible_concept"].fillna(False),
        ["trade_date", "concept_code", signal_column],
    ].rename(columns={signal_column: "active_score"})
    period_members = members.loc[members["trade_date"].isin(signal_dates), [
        "trade_date", "concept_code", "ts_code",
    ]]
    selected = select_deduplicated_concepts(
        period, period_members, "active_score",
        top_n=rules.concepts_per_rebalance,
        preselect=rules.concept_preselect,
        jaccard_limit=rules.jaccard_limit,
    ).rename(columns={"trade_date": "signal_date", "score": "concept_score"})
    if selected.empty:
        return TargetBuildResult(
            pd.DataFrame(columns=["signal_date", "entry_date", "ts_code", "target_weight"]),
            selected, sorted(set(entry_dates + exit_dates)), entry_dates,
        )

    liquidity = stock_panel.loc[stock_panel["trade_date"].isin(signal_dates), [
        "trade_date", "ts_code", "amount_ma20", "is_tradeable",
    ]].rename(columns={"trade_date": "signal_date"})
    candidates = selected.merge(
        period_members.rename(columns={"trade_date": "signal_date"}),
        on=["signal_date", "concept_code"], how="inner", validate="one_to_many",
    ).merge(liquidity, on=["signal_date", "ts_code"], how="inner", validate="many_to_one")
    candidates = candidates.loc[
        candidates["is_tradeable"].fillna(False)
        & candidates["amount_ma20"].ge(rules.minimum_adv20_cny)
    ].sort_values(
        ["signal_date", "concept_code", "amount_ma20", "ts_code"],
        ascending=[True, True, False, True],
    ).groupby(["signal_date", "concept_code"], observed=True).head(rules.stocks_per_concept)
    selected_counts = selected.groupby("signal_date", observed=True)["concept_code"].nunique()
    stock_counts = candidates.groupby(
        ["signal_date", "concept_code"], observed=True
    )["ts_code"].transform("nunique")
    candidates["raw_weight"] = (
        candidates["signal_date"].map(1 / selected_counts) / stock_counts
    )
    targets = candidates.groupby(["signal_date", "ts_code"], observed=True).agg(
        raw_weight=("raw_weight", "sum"),
        supporting_concepts=("concept_code", "nunique"),
        amount_ma20=("amount_ma20", "max"),
    ).reset_index()
    targets["target_weight"] = targets.groupby("signal_date", group_keys=False)["raw_weight"].transform(
        lambda values: cap_and_redistribute(values, rules.maximum_stock_weight)
    )
    entry_map = dict(zip(signal_dates, entry_dates))
    targets["entry_date"] = targets["signal_date"].map(entry_map)
    targets = targets[[
        "signal_date", "entry_date", "ts_code", "target_weight",
        "supporting_concepts", "amount_ma20",
    ]].sort_values(["entry_date", "target_weight"], ascending=[True, False])
    rebalance_dates = sorted(set(entry_dates + exit_dates))
    return TargetBuildResult(targets, selected, rebalance_dates, entry_dates)


def cap_and_redistribute(weights: pd.Series, cap: float) -> pd.Series:
    values = pd.to_numeric(weights, errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
    if values.sum() <= 0 or cap <= 0:
        return pd.Series(0.0, index=weights.index)
    values /= values.sum()
    if len(values) * cap < 1:
        values = np.minimum(values, cap)
        return pd.Series(values, index=weights.index)
    free = np.ones(len(values), dtype=bool)
    output = np.zeros(len(values), dtype=float)
    remaining = 1.0
    while free.any():
        source = values[free]
        allocation = remaining * source / source.sum() if source.sum() else np.full(source.size, remaining / source.size)
        over = allocation > cap + 1e-12
        free_indices = np.flatnonzero(free)
        if not over.any():
            output[free_indices] = allocation
            break
        capped_indices = free_indices[over]
        output[capped_indices] = cap
        free[capped_indices] = False
        remaining = 1 - output.sum()
    return pd.Series(output, index=weights.index)


def rescale_target_build(
    target_build: TargetBuildResult,
    exposure_by_entry: dict[pd.Timestamp, float] | float,
    *, maximum_stock_weight: float,
) -> TargetBuildResult:
    targets = target_build.targets.copy()
    if targets.empty:
        return TargetBuildResult(targets, target_build.selections, target_build.rebalance_dates, target_build.entry_dates)
    frames = []
    for entry_date, frame in targets.groupby("entry_date", sort=True):
        exposure = (
            float(exposure_by_entry.get(pd.Timestamp(entry_date), 0.0))
            if isinstance(exposure_by_entry, dict) else float(exposure_by_entry)
        )
        exposure = float(np.clip(exposure, 0.0, 1.0))
        item = frame.copy()
        if exposure <= 0:
            item["target_weight"] = 0.0
        else:
            relative_cap = min(maximum_stock_weight / exposure, 1.0)
            item["target_weight"] = cap_and_redistribute(
                item["target_weight"], relative_cap
            ).to_numpy() * exposure
        frames.append(item)
    targets = pd.concat(frames, ignore_index=True)
    targets = targets.loc[targets["target_weight"].gt(0)]
    return TargetBuildResult(
        targets, target_build.selections.copy(),
        list(target_build.rebalance_dates), list(target_build.entry_dates),
    )


def blend_target_builds(
    primary: TargetBuildResult,
    baseline: TargetBuildResult,
    regime_by_signal_date: dict[pd.Timestamp, str],
    *, active_regime: str = "repair",
) -> TargetBuildResult:
    primary_targets = primary.targets.copy()
    baseline_targets = baseline.targets.copy()
    primary_targets["_regime"] = primary_targets["signal_date"].map(regime_by_signal_date)
    baseline_targets["_regime"] = baseline_targets["signal_date"].map(regime_by_signal_date)
    targets = pd.concat([
        primary_targets.loc[primary_targets["_regime"].eq(active_regime)],
        baseline_targets.loc[~baseline_targets["_regime"].eq(active_regime)],
    ], ignore_index=True).drop(columns="_regime")
    selections = pd.concat([primary.selections, baseline.selections], ignore_index=True).drop_duplicates()
    return TargetBuildResult(
        targets, selections,
        sorted(set(primary.rebalance_dates) | set(baseline.rebalance_dates)),
        sorted(set(primary.entry_dates) | set(baseline.entry_dates)),
    )


def run_non_overlapping_ledger(
    panel: pd.DataFrame,
    target_build: TargetBuildResult,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    portfolio_rules: PortfolioRules = PortfolioRules(),
    execution_rules: ExecutionRules = ExecutionRules(),
) -> LedgerResult:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    relevant_codes = set(target_build.targets.get("ts_code", pd.Series(dtype=str)).astype(str))
    data = panel.loc[
        panel["trade_date"].between(start, end) & panel["ts_code"].astype(str).isin(relevant_codes)
    ].copy()
    dates = list(pd.Index(data["trade_date"].unique()).sort_values())
    by_date = {pd.Timestamp(date): frame.set_index("ts_code") for date, frame in data.groupby("trade_date")}
    targets_by_entry = {
        pd.Timestamp(date): frame.set_index("ts_code")["target_weight"].to_dict()
        for date, frame in target_build.targets.groupby("entry_date")
    }
    rebalance_dates = set(pd.Timestamp(date) for date in target_build.rebalance_dates)
    if not rebalance_dates:
        return LedgerResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), empty_metrics())

    first_date, last_date = min(rebalance_dates), end
    active_dates = [pd.Timestamp(date) for date in dates if first_date <= pd.Timestamp(date) <= last_date]
    cash = portfolio_rules.initial_cash
    units: dict[str, float] = {}
    pending_units: dict[str, float] = {}
    last_marks: dict[str, float] = {}
    daily_rows: list[dict] = []
    trade_rows: list[dict] = []
    position_rows: list[dict] = []

    for date in active_dates:
        today = by_date[date]
        for code, row in today.iterrows():
            mark = row.get("adj_open")
            if np.isfinite(mark):
                last_marks[code] = float(mark)
        nav_before = cash + sum(
            quantity * last_marks.get(code, 0.0) for code, quantity in units.items()
        )
        if date in rebalance_dates:
            weights = targets_by_entry.get(date, {})
            all_codes = set(units) | set(weights)
            pending_units = {}
            for code in all_codes:
                mark = _mark(today, code, last_marks)
                if not np.isfinite(mark) or mark <= 0:
                    continue
                desired_units = nav_before * weights.get(code, 0.0) / mark
                difference = desired_units - units.get(code, 0.0)
                if abs(difference * mark) >= 1.0:
                    pending_units[code] = difference

        turnover = total_cost = 0.0
        blocked_buys = blocked_sells = 0
        participation_values = []
        for side in ("SELL", "BUY"):
            codes = [
                code for code, value in pending_units.items()
                if (value < -1e-12 if side == "SELL" else value > 1e-12)
            ]
            for code in sorted(codes):
                row = _row(today, code)
                if row is None or not _can_execute(row, side):
                    if side == "SELL": blocked_sells += 1
                    else: blocked_buys += 1
                    continue
                mark = float(row["adj_open"])
                raw_open = float(row["raw_open"])
                adv20 = float(row.get("amount_ma20", np.nan))
                volatility = float(row.get("volatility_20d", np.nan))
                if not np.isfinite(adv20) or adv20 <= 0:
                    continue
                desired_notional = abs(pending_units[code]) * mark
                capacity = execution_rules.maximum_adv_participation * adv20
                notional = min(desired_notional, capacity)
                full_exit = side == "SELL" and desired_notional >= units.get(code, 0.0) * mark - 1.0 and capacity >= desired_notional
                if not full_exit:
                    shares = math.floor(notional / (raw_open * portfolio_rules.lot_size)) * portfolio_rules.lot_size
                    notional = shares * raw_open
                if notional <= 0:
                    continue
                impact_bps = execution_impact_bps(
                    volatility, notional / adv20, execution_rules
                )
                cost, cost_parts = transaction_cost(notional, side, impact_bps, execution_rules)
                if side == "BUY" and notional + cost > cash:
                    affordable = max(cash - execution_rules.minimum_commission_cny, 0)
                    shares = math.floor(affordable / (raw_open * portfolio_rules.lot_size)) * portfolio_rules.lot_size
                    notional = shares * raw_open
                    if notional <= 0:
                        continue
                    impact_bps = execution_impact_bps(volatility, notional / adv20, execution_rules)
                    cost, cost_parts = transaction_cost(notional, side, impact_bps, execution_rules)
                    if notional + cost > cash:
                        continue
                traded_units = notional / mark
                if side == "SELL":
                    traded_units = min(traded_units, units.get(code, 0.0))
                    notional = traded_units * mark
                    cost, cost_parts = transaction_cost(notional, side, impact_bps, execution_rules)
                    units[code] = units.get(code, 0.0) - traded_units
                    cash += notional - cost
                    pending_units[code] += traded_units
                else:
                    units[code] = units.get(code, 0.0) + traded_units
                    cash -= notional + cost
                    pending_units[code] -= traded_units
                if units.get(code, 0.0) * mark < 1.0:
                    units.pop(code, None)
                if abs(pending_units.get(code, 0.0) * mark) < raw_open * portfolio_rules.lot_size:
                    pending_units.pop(code, None)
                turnover += notional
                total_cost += cost
                participation_values.append(notional / adv20)
                trade_rows.append({
                    "trade_date": date, "ts_code": code, "side": side,
                    "notional": notional, "cost": cost, "impact_bps": impact_bps,
                    "participation": notional / adv20, "raw_open": raw_open,
                    **cost_parts,
                })

        nav = cash + sum(quantity * _mark(today, code, last_marks) for code, quantity in units.items())
        previous_nav = daily_rows[-1]["nav"] if daily_rows else portfolio_rules.initial_cash
        values = {
            code: quantity * _mark(today, code, last_marks) for code, quantity in units.items()
        }
        for code, value in values.items():
            position_rows.append({
                "trade_date": date, "ts_code": code, "market_value": value,
                "weight": value / nav if nav > 0 else 0.0,
            })
        daily_rows.append({
            "trade_date": date, "nav": nav, "return": nav / previous_nav - 1,
            "cash": cash, "cash_ratio": cash / nav if nav > 0 else 0.0,
            "holding_count": len(units),
            "largest_position_weight": max(values.values()) / nav if values and nav > 0 else 0.0,
            "turnover": turnover / previous_nav if previous_nav > 0 else 0.0,
            "transaction_cost": total_cost,
            "blocked_buys": blocked_buys, "blocked_sells": blocked_sells,
            "pending_orders": len(pending_units),
            "maximum_participation": max(participation_values, default=0.0),
        })

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)
    positions = pd.DataFrame(position_rows)
    return LedgerResult(daily, trades, positions, ledger_metrics(daily, trades))


def execution_impact_bps(
    volatility_20d: float, participation: float, rules: ExecutionRules,
) -> float:
    volatility = volatility_20d if np.isfinite(volatility_20d) and volatility_20d > 0 else 0.02
    modeled = rules.impact_eta * volatility * math.sqrt(max(participation, 0)) * 10_000
    return min(
        rules.base_slippage_bps_per_side + rules.extra_slippage_bps_per_side + modeled,
        rules.maximum_impact_bps,
    )


def transaction_cost(
    notional: float, side: str, impact_bps: float, rules: ExecutionRules,
) -> tuple[float, dict[str, float]]:
    commission = max(
        rules.minimum_commission_cny,
        notional * rules.commission_bps_per_side / 10_000,
    )
    transfer = notional * rules.transfer_fee_bps_per_side / 10_000
    stamp = notional * rules.stamp_duty_bps_sell / 10_000 if side == "SELL" else 0.0
    impact = notional * impact_bps / 10_000
    components = {
        "commission_cost": commission * rules.cost_multiplier,
        "transfer_cost": transfer * rules.cost_multiplier,
        "stamp_cost": stamp * rules.cost_multiplier,
        "impact_cost": impact * rules.cost_multiplier,
    }
    return sum(components.values()), components


def ledger_metrics(daily: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    if daily.empty:
        return empty_metrics()
    returns = daily["return"].fillna(0.0)
    total = float((1 + returns).prod() - 1)
    periods = max(len(daily) - 1, 1)
    annual = (1 + total) ** (252 / periods) - 1 if total > -1 else -1.0
    volatility = returns.std(ddof=1) * math.sqrt(252) if len(returns) > 1 else 0.0
    drawdown = daily["nav"] / daily["nav"].cummax() - 1
    return {
        "days": int(len(daily)), "total_return": float(total),
        "annualized_return": float(annual), "annualized_volatility": float(volatility),
        "sharpe": float(annual / volatility) if volatility > 0 else None,
        "max_drawdown": float(drawdown.min()),
        "annualized_turnover": float(daily["turnover"].sum() * 252 / periods),
        "average_cash_ratio": float(daily["cash_ratio"].mean()),
        "maximum_position_weight": float(daily["largest_position_weight"].max()),
        "transaction_cost_cny": float(daily["transaction_cost"].sum()),
        "cost_drag": float(daily["transaction_cost"].sum() / daily["nav"].iloc[0]),
        "trades": int(len(trades)),
        "blocked_buys": int(daily["blocked_buys"].sum()),
        "blocked_sells": int(daily["blocked_sells"].sum()),
        "p95_participation": float(trades["participation"].quantile(0.95)) if len(trades) else 0.0,
    }


def paired_portfolio_comparison(
    daily_results: dict[tuple[str, str, int, str], pd.DataFrame],
    *, primary: str = "common_membership_breadth_rrg", baseline: str = "rrg_only",
) -> pd.DataFrame:
    rows = []
    keys = {(split, offset, scenario) for signal, split, offset, scenario in daily_results}
    for split, offset, scenario in sorted(keys):
        left = daily_results.get((primary, split, offset, scenario))
        right = daily_results.get((baseline, split, offset, scenario))
        if left is None or right is None or left.empty or right.empty:
            continue
        joined = left[["trade_date", "return"]].merge(
            right[["trade_date", "return"]], on="trade_date", suffixes=("_primary", "_baseline")
        )
        difference = joined["return_primary"] - joined["return_baseline"]
        stats = newey_west_mean(difference, 4)
        relative_total = (
            (1 + joined["return_primary"]).prod() / (1 + joined["return_baseline"]).prod() - 1
        )
        rows.append({
            "split": split, "offset": offset, "scenario": scenario,
            "primary": primary, "baseline": baseline, "days": len(joined),
            "relative_total_return": relative_total,
            "mean_daily_increment": stats["mean"], "incremental_nw_t": stats["t_value"],
        })
    return pd.DataFrame(rows)


def _row(today: pd.DataFrame, code: str) -> pd.Series | None:
    if code not in today.index:
        return None
    row = today.loc[code]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def _mark(today: pd.DataFrame, code: str, last_marks: dict[str, float]) -> float:
    row = _row(today, code)
    if row is not None:
        value = row.get("adj_open")
        if np.isfinite(value):
            return float(value)
        value = row.get("adj_close")
        if np.isfinite(value):
            return float(value)
    return float(last_marks.get(code, 0.0))


def _can_execute(row: pd.Series, side: str) -> bool:
    if not np.isfinite(row.get("raw_open", np.nan)) or not np.isfinite(row.get("adj_open", np.nan)):
        return False
    if bool(row.get("is_suspended", True)):
        return False
    if side == "BUY":
        return bool(
            not row.get("is_limit_up_open", False)
            and not row.get("is_st", False)
            and not row.get("is_delisting_period", False)
            and row.get("listing_trade_days", 0) >= 60
        )
    return not bool(row.get("is_limit_down_open", False))


def empty_metrics() -> dict[str, Any]:
    return {
        "days": 0, "total_return": None, "annualized_return": None,
        "annualized_volatility": None, "sharpe": None, "max_drawdown": None,
        "annualized_turnover": None, "average_cash_ratio": None,
        "maximum_position_weight": None, "transaction_cost_cny": 0.0,
        "cost_drag": None, "trades": 0, "blocked_buys": 0, "blocked_sells": 0,
        "p95_participation": None,
    }
