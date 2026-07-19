from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factor_forge.research.concept_etf_shadow import (
    CASH,
    _aggregate_sleeve_nav,
    _serialize_weights,
    _turnover,
    staggered_target_weights,
)


EXIT_POLICIES = ("E0_fixed", "E1_price_confirmed", "E2_price_diffusion_state")


@dataclass(frozen=True)
class ExitStateRules:
    holding_days: int = 5
    execution_delay_days: int = 1
    retention_rank: int = 6
    severe_rank: int = 10
    diffusion_rank_max: float = 0.30
    confirmation_days: int = 2
    reduction_fraction: float = 0.50


def attach_exit_state_features(
    panel: pd.DataFrame,
    *,
    rules: ExitStateRules = ExitStateRules(),
) -> pd.DataFrame:
    """Attach causal close-known deterioration flags to the executable ETF panel."""
    result = panel.sort_values(["ts_code", "trade_date"]).copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    grouped = result.groupby("ts_code", sort=False)
    result["etf_return_5d_exit"] = grouped["adj_close"].pct_change(5, fill_method=None)
    result["momentum_acceleration_exit"] = (
        result["etf_return_5d_exit"] - result["etf_momentum_20d"] / 4.0
    )
    result["score_rank_exit"] = result.groupby("trade_date", sort=False)[
        "score_etf_momentum"
    ].rank(method="min", ascending=False)
    result["price_weak_exit"] = (
        result["score_rank_exit"].gt(rules.retention_rank)
        & result["momentum_acceleration_exit"].lt(0.0)
    )
    result["diffusion_weak_exit"] = (
        result["common_delta_rank"].le(rules.diffusion_rank_max)
        & result["common_breadth_delta_smooth5"].lt(0.0)
    )
    result["relative_weak_exit"] = (
        result["rs_momentum_5d"].lt(0.0)
        & result["rrg_quadrant"].isin(["weakening", "lagging"])
    )
    result["absolute_breakdown_exit"] = result["etf_momentum_60d"].le(0.0)
    return result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def classify_exit_state(
    row: pd.Series,
    *,
    policy: str,
    price_streak: int,
    dual_streak: int,
    severe_streak: int,
    already_reduced: bool,
    rules: ExitStateRules = ExitStateRules(),
) -> dict:
    if policy not in EXIT_POLICIES:
        raise ValueError(f"unknown exit policy: {policy}")
    price_weak = bool(row.get("price_weak_exit", False))
    diffusion_weak = bool(row.get("diffusion_weak_exit", False))
    relative_weak = bool(row.get("relative_weak_exit", False))
    absolute_breakdown = bool(row.get("absolute_breakdown_exit", False))
    severe_rank = float(row.get("score_rank_exit", np.nan)) > rules.severe_rank

    price_streak = price_streak + 1 if price_weak else 0
    dual_streak = dual_streak + 1 if price_weak and diffusion_weak else 0
    severe_streak = (
        severe_streak + 1
        if price_weak and diffusion_weak and relative_weak and severe_rank
        else 0
    )
    reasons: list[str] = []
    if price_weak:
        reasons.append("momentum_rank_and_acceleration_weakened")
    if diffusion_weak:
        reasons.append("internal_diffusion_weakened")
    if relative_weak:
        reasons.append("relative_strength_weakened")
    if absolute_breakdown:
        reasons.append("absolute_60d_momentum_nonpositive")

    status, action = "HOLD", "none"
    if policy != "E0_fixed":
        if absolute_breakdown:
            status, action = "SELL", "sell_all"
        elif policy == "E2_price_diffusion_state" and severe_streak >= rules.confirmation_days:
            status, action = "SELL", "sell_all"
        elif (
            policy == "E1_price_confirmed"
            and price_streak >= rules.confirmation_days
            and not already_reduced
        ):
            status, action = "REDUCE", "reduce_half"
        elif (
            policy == "E2_price_diffusion_state"
            and dual_streak >= rules.confirmation_days
            and not already_reduced
        ):
            status, action = "REDUCE", "reduce_half"
        elif reasons:
            status = "WATCH"
    return {
        "status": status,
        "action": action,
        "reasons": ",".join(reasons) if reasons else "signal_intact",
        "price_streak": price_streak,
        "dual_streak": dual_streak,
        "severe_streak": severe_streak,
    }


def simulate_exit_state_sleeves(
    panel: pd.DataFrame,
    policy: str,
    *,
    start: str,
    end: str,
    roundtrip_cost_bps: float,
    rules: ExitStateRules = ExitStateRules(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if policy not in EXIT_POLICIES:
        raise ValueError(f"unknown exit policy: {policy}")
    frame = attach_exit_state_features(panel, rules=rules)
    full_calendar = pd.Index(sorted(frame["trade_date"].unique()))
    active_calendar = pd.Index([
        date for date in full_calendar if pd.Timestamp(start) <= date <= pd.Timestamp(end)
    ])
    positions = {date: index for index, date in enumerate(full_calendar)}
    next_date = {
        date: full_calendar[index + 1]
        for date, index in positions.items()
        if index + 1 < len(full_calendar)
    }
    delayed_execution = {
        date: full_calendar[index + rules.execution_delay_days]
        for date, index in positions.items()
        if index + rules.execution_delay_days < len(full_calendar)
    }
    prices = frame.pivot(index="trade_date", columns="ts_code", values="adj_open").sort_index()
    lookup = frame.set_index(["trade_date", "ts_code"])
    sleeve_parts: list[pd.DataFrame] = []
    signal_rows: list[dict] = []
    action_rows: list[dict] = []

    for sleeve in range(rules.holding_days):
        signal_dates = active_calendar[sleeve::rules.holding_days]
        executions = {
            pd.Timestamp(delayed_execution[signal_date]): pd.Timestamp(signal_date)
            for signal_date in signal_dates
            if signal_date in delayed_execution
            and delayed_execution[signal_date] <= pd.Timestamp(end)
        }
        scheduled_dates = sorted(executions)
        pending: dict[pd.Timestamp, dict[str, str]] = {}
        streaks: dict[str, dict[str, int]] = {}
        reduced_codes: set[str] = set()
        pretrade = {CASH: 1.0}
        sleeve_nav = 1.0
        holding_day = 0
        rows: list[dict] = []

        for date in active_calendar:
            if date not in next_date or next_date[date] > pd.Timestamp(end):
                continue
            date = pd.Timestamp(date)
            following = pd.Timestamp(next_date[date])
            signal_date = executions.get(date)
            target = dict(pretrade)
            scheduled_rebalance = signal_date is not None
            if scheduled_rebalance:
                previous_holdings = {
                    code for code, weight in pretrade.items() if code != CASH and weight > 0
                }
                target = staggered_target_weights(
                    frame.loc[frame["trade_date"].eq(signal_date)],
                    "R4_rank_buffer",
                    previous_holdings=previous_holdings,
                )
                reduced_codes.clear()
                streaks.clear()
                holding_day = 1
            else:
                holding_day += 1

            action_for_open = pending.pop(date, {})
            before_actions = dict(target)
            for code, action in action_for_open.items():
                weight = float(target.get(code, 0.0))
                if weight <= 0:
                    continue
                if action == "sell_all":
                    target.pop(code, None)
                    target[CASH] = float(target.get(CASH, 0.0)) + weight
                    reduced_codes.discard(code)
                    new_weight = 0.0
                elif action == "reduce_half" and code not in reduced_codes:
                    reduction = weight * rules.reduction_fraction
                    target[code] = weight - reduction
                    target[CASH] = float(target.get(CASH, 0.0)) + reduction
                    reduced_codes.add(code)
                    new_weight = target[code]
                else:
                    continue
                future_scheduled = [item for item in scheduled_dates if item > date]
                action_rows.append({
                    "policy": policy,
                    "sleeve": sleeve,
                    "signal_date": full_calendar[positions[date] - 1],
                    "execution_date": date,
                    "scheduled_review_date": future_scheduled[0] if future_scheduled else pd.NaT,
                    "ts_code": code,
                    "action": action,
                    "weight_before": weight,
                    "weight_after": new_weight,
                })

            turnover = _turnover(pretrade, target) if target != before_actions or scheduled_rebalance else 0.0
            returns = (prices.loc[following] / prices.loc[date] - 1.0).to_dict()
            gross = sum(
                weight * returns.get(code, 0.0)
                for code, weight in target.items()
                if code != CASH and weight > 0
            )
            cost = turnover * roundtrip_cost_bps / 10_000.0
            net = gross - cost
            nav_before = sleeve_nav
            sleeve_nav *= 1.0 + net

            held_codes = [
                code for code, weight in target.items() if code != CASH and weight > 0
            ]
            next_actions: dict[str, str] = {}
            future_scheduled = [item for item in scheduled_dates if item > following]
            next_review = future_scheduled[0] if future_scheduled else pd.NaT
            for code in held_codes:
                if (date, code) not in lookup.index:
                    continue
                state = streaks.setdefault(code, {"price": 0, "dual": 0, "severe": 0})
                classified = classify_exit_state(
                    lookup.loc[(date, code)],
                    policy=policy,
                    price_streak=state["price"],
                    dual_streak=state["dual"],
                    severe_streak=state["severe"],
                    already_reduced=code in reduced_codes,
                    rules=rules,
                )
                state.update({
                    "price": classified["price_streak"],
                    "dual": classified["dual_streak"],
                    "severe": classified["severe_streak"],
                })
                if classified["action"] != "none":
                    next_actions[code] = classified["action"]
                item = lookup.loc[(date, code)]
                signal_rows.append({
                    "policy": policy,
                    "sleeve": sleeve,
                    "signal_date": date,
                    "execution_date": following,
                    "scheduled_review_date": next_review,
                    "holding_day": holding_day,
                    "ts_code": code,
                    "etf_name": item.get("etf_name"),
                    "concept_name": item.get("concept_name"),
                    "current_weight": float(target[code]),
                    "status": classified["status"],
                    "planned_action": classified["action"],
                    "reasons": classified["reasons"],
                    "score_rank": item.get("score_rank_exit"),
                    "etf_momentum_20d": item.get("etf_momentum_20d"),
                    "etf_momentum_60d": item.get("etf_momentum_60d"),
                    "momentum_acceleration": item.get("momentum_acceleration_exit"),
                    "common_delta_rank": item.get("common_delta_rank"),
                    "common_breadth_delta_smooth5": item.get("common_breadth_delta_smooth5"),
                    "rs_momentum_5d": item.get("rs_momentum_5d"),
                    "rrg_quadrant": item.get("rrg_quadrant"),
                    "price_streak": classified["price_streak"],
                    "dual_streak": classified["dual_streak"],
                    "severe_streak": classified["severe_streak"],
                })
            pending[following] = next_actions
            active = set(held_codes)
            streaks = {code: state for code, state in streaks.items() if code in active}
            reduced_codes.intersection_update(active)

            rows.append({
                "return_date": following,
                "holding_date": date,
                "signal_date": signal_date,
                "portfolio": policy,
                "variant": policy,
                "sleeve": sleeve,
                "gross_return": gross,
                "net_return": net,
                "net_nav": sleeve_nav,
                "nav_before": nav_before,
                "turnover": turnover,
                "cost_drag": cost,
                "cash_weight": target.get(CASH, 0.0),
                "is_rebalance": scheduled_rebalance,
                "is_trade": scheduled_rebalance or bool(action_for_open),
                "target_weights": _serialize_weights(target),
            })
            denominator = 1.0 + gross
            pretrade = {
                code: weight * (1.0 + returns.get(code, 0.0)) / denominator
                for code, weight in target.items()
            }
        sleeve_parts.append(pd.DataFrame(rows))

    sleeve_daily = pd.concat(sleeve_parts, ignore_index=True)
    aggregate = _aggregate_sleeve_nav(sleeve_daily, policy)
    aggregate["is_trade"] = aggregate["return_date"].isin(
        sleeve_daily.loc[sleeve_daily["is_trade"], "return_date"]
    )
    signals = pd.DataFrame(signal_rows)
    actions = attach_action_outcomes(pd.DataFrame(action_rows), prices)
    return aggregate, sleeve_daily, signals, actions


def attach_action_outcomes(actions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        return actions.assign(
            remaining_return_to_scheduled=np.nan,
            false_exit=False,
            avoided_remaining_return=np.nan,
        )
    result = actions.copy()
    remaining = []
    for item in result.itertuples(index=False):
        if pd.isna(item.scheduled_review_date):
            remaining.append(np.nan)
            continue
        start = prices.at[pd.Timestamp(item.execution_date), item.ts_code]
        end = prices.at[pd.Timestamp(item.scheduled_review_date), item.ts_code]
        remaining.append(float(end / start - 1.0) if np.isfinite(start) and np.isfinite(end) else np.nan)
    result["remaining_return_to_scheduled"] = remaining
    result["false_exit"] = result["remaining_return_to_scheduled"].gt(0.0)
    result["avoided_remaining_return"] = -result["remaining_return_to_scheduled"]
    return result
