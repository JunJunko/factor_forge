from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factor_forge.research.concept_etf_exit_ml import parse_weights
from factor_forge.research.concept_etf_shadow import (
    CASH,
    staggered_target_weights,
)


COORDINATED_R4_VARIANTS = (
    "R4_A_base",
    "R4_B_concentration",
    "R4_C_correlation",
)


@dataclass(frozen=True)
class CoordinatedR4Rules:
    maximum_sleeves_per_etf: int = 3
    maximum_aggregate_etf_weight: float = 0.20
    maximum_aggregate_cluster_weight: float = 0.30
    correlation_window: int = 60
    correlation_minimum_observations: int = 40
    maximum_pairwise_correlation: float = 0.75


def simulate_coordinated_r4(
    panel: pd.DataFrame,
    variant: str,
    *,
    start: str,
    end: str,
    roundtrip_cost_bps: float = 20,
    score_column: str = "score_etf_momentum",
    excluded_etfs: set[str] | None = None,
    rules: CoordinatedR4Rules = CoordinatedR4Rules(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if variant not in COORDINATED_R4_VARIANTS:
        raise ValueError(f"unknown coordinated R4 variant: {variant}")
    _validate_rules(rules)
    frame = panel.sort_values(["ts_code", "trade_date"]).copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    if "volatility_20d" not in frame:
        frame["volatility_20d"] = frame.groupby(
            "ts_code", sort=False
        )["etf_return_1d"].transform(
            lambda values: values.rolling(20, min_periods=18).std(ddof=0)
        )
    full_calendar = pd.Index(sorted(frame["trade_date"].unique()))
    active_calendar = pd.Index([
        date for date in full_calendar
        if pd.Timestamp(start) <= date <= pd.Timestamp(end)
    ])
    positions = {date: index for index, date in enumerate(full_calendar)}
    next_date = {
        date: full_calendar[index + 1]
        for date, index in positions.items()
        if index + 1 < len(full_calendar)
    }
    executions: dict[pd.Timestamp, tuple[int, pd.Timestamp]] = {}
    for sleeve in range(5):
        for signal in active_calendar[sleeve::5]:
            if signal in next_date and next_date[signal] <= pd.Timestamp(end):
                executions[pd.Timestamp(next_date[signal])] = (sleeve, pd.Timestamp(signal))

    prices = frame.pivot(index="trade_date", columns="ts_code", values="adj_open").sort_index()
    returns = frame.pivot(index="trade_date", columns="ts_code", values="etf_return_1d").sort_index()
    states = {sleeve: {CASH: 1.0} for sleeve in range(5)}
    navs = dict.fromkeys(range(5), 1.0)
    rows: list[dict] = []
    attribution_rows: list[dict] = []
    audit_rows: list[dict] = []

    for date in active_calendar:
        if date not in next_date or next_date[date] > pd.Timestamp(end):
            continue
        turnover_by_sleeve = dict.fromkeys(range(5), 0.0)
        signal_by_sleeve: dict[int, pd.Timestamp | None] = dict.fromkeys(range(5), None)
        execution = executions.get(pd.Timestamp(date))
        if execution is not None:
            sleeve, signal_date = execution
            signal_by_sleeve[sleeve] = signal_date
            previous = states[sleeve]
            previous_holdings = {
                code for code, weight in previous.items() if code != CASH and weight > 0
            }
            signal_day = frame.loc[frame["trade_date"].eq(signal_date)]
            target, constraint_audit = coordinated_target_weights(
                signal_day,
                states=states,
                navs=navs,
                rebalancing_sleeve=sleeve,
                variant=variant,
                previous_holdings=previous_holdings,
                return_history=returns.loc[returns.index <= signal_date],
                excluded_etfs=excluded_etfs,
                score_column=score_column,
                rules=rules,
            )
            turnover_by_sleeve[sleeve] = _turnover(previous, target)
            states[sleeve] = target
            audit_rows.append({
                "execution_date": pd.Timestamp(date),
                "signal_date": signal_date,
                "sleeve": sleeve,
                "variant": variant,
                **constraint_audit,
            })

        following = pd.Timestamp(next_date[date])
        daily_returns = (prices.loc[following] / prices.loc[date] - 1).to_dict()
        for sleeve in range(5):
            target = states[sleeve]
            missing = [
                code for code, weight in target.items()
                if code != CASH and weight > 0
                and not np.isfinite(daily_returns.get(code, np.nan))
            ]
            if missing:
                raise ValueError(f"missing open return on {date} for sleeve {sleeve}: {missing}")
            contributions = {
                code: weight * daily_returns.get(code, 0.0)
                for code, weight in target.items()
                if code != CASH and weight > 0
            }
            gross = float(sum(contributions.values()))
            turnover = float(turnover_by_sleeve[sleeve])
            cost = turnover * roundtrip_cost_bps / 10_000
            net = gross - cost
            nav_before = float(navs[sleeve])
            navs[sleeve] *= 1 + net
            for code, contribution in contributions.items():
                attribution_rows.append({
                    "return_date": following,
                    "variant": variant,
                    "sleeve": sleeve,
                    "ts_code": code,
                    "capital_contribution": nav_before * contribution / 5,
                })
            rows.append({
                "return_date": following,
                "holding_date": pd.Timestamp(date),
                "signal_date": signal_by_sleeve[sleeve],
                "portfolio": variant,
                "variant": variant,
                "sleeve": sleeve,
                "universe": "all",
                "excluded_etfs": ",".join(sorted(excluded_etfs or set())),
                "gross_return": gross,
                "net_return": net,
                "net_nav": navs[sleeve],
                "nav_before": nav_before,
                "turnover": turnover,
                "cost_drag": cost,
                "cash_weight": target.get(CASH, 0.0),
                "is_rebalance": signal_by_sleeve[sleeve] is not None,
                "target_weights": _serialize_weights(target),
            })
            denominator = 1 + gross
            states[sleeve] = {
                code: weight * (1 + daily_returns.get(code, 0.0)) / denominator
                for code, weight in target.items()
            }

    sleeve_daily = pd.DataFrame(rows)
    aggregate = _aggregate_sleeve_nav(sleeve_daily, variant)
    attribution = _attribution_table(pd.DataFrame(attribution_rows), frame, variant)
    audit = pd.DataFrame(audit_rows)
    return aggregate, sleeve_daily, attribution, audit


def coordinated_target_weights(
    signal_day: pd.DataFrame,
    *,
    states: dict[int, dict[str, float]],
    navs: dict[int, float],
    rebalancing_sleeve: int,
    variant: str,
    previous_holdings: set[str],
    return_history: pd.DataFrame,
    excluded_etfs: set[str] | None = None,
    score_column: str = "score_etf_momentum",
    rules: CoordinatedR4Rules = CoordinatedR4Rules(),
) -> tuple[dict[str, float], dict]:
    if variant == "R4_A_base":
        target = staggered_target_weights(
            signal_day,
            "R4_rank_buffer",
            previous_holdings=previous_holdings,
            excluded_etfs=excluded_etfs,
            score_column=score_column,
        )
        return target, _constraint_audit(
            target, states, navs, rebalancing_sleeve, signal_day, np.nan,
        )

    other_states = {
        sleeve: weights for sleeve, weights in states.items()
        if sleeve != rebalancing_sleeve
    }
    frequency = _holding_frequency(other_states)
    constraints = set(excluded_etfs or set())
    constraints.update({
        code for code, count in frequency.items()
        if count >= rules.maximum_sleeves_per_etf
    })
    correlation = _trailing_correlation(
        return_history,
        window=rules.correlation_window,
        minimum_observations=rules.correlation_minimum_observations,
    )
    maximum_entry_correlation = np.nan
    if variant == "R4_C_correlation":
        held_codes = {
            code for weights in other_states.values()
            for code, weight in weights.items()
            if code != CASH and weight > 0
        }
        candidates = set(signal_day["ts_code"].astype(str)) - constraints
        for candidate in candidates:
            peers = held_codes - {candidate}
            correlations = [
                correlation.at[candidate, peer]
                for peer in peers
                if candidate in correlation.index and peer in correlation.columns
                and np.isfinite(correlation.at[candidate, peer])
            ]
            if correlations and max(correlations) > rules.maximum_pairwise_correlation:
                constraints.add(candidate)

    target = staggered_target_weights(
        signal_day,
        "R4_rank_buffer",
        previous_holdings=previous_holdings,
        excluded_etfs=constraints,
        score_column=score_column,
    )
    if variant == "R4_C_correlation":
        target, maximum_entry_correlation = _apply_within_target_correlation(
            target,
            signal_day,
            correlation,
            maximum_correlation=rules.maximum_pairwise_correlation,
            score_column=score_column,
        )
    target = _apply_aggregate_caps(
        target,
        states=states,
        navs=navs,
        rebalancing_sleeve=rebalancing_sleeve,
        signal_day=signal_day,
        maximum_etf_weight=rules.maximum_aggregate_etf_weight,
        maximum_cluster_weight=(
            rules.maximum_aggregate_cluster_weight
            if variant == "R4_C_correlation" else None
        ),
    )
    if variant == "R4_C_correlation":
        maximum_entry_correlation = _accepted_pairwise_correlation(
            target, other_states, correlation,
        )
    audit = _constraint_audit(
        target,
        states,
        navs,
        rebalancing_sleeve,
        signal_day,
        maximum_entry_correlation,
    )
    audit["excluded_by_constraints"] = len(constraints - set(excluded_etfs or set()))
    return target, audit


def _apply_aggregate_caps(
    target: dict[str, float],
    *,
    states: dict[int, dict[str, float]],
    navs: dict[int, float],
    rebalancing_sleeve: int,
    signal_day: pd.DataFrame,
    maximum_etf_weight: float,
    maximum_cluster_weight: float | None,
) -> dict[str, float]:
    total_nav = float(sum(navs.values()))
    sleeve_share = float(navs[rebalancing_sleeve]) / total_nav
    metadata = signal_day.drop_duplicates("ts_code").set_index("ts_code")
    other_etf, other_cluster = _aggregate_exposures(
        states, navs, exclude_sleeve=rebalancing_sleeve, metadata=metadata,
    )
    clipped: dict[str, float] = {}
    for code, weight in sorted(target.items(), key=lambda item: item[1], reverse=True):
        if code == CASH or weight <= 0:
            continue
        etf_room = max(maximum_etf_weight - other_etf.get(code, 0.0), 0.0) / sleeve_share
        allowed = min(float(weight), etf_room)
        if maximum_cluster_weight is not None and code in metadata.index:
            cluster = str(metadata.at[code, "cluster"])
            already = other_cluster.get(cluster, 0.0) + sum(
                sleeve_share * existing_weight
                for existing_code, existing_weight in clipped.items()
                if existing_code in metadata.index
                and str(metadata.at[existing_code, "cluster"]) == cluster
            )
            cluster_room = max(maximum_cluster_weight - already, 0.0) / sleeve_share
            allowed = min(allowed, cluster_room)
        if allowed > 1e-12:
            clipped[code] = allowed
    clipped[CASH] = max(1 - sum(clipped.values()), 0.0)
    return clipped


def _apply_within_target_correlation(
    target: dict[str, float],
    signal_day: pd.DataFrame,
    correlation: pd.DataFrame,
    *,
    maximum_correlation: float,
    score_column: str,
) -> tuple[dict[str, float], float]:
    selected = [code for code, weight in target.items() if code != CASH and weight > 0]
    score = signal_day.set_index("ts_code")[score_column].to_dict()
    ordered = sorted(selected, key=lambda code: score.get(code, -np.inf), reverse=True)
    retained: list[str] = []
    observed: list[float] = []
    for code in ordered:
        pairwise = [
            correlation.at[code, peer]
            for peer in retained
            if code in correlation.index and peer in correlation.columns
            and np.isfinite(correlation.at[code, peer])
        ]
        if pairwise and max(pairwise) > maximum_correlation:
            continue
        observed.extend(pairwise)
        retained.append(code)
    result = {code: float(target[code]) for code in retained}
    result[CASH] = max(1 - sum(result.values()), 0.0)
    return result, max(observed) if observed else np.nan


def _accepted_pairwise_correlation(
    target: dict[str, float],
    other_states: dict[int, dict[str, float]],
    correlation: pd.DataFrame,
) -> float:
    target_codes = [
        code for code, weight in target.items() if code != CASH and weight > 0
    ]
    other_codes = {
        code for weights in other_states.values()
        for code, weight in weights.items()
        if code != CASH and weight > 0
    }
    pairs = {
        tuple(sorted((left, right)))
        for left in target_codes
        for right in [*target_codes, *other_codes]
        if left != right
    }
    observed = [
        float(correlation.at[left, right])
        for left, right in pairs
        if left in correlation.index and right in correlation.columns
        and np.isfinite(correlation.at[left, right])
    ]
    return max(observed) if observed else np.nan


def _constraint_audit(
    target: dict[str, float],
    states: dict[int, dict[str, float]],
    navs: dict[int, float],
    rebalancing_sleeve: int,
    signal_day: pd.DataFrame,
    maximum_entry_correlation: float,
) -> dict:
    post = {sleeve: weights.copy() for sleeve, weights in states.items()}
    post[rebalancing_sleeve] = target
    metadata = signal_day.drop_duplicates("ts_code").set_index("ts_code")
    etf, cluster = _aggregate_exposures(post, navs, metadata=metadata)
    frequency = _holding_frequency(post)
    rebalanced_codes = {
        code for code, weight in target.items() if code != CASH and weight > 0
    }
    return {
        "maximum_aggregate_etf_weight": max(etf.values(), default=0.0),
        "maximum_rebalanced_etf_weight": max(
            (etf.get(code, 0.0) for code in rebalanced_codes), default=0.0,
        ),
        "maximum_aggregate_cluster_weight": max(cluster.values(), default=0.0),
        "maximum_sleeve_frequency": max(frequency.values(), default=0),
        "maximum_entry_pairwise_correlation": maximum_entry_correlation,
        "target_cash_weight": float(target.get(CASH, 0.0)),
        "excluded_by_constraints": 0,
    }


def _aggregate_exposures(
    states: dict[int, dict[str, float]],
    navs: dict[int, float],
    *,
    metadata: pd.DataFrame,
    exclude_sleeve: int | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    total_nav = float(sum(navs.values()))
    etf: dict[str, float] = {}
    cluster: dict[str, float] = {}
    for sleeve, weights in states.items():
        if sleeve == exclude_sleeve:
            continue
        capital_share = float(navs[sleeve]) / total_nav
        for code, weight in weights.items():
            if code == CASH or weight <= 0:
                continue
            exposure = capital_share * float(weight)
            etf[code] = etf.get(code, 0.0) + exposure
            if code in metadata.index:
                label = str(metadata.at[code, "cluster"])
                cluster[label] = cluster.get(label, 0.0) + exposure
    return etf, cluster


def _holding_frequency(states: dict[int, dict[str, float]]) -> dict[str, int]:
    frequency: dict[str, int] = {}
    for weights in states.values():
        for code, weight in weights.items():
            if code != CASH and weight > 0:
                frequency[code] = frequency.get(code, 0) + 1
    return frequency


def _trailing_correlation(
    returns: pd.DataFrame,
    *,
    window: int,
    minimum_observations: int,
) -> pd.DataFrame:
    sample = returns.tail(window)
    return sample.corr(min_periods=minimum_observations)


def _aggregate_sleeve_nav(sleeves: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows = []
    previous_total_nav = 1.0
    for date, day in sleeves.groupby("return_date", sort=True):
        total_nav = float(day["net_nav"].sum() / 5)
        pre_capital = day["nav_before"] / day["nav_before"].sum()
        rows.append({
            "return_date": date,
            "portfolio": variant,
            "net_nav": total_nav,
            "net_return": total_nav / previous_total_nav - 1,
            "gross_return": float((pre_capital * day["gross_return"]).sum()),
            "turnover": float((pre_capital * day["turnover"]).sum()),
            "cost_drag": float((pre_capital * day["cost_drag"]).sum()),
            "cash_weight": float((pre_capital * day["cash_weight"]).sum()),
            "is_rebalance": bool(day["is_rebalance"].any()),
        })
        previous_total_nav = total_nav
    return pd.DataFrame(rows)


def _attribution_table(
    rows: pd.DataFrame, frame: pd.DataFrame, variant: str,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=[
            "variant", "ts_code", "capital_contribution", "contribution_days",
            "positive_profit_share", "etf_name", "concept_name",
        ])
    result = rows.groupby(["variant", "ts_code"], as_index=False).agg(
        capital_contribution=("capital_contribution", "sum"),
        contribution_days=("capital_contribution", "size"),
    )
    positive_total = result["capital_contribution"].clip(lower=0).sum()
    result["positive_profit_share"] = (
        result["capital_contribution"].clip(lower=0) / positive_total
        if positive_total > 0 else 0.0
    )
    names = frame[["ts_code", "etf_name", "concept_name"]].drop_duplicates("ts_code")
    return result.merge(names, on="ts_code", how="left").sort_values(
        "positive_profit_share", ascending=False,
    )


def _turnover(previous: dict[str, float], target: dict[str, float]) -> float:
    codes = set(previous) | set(target)
    return 0.5 * sum(abs(target.get(code, 0.0) - previous.get(code, 0.0)) for code in codes)


def _serialize_weights(weights: dict[str, float]) -> str:
    return ";".join(f"{code}:{weight:.12g}" for code, weight in sorted(weights.items()))


def _validate_rules(rules: CoordinatedR4Rules) -> None:
    if not 1 <= rules.maximum_sleeves_per_etf <= 5:
        raise ValueError("maximum_sleeves_per_etf must be in [1, 5]")
    for name, value in (
        ("maximum_aggregate_etf_weight", rules.maximum_aggregate_etf_weight),
        ("maximum_aggregate_cluster_weight", rules.maximum_aggregate_cluster_weight),
        ("maximum_pairwise_correlation", rules.maximum_pairwise_correlation),
    ):
        if not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1]")


def deserialize_weights(value: str) -> dict[str, float]:
    return parse_weights(value)
