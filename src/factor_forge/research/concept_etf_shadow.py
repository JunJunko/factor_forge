from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from factor_forge.research.concept_rotation_alpha import block_bootstrap_mean, newey_west_mean


CASH = "__CASH__"
PORTFOLIOS = ("P0_equal_weight", "P1_etf_momentum", "P2_breadth_overlay", "P3_rrg_filter")
STAGGERED_VARIANTS = (
    "S0_equal_weight", "R1_staggered_momentum", "R2_absolute_momentum",
    "R3_inverse_volatility", "R4_rank_buffer",
)


def target_weights(
    day: pd.DataFrame,
    portfolio: str,
    *,
    top_n: int = 3,
    universe: str = "all",
    excluded_etf: str | None = None,
) -> dict[str, float]:
    candidates = _eligible_universe(day, universe=universe, excluded_etf=excluded_etf)
    if portfolio == "P0_equal_weight":
        if candidates.empty:
            return {CASH: 1.0}
        weight = 1 / len(candidates)
        return {**dict.fromkeys(candidates["ts_code"].astype(str), weight), CASH: 0.0}
    if portfolio not in PORTFOLIOS:
        raise ValueError(f"unknown portfolio: {portfolio}")
    candidates = candidates.dropna(subset=["score_etf_momentum"])
    if portfolio == "P3_rrg_filter":
        candidates = candidates.loc[candidates["rrg_quadrant"].isin(["leading", "improving"])]
    chosen = _cluster_rank(candidates.sort_values("score_etf_momentum", ascending=False), top_n)
    weights = {code: 1 / top_n for code in chosen["ts_code"].astype(str)}
    if portfolio == "P2_breadth_overlay":
        weakening = (
            chosen["common_breadth_delta_smooth5"].lt(0)
            & chosen["rs_momentum_5d"].lt(0)
        )
        for code in chosen.loc[weakening, "ts_code"].astype(str):
            weights[code] *= 0.5
    weights[CASH] = max(1 - sum(weights.values()), 0.0)
    return weights


def simulate_portfolio(
    panel: pd.DataFrame,
    portfolio: str,
    *,
    start: str,
    end: str,
    horizon: int = 5,
    offset: int = 0,
    top_n: int = 3,
    roundtrip_cost_bps: float = 20,
    universe: str = "all",
    excluded_etf: str | None = None,
) -> pd.DataFrame:
    label = f"forward_open_{horizon}d"
    if label not in panel:
        raise KeyError(label)
    period = panel.loc[panel["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    dates = sorted(period.loc[period[label].notna(), "trade_date"].unique())[offset::horizon]
    pretrade = {CASH: 1.0}
    rows = []
    for date in dates:
        day = period.loc[period["trade_date"].eq(date)]
        target = target_weights(
            day, portfolio, top_n=top_n, universe=universe, excluded_etf=excluded_etf,
        )
        turnover = _turnover(pretrade, target)
        returns = day.set_index("ts_code")[label].to_dict()
        missing = [code for code in target if code != CASH and not np.isfinite(returns.get(code, np.nan))]
        if missing:
            raise ValueError(f"missing {horizon}d forward return on {date}: {missing}")
        gross = sum(weight * returns.get(code, 0.0) for code, weight in target.items() if code != CASH)
        cost = turnover * roundtrip_cost_bps / 10000
        net = gross - cost
        rows.append({
            "trade_date": date, "portfolio": portfolio, "offset": offset, "horizon": horizon,
            "top_n": top_n, "universe": universe, "excluded_etf": excluded_etf,
            "gross_return": gross, "net_return": net, "turnover": turnover,
            "cost_drag": cost, "cash_weight": target.get(CASH, 0.0),
            "holdings": sum(code != CASH and weight > 0 for code, weight in target.items()),
            "target_weights": _serialize_weights(target),
        })
        denominator = 1 + gross
        pretrade = {
            code: weight * (1 + returns.get(code, 0.0)) / denominator
            for code, weight in target.items()
        }
    return pd.DataFrame(rows)


def evaluate_specification(
    panel: pd.DataFrame,
    portfolio: str,
    *,
    start: str,
    end: str,
    horizon: int,
    top_n: int,
    roundtrip_cost_bps: float,
    universe: str = "all",
    excluded_etf: str | None = None,
    benchmark: str = "P0_equal_weight",
) -> tuple[dict, pd.DataFrame]:
    period_parts = []
    offset_rows = []
    for offset in range(horizon):
        strategy = simulate_portfolio(
            panel, portfolio, start=start, end=end, horizon=horizon, offset=offset,
            top_n=top_n, roundtrip_cost_bps=roundtrip_cost_bps, universe=universe,
            excluded_etf=excluded_etf,
        )
        baseline = simulate_portfolio(
            panel, benchmark, start=start, end=end, horizon=horizon, offset=offset,
            top_n=top_n, roundtrip_cost_bps=roundtrip_cost_bps, universe=universe,
            excluded_etf=excluded_etf,
        )
        joined = strategy.merge(
            baseline[["trade_date", "net_return"]].rename(columns={"net_return": "benchmark_net_return"}),
            on="trade_date", how="inner", validate="one_to_one",
        )
        joined["net_excess"] = joined["net_return"] - joined["benchmark_net_return"]
        period_parts.append(joined)
        offset_rows.append({
            "offset": offset, "periods": len(joined), "mean_net_excess": joined["net_excess"].mean(),
            "total_net_return": float(np.prod(1 + joined["net_return"]) - 1),
            "mean_turnover": joined["turnover"].mean(),
        })
    periods = pd.concat(period_parts, ignore_index=True).sort_values("trade_date")
    offset_summary = pd.DataFrame(offset_rows)
    stats = newey_west_mean(periods["net_excess"], horizon - 1)
    summary = {
        "portfolio": portfolio, "benchmark": benchmark, "horizon": horizon, "top_n": top_n,
        "roundtrip_cost_bps": roundtrip_cost_bps, "universe": universe,
        "excluded_etf": excluded_etf, "periods": len(periods),
        "mean_net_return": periods["net_return"].mean(), "mean_net_excess": stats["mean"],
        "net_excess_nw_t": stats["t_value"], "positive_period_rate": periods["net_excess"].gt(0).mean(),
        "positive_offsets": int(offset_summary["mean_net_excess"].gt(0).sum()),
        "offsets": len(offset_summary), "worst_offset_excess": offset_summary["mean_net_excess"].min(),
        "mean_turnover": periods["turnover"].mean(), "mean_cash_weight": periods["cash_weight"].mean(),
    }
    return summary, periods


def latest_target_table(
    panel: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp | None = None,
    top_n: int = 3,
    universe: str = "all",
    portfolios: Iterable[str] = PORTFOLIOS,
) -> pd.DataFrame:
    date = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(panel["trade_date"].max())
    day = panel.loc[panel["trade_date"].eq(date)]
    if day.empty:
        raise ValueError(f"no signal data for {date.date()}")
    metadata = day.set_index("ts_code")
    rows = []
    for portfolio in portfolios:
        weights = target_weights(day, portfolio, top_n=top_n, universe=universe)
        for code, weight in weights.items():
            if weight <= 0:
                continue
            row = {
                "signal_date": date, "portfolio": portfolio, "ts_code": code,
                "target_weight": weight,
            }
            if code == CASH:
                row.update({"etf_name": "现金", "concept_name": "现金", "cluster": "cash"})
            else:
                item = metadata.loc[code]
                row.update({
                    "etf_name": item["etf_name"], "concept_name": item["concept_name"],
                    "cluster": item["cluster"], "score_etf_momentum": item["score_etf_momentum"],
                    "rrg_quadrant": item["rrg_quadrant"],
                    "breadth_delta": item["common_breadth_delta_smooth5"],
                    "rs_momentum_5d": item["rs_momentum_5d"],
                })
            rows.append(row)
    return pd.DataFrame(rows)


def simulate_weekly_daily_nav(
    panel: pd.DataFrame,
    portfolio: str,
    *,
    start: str,
    end: str,
    top_n: int = 3,
    roundtrip_cost_bps: float = 20,
    universe: str = "all",
) -> pd.DataFrame:
    frame = panel.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    calendar = pd.Index(sorted(frame["trade_date"].unique()))
    execution_map = _weekly_execution_map(calendar, pd.Timestamp(start), pd.Timestamp(end))
    if not execution_map:
        return pd.DataFrame()
    next_date = {calendar[index]: calendar[index + 1] for index in range(len(calendar) - 1)}
    prices = frame.pivot(index="trade_date", columns="ts_code", values="adj_open").sort_index()
    pretrade = {CASH: 1.0}
    net_nav = 1.0
    rows = []
    first_execution = min(execution_map)
    trading_dates = [
        date for date in calendar
        if first_execution <= date <= pd.Timestamp(end) and date in next_date
        and next_date[date] <= pd.Timestamp(end)
    ]
    for date in trading_dates:
        signal_date = execution_map.get(date)
        target = pretrade
        turnover = 0.0
        if signal_date is not None:
            signal_day = frame.loc[frame["trade_date"].eq(signal_date)]
            target = target_weights(signal_day, portfolio, top_n=top_n, universe=universe)
            turnover = _turnover(pretrade, target)
        following = next_date[date]
        returns = (prices.loc[following] / prices.loc[date] - 1).to_dict()
        missing = [code for code, weight in target.items() if code != CASH and weight > 0 and not np.isfinite(returns.get(code, np.nan))]
        if missing:
            raise ValueError(f"missing open-to-open return on {date}: {missing}")
        gross = sum(weight * returns.get(code, 0.0) for code, weight in target.items() if code != CASH)
        cost = turnover * roundtrip_cost_bps / 10000
        net = gross - cost
        net_nav *= 1 + net
        rows.append({
            "return_date": following, "holding_date": date, "signal_date": signal_date,
            "portfolio": portfolio, "gross_return": gross, "net_return": net, "net_nav": net_nav,
            "turnover": turnover, "cost_drag": cost, "cash_weight": target.get(CASH, 0.0),
            "is_rebalance": signal_date is not None, "target_weights": _serialize_weights(target),
        })
        denominator = 1 + gross
        pretrade = {
            code: weight * (1 + returns.get(code, 0.0)) / denominator
            for code, weight in target.items()
        }
    return pd.DataFrame(rows)


def monthly_performance(
    daily: pd.DataFrame, *, benchmark_portfolio: str = "P0_equal_weight",
) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    rows = []
    for portfolio, portfolio_frame in daily.groupby("portfolio", observed=True):
        portfolio_frame = portfolio_frame.sort_values("return_date").copy()
        portfolio_frame["month"] = portfolio_frame["return_date"].dt.to_period("M").astype(str)
        portfolio_frame["running_peak"] = portfolio_frame["net_nav"].cummax().clip(lower=1.0)
        portfolio_frame["drawdown"] = portfolio_frame["net_nav"] / portfolio_frame["running_peak"] - 1
        previous_nav = 1.0
        for month, group in portfolio_frame.groupby("month", sort=True):
            nav_path = pd.concat([pd.Series([previous_nav]), group["net_nav"].reset_index(drop=True)], ignore_index=True)
            within_drawdown = nav_path / nav_path.cummax() - 1
            rows.append({
                "month": month, "portfolio": portfolio,
                "monthly_return": float(group["net_nav"].iloc[-1] / previous_nav - 1),
                "monthly_max_drawdown": float(within_drawdown.min()),
                "month_end_drawdown": float(group["drawdown"].iloc[-1]),
                "month_end_nav": float(group["net_nav"].iloc[-1]),
                "turnover": float(group["turnover"].sum()),
                "rebalances": int(group["is_rebalance"].sum()),
                "trading_days": len(group),
            })
            previous_nav = float(group["net_nav"].iloc[-1])
    result = pd.DataFrame(rows)
    benchmark = result.loc[result["portfolio"].eq(benchmark_portfolio), ["month", "monthly_return"]].rename(
        columns={"monthly_return": "benchmark_monthly_return"}
    )
    result = result.merge(benchmark, on="month", how="left", validate="many_to_one")
    result["monthly_excess_vs_p0"] = result["monthly_return"] - result["benchmark_monthly_return"]
    return result.sort_values(["month", "portfolio"]).reset_index(drop=True)


def staggered_target_weights(
    day: pd.DataFrame,
    variant: str,
    *,
    previous_holdings: set[str] | None = None,
    universe: str = "all",
    excluded_etfs: set[str] | None = None,
    score_column: str = "score_etf_momentum",
    r4_selection_count: int = 4,
    r4_retention_rank: int = 6,
    r4_maximum_etf_weight: float = 0.30,
    r4_absolute_momentum_column: str = "etf_momentum_60d",
) -> dict[str, float]:
    candidates = _eligible_universe(day, universe=universe, excluded_etf=None)
    if excluded_etfs:
        candidates = candidates.loc[~candidates["ts_code"].isin(excluded_etfs)]
    if score_column not in candidates:
        raise KeyError(score_column)
    candidates = candidates.dropna(subset=[score_column]).sort_values(
        score_column, ascending=False
    ).copy()
    if variant == "S0_equal_weight":
        if candidates.empty:
            return {CASH: 1.0}
        return {**dict.fromkeys(candidates["ts_code"].astype(str), 1 / len(candidates)), CASH: 0.0}
    if variant == "R1_staggered_momentum":
        selected = _cluster_rank(candidates, 3)
        weights = dict.fromkeys(selected["ts_code"].astype(str), 1 / 3)
        return {**weights, CASH: max(1 - sum(weights.values()), 0.0)}
    if variant == "R2_absolute_momentum":
        selected = _cluster_rank(candidates, 3)
        selected = selected.loc[selected["etf_momentum_60d"].gt(0)]
        weights = dict.fromkeys(selected["ts_code"].astype(str), 1 / 3)
        return {**weights, CASH: max(1 - sum(weights.values()), 0.0)}
    if variant == "R3_inverse_volatility":
        selected = _cluster_rank(candidates, 4)
        selected = selected.loc[selected["etf_momentum_60d"].gt(0)]
        return _inverse_volatility_weights(selected, cap=0.30)
    if variant == "R4_rank_buffer":
        if r4_selection_count < 1:
            raise ValueError("r4_selection_count must be positive")
        if r4_retention_rank < r4_selection_count:
            raise ValueError("r4_retention_rank must be at least r4_selection_count")
        if not 0 < r4_maximum_etf_weight <= 1:
            raise ValueError("r4_maximum_etf_weight must be in (0, 1]")
        if r4_absolute_momentum_column not in candidates.columns:
            raise KeyError(r4_absolute_momentum_column)
        selected = _rank_buffer_selection(
            candidates,
            previous_holdings or set(),
            selection_count=r4_selection_count,
            retention_rank=r4_retention_rank,
        )
        selected = selected.loc[selected[r4_absolute_momentum_column].gt(0)]
        return _inverse_volatility_weights(selected, cap=r4_maximum_etf_weight)
    raise ValueError(f"unknown staggered variant: {variant}")


def simulate_staggered_sleeves(
    panel: pd.DataFrame,
    variant: str,
    *,
    start: str,
    end: str,
    roundtrip_cost_bps: float = 20,
    universe: str = "all",
    excluded_etfs: set[str] | None = None,
    score_column: str = "score_etf_momentum",
    holding_days: int = 5,
    execution_delay_days: int = 1,
    r4_selection_count: int = 4,
    r4_retention_rank: int = 6,
    r4_maximum_etf_weight: float = 0.30,
    r4_absolute_momentum_column: str = "etf_momentum_60d",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if variant not in STAGGERED_VARIANTS:
        raise ValueError(f"unknown staggered variant: {variant}")
    if holding_days < 1:
        raise ValueError("holding_days must be positive")
    if execution_delay_days < 1:
        raise ValueError("execution_delay_days must be positive")
    frame = panel.sort_values(["ts_code", "trade_date"]).copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    if "volatility_20d" not in frame:
        frame["volatility_20d"] = frame.groupby("ts_code", sort=False)["etf_return_1d"].transform(
            lambda values: values.rolling(20, min_periods=18).std(ddof=0)
        )
    full_calendar = pd.Index(sorted(frame["trade_date"].unique()))
    active_calendar = pd.Index([
        date for date in full_calendar if pd.Timestamp(start) <= date <= pd.Timestamp(end)
    ])
    positions = {date: index for index, date in enumerate(full_calendar)}
    next_date = {
        date: full_calendar[index + 1] for date, index in positions.items()
        if index + 1 < len(full_calendar)
    }
    delayed_execution = {
        date: full_calendar[index + execution_delay_days]
        for date, index in positions.items()
        if index + execution_delay_days < len(full_calendar)
    }
    prices = frame.pivot(index="trade_date", columns="ts_code", values="adj_open").sort_index()
    sleeve_parts, attribution_rows = [], []
    for sleeve in range(holding_days):
        signals = active_calendar[sleeve::holding_days]
        executions = {
            pd.Timestamp(delayed_execution[signal]): pd.Timestamp(signal)
            for signal in signals
            if signal in delayed_execution
            and delayed_execution[signal] <= pd.Timestamp(end)
        }
        pretrade = {CASH: 1.0}
        sleeve_nav = 1.0
        rows = []
        for date in active_calendar:
            if date not in next_date or next_date[date] > pd.Timestamp(end):
                continue
            signal_date = executions.get(pd.Timestamp(date))
            target = pretrade
            turnover = 0.0
            if signal_date is not None:
                previous_holdings = {code for code, weight in pretrade.items() if code != CASH and weight > 0}
                signal_day = frame.loc[frame["trade_date"].eq(signal_date)]
                target = staggered_target_weights(
                    signal_day, variant, previous_holdings=previous_holdings,
                    universe=universe, excluded_etfs=excluded_etfs,
                    score_column=score_column,
                    r4_selection_count=r4_selection_count,
                    r4_retention_rank=r4_retention_rank,
                    r4_maximum_etf_weight=r4_maximum_etf_weight,
                    r4_absolute_momentum_column=r4_absolute_momentum_column,
                )
                turnover = _turnover(pretrade, target)
            following = pd.Timestamp(next_date[date])
            returns = (prices.loc[following] / prices.loc[date] - 1).to_dict()
            gross_contributions = {
                code: weight * returns.get(code, 0.0)
                for code, weight in target.items() if code != CASH and weight > 0
            }
            gross = sum(gross_contributions.values())
            cost = turnover * roundtrip_cost_bps / 10000
            net = gross - cost
            nav_before = sleeve_nav
            sleeve_nav *= 1 + net
            for code, contribution in gross_contributions.items():
                attribution_rows.append({
                    "return_date": following, "variant": variant, "sleeve": sleeve,
                    "ts_code": code,
                    "capital_contribution": nav_before * contribution / holding_days,
                })
            rows.append({
                "return_date": following, "holding_date": date, "signal_date": signal_date,
                "portfolio": variant, "variant": variant, "sleeve": sleeve,
                "universe": universe,
                "excluded_etfs": ",".join(sorted(excluded_etfs or set())),
                "gross_return": gross, "net_return": net, "net_nav": sleeve_nav,
                "nav_before": nav_before, "turnover": turnover, "cost_drag": cost,
                "cash_weight": target.get(CASH, 0.0), "is_rebalance": signal_date is not None,
                "target_weights": _serialize_weights(target),
            })
            denominator = 1 + gross
            pretrade = {
                code: weight * (1 + returns.get(code, 0.0)) / denominator
                for code, weight in target.items()
            }
        sleeve_parts.append(pd.DataFrame(rows))
    sleeve_daily = pd.concat(sleeve_parts, ignore_index=True)
    aggregate = _aggregate_sleeve_nav(sleeve_daily, variant)
    attribution = pd.DataFrame(attribution_rows).groupby(["variant", "ts_code"], as_index=False).agg(
        capital_contribution=("capital_contribution", "sum"),
        contribution_days=("capital_contribution", "size"),
    )
    positive_total = attribution["capital_contribution"].clip(lower=0).sum()
    attribution["positive_profit_share"] = (
        attribution["capital_contribution"].clip(lower=0) / positive_total if positive_total > 0 else 0.0
    )
    names = frame[["ts_code", "etf_name", "concept_name"]].drop_duplicates("ts_code")
    attribution = attribution.merge(names, on="ts_code", how="left").sort_values(
        "positive_profit_share", ascending=False
    )
    return aggregate, sleeve_daily, attribution


def _aggregate_sleeve_nav(sleeve_daily: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows = []
    previous_total_nav = 1.0
    for date, day in sleeve_daily.groupby("return_date", sort=True):
        total_nav = float(day["net_nav"].mean())
        pre_capital = day["nav_before"] / day["nav_before"].sum()
        rows.append({
            "return_date": date, "portfolio": variant, "net_nav": total_nav,
            "net_return": total_nav / previous_total_nav - 1,
            "gross_return": float((pre_capital * day["gross_return"]).sum()),
            "turnover": float((pre_capital * day["turnover"]).sum()),
            "cost_drag": float((pre_capital * day["cost_drag"]).sum()),
            "cash_weight": float((pre_capital * day["cash_weight"]).sum()),
            "is_rebalance": bool(day["is_rebalance"].any()),
        })
        previous_total_nav = total_nav
    return pd.DataFrame(rows)


def _rank_buffer_selection(
    candidates: pd.DataFrame,
    previous_holdings: set[str],
    *,
    selection_count: int = 4,
    retention_rank: int = 6,
) -> pd.DataFrame:
    ranked = candidates.copy()
    ranked["raw_rank"] = np.arange(1, len(ranked) + 1)
    retained = ranked.loc[
        ranked["ts_code"].isin(previous_holdings) & ranked["raw_rank"].le(retention_rank)
    ].sort_values("raw_rank")
    selected_indices, clusters = [], set()
    for index, row in retained.iterrows():
        if row["cluster"] in clusters:
            continue
        selected_indices.append(index)
        clusters.add(row["cluster"])
        if len(selected_indices) == selection_count:
            return ranked.loc[selected_indices]
    entrants = ranked.loc[ranked["raw_rank"].le(selection_count)]
    for index, row in entrants.iterrows():
        if index in selected_indices or row["cluster"] in clusters:
            continue
        selected_indices.append(index)
        clusters.add(row["cluster"])
        if len(selected_indices) == selection_count:
            break
    return ranked.loc[selected_indices]


def _inverse_volatility_weights(selected: pd.DataFrame, *, cap: float) -> dict[str, float]:
    selected = selected.loc[selected["volatility_20d"].gt(0)].copy()
    if selected.empty:
        return {CASH: 1.0}
    inverse = dict(zip(selected["ts_code"].astype(str), 1 / selected["volatility_20d"]))
    weights: dict[str, float] = {}
    active = dict(inverse)
    remaining = 1.0
    while active and remaining > 1e-12:
        total = sum(active.values())
        proposed = {code: remaining * value / total for code, value in active.items()}
        capped = [code for code, weight in proposed.items() if weight > cap]
        if not capped:
            weights.update(proposed)
            remaining = 0.0
            break
        for code in capped:
            weights[code] = cap
            remaining -= cap
            active.pop(code)
    weights[CASH] = max(1 - sum(weights.values()), 0.0)
    return weights


def nonoverlapping_holding_periods(sleeve_daily: pd.DataFrame, *, holding_days: int = 5) -> pd.DataFrame:
    rows = []
    group_columns = [
        column for column in ("scenario", "variant", "roundtrip_cost_bps", "sleeve")
        if column in sleeve_daily
    ]
    for keys, sleeve in sleeve_daily.groupby(group_columns, observed=True):
        sleeve = sleeve.sort_values("holding_date").copy()
        sleeve["period_id"] = sleeve["is_rebalance"].cumsum()
        for period_id, period in sleeve.loc[sleeve["period_id"].gt(0)].groupby("period_id"):
            if len(period) != holding_days or not bool(period.iloc[0]["is_rebalance"]):
                continue
            row = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
            row.update({
                "period_id": int(period_id), "signal_date": period.iloc[0]["signal_date"],
                "entry_date": period.iloc[0]["holding_date"], "exit_date": period.iloc[-1]["return_date"],
                "holding_days": len(period),
                "gross_return": float(np.prod(1 + period["gross_return"]) - 1),
                "net_return": float(np.prod(1 + period["net_return"]) - 1),
                "turnover": float(period["turnover"].sum()),
                "cost_drag": float(period["cost_drag"].sum()),
                "mean_cash_weight": float(period["cash_weight"].mean()),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def nonoverlap_sleeve_statistics(
    strategy_periods: pd.DataFrame,
    benchmark_periods: pd.DataFrame,
    *,
    bootstrap_samples: int = 2_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["sleeve", "signal_date", "entry_date", "exit_date"]
    benchmark = benchmark_periods[keys + ["net_return"]].rename(
        columns={"net_return": "benchmark_net_return"}
    )
    paired = strategy_periods.merge(benchmark, on=keys, how="inner", validate="one_to_one")
    paired["net_excess"] = paired["net_return"] - paired["benchmark_net_return"]
    rows = []
    for sleeve, group in paired.groupby("sleeve", observed=True):
        group = group.sort_values("entry_date")
        stats = newey_west_mean(group["net_excess"], 0)
        low, high = block_bootstrap_mean(
            group["net_excess"].to_numpy(float), block=4,
            samples=bootstrap_samples, seed=20260717 + int(sleeve),
        )
        nav = (1 + group["net_return"]).cumprod()
        drawdown = nav / nav.cummax().clip(lower=1.0) - 1
        rows.append({
            "sleeve": int(sleeve), "periods": len(group),
            "mean_net_return": float(group["net_return"].mean()),
            "mean_net_excess": stats["mean"], "net_excess_t": stats["t_value"],
            "bootstrap_95_low": low, "bootstrap_95_high": high,
            "positive_period_rate": float(group["net_excess"].gt(0).mean()),
            "total_net_return": float(nav.iloc[-1] - 1),
            "maximum_drawdown": float(drawdown.min()),
            "mean_turnover": float(group["turnover"].mean()),
        })
    return pd.DataFrame(rows), paired


def _weekly_execution_map(
    calendar: pd.Index, start: pd.Timestamp, end: pd.Timestamp,
) -> dict[pd.Timestamp, pd.Timestamp]:
    available = pd.Series(calendar[(calendar >= start) & (calendar <= end)])
    if available.empty:
        return {}
    weeks = available.dt.to_period("W-FRI")
    signals = available.groupby(weeks).max().tolist()
    # The final partial week is not a valid signal unless Friday is present.
    if signals and pd.Timestamp(signals[-1]).weekday() != 4:
        signals = signals[:-1]
    positions = {date: index for index, date in enumerate(calendar)}
    result = {}
    for signal in signals:
        position = positions[pd.Timestamp(signal)]
        if position + 1 >= len(calendar):
            continue
        execution = pd.Timestamp(calendar[position + 1])
        if execution <= end:
            result[execution] = pd.Timestamp(signal)
    return result


def _eligible_universe(day: pd.DataFrame, *, universe: str, excluded_etf: str | None) -> pd.DataFrame:
    mapping = day["mapping_pass"].astype("boolean").fillna(False).astype(bool)
    concept = day["eligible_concept"].astype("boolean").fillna(False).astype(bool)
    result = day.loc[mapping & concept].copy()
    if universe == "no_proxy":
        result = result.loc[result["match_type"].ne("proxy")]
    elif universe == "exact":
        result = result.loc[result["match_type"].eq("exact")]
    elif universe != "all":
        raise ValueError(f"unknown universe: {universe}")
    if excluded_etf:
        result = result.loc[result["ts_code"].ne(excluded_etf)]
    return result


def _cluster_rank(candidates: pd.DataFrame, top_n: int) -> pd.DataFrame:
    indices, clusters = [], set()
    for index, row in candidates.iterrows():
        if row["cluster"] in clusters:
            continue
        indices.append(index)
        clusters.add(row["cluster"])
        if len(indices) == top_n:
            break
    return candidates.loc[indices]


def _turnover(current: dict[str, float], target: dict[str, float]) -> float:
    keys = set(current) | set(target)
    return 0.5 * sum(abs(target.get(key, 0.0) - current.get(key, 0.0)) for key in keys)


def _serialize_weights(weights: dict[str, float]) -> str:
    return ";".join(f"{code}:{weight:.8f}" for code, weight in sorted(weights.items()) if weight > 0)
