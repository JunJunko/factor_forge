from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads


HORIZONS = (1, 3, 5, 10, 20)
SIGNALS = {
    "hot_1d": "signal_hot_1d",
    "momentum_20d": "signal_momentum_20d",
    "breadth_delta": "signal_breadth_delta",
    "common_breadth_delta": "signal_common_breadth_delta",
    "rrg_only": "signal_rrg_only",
    "current_membership_breadth_rrg": "signal_current_breadth_rrg",
    "common_membership_breadth_rrg": "signal_common_breadth_rrg",
    "membership_churn_placebo": "signal_churn",
}


def load_dc_snapshot(
    root: str | Path, *, trade_dates: Iterable[pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    # Read only the fields used below. Historical monthly files contain an
    # all-null ``level`` column in some partitions and a string column in
    # others, which Arrow cannot unify even though the field is irrelevant.
    index = pd.read_parquet(
        root / "index_monthly", columns=["trade_date", "ts_code", "name", "idx_type"]
    )
    index = index.loc[index["idx_type"].eq("概念板块")].drop_duplicates(
        ["trade_date", "ts_code"], keep="last"
    ).rename(columns={"ts_code": "concept_code", "name": "concept_name"})
    if trade_dates is not None:
        allowed = {pd.Timestamp(date).strftime("%Y%m%d") for date in trade_dates}
        index = index.loc[index["trade_date"].astype(str).isin(allowed)]
    active_dates = index["trade_date"].astype(str).unique().tolist()
    table = pads.dataset(root / "members_by_concept", format="parquet").to_table(
        columns=["trade_date", "ts_code", "con_code"],
        filter=pads.field("trade_date").isin(active_dates),
    )
    members = table.to_pandas(categories=["ts_code", "con_code"])
    index["trade_date"] = pd.to_datetime(index["trade_date"].astype(str))
    members["trade_date"] = pd.to_datetime(members["trade_date"].astype(str))
    members = members.rename(columns={
        "ts_code": "concept_code", "con_code": "ts_code",
    }).drop_duplicates(["trade_date", "concept_code", "ts_code"], keep="last")
    members, repaired_dates = repair_partial_member_snapshots(members)
    # dc_index has known partial single-day snapshots (for example 2026-04-09),
    # while per-concept dc_member still carries the complete active key set.
    # Reconstruct active concept-days from actual memberships and use dc_index as
    # descriptive metadata only.
    names = index.sort_values("trade_date").drop_duplicates("concept_code", keep="last")[[
        "concept_code", "concept_name",
    ]]
    index = members[["trade_date", "concept_code"]].drop_duplicates().merge(
        names, on="concept_code", how="left", validate="many_to_one",
    )
    index = index.sort_values(["trade_date", "concept_code"])
    members = members.sort_values(["trade_date", "concept_code", "ts_code"])
    index.attrs["repaired_member_dates"] = repaired_dates
    members.attrs["repaired_member_dates"] = repaired_dates
    return index, members


def repair_partial_member_snapshots(
    members: pd.DataFrame, *, neighbor_ratio: float = 0.95,
) -> tuple[pd.DataFrame, list[str]]:
    """Replace obvious partial source snapshots with the prior complete snapshot."""
    counts = members.groupby("trade_date", observed=True)["concept_code"].nunique().sort_index()
    neighbor_reference = pd.concat(
        [counts.shift(2), counts.shift(1), counts.shift(-1), counts.shift(-2)], axis=1
    ).median(axis=1)
    anomalous = counts.lt(neighbor_reference * neighbor_ratio)
    dates = counts.index[anomalous.fillna(False)].tolist()
    if not dates:
        return members, []
    good_dates = counts.index[~anomalous.fillna(False)]
    replacements = []
    for date in dates:
        previous = good_dates[good_dates < date]
        if previous.empty:
            continue
        source_date = previous.max()
        frame = members.loc[members["trade_date"].eq(source_date)].copy()
        frame["trade_date"] = date
        replacements.append(frame)
    result = members.loc[~members["trade_date"].isin(dates)]
    if replacements:
        result = pd.concat([result, *replacements], ignore_index=True)
    return result, [pd.Timestamp(date).strftime("%Y-%m-%d") for date in dates]


def latest_membership_backfill(
    concept_index: pd.DataFrame, members: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand the final observed membership backwards as an explicit look-ahead diagnostic."""
    latest_date = members["trade_date"].max()
    latest_members = members.loc[
        members["trade_date"].eq(latest_date), ["concept_code", "ts_code"]
    ].drop_duplicates(["concept_code", "ts_code"])
    latest_index = concept_index.loc[
        concept_index["trade_date"].eq(concept_index["trade_date"].max()),
        ["concept_code", "concept_name"],
    ].drop_duplicates("concept_code")
    dates = pd.DataFrame({"trade_date": sorted(concept_index["trade_date"].unique())})
    return dates.merge(latest_index, how="cross"), dates.merge(latest_members, how="cross")


def prepare_stock_panel(panel: pd.DataFrame, *, breadth_weight_lag: int = 0) -> pd.DataFrame:
    if breadth_weight_lag not in (0, 1):
        raise ValueError("breadth_weight_lag must be 0 or 1")
    stocks = panel.sort_values(["ts_code", "trade_date"]).copy()
    stocks["trade_date"] = pd.to_datetime(stocks["trade_date"])
    grouped = stocks.groupby("ts_code", sort=False)
    stocks["stock_return_1d"] = grouped["adj_close"].pct_change(fill_method=None)
    stocks["stock_return_20d"] = grouped["adj_close"].pct_change(20, fill_method=None)
    stocks["cap_lag1"] = grouped["circ_mv_cny"].shift(1)
    if "free_float_mv_cny" in stocks:
        free_float = pd.to_numeric(stocks["free_float_mv_cny"], errors="coerce")
        stocks["breadth_mv_cny"] = free_float.where(free_float.gt(0), stocks["circ_mv_cny"])
    else:
        stocks["breadth_mv_cny"] = stocks["circ_mv_cny"]
    if breadth_weight_lag:
        stocks["breadth_mv_cny"] = stocks.groupby("ts_code", sort=False)["breadth_mv_cny"].shift(1)
    stocks["amount_ma20"] = grouped["amount_cny"].transform(
        lambda values: values.rolling(20, min_periods=18).mean()
    )
    stocks["up_state_20d"] = stocks["stock_return_20d"].gt(0)
    for horizon in HORIZONS:
        entry = grouped["adj_open"].shift(-1)
        exit_price = grouped["adj_open"].shift(-(horizon + 1))
        stocks[f"forward_open_{horizon}d"] = exit_price / entry - 1
    return stocks


def build_concept_dataset(
    stock_panel: pd.DataFrame,
    concept_index: pd.DataFrame,
    members: pd.DataFrame,
    *,
    minimum_members: int = 8,
    maximum_members: int = 500,
    minimum_age: int = 60,
    maximum_churn: float = 0.30,
    breadth_weight_lag: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    stocks = prepare_stock_panel(stock_panel, breadth_weight_lag=breadth_weight_lag)
    supported = members["ts_code"].isin(set(stocks["ts_code"].astype(str)))
    member_totals = members.groupby(["trade_date", "concept_code"], observed=True).size().rename("source_member_count")
    stock_columns = [
        "trade_date", "ts_code", "circ_mv_cny", "breadth_mv_cny", "amount_cny", "up_state_20d",
        *[f"forward_open_{horizon}d" for horizon in HORIZONS],
    ]
    relations = members.loc[supported].merge(
        stocks[stock_columns],
        on=["trade_date", "ts_code"], how="inner", validate="many_to_one",
    )
    relations["cap_up"] = relations["breadth_mv_cny"] * relations["up_state_20d"].astype(float)
    current = relations.groupby(["trade_date", "concept_code"], observed=True).agg(
        matched_member_count=("ts_code", "nunique"),
        concept_circ_mv=("breadth_mv_cny", "sum"), cap_up=("cap_up", "sum"),
        breadth_equal_raw=("up_state_20d", "mean"), concept_amount=("amount_cny", "sum"),
    ).reset_index()
    current = current.merge(member_totals.reset_index(), on=["trade_date", "concept_code"], how="left")
    current["member_match_coverage"] = current["matched_member_count"] / current["source_member_count"]
    current["breadth_float_raw"] = current["cap_up"] / current["concept_circ_mv"]
    names = concept_index[["trade_date", "concept_code", "concept_name"]].drop_duplicates(
        ["trade_date", "concept_code"]
    )
    concepts = names.merge(current, on=["trade_date", "concept_code"], how="left", validate="one_to_one")
    calendar = pd.Index(sorted(stocks["trade_date"].dropna().unique()))
    next_date = {calendar[i]: calendar[i + 1] for i in range(len(calendar) - 1)}
    previous_relations = members.loc[supported, ["trade_date", "concept_code", "ts_code"]].copy()
    previous_relations["return_date"] = previous_relations["trade_date"].map(next_date)
    return_fields = stocks[["trade_date", "ts_code", "stock_return_1d", "cap_lag1"]].rename(
        columns={"trade_date": "return_date"}
    )
    returns = previous_relations.dropna(subset=["return_date"]).merge(
        return_fields, on=["return_date", "ts_code"], how="inner", validate="many_to_one"
    )
    returns["weighted_return"] = returns["cap_lag1"] * returns["stock_return_1d"]
    concept_returns = returns.groupby(["return_date", "concept_code"], observed=True).agg(
        return_weight=("cap_lag1", "sum"), weighted_return=("weighted_return", "sum")
    ).reset_index().rename(columns={"return_date": "trade_date"})
    concept_returns["concept_return_1d"] = concept_returns["weighted_return"] / concept_returns["return_weight"]
    concepts = concepts.merge(
        concept_returns[["trade_date", "concept_code", "concept_return_1d"]],
        on=["trade_date", "concept_code"], how="left", validate="one_to_one",
    )
    market = _market_returns(stocks)
    concepts = concepts.merge(market, on="trade_date", how="left", validate="many_to_one")
    concepts = _attach_common_membership_metrics(concepts, relations, calendar)
    concepts = concepts.sort_values(["concept_code", "trade_date"]).reset_index(drop=True)
    by_concept = concepts.groupby("concept_code", sort=False)
    concepts["concept_age_days"] = by_concept.cumcount() + 1
    concepts["breadth_float"] = by_concept["breadth_float_raw"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    concepts["breadth_delta_5d"] = concepts["breadth_float"] - by_concept["breadth_float"].shift(5)
    concepts["common_breadth_delta_smooth5"] = by_concept["common_breadth_delta_5d"].transform(
        lambda values: values.rolling(5, min_periods=3).mean()
    )
    for window in (5, 20, 60):
        concepts[f"concept_return_{window}d"] = by_concept["concept_return_1d"].transform(
            lambda values, w=window: _rolling_compound(values, w)
        )
    concepts["rs_20d"] = (
        (1 + concepts["concept_return_20d"]) / (1 + concepts["market_return_20d"]) - 1
    )
    by_concept = concepts.groupby("concept_code", sort=False)
    concepts["rs_momentum_5d"] = concepts["rs_20d"] - by_concept["rs_20d"].shift(5)
    concepts["rrg_quadrant"] = np.select(
        [
            concepts["rs_20d"].gt(0) & concepts["rs_momentum_5d"].gt(0),
            concepts["rs_20d"].le(0) & concepts["rs_momentum_5d"].gt(0),
            concepts["rs_20d"].gt(0) & concepts["rs_momentum_5d"].le(0),
        ], ["leading", "improving", "weakening"], default="lagging",
    )
    concepts["eligible_concept"] = (
        concepts["matched_member_count"].between(minimum_members, maximum_members)
        & concepts["member_match_coverage"].ge(0.90)
        & concepts["concept_age_days"].ge(minimum_age)
        & concepts["membership_churn_5d"].le(maximum_churn)
    )
    concepts = _attach_labels(concepts, relations)
    concepts = _attach_signals(concepts)
    audit = concept_data_audit(concept_index, members, relations, concepts, calendar)
    return relations, concepts.sort_values(["trade_date", "concept_code"]).reset_index(drop=True), audit


def _market_returns(stocks: pd.DataFrame) -> pd.DataFrame:
    valid = stocks["is_tradeable"].fillna(False).astype(bool) & stocks["cap_lag1"].gt(0)
    frame = stocks.loc[valid, ["trade_date", "cap_lag1", "stock_return_1d"]].dropna()
    frame["weighted"] = frame["cap_lag1"] * frame["stock_return_1d"]
    market = frame.groupby("trade_date").agg(weight=("cap_lag1", "sum"), weighted=("weighted", "sum")).reset_index()
    market["market_return_1d"] = market["weighted"] / market["weight"]
    market = market.sort_values("trade_date")
    for window in (5, 20, 60):
        market[f"market_return_{window}d"] = _rolling_compound(market["market_return_1d"], window)
    for horizon in HORIZONS:
        label = stocks.loc[valid, ["trade_date", "circ_mv_cny", f"forward_open_{horizon}d"]].dropna()
        label["weighted_label"] = label["circ_mv_cny"] * label[f"forward_open_{horizon}d"]
        grouped = label.groupby("trade_date").agg(w=("circ_mv_cny", "sum"), y=("weighted_label", "sum"))
        market = market.merge((grouped["y"] / grouped["w"]).rename(f"forward_market_{horizon}d"), left_on="trade_date", right_index=True, how="left")
    return market[[column for column in market if column.startswith("market_") or column.startswith("forward_") or column == "trade_date"]]


def _attach_common_membership_metrics(
    concepts: pd.DataFrame, relations: pd.DataFrame, calendar: pd.Index
) -> pd.DataFrame:
    # A whole-history self-join is several times larger than the source relation table.
    # Join one date pair at a time to keep peak memory bounded.
    positions = relations.groupby("trade_date", sort=False).indices
    aggregates = []
    for position in range(5, len(calendar)):
        date, previous_date = calendar[position], calendar[position - 5]
        current = relations.iloc[positions.get(date, [])][[
            "concept_code", "ts_code", "breadth_mv_cny", "up_state_20d",
        ]].rename(columns={"breadth_mv_cny": "current_cap", "up_state_20d": "current_up"})
        previous = relations.iloc[positions.get(previous_date, [])][[
            "concept_code", "ts_code", "breadth_mv_cny", "up_state_20d",
        ]].rename(columns={"breadth_mv_cny": "previous_cap", "up_state_20d": "previous_up"})
        common = current.merge(
            previous, on=["concept_code", "ts_code"], how="inner", validate="one_to_one",
        )
        if common.empty:
            continue
        common["current_cap_up"] = common["current_cap"] * common["current_up"].astype(float)
        common["previous_cap_up"] = common["previous_cap"] * common["previous_up"].astype(float)
        aggregated = common.groupby("concept_code", observed=True).agg(
            common_member_count=("ts_code", "nunique"), current_cap=("current_cap", "sum"),
            current_cap_up=("current_cap_up", "sum"), previous_cap=("previous_cap", "sum"),
            previous_cap_up=("previous_cap_up", "sum"),
        ).reset_index()
        aggregated["trade_date"] = date
        aggregates.append(aggregated)
    aggregated = pd.concat(aggregates, ignore_index=True) if aggregates else pd.DataFrame(columns=[
        "trade_date", "concept_code", "common_member_count", "current_cap", "current_cap_up",
        "previous_cap", "previous_cap_up",
    ])
    aggregated["common_breadth_current"] = aggregated["current_cap_up"] / aggregated["current_cap"]
    aggregated["common_breadth_previous"] = aggregated["previous_cap_up"] / aggregated["previous_cap"]
    aggregated["common_breadth_delta_5d"] = aggregated["common_breadth_current"] - aggregated["common_breadth_previous"]
    result = concepts.merge(
        aggregated[["trade_date", "concept_code", "common_member_count", "common_breadth_delta_5d"]],
        on=["trade_date", "concept_code"], how="left", validate="one_to_one",
    )
    ordered = result.sort_values(["concept_code", "trade_date"]).copy()
    previous_count = ordered.groupby("concept_code", sort=False)["matched_member_count"].shift(5)
    union = ordered["matched_member_count"] + previous_count - ordered["common_member_count"]
    ordered["membership_churn_5d"] = 1 - ordered["common_member_count"] / union.replace(0, np.nan)
    return ordered.sort_index()


def _attach_labels(concepts: pd.DataFrame, relations: pd.DataFrame) -> pd.DataFrame:
    result = concepts.copy()
    for horizon in HORIZONS:
        label = relations[[
            "trade_date", "concept_code", "breadth_mv_cny", f"forward_open_{horizon}d",
        ]].dropna()
        label["weighted"] = label["breadth_mv_cny"] * label[f"forward_open_{horizon}d"]
        grouped = label.groupby(["trade_date", "concept_code"], observed=True).agg(
            w=("breadth_mv_cny", "sum"), y=("weighted", "sum")
        ).reset_index()
        grouped[f"forward_concept_{horizon}d"] = grouped["y"] / grouped["w"]
        result = result.merge(
            grouped[["trade_date", "concept_code", f"forward_concept_{horizon}d"]],
            on=["trade_date", "concept_code"], how="left", validate="one_to_one",
        )
        result[f"forward_excess_{horizon}d"] = (
            result[f"forward_concept_{horizon}d"] - result[f"forward_market_{horizon}d"]
        )
    return result


def _attach_signals(concepts: pd.DataFrame) -> pd.DataFrame:
    result = concepts.copy()
    result["breadth_delta_rank"] = result.groupby("trade_date")["breadth_delta_5d"].rank(pct=True)
    result["common_delta_rank"] = result.groupby("trade_date")["common_breadth_delta_smooth5"].rank(pct=True)
    result["rs_z"] = result.groupby("trade_date")["rs_20d"].transform(_cs_zscore)
    result["rs_momentum_z"] = result.groupby("trade_date")["rs_momentum_5d"].transform(_cs_zscore)
    result["signal_hot_1d"] = result["concept_return_1d"]
    result["signal_momentum_20d"] = result["concept_return_20d"]
    result["signal_breadth_delta"] = result["breadth_delta_5d"]
    result["signal_common_breadth_delta"] = result["common_breadth_delta_smooth5"]
    result["signal_rrg_only"] = result["rs_z"] + result["rs_momentum_z"]
    current_mask = (
        result["eligible_concept"] & result["breadth_float"].gt(0.50)
        & result["breadth_delta_rank"].ge(0.70) & result["rs_momentum_5d"].gt(0)
    )
    common_mask = (
        result["eligible_concept"] & result["breadth_float"].gt(0.50)
        & result["common_delta_rank"].ge(0.70) & result["rs_momentum_5d"].gt(0)
    )
    result["signal_current_breadth_rrg"] = result["breadth_delta_5d"].where(current_mask)
    result["signal_common_breadth_rrg"] = result["common_breadth_delta_smooth5"].where(common_mask)
    result["signal_churn"] = result["membership_churn_5d"]
    return result


def concept_data_audit(index: pd.DataFrame, members: pd.DataFrame, relations: pd.DataFrame,
                       concepts: pd.DataFrame, calendar: pd.Index) -> dict:
    supported_member_rows = len(relations)
    member_dates = set(members["trade_date"].unique())
    index_dates = set(index["trade_date"].unique())
    counts = index.groupby("trade_date")["concept_code"].nunique()
    multi = members.groupby(["trade_date", "ts_code"], observed=True).size()
    return {
        "start_date": pd.Timestamp(index["trade_date"].min()).strftime("%Y-%m-%d"),
        "end_date": pd.Timestamp(index["trade_date"].max()).strftime("%Y-%m-%d"),
        "index_rows": int(len(index)), "member_rows": int(len(members)),
        "supported_member_rows": int(supported_member_rows),
        "member_support_coverage": float(supported_member_rows / len(members)) if len(members) else 0,
        "concepts": int(index["concept_code"].nunique()),
        "stocks": int(members["ts_code"].nunique()),
        "snapshot_dates": int(len(index_dates)),
        "missing_member_snapshot_dates": int(len(index_dates - member_dates)),
        "duplicate_member_keys": int(members.duplicated(["trade_date", "concept_code", "ts_code"]).sum()),
        "min_daily_concepts": int(counts.min()), "median_daily_concepts": float(counts.median()),
        "max_daily_concepts": int(counts.max()),
        "median_concepts_per_stock": float(multi.median()),
        "p95_concepts_per_stock": float(multi.quantile(0.95)),
        "eligible_concept_days": int(concepts["eligible_concept"].fillna(False).sum()),
    }


def select_deduplicated_concepts(
    features: pd.DataFrame, members: pd.DataFrame, score_column: str,
    *, top_n: int = 10, preselect: int = 30, jaccard_limit: float = 0.80,
) -> pd.DataFrame:
    scored = features.dropna(subset=[score_column]).sort_values(
        ["trade_date", score_column], ascending=[True, False]
    ).groupby("trade_date").head(preselect)
    needed = members.merge(
        scored[["trade_date", "concept_code"]], on=["trade_date", "concept_code"], how="inner"
    )
    sets = needed.groupby(["trade_date", "concept_code"])["ts_code"].agg(lambda values: frozenset(values))
    rows = []
    for date, day in scored.groupby("trade_date", sort=True):
        accepted: list[tuple[str, frozenset]] = []
        for _, row in day.iterrows():
            code = row["concept_code"]
            members_set = sets.get((date, code), frozenset())
            if not members_set:
                continue
            if any(_jaccard(members_set, existing) > jaccard_limit for _, existing in accepted):
                continue
            accepted.append((code, members_set))
            rows.append({
                "trade_date": date, "concept_code": code,
                "selection_rank": len(accepted), "score": row[score_column],
            })
            if len(accepted) >= top_n:
                break
    return pd.DataFrame(rows)


def evaluate_concept_signals(
    features: pd.DataFrame, members: pd.DataFrame,
    *, splits: dict[str, tuple[str, str]], horizon: int = 5,
    top_n: int = 10, roundtrip_cost_bps: float = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    daily_rows = []
    label = f"forward_excess_{horizon}d"
    for split, (start, end) in splits.items():
        period = features.loc[features["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        period = period.loc[period["eligible_concept"].fillna(False)]
        period_members = members.loc[members["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        for name, column in SIGNALS.items():
            sample = period[["trade_date", "concept_code", column, label]].rename(columns={column: "score"}).dropna()
            ic = sample.groupby("trade_date").apply(
                lambda day: day["score"].corr(day[label], method="spearman") if len(day) >= 20 else np.nan,
                include_groups=False,
            )
            selected = select_deduplicated_concepts(
                period.rename(columns={column: "_active_score"}), period_members, "_active_score",
                top_n=top_n,
            )
            payoff = selected.merge(
                period[["trade_date", "concept_code", label]], on=["trade_date", "concept_code"],
                how="left", validate="one_to_one",
            ).groupby("trade_date")[label].mean()
            net = payoff - roundtrip_cost_bps / 10_000
            gross_stats = newey_west_mean(payoff, horizon - 1)
            net_stats = newey_west_mean(net, horizon - 1)
            lower, upper = block_bootstrap_mean(payoff.to_numpy(float), block=20, samples=2_000, seed=42)
            rows.append({
                "split": split, "signal": name, "horizon": horizon,
                "days": int(len(payoff)), "mean_candidates": float(sample.groupby("trade_date").size().mean()) if len(sample) else None,
                "rank_ic": float(ic.mean()) if len(ic.dropna()) else None,
                "rank_ic_nw_t": newey_west_mean(ic, horizon - 1)["t_value"],
                "top_gross_excess": gross_stats["mean"], "top_gross_nw_t": gross_stats["t_value"],
                "bootstrap_95_low": lower, "bootstrap_95_high": upper,
                "top_net_excess_20bps": net_stats["mean"], "top_net_nw_t": net_stats["t_value"],
            })
            if len(payoff):
                daily_rows.append(pd.DataFrame({
                    "trade_date": payoff.index, "split": split, "signal": name,
                    "gross_excess": payoff.values, "net_excess_20bps": net.values,
                }))
    return pd.DataFrame(rows), pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()


def paired_signal_differences(
    daily: pd.DataFrame,
    comparisons: Iterable[tuple[str, str]] = (
        ("common_membership_breadth_rrg", "rrg_only"),
        ("current_membership_breadth_rrg", "rrg_only"),
        ("common_membership_breadth_rrg", "momentum_20d"),
        ("current_membership_breadth_rrg", "momentum_20d"),
    ),
    *, lags: int = 4,
) -> pd.DataFrame:
    rows = []
    for (mode, split), group in daily.groupby(["membership_mode", "split"], observed=True):
        pivot = group.pivot(index="trade_date", columns="signal", values="net_excess_20bps")
        for left, right in comparisons:
            if left not in pivot or right not in pivot:
                continue
            difference = (pivot[left] - pivot[right]).dropna()
            stats = newey_west_mean(difference, lags)
            low, high = block_bootstrap_mean(
                difference.to_numpy(float), block=20, samples=2_000, seed=314159,
            )
            rows.append({
                "membership_mode": mode, "split": split,
                "signal": left, "benchmark_signal": right, "days": len(difference),
                "incremental_net_excess": stats["mean"],
                "incremental_nw_t": stats["t_value"],
                "bootstrap_95_low": low, "bootstrap_95_high": high,
            })
    return pd.DataFrame(rows)


def newey_west_mean(values: pd.Series, lags: int) -> dict:
    clean = pd.Series(values).dropna().to_numpy(float)
    n = len(clean)
    if n < 3:
        return {"mean": float(clean.mean()) if n else None, "t_value": None}
    lags = min(max(int(lags), 0), n - 1)
    centered = clean - clean.mean()
    variance = float(centered @ centered / n)
    for lag in range(1, lags + 1):
        variance += 2 * (1 - lag / (lags + 1)) * float(centered[lag:] @ centered[:-lag] / n)
    variance = max(variance, 0) / n
    return {"mean": float(clean.mean()), "t_value": float(clean.mean() / math.sqrt(variance)) if variance > 0 else None}


def block_bootstrap_mean(values: np.ndarray, *, block: int, samples: int, seed: int) -> tuple[float | None, float | None]:
    values = values[np.isfinite(values)]
    if len(values) < block:
        return None, None
    rng = np.random.default_rng(seed)
    starts = np.arange(len(values) - block + 1)
    blocks = int(math.ceil(len(values) / block))
    means = []
    for _ in range(samples):
        chosen = rng.choice(starts, blocks, replace=True)
        draw = np.concatenate([values[start:start + block] for start in chosen])[:len(values)]
        means.append(draw.mean())
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _rolling_compound(values: pd.Series, window: int) -> pd.Series:
    return np.expm1(np.log1p(values.clip(lower=-0.999999)).rolling(window, min_periods=window).sum())


def _cs_zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    return (values - values.mean()) / std if np.isfinite(std) and std > 0 else pd.Series(np.nan, index=values.index)


def _jaccard(left: frozenset, right: frozenset) -> float:
    union = len(left | right)
    return len(left & right) / union if union else 0.0
