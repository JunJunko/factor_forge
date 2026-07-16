from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from factor_forge.research.concept_portfolio_backtest import (
    TargetBuildResult,
    cap_and_redistribute,
)


EXPOSURE_BY_REGIME = {
    "retreat": 0.15,
    "repair": 1.00,
    "divergence": 0.65,
    "overheat": 0.40,
    "neutral": 0.35,
}


@dataclass(frozen=True)
class LifecycleRules:
    decision_interval: int = 5
    concepts_per_rebalance: int = 5
    core_stocks_per_concept: int = 2
    catchup_stocks_per_concept: int = 3
    minimum_adv20_cny: float = 20_000_000.0
    maximum_stock_weight: float = 0.10
    maximum_holding_days: int = 10
    jaccard_limit: float = 0.80


def attach_enhanced_stock_features(
    panel: pd.DataFrame, daily_basic: pd.DataFrame,
) -> pd.DataFrame:
    result = panel.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    basic = daily_basic.copy()
    basic["trade_date"] = pd.to_datetime(basic["trade_date"])
    basic = basic[[
        "trade_date", "ts_code", "free_share", "turnover_rate_f", "volume_ratio",
    ]].drop_duplicates(["trade_date", "ts_code"], keep="last")
    result = result.merge(basic, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    result["free_float_mv_cny"] = (
        pd.to_numeric(result["raw_close"], errors="coerce")
        * pd.to_numeric(result["free_share"], errors="coerce") * 10_000.0
    )
    result["free_float_mv_cny"] = result["free_float_mv_cny"].where(
        result["free_float_mv_cny"].gt(0), result["circ_mv_cny"]
    )
    result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    grouped = result.groupby("ts_code", sort=False)
    result["stock_return_1d"] = grouped["adj_close"].pct_change(fill_method=None)
    result["stock_return_5d"] = grouped["adj_close"].pct_change(5, fill_method=None)
    result["stock_return_20d"] = grouped["adj_close"].pct_change(20, fill_method=None)
    result["amount_ma5"] = grouped["amount_cny"].transform(
        lambda values: values.rolling(5, min_periods=4).mean()
    )
    result["amount_ma20"] = grouped["amount_cny"].transform(
        lambda values: values.rolling(20, min_periods=18).mean()
    )
    result["amount_ratio_5_20"] = result["amount_ma5"] / result["amount_ma20"]
    result["turnover_f_ma5"] = grouped["turnover_rate_f"].transform(
        lambda values: values.rolling(5, min_periods=4).mean()
    )
    result["turnover_f_delta5"] = result["turnover_f_ma5"] - grouped["turnover_f_ma5"].shift(5)
    result["volatility_20d"] = result["stock_return_1d"].groupby(result["ts_code"]).transform(
        lambda values: values.rolling(20, min_periods=18).std()
    ).groupby(result["ts_code"]).shift(1)
    return result


def build_document_market_regimes(stock_panel: pd.DataFrame) -> pd.DataFrame:
    stocks = stock_panel.sort_values(["ts_code", "trade_date"]).copy()
    valid = stocks["is_tradeable"].fillna(False) & stocks["stock_return_20d"].notna()
    eligible = stocks.loc[valid].copy()
    daily = eligible.groupby("trade_date").agg(
        market_breadth=("stock_return_20d", lambda x: float(x.gt(0).mean())),
        median_amount_ratio=("amount_ratio_5_20", "median"),
    )
    one_day = stocks.loc[
        stocks["is_tradeable"].fillna(False) & stocks["stock_return_1d"].notna()
        & stocks["circ_mv_cny"].gt(0)
    ].copy()
    one_day["cap_lag"] = one_day.groupby("ts_code", sort=False)["circ_mv_cny"].shift(1)
    one_day = one_day.loc[one_day["cap_lag"].gt(0)]
    one_day["weighted"] = one_day["stock_return_1d"] * one_day["cap_lag"]
    cap_return = one_day.groupby("trade_date").apply(
        lambda day: day["weighted"].sum() / day["cap_lag"].sum(), include_groups=False,
    ).rename("cap_return_1d")
    equal_return = one_day.groupby("trade_date")["stock_return_1d"].mean().rename("equal_return_1d")
    one_day["size_bucket"] = one_day.groupby("trade_date")["cap_lag"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 3, labels=False, duplicates="drop")
    )
    size_return = one_day.groupby(["trade_date", "size_bucket"], observed=True)["stock_return_1d"].mean().unstack()
    small_large = (size_return.get(0, 0.0) - size_return.get(2, 0.0)).rename("small_minus_large_1d")
    daily = daily.join([cap_return, equal_return, small_large], how="outer").sort_index().reset_index()
    daily["breadth_delta_5d"] = daily["market_breadth"] - daily["market_breadth"].shift(5)
    for column, window, output in (
        ("cap_return_1d", 20, "market_return_20d"),
        ("equal_return_1d", 20, "equal_return_20d"),
        ("small_minus_large_1d", 10, "small_minus_large_10d"),
    ):
        daily[output] = np.expm1(
            np.log1p(daily[column].clip(lower=-0.999999)).rolling(window, min_periods=window).sum()
        )
    daily["breadth_q80_prior"] = daily["market_breadth"].shift(1).rolling(120, min_periods=60).quantile(0.80)
    overheat = (
        daily["market_return_20d"].gt(0)
        & daily["market_breadth"].ge(daily["breadth_q80_prior"])
        & daily["median_amount_ratio"].ge(1.15)
    )
    retreat = daily["market_return_20d"].le(0) & daily["breadth_delta_5d"].le(0)
    repair = (
        daily["market_return_20d"].gt(0) & daily["breadth_delta_5d"].gt(0)
        & daily["small_minus_large_10d"].gt(0)
    )
    divergence = daily["market_return_20d"].gt(0) & daily["small_minus_large_10d"].le(0)
    daily["regime"] = np.select(
        [overheat, retreat, repair, divergence],
        ["overheat", "retreat", "repair", "divergence"],
        default="neutral",
    )
    daily["target_exposure"] = daily["regime"].map(EXPOSURE_BY_REGIME).astype(float)
    daily["stock_mode"] = np.where(daily["regime"].eq("repair"), "catchup", "core")
    return daily


def attach_lifecycle_fields(features: pd.DataFrame) -> pd.DataFrame:
    result = features.sort_values(["concept_code", "trade_date"]).copy()
    grouped = result.groupby("concept_code", sort=False)
    result["previous_rrg_quadrant"] = grouped["rrg_quadrant"].shift(1)
    result["breadth_negative_2d"] = (
        result["common_breadth_delta_smooth5"].lt(0)
        & grouped["common_breadth_delta_smooth5"].shift(1).lt(0)
    )
    result["lifecycle"] = np.select(
        [
            result["rrg_quadrant"].eq("improving") & ~result["previous_rrg_quadrant"].eq("improving"),
            result["rrg_quadrant"].eq("leading") & result["previous_rrg_quadrant"].eq("improving"),
            result["rrg_quadrant"].eq("leading"),
            result["rrg_quadrant"].isin(["weakening", "lagging"]),
        ],
        ["new_improving", "confirmed_leading", "persistent_leading", "exit"],
        default="persistent_improving",
    )
    result["lifecycle_weight"] = result["lifecycle"].map({
        "new_improving": 0.50,
        "persistent_improving": 0.50,
        "confirmed_leading": 1.00,
        "persistent_leading": 0.75,
        "exit": 0.0,
    }).fillna(0.0)
    result["breadth_placebo_rank"] = np.nan
    for date, indices in result.groupby("trade_date", sort=True).groups.items():
        values = result.loc[indices, "common_delta_rank"]
        valid = values.notna()
        rng = np.random.default_rng(int(pd.Timestamp(date).strftime("%Y%m%d")) + 9173)
        result.loc[values.index[valid], "breadth_placebo_rank"] = rng.permutation(values.loc[valid].to_numpy())
    return result.sort_values(["trade_date", "concept_code"]).reset_index(drop=True)


def build_lifecycle_targets(
    features: pd.DataFrame,
    members: pd.DataFrame,
    stock_panel: pd.DataFrame,
    regimes: pd.DataFrame,
    *,
    variant: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    offset: int,
    rules: LifecycleRules = LifecycleRules(),
) -> TargetBuildResult:
    if variant not in {"A_rrg_baseline", "B_breadth_filter", "C_full_lifecycle", "D_breadth_placebo"}:
        raise KeyError(variant)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    calendar = pd.Index(sorted(stock_panel.loc[stock_panel["trade_date"].between(start, end), "trade_date"].unique()))
    if not 0 <= offset < rules.decision_interval:
        raise ValueError("offset outside decision interval")
    feature_days = {pd.Timestamp(d): day.set_index("concept_code") for d, day in features.loc[
        features["trade_date"].isin(calendar)
    ].groupby("trade_date", sort=False)}
    member_days = {pd.Timestamp(d): day[["concept_code", "ts_code"]] for d, day in members.loc[
        members["trade_date"].isin(calendar)
    ].groupby("trade_date", sort=False)}
    stock_days = {pd.Timestamp(d): day.set_index("ts_code") for d, day in stock_panel.loc[
        stock_panel["trade_date"].isin(calendar)
    ].groupby("trade_date", sort=False)}
    regime_days = regimes.set_index("trade_date").to_dict("index")
    active: dict[str, dict] = {}
    stock_lists: dict[str, list[str]] = {}
    previous_mode: str | None = None
    target_rows: list[dict] = []
    selection_rows: list[dict] = []
    entry_dates: list[pd.Timestamp] = []

    for position, date in enumerate(calendar[:-1]):
        date = pd.Timestamp(date)
        entry_date = pd.Timestamp(calendar[position + 1])
        day = feature_days.get(date)
        day_members = member_days.get(date, pd.DataFrame(columns=["concept_code", "ts_code"]))
        stocks = stock_days.get(date)
        regime_info = regime_days.get(date, {"regime": "neutral", "target_exposure": 0.35, "stock_mode": "core"})
        regime = str(regime_info["regime"])
        mode = str(regime_info["stock_mode"])
        exposure = float(regime_info["target_exposure"])
        decision_day = position % rules.decision_interval == offset
        if day is None or stocks is None:
            continue

        if regime == "retreat":
            active.clear()
            stock_lists.clear()
        elif variant in {"C_full_lifecycle", "D_breadth_placebo"}:
            for code in list(active):
                if code not in day.index:
                    active.pop(code, None); stock_lists.pop(code, None); continue
                row = day.loc[code]
                held = position - int(active[code]["entry_position"])
                breadth_exit = bool(row.get("breadth_negative_2d", False))
                if row["rrg_quadrant"] in {"weakening", "lagging"} or breadth_exit or held >= rules.maximum_holding_days:
                    active.pop(code, None); stock_lists.pop(code, None)

        if decision_day and regime not in {"retreat", "overheat"}:
            candidates = _concept_candidates(day, variant)
            selected_codes = _deduplicate_codes(candidates, day_members, rules)
            if variant in {"A_rrg_baseline", "B_breadth_filter"}:
                active = {
                    code: {"entry_position": position, "score": float(candidates.loc[code, "selection_score"])}
                    for code in selected_codes
                }
                stock_lists = {}
            else:
                for code in selected_codes:
                    if code not in active and len(active) < rules.concepts_per_rebalance:
                        active[code] = {"entry_position": position, "score": float(candidates.loc[code, "selection_score"])}

        reselection = decision_day or mode != previous_mode
        if reselection:
            for code in list(active):
                if code not in day.index:
                    continue
                picks = _select_stocks(code, day.loc[code], day_members, stocks, mode, rules)
                if picks:
                    stock_lists[code] = picks
                else:
                    stock_lists.pop(code, None)
        previous_mode = mode

        contributions: list[dict] = []
        concept_weight_sum = sum(
            float(day.loc[code, "lifecycle_weight"]) for code in active if code in day.index and code in stock_lists
        )
        if concept_weight_sum > 0 and exposure > 0:
            concept_counts = day_members.groupby("ts_code", observed=True)["concept_code"].nunique()
            for code in active:
                if code not in day.index or code not in stock_lists:
                    continue
                lifecycle_weight = float(day.loc[code, "lifecycle_weight"])
                picks = stock_lists[code]
                for stock in picks:
                    dehub = 1 / math.sqrt(max(float(concept_counts.get(stock, 1)), 1.0))
                    contributions.append({
                        "ts_code": stock,
                        "raw_weight": lifecycle_weight / concept_weight_sum / len(picks) * dehub,
                        "concept_code": code,
                    })
                selection_rows.append({
                    "signal_date": date, "entry_date": entry_date, "concept_code": code,
                    "variant": variant, "regime": regime, "stock_mode": mode,
                    "lifecycle": day.loc[code, "lifecycle"], "concept_score": active[code]["score"],
                    "selected_stocks": len(picks),
                })
        if contributions:
            contribution = pd.DataFrame(contributions)
            aggregated = contribution.groupby("ts_code", observed=True).agg(
                raw_weight=("raw_weight", "sum"), supporting_concepts=("concept_code", "nunique")
            ).reset_index()
            relative_cap = min(rules.maximum_stock_weight / exposure, 1.0)
            aggregated["target_weight"] = cap_and_redistribute(
                aggregated["raw_weight"], relative_cap
            ).to_numpy() * exposure
            for row in aggregated.itertuples(index=False):
                target_rows.append({
                    "signal_date": date, "entry_date": entry_date, "ts_code": row.ts_code,
                    "target_weight": row.target_weight, "supporting_concepts": row.supporting_concepts,
                    "variant": variant, "regime": regime, "stock_mode": mode,
                })
        entry_dates.append(entry_date)

    targets = pd.DataFrame(target_rows)
    selections = pd.DataFrame(selection_rows)
    return TargetBuildResult(targets, selections, sorted(set(entry_dates)), sorted(set(entry_dates)))


def _concept_candidates(day: pd.DataFrame, variant: str) -> pd.DataFrame:
    candidates = day.loc[
        day["eligible_concept"].fillna(False) & day["rrg_quadrant"].isin(["leading", "improving"])
    ].copy()
    candidates["selection_score"] = candidates["signal_rrg_only"]
    if variant != "A_rrg_baseline":
        rank_column = "breadth_placebo_rank" if variant == "D_breadth_placebo" else "common_delta_rank"
        candidates = candidates.loc[
            candidates["breadth_float"].gt(0.50) & candidates[rank_column].ge(0.70)
        ].copy()
        candidates["selection_score"] = (
            candidates["signal_rrg_only"] + candidates[rank_column].fillna(0.0)
        )
    return candidates.sort_values("selection_score", ascending=False)


def _deduplicate_codes(candidates: pd.DataFrame, members: pd.DataFrame, rules: LifecycleRules) -> list[str]:
    sets = members.groupby("concept_code", observed=True)["ts_code"].agg(lambda values: frozenset(values))
    selected: list[str] = []
    selected_sets: list[frozenset] = []
    for code in candidates.index:
        member_set = sets.get(code, frozenset())
        if not member_set:
            continue
        if any(len(member_set & other) / len(member_set | other) > rules.jaccard_limit for other in selected_sets):
            continue
        selected.append(str(code)); selected_sets.append(member_set)
        if len(selected) >= rules.concepts_per_rebalance:
            break
    return selected


def _select_stocks(
    concept_code: str,
    concept: pd.Series,
    members: pd.DataFrame,
    stocks: pd.DataFrame,
    mode: str,
    rules: LifecycleRules,
) -> list[str]:
    codes = members.loc[members["concept_code"].eq(concept_code), "ts_code"].astype(str)
    available = stocks.loc[stocks.index.intersection(codes)].copy()
    available = available.loc[
        available["is_tradeable"].fillna(False)
        & available["amount_ma20"].ge(rules.minimum_adv20_cny)
    ]
    if available.empty:
        return []
    if mode == "catchup":
        percentile = available["stock_return_20d"].rank(pct=True)
        available = available.loc[
            percentile.between(0.20, 0.60)
            & available["stock_return_5d"].gt(0)
            & available["amount_ratio_5_20"].gt(1.0)
            & available["turnover_f_delta5"].gt(0)
        ].copy()
        if available.empty:
            return []
        available["selection_score"] = (
            (-np.log(available["free_float_mv_cny"].clip(lower=1))).rank(pct=True)
            + available["turnover_f_delta5"].rank(pct=True)
            + available["amount_ratio_5_20"].rank(pct=True)
            + available["stock_return_5d"].rank(pct=True)
        ) / 4
        count = rules.catchup_stocks_per_concept
    else:
        available = available.loc[available["stock_return_20d"].gt(0)].copy()
        if available.empty:
            return []
        relative = available["stock_return_20d"] - float(concept.get("concept_return_20d", 0.0))
        available["selection_score"] = (
            np.log(available["amount_ma20"].clip(lower=1)).rank(pct=True) * 0.30
            + np.log(available["free_float_mv_cny"].clip(lower=1)).rank(pct=True) * 0.25
            + available["stock_return_20d"].rank(pct=True) * 0.25
            + relative.rank(pct=True) * 0.20
        )
        count = rules.core_stocks_per_concept
    return available.sort_values(["selection_score", "amount_ma20"], ascending=False).head(count).index.astype(str).tolist()
