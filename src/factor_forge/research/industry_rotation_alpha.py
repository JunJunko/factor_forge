from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


PRIMARY_HORIZONS = (1, 3, 5, 10, 20)
SIGNAL_COLUMNS = {
    "hot_1d": "signal_hot_1d",
    "momentum_5d": "signal_momentum_5d",
    "momentum_20d": "signal_momentum_20d",
    "breadth_level": "signal_breadth_level",
    "breadth_delta": "signal_breadth_delta",
    "rrg_only": "signal_rrg_only",
    "breadth_rrg": "signal_breadth_rrg",
    "existing_selector_proxy": "signal_existing_proxy",
    "weakening_placebo": "signal_weakening_placebo",
}


@dataclass(frozen=True)
class RotationAudit:
    rows: int
    start_date: str
    end_date: str
    stock_count: int
    group_count: int
    membership_coverage: float
    tradeable_membership_coverage: float
    min_daily_groups: int
    median_daily_groups: float
    max_daily_groups: int
    duplicate_keys: int
    overlapping_intervals: int
    current_membership_violations: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def normalize_sw_l2_membership(membership: pd.DataFrame) -> pd.DataFrame:
    required = {
        "ts_code", "l2_code", "l2_name", "in_date", "out_date", "is_new",
    }
    missing = sorted(required - set(membership.columns))
    if missing:
        raise ValueError(f"SW L2 membership missing columns: {missing}")
    result = membership.loc[membership["l2_code"].notna(), list(required)].copy()
    result = result.rename(columns={"l2_code": "industry_code", "l2_name": "industry_name"})
    result["ts_code"] = result["ts_code"].astype(str)
    result["industry_code"] = result["industry_code"].astype(str)
    result["in_date"] = pd.to_datetime(result["in_date"], errors="coerce")
    result["out_date"] = pd.to_datetime(result["out_date"], errors="coerce")
    result = result.dropna(subset=["ts_code", "industry_code", "in_date"])
    result = result.drop_duplicates(
        ["ts_code", "industry_code", "in_date", "out_date"], keep="last"
    ).sort_values(["ts_code", "in_date", "industry_code"])
    current_counts = result.loc[result["out_date"].isna()].groupby("ts_code").size()
    if (current_counts > 1).any():
        raise ValueError("multiple current SW L2 memberships for a stock")
    overlap_count = _count_interval_overlaps(result)
    if overlap_count:
        raise ValueError(f"overlapping SW L2 membership intervals: {overlap_count}")
    return result.reset_index(drop=True)


def _count_interval_overlaps(membership: pd.DataFrame) -> int:
    ordered = membership.sort_values(["ts_code", "in_date"])
    previous_out = ordered.groupby("ts_code", sort=False)["out_date"].shift(1)
    previous_code = ordered.groupby("ts_code", sort=False)["industry_code"].shift(1)
    overlap = previous_out.notna() & ordered["in_date"].lt(previous_out)
    overlap &= ordered["industry_code"].ne(previous_code)
    return int(overlap.sum())


def attach_pit_membership(panel: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Attach the last effective interval and reject rows outside its out_date.

    Sorting by the as-of key first is required by pandas even when ``by`` is used.
    The original row order is restored before returning.
    """
    members = normalize_sw_l2_membership(membership)
    left = panel.copy()
    left["trade_date"] = pd.to_datetime(left["trade_date"])
    left["ts_code"] = left["ts_code"].astype(str)
    left["_rotation_row"] = np.arange(len(left), dtype=np.int64)
    left = left.drop(columns=[
        column for column in ("industry_code", "industry_name", "membership_in_date", "membership_out_date")
        if column in left
    ])
    right = members.rename(columns={
        "in_date": "membership_in_date", "out_date": "membership_out_date",
    })[[
        "ts_code", "industry_code", "industry_name", "membership_in_date",
        "membership_out_date",
    ]]
    merged = pd.merge_asof(
        left.sort_values(["trade_date", "ts_code"]),
        right.sort_values(["membership_in_date", "ts_code"]),
        by="ts_code", left_on="trade_date", right_on="membership_in_date",
        direction="backward", allow_exact_matches=True,
    )
    expired = merged["membership_out_date"].notna() & merged["trade_date"].ge(
        merged["membership_out_date"]
    )
    merged.loc[expired, [
        "industry_code", "industry_name", "membership_in_date", "membership_out_date",
    ]] = [None, None, pd.NaT, pd.NaT]
    return merged.sort_values("_rotation_row").drop(columns="_rotation_row").reset_index(drop=True)


def attach_stitched_pit_membership(
    panel: pd.DataFrame,
    legacy_membership: pd.DataFrame,
    current_membership: pd.DataFrame,
    *,
    switch_date: str = "2021-07-30",
) -> pd.DataFrame:
    """Use the taxonomy available at the time and reset group histories at the switch."""
    dates = pd.to_datetime(panel["trade_date"])
    switch = pd.Timestamp(switch_date)
    legacy = attach_pit_membership(panel.loc[dates.lt(switch)].copy(), legacy_membership)
    current = attach_pit_membership(panel.loc[dates.ge(switch)].copy(), current_membership)
    legacy["industry_code"] = "SW2014:" + legacy["industry_code"].astype("string")
    current["industry_code"] = "SW2021:" + current["industry_code"].astype("string")
    legacy.loc[legacy["industry_name"].isna(), "industry_code"] = pd.NA
    current.loc[current["industry_name"].isna(), "industry_code"] = pd.NA
    return pd.concat([legacy, current], ignore_index=True).sort_values(
        ["trade_date", "ts_code"]
    ).reset_index(drop=True)


def audit_rotation_panel(panel: pd.DataFrame, membership: pd.DataFrame) -> RotationAudit:
    normalized = normalize_sw_l2_membership(membership)
    duplicate_keys = int(panel.duplicated(["trade_date", "ts_code"]).sum())
    daily_groups = panel.dropna(subset=["industry_code"]).groupby("trade_date")[
        "industry_code"
    ].nunique()
    tradeable = panel["is_tradeable"].fillna(False).astype(bool)
    current_counts = normalized.loc[normalized["out_date"].isna()].groupby("ts_code").size()
    return RotationAudit(
        rows=int(len(panel)),
        start_date=pd.Timestamp(panel["trade_date"].min()).strftime("%Y-%m-%d"),
        end_date=pd.Timestamp(panel["trade_date"].max()).strftime("%Y-%m-%d"),
        stock_count=int(panel["ts_code"].nunique()),
        group_count=int(panel["industry_code"].nunique()),
        membership_coverage=float(panel["industry_code"].notna().mean()),
        tradeable_membership_coverage=float(panel.loc[tradeable, "industry_code"].notna().mean()),
        min_daily_groups=int(daily_groups.min()),
        median_daily_groups=float(daily_groups.median()),
        max_daily_groups=int(daily_groups.max()),
        duplicate_keys=duplicate_keys,
        overlapping_intervals=_count_interval_overlaps(normalized),
        current_membership_violations=int((current_counts != 1).sum()),
    )


def build_rotation_dataset(
    panel: pd.DataFrame,
    *,
    horizons: Iterable[int] = PRIMARY_HORIZONS,
    minimum_members: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return stock-level trailing features and group-day features/labels."""
    stocks = panel.sort_values(["ts_code", "trade_date"]).copy()
    grouped_stock = stocks.groupby("ts_code", sort=False)
    stocks["stock_return_1d"] = grouped_stock["adj_close"].pct_change(fill_method=None)
    stocks["stock_return_20d"] = grouped_stock["adj_close"].pct_change(20, fill_method=None)
    stocks["weight_lag1"] = grouped_stock["circ_mv_cny"].shift(1)
    stocks["amount_ma20"] = grouped_stock["amount_cny"].transform(
        lambda values: values.rolling(20, min_periods=18).mean()
    )
    valid = (
        stocks["is_tradeable"].fillna(False).astype(bool)
        & stocks["industry_code"].notna()
        & stocks["adj_close"].notna()
        & stocks["circ_mv_cny"].gt(0)
    )
    base = stocks.loc[valid].copy()
    base["up_state_20d"] = base["stock_return_20d"].gt(0)
    base["cap_up"] = base["circ_mv_cny"] * base["up_state_20d"].astype(float)
    base["weighted_return"] = base["weight_lag1"] * base["stock_return_1d"]
    keys = ["trade_date", "industry_code"]
    groups = base.groupby(keys, sort=False, observed=True).agg(
        industry_name=("industry_name", "first"),
        member_count=("ts_code", "nunique"),
        industry_circ_mv=("circ_mv_cny", "sum"),
        cap_up=("cap_up", "sum"),
        breadth_equal_raw=("up_state_20d", "mean"),
        return_weight=("weight_lag1", "sum"),
        weighted_return=("weighted_return", "sum"),
        industry_amount=("amount_cny", "sum"),
    ).reset_index()
    groups["breadth_float_raw"] = groups["cap_up"] / groups["industry_circ_mv"]
    groups["industry_return_1d"] = groups["weighted_return"] / groups["return_weight"]
    market = base.groupby("trade_date", sort=False).agg(
        market_weight=("weight_lag1", "sum"),
        market_weighted_return=("weighted_return", "sum"),
        market_amount=("amount_cny", "sum"),
    ).reset_index()
    market["market_return_1d"] = market["market_weighted_return"] / market["market_weight"]
    groups = groups.merge(
        market[["trade_date", "market_return_1d", "market_amount"]],
        on="trade_date", how="left", validate="many_to_one",
    )
    groups["eligible_group"] = groups["member_count"].ge(minimum_members)
    groups["industry_amount_share"] = groups["industry_amount"] / groups["market_amount"]
    groups = groups.sort_values(["industry_code", "trade_date"]).reset_index(drop=True)
    industry_group = groups.groupby("industry_code", sort=False)
    groups["breadth_float"] = industry_group["breadth_float_raw"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    groups["breadth_equal"] = industry_group["breadth_equal_raw"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    groups["breadth_delta_5d"] = groups["breadth_float"] - industry_group["breadth_float"].shift(5)
    for window in (5, 20, 60):
        groups[f"industry_return_{window}d"] = industry_group["industry_return_1d"].transform(
            lambda values, w=window: _rolling_compound(values, w)
        )
    market = market.sort_values("trade_date")
    market["market_return_20d"] = _rolling_compound(market["market_return_1d"], 20)
    groups = groups.merge(
        market[["trade_date", "market_return_20d"]], on="trade_date", how="left",
        validate="many_to_one",
    )
    groups["rs_20d"] = (
        (1 + groups["industry_return_20d"]) / (1 + groups["market_return_20d"]) - 1
    )
    industry_group = groups.groupby("industry_code", sort=False)
    groups["rs_momentum_5d"] = groups["rs_20d"] - industry_group["rs_20d"].shift(5)
    excess_1d = groups["industry_return_1d"] - groups["market_return_1d"]
    short = excess_1d.groupby(groups["industry_code"], sort=False).transform(
        lambda values: values.ewm(span=3, adjust=False, min_periods=3).mean()
    )
    long = excess_1d.groupby(groups["industry_code"], sort=False).transform(
        lambda values: values.ewm(span=10, adjust=False, min_periods=10).mean()
    )
    groups["relative_strength_velocity"] = short - long
    groups["rrg_quadrant"] = np.select(
        [
            groups["rs_20d"].gt(0) & groups["rs_momentum_5d"].gt(0),
            groups["rs_20d"].le(0) & groups["rs_momentum_5d"].gt(0),
            groups["rs_20d"].gt(0) & groups["rs_momentum_5d"].le(0),
        ],
        ["leading", "improving", "weakening"], default="lagging",
    )
    groups["breadth_delta_rank"] = groups.groupby("trade_date")["breadth_delta_5d"].rank(pct=True)
    groups["breadth_level_rank"] = groups.groupby("trade_date")["breadth_float"].rank(pct=True)
    groups["rs_z"] = groups.groupby("trade_date")["rs_20d"].transform(_cs_zscore)
    groups["rs_momentum_z"] = groups.groupby("trade_date")["rs_momentum_5d"].transform(_cs_zscore)
    groups["breadth_delta_z"] = groups.groupby("trade_date")["breadth_delta_5d"].transform(_cs_zscore)
    groups["relative_velocity_z"] = groups.groupby("trade_date")["relative_strength_velocity"].transform(_cs_zscore)
    groups["signal_hot_1d"] = groups["industry_return_1d"]
    groups["signal_momentum_5d"] = groups["industry_return_5d"]
    groups["signal_momentum_20d"] = groups["industry_return_20d"]
    groups["signal_breadth_level"] = groups["breadth_float"]
    groups["signal_breadth_delta"] = groups["breadth_delta_5d"]
    groups["signal_rrg_only"] = groups["rs_z"] + groups["rs_momentum_z"]
    main_mask = (
        groups["eligible_group"]
        & groups["breadth_float"].gt(0.50)
        & groups["breadth_delta_rank"].ge(0.70)
        & groups["rs_momentum_5d"].gt(0)
    )
    groups["signal_breadth_rrg"] = groups["breadth_delta_z"].where(main_mask)
    groups["signal_existing_proxy"] = (
        0.5 * groups["breadth_delta_z"] + 0.5 * groups["relative_velocity_z"]
    )
    weakening = (
        groups["eligible_group"]
        & groups["breadth_float"].gt(0.50)
        & groups["breadth_delta_rank"].ge(0.70)
        & groups["rs_20d"].gt(0)
        & groups["rs_momentum_5d"].le(0)
    )
    groups["signal_weakening_placebo"] = groups["breadth_delta_z"].where(weakening)
    groups = _attach_frozen_membership_labels(stocks, groups, horizons)
    return stocks, groups.sort_values(["trade_date", "industry_code"]).reset_index(drop=True)


def _attach_frozen_membership_labels(
    stocks: pd.DataFrame, groups: pd.DataFrame, horizons: Iterable[int]
) -> pd.DataFrame:
    result = groups.copy()
    ordered = stocks.sort_values(["ts_code", "trade_date"]).copy()
    stock_group = ordered.groupby("ts_code", sort=False)["adj_open"]
    valid_base = (
        ordered["is_tradeable"].fillna(False).astype(bool)
        & ordered["industry_code"].notna()
        & ordered["circ_mv_cny"].gt(0)
    )
    for horizon in sorted(set(int(item) for item in horizons)):
        forward = stock_group.shift(-(horizon + 1)) / stock_group.shift(-1) - 1
        label = ordered.loc[valid_base, ["trade_date", "industry_code", "circ_mv_cny"]].copy()
        label["forward"] = forward.loc[valid_base]
        label = label.dropna(subset=["forward"])
        label["weighted_forward"] = label["circ_mv_cny"] * label["forward"]
        industry = label.groupby(["trade_date", "industry_code"], observed=True).agg(
            label_weight=("circ_mv_cny", "sum"), weighted_forward=("weighted_forward", "sum")
        ).reset_index()
        industry[f"forward_industry_{horizon}d"] = industry["weighted_forward"] / industry["label_weight"]
        market = label.groupby("trade_date").agg(
            market_label_weight=("circ_mv_cny", "sum"),
            market_weighted_forward=("weighted_forward", "sum"),
        )
        market[f"forward_market_{horizon}d"] = (
            market["market_weighted_forward"] / market["market_label_weight"]
        )
        industry = industry.merge(
            market[[f"forward_market_{horizon}d"]], left_on="trade_date", right_index=True,
            how="left", validate="many_to_one",
        )
        industry[f"forward_excess_{horizon}d"] = (
            industry[f"forward_industry_{horizon}d"] - industry[f"forward_market_{horizon}d"]
        )
        result = result.merge(
            industry[["trade_date", "industry_code", f"forward_industry_{horizon}d",
                      f"forward_market_{horizon}d", f"forward_excess_{horizon}d"]],
            on=["trade_date", "industry_code"], how="left", validate="one_to_one",
        )
    return result


def evaluate_rotation_signals(
    groups: pd.DataFrame,
    *,
    splits: dict[str, tuple[str, str]],
    horizons: Iterable[int] = PRIMARY_HORIZONS,
    top_n: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    daily_rows: list[pd.DataFrame] = []
    for split, (start, end) in splits.items():
        period = groups.loc[groups["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        for horizon in horizons:
            label = f"forward_excess_{int(horizon)}d"
            for signal, column in SIGNAL_COLUMNS.items():
                sample = period[["trade_date", "industry_code", column, label]].rename(
                    columns={column: "score", label: "forward_excess"}
                ).dropna()
                sizes = sample.groupby("trade_date").size()
                sample = sample[sample["trade_date"].isin(sizes[sizes >= 8].index)]
                ic = sample.groupby("trade_date").apply(
                    lambda day: day["score"].corr(day["forward_excess"], method="spearman"),
                    include_groups=False,
                ) if len(sample) else pd.Series(dtype=float)
                payoff_records = []
                for date, day in sample.groupby("trade_date", sort=True):
                    ordered = day.sort_values("score", ascending=False)
                    n = min(top_n, len(ordered))
                    if n == 0:
                        continue
                    top = ordered.head(n)["forward_excess"].mean()
                    bottom = ordered.tail(n)["forward_excess"].mean()
                    rest = ordered.iloc[n:]["forward_excess"].mean() if len(ordered) > n else np.nan
                    payoff_records.append({
                        "trade_date": date, "top_excess": top,
                        "top_minus_bottom": top - bottom,
                        "top_minus_rest": top - rest if np.isfinite(rest) else np.nan,
                        "candidate_groups": len(ordered),
                    })
                payoff = pd.DataFrame(payoff_records)
                ic_stats = newey_west_mean(ic, max(int(horizon) - 1, 0))
                top_stats = newey_west_mean(
                    payoff["top_excess"] if len(payoff) else pd.Series(dtype=float),
                    max(int(horizon) - 1, 0),
                )
                spread_stats = newey_west_mean(
                    payoff["top_minus_bottom"] if len(payoff) else pd.Series(dtype=float),
                    max(int(horizon) - 1, 0),
                )
                year_positive = None
                if len(payoff):
                    annual = payoff.assign(year=pd.to_datetime(payoff["trade_date"]).dt.year).groupby("year")["top_excess"].mean()
                    year_positive = float(annual.gt(0).mean())
                rows.append({
                    "split": split, "signal": signal, "horizon": int(horizon),
                    "observations": int(len(sample)), "days": int(sample["trade_date"].nunique()),
                    "mean_candidates": float(sizes.loc[sizes.index.intersection(sample["trade_date"].unique())].mean()) if len(sample) else None,
                    "rank_ic": float(ic.mean()) if len(ic.dropna()) else None,
                    "rank_ic_nw_t": ic_stats["t_value"],
                    "top_excess": top_stats["mean"], "top_excess_nw_t": top_stats["t_value"],
                    "top_minus_bottom": spread_stats["mean"],
                    "top_minus_bottom_nw_t": spread_stats["t_value"],
                    "positive_year_ratio": year_positive,
                })
                if len(payoff):
                    payoff = payoff.assign(split=split, signal=signal, horizon=int(horizon))
                    daily_rows.append(payoff)
    return pd.DataFrame(rows), pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()


def conditional_payoff_matrix(groups: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    frame = groups.dropna(subset=["breadth_delta_5d", f"forward_excess_{horizon}d"]).copy()
    frame["breadth_quintile"] = frame.groupby("trade_date")["breadth_delta_5d"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 5, labels=False, duplicates="drop") + 1
    )
    return frame.groupby(["rrg_quadrant", "breadth_quintile"], observed=True).agg(
        observations=(f"forward_excess_{horizon}d", "size"),
        mean_forward_excess=(f"forward_excess_{horizon}d", "mean"),
        positive_ratio=(f"forward_excess_{horizon}d", lambda values: values.gt(0).mean()),
    ).reset_index()


def evaluate_rotation_sensitivity(
    groups: pd.DataFrame,
    *,
    start: str = "2024-01-01",
    end: str = "2026-06-30",
    horizon: int = 5,
    top_n: int = 3,
    bootstrap_samples: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """One-at-a-time neighbours around the frozen breadth/RRG rule."""
    frame = groups.sort_values(["industry_code", "trade_date"]).copy()
    by_group = frame.groupby("industry_code", sort=False)
    variants: dict[str, pd.Series] = {
        "primary_smooth20_delta5_level50": frame["breadth_delta_5d"],
    }
    masks: dict[str, pd.Series] = {}
    base_rank = frame.groupby("trade_date")["breadth_delta_5d"].rank(pct=True)
    masks["primary_smooth20_delta5_level50"] = (
        frame["eligible_group"] & frame["breadth_float"].gt(0.50)
        & base_rank.ge(0.70) & frame["rs_momentum_5d"].gt(0)
    )
    for window in (5, 10):
        smooth = by_group["breadth_float_raw"].transform(
            lambda values, w=window: values.rolling(w, min_periods=w).mean()
        )
        delta = smooth - smooth.groupby(frame["industry_code"], sort=False).shift(5)
        name = f"smooth{window}_delta5_level50"
        variants[name] = delta
        rank = delta.groupby(frame["trade_date"]).rank(pct=True)
        masks[name] = frame["eligible_group"] & smooth.gt(0.50) & rank.ge(0.70) & frame["rs_momentum_5d"].gt(0)
    for delta_window in (3, 10):
        delta = frame["breadth_float"] - by_group["breadth_float"].shift(delta_window)
        name = f"smooth20_delta{delta_window}_level50"
        variants[name] = delta
        rank = delta.groupby(frame["trade_date"]).rank(pct=True)
        masks[name] = frame["eligible_group"] & frame["breadth_float"].gt(0.50) & rank.ge(0.70) & frame["rs_momentum_5d"].gt(0)
    equal_smooth = by_group["breadth_equal_raw"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    equal_delta = equal_smooth - equal_smooth.groupby(frame["industry_code"], sort=False).shift(5)
    variants["equal_weight_smooth20_delta5"] = equal_delta
    equal_rank = equal_delta.groupby(frame["trade_date"]).rank(pct=True)
    masks["equal_weight_smooth20_delta5"] = frame["eligible_group"] & equal_smooth.gt(0.50) & equal_rank.ge(0.70) & frame["rs_momentum_5d"].gt(0)
    raw_delta = frame["breadth_float_raw"] - by_group["breadth_float_raw"].shift(5)
    variants["raw_breadth_delta5"] = raw_delta
    raw_rank = raw_delta.groupby(frame["trade_date"]).rank(pct=True)
    masks["raw_breadth_delta5"] = frame["eligible_group"] & frame["breadth_float_raw"].gt(0.50) & raw_rank.ge(0.70) & frame["rs_momentum_5d"].gt(0)
    for level in (0.40, 0.60):
        name = f"smooth20_delta5_level{int(level * 100)}"
        variants[name] = frame["breadth_delta_5d"]
        masks[name] = frame["eligible_group"] & frame["breadth_float"].gt(level) & base_rank.ge(0.70) & frame["rs_momentum_5d"].gt(0)
    variants["no_breadth_level_gate"] = frame["breadth_delta_5d"]
    masks["no_breadth_level_gate"] = frame["eligible_group"] & base_rank.ge(0.70) & frame["rs_momentum_5d"].gt(0)
    for quadrant in ("leading", "improving"):
        name = f"primary_{quadrant}_only"
        variants[name] = frame["breadth_delta_5d"]
        masks[name] = masks["primary_smooth20_delta5_level50"] & frame["rrg_quadrant"].eq(quadrant)

    period = frame["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))
    label = f"forward_excess_{horizon}d"
    rows = []
    rng = np.random.default_rng(seed)
    for name, score in variants.items():
        sample = frame.loc[period & masks[name], ["trade_date", "industry_code", label]].copy()
        sample["score"] = score.loc[sample.index]
        sample = sample.dropna(subset=[label, "score"])
        ic = sample.groupby("trade_date").apply(
            lambda day: day["score"].corr(day[label], method="spearman") if len(day) >= 3 else np.nan,
            include_groups=False,
        )
        top = sample.sort_values(["trade_date", "score"], ascending=[True, False]).groupby("trade_date").head(top_n)
        daily = top.groupby("trade_date")[label].mean()
        stats = newey_west_mean(daily, horizon - 1)
        lower, upper = _block_bootstrap_mean(daily.to_numpy(float), 20, bootstrap_samples, rng)
        rows.append({
            "variant": name, "days": int(len(daily)),
            "mean_candidates": float(sample.groupby("trade_date").size().mean()) if len(sample) else None,
            "rank_ic": float(ic.mean()) if len(ic.dropna()) else None,
            "top3_forward_excess": stats["mean"], "nw_t": stats["t_value"],
            "bootstrap_95_low": lower, "bootstrap_95_high": upper,
        })
    return pd.DataFrame(rows)


def _block_bootstrap_mean(
    values: np.ndarray, block: int, samples: int, rng: np.random.Generator
) -> tuple[float | None, float | None]:
    clean = values[np.isfinite(values)]
    count = len(clean)
    if count < block or samples <= 0:
        return None, None
    starts = np.arange(max(count - block + 1, 1))
    draws = []
    blocks_needed = int(math.ceil(count / block))
    for _ in range(samples):
        selected = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([clean[start:start + block] for start in selected])[:count]
        draws.append(sample.mean())
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def fama_macbeth_breadth_increment(groups: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    predictors = [
        "breadth_delta_5d", "rs_20d", "rs_momentum_5d", "industry_return_1d",
        "industry_return_5d", "industry_return_20d", "industry_return_60d",
        "industry_amount_share", "member_count",
    ]
    label = f"forward_excess_{horizon}d"
    coefficients = []
    for date, day in groups[["trade_date", label, *predictors]].dropna().groupby("trade_date"):
        if len(day) < len(predictors) + 5:
            continue
        x = day[predictors].apply(_cs_zscore)
        usable = x.notna().all(axis=1) & day[label].notna()
        if usable.sum() < len(predictors) + 5:
            continue
        matrix = np.column_stack([np.ones(int(usable.sum())), x.loc[usable].to_numpy(float)])
        beta = np.linalg.lstsq(matrix, day.loc[usable, label].to_numpy(float), rcond=None)[0]
        coefficients.append({"trade_date": date, "intercept": beta[0], **dict(zip(predictors, beta[1:]))})
    daily = pd.DataFrame(coefficients)
    rows = []
    for column in ["intercept", *predictors]:
        stats = newey_west_mean(daily[column] if column in daily else pd.Series(dtype=float), horizon - 1)
        rows.append({"term": column, **stats, "days": int(len(daily))})
    return pd.DataFrame(rows)


def newey_west_mean(values: pd.Series, lags: int) -> dict[str, float | None]:
    clean = pd.Series(values).dropna().to_numpy(float)
    count = len(clean)
    if count < 3:
        return {"mean": float(clean.mean()) if count else None, "t_value": None, "p_value": None}
    lags = min(max(int(lags), 0), count - 1)
    centered = clean - clean.mean()
    long_run = float(centered @ centered / count)
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        long_run += 2 * weight * float(centered[lag:] @ centered[:-lag] / count)
    variance = max(long_run, 0.0) / count
    t_value = float(clean.mean() / math.sqrt(variance)) if variance > 0 else None
    return {
        "mean": float(clean.mean()), "t_value": t_value,
        "p_value": math.erfc(abs(t_value) / math.sqrt(2)) if t_value is not None else None,
    }


def _rolling_compound(values: pd.Series, window: int) -> pd.Series:
    safe = values.clip(lower=-0.999999)
    return np.expm1(np.log1p(safe).rolling(window, min_periods=window).sum())


def _cs_zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 0:
        return pd.Series(np.nan, index=values.index)
    return (values - values.mean()) / std
