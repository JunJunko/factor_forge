from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from factor_forge.research.concept_rotation_alpha import newey_west_mean
from factor_forge.research.concept_portfolio_backtest import attach_continuous_breadth_signals


SIGNALS = {
    "E1_etf_momentum": "score_etf_momentum",
    "E2_concept_rrg": "score_rrg",
    "E3_rrg_breadth": "score_rrg_breadth",
    "E4_literal_gate": "score_literal_gate",
    "E5_placebo_mapping": "score_placebo_mapping",
}


def prepare_etf_panel(
    daily: pd.DataFrame,
    share: pd.DataFrame,
    nav: pd.DataFrame,
    basic: pd.DataFrame,
    *,
    share_availability_lag_sessions: int = 0,
) -> pd.DataFrame:
    if share_availability_lag_sessions < 0:
        raise ValueError("share_availability_lag_sessions cannot be negative")
    frame = daily.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"].astype(str))
    numeric = ["open", "high", "low", "close", "pre_close", "pct_chg", "amount", "vol"]
    for column in numeric:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["amount_cny"] = frame["amount"] * 1000
    frame["volume_shares"] = frame["vol"] * 100
    shares = share.copy()
    shares["trade_date"] = pd.to_datetime(shares["trade_date"].astype(str))
    shares["fd_share"] = pd.to_numeric(shares["fd_share"], errors="coerce")
    if share_availability_lag_sessions:
        shares = _lag_observation_dates(
            shares,
            pd.DatetimeIndex(sorted(frame["trade_date"].unique())),
            sessions=share_availability_lag_sessions,
        )
    frame = _asof_by_code(frame, shares[["ts_code", "trade_date", "fd_share"]], "trade_date")
    navs = nav.copy()
    if not navs.empty and "nav_date" in navs:
        navs["trade_date"] = pd.to_datetime(navs["nav_date"].astype(str))
        for column in ("unit_nav", "adj_nav"):
            if column in navs:
                navs[column] = pd.to_numeric(navs[column], errors="coerce")
        nav_columns = [c for c in ["ts_code", "trade_date", "unit_nav", "adj_nav"] if c in navs]
        frame = _asof_by_code(frame, navs[nav_columns], "trade_date")
    frame["price_adjustment"] = 1.0
    if {"unit_nav", "adj_nav"}.issubset(frame):
        ratio = frame["adj_nav"] / frame["unit_nav"]
        frame["price_adjustment"] = ratio.where(ratio.gt(0), 1.0).fillna(1.0)
    # Fund NAV adjustment factors can switch on the ex-right date while the
    # exchange close is still quoted on the pre-split basis. Reconstructing the
    # factor from the exchange-reported daily return keeps that boundary clean.
    reported_return = pd.Series(index=frame.index, dtype=float)
    if "pct_chg" in frame:
        reported_return = frame["pct_chg"] / 100.0
    pre_close_return = frame["close"] / frame["pre_close"] - 1.0
    reported_return = reported_return.where(reported_return.notna(), pre_close_return)
    raw_return = frame.groupby("ts_code", sort=False)["close"].pct_change(fill_method=None)
    reported_return = reported_return.where(reported_return.notna(), raw_return)
    growth = (1.0 + reported_return).where(reported_return.gt(-1.0))
    first_row = frame.groupby("ts_code", sort=False).cumcount().eq(0)
    growth = growth.mask(first_row, 1.0)
    synthetic_close = growth.groupby(frame["ts_code"], sort=False).cumprod()
    first_close = frame.groupby("ts_code", sort=False)["close"].transform("first")
    exchange_adjustment = synthetic_close * first_close / frame["close"]
    frame["price_adjustment"] = exchange_adjustment.where(
        exchange_adjustment.gt(0), frame["price_adjustment"]
    )
    frame["adj_open"] = frame["open"] * frame["price_adjustment"]
    frame["adj_close"] = frame["close"] * frame["price_adjustment"]
    frame["aum_cny"] = frame["close"] * frame["fd_share"] * 10000
    keep = [c for c in ["ts_code", "name", "list_date", "benchmark", "m_fee", "c_fee"] if c in basic]
    frame = frame.merge(basic[keep].drop_duplicates("ts_code"), on="ts_code", how="left")
    frame = frame.sort_values(["ts_code", "trade_date"])
    grouped = frame.groupby("ts_code", sort=False)
    frame["etf_return_1d"] = grouped["adj_close"].pct_change(fill_method=None)
    frame["etf_momentum_20d"] = grouped["adj_close"].pct_change(20, fill_method=None)
    frame["etf_momentum_60d"] = grouped["adj_close"].pct_change(60, fill_method=None)
    for horizon in (5, 10, 20):
        entry = grouped["adj_open"].shift(-1)
        exit_price = grouped["adj_open"].shift(-(horizon + 1))
        frame[f"forward_open_{horizon}d"] = exit_price / entry - 1
    return frame.reset_index(drop=True)


def _asof_by_code(left: pd.DataFrame, right: pd.DataFrame, date_column: str) -> pd.DataFrame:
    chunks = []
    for code, group in left.groupby("ts_code", sort=False):
        lookup = right.loc[right["ts_code"].eq(code)].drop(columns="ts_code").sort_values(date_column)
        ordered = group.sort_values(date_column)
        if lookup.empty:
            for column in right.columns.difference(["ts_code", date_column]):
                ordered[column] = np.nan
            chunks.append(ordered)
        else:
            chunks.append(pd.merge_asof(ordered, lookup, on=date_column, direction="backward"))
    return pd.concat(chunks, ignore_index=True)


def _lag_observation_dates(
    observations: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    sessions: int,
) -> pd.DataFrame:
    """Move dated observations to their conservative first-usable session."""
    if sessions < 1 or observations.empty:
        return observations.copy()
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(calendar).unique()))
    positions = dates.searchsorted(pd.to_datetime(observations["trade_date"]), side="right")
    positions = positions + sessions - 1
    valid = positions < len(dates)
    result = observations.loc[valid].copy()
    result["trade_date"] = dates[positions[valid]]
    return result


def build_etf_signal_panel(
    concept_features: pd.DataFrame,
    etfs: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    selection_cutoff: str,
    minimum_mapping_correlation: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = mapping.loc[mapping["selected"].fillna(False)].copy()
    selected = selected.rename(columns={"etf_code": "ts_code"})
    panel = etfs.merge(selected, on="ts_code", how="inner", suffixes=("", "_mapping"))
    wanted = [
        "trade_date", "concept_code", "concept_return_1d", "eligible_concept", "breadth_float",
        "common_delta_rank", "rs_momentum_5d", "rrg_quadrant", "signal_rrg_only",
        "common_breadth_delta_smooth5",
    ]
    features = attach_continuous_breadth_signals(concept_features)
    wanted += ["signal_common_breadth_residual", "signal_rrg_plus_common_breadth_residual"]
    panel = panel.merge(features[wanted], on=["trade_date", "concept_code"], how="left", validate="many_to_one")
    cutoff = pd.Timestamp(selection_cutoff)
    correlations = panel.loc[panel["trade_date"].le(cutoff)].groupby(
        ["concept_code", "ts_code"], observed=True
    ).apply(lambda x: x["concept_return_1d"].corr(x["etf_return_1d"]), include_groups=False).rename("mapping_correlation").reset_index()
    observations = panel.loc[panel["trade_date"].le(cutoff)].groupby(
        ["concept_code", "ts_code"], observed=True
    )[["concept_return_1d", "etf_return_1d"]].count().min(axis=1).rename("correlation_observations").reset_index()
    mapping_audit = selected.merge(correlations, on=["concept_code", "ts_code"], how="left").merge(
        observations, on=["concept_code", "ts_code"], how="left"
    )
    mapping_audit["mapping_pass"] = (
        mapping_audit["mapping_correlation"].ge(minimum_mapping_correlation)
        & mapping_audit["correlation_observations"].ge(60)
    )
    panel = panel.merge(
        mapping_audit[["concept_code", "ts_code", "mapping_correlation", "mapping_pass"]],
        on=["concept_code", "ts_code"], how="left", validate="many_to_one",
    )
    panel["score_etf_momentum"] = (
        0.6 * panel.groupby("trade_date")["etf_momentum_20d"].transform(_zscore)
        + 0.4 * panel.groupby("trade_date")["etf_momentum_60d"].transform(_zscore)
    )
    panel["score_rrg"] = panel["signal_rrg_only"]
    panel["score_rrg_breadth"] = panel["signal_rrg_plus_common_breadth_residual"]
    eligible_flag = panel["eligible_concept"].astype("boolean").fillna(False).astype(bool)
    literal = (
        eligible_flag & panel["breadth_float"].gt(0.50)
        & panel["common_delta_rank"].ge(0.70)
        & panel["rrg_quadrant"].isin(["leading", "improving"])
    )
    panel["score_literal_gate"] = panel["score_rrg_breadth"].where(literal)
    codes = sorted(panel["ts_code"].unique())
    placebo = dict(zip(codes, codes[1:] + codes[:1]))
    lookup = panel[["trade_date", "ts_code", "score_rrg_breadth"]].rename(
        columns={"ts_code": "placebo_source", "score_rrg_breadth": "score_placebo_mapping"}
    )
    panel["placebo_source"] = panel["ts_code"].map(placebo)
    panel = panel.merge(lookup, on=["trade_date", "placebo_source"], how="left", validate="many_to_one")
    return panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True), mapping_audit


def evaluate_etf_signals(
    panel: pd.DataFrame,
    *,
    splits: dict[str, tuple[str, str]],
    costs_bps: Iterable[float] = (10, 20, 40),
    top_n: int = 3,
    horizon: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping_flag = panel["mapping_pass"].astype("boolean").fillna(False).astype(bool)
    concept_flag = panel["eligible_concept"].astype("boolean").fillna(False).astype(bool)
    eligible = panel.loc[mapping_flag & concept_flag].copy()
    label = f"forward_open_{horizon}d"
    period_rows, summaries = [], []
    for split, (start, end) in splits.items():
        split_frame = eligible.loc[eligible["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        dates = sorted(split_frame.loc[split_frame[label].notna(), "trade_date"].unique())
        for offset in range(horizon):
            rebalance_dates = set(dates[offset::horizon])
            for date, day in split_frame.loc[split_frame["trade_date"].isin(rebalance_dates)].groupby("trade_date"):
                day = day.dropna(subset=[label])
                if day.empty:
                    continue
                benchmark = float(day[label].mean())
                period_rows.append(_period_row(split, offset, date, "E0_equal_weight", day, day, benchmark, label, costs_bps))
                for signal, column in SIGNALS.items():
                    candidates = day.dropna(subset=[column]).sort_values(column, ascending=False)
                    chosen = _select_clusters(candidates, top_n)
                    if chosen.empty:
                        continue
                    period_rows.append(_period_row(split, offset, date, signal, day, chosen, benchmark, label, costs_bps))
    periods = pd.DataFrame(period_rows)
    if periods.empty:
        return pd.DataFrame(), periods, pd.DataFrame()
    for keys, group in periods.groupby(["split", "signal"], observed=True):
        split, signal = keys
        group = group.sort_values("trade_date")
        row = {"split": split, "signal": signal, "periods": len(group), "mean_holdings": group["holdings"].mean(), "mean_exposure": group["exposure"].mean()}
        row.update({"gross_return": group["gross_return"].mean(), "gross_excess": group["gross_excess"].mean()})
        row["gross_excess_nw_t"] = newey_west_mean(group["gross_excess"], horizon - 1)["t_value"]
        for cost in costs_bps:
            column = f"net_excess_{int(cost)}bps"
            row[column] = group[column].mean()
            row[f"{column}_nw_t"] = newey_west_mean(group[column], horizon - 1)["t_value"]
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    paired = paired_incremental(periods, "E3_rrg_breadth", "E2_concept_rrg", costs_bps=20)
    return summary, periods, paired


def _period_row(split, offset, date, signal, universe, chosen, benchmark, label, costs_bps):
    exposure = 1.0 if signal == "E0_equal_weight" else min(len(chosen) / 3, 1.0)
    gross = float(chosen[label].mean()) * exposure
    result = {
        "split": split, "offset": offset, "trade_date": date, "signal": signal,
        "holdings": len(chosen), "universe": len(universe), "exposure": exposure,
        "gross_return": gross, "benchmark_return": benchmark, "gross_excess": gross - benchmark,
        "selected_etfs": ",".join(chosen["ts_code"].astype(str)),
    }
    for cost in costs_bps:
        # ``cost`` is the assumed full round-trip drag for one holding period.
        result[f"net_excess_{int(cost)}bps"] = gross - benchmark - (float(cost) / 10000)
    return result


def _select_clusters(candidates: pd.DataFrame, top_n: int) -> pd.DataFrame:
    indices, clusters = [], set()
    for index, row in candidates.iterrows():
        if row["cluster"] in clusters:
            continue
        indices.append(index)
        clusters.add(row["cluster"])
        if len(indices) == top_n:
            break
    return candidates.loc[indices]


def paired_incremental(periods: pd.DataFrame, left: str, right: str, *, costs_bps: int) -> pd.DataFrame:
    column = f"net_excess_{costs_bps}bps"
    pivot = periods.pivot_table(index=["split", "offset", "trade_date"], columns="signal", values=column)
    if left not in pivot or right not in pivot:
        return pd.DataFrame()
    difference = (pivot[left] - pivot[right]).dropna().rename("incremental").reset_index()
    rows = []
    for split, group in difference.groupby("split", observed=True):
        group = group.sort_values("trade_date")
        stats = newey_west_mean(group["incremental"], 4)
        rows.append({"split": split, "left": left, "right": right, "periods": len(group), "incremental_net_excess": stats["mean"], "nw_t": stats["t_value"], "positive_rate": group["incremental"].gt(0).mean()})
    return pd.DataFrame(rows)


def evaluate_signal_ic(panel: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    mapping_flag = panel["mapping_pass"].astype("boolean").fillna(False).astype(bool)
    concept_flag = panel["eligible_concept"].astype("boolean").fillna(False).astype(bool)
    sample = panel.loc[
        mapping_flag & concept_flag
        & panel["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))
    ]
    rows = []
    for signal, column in SIGNALS.items():
        for horizon in (5, 10, 20):
            label = f"forward_open_{horizon}d"
            values = sample.dropna(subset=[column, label])
            daily = values.groupby("trade_date").apply(
                lambda day: day[column].corr(day[label], method="spearman") if len(day) >= 6 else np.nan,
                include_groups=False,
            ).dropna()
            stats = newey_west_mean(daily.sort_index(), horizon - 1)
            rows.append({
                "signal": signal, "horizon": horizon, "days": len(daily),
                "mean_rank_ic": stats["mean"], "rank_ic_nw_t": stats["t_value"],
            })
    return pd.DataFrame(rows)


def _zscore(values: pd.Series) -> pd.Series:
    standard_deviation = values.std(ddof=0)
    return (values - values.mean()) / standard_deviation if standard_deviation and np.isfinite(standard_deviation) else pd.Series(0.0, index=values.index)
