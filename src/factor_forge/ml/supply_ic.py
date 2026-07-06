"""Factor-level IC diagnostics for the supply-contraction family (no ML, no backtest).

This is the document's 阶段一 / phase-1 check: before committing to LightGBM, look at
whether the raw structures (volume_residual, scarcity, the composites) have any
cross-sectional predictive power against the forward industry-neutral label.  It builds
the same dataset as the trainer (so the label and sample filter are identical) but skips
Qlib entirely and just computes daily IC summaries + a 2-D quantile sort.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .supply_config import SupplyFeatureConfig, SupplyLabelConfig
from .supply_dataset import build_supply_dataset
from .supply_features import _industry_loo_mean


def compute_factor_ic(
    dataset: pd.DataFrame,
    factor_names: list[str],
    eval_start: str,
    eval_end: str,
    min_daily_n: int = 20,
    label_col: str = "label",
) -> pd.DataFrame:
    """Daily Rank/Pearson IC summary for each factor over ``[eval_start, eval_end]``.

    Returns a DataFrame indexed by factor with columns: rank_ic_mean, rank_ic_ir,
    rank_ic_positive_ratio, rank_ic_newey_t, pearson_ic_mean, pearson_ic_ir, n_days.
    """
    mask = dataset["datetime"].between(pd.Timestamp(eval_start), pd.Timestamp(eval_end))
    sub = dataset.loc[mask]
    n_lag = 5  # Newey-West lag for the t-stat (round-trip IC autocorrelation)
    records: list[dict] = []
    for f in factor_names:
        if f not in sub.columns:
            records.append({"factor": f})
            continue
        pair = sub[["datetime", f, label_col]].dropna()
        if pair.empty:
            records.append({"factor": f})
            continue
        grp = pair.groupby("datetime")
        rank_ic = grp.apply(
            lambda g: g[f].corr(g[label_col], method="spearman") if len(g) >= min_daily_n else np.nan,
            include_groups=False,
        ).dropna()
        pear_ic = grp.apply(
            lambda g: g[f].corr(g[label_col]) if len(g) >= min_daily_n else np.nan,
            include_groups=False,
        ).dropna()
        records.append({
            "factor": f,
            "rank_ic_mean": float(rank_ic.mean()),
            "rank_ic_ir": float(rank_ic.mean() / rank_ic.std() * np.sqrt(252)) if rank_ic.std() > 0 else np.nan,
            "rank_ic_positive_ratio": float((rank_ic > 0).mean()),
            "rank_ic_newey_t": _newey_west_t(rank_ic.to_numpy(), n_lag),
            "pearson_ic_mean": float(pear_ic.mean()),
            "pearson_ic_ir": float(pear_ic.mean() / pear_ic.std() * np.sqrt(252)) if pear_ic.std() > 0 else np.nan,
            "n_days": int(len(rank_ic)),
        })
    return pd.DataFrame(records).set_index("factor")


def neutralized_residual(
    dataset: pd.DataFrame,
    factor: str,
    controls: list[str],
    eval_start: str,
    eval_end: str,
) -> pd.Series:
    """Daily cross-sectional OLS residual of ``factor`` on ``controls`` (+ intercept).

    Returned Series is aligned to ``dataset.index``; rows outside the window or with any
    missing value are NaN.  This is the standard "does the IC survive control X" test.
    """
    mask = dataset["datetime"].between(pd.Timestamp(eval_start), pd.Timestamp(eval_end))
    cols = ["datetime", factor, *controls]
    sub = dataset.loc[mask, cols].dropna()
    out = pd.Series(np.nan, index=dataset.index, dtype=float)
    k = len(controls) + 1
    for _dt, g in sub.groupby("datetime"):
        if len(g) < k + 1:
            continue
        x = np.column_stack([np.ones(len(g)), g[controls].to_numpy(dtype=float)])
        y = g[factor].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        out.loc[g.index] = y - x @ beta
    return out


def ic_of_series(
    dataset: pd.DataFrame, factor_values: pd.Series, eval_start: str, eval_end: str, name: str, min_daily_n: int = 20
) -> dict:
    """IC summary for an arbitrary per-row Series (e.g. a neutralized residual)."""
    tmp = dataset[["datetime", "label"]].copy()
    tmp[name] = factor_values.reindex(tmp.index).to_numpy()
    ic = compute_factor_ic(tmp, [name], eval_start, eval_end, min_daily_n=min_daily_n)
    row = ic.iloc[0].to_dict()
    row["factor"] = name
    return row


def _newey_west_t(series: np.ndarray, lag: int) -> float:
    """Newey-West HAC t-stat for the mean of a daily-IC series."""
    series = series[np.isfinite(series)]
    n = len(series)
    if n < lag + 2:
        return float("nan")
    mean = series.mean()
    gamma0 = np.mean((series - mean) ** 2)
    var = gamma0
    for k in range(1, lag + 1):
        gamma_k = np.mean((series[k:] - mean) * (series[:-k] - mean))
        var += 2 * gamma_k
    var = max(var, 1e-12)
    return float(mean / np.sqrt(var / n))


def quantile_2d_sort(
    dataset: pd.DataFrame,
    row_factor: str,
    col_factor: str,
    eval_start: str,
    eval_end: str,
    n_bins: int = 5,
    min_daily_n: int = 20,
) -> pd.DataFrame:
    """Daily quintile buckets of ``row_factor`` crossed with ``col_factor`` -> mean label.

    Document 阶段一: e.g. row=excess_ret_5 (price strength), col=volume_residual (squeeze).
    Each day both factors are independently ranked into quintiles; the cell value is the
    time-series mean of the daily cross-sectional mean forward return within that cell.
    """
    mask = dataset["datetime"].between(pd.Timestamp(eval_start), pd.Timestamp(eval_end))
    sub = dataset.loc[mask, ["datetime", row_factor, col_factor, "label"]].dropna()
    if sub.empty:
        return pd.DataFrame()
    # Daily independent quintile buckets.
    sub["_row_q"] = sub.groupby("datetime")[row_factor].transform(
        lambda s: pd.qcut(s, n_bins, labels=False, duplicates="drop") if len(s) >= min_daily_n else np.nan
    )
    sub["_col_q"] = sub.groupby("datetime")[col_factor].transform(
        lambda s: pd.qcut(s, n_bins, labels=False, duplicates="drop") if len(s) >= min_daily_n else np.nan
    )
    sub = sub.dropna(subset=["_row_q", "_col_q"])
    daily_cell = sub.groupby(["datetime", "_row_q", "_col_q"])["label"].mean().reset_index()
    table = daily_cell.groupby(["_row_q", "_col_q"])["label"].mean().unstack("_col_q")
    table.index.name = f"{row_factor}_quintile"
    table.columns.name = f"{col_factor}_quintile"
    return table


def multi_horizon_labels(
    panel: pd.DataFrame, horizons: list[int], method: str = "open_to_open"
) -> dict[int, pd.Series]:
    """Industry-neutral (leave-one-out) forward returns at each horizon.

    Returns ``{horizon: Series}`` indexed by ``(datetime, instrument)`` (instrument ==
    ts_code), matching the IC dataset's keys.  Open-to-open mirrors the trainer label.
    """
    df = panel.sort_values(["ts_code", "trade_date"]).copy()
    df["datetime"] = pd.to_datetime(df["trade_date"])
    df["instrument"] = df["ts_code"].to_numpy()
    df["industry"] = df.get("industry_l1_code").to_numpy()
    g = df.groupby("ts_code", sort=False)
    out: dict[int, pd.Series] = {}
    for h in horizons:
        if method == "open_to_open":
            fwd = g["adj_open"].shift(-(h + 1)) / g["adj_open"].shift(-1) - 1.0
        else:  # open_to_close
            fwd = g["adj_close"].shift(-h) / g["adj_open"].shift(-1) - 1.0
        ind = _industry_loo_mean(fwd, df["datetime"], df["industry"])
        labelled = pd.DataFrame({"datetime": df["datetime"].to_numpy(),
                                 "instrument": df["instrument"].to_numpy(),
                                 "v": (fwd.to_numpy() - ind.to_numpy())})
        out[h] = labelled.dropna().set_index(["datetime", "instrument"])["v"]
    return out


def yearly_ic(
    dataset: pd.DataFrame, factor: str, label_col: str, eval_start: str, eval_end: str, min_daily_n: int = 20
) -> pd.DataFrame:
    """Per-year daily RankIC for one factor (yearly stability check)."""
    mask = dataset["datetime"].between(pd.Timestamp(eval_start), pd.Timestamp(eval_end))
    pair = dataset.loc[mask, ["datetime", factor, label_col]].dropna()
    pair["year"] = pair["datetime"].dt.year
    rows = []
    for year, g in pair.groupby("year"):
        daily = g.groupby("datetime").apply(
            lambda x: x[factor].corr(x[label_col], method="spearman") if len(x) >= min_daily_n else np.nan,
            include_groups=False,
        ).dropna()
        rows.append({
            "year": int(year),
            "rank_ic_mean": float(daily.mean()),
            "rank_ic_ir": float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else np.nan,
            "positive_ratio": float((daily > 0).mean()),
            "n_days": int(len(daily)),
        })
    return pd.DataFrame(rows).set_index("year")


def quantile_decomp(
    dataset: pd.DataFrame, factor: str, label_col: str, eval_start: str, eval_end: str,
    n_bins: int = 10, min_daily_n: int = 20,
) -> pd.DataFrame:
    """Daily ``n_bins`` quantile buckets -> time-series mean forward return per quantile.

    Also reports the long-short spread (Q-top − Q-bottom), the top/bottom legs separately,
    and a monotonicity score (Spearman of quantile index vs mean return).
    """
    mask = dataset["datetime"].between(pd.Timestamp(eval_start), pd.Timestamp(eval_end))
    sub = dataset.loc[mask, ["datetime", factor, label_col]].dropna()
    sub["_q"] = sub.groupby("datetime")[factor].transform(
        lambda s: pd.qcut(s, n_bins, labels=False, duplicates="drop") if len(s) >= min_daily_n else np.nan
    )
    sub = sub.dropna(subset=["_q"])
    daily_cell = sub.groupby(["datetime", "_q"])[label_col].mean().reset_index()
    means = daily_cell.groupby("_q")[label_col].mean()
    means.index = [f"Q{int(i)+1}" for i in means.index]
    out = means.to_frame("mean_fwd_return")
    return out


def topn_returns(
    dataset: pd.DataFrame, factor: str, label_col: str, eval_start: str, eval_end: str,
    top_ns: list[int], n_per_year: int = 252,
) -> pd.DataFrame:
    """Daily Top-N mean forward return vs universe mean, plus annualized net-of-cost.

    Cost is a round-trip ``cost_bps`` applied per rebalance; with ``n_per_year`` rebalances
    a year the per-rebalance forward return is scaled to annualized gross and a per-period
    cost is subtracted.
    """
    mask = dataset["datetime"].between(pd.Timestamp(eval_start), pd.Timestamp(eval_end))
    sub = dataset.loc[mask, ["datetime", factor, label_col]].dropna()
    rows = []
    for top_n in top_ns:
        daily_top = sub.sort_values(["datetime", factor], ascending=[True, False]).groupby("datetime").head(top_n)
        daily_universe = sub.groupby("datetime")[label_col].mean()
        daily_top_mean = daily_top.groupby("datetime")[label_col].mean()
        excess = (daily_top_mean - daily_universe).dropna()
        rows.append({
            "top_n": top_n,
            "avg_daily_fwd": float(daily_top.mean().get(label_col, np.nan)),
            "avg_universe_fwd": float(daily_universe.mean()),
            "avg_daily_excess": float(excess.mean()),
            "ann_gross_excess": float((1 + excess.mean()) ** n_per_year - 1) if excess.mean() > -1 else np.nan,
            "n_days": int(len(excess)),
        })
    return pd.DataFrame(rows).set_index("top_n")


def factor_rank_correlation(
    dataset: pd.DataFrame, factor: str, others: list[str], eval_start: str, eval_end: str,
) -> pd.Series:
    """Time-series-averaged daily cross-sectional Spearman correlation of ``factor`` with each other."""
    mask = dataset["datetime"].between(pd.Timestamp(eval_start), pd.Timestamp(eval_end))
    sub = dataset.loc[mask, ["datetime", factor, *others]]
    out = {}
    for o in others:
        pair = sub[[factor, o]].dropna()
        if pair.empty:
            out[o] = np.nan
            continue
        daily = pair.groupby(sub["datetime"]).apply(
            lambda g: g[factor].corr(g[o], method="spearman") if len(g) >= 20 else np.nan,
            include_groups=False,
        ).dropna()
        out[o] = float(daily.mean()) if len(daily) else np.nan
    return pd.Series(out, name=f"corr_with_{factor}").sort_values(key=lambda s: s.abs(), ascending=False)


def monthly_ic(
    dataset: pd.DataFrame, factor_names: list[str], label_col: str,
    eval_start: str, eval_end: str, min_daily_n: int = 20,
) -> pd.DataFrame:
    """Mean-of-daily RankIC per calendar month, one column per factor."""
    mask = dataset["datetime"].between(pd.Timestamp(eval_start), pd.Timestamp(eval_end))
    sub = dataset.loc[mask]
    cols = {}
    for f in factor_names:
        pair = sub[["datetime", f, label_col]].dropna()
        daily = pair.groupby("datetime").apply(
            lambda g, f=f: g[f].corr(g[label_col], method="spearman") if len(g) >= min_daily_n else np.nan,
            include_groups=False,
        ).dropna()
        daily.index = pd.to_datetime(daily.index)
        cols[f] = daily.resample("MS").mean()
    out = pd.DataFrame(cols)
    out.index.name = "month"
    return out


def rolling_ic(
    dataset: pd.DataFrame, factor: str, label_col: str,
    eval_start: str, eval_end: str, window: int = 60, min_daily_n: int = 20,
) -> pd.Series:
    """Rolling ``window``-day mean of daily RankIC for one factor."""
    mask = dataset["datetime"].between(pd.Timestamp(eval_start), pd.Timestamp(eval_end))
    pair = dataset.loc[mask, ["datetime", factor, label_col]].dropna()
    daily = pair.groupby("datetime").apply(
        lambda g: g[factor].corr(g[label_col], method="spearman") if len(g) >= min_daily_n else np.nan,
        include_groups=False,
    ).dropna()
    daily.index = pd.to_datetime(daily.index)
    return daily.rolling(window, min_periods=window // 2).mean()


def load_dataset_for_ic(
    panel: pd.DataFrame,
    features: SupplyFeatureConfig | None = None,
    label: SupplyLabelConfig | None = None,
) -> pd.DataFrame:
    """Build the dataset RAW (no cross-sectional z-score) for IC inspection.

    Rank IC is invariant to monotonic transforms, but raw values keep Pearson IC and the
    quantile sort interpretable.  Sample weights are irrelevant here.
    """
    features = features or SupplyFeatureConfig(cross_sectional_zscore=False, use_sample_weight=False)
    if features.cross_sectional_zscore:
        features = features.model_copy(update={"cross_sectional_zscore": False, "use_sample_weight": False})
    label = label or SupplyLabelConfig()
    ds, _ = build_supply_dataset(panel, None, features, label, sample_weight_train=None)
    return ds
