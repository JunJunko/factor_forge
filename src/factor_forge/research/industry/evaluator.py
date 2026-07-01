from __future__ import annotations

import math

import numpy as np
import pandas as pd

from factor_forge.evaluation.l1 import _daily_correlation, _summary


def _newey_west_t(values: pd.Series, lag: int | None = None) -> float | None:
    x = values.dropna().to_numpy(float)
    n = len(x)
    if n < 3:
        return None
    lag = min(lag if lag is not None else int(4 * (n / 100) ** (2 / 9)), n - 1)
    centered = x - x.mean()
    variance = float(centered @ centered / n)
    for k in range(1, lag + 1):
        covariance = float(centered[k:] @ centered[:-k] / n)
        variance += 2 * (1 - k / (lag + 1)) * covariance
    se = math.sqrt(max(variance, 0) / n)
    return float(x.mean() / se) if se > 0 else None


class IndustrySliceEvaluator:
    """Evaluate stock factors with the same daily IC/statistics primitives as L1."""

    scope_columns = {
        "top2": "sw_l1_rotation_top2_flag", "top5": "sw_l1_rotation_top5_flag",
        "top10": "sw_l1_rotation_top10_flag", "bottom5": "sw_l1_rotation_bottom5_flag",
    }

    def evaluate_stock(self, stocks: pd.DataFrame, factor_values: pd.DataFrame,
                       targets: dict[str, dict[int, pd.Series]], scopes: list[str],
                       min_cross_section: int, factor_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        default_values = factor_values["all"] if isinstance(factor_values, dict) else factor_values
        merged = stocks.merge(default_values[["trade_date", "ts_code", "factor_value"]],
                              on=["trade_date", "ts_code"], how="left", validate="one_to_one")
        rows, yearly_rows = [], []
        for target_mode, by_horizon in targets.items():
            for horizon, target in by_horizon.items():
                merged["target"] = target.to_numpy()
                for scope in scopes:
                    if isinstance(factor_values, dict):
                        scoped = factor_values[scope][["trade_date", "ts_code", "factor_value"]].rename(columns={"factor_value": "scoped_factor"})
                        if "scoped_factor" in merged: merged = merged.drop(columns="scoped_factor")
                        merged = merged.merge(scoped, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
                        factor_column = "scoped_factor"
                    else:
                        factor_column = "factor_value"
                    mask = pd.Series(True, index=merged.index) if scope == "all" else merged[self.scope_columns[scope]].fillna(False)
                    sample = merged.loc[mask, ["trade_date", factor_column, "target"]].rename(
                        columns={factor_column: "factor", "target": "forward_return"}).dropna()
                    sizes = sample.groupby("trade_date").size()
                    sample = sample[sample["trade_date"].isin(sizes[sizes >= min_cross_section].index)]
                    rank_ic = _daily_correlation(sample, "spearman") if len(sample) else pd.Series(dtype=float)
                    pearson = _daily_correlation(sample, "pearson") if len(sample) else pd.Series(dtype=float)
                    rank, pear = _summary(rank_ic), _summary(pearson)
                    cutoff = rank_ic.index.sort_values()[int(len(rank_ic) * .8)] if len(rank_ic) >= 5 else None
                    oos = _summary(rank_ic[rank_ic.index >= cutoff]) if cutoff is not None else _summary(pd.Series(dtype=float))
                    year_means = rank_ic.groupby(pd.to_datetime(rank_ic.index).year).mean() if len(rank_ic) else pd.Series(dtype=float)
                    rows.append({"factor_id": factor_id, "scope": scope, "target_mode": target_mode,
                                 "horizon": horizon, "observations": len(sample), "days": sample["trade_date"].nunique(),
                                 "mean_rank_ic": rank["mean"], "pearson_ic": pear["mean"], "ic_std": rank["std"],
                                 "icir": rank["icir"], "positive_ratio": rank["positive_ratio"], "t_value": rank["t_value"],
                                 "newey_west_t": _newey_west_t(rank_ic), "oos_rank_ic": oos["mean"],
                                 "oos_icir": oos["icir"], "yearly_positive_ratio": float((year_means > 0).mean()) if len(year_means) else None})
                    yearly_rows.extend({"factor_id": factor_id, "scope": scope, "target_mode": target_mode,
                                        "horizon": horizon, "year": int(year), "mean_rank_ic": float(value)}
                                       for year, value in year_means.items())
        return pd.DataFrame(rows), pd.DataFrame(yearly_rows)

    def evaluate_selector(self, industries: pd.DataFrame,
                          future_targets: dict[int, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows, yearly = [], []
        ordered = industries.sort_values(["industry_code", "trade_date"]).copy()
        for horizon, future in future_targets.items():
            sample = ordered.assign(factor=ordered["neutral_score"], forward_return=future.reindex(ordered.index)).dropna(
                subset=["factor", "forward_return"])
            ic = _daily_correlation(sample, "spearman")
            summary = _summary(ic)
            tail = ordered.assign(forward_return=future.reindex(ordered.index))
            daily_tail = []
            for _, day in tail.groupby("trade_date"):
                valid = day.dropna(subset=["forward_return"])
                if valid.empty:
                    continue
                top5 = valid.loc[valid["top5_flag"], "forward_return"].mean()
                bottom5 = valid.loc[valid["bottom5_flag"], "forward_return"].mean()
                rest = valid.loc[~valid["top5_flag"], "forward_return"].mean()
                daily_tail.append({
                    "top2_mean_return": valid.loc[valid["top2_flag"], "forward_return"].mean(),
                    "top5_mean_return": top5,
                    "top10_mean_return": valid.loc[valid["top10_flag"], "forward_return"].mean(),
                    "top5_minus_rest": top5 - rest,
                    "top5_minus_bottom5": top5 - bottom5,
                })
            tail_mean = pd.DataFrame(daily_tail).mean().to_dict() if daily_tail else {}
            rows.append({"horizon": horizon, "rank_ic": summary["mean"], "icir": summary["icir"],
                         "positive_ratio": summary["positive_ratio"], "newey_west_t": _newey_west_t(ic),
                         **tail_mean})
            for year, value in ic.groupby(pd.to_datetime(ic.index).year).mean().items():
                yearly.append({"horizon": horizon, "year": int(year), "rank_ic": float(value)})
        return pd.DataFrame(rows), pd.DataFrame(yearly)
