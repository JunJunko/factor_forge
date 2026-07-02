from __future__ import annotations

import math

import numpy as np
import pandas as pd

from factor_forge.config import FactorSpec, L1Config
from .neutralization import build_variants


def _forward_open_return(panel: pd.DataFrame, horizon: int) -> pd.Series:
    ordered = panel[["ts_code", "trade_date", "adj_open"]].sort_values(["ts_code", "trade_date"])
    grouped = ordered.groupby("ts_code", sort=False)["adj_open"]
    entry = grouped.shift(-1)
    exit_price = grouped.shift(-(horizon + 1))
    result = exit_price / entry - 1.0
    return result.reindex(panel.index)


def _daily_correlation(group: pd.DataFrame, method: str) -> pd.Series:
    return group.groupby("trade_date")[["factor", "forward_return"]].apply(
        lambda x: x["factor"].corr(x["forward_return"], method=method)
        if len(x.dropna(subset=["factor", "forward_return"])) >= 3 else np.nan
    )


def _summary(ic: pd.Series) -> dict:
    values = ic.dropna()
    if len(values) < 2:
        return {"mean": None, "std": None, "icir": None, "positive_ratio": None, "t_value": None, "p_value": None}
    mean, std = float(values.mean()), float(values.std(ddof=1))
    t_value = mean / (std / math.sqrt(len(values))) if std > 0 else None
    p_value = math.erfc(abs(t_value) / math.sqrt(2)) if t_value is not None else None
    return {
        "mean": mean, "std": std, "icir": mean / std * math.sqrt(252) if std else None,
        "positive_ratio": float((values > 0).mean()), "t_value": t_value, "p_value": p_value,
    }


def _quantile_metrics(frame: pd.DataFrame, groups: int) -> dict:
    records = []
    for date, daily in frame.groupby("trade_date"):
        daily = daily.dropna(subset=["factor", "forward_return"])
        q = min(groups, 5 if len(daily) < groups * 10 else groups)
        if len(daily) < q * 2:
            continue
        try:
            bucket = pd.qcut(daily["factor"].rank(method="first"), q, labels=False) + 1
        except ValueError:
            continue
        values = daily.assign(quantile=bucket).groupby("quantile")["forward_return"].mean()
        records.append(pd.DataFrame({"trade_date": date, "quantile": values.index, "return": values.values}))
    if not records:
        return {"top_bottom_mean": None, "monotonicity": None, "quantile_returns": {}}
    all_returns = pd.concat(records, ignore_index=True)
    mean_by_q = all_returns.groupby("quantile")["return"].mean()
    top_bottom = float(mean_by_q.iloc[-1] - mean_by_q.iloc[0])
    monotonicity = float(pd.Series(mean_by_q.index).corr(pd.Series(mean_by_q.values), method="spearman"))
    return {
        "top_bottom_mean": top_bottom, "monotonicity": monotonicity,
        "quantile_returns": {str(int(k)): float(v) for k, v in mean_by_q.items()},
    }


def _fdr_bh(rows: list[dict]) -> None:
    valid = [(i, row["rank_ic"]["p_value"]) for i, row in enumerate(rows) if row["rank_ic"]["p_value"] is not None]
    valid.sort(key=lambda item: item[1])
    count = len(valid)
    adjusted = [0.0] * count
    running = 1.0
    for position in range(count - 1, -1, -1):
        _, p_value = valid[position]
        running = min(running, p_value * count / (position + 1))
        adjusted[position] = running
    for (row_index, _), q_value in zip(valid, adjusted):
        rows[row_index]["fdr_q"] = float(q_value)


def evaluate_predictive_power(
    panel: pd.DataFrame, factor_values: pd.DataFrame, spec: FactorSpec, config: L1Config
) -> dict:
    candidates = ["trade_date", "ts_code", "adj_open", "log_total_mv", "industry_l1_code"]
    candidates += [f"is_{universe}" for universe in config.universes]
    seen = set()
    needed = [c for c in candidates if c in panel.columns and not (c in seen or seen.add(c))]
    merged = panel[needed].merge(
        factor_values[["trade_date", "ts_code", "factor_value"]],
        on=["trade_date", "ts_code"], how="left",
    ).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    variants = build_variants(merged, spec.scope.cross_section)
    if spec.factor.direction == "unknown":
        variants.update({f"{name}_negative": -value for name, value in list(variants.items())})
    rows: list[dict] = []
    for horizon in config.forward_horizons:
        merged[f"forward_{horizon}"] = _forward_open_return(merged, horizon)
        for universe in config.universes:
            universe_mask = merged[f"is_{universe}"].fillna(False).astype(bool)
            for variant_name, values in variants.items():
                sample = merged.loc[universe_mask, ["trade_date", f"forward_{horizon}"]].copy()
                sample["factor"] = values.loc[universe_mask]
                sample = sample.rename(columns={f"forward_{horizon}": "forward_return"})
                sample = sample.dropna()
                daily_size = sample.groupby("trade_date").size()
                sample = sample[sample["trade_date"].isin(daily_size[daily_size >= config.min_cross_section].index)]
                rank_ic = _daily_correlation(sample, "spearman") if len(sample) else pd.Series(dtype=float)
                pearson_ic = _daily_correlation(sample, "pearson") if len(sample) else pd.Series(dtype=float)
                cutoff = rank_ic.index.sort_values()[int(len(rank_ic) * 0.8)] if len(rank_ic) >= 5 else None
                oos = rank_ic[rank_ic.index >= cutoff] if cutoff is not None else pd.Series(dtype=float)
                row = {
                    "variant": variant_name, "universe": universe, "horizon": horizon,
                    "observations": int(len(sample)), "days": int(sample["trade_date"].nunique()),
                    "rank_ic": _summary(rank_ic), "pearson_ic": _summary(pearson_ic),
                    "oos_rank_ic": _summary(oos), **_quantile_metrics(sample, config.quantile_groups),
                    "fdr_q": None,
                }
                rows.append(row)
    _fdr_bh(rows)
    full_path = any(
        row["rank_ic"]["mean"] is not None and row["rank_ic"]["mean"] >= 0.01
        and (row["rank_ic"]["positive_ratio"] or 0) >= 0.50 for row in rows
    )
    tail_path = any((row["top_bottom_mean"] or 0) > 0 for row in rows)
    return {"passed": full_path or tail_path, "gate_paths": {"cross_section": full_path, "top_tail": tail_path}, "results": rows}
