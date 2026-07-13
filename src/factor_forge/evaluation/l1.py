from __future__ import annotations

import math

import numpy as np
import pandas as pd

from factor_forge.config import FactorSpec, L1Config, PrimaryGateConfig
from .neutralization import build_variants


def _forward_open_return(panel: pd.DataFrame, horizon: int) -> pd.Series:
    ordered = panel[["ts_code", "trade_date", "adj_open"]].sort_values(["ts_code", "trade_date"])
    grouped = ordered.groupby("ts_code", sort=False)["adj_open"]
    entry = grouped.shift(-1)
    exit_price = grouped.shift(-(horizon + 1))
    result = exit_price / entry - 1.0
    return result.reindex(panel.index)


def build_forward_targets(panel: pd.DataFrame, horizon: int) -> dict[str, pd.Series]:
    """Build PIT signal-date stock and leave-one-out SW L1 relative returns."""
    stock_return = _forward_open_return(panel, horizon)
    industry = panel["industry_l1_code"]
    valid = stock_return.notna() & industry.notna()
    keys = [panel["trade_date"], industry]
    total = stock_return.where(valid).groupby(keys).transform("sum")
    count = stock_return.where(valid).groupby(keys).transform("count")
    peer_return = (total - stock_return) / (count - 1).replace(0, np.nan)
    return {
        "stock_return": stock_return,
        "stock_minus_sw_l1_return": stock_return - peer_return,
    }


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


def _newey_west_stats(values: pd.Series, max_lags: int) -> dict:
    """HAC inference for the mean of a daily IC series."""
    clean = values.dropna().to_numpy(dtype=float)
    count = len(clean)
    if count < 2:
        return {"nw_lags": 0, "nw_t_value": None, "nw_p_value": None}
    lags = min(max(int(max_lags), 0), count - 1)
    centered = clean - clean.mean()
    long_run_variance = float(np.dot(centered, centered) / count)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        autocovariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        long_run_variance += 2.0 * weight * autocovariance
    variance_of_mean = max(long_run_variance, 0.0) / count
    if variance_of_mean <= 0:
        return {"nw_lags": lags, "nw_t_value": None, "nw_p_value": None}
    t_value = float(clean.mean() / math.sqrt(variance_of_mean))
    return {
        "nw_lags": lags,
        "nw_t_value": t_value,
        "nw_p_value": math.erfc(abs(t_value) / math.sqrt(2)),
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


def _fdr_bh(rows: list[dict], p_value_key: str = "p_value") -> None:
    valid = [
        (i, row["rank_ic"][p_value_key])
        for i, row in enumerate(rows)
        if row["rank_ic"].get(p_value_key) is not None
    ]
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


def _evaluate_primary_gate(
    rows: list[dict],
    gate: PrimaryGateConfig,
    *,
    condition_quantile: int | None = None,
) -> dict:
    matches = [
        row for row in rows
        if row.get("target") == gate.target
        and row.get("variant") == gate.variant
        and row.get("universe") == gate.universe
        and row.get("horizon") == gate.horizon
        and (
            condition_quantile is None
            or row.get("condition_quantile") == condition_quantile
        )
    ]
    selector = {
        "target": gate.target,
        "variant": gate.variant,
        "universe": gate.universe,
        "horizon": gate.horizon,
        "metric": gate.metric,
    }
    if condition_quantile is not None:
        selector["condition_quantile"] = condition_quantile
    if len(matches) != 1:
        return {
            "passed": False,
            "selector": selector,
            "reason": f"expected exactly one primary result, found {len(matches)}",
            "checks": {},
        }
    row = matches[0]
    summary = row[gate.metric]
    mean = summary.get("mean")
    positive_ratio = summary.get("positive_ratio")
    fdr_q = row.get("fdr_q")
    checks = {
        "mean": mean is not None and mean >= gate.min_mean,
        "positive_ratio": (
            positive_ratio is not None and positive_ratio >= gate.min_positive_ratio
        ),
        "fdr_q": gate.max_fdr_q is None or (fdr_q is not None and fdr_q <= gate.max_fdr_q),
    }
    tail_fallback = bool(
        gate.allow_top_tail_fallback and (row.get("top_bottom_mean") or 0) > 0
    )
    return {
        "passed": all(checks.values()) or tail_fallback,
        "selector": selector,
        "thresholds": {
            "min_mean": gate.min_mean,
            "min_positive_ratio": gate.min_positive_ratio,
            "max_fdr_q": gate.max_fdr_q,
            "allow_top_tail_fallback": gate.allow_top_tail_fallback,
        },
        "observed": {
            "mean": mean,
            "positive_ratio": positive_ratio,
            "fdr_q": fdr_q,
            "top_bottom_mean": row.get("top_bottom_mean"),
        },
        "checks": checks,
        "tail_fallback_used": tail_fallback and not all(checks.values()),
    }


def _assign_daily_condition_quantiles(frame: pd.DataFrame, groups: int) -> pd.Series:
    output = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    for _, indexes in frame.groupby("trade_date").groups.items():
        values = frame.loc[indexes, "condition_factor"]
        if values.notna().sum() < groups:
            continue
        percentile = values.rank(method="average", pct=True)
        buckets = np.ceil(percentile * groups).clip(1, groups)
        output.loc[indexes] = buckets.astype("Int64")
    return output


def evaluate_conditional_ic(
    panel: pd.DataFrame,
    factor_values: pd.DataFrame,
    conditioning_factor_values: pd.DataFrame,
    spec: FactorSpec,
    config: L1Config,
    conditioning_factor_name: str,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate the main factor's cross-sectional IC inside condition-factor bins."""
    conditional = config.conditional_ic
    candidates = ["trade_date", "ts_code", "adj_open", "log_total_mv", "industry_l1_code"]
    candidates += [f"is_{universe}" for universe in config.universes]
    seen = set()
    needed = [c for c in candidates if c in panel.columns and not (c in seen or seen.add(c))]
    merged = panel[needed].merge(
        factor_values[["trade_date", "ts_code", "factor_value"]],
        on=["trade_date", "ts_code"], how="left",
    ).merge(
        conditioning_factor_values[["trade_date", "ts_code", "factor_value"]].rename(
            columns={"factor_value": "condition_factor"}
        ),
        on=["trade_date", "ts_code"], how="left",
    ).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    variants = build_variants(merged, spec.scope.cross_section)
    if spec.factor.direction == "unknown":
        variants.update({f"{name}_negative": -value for name, value in list(variants.items())})

    rows: list[dict] = []
    daily_records: list[dict] = []
    for horizon in config.forward_horizons:
        targets = build_forward_targets(merged, horizon)
        for target_name in config.targets:
            forward_column = f"forward_{target_name}_{horizon}"
            merged[forward_column] = targets[target_name]
            for universe in config.universes:
                universe_mask = merged[f"is_{universe}"].fillna(False).astype(bool)
                for variant_name, values in variants.items():
                    sample = merged.loc[
                        universe_mask, ["trade_date", "condition_factor", forward_column]
                    ].copy()
                    sample = sample.dropna(subset=["condition_factor"])
                    sample["condition_quantile"] = _assign_daily_condition_quantiles(
                        sample, conditional.quantile_groups
                    )
                    sample["factor"] = values.loc[universe_mask]
                    sample = sample.rename(columns={forward_column: "forward_return"})
                    sample = sample.dropna(subset=["factor", "forward_return", "condition_quantile"])
                    daily_size = sample.groupby("trade_date").size()
                    valid_dates = daily_size[daily_size >= config.min_cross_section].index
                    sample = sample[sample["trade_date"].isin(valid_dates)].copy()

                    for quantile in range(1, conditional.quantile_groups + 1):
                        bucket = sample.loc[sample["condition_quantile"] == quantile].copy()
                        bucket_size = bucket.groupby("trade_date").size()
                        bucket_dates = bucket_size[bucket_size >= conditional.min_group_size].index
                        bucket = bucket[bucket["trade_date"].isin(bucket_dates)]
                        rank_ic = (
                            _daily_correlation(bucket, "spearman")
                            if len(bucket) else pd.Series(dtype=float)
                        )
                        summary = _summary(rank_ic)
                        summary.update(_newey_west_stats(rank_ic, max_lags=max(horizon - 1, 0)))
                        row = {
                            "conditioning_factor": conditioning_factor_name,
                            "target": target_name,
                            "variant": variant_name,
                            "universe": universe,
                            "horizon": horizon,
                            "condition_quantile": quantile,
                            "observations": int(len(bucket)),
                            "days": int(rank_ic.notna().sum()),
                            "mean_group_size": float(bucket_size.loc[bucket_dates].mean()) if len(bucket_dates) else None,
                            "condition_value_min": float(bucket["condition_factor"].min()) if len(bucket) else None,
                            "condition_value_median": float(bucket["condition_factor"].median()) if len(bucket) else None,
                            "condition_value_max": float(bucket["condition_factor"].max()) if len(bucket) else None,
                            "mean_forward_return": float(bucket["forward_return"].mean()) if len(bucket) else None,
                            "rank_ic": summary,
                            "significance_rank": None,
                            "fdr_q": None,
                        }
                        rows.append(row)
                        for trade_date, ic in rank_ic.dropna().items():
                            daily_records.append({
                                "trade_date": trade_date,
                                "conditioning_factor": conditioning_factor_name,
                                "target": target_name,
                                "variant": variant_name,
                                "universe": universe,
                                "horizon": horizon,
                                "condition_quantile": quantile,
                                "observations": int(bucket_size.get(trade_date, 0)),
                                "rank_ic": float(ic),
                            })

    _fdr_bh(rows, p_value_key="nw_p_value")
    context_groups: dict[tuple, list[dict]] = {}
    for row in rows:
        context = (row["target"], row["variant"], row["universe"], row["horizon"])
        context_groups.setdefault(context, []).append(row)
    strongest_by_context = []
    for context_rows in context_groups.values():
        ranked = sorted(
            (row for row in context_rows if row["rank_ic"].get("nw_t_value") is not None),
            key=lambda row: abs(row["rank_ic"]["nw_t_value"]), reverse=True,
        )
        for rank, row in enumerate(ranked, start=1):
            row["significance_rank"] = rank
        if ranked:
            strongest_by_context.append(ranked[0])
    significant = [
        row for row in rows if row["rank_ic"].get("nw_t_value") is not None
    ]
    strongest = max(
        significant, key=lambda row: abs(row["rank_ic"]["nw_t_value"]), default=None
    )
    primary_gate = None
    if conditional.primary_gate is not None:
        primary_gate = _evaluate_primary_gate(
            rows,
            conditional.primary_gate,
            condition_quantile=conditional.primary_gate.condition_quantile,
        )
    result = {
        "enabled": True,
        "conditioning_factor": conditioning_factor_name,
        "quantile_groups": conditional.quantile_groups,
        "min_group_size": conditional.min_group_size,
        "inference": "Newey-West HAC with max_lags=horizon-1; FDR is Benjamini-Hochberg across conditional tests",
        "passed": primary_gate["passed"] if primary_gate is not None else None,
        "primary_gate": primary_gate,
        "strongest_result": strongest,
        "strongest_by_context": strongest_by_context,
        "results": rows,
    }
    daily_columns = [
        "trade_date", "conditioning_factor", "target", "variant", "universe", "horizon",
        "condition_quantile", "observations", "rank_ic",
    ]
    return result, pd.DataFrame(daily_records, columns=daily_columns)


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
    daily_records: list[dict] = []
    for horizon in config.forward_horizons:
        targets = build_forward_targets(merged, horizon)
        for target_name in config.targets:
            forward_column = f"forward_{target_name}_{horizon}"
            merged[forward_column] = targets[target_name]
            for universe in config.universes:
                universe_mask = merged[f"is_{universe}"].fillna(False).astype(bool)
                for variant_name, values in variants.items():
                    sample = merged.loc[universe_mask, ["trade_date", forward_column]].copy()
                    sample["factor"] = values.loc[universe_mask]
                    sample = sample.rename(columns={forward_column: "forward_return"})
                    sample = sample.dropna()
                    daily_size = sample.groupby("trade_date").size()
                    sample = sample[sample["trade_date"].isin(daily_size[daily_size >= config.min_cross_section].index)]
                    rank_ic = _daily_correlation(sample, "spearman") if len(sample) else pd.Series(dtype=float)
                    pearson_ic = _daily_correlation(sample, "pearson") if len(sample) else pd.Series(dtype=float)
                    cutoff = rank_ic.index.sort_values()[int(len(rank_ic) * 0.8)] if len(rank_ic) >= 5 else None
                    oos = rank_ic[rank_ic.index >= cutoff] if cutoff is not None else pd.Series(dtype=float)
                    rank_summary = _summary(rank_ic)
                    rank_summary.update(_newey_west_stats(rank_ic, max_lags=max(horizon - 1, 0)))
                    row = {
                        "target": target_name, "variant": variant_name,
                        "universe": universe, "horizon": horizon,
                        "observations": int(len(sample)), "days": int(sample["trade_date"].nunique()),
                        "rank_ic": rank_summary, "pearson_ic": _summary(pearson_ic),
                        "oos_rank_ic": _summary(oos), **_quantile_metrics(sample, config.quantile_groups),
                        "fdr_q": None,
                    }
                    rows.append(row)
                    for trade_date, ic in rank_ic.dropna().items():
                        daily_records.append({
                            "trade_date": trade_date,
                            "target": target_name,
                            "variant": variant_name,
                            "universe": universe,
                            "horizon": horizon,
                            "rank_ic": float(ic),
                        })
    _fdr_bh(rows, p_value_key="nw_p_value")
    if config.primary_gate is not None:
        primary_gate = _evaluate_primary_gate(rows, config.primary_gate)
        return {
            "passed": primary_gate["passed"],
            "gate_paths": {"primary": primary_gate["passed"], "top_tail": False},
            "primary_gate": primary_gate,
            "inference": "Newey-West HAC with max_lags=horizon-1; FDR is Benjamini-Hochberg across standard L1 tests",
            "results": rows,
            "daily_rank_ic": daily_records,
        }
    full_path = any(
        row["rank_ic"]["mean"] is not None and row["rank_ic"]["mean"] >= 0.01
        and (row["rank_ic"]["positive_ratio"] or 0) >= 0.50 for row in rows
    )
    tail_path = any((row["top_bottom_mean"] or 0) > 0 for row in rows)
    return {
        "passed": full_path or tail_path,
        "gate_paths": {"cross_section": full_path, "top_tail": tail_path},
        "inference": "Newey-West HAC with max_lags=horizon-1; FDR is Benjamini-Hochberg across standard L1 tests",
        "results": rows,
        "daily_rank_ic": daily_records,
    }
