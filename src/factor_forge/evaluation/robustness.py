from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_robustness(panel: pd.DataFrame, rows: list[dict], parameter_results: list[dict]) -> dict:
    primary = [r for r in rows if r["universe"] == "liquid" and r["top_n"] == 2 and r["cost_bps"] == 20]
    if not primary:
        return {"status": "NO_PRIMARY_RESULT"}
    yearly, rolling, walk = [], [], []
    concentrations, removed_best = [], []
    for row in primary:
        daily = row["daily"].copy()
        annual = daily.resample("YE", on="trade_date")["excess_return"].sum()
        yearly.append({str(index.year): float(value) for index, value in annual.items()})
        roll = daily.set_index("trade_date")["excess_return"].rolling(252, min_periods=63).sum().dropna()
        rolling.append(float((roll > 0).mean()) if len(roll) else None)
        years = sorted(daily["trade_date"].dt.year.unique())
        tests = []
        for year in years[3:]:
            value = float(daily.loc[daily["trade_date"].dt.year == year, "excess_return"].sum())
            tests.append({"test_year": int(year), "excess_return": value})
        walk.append(tests)
        positive = daily["excess_return"].clip(lower=0).sort_values(ascending=False)
        concentrations.append(float(positive.head(10).sum() / positive.sum()) if positive.sum() else 1.0)
        trimmed = daily["excess_return"].drop(index=positive.head(10).index)
        removed_best.append(float(trimmed.mean() * 252) if len(trimmed) else None)
    curve = {}
    for n in sorted({r["top_n"] for r in rows}):
        values = [r["metrics"]["annualized_excess_return"] for r in rows
                  if r["universe"] == "liquid" and r["top_n"] == n and r["cost_bps"] == 20]
        curve[str(n)] = float(np.median(values)) if values else None
    holding = {}
    for days in sorted({r["holding_days"] for r in rows}):
        values = [r["metrics"]["annualized_excess_return"] for r in rows
                  if r["universe"] == "liquid" and r["top_n"] == 2 and r["holding_days"] == days and r["cost_bps"] == 20]
        holding[str(days)] = float(np.median(values)) if values else None
    costs = {}
    for cost in sorted({r["cost_bps"] for r in rows}):
        values = [r["metrics"]["annualized_excess_return"] for r in rows
                  if r["universe"] == "liquid" and r["top_n"] == 2 and r["cost_bps"] == cost]
        costs[str(cost)] = float(np.median(values)) if values else None
    exposures = _position_exposures(panel, primary)
    return {
        "status": "COMPLETED",
        "year_by_year": yearly,
        "rolling_252d_positive_ratio": [x for x in rolling if x is not None],
        "walk_forward": {"train_years": 3, "test_years": 1, "folds": walk,
                         "note": "The factor has no fitted parameters; train windows only precede untouched yearly tests."},
        "topn_curve": curve, "holding_decay": holding, "cost_sensitivity": costs,
        "parameter_neighborhood": parameter_results,
        "contribution_concentration": {"median_top10_positive_days": float(np.median(concentrations))},
        "annualized_excess_without_best_10_days": [x for x in removed_best if x is not None],
        **exposures,
    }


def _position_exposures(panel: pd.DataFrame, primary: list[dict]) -> dict:
    frames = []
    for row in primary:
        positions = row.get("positions")
        if positions is not None and not positions.empty:
            frames.append(positions)
    if not frames:
        return {"industry_exposure": {}, "weighted_log_market_cap": None}
    positions = pd.concat(frames, ignore_index=True)
    industry_field = next(
        (field for field in ["industry_l1_code", "industry_l2_code"] if field in panel),
        None,
    )
    if industry_field is None:
        return {"industry_exposure": {}, "weighted_log_market_cap": None}
    joined = positions.merge(
        panel[["trade_date", "ts_code", industry_field, "log_total_mv"]],
        on=["trade_date", "ts_code"], how="left",
    )
    joined["weight"] = joined["market_value"] / joined.groupby("trade_date")["market_value"].transform("sum")
    industry = joined.groupby(industry_field, dropna=True)["weight"].mean().sort_values(ascending=False)
    weighted_size = (joined["weight"] * joined["log_total_mv"]).groupby(joined["trade_date"]).sum().mean()
    return {
        "industry_exposure": {str(k): float(v) for k, v in industry.items()},
        "weighted_log_market_cap": float(weighted_size) if np.isfinite(weighted_size) else None,
    }
