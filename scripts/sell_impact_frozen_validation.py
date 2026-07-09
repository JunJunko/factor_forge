from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sell_impact_ranker_timing_compare as timing_compare
import sell_impact_sorting_repair as base
from factor_forge.config import CostModel, ExecutionConstraints


OUTPUT_ROOT = Path("artifacts/strategy_reviews/experiment10_frozen_validation")
PREDICTIONS = Path(
    "artifacts/strategy_reviews/experiment8_signal_reliability/"
    "stock_signal_reliability_20260709T151758Z/model_predictions.csv"
)
MODEL = "lightgbm_shallow"
HORIZON = 10
FROZEN_LAMBDA = 0.05
TOP_N = 5
HOLDING_DAYS = 10
COST_BPS = 20.0
INITIAL_CASH = 1_000_000
LOT_SIZE = 100
RANDOM_SEED = 20260709


PERIODS = [
    {"period": "2024", "start": "2024-01-01", "end": "2024-12-31", "status": "train_in_sample"},
    {"period": "2025", "start": "2025-01-01", "end": "2025-12-31", "status": "mixed_train_valid"},
    {"period": "2025H1", "start": "2025-01-01", "end": "2025-06-30", "status": "train_in_sample"},
    {"period": "2025H2", "start": "2025-07-01", "end": "2025-12-31", "status": "validation"},
    {"period": "2026H1", "start": "2026-01-01", "end": "2026-06-30", "status": "test_oos"},
]

ROLLING_WINDOWS = [
    {"window": "2024H1", "start": "2024-01-01", "end": "2024-06-30", "status": "train_in_sample"},
    {"window": "2024H2", "start": "2024-07-01", "end": "2024-12-31", "status": "train_in_sample"},
    {"window": "2025H1", "start": "2025-01-01", "end": "2025-06-30", "status": "train_in_sample"},
    {"window": "2025H2", "start": "2025-07-01", "end": "2025-12-31", "status": "validation"},
    {"window": "2026H1", "start": "2026-01-01", "end": "2026-06-30", "status": "test_oos"},
]


def main() -> None:
    args = parse_args()
    output = OUTPUT_ROOT / f"frozen_validation_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("loading frozen stock-level reliability predictions")
    signals = load_predictions(args.predictions)
    version, panel = base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    market_benchmark = timing_compare.load_market_benchmark(version)
    position_multiplier = timing_compare.load_position_multiplier(timing_compare.TIMING_DAILY)
    log(f"signals rows={len(signals):,} dates={signals['trade_date'].nunique():,} range={signals['trade_date'].min().date()}..{signals['trade_date'].max().date()}")

    variants = build_variant_scores(signals)
    period_results = run_period_validation(variants, panel, market_benchmark, position_multiplier, PERIODS, "period", log)
    walk_forward = run_period_validation(variants, panel, market_benchmark, position_multiplier, ROLLING_WINDOWS, "window", log)
    alpha_period = alpha_quality_by_period(variants, PERIODS, "period")
    alpha_walk = alpha_quality_by_period(variants, ROLLING_WINDOWS, "window")
    top5_period = top5_quality_by_period(variants, PERIODS, "period")

    period_results = period_results.merge(alpha_period, on=["period", "status", "variant"], how="left")
    walk_forward = walk_forward.merge(alpha_walk, on=["window", "status", "variant"], how="left")
    contribution = contribution_tables(period_results, "period")
    placebo = period_results.loc[period_results["variant"].isin(["baseline_alpha", "frozen_reliability_lambda005", "random_reliability_lambda005"])].copy()
    lambda_sensitivity = period_results.loc[period_results["variant"].isin(["baseline_alpha", "frozen_reliability_lambda005", "lambda010_sensitivity"])].copy()
    decay = decay_analysis(period_results)
    stability = improvement_win_rate(walk_forward)

    period_results.to_csv(output / "period_results.csv", index=False, encoding="utf-8-sig")
    walk_forward.to_csv(output / "walk_forward_results.csv", index=False, encoding="utf-8-sig")
    placebo.to_csv(output / "placebo_results.csv", index=False, encoding="utf-8-sig")
    lambda_sensitivity.to_csv(output / "lambda_sensitivity.csv", index=False, encoding="utf-8-sig")
    decay.to_csv(output / "decay_analysis.csv", index=False, encoding="utf-8-sig")
    contribution.to_csv(output / "reliability_contribution.csv", index=False, encoding="utf-8-sig")
    top5_period.to_csv(output / "top5_quality.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(output / "stability_summary.csv", index=False, encoding="utf-8-sig")

    write_report(output, period_results, walk_forward, placebo, lambda_sensitivity, decay, contribution, top5_period, stability)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "experiment": "Experiment 10: Frozen Model Validation Framework",
                "predictions": str(args.predictions),
                "model": MODEL,
                "horizon": HORIZON,
                "frozen_lambda": FROZEN_LAMBDA,
                "available_range": [str(signals["trade_date"].min().date()), str(signals["trade_date"].max().date())],
                "data_version": version,
                "important_limitation": (
                    "No frozen stock-level reliability predictions are available before 2024. "
                    "2024 and 2025H1 are in-sample for the reliability model; 2026H1 is the clean OOS period."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 10: frozen validation.")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    return parser.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["trade_date"])
    frame = frame.loc[frame["model"].eq(MODEL) & frame["horizon"].eq(HORIZON)].copy()
    frame["signal_probability"] = pd.to_numeric(frame["signal_probability"], errors="coerce").clip(0.0, 1.0)
    frame["raw_score"] = pd.to_numeric(frame["raw_score"], errors="coerce")
    frame["future_trade_return"] = pd.to_numeric(frame["future_trade_return"], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "ts_code", "raw_score", "signal_probability"])
    frame["reliability_z"] = frame.groupby("trade_date")["signal_probability"].transform(cs_zscore)
    frame["random_reliability_z"] = random_reliability_z(frame)
    return frame


def random_reliability_z(frame: pd.DataFrame) -> pd.Series:
    rng = np.random.default_rng(RANDOM_SEED)
    out = pd.Series(index=frame.index, dtype=float)
    for _, group in frame.groupby("trade_date"):
        values = group["signal_probability"].to_numpy(copy=True)
        rng.shuffle(values)
        z = cs_zscore(pd.Series(values, index=group.index))
        out.loc[group.index] = z
    return out.fillna(0.0)


def build_variant_scores(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    out["baseline_alpha"] = out["raw_score"]
    out["frozen_reliability_lambda005"] = out["raw_score"] + FROZEN_LAMBDA * out["reliability_z"].fillna(0.0)
    out["lambda010_sensitivity"] = out["raw_score"] + 0.10 * out["reliability_z"].fillna(0.0)
    out["random_reliability_lambda005"] = out["raw_score"] + FROZEN_LAMBDA * out["random_reliability_z"].fillna(0.0)
    return out


def variant_names() -> list[str]:
    return ["baseline_alpha", "frozen_reliability_lambda005", "lambda010_sensitivity", "random_reliability_lambda005"]


def run_period_validation(
    signals: pd.DataFrame,
    panel: pd.DataFrame,
    market_benchmark: pd.DataFrame,
    position_multiplier: pd.Series,
    periods: list[dict[str, str]],
    period_col: str,
    log,
) -> pd.DataFrame:
    rows = []
    for item in periods:
        name = item[period_col]
        start = pd.Timestamp(item["start"])
        end = pd.Timestamp(item["end"])
        period_signals = signals.loc[signals["trade_date"].between(start, end)].copy()
        if period_signals.empty:
            for variant in variant_names():
                rows.append(empty_result(name, item["status"], variant, period_col, "no_signal_predictions"))
            continue
        panel_slice = panel.loc[panel["trade_date"].between(start, end)].copy()
        member = period_signals[["trade_date", "ts_code"]].drop_duplicates().copy()
        member["selection_eligible"] = True
        member["condition_quantile"] = 0
        for variant in variant_names():
            log(f"backtest {period_col}={name} variant={variant}")
            factor_values = period_signals[["trade_date", "ts_code", variant]].rename(columns={variant: "factor_value"})
            try:
                result = base.BacktestEngine().run(
                    panel_slice,
                    factor_values,
                    universe="liquid",
                    top_n=TOP_N,
                    holding_days=HOLDING_DAYS,
                    initial_cash=INITIAL_CASH,
                    lot_size=LOT_SIZE,
                    constraints=ExecutionConstraints(
                        exclude_suspended=True,
                        cannot_buy_limit_up=True,
                        cannot_sell_limit_down=True,
                        exclude_st=True,
                        exclude_delisting_period=True,
                        min_listing_days=60,
                    ),
                    cost_model=CostModel(commission_bps_per_side=3, slippage_bps_per_side=5, stamp_duty_bps_sell=5),
                    cost_scenario_bps=COST_BPS,
                    selection_membership=member,
                    position_multiplier=position_multiplier,
                    market_benchmark=market_benchmark,
                )
                rows.append(
                    {
                        period_col: name,
                        "status": item["status"],
                        "variant": variant,
                        "start": item["start"],
                        "end": item["end"],
                        "days": int(period_signals["trade_date"].nunique()),
                        **portfolio_metric_subset(result.metrics),
                        "annualized_turnover": float(result.daily["portfolio_turnover"].mean() * 252),
                        "avg_holding_count": float(result.daily["holding_count"].mean()),
                        "note": "",
                    }
                )
            except Exception as exc:
                row = empty_result(name, item["status"], variant, period_col, str(exc))
                row.update({"start": item["start"], "end": item["end"], "days": int(period_signals["trade_date"].nunique())})
                rows.append(row)
    return pd.DataFrame(rows)


def portfolio_metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "calmar",
        "executed_buys",
        "executed_sells",
        "trade_count",
        "execution_rate",
        "market_index_annualized_return",
    ]
    return {key: metrics.get(key, np.nan) for key in keys}


def empty_result(name: str, status: str, variant: str, period_col: str, note: str) -> dict[str, Any]:
    return {
        period_col: name,
        "status": status,
        "variant": variant,
        "start": None,
        "end": None,
        "days": 0,
        "total_return": np.nan,
        "annualized_return": np.nan,
        "annualized_volatility": np.nan,
        "sharpe": np.nan,
        "max_drawdown": np.nan,
        "calmar": np.nan,
        "executed_buys": 0,
        "executed_sells": 0,
        "trade_count": 0,
        "execution_rate": np.nan,
        "market_index_annualized_return": np.nan,
        "annualized_turnover": np.nan,
        "avg_holding_count": np.nan,
        "note": note,
    }


def alpha_quality_by_period(signals: pd.DataFrame, periods: list[dict[str, str]], period_col: str) -> pd.DataFrame:
    rows = []
    for item in periods:
        name = item[period_col]
        frame = signals.loc[signals["trade_date"].between(pd.Timestamp(item["start"]), pd.Timestamp(item["end"]))].copy()
        for variant in variant_names():
            values = []
            for _, group in frame.groupby("trade_date"):
                data = group.dropna(subset=[variant, "future_trade_return"])
                if len(data) < 5 or data[variant].nunique() < 2 or data["future_trade_return"].nunique() < 2:
                    continue
                value = data[variant].corr(data["future_trade_return"], method="spearman")
                if pd.notna(value):
                    values.append(float(value))
            series = pd.Series(values, dtype=float)
            mean = float(series.mean()) if len(series) else np.nan
            std = float(series.std(ddof=1)) if len(series) > 1 else np.nan
            rows.append(
                {
                    period_col: name,
                    "status": item["status"],
                    "variant": variant,
                    "rank_ic": mean,
                    "rank_ic_std": std,
                    "icir": float(mean / std * np.sqrt(252)) if std and np.isfinite(std) and std > 0 else np.nan,
                    "positive_ic_ratio": float((series > 0).mean()) if len(series) else np.nan,
                    "ic_days": int(len(series)),
                }
            )
    return pd.DataFrame(rows)


def top5_quality_by_period(signals: pd.DataFrame, periods: list[dict[str, str]], period_col: str) -> pd.DataFrame:
    rows = []
    for item in periods:
        frame = signals.loc[signals["trade_date"].between(pd.Timestamp(item["start"]), pd.Timestamp(item["end"]))].copy()
        for variant in variant_names():
            daily = []
            for _, group in frame.groupby("trade_date"):
                top = group.sort_values(variant, ascending=False).head(TOP_N)
                if top.empty:
                    continue
                daily.append(
                    {
                        "future_return": float(top["future_trade_return"].mean()),
                        "win_rate": float(top["future_trade_return"].gt(0.002).mean()),
                        "avg_probability": float(top["signal_probability"].mean()),
                        "avg_raw_score": float(top["raw_score"].mean()),
                    }
                )
            d = pd.DataFrame(daily)
            rows.append(
                {
                    period_col: item[period_col],
                    "status": item["status"],
                    "variant": variant,
                    "top5_future_return": float(d["future_return"].mean()) if not d.empty else np.nan,
                    "top5_win_rate": float(d["win_rate"].mean()) if not d.empty else np.nan,
                    "top5_avg_probability": float(d["avg_probability"].mean()) if not d.empty else np.nan,
                    "top5_avg_raw_score": float(d["avg_raw_score"].mean()) if not d.empty else np.nan,
                    "days": int(len(d)),
                }
            )
    return pd.DataFrame(rows)


def contribution_tables(results: pd.DataFrame, period_col: str) -> pd.DataFrame:
    base = results.loc[results["variant"].eq("baseline_alpha")].set_index(period_col)
    rows = []
    for variant in ["frozen_reliability_lambda005", "lambda010_sensitivity", "random_reliability_lambda005"]:
        comp = results.loc[results["variant"].eq(variant)].set_index(period_col)
        for period, row in comp.iterrows():
            if period not in base.index:
                continue
            b = base.loc[period]
            rows.append(
                {
                    period_col: period,
                    "status": row.get("status"),
                    "variant": variant,
                    "delta_annualized_return": safe_delta(row.get("annualized_return"), b.get("annualized_return")),
                    "delta_sharpe": safe_delta(row.get("sharpe"), b.get("sharpe")),
                    "delta_max_drawdown": safe_delta(row.get("max_drawdown"), b.get("max_drawdown")),
                    "delta_ic": safe_delta(row.get("rank_ic"), b.get("rank_ic")),
                    "delta_icir": safe_delta(row.get("icir"), b.get("icir")),
                    "delta_positive_ic_ratio": safe_delta(row.get("positive_ic_ratio"), b.get("positive_ic_ratio")),
                }
            )
    return pd.DataFrame(rows)


def decay_analysis(period_results: pd.DataFrame) -> pd.DataFrame:
    contrib = contribution_tables(period_results, "period")
    rows = []
    for period in ["2024", "2025", "2026H1"]:
        real = contrib.loc[
            contrib["period"].eq(period) & contrib["variant"].eq("frozen_reliability_lambda005")
        ]
        if real.empty:
            continue
        row = real.iloc[0]
        rows.append(
            {
                "period": period,
                "status": row["status"],
                "delta_icir": row["delta_icir"],
                "delta_rank_ic": row["delta_ic"],
                "delta_annualized_return": row["delta_annualized_return"],
                "delta_sharpe": row["delta_sharpe"],
                "delta_max_drawdown": row["delta_max_drawdown"],
            }
        )
    return pd.DataFrame(rows)


def improvement_win_rate(walk: pd.DataFrame) -> pd.DataFrame:
    base = walk.loc[walk["variant"].eq("baseline_alpha")].set_index("window")
    rows = []
    for variant in ["frozen_reliability_lambda005", "random_reliability_lambda005", "lambda010_sensitivity"]:
        comp = walk.loc[walk["variant"].eq(variant)].set_index("window")
        common = [idx for idx in comp.index if idx in base.index]
        for metric in ["annualized_return", "sharpe", "icir", "max_drawdown"]:
            wins = []
            for idx in common:
                c = comp.loc[idx, metric]
                b = base.loc[idx, metric]
                if not np.isfinite(c) or not np.isfinite(b):
                    continue
                if metric == "max_drawdown":
                    wins.append(c > b)
                else:
                    wins.append(c > b)
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "windows": int(len(wins)),
                    "improvement_count": int(sum(wins)),
                    "improvement_rate": float(np.mean(wins)) if wins else np.nan,
                }
            )
    return pd.DataFrame(rows)


def safe_delta(a: Any, b: Any) -> float:
    try:
        a = float(a)
        b = float(b)
    except Exception:
        return np.nan
    return a - b if np.isfinite(a) and np.isfinite(b) else np.nan


def cs_zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(
    output: Path,
    period_results: pd.DataFrame,
    walk_forward: pd.DataFrame,
    placebo: pd.DataFrame,
    lambda_sensitivity: pd.DataFrame,
    decay: pd.DataFrame,
    contribution: pd.DataFrame,
    top5: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    period_view = period_results[
        [
            "period",
            "status",
            "variant",
            "annualized_return",
            "sharpe",
            "max_drawdown",
            "calmar",
            "rank_ic",
            "icir",
            "positive_ic_ratio",
            "executed_buys",
        ]
    ].sort_values(["period", "variant"])
    stability_table = period_results.pivot_table(index=["period", "status"], columns="variant", values="sharpe", aggfunc="first").reset_index()
    if {"baseline_alpha", "frozen_reliability_lambda005"} <= set(stability_table.columns):
        stability_table["delta_real_vs_baseline"] = stability_table["frozen_reliability_lambda005"] - stability_table["baseline_alpha"]
    lines = [
        "# Experiment 10: Frozen Model Validation Framework",
        "",
        "## Scope",
        "- Frozen rule: `final_score = alpha_score + 0.05 * reliability_zscore`.",
        "- No Alpha model retraining, no reliability model retraining, no lambda search.",
        "- Random reliability is a same-date permutation of the real reliability distribution.",
        "- Important limitation: frozen stock-level reliability predictions only exist from 2024 onward. 2024 and 2025H1 are in-sample for the reliability model; 2026H1 is the clean OOS period.",
        "",
        "## Period Stability Table",
        md_table(stability_table, 20),
        "",
        "## Period Results",
        md_table(period_view, 80),
        "",
        "## Reliability Contribution",
        md_table(contribution, 80),
        "",
        "## Walk-Forward Style Windows",
        "These are frozen semiannual windows, not true retrained walk-forward windows, because pre-2024 frozen reliability predictions are unavailable.",
        md_table(walk_forward[["window", "status", "variant", "annualized_return", "sharpe", "max_drawdown", "rank_ic", "icir"]].sort_values(["window", "variant"]), 80),
        "",
        "## Win Rate Of Improvement",
        md_table(stability, 40),
        "",
        "## Placebo Test",
        md_table(placebo[["period", "status", "variant", "annualized_return", "sharpe", "max_drawdown", "rank_ic", "icir"]].sort_values(["period", "variant"]), 80),
        "",
        "## Lambda Sensitivity",
        md_table(lambda_sensitivity[["period", "status", "variant", "annualized_return", "sharpe", "max_drawdown", "rank_ic", "icir"]].sort_values(["period", "variant"]), 80),
        "",
        "## Decay Analysis",
        md_table(decay, 20),
        "",
        "## Top5 Quality",
        md_table(top5.sort_values(["period", "variant"]), 80),
        "",
        "## Required Answers",
        "- Cross-period reliability lift: judge by `Period Stability Table` and `Win Rate Of Improvement`.",
        "- Whether lift only comes from 2026H1: compare 2024/2025/2026H1 in `Reliability Contribution`.",
        "- Whether random reliability replicates it: see `Placebo Test`.",
        "- Whether lambda=0.05 is stable: see `Lambda Sensitivity`; this is diagnostic only, not optimization.",
        "- Whether ready for live validation: see the final conclusion in the assistant response.",
        "",
        "## Files",
        "- `period_results.csv`",
        "- `walk_forward_results.csv`",
        "- `placebo_results.csv`",
        "- `lambda_sensitivity.csv`",
        "- `decay_analysis.csv`",
        "- `frozen_validation_report.md`",
    ]
    text = "\n".join(lines) + "\n"
    (output / "frozen_validation_report.md").write_text(text, encoding="utf-8")
    (output / "report.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
