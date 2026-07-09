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
import sell_impact_trade_quality_optimization as tq
from factor_forge.config import CostModel, ExecutionConstraints


OUTPUT_ROOT = Path("artifacts/strategy_reviews")
DEFAULT_SAMPLES = 120
RANDOM_SEED = 20260709

PARAM_COLUMNS = [
    "max_positions",
    "entry_band_rank_min",
    "entry_raw_rank_min",
    "continue_band_rank_min",
    "continue_raw_rank_min",
    "min_hold_days",
    "max_hold_days",
    "max_microcap_score",
    "min_liquidity",
    "min_price_reversal",
]


def main() -> None:
    args = parse_args()
    output = OUTPUT_ROOT / f"sell_impact_trade_param_ml_surface_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"samples={args.samples} seed={args.seed}")
    signals = tq.load_signals()
    version, panel = base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel.loc[panel["trade_date"].between(pd.Timestamp(tq.TEST_START), pd.Timestamp(tq.TEST_END))].copy()
    timing = timing_compare.load_position_multiplier(timing_compare.TIMING_DAILY)
    market_benchmark = timing_compare.load_market_benchmark(version)
    constraints = ExecutionConstraints(
        exclude_suspended=True,
        cannot_buy_limit_up=True,
        cannot_sell_limit_down=True,
        exclude_st=True,
        exclude_delisting_period=True,
        min_listing_days=60,
    )
    cost_model = CostModel(commission_bps_per_side=3, slippage_bps_per_side=5, stamp_duty_bps_sell=5)
    configs = sample_configs(args.samples, args.seed)
    pd.DataFrame(configs).to_csv(output / "sampled_trade_params.csv", index=False, encoding="utf-8-sig")

    metric_rows: list[dict[str, Any]] = []
    monthly_rows: list[pd.DataFrame] = []
    trade_quality_rows: list[pd.DataFrame] = []
    for idx, cfg in enumerate(configs, start=1):
        name = cfg["variant"]
        daily, trades, _positions, metrics = tq.run_trade_quality_backtest(
            panel=panel,
            signals=signals,
            timing=timing,
            market_benchmark=market_benchmark,
            constraints=constraints,
            cost_model=cost_model,
            cfg=cfg,
        )
        monthly = tq.period_breakdown(daily, trades, name, "M")
        quality = tq.trade_quality_summary(trades.assign(variant=name), panel)
        monthly_rows.append(monthly)
        if not quality.empty:
            trade_quality_rows.append(quality)
        row = {**cfg, **metrics}
        row.update(quality_to_row(quality))
        row.update(monthly_to_row(monthly))
        row["objective"] = objective(row)
        metric_rows.append(row)
        if idx == 1 or idx % 10 == 0 or idx == len(configs):
            log(
                f"{idx:03d}/{len(configs)} best_obj={max(r['objective'] for r in metric_rows):.4f} "
                f"last={name} ann={metrics['annualized_return']:.2%} "
                f"mdd={metrics['max_drawdown']:.2%} trades={metrics['trade_count']}"
            )

    metrics_df = pd.DataFrame(metric_rows).sort_values("objective", ascending=False)
    monthly_df = pd.concat(monthly_rows, ignore_index=True)
    trade_quality_df = pd.concat(trade_quality_rows, ignore_index=True) if trade_quality_rows else pd.DataFrame()
    metrics_df.to_csv(output / "param_search_metrics.csv", index=False, encoding="utf-8-sig")
    monthly_df.to_csv(output / "param_search_monthly.csv", index=False, encoding="utf-8-sig")
    trade_quality_df.to_csv(output / "param_search_trade_quality.csv", index=False, encoding="utf-8-sig")

    model_result = fit_response_surface(metrics_df, args.seed)
    model_result["importance"].to_csv(output / "response_surface_feature_importance.csv", index=False, encoding="utf-8-sig")
    model_result["predictions"].to_csv(output / "response_surface_predictions.csv", index=False, encoding="utf-8-sig")
    robust = robust_candidates(metrics_df)
    robust.to_csv(output / "robust_parameter_candidates.csv", index=False, encoding="utf-8-sig")
    write_report(output, metrics_df, robust, model_result["importance"], model_result["cv_metrics"])
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "source_run": str(tq.SOURCE_RUN),
                "model": tq.MODEL,
                "band_target": tq.BAND_TARGET,
                "samples": args.samples,
                "seed": args.seed,
                "data_version": version,
                "objective": "annualized_return - drawdown/turnover/concentration/month-loss penalties",
                "response_surface": model_result["cv_metrics"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML response-surface search for sell-impact trade parameters.")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def sample_configs(n: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    anchors = [
        {
            "max_positions": 5,
            "entry_band_rank_min": 0.95,
            "entry_raw_rank_min": 0.65,
            "continue_band_rank_min": 0.85,
            "continue_raw_rank_min": 0.55,
            "min_hold_days": 3,
            "max_hold_days": 20,
            "max_microcap_score": 1.0,
            "min_liquidity": -1.0,
            "min_price_reversal": -0.5,
        },
        {
            "max_positions": 8,
            "entry_band_rank_min": 0.95,
            "entry_raw_rank_min": 0.65,
            "continue_band_rank_min": 0.85,
            "continue_raw_rank_min": 0.55,
            "min_hold_days": 3,
            "max_hold_days": 20,
            "max_microcap_score": 1.0,
            "min_liquidity": -1.0,
            "min_price_reversal": -0.5,
        },
    ]
    rows.extend(anchors)
    while len(rows) < n:
        min_hold = int(rng.choice([2, 3, 5]))
        max_hold = int(rng.choice([10, 15, 20, 30]))
        if max_hold <= min_hold:
            max_hold = min_hold + 5
        rows.append(
            {
                "max_positions": int(rng.choice([4, 5, 6, 8])),
                "entry_band_rank_min": float(rng.choice([0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98])),
                "entry_raw_rank_min": float(rng.choice([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])),
                "continue_band_rank_min": float(rng.choice([0.75, 0.80, 0.85, 0.90, 0.95])),
                "continue_raw_rank_min": float(rng.choice([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])),
                "min_hold_days": min_hold,
                "max_hold_days": max_hold,
                "max_microcap_score": float(rng.choice([0.5, 1.0, 1.5, 2.0])),
                "min_liquidity": float(rng.choice([-1.5, -1.0, -0.5, 0.0])),
                "min_price_reversal": float(rng.choice([-1.0, -0.5, 0.0, 0.5])),
            }
        )
    configs = []
    seen: set[tuple] = set()
    for i, row in enumerate(rows):
        key = tuple(row[col] for col in PARAM_COLUMNS)
        if key in seen:
            continue
        seen.add(key)
        configs.append(
            {
                "variant": f"param_{len(configs):03d}",
                "description": "randomized high-threshold account trade rule",
                "entry_pool": "threshold",
                "sell_rule": "continue",
                **row,
            }
        )
    return configs


def quality_to_row(quality: pd.DataFrame) -> dict[str, float]:
    if quality.empty:
        return {
            "round_trips": 0,
            "mean_trade_return": np.nan,
            "median_trade_return": np.nan,
            "win_rate": np.nan,
            "payoff_ratio": np.nan,
        }
    row = quality.iloc[0]
    return {
        "round_trips": float(row.get("round_trips", np.nan)),
        "mean_trade_return": float(row.get("mean_trade_return", np.nan)),
        "median_trade_return": float(row.get("median_trade_return", np.nan)),
        "win_rate": float(row.get("win_rate", np.nan)),
        "payoff_ratio": float(row.get("payoff_ratio", np.nan)),
    }


def monthly_to_row(monthly: pd.DataFrame) -> dict[str, float]:
    if monthly.empty:
        return {"worst_month_return": np.nan, "monthly_return_std": np.nan, "positive_months": np.nan}
    returns = monthly["period_return"]
    return {
        "worst_month_return": float(returns.min()),
        "monthly_return_std": float(returns.std(ddof=1)),
        "positive_months": float(returns.gt(0).sum()),
    }


def objective(row: dict[str, Any]) -> float:
    annual = float(row.get("annualized_return", 0.0) or 0.0)
    mdd = abs(float(row.get("max_drawdown", 0.0) or 0.0))
    trades = float(row.get("trade_count", 0.0) or 0.0)
    month_share = float(row.get("top_month_return_share", 0.0) or 0.0)
    worst_month = float(row.get("worst_month_return", 0.0) or 0.0)
    round_trips = float(row.get("round_trips", 0.0) or 0.0)
    trade_penalty = max(0.0, (trades - 320.0) / 500.0)
    concentration_penalty = max(0.0, month_share - 0.55)
    month_loss_penalty = max(0.0, -worst_month - 0.08)
    sample_penalty = 0.08 if round_trips < 60 else 0.0
    return annual - 1.15 * mdd - 0.18 * trade_penalty - 0.35 * concentration_penalty - 0.70 * month_loss_penalty - sample_penalty


def fit_response_surface(metrics: pd.DataFrame, seed: int) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold, cross_val_predict

    frame = metrics.dropna(subset=["objective"]).copy()
    x = frame[PARAM_COLUMNS].astype(float)
    y = frame["objective"].astype(float)
    if len(frame) < 20:
        raise ValueError("not enough rows for response-surface fitting")
    model = ExtraTreesRegressor(
        n_estimators=500,
        max_features=0.85,
        min_samples_leaf=3,
        random_state=seed,
    )
    cv = KFold(n_splits=min(5, len(frame)), shuffle=True, random_state=seed)
    pred = cross_val_predict(model, x, y, cv=cv)
    cv_r2 = float(r2_score(y, pred))
    model.fit(x, y)
    perm = permutation_importance(model, x, y, n_repeats=25, random_state=seed)
    importance = pd.DataFrame(
        {
            "parameter": PARAM_COLUMNS,
            "model_importance": model.feature_importances_,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
        }
    ).sort_values("permutation_importance_mean", ascending=False)
    predictions = frame[["variant", *PARAM_COLUMNS, "objective", "annualized_return", "max_drawdown", "trade_count"]].copy()
    predictions["cv_pred_objective"] = pred
    predictions["fit_pred_objective"] = model.predict(x)
    return {
        "model": model,
        "importance": importance,
        "predictions": predictions,
        "cv_metrics": {
            "model": "ExtraTreesRegressor",
            "rows": int(len(frame)),
            "cv_r2": cv_r2,
            "objective_mean": float(y.mean()),
            "objective_std": float(y.std(ddof=1)),
        },
    }


def robust_candidates(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics.copy()
    frame["robust_score"] = (
        frame["objective"]
        + 0.20 * frame["annualized_return"].rank(pct=True)
        + 0.15 * (-frame["max_drawdown"].abs()).rank(pct=True)
        + 0.10 * (-frame["trade_count"]).rank(pct=True)
        + 0.10 * frame["mean_trade_return"].fillna(-1).rank(pct=True)
    )
    filtered = frame.loc[
        frame["trade_count"].between(80, 450)
        & frame["max_drawdown"].gt(-0.22)
        & frame["round_trips"].ge(60)
        & frame["top_month_return_share"].le(0.75)
    ].copy()
    if filtered.empty:
        filtered = frame.copy()
    cols = [
        "variant",
        *PARAM_COLUMNS,
        "annualized_return",
        "total_return",
        "max_drawdown",
        "sharpe",
        "trade_count",
        "mean_trade_return",
        "win_rate",
        "payoff_ratio",
        "worst_month_return",
        "top_month_return_share",
        "objective",
        "robust_score",
    ]
    return filtered.sort_values("robust_score", ascending=False)[cols].head(25)


def md_table(frame: pd.DataFrame, max_rows: int = 60) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(
    output: Path,
    metrics: pd.DataFrame,
    robust: pd.DataFrame,
    importance: pd.DataFrame,
    cv_metrics: dict[str, Any],
) -> None:
    top_cols = [
        "variant",
        *PARAM_COLUMNS,
        "annualized_return",
        "max_drawdown",
        "sharpe",
        "trade_count",
        "mean_trade_return",
        "worst_month_return",
        "top_month_return_share",
        "objective",
    ]
    lines = [
        "# Trade Parameter ML Response Surface",
        "",
        "## Scope",
        f"- Source tactical run: `{tq.SOURCE_RUN}`",
        f"- Model: `{tq.MODEL}`, score-band target `{tq.BAND_TARGET}`.",
        "- Parameters are sampled randomly; ML estimates the response surface and parameter sensitivity.",
        "- The selected candidates are robust-region candidates, not an OOS proof.",
        "",
        "## Response Surface CV",
        json.dumps(cv_metrics, ensure_ascii=False, indent=2),
        "",
        "## Parameter Importance",
        md_table(importance, 20),
        "",
        "## Top Objective Rows",
        md_table(metrics[top_cols], 25),
        "",
        "## Robust Candidates",
        md_table(robust, 25),
        "",
        "## Files",
        "- `sampled_trade_params.csv`",
        "- `param_search_metrics.csv`",
        "- `param_search_monthly.csv`",
        "- `param_search_trade_quality.csv`",
        "- `response_surface_feature_importance.csv`",
        "- `response_surface_predictions.csv`",
        "- `robust_parameter_candidates.csv`",
    ]
    (output / "trade_param_ml_surface_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
