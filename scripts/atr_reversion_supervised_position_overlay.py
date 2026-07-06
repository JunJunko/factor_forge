"""Supervised position overlay for ATR reversion.

The stock-selection model and risk gates stay fixed.  This script trains small
OOS supervised models on rebalance-cycle samples and maps predictions to an
exposure multiplier in {0, 25%, 50%, 75%, 100%}.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from atr_reversion_ppo_position_overlay import (
    ACTIONS,
    BASE_DIR,
    FEATURE_COLUMNS,
    INITIAL_CASH,
    REBALANCE_DAYS,
    TEST_FOLDS,
    TOP_N,
    TRAIN_FOLDS,
    _comparison,
    _cycles_for_fold,
    _load_fold,
    _report as _ppo_report,
    _yearly_rows,
)
from atr_reversion_small_portfolio_backtest import _json_default, _metrics


OUTPUT_ROOT = Path("artifacts/atr_reversion_supervised_overlay")


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).copy()
    return x


def _prob_to_multiplier(prob: np.ndarray) -> np.ndarray:
    return np.select(
        [
            prob < 0.45,
            prob < 0.50,
            prob < 0.55,
            prob < 0.60,
        ],
        [0.0, 0.25, 0.50, 0.75],
        default=1.0,
    )


def _expected_to_multiplier(expected: np.ndarray) -> np.ndarray:
    return np.select(
        [
            expected <= -0.005,
            expected <= 0.000,
            expected <= 0.005,
            expected <= 0.015,
        ],
        [0.0, 0.25, 0.50, 0.75],
        default=1.0,
    )


def _fit_predict_logistic(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    y = (train["base_cycle_return"] > 0.0).astype(int)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.35, class_weight="balanced", max_iter=2000, random_state=42),
    )
    model.fit(_feature_frame(train), y)
    prob = model.predict_proba(_feature_frame(test))[:, 1]
    out = test.copy()
    out["supervised_score"] = prob
    out["supervised_multiplier"] = _prob_to_multiplier(prob)
    out["supervised_model"] = "logistic_win_prob"
    return out


def _fit_predict_logistic_veto(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    out = _fit_predict_logistic(train, test)
    prob = out["supervised_score"].to_numpy(float)
    out["supervised_multiplier"] = np.select(
        [prob < 0.35, prob < 0.45],
        [0.0, 0.50],
        default=1.0,
    )
    out["supervised_model"] = "logistic_veto"
    return out


def _fit_predict_ridge_ev(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    y = train["base_cycle_return"].clip(-0.12, 0.12)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RidgeCV(alphas=np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])),
    )
    model.fit(_feature_frame(train), y)
    pred = model.predict(_feature_frame(test))
    out = test.copy()
    out["supervised_score"] = pred
    out["supervised_multiplier"] = _expected_to_multiplier(pred)
    out["supervised_model"] = "ridge_expected_return"
    return out


def _fit_predict_ridge_veto(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    out = _fit_predict_ridge_ev(train, test)
    pred = out["supervised_score"].to_numpy(float)
    out["supervised_multiplier"] = np.select(
        [pred < -0.020, pred < -0.005],
        [0.0, 0.50],
        default=1.0,
    )
    out["supervised_model"] = "ridge_veto"
    return out


def _fit_predict_hgb_ev(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    # Tiny tree ensemble with strong regularization.  It is included as a
    # nonlinear supervised baseline, not as an optimized production model.
    y = train["base_cycle_return"].clip(-0.12, 0.12)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=40,
        learning_rate=0.03,
        max_leaf_nodes=5,
        l2_regularization=0.2,
        random_state=42,
    )
    model.fit(_feature_frame(train), y)
    pred = model.predict(_feature_frame(test))
    out = test.copy()
    out["supervised_score"] = pred
    out["supervised_multiplier"] = _expected_to_multiplier(pred)
    out["supervised_model"] = "hgb_expected_return"
    return out


def _daily_with_supervised_overlay(fold: str, cost: int, cycles: pd.DataFrame) -> pd.DataFrame:
    daily, _ = _load_fold(fold, cost)
    out = daily.copy()
    out["base_return"] = out["return"]
    out["base_nav"] = out["nav"]
    out["supervised_multiplier"] = 1.0
    out["return"] = 0.0
    for _, cycle in cycles.iterrows():
        mask = out["trade_date"].between(pd.Timestamp(cycle["start_date"]), pd.Timestamp(cycle["end_date"]))
        multiplier = float(cycle["supervised_multiplier"])
        out.loc[mask, "supervised_multiplier"] = multiplier
        out.loc[mask, "return"] = out.loc[mask, "base_return"].fillna(0.0) * multiplier
    out["nav"] = INITIAL_CASH * (1.0 + out["return"].fillna(0.0)).cumprod()
    out["excess_return"] = out["return"] - out["benchmark_return"]
    out["exposure"] = out["exposure"].fillna(0.0) * out["supervised_multiplier"]
    out["turnover"] = out["turnover"].fillna(0.0) * out["supervised_multiplier"]
    out["transaction_cost"] = out["transaction_cost"].fillna(0.0) * out["supervised_multiplier"]
    out["cash_ratio"] = 1.0 - out["exposure"].clip(0.0, 1.0)
    return out


def _model_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    base = metrics[metrics["policy"].eq("risk_kill_only")].copy()
    overlays = metrics[metrics["policy"].ne("risk_kill_only")].copy()
    keep = ["fold", "cost_bps", "annualized_return", "annualized_excess_return", "sharpe", "max_drawdown", "avg_exposure"]
    left = base[keep].rename(columns={
        "annualized_return": "base_ann",
        "annualized_excess_return": "base_excess",
        "sharpe": "base_sharpe",
        "max_drawdown": "base_maxdd",
        "avg_exposure": "base_exposure",
    })
    right = overlays[["policy", *keep]].rename(columns={
        "annualized_return": "model_ann",
        "annualized_excess_return": "model_excess",
        "sharpe": "model_sharpe",
        "max_drawdown": "model_maxdd",
        "avg_exposure": "model_exposure",
    })
    out = left.merge(right, on=["fold", "cost_bps"], how="inner")
    out["ann_delta"] = out["model_ann"] - out["base_ann"]
    out["excess_delta"] = out["model_excess"] - out["base_excess"]
    out["maxdd_delta"] = out["model_maxdd"] - out["base_maxdd"]
    out["exposure_delta"] = out["model_exposure"] - out["base_exposure"]
    return out


def _report(comparison: pd.DataFrame, yearly: pd.DataFrame, actions: pd.DataFrame) -> str:
    comp = comparison.copy()
    for col in [c for c in comp.columns if c.endswith(("ann", "excess", "maxdd", "exposure", "delta"))]:
        comp[col] = comp[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    for col in ["base_sharpe", "model_sharpe"]:
        comp[col] = comp[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    y = yearly[yearly["year"].isin([2025, 2026])].copy()
    for col in ["return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"]:
        y[col] = y[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    a = actions.groupby(["policy", "fold", "cost_bps", "supervised_multiplier"]).size().reset_index(name="cycles")
    return "\n".join([
        "# Supervised Position Overlay",
        "",
        "## OOS Comparison",
        "",
        comp.to_markdown(index=False),
        "",
        "## Yearly 2025/2026",
        "",
        y.to_markdown(index=False),
        "",
        "## Action Counts",
        "",
        a.to_markdown(index=False),
        "",
    ])


def main() -> None:
    output = OUTPUT_ROOT / f"supervised_position_overlay_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    all_cycles = []
    for cost in [10, 20]:
        for fold in ["test_2022", "test_2023", "test_2024", "test_2025", "test_2026h1"]:
            c = _cycles_for_fold(fold, cost)
            all_cycles.append(c)
            log(f"loaded cycles fold={fold} cost={cost} cycles={len(c)}")
    cycles_all = pd.concat(all_cycles, ignore_index=True)
    cycles_all.to_csv(output / "cycle_dataset.csv", index=False, encoding="utf-8-sig")

    model_fns = {
        "logistic_win_prob": _fit_predict_logistic,
        "logistic_veto": _fit_predict_logistic_veto,
        "ridge_expected_return": _fit_predict_ridge_ev,
        "ridge_veto": _fit_predict_ridge_veto,
        "hgb_expected_return": _fit_predict_hgb_ev,
    }
    metrics_rows = []
    yearly_rows = []
    action_frames = []
    for cost in [10, 20]:
        for test_fold in TEST_FOLDS:
            train = cycles_all[
                cycles_all["cost_bps"].eq(cost) & cycles_all["fold"].isin(TRAIN_FOLDS[test_fold])
            ].copy()
            test = cycles_all[
                cycles_all["cost_bps"].eq(cost) & cycles_all["fold"].eq(test_fold)
            ].copy()
            base_daily, _ = _load_fold(test_fold, cost)
            base_metrics = _metrics(base_daily, pd.DataFrame())
            base_metrics.update({
                "policy": "risk_kill_only",
                "fold": test_fold,
                "cost_bps": cost,
                "top_n": TOP_N,
                "rebalance_days": REBALANCE_DAYS,
                "avg_exposure": float(base_daily["exposure"].mean()),
                "avg_daily_turnover": float(base_daily["turnover"].mean()),
            })
            metrics_rows.append(base_metrics)
            yearly_rows.extend(_yearly_rows(base_daily, "risk_kill_only", test_fold, cost))
            log(f"fit supervised models cost={cost} test={test_fold} train_cycles={len(train)} test_cycles={len(test)}")

            for model_name, fn in model_fns.items():
                acted = fn(train, test)
                acted["policy"] = model_name
                acted.to_csv(output / f"actions_{model_name}_{test_fold}_cost{cost}.csv", index=False, encoding="utf-8-sig")
                action_frames.append(acted)
                daily = _daily_with_supervised_overlay(test_fold, cost, acted)
                daily.to_parquet(output / f"daily_{model_name}_{test_fold}_top{TOP_N}_cost{cost}.parquet", index=False)
                metrics = _metrics(daily, pd.DataFrame())
                metrics.update({
                    "policy": model_name,
                    "fold": test_fold,
                    "cost_bps": cost,
                    "top_n": TOP_N,
                    "rebalance_days": REBALANCE_DAYS,
                    "avg_exposure": float(daily["exposure"].mean()),
                    "avg_daily_turnover": float(daily["turnover"].mean()),
                })
                metrics_rows.append(metrics)
                yearly_rows.extend(_yearly_rows(daily, model_name, test_fold, cost))
                log(
                    f"{model_name} {test_fold} cost={cost} "
                    f"ann={metrics['annualized_return']:.2%} avg_exposure={daily['exposure'].mean():.2%}"
                )

    metrics_df = pd.DataFrame(metrics_rows)
    yearly_df = pd.DataFrame(yearly_rows)
    actions_df = pd.concat(action_frames, ignore_index=True)
    comparison = _model_comparison(metrics_df)
    metrics_df.to_csv(output / "supervised_overlay_metrics.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(output / "supervised_overlay_yearly.csv", index=False, encoding="utf-8-sig")
    actions_df.to_csv(output / "supervised_overlay_actions.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "supervised_overlay_comparison.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "base_dir": str(BASE_DIR),
                "run_dir": str(output),
                "actions": ACTIONS.tolist(),
                "test_folds": TEST_FOLDS,
                "train_folds": TRAIN_FOLDS,
                "models": list(model_fns),
                "metrics": metrics_df.to_dict("records"),
                "comparison": comparison.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(comparison, yearly_df, actions_df), encoding="utf-8")
    log(f"done output={output}")
    print(f"run_dir={output}")


if __name__ == "__main__":
    main()
