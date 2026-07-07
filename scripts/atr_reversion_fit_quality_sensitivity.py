"""Parameter sensitivity for the frozen fit-quality flip rule."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository

from atr_reversion_fit_quality_gate import (
    BASELINE_RUN,
    HMM_VARIANT,
    PIT_RUN,
    SOURCE_RUN,
    TOP_N,
    _apply_score_direction,
    _daily_fit_metrics,
    _load_baseline_metrics,
    _report,
    _yearly,
)
from atr_reversion_pit_hmm_calibrated_backtest import _rank_states_from_validation, _tiered_weight
from atr_reversion_pit_regime_backtest import _run_regime_backtest_pit
from atr_reversion_small_portfolio_backtest import _json_default, _metrics
from atr_reversion_walk_forward import FOLDS, REBALANCE_DAYS


LOOKBACKS = [20, 30, 40, 60, 80]
MIN_OBS_VALUES = [10, 15, 20, 30]
COST = 20
POLICY = "fit_quality_flip_only"


def main(
    source_run: str = str(SOURCE_RUN),
    baseline_run: str = str(BASELINE_RUN),
) -> None:
    source = Path(source_run)
    baseline = Path(baseline_run)
    output = PIT_RUN / f"fit_quality_sensitivity_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    version, panel = _load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    pit = pd.read_parquet(PIT_RUN / "pit_liquidity_flags.parquet")
    pit["trade_date"] = pd.to_datetime(pit["trade_date"])
    log(f"loaded panel={len(panel):,} pit={len(pit):,} version={version}")

    prepared = {}
    for fold in FOLDS:
        fold_name = fold["name"]
        pred = pd.read_parquet(source / fold_name / "predictions_valid_test_rolling_2y.parquet")
        pred["trade_date"] = pd.to_datetime(pred["trade_date"])
        states = pd.read_csv(source / fold_name / HMM_VARIANT / "hmm_daily_states.csv")
        states["trade_date"] = pd.to_datetime(states["trade_date"])
        fit_daily = _daily_fit_metrics(panel, pit, pred, fold, log)
        panel_bt = panel[
            panel["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["test_end"]))
        ].merge(pit, on=["trade_date", "ts_code"], how="left")
        panel_bt["pit_top1000"] = panel_bt["pit_top1000"].fillna(False).astype(bool)
        prepared[fold_name] = {
            "fold": fold,
            "pred": pred,
            "states": states,
            "fit_daily": fit_daily,
            "valid_panel": panel_bt[
                panel_bt["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["valid_end"]))
            ],
            "test_panel": panel_bt[
                panel_bt["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))
            ],
        }

    rows: list[dict] = []
    yearly_frames: list[pd.DataFrame] = []
    gate_frames: list[pd.DataFrame] = []
    for lookback in LOOKBACKS:
        for min_obs in MIN_OBS_VALUES:
            log(f"sensitivity lookback={lookback} min_obs={min_obs}")
            for fold_name, item in prepared.items():
                fold = item["fold"]
                pred = item["pred"]
                states = item["states"]
                controls = _rolling_controls(states["trade_date"], item["fit_daily"], lookback, min_obs)
                controls["fold"] = fold_name
                controls["lookback"] = lookback
                controls["min_obs"] = min_obs
                controls["cost_bps"] = COST
                controls["policy"] = POLICY
                gate_frames.append(controls)
                adjusted_pred = _apply_score_direction(pred, controls)
                valid_pred = adjusted_pred[
                    adjusted_pred["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["valid_end"]))
                ]
                test_pred = adjusted_pred[
                    adjusted_pred["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))
                ]
                states_ext = states.merge(controls[["trade_date", "strategy_gate"]], on="trade_date", how="left")
                states_ext["strategy_gate"] = states_ext["strategy_gate"].fillna(1.0)
                valid_daily, _ = _run_regime_backtest_pit(
                    item["valid_panel"],
                    valid_pred,
                    states_ext,
                    top_n=TOP_N,
                    rebalance_days=REBALANCE_DAYS,
                    cost_bps=COST,
                    policy=lambda row: float(row.get("strategy_gate", 1.0)),
                )
                ranks, _ = _rank_states_from_validation(valid_daily, states_ext)
                daily, trades = _run_regime_backtest_pit(
                    item["test_panel"],
                    test_pred,
                    states_ext,
                    top_n=TOP_N,
                    rebalance_days=REBALANCE_DAYS,
                    cost_bps=COST,
                    policy=lambda row, ranks=ranks: _tiered_weight(row, ranks) * float(row.get("strategy_gate", 1.0)),
                )
                tag = f"lookback{lookback}_minobs{min_obs}_{fold_name}_{POLICY}_top{TOP_N}_cost{COST}"
                daily.to_parquet(output / f"daily_{tag}.parquet", index=False)
                trades.to_parquet(output / f"trades_{tag}.parquet", index=False)
                metrics = _metrics(daily, trades)
                test_controls = controls[
                    controls["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))
                ]
                row = {
                    "lookback": lookback,
                    "min_obs": min_obs,
                    "fold": fold_name,
                    "policy": POLICY,
                    "top_n": TOP_N,
                    "cost_bps": COST,
                    "hmm_variant": HMM_VARIANT,
                    "avg_exposure": float(daily["exposure"].mean()),
                    "avg_daily_turnover": float(daily["turnover"].mean()),
                    "avg_strategy_gate": float(test_controls["strategy_gate"].mean()),
                    "flip_ratio": float(test_controls["score_direction"].lt(0.0).mean()),
                    **metrics,
                }
                rows.append(row)
                y = _yearly(row, daily)
                y["lookback"] = lookback
                y["min_obs"] = min_obs
                yearly_frames.append(y)
                log(
                    f"{tag} ann={metrics['annualized_return']:.2%} "
                    f"excess={metrics['annualized_excess_return']:.2%} "
                    f"flip={row['flip_ratio']:.1%}"
                )

    metrics_df = pd.DataFrame(rows)
    yearly_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    gates_df = pd.concat(gate_frames, ignore_index=True) if gate_frames else pd.DataFrame()
    summary = _summary(metrics_df)
    baseline_df = _load_baseline_metrics(baseline)
    baseline_df = baseline_df[baseline_df["cost_bps"].eq(COST)]
    comparison = _comparison(metrics_df, baseline_df)

    metrics_df.to_csv(output / "sensitivity_metrics.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(output / "sensitivity_yearly.csv", index=False, encoding="utf-8-sig")
    gates_df.to_csv(output / "sensitivity_gate_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "sensitivity_comparison.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "source_run": str(source),
                "baseline_run": str(baseline),
                "run_dir": str(output),
                "lookbacks": LOOKBACKS,
                "min_obs_values": MIN_OBS_VALUES,
                "cost_bps": COST,
                "policy": POLICY,
                "metrics": metrics_df.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_sensitivity_report(summary, comparison, yearly_df), encoding="utf-8")
    log("done")
    print(f"run_dir={output}")


def _load_panel() -> tuple[str, pd.DataFrame]:
    project = load_project("configs/project.yaml")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest("latest")
    _, panel = repo.load_panel(version)
    return version, panel


def _rolling_controls(
    state_dates: pd.Series,
    fit_daily: pd.DataFrame,
    lookback: int,
    min_obs: int,
) -> pd.DataFrame:
    fit = fit_daily.sort_values("known_date").reset_index(drop=True)
    rows = []
    for date in pd.to_datetime(state_dates).sort_values():
        hist = fit[fit["known_date"].le(date)].tail(lookback)
        rank_ic = float(hist["rank_ic"].mean()) if len(hist) else np.nan
        decile_spread = float(hist["decile_spread"].mean()) if len(hist) else np.nan
        flip = len(hist) >= min_obs and rank_ic < 0.0 and decile_spread < 0.0
        rows.append(
            {
                "trade_date": date,
                "fit_obs": int(len(hist)),
                "rank_ic_rolling": rank_ic,
                "decile_spread_rolling": decile_spread,
                "top5_excess_rolling": float(hist["top5_excess_forward_return"].mean()) if len(hist) else np.nan,
                "top5_hit_rolling": float(hist["top5_hit_rate"].mean()) if len(hist) else np.nan,
                "strategy_gate": 1.0,
                "score_direction": -1.0 if flip else 1.0,
            }
        )
    return pd.DataFrame(rows)


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["lookback", "min_obs", "cost_bps"])
        .agg(
            mean_ann=("annualized_return", "mean"),
            median_ann=("annualized_return", "median"),
            mean_excess=("annualized_excess_return", "mean"),
            median_excess=("annualized_excess_return", "median"),
            positive_excess_folds=("annualized_excess_return", lambda s: int((s > 0.0).sum())),
            mean_sharpe=("sharpe", "mean"),
            worst_drawdown=("max_drawdown", "min"),
            mean_exposure=("avg_exposure", "mean"),
            mean_flip_ratio=("flip_ratio", "mean"),
        )
        .reset_index()
        .sort_values(["mean_excess", "mean_sharpe"], ascending=False)
    )


def _comparison(metrics: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    base = baseline[[
        "fold",
        "cost_bps",
        "annualized_return",
        "annualized_excess_return",
        "max_drawdown",
        "avg_exposure",
    ]].rename(
        columns={
            "annualized_return": "base_ann",
            "annualized_excess_return": "base_excess",
            "max_drawdown": "base_max_drawdown",
            "avg_exposure": "base_exposure",
        }
    )
    out = metrics.merge(base, on=["fold", "cost_bps"], how="left")
    out["delta_ann"] = out["annualized_return"] - out["base_ann"]
    out["delta_excess"] = out["annualized_excess_return"] - out["base_excess"]
    out["delta_drawdown"] = out["max_drawdown"] - out["base_max_drawdown"]
    return out


def _fmt_pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    return out


def _sensitivity_report(summary: pd.DataFrame, comparison: pd.DataFrame, yearly: pd.DataFrame) -> str:
    summ = _fmt_pct(
        summary,
        [
            "mean_ann",
            "median_ann",
            "mean_excess",
            "median_excess",
            "worst_drawdown",
            "mean_exposure",
            "mean_flip_ratio",
        ],
    )
    summ["mean_sharpe"] = summ["mean_sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    comp = comparison[
        ["lookback", "min_obs", "fold", "annualized_return", "base_ann", "annualized_excess_return", "base_excess", "max_drawdown", "base_max_drawdown", "flip_ratio"]
    ].copy()
    comp = _fmt_pct(
        comp,
        [
            "annualized_return",
            "base_ann",
            "annualized_excess_return",
            "base_excess",
            "max_drawdown",
            "base_max_drawdown",
            "flip_ratio",
        ],
    )
    y = yearly[["lookback", "min_obs", "year", "return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"]].copy()
    y = _fmt_pct(y, ["return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"])
    return "\n".join(
        [
            "# Fit-Quality Flip Sensitivity",
            "",
            "Rule family: flip score direction when completed rolling RankIC < 0 and completed rolling decile_spread < 0.",
            "",
            "## Parameter Summary",
            "",
            summ.to_markdown(index=False),
            "",
            "## Fold Comparison",
            "",
            comp.to_markdown(index=False),
            "",
            "## Yearly",
            "",
            y.to_markdown(index=False),
            "",
        ]
    )


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else str(SOURCE_RUN)
    baseline = sys.argv[2] if len(sys.argv) > 2 else str(BASELINE_RUN)
    main(source, baseline)
