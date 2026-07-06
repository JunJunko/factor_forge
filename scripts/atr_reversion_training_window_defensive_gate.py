"""Apply defensive gates to predictions from a training-window experiment."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository

from atr_reversion_defensive_gate import (
    _defensive_gate,
    _defensive_gate_soft,
    _fmt_pct,
    _risk_kill_only_gate,
    _rule_text,
)
from atr_reversion_pit_hmm_calibrated_backtest import _rank_states_from_validation, _tiered_weight
from atr_reversion_pit_regime_backtest import _run_regime_backtest_pit
from atr_reversion_small_portfolio_backtest import _json_default, _metrics
from atr_reversion_training_window_experiment import _yearly
from atr_reversion_walk_forward import FOLDS, REBALANCE_DAYS


TOP_N = 5
COSTS = [10, 20]
POLICIES = {
    "risk_kill_only": _risk_kill_only_gate,
    "defensive_gate": _defensive_gate,
    "defensive_gate_soft": _defensive_gate_soft,
}


def main(
    training_window_run: str = "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/training_window_experiment_20260706T130710Z",
    feature_dir: str = "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/three_layer_gate_20260706T113826Z",
    variant: str = "rolling_2y",
) -> None:
    train_path = Path(training_window_run)
    pit_run = train_path.parent
    feature_path = Path(feature_dir)
    output = pit_run / f"training_window_defensive_gate_{variant}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
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
    pit = pd.read_parquet(pit_run / "pit_liquidity_flags.parquet")
    pit["trade_date"] = pd.to_datetime(pit["trade_date"])
    log(f"loaded panel={len(panel):,} pit={len(pit):,} version={version}")
    log(f"using predictions from {train_path / variant}")

    rows: list[dict] = []
    yearly_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []

    for fold in FOLDS:
        fold_name = fold["name"]
        pred_path = train_path / variant / fold_name / "predictions_valid_test.parquet"
        pred = pd.read_parquet(pred_path)
        pred["trade_date"] = pd.to_datetime(pred["trade_date"])
        states = pd.read_csv(pit_run / "walk_forward_20260706T102017Z" / fold_name / "hmm_daily_states.csv")
        states["trade_date"] = pd.to_datetime(states["trade_date"])
        panel_bt = panel[
            panel["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["test_end"]))
        ].merge(pit, on=["trade_date", "ts_code"], how="left")
        panel_bt["pit_top1000"] = panel_bt["pit_top1000"].fillna(False).astype(bool)
        valid_panel = panel_bt[
            panel_bt["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["valid_end"]))
        ]
        test_panel = panel_bt[
            panel_bt["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))
        ]
        valid_pred = pred[pred["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["valid_end"]))]
        test_pred = pred[pred["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))]

        for cost in COSTS:
            valid_daily, _ = _run_regime_backtest_pit(
                valid_panel,
                valid_pred,
                states,
                top_n=TOP_N,
                rebalance_days=REBALANCE_DAYS,
                cost_bps=cost,
                policy=lambda _row: 1.0,
            )
            ranks, _ = _rank_states_from_validation(valid_daily, states)
            scores = pd.read_csv(feature_path / f"gate_scores_{fold_name}_cost{cost}.csv")
            scores["trade_date"] = pd.to_datetime(scores["trade_date"])
            scores = scores[scores["trade_date"].between(
                pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"])
            )].copy()

            for policy_name, gate_fn in POLICIES.items():
                scored = scores.copy()
                scored["variant"] = variant
                scored["fold"] = fold_name
                scored["cost_bps"] = cost
                scored["policy"] = policy_name
                scored["strategy_gate"] = scored.apply(gate_fn, axis=1).astype(float)
                scored.to_csv(output / f"gate_scores_{variant}_{fold_name}_{policy_name}_cost{cost}.csv", index=False, encoding="utf-8-sig")
                score_frames.append(scored)
                states_ext = states.merge(scored[["trade_date", "strategy_gate"]], on="trade_date", how="left")
                states_ext["strategy_gate"] = states_ext["strategy_gate"].fillna(1.0)
                policy: Callable[[pd.Series], float] = (
                    lambda row, ranks=ranks: _tiered_weight(row, ranks) * float(row.get("strategy_gate", 1.0))
                )
                daily, trades = _run_regime_backtest_pit(
                    test_panel,
                    test_pred,
                    states_ext,
                    top_n=TOP_N,
                    rebalance_days=REBALANCE_DAYS,
                    cost_bps=cost,
                    policy=policy,
                )
                tag = f"{variant}_{fold_name}_{policy_name}_top{TOP_N}_cost{cost}"
                daily.to_parquet(output / f"daily_{tag}.parquet", index=False)
                trades.to_parquet(output / f"trades_{tag}.parquet", index=False)
                metrics = _metrics(daily, trades)
                metrics.update({
                    "variant": variant,
                    "fold": fold_name,
                    "policy": policy_name,
                    "top_n": TOP_N,
                    "cost_bps": cost,
                    "avg_exposure": float(daily["exposure"].mean()),
                    "avg_daily_turnover": float(daily["turnover"].mean()),
                    "avg_strategy_gate": float(scored["strategy_gate"].mean()),
                    "flat_gate_ratio": float(scored["strategy_gate"].eq(0.0).mean()),
                    "half_gate_ratio": float(scored["strategy_gate"].eq(0.5).mean()),
                    "rule_text": _rule_text(policy_name),
                })
                rows.append(metrics)
                yearly_frames.append(_yearly(metrics, daily))
                log(
                    f"{tag} ann={metrics['annualized_return']:.2%} "
                    f"excess={metrics['annualized_excess_return']:.2%} "
                    f"sharpe={metrics['sharpe']:.2f} maxdd={metrics['max_drawdown']:.2%}"
                )

    metrics_df = pd.DataFrame(rows)
    yearly = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    scores_all = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    baseline = _load_training_baseline(train_path, variant)
    comparison = _compare_to_baseline(metrics_df, baseline)
    metrics_df.to_csv(output / "training_window_defensive_gate_metrics.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "training_window_defensive_gate_yearly.csv", index=False, encoding="utf-8-sig")
    scores_all.to_csv(output / "training_window_defensive_gate_scores.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "training_window_defensive_gate_comparison.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "data_version": version,
                "training_window_run": str(train_path),
                "feature_dir": str(feature_path),
                "variant": variant,
                "run_dir": str(output),
                "metrics": metrics_df.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(metrics_df, comparison, yearly, variant), encoding="utf-8")
    log("wrote training-window defensive gate report")
    log("done")
    print(f"run_dir={output}")


def _load_panel() -> tuple[str, pd.DataFrame]:
    project = load_project("configs/project.yaml")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest("latest")
    _, panel = repo.load_panel(version)
    return version, panel


def _load_training_baseline(train_path: Path, variant: str) -> pd.DataFrame:
    metrics = pd.read_csv(train_path / "training_window_metrics.csv")
    return metrics[
        metrics["variant"].eq(variant)
        & metrics["policy"].eq("atr_hmm_tiered")
        & metrics["top_n"].eq(TOP_N)
        & metrics["cost_bps"].isin(COSTS)
    ].copy()


def _compare_to_baseline(metrics: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    left = baseline[[
        "variant",
        "fold",
        "cost_bps",
        "annualized_return",
        "annualized_excess_return",
        "sharpe",
        "max_drawdown",
        "avg_exposure",
    ]].rename(columns={
        "annualized_return": "base_ann",
        "annualized_excess_return": "base_excess",
        "sharpe": "base_sharpe",
        "max_drawdown": "base_maxdd",
        "avg_exposure": "base_exposure",
    })
    right = metrics[[
        "variant",
        "fold",
        "cost_bps",
        "policy",
        "annualized_return",
        "annualized_excess_return",
        "sharpe",
        "max_drawdown",
        "avg_exposure",
    ]].rename(columns={
        "annualized_return": "gate_ann",
        "annualized_excess_return": "gate_excess",
        "sharpe": "gate_sharpe",
        "max_drawdown": "gate_maxdd",
        "avg_exposure": "gate_exposure",
    })
    out = left.merge(right, on=["variant", "fold", "cost_bps"], how="inner")
    out["ann_delta"] = out["gate_ann"] - out["base_ann"]
    out["excess_delta"] = out["gate_excess"] - out["base_excess"]
    out["sharpe_delta"] = out["gate_sharpe"] - out["base_sharpe"]
    out["maxdd_delta"] = out["gate_maxdd"] - out["base_maxdd"]
    return out


def _report(metrics: pd.DataFrame, comparison: pd.DataFrame, yearly: pd.DataFrame, variant: str) -> str:
    show = metrics[[
        "fold",
        "policy",
        "cost_bps",
        "annualized_return",
        "annualized_excess_return",
        "sharpe",
        "max_drawdown",
        "avg_exposure",
        "avg_strategy_gate",
        "flat_gate_ratio",
    ]].copy()
    show = _fmt_pct(show, [
        "annualized_return",
        "annualized_excess_return",
        "max_drawdown",
        "avg_exposure",
        "avg_strategy_gate",
        "flat_gate_ratio",
    ])
    show["sharpe"] = show["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    comp = comparison.copy()
    comp = comp[[
        "fold",
        "cost_bps",
        "policy",
        "base_ann",
        "gate_ann",
        "ann_delta",
        "base_excess",
        "gate_excess",
        "excess_delta",
        "base_maxdd",
        "gate_maxdd",
        "gate_exposure",
    ]]
    comp = _fmt_pct(comp, [
        "base_ann",
        "gate_ann",
        "ann_delta",
        "base_excess",
        "gate_excess",
        "excess_delta",
        "base_maxdd",
        "gate_maxdd",
        "gate_exposure",
    ])
    y = yearly.copy()
    y = y[["variant", "fold", "policy", "cost_bps", "year", "return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"]]
    y = _fmt_pct(y, ["return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"])
    return "\n".join([
        f"# ATR {variant} + Defensive Gate",
        "",
        "Predictions are reused from the training-window experiment; only defensive exposure gates are added.",
        "",
        "## Metrics",
        "",
        show.to_markdown(index=False),
        "",
        "## Versus Training-Window Baseline",
        "",
        comp.to_markdown(index=False),
        "",
        "## Yearly",
        "",
        y.to_markdown(index=False),
        "",
    ])


if __name__ == "__main__":
    training_run = sys.argv[1] if len(sys.argv) > 1 else "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/training_window_experiment_20260706T130710Z"
    feature_dir = sys.argv[2] if len(sys.argv) > 2 else "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/three_layer_gate_20260706T113826Z"
    variant = sys.argv[3] if len(sys.argv) > 3 else "rolling_2y"
    main(training_run, feature_dir, variant)
