"""Defensive strategy-aware gate for ATR lower-shadow reversion.

This is a less aggressive successor to ``atr_reversion_three_layer_gate.py``.
It keeps ATR-HMM tiered exposure as the base exposure and only applies:

1. Extreme risk kill: flat in 2026-like high-dispersion strong-market regimes.
2. Health + mainline mismatch de-risk: half exposure when the strategy is
   recently weak and selected names do not fit the market mainline.

Mainline mismatch alone is not allowed to kill exposure.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository

from atr_reversion_pit_hmm_calibrated_backtest import _rank_states_from_validation, _tiered_weight
from atr_reversion_pit_regime_backtest import _run_regime_backtest_pit
from atr_reversion_small_portfolio_backtest import _json_default, _metrics
from atr_reversion_strategy_regime_mining import _compare
from atr_reversion_three_layer_gate import _load_baseline, _yearly
from atr_reversion_walk_forward import FOLDS, REBALANCE_DAYS


TOP_N = 5
COSTS = [10, 20]
DEFAULT_FEATURE_DIR = (
    "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/"
    "three_layer_gate_20260706T113826Z"
)


def main(
    walk_forward_dir: str = "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/walk_forward_20260706T102017Z",
    feature_dir: str = DEFAULT_FEATURE_DIR,
) -> None:
    wf_path = Path(walk_forward_dir)
    feature_path = Path(feature_dir)
    pit_run = wf_path.parent
    output = pit_run / f"defensive_gate_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
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
    log(f"using precomputed three-layer features from {feature_path}")

    rows: list[dict] = []
    yearly_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []

    policies = {
        "risk_kill_only": _risk_kill_only_gate,
        "defensive_gate": _defensive_gate,
        "defensive_gate_soft": _defensive_gate_soft,
    }

    for fold in FOLDS:
        fold_name = fold["name"]
        fold_dir = wf_path / fold_name
        pred = pd.read_parquet(fold_dir / "predictions_valid_test.parquet")
        pred["trade_date"] = pd.to_datetime(pred["trade_date"])
        states = pd.read_csv(fold_dir / "hmm_daily_states.csv")
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
            log(f"{fold_name} cost={cost}: calibrating ATR-HMM tiered base")
            valid_ungated, _ = _run_regime_backtest_pit(
                valid_panel,
                valid_pred,
                states,
                top_n=TOP_N,
                rebalance_days=REBALANCE_DAYS,
                cost_bps=cost,
                policy=lambda _row: 1.0,
            )
            ranks, _ = _rank_states_from_validation(valid_ungated, states)

            scores = pd.read_csv(feature_path / f"gate_scores_{fold_name}_cost{cost}.csv")
            scores["trade_date"] = pd.to_datetime(scores["trade_date"])
            test_scores = scores[scores["trade_date"].between(
                pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"])
            )].copy()

            for policy_name, gate_fn in policies.items():
                scored = _apply_gate_columns(test_scores, gate_fn, fold_name, cost, policy_name)
                scored.to_csv(output / f"gate_scores_{fold_name}_{policy_name}_cost{cost}.csv", index=False, encoding="utf-8-sig")
                score_frames.append(scored)
                states_ext = states.merge(
                    scored[["trade_date", "strategy_gate"]],
                    on="trade_date",
                    how="left",
                )
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
                tag = f"{fold_name}_{policy_name}_top{TOP_N}_cost{cost}"
                daily.to_parquet(output / f"daily_{tag}.parquet", index=False)
                trades.to_parquet(output / f"trades_{tag}.parquet", index=False)
                metrics = _metrics(daily, trades)
                metrics.update({
                    "fold": fold_name,
                    "policy": policy_name,
                    "top_n": TOP_N,
                    "cost_bps": cost,
                    "avg_exposure": float(daily["exposure"].mean()),
                    "avg_daily_turnover": float(daily["turnover"].mean()),
                    "avg_strategy_gate": float(scored["strategy_gate"].mean()),
                    "flat_gate_ratio": float(scored["strategy_gate"].eq(0.0).mean()),
                    "half_gate_ratio": float(scored["strategy_gate"].eq(0.5).mean()),
                    "risk_kill_ratio": float(scored["extreme_risk_kill"].mean()),
                    "derisk_ratio": float(scored["health_mainline_derisk"].mean()),
                    "rule_text": _rule_text(policy_name),
                })
                rows.append(metrics)
                yearly_frames.append(_yearly(metrics, daily))
                log(
                    f"{tag} ann={metrics['annualized_return']:.2%} "
                    f"excess={metrics['annualized_excess_return']:.2%} "
                    f"sharpe={metrics['sharpe']:.2f} maxdd={metrics['max_drawdown']:.2%} "
                    f"avg_gate={metrics['avg_strategy_gate']:.1%}"
                )

    metrics_df = pd.DataFrame(rows)
    yearly = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    scores_all = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    baseline = _load_baseline(wf_path)
    comparison = _compare(metrics_df, baseline)
    metrics_df.to_csv(output / "defensive_gate_metrics.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "defensive_gate_yearly.csv", index=False, encoding="utf-8-sig")
    scores_all.to_csv(output / "defensive_gate_scores.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "walk_forward_comparison.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "data_version": version,
                "walk_forward_dir": str(wf_path),
                "feature_dir": str(feature_path),
                "run_dir": str(output),
                "metrics": metrics_df.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(metrics_df, comparison, yearly), encoding="utf-8")
    log("wrote defensive gate report")
    log("done")
    print(f"run_dir={output}")


def _load_panel() -> tuple[str, pd.DataFrame]:
    project = load_project("configs/project.yaml")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest("latest")
    _, panel = repo.load_panel(version)
    return version, panel


def _risk_kill_only_gate(row: pd.Series) -> float:
    return 0.0 if _extreme_risk_kill(row) else 1.0


def _defensive_gate(row: pd.Series) -> float:
    if _extreme_risk_kill(row) or (_health_weak(row) and float(row.get("gate_score", 1.0)) <= 0.0):
        return 0.0
    if _health_weak(row):
        return 0.5
    return 1.0


def _defensive_gate_soft(row: pd.Series) -> float:
    if _extreme_risk_kill(row) or (_health_weak(row) and float(row.get("gate_score", 1.0)) <= 0.0):
        return 0.0
    if _health_weak(row):
        return 0.7
    return 1.0


def _extreme_risk_kill(row: pd.Series) -> bool:
    xsec_vol = float(row.get("xsec_vol_20", np.nan))
    market_ret_20 = float(row.get("market_ret_20", np.nan))
    market_ret_60 = float(row.get("market_ret_60", np.nan))
    momentum_minus_reversal = float(row.get("momentum_minus_reversal_20", np.nan))
    signal_health = float(row.get("top5_excess_5round", 0.0))
    return bool(
        np.isfinite(xsec_vol)
        and xsec_vol >= 0.036
        and (market_ret_20 >= 0.02 or market_ret_60 >= 0.06)
        and momentum_minus_reversal > 0.0
        and signal_health < 0.0
    )


def _health_mainline_derisk(row: pd.Series) -> bool:
    return _health_weak(row) and _mainline_mismatch(row)


def _health_reversal_derisk(row: pd.Series) -> bool:
    return _health_weak(row) and _reversal_unrewarded(row)


def _health_weak(row: pd.Series) -> bool:
    excess = float(row.get("top5_excess_5round", 0.0))
    winrate = float(row.get("top5_winrate_5round", 0.5))
    return bool(excess < 0.0 or winrate < 0.4)


def _mainline_mismatch(row: pd.Series) -> bool:
    strong_market = (
        float(row.get("market_ret_20", 0.0)) > 0.02
        or float(row.get("market_ret_60", 0.0)) > 0.06
        or float(row.get("market_breadth_20", 0.0)) > 0.49
    )
    selected_weak = (
        float(row.get("selected_momentum_rank", 0.5)) < 0.45
        and float(row.get("selected_industry_strength", 0.5)) < 0.45
        and float(row.get("selected_hot_industry_ratio", 0.0)) < 0.20
    )
    return bool(strong_market and selected_weak)


def _reversal_unrewarded(row: pd.Series) -> bool:
    reversal = float(row.get("reversal_strength_20", 0.0))
    lower_shadow = float(row.get("lower_shadow_style_20", 0.0))
    core = float(row.get("core_signal_style_20", 0.0))
    momentum_minus_reversal = float(row.get("momentum_minus_reversal_20", 0.0))
    return bool((reversal < 0.0 and lower_shadow < 0.0) or (core < 0.0 and momentum_minus_reversal > 0.0))


def _apply_gate_columns(
    scores: pd.DataFrame,
    gate_fn: Callable[[pd.Series], float],
    fold: str,
    cost: int,
    policy: str,
) -> pd.DataFrame:
    out = scores.copy()
    out["fold"] = fold
    out["cost_bps"] = cost
    out["policy"] = policy
    out["extreme_risk_kill"] = out.apply(_extreme_risk_kill, axis=1)
    out["health_weak"] = out.apply(_health_weak, axis=1)
    out["mainline_mismatch"] = out.apply(_mainline_mismatch, axis=1)
    out["reversal_unrewarded"] = out.apply(_reversal_unrewarded, axis=1)
    out["health_mainline_derisk"] = out.apply(_health_mainline_derisk, axis=1)
    out["health_reversal_derisk"] = out.apply(_health_reversal_derisk, axis=1)
    out["strategy_gate"] = out.apply(gate_fn, axis=1).astype(float)
    return out


def _rule_text(policy: str) -> str:
    if policy == "risk_kill_only":
        return "flat only on extreme high-dispersion strong-market risk"
    if policy == "defensive_gate_soft":
        return "flat on extreme risk or weak health with nonpositive gate_score; 70% gate on weak health"
    return "flat on extreme risk or weak health with nonpositive gate_score; 50% gate on weak health"


def _fmt_pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    return out


def _report(metrics: pd.DataFrame, comparison: pd.DataFrame, yearly: pd.DataFrame) -> str:
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
        "risk_kill_ratio",
        "derisk_ratio",
    ]].copy()
    show = _fmt_pct(show, [
        "annualized_return",
        "annualized_excess_return",
        "max_drawdown",
        "avg_exposure",
        "avg_strategy_gate",
        "risk_kill_ratio",
        "derisk_ratio",
    ])
    show["sharpe"] = show["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    comp = comparison.copy()
    if not comp.empty:
        comp = comp[[
            "fold",
            "cost_bps",
            "rule_text",
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
    if not y.empty:
        y = y[["fold", "policy", "cost_bps", "year", "return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"]]
        y = _fmt_pct(y, ["return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"])

    return "\n".join([
        "# ATR Defensive Strategy Gate",
        "",
        "Base exposure is ATR-HMM tiered.  Mainline mismatch alone does not reduce exposure.",
        "The gate only cuts exposure when risk is extreme or strategy health is weak.",
        "",
        "## Metrics",
        "",
        show.to_markdown(index=False),
        "",
        "## Versus ATR-HMM Tiered Baseline",
        "",
        comp.to_markdown(index=False) if not comp.empty else "No baseline metrics found.",
        "",
        "## Yearly",
        "",
        y.to_markdown(index=False) if not y.empty else "No yearly metrics.",
        "",
    ])


if __name__ == "__main__":
    wf = sys.argv[1] if len(sys.argv) > 1 else "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/walk_forward_20260706T102017Z"
    features = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_FEATURE_DIR
    main(wf, features)
