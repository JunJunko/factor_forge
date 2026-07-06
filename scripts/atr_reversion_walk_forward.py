"""Walk-forward validation for ATR PIT-liquidity + ATR-calibrated HMM gates."""

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
from factor_forge.ml.atr_reversion_config import load_atr_reversion_config
from factor_forge.ml.atr_reversion_dataset import FEATURE_GROUPS

from atr_reversion_pit_hmm_calibrated_backtest import (
    _market_features,
    _rank_states_from_validation,
    _run_regime_backtest_pit,
    _soft_prob_weight,
    _tiered_weight,
    _train_predict_valid_test,
    _walk_forward_hmm,
)
from atr_reversion_small_portfolio_backtest import _json_default, _metrics


FOLDS = [
    {
        "name": "test_2022",
        "train_start": "2017-01-01",
        "train_end": "2020-12-31",
        "valid_start": "2021-01-01",
        "valid_end": "2021-12-31",
        "test_start": "2022-01-01",
        "test_end": "2022-12-31",
    },
    {
        "name": "test_2023",
        "train_start": "2017-01-01",
        "train_end": "2021-12-31",
        "valid_start": "2022-01-01",
        "valid_end": "2022-12-31",
        "test_start": "2023-01-01",
        "test_end": "2023-12-31",
    },
    {
        "name": "test_2024",
        "train_start": "2017-01-01",
        "train_end": "2022-12-31",
        "valid_start": "2023-01-01",
        "valid_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
    },
    {
        "name": "test_2025",
        "train_start": "2018-01-01",
        "train_end": "2023-12-31",
        "valid_start": "2024-01-01",
        "valid_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31",
    },
    {
        "name": "test_2026h1",
        "train_start": "2019-01-01",
        "train_end": "2024-12-31",
        "valid_start": "2025-01-01",
        "valid_end": "2025-12-31",
        "test_start": "2026-01-01",
        "test_end": "2026-06-30",
    },
]

TOP_NS = [5]
COST_BPS = [10, 20]
POLICIES = ["ungated", "atr_hmm_tiered", "atr_hmm_soft_prob"]
REBALANCE_DAYS = 10


def main(
    config_path: str = "configs/ml/atr_reversion_lightgbm_v1.yaml",
    pit_run: str = "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z",
) -> None:
    cfg_base = load_atr_reversion_config(config_path)
    pit_run_path = Path(pit_run)
    output = pit_run_path / f"walk_forward_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
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
    pit = pd.read_parquet(pit_run_path / "pit_liquidity_flags.parquet")
    pit["trade_date"] = pd.to_datetime(pit["trade_date"])
    dataset = pd.read_parquet(pit_run_path / "pit_model_dataset.parquet")
    dataset["datetime"] = pd.to_datetime(dataset["datetime"])
    features = FEATURE_GROUPS["all"]
    market = _market_features(panel)
    log(f"loaded panel={len(panel):,} dataset={len(dataset):,} version={version}")

    all_rows = []
    all_yearly = []
    for fold in FOLDS:
        log(f"fold start {fold['name']}")
        fold_dir = output / fold["name"]
        fold_dir.mkdir(parents=True, exist_ok=False)
        cfg = cfg_base.model_copy(
            update={
                "segments": cfg_base.segments.model_copy(
                    update={
                        "train": cfg_base.segments.train.model_copy(
                            update={"start": fold["train_start"], "end": fold["train_end"]}
                        ),
                        "valid": cfg_base.segments.valid.model_copy(
                            update={"start": fold["valid_start"], "end": fold["valid_end"]}
                        ),
                        "test": cfg_base.segments.test.model_copy(
                            update={"start": fold["test_start"], "end": fold["test_end"]}
                        ),
                    }
                )
            }
        )
        pred = _train_predict_valid_test(dataset, features, cfg, log)
        pred.to_parquet(fold_dir / "predictions_valid_test.parquet", index=False)
        states = _walk_forward_hmm(market, fold["valid_start"], fold["test_end"], log)
        states.to_csv(fold_dir / "hmm_daily_states.csv", index=False, encoding="utf-8-sig")

        panel_bt = panel[
            panel["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["test_end"]))
        ].merge(pit, on=["trade_date", "ts_code"], how="left")
        panel_bt["pit_top1000"] = panel_bt["pit_top1000"].fillna(False).astype(bool)
        valid_panel = panel_bt[panel_bt["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["valid_end"]))]
        test_panel = panel_bt[panel_bt["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))]
        valid_pred = pred[pred["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["valid_end"]))]
        test_pred = pred[pred["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))]

        for top_n in TOP_NS:
            for cost in COST_BPS:
                valid_daily, _ = _run_regime_backtest_pit(
                    valid_panel,
                    valid_pred,
                    states,
                    top_n=top_n,
                    rebalance_days=REBALANCE_DAYS,
                    cost_bps=cost,
                    policy=lambda _row: 1.0,
                )
                ranks, state_perf = _rank_states_from_validation(valid_daily, states)
                state_perf.to_csv(fold_dir / f"state_validation_perf_top{top_n}_cost{cost}.csv", index=False, encoding="utf-8-sig")
                policies = {
                    "ungated": lambda _row, ranks=ranks: 1.0,
                    "atr_hmm_tiered": lambda row, ranks=ranks: _tiered_weight(row, ranks),
                    "atr_hmm_soft_prob": lambda row, ranks=ranks: _soft_prob_weight(row, ranks),
                }
                for policy_name in POLICIES:
                    daily, trades = _run_regime_backtest_pit(
                        test_panel,
                        test_pred,
                        states,
                        top_n=top_n,
                        rebalance_days=REBALANCE_DAYS,
                        cost_bps=cost,
                        policy=policies[policy_name],
                    )
                    tag = f"{fold['name']}_{policy_name}_top{top_n}_cost{cost}"
                    daily.to_parquet(fold_dir / f"daily_{tag}.parquet", index=False)
                    trades.to_parquet(fold_dir / f"trades_{tag}.parquet", index=False)
                    metrics = _metrics(daily, trades)
                    row = {
                        "fold": fold["name"],
                        "test_start": fold["test_start"],
                        "test_end": fold["test_end"],
                        "policy": policy_name,
                        "top_n": top_n,
                        "cost_bps": cost,
                        "best_state": ranks["best"],
                        "neutral_state": ranks["neutral"],
                        "worst_state": ranks["worst"],
                        "avg_exposure": float(daily["exposure"].mean()),
                        "avg_daily_turnover": float(daily["turnover"].mean()),
                        **metrics,
                    }
                    all_rows.append(row)
                    all_yearly.append(_yearly_row(row, daily))
                    log(
                        f"{tag} ann={metrics['annualized_return']:.2%} "
                        f"excess={metrics['annualized_excess_return']:.2%} "
                        f"sharpe={metrics['sharpe']:.2f} maxdd={metrics['max_drawdown']:.2%}"
                    )
    metrics_df = pd.DataFrame(all_rows)
    yearly_df = pd.concat(all_yearly, ignore_index=True) if all_yearly else pd.DataFrame()
    metrics_df.to_csv(output / "walk_forward_metrics.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(output / "walk_forward_yearly.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "data_version": version,
                "pit_run": str(pit_run_path),
                "run_dir": str(output),
                "folds": FOLDS,
                "metrics": metrics_df.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(metrics_df), encoding="utf-8")
    log("done")
    print(f"run_dir={output}")


def _load_panel():
    project = load_project("configs/project.yaml")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest("latest")
    _, panel = repo.load_panel(version)
    return version, panel


def _yearly_row(meta: dict, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in daily.groupby(pd.to_datetime(daily["trade_date"]).dt.year):
        total = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1
        bench = (1 + g["benchmark_return"]).prod() - 1
        dd = g["nav"] / g["nav"].cummax() - 1
        rows.append({
            "fold": meta["fold"],
            "policy": meta["policy"],
            "top_n": meta["top_n"],
            "cost_bps": meta["cost_bps"],
            "year": int(year),
            "return": float(total),
            "benchmark_return": float(bench),
            "excess_return": float(total - bench),
            "max_drawdown": float(dd.min()),
            "avg_exposure": float(g["exposure"].mean()),
        })
    return pd.DataFrame(rows)


def _report(metrics: pd.DataFrame) -> str:
    grouped = metrics.groupby(["policy", "cost_bps"]).agg(
        mean_excess=("annualized_excess_return", "mean"),
        median_excess=("annualized_excess_return", "median"),
        positive_folds=("annualized_excess_return", lambda s: int((s > 0).sum())),
        mean_sharpe=("sharpe", "mean"),
        worst_drawdown=("max_drawdown", "min"),
        mean_exposure=("avg_exposure", "mean"),
    ).reset_index()
    show = metrics[[
        "fold", "policy", "cost_bps", "annualized_return", "benchmark_annualized_return",
        "annualized_excess_return", "sharpe", "max_drawdown", "avg_exposure",
    ]].copy()
    for df in [show, grouped]:
        for col in df.columns:
            if col in {"annualized_return", "benchmark_annualized_return", "annualized_excess_return", "max_drawdown", "avg_exposure", "mean_excess", "median_excess", "worst_drawdown", "mean_exposure"}:
                df[col] = df[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
            if col in {"sharpe", "mean_sharpe"}:
                df[col] = df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    return "\n".join([
        "# ATR PIT Walk-Forward Validation",
        "",
        "## Fold Metrics",
        "",
        show.to_markdown(index=False),
        "",
        "## Policy Summary",
        "",
        grouped.to_markdown(index=False),
        "",
    ])


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/ml/atr_reversion_lightgbm_v1.yaml"
    pit_run = sys.argv[2] if len(sys.argv) > 2 else "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z"
    main(config, pit_run)

