"""Retrain ATR reversion on the investor-permission universe and backtest.

This is stricter than post-filtering predictions:
- dataset construction uses only allowed stocks for feature histories;
- PIT Top1000 flags are set to False for excluded boards;
- cross-sectional scaling and label ranking happen inside the allowed PIT subset;
- LightGBM is retrained rolling_2y per fold;
- fit-quality flip is recomputed inside the allowed PIT subset.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository
from factor_forge.ml.atr_reversion_config import load_atr_reversion_config
from factor_forge.ml.atr_reversion_dataset import FEATURE_GROUPS, build_atr_reversion_dataset

from atr_reversion_fit_quality_gate import HMM_VARIANT, PIT_RUN, SOURCE_RUN, _apply_score_direction, _yearly
from atr_reversion_fit_quality_sensitivity import _rolling_controls
from atr_reversion_permission_filtered_backtest import (
    COST,
    LOOKBACK,
    MIN_OBS,
    TOP_N,
    _permission_eligible,
    _restricted_fit_metrics,
)
from atr_reversion_pit_hmm_calibrated_backtest import _rank_states_from_validation, _tiered_weight
from atr_reversion_pit_liquidity_backtest import _attach_and_preprocess_pit
from atr_reversion_pit_regime_backtest import _run_regime_backtest_pit
from atr_reversion_small_portfolio_backtest import _json_default, _metrics
from atr_reversion_training_window_experiment import _train_predict_variant
from atr_reversion_walk_forward import FOLDS, REBALANCE_DAYS


POLICY = "permission_retrained_rolling2y_fit_quality_flip"
ALPHA_VARIANT = "rolling_2y"


def main(
    config_path: str = "configs/ml/atr_reversion_lightgbm_v1.yaml",
    source_run: str = str(SOURCE_RUN),
) -> None:
    cfg0 = load_atr_reversion_config(config_path)
    cfg = cfg0.model_copy(
        update={
            "universe_top_n": None,
            "features": cfg0.features.model_copy(
                update={"cross_sectional_zscore": False, "winsor_quantile": 0.0}
            ),
            "label": cfg0.label.model_copy(update={"cross_sectional_rank_label": False}),
        }
    )
    source = Path(source_run)
    output = PIT_RUN / f"permission_retrain_frozen_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
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
    panel["permission_eligible"] = panel["ts_code"].map(_permission_eligible).astype(bool)
    pit = pd.read_parquet(PIT_RUN / "pit_liquidity_flags.parquet")
    pit["trade_date"] = pd.to_datetime(pit["trade_date"])
    pit = pit.copy()
    pit["permission_eligible"] = pit["ts_code"].map(_permission_eligible).astype(bool)
    pit.loc[~pit["permission_eligible"], "pit_top1000"] = False
    pit = pit.drop(columns=["permission_eligible"])
    features = FEATURE_GROUPS["all"]
    log(f"loaded panel={len(panel):,} pit={len(pit):,} version={version}")

    dataset = _build_or_load_dataset(panel, pit, cfg, features, output, log)

    rows: list[dict] = []
    yearly_frames: list[pd.DataFrame] = []
    train_summaries: list[dict] = []
    gate_frames: list[pd.DataFrame] = []
    fit_frames: list[pd.DataFrame] = []
    universe_frames: list[pd.DataFrame] = []

    for fold in FOLDS:
        fold_name = fold["name"]
        fold_dir = output / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        log(f"{fold_name}: retrain LightGBM variant={ALPHA_VARIANT}")
        pred, train_summary = _train_predict_variant(dataset, features, cfg0, fold, ALPHA_VARIANT, set(), log)
        pred = pred[pred["ts_code"].map(_permission_eligible)].copy()
        pred.to_parquet(fold_dir / "predictions_valid_test_permission_retrained_rolling_2y.parquet", index=False)
        train_summary.update({"fold": fold_name, "variant": ALPHA_VARIANT})
        train_summaries.append(train_summary)

        states = pd.read_csv(source / fold_name / HMM_VARIANT / "hmm_daily_states.csv")
        states["trade_date"] = pd.to_datetime(states["trade_date"])
        fit_daily = _restricted_fit_metrics(panel, pit, pred, fold, log)
        controls = _rolling_controls(states["trade_date"], fit_daily, LOOKBACK, MIN_OBS)
        controls["fold"] = fold_name
        controls["lookback"] = LOOKBACK
        controls["min_obs"] = MIN_OBS
        controls["cost_bps"] = COST
        controls["policy"] = POLICY
        controls.to_csv(fold_dir / "fit_quality_controls.csv", index=False, encoding="utf-8-sig")
        gate_frames.append(controls)
        fit_daily["fold"] = fold_name
        fit_frames.append(fit_daily)

        adjusted_pred = _apply_score_direction(pred, controls)
        adjusted_pred.to_parquet(fold_dir / "predictions_adjusted_permission_retrained.parquet", index=False)

        panel_bt = panel[
            panel["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["test_end"]))
        ].merge(pit, on=["trade_date", "ts_code"], how="left")
        panel_bt["pit_top1000"] = panel_bt["pit_top1000"].fillna(False).astype(bool)
        panel_bt.loc[~panel_bt["permission_eligible"], "pit_top1000"] = False
        valid_panel = panel_bt[
            panel_bt["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["valid_end"]))
        ]
        test_panel = panel_bt[
            panel_bt["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))
        ]
        valid_pred = adjusted_pred[
            adjusted_pred["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["valid_end"]))
        ]
        test_pred = adjusted_pred[
            adjusted_pred["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))
        ]
        states_ext = states.merge(controls[["trade_date", "strategy_gate"]], on="trade_date", how="left")
        states_ext["strategy_gate"] = states_ext["strategy_gate"].fillna(1.0)

        valid_daily, _ = _run_regime_backtest_pit(
            valid_panel,
            valid_pred,
            states_ext,
            top_n=TOP_N,
            rebalance_days=REBALANCE_DAYS,
            cost_bps=COST,
            policy=lambda row: float(row.get("strategy_gate", 1.0)),
        )
        ranks, state_perf = _rank_states_from_validation(valid_daily, states_ext)
        state_perf.to_csv(fold_dir / f"state_validation_perf_cost{COST}.csv", index=False, encoding="utf-8-sig")
        daily, trades = _run_regime_backtest_pit(
            test_panel,
            test_pred,
            states_ext,
            top_n=TOP_N,
            rebalance_days=REBALANCE_DAYS,
            cost_bps=COST,
            policy=lambda row, ranks=ranks: _tiered_weight(row, ranks) * float(row.get("strategy_gate", 1.0)),
        )
        tag = f"{fold_name}_{POLICY}_top{TOP_N}_cost{COST}"
        daily.to_parquet(output / f"daily_{tag}.parquet", index=False)
        trades.to_parquet(output / f"trades_{tag}.parquet", index=False)
        metrics = _metrics(daily, trades)
        test_controls = controls[
            controls["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))
        ]
        row = {
            "fold": fold_name,
            "policy": POLICY,
            "variant": ALPHA_VARIANT,
            "top_n": TOP_N,
            "cost_bps": COST,
            "hmm_variant": HMM_VARIANT,
            "lookback": LOOKBACK,
            "min_obs": MIN_OBS,
            "best_state": ranks["best"],
            "neutral_state": ranks["neutral"],
            "worst_state": ranks["worst"],
            "avg_exposure": float(daily["exposure"].mean()),
            "avg_daily_turnover": float(daily["turnover"].mean()),
            "flip_ratio": float(test_controls["score_direction"].lt(0.0).mean()),
            **metrics,
        }
        rows.append(row)
        y = _yearly(row, daily)
        y["variant"] = ALPHA_VARIANT
        y["lookback"] = LOOKBACK
        y["min_obs"] = MIN_OBS
        yearly_frames.append(y)
        universe_frames.append(_universe_summary(panel_bt, pred, fold, fold_name))
        log(
            f"{tag} ann={metrics['annualized_return']:.2%} "
            f"excess={metrics['annualized_excess_return']:.2%} "
            f"maxdd={metrics['max_drawdown']:.2%} flip={row['flip_ratio']:.1%}"
        )

    metrics_df = pd.DataFrame(rows)
    yearly_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    train_df = pd.DataFrame(train_summaries)
    gates_df = pd.concat(gate_frames, ignore_index=True) if gate_frames else pd.DataFrame()
    fit_df = pd.concat(fit_frames, ignore_index=True) if fit_frames else pd.DataFrame()
    universe_df = pd.concat(universe_frames, ignore_index=True) if universe_frames else pd.DataFrame()
    comparison = _comparison_to_post_filter(metrics_df)

    metrics_df.to_csv(output / "permission_retrain_metrics.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(output / "permission_retrain_yearly.csv", index=False, encoding="utf-8-sig")
    train_df.to_csv(output / "permission_retrain_train_summary.csv", index=False, encoding="utf-8-sig")
    gates_df.to_csv(output / "permission_retrain_gate_scores.csv", index=False, encoding="utf-8-sig")
    fit_df.to_csv(output / "permission_retrain_fit_daily.csv", index=False, encoding="utf-8-sig")
    universe_df.to_csv(output / "permission_retrain_universe_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "permission_retrain_vs_post_filter.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "data_version": version,
                "source_run": str(source),
                "run_dir": str(output),
                "policy": POLICY,
                "variant": ALPHA_VARIANT,
                "lookback": LOOKBACK,
                "min_obs": MIN_OBS,
                "top_n": TOP_N,
                "cost_bps": COST,
                "metrics": metrics_df.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(metrics_df, yearly_df, train_df, comparison, universe_df), encoding="utf-8")
    log("done")
    print(f"run_dir={output}")


def _load_panel() -> tuple[str, pd.DataFrame]:
    project = load_project("configs/project.yaml")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest("latest")
    _, panel = repo.load_panel(version)
    return version, panel


def _build_or_load_dataset(panel: pd.DataFrame, pit: pd.DataFrame, cfg, features: list[str], output: Path, log) -> pd.DataFrame:
    cache = PIT_RUN / "atr_reversion_permission_pit_dataset_w20_top1000.parquet"
    if cache.exists():
        log(f"loading cached permission dataset {cache}")
        dataset = pd.read_parquet(cache)
        dataset["datetime"] = pd.to_datetime(dataset["datetime"])
        return dataset
    start = pd.Timestamp("2017-01-01")
    end = pd.Timestamp("2026-07-03")
    ever = pit.loc[pit["trade_date"].between(start, end) & pit["pit_top1000"], "ts_code"].unique()
    work_panel = panel[panel["ts_code"].isin(ever) & panel["permission_eligible"]].copy()
    log(f"building permission ATR dataset rows={len(work_panel):,} stocks={len(ever):,}")
    raw, _ = build_atr_reversion_dataset(work_panel, cfg.features, cfg.label)
    dataset = _attach_and_preprocess_pit(raw, pit, features, 0.0)
    dataset.to_parquet(cache, index=False)
    dataset.to_parquet(output / "permission_pit_model_dataset.parquet", index=False)
    log(f"cached permission dataset -> {cache}; rows={len(dataset):,}; usable={int(dataset['pit_top1000'].sum()):,}")
    return dataset


def _universe_summary(panel_bt: pd.DataFrame, pred: pd.DataFrame, fold: dict, fold_name: str) -> pd.DataFrame:
    test = panel_bt[panel_bt["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))]
    by_date = test.groupby("trade_date").agg(
        pit_permission_count=("pit_top1000", "sum"),
    )
    pred_count = pred[pred["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))].groupby("trade_date").size()
    out = by_date.join(pred_count.rename("predictable_permission_candidates"), how="left").reset_index()
    out["fold"] = fold_name
    return out


def _comparison_to_post_filter(metrics: pd.DataFrame) -> pd.DataFrame:
    prev = PIT_RUN / "permission_filtered_frozen_20260707T072213Z" / "permission_filtered_metrics.csv"
    if not prev.exists():
        return pd.DataFrame()
    old = pd.read_csv(prev)
    cols = ["fold", "annualized_return", "annualized_excess_return", "max_drawdown", "avg_exposure", "flip_ratio", "trade_count"]
    left = metrics[cols].rename(columns={c: f"retrain_{c}" for c in cols if c != "fold"})
    right = old[cols].rename(columns={c: f"post_filter_{c}" for c in cols if c != "fold"})
    out = left.merge(right, on="fold", how="left")
    out["delta_ann"] = out["retrain_annualized_return"] - out["post_filter_annualized_return"]
    out["delta_excess"] = out["retrain_annualized_excess_return"] - out["post_filter_annualized_excess_return"]
    out["delta_drawdown"] = out["retrain_max_drawdown"] - out["post_filter_max_drawdown"]
    return out


def _fmt_pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    return out


def _report(
    metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    train: pd.DataFrame,
    comparison: pd.DataFrame,
    universe: pd.DataFrame,
) -> str:
    show = metrics[[
        "fold", "annualized_return", "benchmark_annualized_return", "annualized_excess_return",
        "sharpe", "max_drawdown", "avg_exposure", "avg_daily_turnover", "flip_ratio", "trade_count",
    ]].copy()
    show = _fmt_pct(show, [
        "annualized_return", "benchmark_annualized_return", "annualized_excess_return",
        "max_drawdown", "avg_exposure", "avg_daily_turnover", "flip_ratio",
    ])
    show["sharpe"] = show["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    y = yearly.copy()
    y = _fmt_pct(y, ["return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"])
    t = train[["fold", "variant", "actual_train_start", "actual_train_end", "train_rows", "predict_rows"]].copy()
    c = comparison.copy()
    c = _fmt_pct(c, [
        "retrain_annualized_return", "post_filter_annualized_return", "delta_ann",
        "retrain_annualized_excess_return", "post_filter_annualized_excess_return", "delta_excess",
        "retrain_max_drawdown", "post_filter_max_drawdown", "delta_drawdown",
    ])
    u = universe.groupby("fold").agg(
        avg_pit_permission_count=("pit_permission_count", "mean"),
        avg_predictable_permission_candidates=("predictable_permission_candidates", "mean"),
    ).reset_index()
    return "\n".join(
        [
            "# Permission-Retrained Frozen ATR Reversion Backtest",
            "",
            "- universe: allowed main-board subset inside PIT rolling liquidity Top1000",
            "- training: LightGBM rolling_2y retrained on allowed PIT rows only",
            "- rule: lookback=40, min_obs=15, flip when completed rolling RankIC < 0 and decile_spread < 0",
            "- portfolio: Top5, 10-day rebalance/holding, 20bps cost",
            "",
            "## Overall",
            "",
            show.to_markdown(index=False),
            "",
            "## Versus Post-Filter-Only",
            "",
            c.to_markdown(index=False) if not c.empty else "_No post-filter comparison found._",
            "",
            "## Yearly",
            "",
            y.to_markdown(index=False),
            "",
            "## Training Summary",
            "",
            t.to_markdown(index=False),
            "",
            "## Universe Size",
            "",
            u.to_markdown(index=False),
            "",
        ]
    )


if __name__ == "__main__":
    main()
