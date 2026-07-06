"""Training-window experiments for ATR lower-shadow reversion.

The trading/backtest protocol is fixed:
- PIT rolling top1000 universe
- Top5, 10-day rebalance
- 10/20 bps round-trip costs
- ATR-HMM tiered gate calibrated per fold validation period

Only LightGBM training samples / sample weights differ across variants.
"""

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

from atr_reversion_pit_hmm_calibrated_backtest import _rank_states_from_validation, _tiered_weight
from atr_reversion_pit_regime_backtest import _run_regime_backtest_pit
from atr_reversion_small_portfolio_backtest import _json_default, _metrics
from atr_reversion_walk_forward import FOLDS, REBALANCE_DAYS


TOP_N = 5
COSTS = [10, 20]
VARIANTS = [
    "expanding",
    "rolling_3y",
    "rolling_2y",
    "recency_weighted",
    "regime_filtered_train",
]
DEFAULT_FEATURE_DIR = (
    "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/"
    "three_layer_gate_20260706T113826Z"
)


def main(
    config_path: str = "configs/ml/atr_reversion_lightgbm_v1.yaml",
    pit_run: str = "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z",
    feature_dir: str = DEFAULT_FEATURE_DIR,
) -> None:
    cfg = load_atr_reversion_config(config_path)
    pit_run_path = Path(pit_run)
    feature_path = Path(feature_dir)
    output = pit_run_path / f"training_window_experiment_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
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
    regime_features = pd.read_parquet(feature_path / "base_three_layer_features.parquet")
    regime_features["trade_date"] = pd.to_datetime(regime_features["trade_date"])
    reversal_ok_dates = _reversal_training_dates(regime_features)
    features = FEATURE_GROUPS["all"]
    log(
        f"loaded panel={len(panel):,} pit={len(pit):,} dataset={len(dataset):,} "
        f"reversal_ok_dates={len(reversal_ok_dates):,} version={version}"
    )

    rows: list[dict] = []
    yearly_frames: list[pd.DataFrame] = []
    train_summaries: list[dict] = []
    for variant in VARIANTS:
        for fold in FOLDS:
            fold_name = fold["name"]
            log(f"variant={variant} fold={fold_name}: train/predict")
            pred, train_summary = _train_predict_variant(dataset, features, cfg, fold, variant, reversal_ok_dates, log)
            train_summary.update({"variant": variant, "fold": fold_name})
            train_summaries.append(train_summary)
            variant_dir = output / variant / fold_name
            variant_dir.mkdir(parents=True, exist_ok=True)
            pred.to_parquet(variant_dir / "predictions_valid_test.parquet", index=False)

            states = pd.read_csv(pit_run_path / "walk_forward_20260706T102017Z" / fold_name / "hmm_daily_states.csv")
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
                ranks, state_perf = _rank_states_from_validation(valid_daily, states)
                state_perf.to_csv(variant_dir / f"state_validation_perf_cost{cost}.csv", index=False, encoding="utf-8-sig")
                daily, trades = _run_regime_backtest_pit(
                    test_panel,
                    test_pred,
                    states,
                    top_n=TOP_N,
                    rebalance_days=REBALANCE_DAYS,
                    cost_bps=cost,
                    policy=lambda row, ranks=ranks: _tiered_weight(row, ranks),
                )
                tag = f"{variant}_{fold_name}_tiered_top{TOP_N}_cost{cost}"
                daily.to_parquet(variant_dir / f"daily_{tag}.parquet", index=False)
                trades.to_parquet(variant_dir / f"trades_{tag}.parquet", index=False)
                metrics = _metrics(daily, trades)
                metrics.update({
                    "variant": variant,
                    "fold": fold_name,
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "policy": "atr_hmm_tiered",
                    "top_n": TOP_N,
                    "cost_bps": cost,
                    "best_state": ranks["best"],
                    "neutral_state": ranks["neutral"],
                    "worst_state": ranks["worst"],
                    "avg_exposure": float(daily["exposure"].mean()),
                    "avg_daily_turnover": float(daily["turnover"].mean()),
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
    train_df = pd.DataFrame(train_summaries)
    summary = _variant_summary(metrics_df)
    metrics_df.to_csv(output / "training_window_metrics.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "training_window_yearly.csv", index=False, encoding="utf-8-sig")
    train_df.to_csv(output / "training_window_train_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "training_window_variant_summary.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "data_version": version,
                "pit_run": str(pit_run_path),
                "feature_dir": str(feature_path),
                "run_dir": str(output),
                "variants": VARIANTS,
                "metrics": metrics_df.to_dict("records"),
                "train_summary": train_df.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(metrics_df, summary, train_df), encoding="utf-8")
    log("wrote training-window experiment report")
    log("done")
    print(f"run_dir={output}")


def _load_panel() -> tuple[str, pd.DataFrame]:
    project = load_project("configs/project.yaml")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest("latest")
    _, panel = repo.load_panel(version)
    return version, panel


def _reversal_training_dates(regime_features: pd.DataFrame) -> set[pd.Timestamp]:
    f = regime_features.copy()
    ok = (
        (
            f["reversal_strength_20"].fillna(0.0).gt(0.0)
            | f["lower_shadow_style_20"].fillna(0.0).gt(0.0)
            | f["core_signal_style_20"].fillna(0.0).gt(0.0)
        )
        & f["momentum_minus_reversal_20"].fillna(0.0).le(0.005)
    )
    return set(pd.to_datetime(f.loc[ok, "trade_date"]))


def _train_predict_variant(
    dataset: pd.DataFrame,
    features: list[str],
    cfg,
    fold: dict,
    variant: str,
    reversal_ok_dates: set[pd.Timestamp],
    log,
) -> tuple[pd.DataFrame, dict]:
    from lightgbm import LGBMRegressor

    train_start = pd.Timestamp(fold["train_start"])
    train_end = pd.Timestamp(fold["train_end"])
    if variant == "rolling_3y":
        train_start = train_end - pd.DateOffset(years=3) + pd.Timedelta(days=1)
    elif variant == "rolling_2y":
        train_start = train_end - pd.DateOffset(years=2) + pd.Timedelta(days=1)
    elif variant in {"expanding", "recency_weighted", "regime_filtered_train"}:
        pass
    else:
        raise ValueError(f"unknown variant: {variant}")

    cols = ["datetime", "instrument", *features, "label", "sample_weight", "pit_top1000"]
    train_mask = (
        dataset["datetime"].between(train_start, train_end)
        & dataset["pit_top1000"].fillna(False).astype(bool)
    )
    if variant == "regime_filtered_train":
        train_mask &= dataset["datetime"].isin(reversal_ok_dates)
    train = dataset.loc[train_mask, cols].dropna(subset=[*features, "label"]).copy()
    valid_test = dataset.loc[
        dataset["datetime"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["test_end"]))
        & dataset["pit_top1000"].fillna(False).astype(bool),
        cols,
    ].dropna(subset=features)
    weights = train["sample_weight"].fillna(1.0).astype(float)
    if variant == "recency_weighted":
        recency_start = train_end - pd.DateOffset(years=2) + pd.Timedelta(days=1)
        weights.loc[train["datetime"].ge(recency_start)] *= 2.0
    if train.empty:
        raise RuntimeError(f"empty training set variant={variant} fold={fold['name']}")
    params = cfg.model.model_dump()
    params.pop("objective", None)
    params.setdefault("verbosity", -1)
    params.setdefault("force_col_wise", True)
    model = LGBMRegressor(objective="regression", **params)
    log(
        f"fit variant={variant} fold={fold['name']} train={len(train):,} "
        f"date={train['datetime'].min().date()}..{train['datetime'].max().date()} "
        f"predict={len(valid_test):,} weight_mean={weights.mean():.3f}"
    )
    model.fit(train[features], train["label"], sample_weight=weights)
    out = valid_test[["datetime", "instrument"]].copy()
    out["factor_value"] = model.predict(valid_test[features])
    out = out.rename(columns={"datetime": "trade_date", "instrument": "ts_code"})
    summary = {
        "configured_train_start": fold["train_start"],
        "configured_train_end": fold["train_end"],
        "actual_train_start": train["datetime"].min(),
        "actual_train_end": train["datetime"].max(),
        "train_rows": int(len(train)),
        "predict_rows": int(len(valid_test)),
        "sample_weight_mean": float(weights.mean()),
        "sample_weight_median": float(weights.median()),
        "sample_weight_max": float(weights.max()),
    }
    return out, summary


def _yearly(meta: dict, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = daily.copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    for year, g in d.groupby(d["trade_date"].dt.year):
        if len(g) < 2:
            continue
        total = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1.0
        bench = (1.0 + g["benchmark_return"]).prod() - 1.0
        dd = g["nav"] / g["nav"].cummax() - 1.0
        rows.append({
            "variant": meta["variant"],
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


def _variant_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    return metrics.groupby(["variant", "cost_bps"]).agg(
        mean_ann=("annualized_return", "mean"),
        median_ann=("annualized_return", "median"),
        mean_excess=("annualized_excess_return", "mean"),
        median_excess=("annualized_excess_return", "median"),
        positive_excess_folds=("annualized_excess_return", lambda s: int((s > 0).sum())),
        mean_sharpe=("sharpe", "mean"),
        worst_drawdown=("max_drawdown", "min"),
        mean_exposure=("avg_exposure", "mean"),
    ).reset_index()


def _fmt_pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    return out


def _report(metrics: pd.DataFrame, summary: pd.DataFrame, train: pd.DataFrame) -> str:
    show = metrics[[
        "variant",
        "fold",
        "cost_bps",
        "annualized_return",
        "annualized_excess_return",
        "sharpe",
        "max_drawdown",
        "avg_exposure",
    ]].copy()
    show = _fmt_pct(show, ["annualized_return", "annualized_excess_return", "max_drawdown", "avg_exposure"])
    show["sharpe"] = show["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    summ = summary.copy()
    summ = _fmt_pct(summ, [
        "mean_ann",
        "median_ann",
        "mean_excess",
        "median_excess",
        "worst_drawdown",
        "mean_exposure",
    ])
    if "mean_sharpe" in summ:
        summ["mean_sharpe"] = summ["mean_sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    tr = train[[
        "variant",
        "fold",
        "actual_train_start",
        "actual_train_end",
        "train_rows",
        "sample_weight_mean",
        "sample_weight_max",
    ]].copy()
    return "\n".join([
        "# ATR Training Window Experiment",
        "",
        "Only the LightGBM training window / sample weighting changes. Backtest protocol is fixed.",
        "",
        "## Variant Summary",
        "",
        summ.to_markdown(index=False),
        "",
        "## Fold Metrics",
        "",
        show.to_markdown(index=False),
        "",
        "## Training Samples",
        "",
        tr.to_markdown(index=False),
        "",
    ])


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/ml/atr_reversion_lightgbm_v1.yaml"
    pit_run = sys.argv[2] if len(sys.argv) > 2 else "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z"
    feature_dir = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_FEATURE_DIR
    main(config, pit_run, feature_dir)
