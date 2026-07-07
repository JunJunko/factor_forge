"""Small weight matrix for main-board ATR reversion with upper-shadow features."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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
from atr_reversion_walk_forward import FOLDS, REBALANCE_DAYS


VARIANTS = [
    "original_weight",
    "equal_weight",
    "low_elasticity_bias",
    "stable_reversal_weight",
]
POLICY = "permission_upper_shadow_weight_matrix"


def main(config_path: str = "configs/ml/atr_reversion_lightgbm_v1.yaml", source_run: str = str(SOURCE_RUN)) -> None:
    cfg0 = load_atr_reversion_config(config_path)
    cfg = cfg0.model_copy(
        update={
            "universe_top_n": None,
            "features": cfg0.features.model_copy(update={"cross_sectional_zscore": False, "winsor_quantile": 0.0}),
            "label": cfg0.label.model_copy(update={"cross_sectional_rank_label": False}),
        }
    )
    source = Path(source_run)
    output = PIT_RUN / f"permission_upper_shadow_matrix_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
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
    pit.loc[~pit["ts_code"].map(_permission_eligible), "pit_top1000"] = False
    features = FEATURE_GROUPS["all"]
    if "upper_shadow_atr" not in features:
        raise RuntimeError("upper-shadow features are not registered in FEATURE_GROUPS['all']")
    dataset = _build_or_load_dataset(panel, pit, cfg, features, output, log)
    log(f"loaded data_version={version} dataset={len(dataset):,} features={len(features)} variants={VARIANTS}")

    rows: list[dict] = []
    yearly_frames: list[pd.DataFrame] = []
    train_frames: list[dict] = []
    gate_frames: list[pd.DataFrame] = []

    for variant in VARIANTS:
        for fold in FOLDS:
            fold_name = fold["name"]
            fold_dir = output / variant / fold_name
            fold_dir.mkdir(parents=True, exist_ok=True)
            pred, train_summary = _train_predict_weight_variant(dataset, features, cfg0, fold, variant, log)
            pred.to_parquet(fold_dir / "predictions_valid_test.parquet", index=False)
            train_summary.update({"variant": variant, "fold": fold_name})
            train_frames.append(train_summary)

            states = pd.read_csv(source / fold_name / HMM_VARIANT / "hmm_daily_states.csv")
            states["trade_date"] = pd.to_datetime(states["trade_date"])
            fit_daily = _restricted_fit_metrics(panel, pit, pred, fold, log)
            controls = _rolling_controls(states["trade_date"], fit_daily, LOOKBACK, MIN_OBS)
            controls["variant"] = variant
            controls["fold"] = fold_name
            controls["lookback"] = LOOKBACK
            controls["min_obs"] = MIN_OBS
            controls["cost_bps"] = COST
            controls["policy"] = POLICY
            gate_frames.append(controls)
            controls.to_csv(fold_dir / "fit_quality_controls.csv", index=False, encoding="utf-8-sig")

            adjusted_pred = _apply_score_direction(pred, controls)
            adjusted_pred.to_parquet(fold_dir / "predictions_adjusted.parquet", index=False)

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
            ranks, _state_perf = _rank_states_from_validation(valid_daily, states_ext)
            daily, trades = _run_regime_backtest_pit(
                test_panel,
                test_pred,
                states_ext,
                top_n=TOP_N,
                rebalance_days=REBALANCE_DAYS,
                cost_bps=COST,
                policy=lambda row, ranks=ranks: _tiered_weight(row, ranks) * float(row.get("strategy_gate", 1.0)),
            )
            tag = f"{variant}_{fold_name}_top{TOP_N}_cost{COST}"
            daily.to_parquet(output / f"daily_{tag}.parquet", index=False)
            trades.to_parquet(output / f"trades_{tag}.parquet", index=False)
            metrics = _metrics(daily, trades)
            test_controls = controls[
                controls["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))
            ]
            row = {
                "variant": variant,
                "fold": fold_name,
                "policy": POLICY,
                "top_n": TOP_N,
                "cost_bps": COST,
                "hmm_variant": HMM_VARIANT,
                "lookback": LOOKBACK,
                "min_obs": MIN_OBS,
                "avg_exposure": float(daily["exposure"].mean()),
                "avg_daily_turnover": float(daily["turnover"].mean()),
                "flip_ratio": float(test_controls["score_direction"].lt(0.0).mean()),
                **metrics,
            }
            rows.append(row)
            y = _yearly(row, daily)
            y["variant"] = variant
            y["lookback"] = LOOKBACK
            y["min_obs"] = MIN_OBS
            yearly_frames.append(y)
            log(
                f"{tag} ann={metrics['annualized_return']:.2%} "
                f"excess={metrics['annualized_excess_return']:.2%} "
                f"maxdd={metrics['max_drawdown']:.2%} flip={row['flip_ratio']:.1%}"
            )

    metrics_df = pd.DataFrame(rows)
    yearly_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    train_df = pd.DataFrame(train_frames)
    gates_df = pd.concat(gate_frames, ignore_index=True) if gate_frames else pd.DataFrame()
    summary = _summary(metrics_df)
    metrics_df.to_csv(output / "upper_shadow_weight_metrics.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(output / "upper_shadow_weight_yearly.csv", index=False, encoding="utf-8-sig")
    train_df.to_csv(output / "upper_shadow_weight_train_summary.csv", index=False, encoding="utf-8-sig")
    gates_df.to_csv(output / "upper_shadow_weight_gate_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "upper_shadow_weight_summary.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "data_version": version,
                "run_dir": str(output),
                "features": features,
                "variants": VARIANTS,
                "metrics": metrics_df.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(metrics_df, yearly_df, train_df, summary), encoding="utf-8")
    log("done")
    print(f"run_dir={output}")


def _load_panel() -> tuple[str, pd.DataFrame]:
    project = load_project("configs/project.yaml")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest("latest")
    _, panel = repo.load_panel(version)
    return version, panel


def _build_or_load_dataset(panel: pd.DataFrame, pit: pd.DataFrame, cfg, features: list[str], output: Path, log) -> pd.DataFrame:
    cache = PIT_RUN / "atr_reversion_permission_upper_shadow_pit_dataset_w20_top1000.parquet"
    if cache.exists():
        log(f"loading cached upper-shadow permission dataset {cache}")
        dataset = pd.read_parquet(cache)
        dataset["datetime"] = pd.to_datetime(dataset["datetime"])
        missing = set(features) - set(dataset.columns)
        if not missing:
            return dataset
        log(f"cached dataset missing {sorted(missing)}, rebuilding")
    start = pd.Timestamp("2017-01-01")
    end = pd.Timestamp("2026-07-03")
    ever = pit.loc[pit["trade_date"].between(start, end) & pit["pit_top1000"], "ts_code"].unique()
    work_panel = panel[panel["ts_code"].isin(ever) & panel["permission_eligible"]].copy()
    log(f"building upper-shadow permission dataset rows={len(work_panel):,} stocks={len(ever):,}")
    raw, _ = build_atr_reversion_dataset(work_panel, cfg.features, cfg.label)
    dataset = _attach_and_preprocess_pit(raw, pit, features, 0.0)
    dataset.to_parquet(cache, index=False)
    dataset.to_parquet(output / "permission_upper_shadow_pit_model_dataset.parquet", index=False)
    log(f"cached upper-shadow dataset -> {cache}; rows={len(dataset):,}; usable={int(dataset['pit_top1000'].sum()):,}")
    return dataset


def _train_predict_weight_variant(dataset: pd.DataFrame, features: list[str], cfg, fold: dict, variant: str, log) -> tuple[pd.DataFrame, dict]:
    from lightgbm import LGBMRegressor

    train_end = pd.Timestamp(fold["train_end"])
    train_start = train_end - pd.DateOffset(years=2) + pd.Timedelta(days=1)
    cols = ["datetime", "instrument", *features, "label", "sample_weight", "pit_top1000"]
    train_mask = dataset["datetime"].between(train_start, train_end) & dataset["pit_top1000"].fillna(False).astype(bool)
    train = dataset.loc[train_mask, cols].dropna(subset=[*features, "label"]).copy()
    valid_test = dataset.loc[
        dataset["datetime"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["test_end"]))
        & dataset["pit_top1000"].fillna(False).astype(bool),
        cols,
    ].dropna(subset=features)
    if train.empty:
        raise RuntimeError(f"empty training set variant={variant} fold={fold['name']}")
    weights = _variant_weights(train, variant)
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
    summary = {
        "actual_train_start": train["datetime"].min(),
        "actual_train_end": train["datetime"].max(),
        "train_rows": int(len(train)),
        "predict_rows": int(len(valid_test)),
        "sample_weight_mean": float(weights.mean()),
        "sample_weight_median": float(weights.median()),
        "sample_weight_max": float(weights.max()),
    }
    return out.rename(columns={"datetime": "trade_date", "instrument": "ts_code"}), summary


def _variant_weights(train: pd.DataFrame, variant: str) -> pd.Series:
    base = train["sample_weight"].fillna(1.0).astype(float)
    if variant == "original_weight":
        return base
    if variant == "equal_weight":
        return pd.Series(1.0, index=train.index)
    vol_high = train["vol_state_pct"].ge(0.8).fillna(False)
    amount_q80 = train.groupby("datetime")["amount_shock"].transform(lambda s: s.quantile(0.8))
    amount_high = train["amount_shock"].ge(amount_q80).fillna(False)
    upper_high = train["upper_shadow_pct"].ge(0.75).fillna(False)
    lower_good = train["lower_shadow_pct"].ge(0.65).fillna(False)
    repair_good = train["intraday_repair"].ge(0.5).fillna(False)
    down_good = train["down_deviation_pct"].ge(0.5).fillna(False)
    if variant == "low_elasticity_bias":
        w = base.copy()
        w.loc[vol_high | amount_high] *= 0.55
        w.loc[upper_high] *= 0.70
        return w.clip(lower=0.1, upper=2.0)
    if variant == "stable_reversal_weight":
        w = base.copy()
        stable = lower_good & repair_good & down_good & ~vol_high & ~amount_high & ~upper_high
        w.loc[stable] *= 1.50
        w.loc[vol_high | amount_high | upper_high] *= 0.60
        return w.clip(lower=0.1, upper=2.0)
    raise ValueError(f"unknown variant: {variant}")


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby("variant")
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


def _fmt_pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    return out


def _report(metrics: pd.DataFrame, yearly: pd.DataFrame, train: pd.DataFrame, summary: pd.DataFrame) -> str:
    s = _fmt_pct(summary, ["mean_ann", "median_ann", "mean_excess", "median_excess", "worst_drawdown", "mean_exposure", "mean_flip_ratio"])
    s["mean_sharpe"] = s["mean_sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    show = metrics[[
        "variant", "fold", "annualized_return", "benchmark_annualized_return", "annualized_excess_return",
        "sharpe", "max_drawdown", "avg_exposure", "flip_ratio", "trade_count",
    ]].copy()
    show = _fmt_pct(show, ["annualized_return", "benchmark_annualized_return", "annualized_excess_return", "max_drawdown", "avg_exposure", "flip_ratio"])
    show["sharpe"] = show["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    y = yearly.copy()
    y = _fmt_pct(y, ["return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"])
    t = train[["variant", "fold", "actual_train_start", "actual_train_end", "train_rows", "predict_rows", "sample_weight_mean"]].copy()
    return "\n".join(
        [
            "# Upper-Shadow Weight Matrix",
            "",
            "- universe: allowed main-board subset inside PIT rolling liquidity Top1000",
            "- features: existing ATR lower-shadow set plus upper_shadow_atr and upper_shadow_pct",
            "- model: LightGBM rolling_2y per fold",
            "- rule: lookback=40, min_obs=15 fit-quality flip",
            "- portfolio: Top5, 10-day rebalance/holding, 20bps cost",
            "",
            "## Variant Summary",
            "",
            s.to_markdown(index=False),
            "",
            "## Fold Metrics",
            "",
            show.to_markdown(index=False),
            "",
            "## Yearly",
            "",
            y.to_markdown(index=False),
            "",
            "## Training Summary",
            "",
            t.to_markdown(index=False),
            "",
        ]
    )


if __name__ == "__main__":
    main()
