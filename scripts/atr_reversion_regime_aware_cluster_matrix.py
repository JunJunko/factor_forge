"""Regime-aware Factor Cluster LightGBM matrix for main-board ATR reversion."""

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

from atr_reversion_defensive_gate import _risk_kill_only_gate
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
from atr_reversion_strategy_regime_mining import _build_regime_features
from atr_reversion_walk_forward import FOLDS, REBALANCE_DAYS


VARIANTS = [
    "raw_features_lgbm",
    "factor_cluster_lgbm",
    "regime_aware_cluster_lgbm",
    "regime_aware_cluster_defensive",
]
CLUSTER_COLS = [
    "cluster_deviation_atr",
    "cluster_lower_shadow_shock",
    "cluster_close_repair_quality",
    "cluster_no_upper_pressure",
    "cluster_stock_state",
    "cluster_liquidity_shock",
    "cluster_market_context",
]
REGIME_COLS = [
    "market_ret_20",
    "market_ret_60",
    "market_vol_20",
    "market_breadth_20",
    "xsec_vol_20",
    "turnover_chg_5_20",
    "reversal_strength_20",
    "momentum_minus_reversal_20",
]
POLICY = "regime_aware_factor_cluster_matrix"


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
    output = PIT_RUN / f"regime_aware_cluster_matrix_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
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
    raw_features = FEATURE_GROUPS["all"]
    dataset = _build_or_load_dataset(panel, pit, cfg, raw_features, output, log)
    regime = _build_permission_regime(panel, pit, dataset, output, log)
    model_dataset = _prepare_model_dataset(dataset, regime, raw_features)
    model_dataset.to_parquet(output / "cluster_regime_model_dataset.parquet", index=False)
    log(
        f"loaded data_version={version} dataset={len(model_dataset):,} "
        f"raw_features={len(raw_features)} cluster_features={len(CLUSTER_COLS)}"
    )

    rows: list[dict] = []
    yearly_frames: list[pd.DataFrame] = []
    gain_frames: list[pd.DataFrame] = []
    train_rows: list[dict] = []
    cluster_ic_frames: list[pd.DataFrame] = []
    gate_frames: list[pd.DataFrame] = []

    for variant in VARIANTS:
        feature_cols = _feature_cols_for_variant(variant, raw_features)
        for fold in FOLDS:
            fold_name = fold["name"]
            fold_dir = output / variant / fold_name
            fold_dir.mkdir(parents=True, exist_ok=True)
            pred, train_summary, gain = _train_predict(
                model_dataset,
                feature_cols,
                cfg0,
                fold,
                variant,
                log,
            )
            pred.to_parquet(fold_dir / "predictions_valid_test.parquet", index=False)
            train_summary.update({"variant": variant, "fold": fold_name, "feature_count": len(feature_cols)})
            train_rows.append(train_summary)
            gain["variant"] = variant
            gain["fold"] = fold_name
            gain_frames.append(gain)

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
            if variant == "regime_aware_cluster_defensive":
                controls = _apply_defensive_gate(controls, regime)
            else:
                controls["strategy_gate"] = controls["strategy_gate"].astype(float)
                controls["defensive_gate"] = 1.0
            controls.to_csv(fold_dir / "fit_quality_controls.csv", index=False, encoding="utf-8-sig")
            gate_frames.append(controls)
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
                "feature_count": len(feature_cols),
                "lookback": LOOKBACK,
                "min_obs": MIN_OBS,
                "best_state": ranks["best"],
                "neutral_state": ranks["neutral"],
                "worst_state": ranks["worst"],
                "avg_exposure": float(daily["exposure"].mean()),
                "avg_daily_turnover": float(daily["turnover"].mean()),
                "flip_ratio": float(test_controls["score_direction"].lt(0.0).mean()),
                "defensive_flat_ratio": float(test_controls["defensive_gate"].eq(0.0).mean()),
                **metrics,
            }
            rows.append(row)
            y = _yearly(row, daily)
            y["variant"] = variant
            yearly_frames.append(y)
            cluster_ic_frames.append(_cluster_ic_by_state(model_dataset, states_ext, fold, variant))
            log(
                f"{tag} ann={metrics['annualized_return']:.2%} "
                f"excess={metrics['annualized_excess_return']:.2%} "
                f"maxdd={metrics['max_drawdown']:.2%} flip={row['flip_ratio']:.1%} "
                f"flat={row['defensive_flat_ratio']:.1%}"
            )

    metrics_df = pd.DataFrame(rows)
    yearly_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    gains_df = pd.concat(gain_frames, ignore_index=True) if gain_frames else pd.DataFrame()
    train_df = pd.DataFrame(train_rows)
    cluster_ic_df = pd.concat(cluster_ic_frames, ignore_index=True) if cluster_ic_frames else pd.DataFrame()
    gates_df = pd.concat(gate_frames, ignore_index=True) if gate_frames else pd.DataFrame()
    summary = _summary(metrics_df)
    gain_summary = _gain_summary(gains_df)
    cluster_effective = _cluster_effectiveness_summary(cluster_ic_df)

    metrics_df.to_csv(output / "regime_cluster_metrics.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(output / "regime_cluster_yearly.csv", index=False, encoding="utf-8-sig")
    gains_df.to_csv(output / "feature_gain_by_fold.csv", index=False, encoding="utf-8-sig")
    gain_summary.to_csv(output / "feature_gain_by_cluster.csv", index=False, encoding="utf-8-sig")
    train_df.to_csv(output / "train_summary.csv", index=False, encoding="utf-8-sig")
    cluster_ic_df.to_csv(output / "cluster_ic_by_year_state.csv", index=False, encoding="utf-8-sig")
    cluster_effective.to_csv(output / "cluster_effectiveness_summary.csv", index=False, encoding="utf-8-sig")
    gates_df.to_csv(output / "gate_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "variant_summary.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "data_version": version,
                "run_dir": str(output),
                "variants": VARIANTS,
                "cluster_cols": CLUSTER_COLS,
                "regime_cols": REGIME_COLS,
                "metrics": metrics_df.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(
        _report(metrics_df, yearly_df, summary, gain_summary, cluster_effective),
        encoding="utf-8",
    )
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
        log(f"loading cached permission upper-shadow dataset {cache}")
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
    log(f"building permission upper-shadow dataset rows={len(work_panel):,} stocks={len(ever):,}")
    raw, _ = build_atr_reversion_dataset(work_panel, cfg.features, cfg.label)
    dataset = _attach_and_preprocess_pit(raw, pit, features, 0.0)
    dataset.to_parquet(cache, index=False)
    dataset.to_parquet(output / "permission_upper_shadow_pit_model_dataset.parquet", index=False)
    return dataset


def _build_permission_regime(panel: pd.DataFrame, pit: pd.DataFrame, dataset: pd.DataFrame, output: Path, log) -> pd.DataFrame:
    cache = PIT_RUN / "permission_daily_regime_features.parquet"
    if cache.exists():
        regime = pd.read_parquet(cache)
        regime["trade_date"] = pd.to_datetime(regime["trade_date"])
        if set(REGIME_COLS).issubset(regime.columns):
            log(f"loading cached permission regime features {cache}")
            return regime
    allowed_panel = panel[panel["permission_eligible"]].copy()
    regime = _build_regime_features(allowed_panel, pit, dataset)
    regime.to_parquet(cache, index=False)
    regime.to_parquet(output / "permission_daily_regime_features.parquet", index=False)
    log(f"built permission regime features rows={len(regime):,}")
    return regime


def _prepare_model_dataset(dataset: pd.DataFrame, regime: pd.DataFrame, raw_features: list[str]) -> pd.DataFrame:
    d = dataset.copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    d = _add_clusters(d)
    r = regime[["trade_date", *REGIME_COLS]].copy()
    r = r.rename(columns={"trade_date": "datetime"})
    out = d.merge(r, on="datetime", how="left")
    for col in REGIME_COLS:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
        mean = out[col].mean(skipna=True)
        std = out[col].std(skipna=True, ddof=0)
        out[f"regime_{col}"] = (out[col] - mean) / (std if std and np.isfinite(std) else np.nan)
    for cluster in CLUSTER_COLS:
        for reg in REGIME_COLS:
            out[f"{cluster}__x__{reg}"] = out[cluster] * out[f"regime_{reg}"]
    # Keep atomic raw features available for the baseline model.
    out[raw_features] = out[raw_features].replace([np.inf, -np.inf], np.nan)
    return out


def _add_clusters(dataset: pd.DataFrame) -> pd.DataFrame:
    d = dataset.copy()
    d["cluster_deviation_atr"] = _mean_cols(d, ["down_deviation_atr", "down_deviation_pct"])
    d["cluster_lower_shadow_shock"] = _mean_cols(d, ["lower_shadow_atr", "lower_shadow_pct", "core_signal"])
    d["cluster_close_repair_quality"] = d["intraday_repair"]
    d["cluster_no_upper_pressure"] = _mean_cols(d, [-d["upper_shadow_atr"], 1.0 - d["upper_shadow_pct"]])
    d["cluster_stock_state"] = _mean_cols(d, ["trend_state", -d["vol_state"], 1.0 - d["vol_state_pct"]])
    d["cluster_liquidity_shock"] = _mean_cols(d, ["liquidity_log_amount_20", -d["amount_shock"]])
    d["cluster_market_context"] = _mean_cols(
        d,
        ["market_ret_1d", "market_ret_5d", "industry_ret_1d", "industry_ret_5d", "stock_minus_industry_5d"],
    )
    valid = d["pit_top1000"].fillna(False).astype(bool)
    for col in CLUSTER_COLS:
        grouped = d.loc[valid].groupby("datetime")[col]
        mean = grouped.transform("mean")
        std = grouped.transform("std", ddof=0)
        d.loc[valid, col] = (d.loc[valid, col] - mean) / std.replace(0, np.nan)
    return d


def _mean_cols(df: pd.DataFrame, cols: list[str | pd.Series]) -> pd.Series:
    parts = []
    for col in cols:
        if isinstance(col, str):
            parts.append(df[col])
        else:
            parts.append(col)
    return pd.concat(parts, axis=1).mean(axis=1)


def _feature_cols_for_variant(variant: str, raw_features: list[str]) -> list[str]:
    if variant == "raw_features_lgbm":
        return raw_features
    if variant == "factor_cluster_lgbm":
        return CLUSTER_COLS
    if variant in {"regime_aware_cluster_lgbm", "regime_aware_cluster_defensive"}:
        regime_z = [f"regime_{c}" for c in REGIME_COLS]
        interactions = [f"{cluster}__x__{reg}" for cluster in CLUSTER_COLS for reg in REGIME_COLS]
        return [*CLUSTER_COLS, *regime_z, *interactions]
    raise ValueError(f"unknown variant: {variant}")


def _train_predict(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    cfg,
    fold: dict,
    variant: str,
    log,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    from lightgbm import LGBMRegressor

    train_end = pd.Timestamp(fold["train_end"])
    train_start = train_end - pd.DateOffset(years=2) + pd.Timedelta(days=1)
    cols = ["datetime", "instrument", *feature_cols, "label", "sample_weight", "pit_top1000"]
    train = dataset.loc[
        dataset["datetime"].between(train_start, train_end)
        & dataset["pit_top1000"].fillna(False).astype(bool),
        cols,
    ].dropna(subset=[*feature_cols, "label"]).copy()
    valid_test = dataset.loc[
        dataset["datetime"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["test_end"]))
        & dataset["pit_top1000"].fillna(False).astype(bool),
        cols,
    ].dropna(subset=feature_cols)
    if train.empty:
        raise RuntimeError(f"empty training set variant={variant} fold={fold['name']}")
    weights = train["sample_weight"].fillna(1.0).astype(float)
    params = cfg.model.model_dump()
    params.pop("objective", None)
    params.setdefault("verbosity", -1)
    params.setdefault("force_col_wise", True)
    model = LGBMRegressor(objective="regression", **params)
    log(
        f"fit variant={variant} fold={fold['name']} train={len(train):,} "
        f"date={train['datetime'].min().date()}..{train['datetime'].max().date()} "
        f"predict={len(valid_test):,} features={len(feature_cols)}"
    )
    model.fit(train[feature_cols], train["label"], sample_weight=weights)
    out = valid_test[["datetime", "instrument"]].copy()
    out["factor_value"] = model.predict(valid_test[feature_cols])
    gain = pd.DataFrame(
        {
            "feature": feature_cols,
            "gain": model.booster_.feature_importance(importance_type="gain"),
        }
    )
    gain["feature_group"] = gain["feature"].map(_feature_group)
    gain["gain_share"] = gain["gain"] / gain["gain"].sum() if gain["gain"].sum() else 0.0
    summary = {
        "actual_train_start": train["datetime"].min(),
        "actual_train_end": train["datetime"].max(),
        "train_rows": int(len(train)),
        "predict_rows": int(len(valid_test)),
        "sample_weight_mean": float(weights.mean()),
        "sample_weight_median": float(weights.median()),
        "sample_weight_max": float(weights.max()),
    }
    return out.rename(columns={"datetime": "trade_date", "instrument": "ts_code"}), summary, gain


def _feature_group(feature: str) -> str:
    if "__x__" in feature:
        return feature.split("__x__", 1)[0]
    if feature.startswith("regime_"):
        return "regime_context"
    if feature in CLUSTER_COLS:
        return feature
    raw_map = {
        "down_deviation_atr": "cluster_deviation_atr",
        "down_deviation_pct": "cluster_deviation_atr",
        "lower_shadow_atr": "cluster_lower_shadow_shock",
        "lower_shadow_pct": "cluster_lower_shadow_shock",
        "core_signal": "cluster_lower_shadow_shock",
        "intraday_repair": "cluster_close_repair_quality",
        "upper_shadow_atr": "cluster_no_upper_pressure",
        "upper_shadow_pct": "cluster_no_upper_pressure",
        "trend_state": "cluster_stock_state",
        "vol_state": "cluster_stock_state",
        "vol_state_pct": "cluster_stock_state",
        "amount_shock": "cluster_liquidity_shock",
        "liquidity_log_amount_20": "cluster_liquidity_shock",
        "market_ret_1d": "cluster_market_context",
        "market_ret_5d": "cluster_market_context",
        "industry_ret_1d": "cluster_market_context",
        "industry_ret_5d": "cluster_market_context",
        "stock_minus_industry_5d": "cluster_market_context",
        "limit_flag": "limit_risk",
        "near_down_limit_flag": "limit_risk",
    }
    return raw_map.get(feature, "other")


def _apply_defensive_gate(controls: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    r = regime[["trade_date", *REGIME_COLS]].copy()
    out = controls.merge(r, on="trade_date", how="left")
    out["top5_excess_5round"] = out["top5_excess_rolling"]
    out["defensive_gate"] = out.apply(_risk_kill_only_gate, axis=1).astype(float)
    out["strategy_gate"] = out["strategy_gate"].astype(float) * out["defensive_gate"]
    return out


def _cluster_ic_by_state(dataset: pd.DataFrame, states: pd.DataFrame, fold: dict, variant: str) -> pd.DataFrame:
    test = dataset[
        dataset["datetime"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))
        & dataset["pit_top1000"].fillna(False).astype(bool)
    ][["datetime", "instrument", "label", *CLUSTER_COLS]].dropna(subset=["label"])
    s = states[["trade_date", "predicted_state"]].rename(columns={"trade_date": "datetime"})
    test = test.merge(s, on="datetime", how="left")
    rows = []
    for (year, state), g in test.groupby([test["datetime"].dt.year, "predicted_state"], dropna=False):
        for cluster in CLUSTER_COLS:
            h = g[[cluster, "label"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(h) < 100 or h[cluster].nunique() < 5:
                ic = np.nan
                spread = np.nan
            else:
                ic = float(h[cluster].corr(h["label"], method="spearman"))
                ranked = h[cluster].rank(method="first")
                bucket = pd.qcut(ranked, 5, labels=False, duplicates="drop")
                by_bucket = h.assign(bucket=bucket).groupby("bucket", observed=True)["label"].mean()
                spread = float(by_bucket.loc[by_bucket.index.max()] - by_bucket.loc[by_bucket.index.min()]) if len(by_bucket) >= 2 else np.nan
            rows.append(
                {
                    "variant": variant,
                    "fold": fold["name"],
                    "year": int(year),
                    "predicted_state": None if pd.isna(state) else int(state),
                    "cluster": cluster,
                    "rank_ic": ic,
                    "decile_spread": spread,
                    "obs": int(len(h)),
                }
            )
    return pd.DataFrame(rows)


def _gain_summary(gains: pd.DataFrame) -> pd.DataFrame:
    if gains.empty:
        return pd.DataFrame()
    return (
        gains.groupby(["variant", "fold", "feature_group"], dropna=False)
        .agg(gain=("gain", "sum"))
        .reset_index()
        .assign(gain_share=lambda d: d["gain"] / d.groupby(["variant", "fold"])["gain"].transform("sum"))
        .sort_values(["variant", "fold", "gain_share"], ascending=[True, True, False])
    )


def _cluster_effectiveness_summary(cluster_ic: pd.DataFrame) -> pd.DataFrame:
    if cluster_ic.empty:
        return pd.DataFrame()
    return (
        cluster_ic.groupby(["variant", "year", "cluster"], dropna=False)
        .agg(
            mean_rank_ic=("rank_ic", "mean"),
            mean_spread=("decile_spread", "mean"),
            states=("predicted_state", "nunique"),
            obs=("obs", "sum"),
        )
        .reset_index()
        .sort_values(["variant", "year", "mean_rank_ic"], ascending=[True, True, False])
    )


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


def _report(
    metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    summary: pd.DataFrame,
    gains: pd.DataFrame,
    cluster_effective: pd.DataFrame,
) -> str:
    s = _fmt_pct(summary, ["mean_ann", "median_ann", "mean_excess", "median_excess", "worst_drawdown", "mean_exposure", "mean_flip_ratio"])
    s["mean_sharpe"] = s["mean_sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    show = metrics[[
        "variant", "fold", "annualized_return", "benchmark_annualized_return", "annualized_excess_return",
        "sharpe", "max_drawdown", "avg_exposure", "flip_ratio", "defensive_flat_ratio", "trade_count",
    ]].copy()
    show = _fmt_pct(show, [
        "annualized_return", "benchmark_annualized_return", "annualized_excess_return", "max_drawdown",
        "avg_exposure", "flip_ratio", "defensive_flat_ratio",
    ])
    show["sharpe"] = show["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    y = yearly.copy()
    y = _fmt_pct(y, ["return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"])
    top_gain = gains.groupby(["variant", "feature_group"], dropna=False)["gain_share"].mean().reset_index()
    top_gain = top_gain.sort_values(["variant", "gain_share"], ascending=[True, False]).groupby("variant").head(8)
    top_gain = _fmt_pct(top_gain, ["gain_share"])
    effective = cluster_effective.copy()
    effective["mean_rank_ic"] = effective["mean_rank_ic"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    effective["mean_spread"] = effective["mean_spread"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    return "\n".join(
        [
            "# Regime-Aware Factor Cluster Matrix",
            "",
            "- universe: allowed main-board subset inside PIT rolling liquidity Top1000",
            "- clusters: deviation, lower-shadow shock, close repair, no-upper pressure, stock state, liquidity shock, market context",
            "- regime-aware model: cluster + regime + cluster*regime interactions",
            "- execution: frozen fit-quality flip, HMM rolling_3y tiered gate; defensive variant also applies risk-kill gate",
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
            "## Mean Gain By Feature Group",
            "",
            top_gain.to_markdown(index=False),
            "",
            "## Cluster IC By Year",
            "",
            effective.to_markdown(index=False),
            "",
        ]
    )


if __name__ == "__main__":
    main()
