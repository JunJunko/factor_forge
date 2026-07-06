"""IC and LightGBM runner for ATR lower-shadow reversion research."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository

from .atr_reversion_config import ATRReversionPipelineConfig, load_atr_reversion_config
from .atr_reversion_dataset import CORE_FEATURES, FEATURE_GROUPS, build_atr_reversion_dataset
from .supply_ic import compute_factor_ic, quantile_decomp, yearly_ic


def run_atr_reversion(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    cfg = load_atr_reversion_config(config_path)
    output = cfg.output_root / f"{cfg.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"start config={config_path}")
    log("loading panel")
    version, panel = _load_panel(cfg)
    log(f"loaded panel rows={len(panel):,} version={version}")
    if cfg.universe_top_n:
        log(f"selecting liquid universe top_n={cfg.universe_top_n}")
        keep = panel.groupby("ts_code")["amount_cny"].median().nlargest(cfg.universe_top_n).index
        panel = panel[panel["ts_code"].isin(keep)].copy()
        log(f"universe rows={len(panel):,} stocks={panel['ts_code'].nunique():,}")

    cache = (
        cfg.output_root
        / f"atr_reversion_ic_dataset_top{cfg.universe_top_n or 'all'}_p{cfg.features.percentile_window}_h{cfg.label.primary_horizon}.parquet"
    )
    if cfg.cache_dataset and cache.exists():
        log(f"loading cached dataset {cache}")
        dataset = pd.read_parquet(cache)
        feature_names = FEATURE_GROUPS["all"]
        log(f"loaded dataset rows={len(dataset):,}")
    else:
        log("building ATR reversion dataset")
        dataset, feature_names = build_atr_reversion_dataset(panel, cfg.features, cfg.label)
        log(f"built dataset rows={len(dataset):,} features={len(feature_names)}")
        if cfg.cache_dataset:
            cfg.output_root.mkdir(parents=True, exist_ok=True)
            dataset.to_parquet(cache, index=False)
            log(f"cached dataset -> {cache}")

    eval_start, eval_end = cfg.segments.test.start, cfg.segments.test.end
    log(f"computing factor IC test={eval_start}..{eval_end}")
    ic = compute_factor_ic(
        dataset,
        feature_names,
        eval_start,
        eval_end,
        label_col="label",
        n_lag=max(1, cfg.label.primary_horizon - 1),
    )
    ic.to_csv(output / "factor_ic.csv", encoding="utf-8-sig")
    log("wrote factor_ic.csv")

    q = quantile_decomp(dataset, "core_signal", "label", eval_start, eval_end, n_bins=10)
    q.to_csv(output / "core_signal_deciles.csv", encoding="utf-8-sig")
    log("wrote core_signal_deciles.csv")
    yic = yearly_ic(dataset, "core_signal", "label", eval_start, eval_end)
    yic.to_csv(output / "core_signal_yearly_ic.csv", encoding="utf-8-sig")
    log("wrote core_signal_yearly_ic.csv")

    model_results = {}
    for name, cols in {"core_only": CORE_FEATURES, "all_features": feature_names}.items():
        log(f"training LightGBM model={name} features={len(cols)}")
        model_results[name] = _fit_predict_ic(dataset, cols, cfg, log=log)
        log(f"finished model={name}")

    summary = {
        "name": cfg.name,
        "data_version": version,
        "run_dir": str(output),
        "rows": int(len(dataset)),
        "features": feature_names,
        "factor_ic_top": ic.sort_values("rank_ic_mean", ascending=False)
        .head(8)
        .reset_index()
        .to_dict("records"),
        "model_results": model_results,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (output / "report.md").write_text(
        _report(cfg, version, ic, q, yic, model_results),
        encoding="utf-8",
    )
    log("wrote summary.json and report.md")
    log("done")
    return summary


def _fit_predict_ic(
    dataset: pd.DataFrame,
    features: list[str],
    cfg: ATRReversionPipelineConfig,
    log=None,
) -> dict:
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        return {"error": "lightgbm is not installed; install project optional dependency .[ml]"}

    train = _slice(dataset, cfg.segments.train.start, cfg.segments.train.end, features)
    valid = _slice(dataset, cfg.segments.valid.start, cfg.segments.valid.end, features)
    test = _slice(dataset, cfg.segments.test.start, cfg.segments.test.end, features)
    if log:
        log(f"model slices train={len(train):,} valid={len(valid):,} test={len(test):,}")
    if len(train) == 0 or len(test) == 0:
        return {
            "error": "empty train or test slice after feature/label intersection",
            "n_train": int(len(train)),
            "n_valid": int(len(valid)),
            "n_test": int(len(test)),
        }
    params = cfg.model.model_dump()
    params.pop("objective", None)
    params.setdefault("verbosity", -1)
    params.setdefault("force_col_wise", True)
    model = LGBMRegressor(objective="regression", **params)
    fit_kwargs = {}
    if "sample_weight" in train:
        fit_kwargs["sample_weight"] = train["sample_weight"].fillna(1.0)
    eval_set = [(valid[features], valid["label"])] if len(valid) else None
    if log:
        log("fitting LightGBM")
    model.fit(train[features], train["label"], eval_set=eval_set, **fit_kwargs)
    if log:
        log("predicting test")
    pred = pd.Series(model.predict(test[features]), index=test.index, name="score")
    joined = test[["datetime", "instrument", "label"]].copy()
    joined["score"] = pred
    daily = joined.groupby("datetime").apply(
        lambda g: g["score"].corr(g["label"], method="spearman") if len(g) >= 20 else np.nan,
        include_groups=False,
    ).dropna()
    return {
        "rank_ic_mean": float(daily.mean()) if len(daily) else np.nan,
        "rank_ic_ir": float(daily.mean() / daily.std() * np.sqrt(252)) if len(daily) > 1 and daily.std() > 0 else np.nan,
        "n_days": int(len(daily)),
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "n_test": int(len(test)),
        "feature_importance": dict(zip(features, [float(x) for x in model.feature_importances_])),
    }


def _slice(dataset: pd.DataFrame, start: str, end: str, features: list[str]) -> pd.DataFrame:
    cols = ["datetime", "instrument", *features, "label", "sample_weight"]
    return dataset.loc[
        dataset["datetime"].between(pd.Timestamp(start), pd.Timestamp(end)),
        cols,
    ].dropna(subset=[*features, "label"])


def _load_panel(cfg: ATRReversionPipelineConfig) -> tuple[str, pd.DataFrame]:
    project = load_project(cfg.project_config)
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, manifest = repo.load_manifest(cfg.data_version)
    if cfg.require_full_segment_coverage:
        if (
            pd.Timestamp(manifest["start_date"]) > pd.Timestamp(cfg.segments.train.start)
            or pd.Timestamp(manifest["end_date"]) < pd.Timestamp(cfg.segments.test.end)
        ):
            raise ValueError(f"data version {version} does not cover configured train/test segments")
    _, panel = repo.load_panel(version)
    return version, panel


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"not serializable: {type(obj)}")


def _report(cfg, version, ic, deciles, yearly, model_results) -> str:
    lines = [
        "# ATR Lower-Shadow Reversion Report",
        "",
        f"- data version: `{version}`",
        f"- Train: {cfg.segments.train.start} ~ {cfg.segments.train.end}",
        f"- Valid: {cfg.segments.valid.start} ~ {cfg.segments.valid.end}",
        f"- Test: {cfg.segments.test.start} ~ {cfg.segments.test.end}",
        f"- label: {cfg.label.label_method}, horizon={cfg.label.primary_horizon}, industry_neutral={cfg.label.industry_neutralize}",
        "",
        "## Factor RankIC",
        "",
        ic.sort_values("rank_ic_mean", ascending=False).round(4).to_markdown(),
        "",
        "## CoreSignal Deciles",
        "",
        deciles.round(5).to_markdown(),
        "",
        "## CoreSignal Yearly IC",
        "",
        yearly.round(4).to_markdown(),
        "",
        "## LightGBM Test RankIC",
        "",
        "|model|RankIC|ICIR|train samples|test days|",
        "|---|---:|---:|---:|---:|",
    ]
    for name, res in model_results.items():
        if "error" in res:
            lines.append(f"|{name}|ERROR: {res['error']}||||")
        else:
            lines.append(
                f"|{name}|{res['rank_ic_mean']:.4f}|{res['rank_ic_ir']:.2f}|"
                f"{res['n_train']}|{res['n_days']}|"
            )
    return "\n".join(lines) + "\n"
