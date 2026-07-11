"""Frozen M0--M7 comparison for Bollinger lower-band rejection research."""

from __future__ import annotations

import json
import time
import gc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository

from .atr_reversion_config import ATRReversionPipelineConfig, load_atr_reversion_config
from .atr_reversion_dataset import FEATURE_GROUPS, REQUIRED_PANEL_COLUMNS, build_atr_reversion_dataset
from .supply_ic import compute_factor_ic


# These variants deliberately reproduce the pre-registered comparison in the
# research conversation.  Do not tune a variant's features or model parameters
# independently during this first selection round.
MODEL_SPECS: dict[str, tuple[str, ...]] = {
    "M0_shape": ("S",),
    "M1_shape_price": ("S", "P"),
    "M2_shape_volatility": ("S", "V"),
    "M3_shape_flow": ("S", "F"),
    "M4_first_order_full": ("S", "P", "V", "F"),
    "M5_full": ("S", "P", "V", "F", "A"),
    "M6_full_minus_flow": ("S", "P", "V", "A"),
    "M7_full_minus_acceleration": ("S", "P", "V", "F"),
}


def run_atr_reversion(config_path: str | Path, model_variants: list[str] | None = None) -> dict:
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
    # A cache is intentionally keyed by the data version and experiment name.  The
    # earlier generic cache pre-dated this feature specification and is not reusable.
    version = _resolve_version(cfg)
    cache = cfg.output_root / f"{cfg.name}_{version}_dataset.parquet"
    if cfg.cache_dataset and cache.exists():
        dataset = pd.read_parquet(cache)
        feature_names = FEATURE_GROUPS["all"]
        log(f"loaded cached dataset rows={len(dataset):,}")
    else:
        version, panel = _load_panel(cfg)
        log(f"loaded panel rows={len(panel):,} version={version}")
        log(f"universe stocks={panel['ts_code'].nunique():,}")
        log("building point-in-time Bollinger event dataset")
        dataset, feature_names = build_atr_reversion_dataset(panel, cfg.features, cfg.label)
        del panel
        gc.collect()
        if cfg.cache_dataset:
            cfg.output_root.mkdir(parents=True, exist_ok=True)
            dataset.to_parquet(cache, index=False)
            log(f"cached dataset -> {cache}")
        log(f"built dataset rows={len(dataset):,} features={len(feature_names)}")

    dataset = dataset.loc[dataset["event_pool"].eq(True)].copy()
    gc.collect()
    log(
        f"event pool touch_depth_atr >= {cfg.features.event_pool_threshold:g}: "
        f"rows={len(dataset):,}, stocks={dataset['instrument'].nunique():,}"
    )
    available_groups = _available_groups(dataset)
    log(f"available feature groups={available_groups}")

    eval_start, eval_end = cfg.segments.test.start, cfg.segments.test.end
    factor_ic = compute_factor_ic(
        dataset, [f for f in feature_names if f not in FEATURE_GROUPS["F"] or available_groups["F"]],
        eval_start, eval_end, label_col="label", n_lag=max(1, cfg.label.primary_horizon - 1),
    )
    factor_ic.to_csv(output / "atomic_feature_ic.csv", encoding="utf-8-sig")
    _touch_wick_heatmap(dataset, eval_start, eval_end).to_csv(output / "touch_wick_2d_return.csv", encoding="utf-8-sig")

    model_results: dict[str, dict] = {}
    comparison_rows: list[dict] = []
    names = model_variants if model_variants is not None else cfg.model_variants
    selected_specs = MODEL_SPECS if names is None else {
        name: MODEL_SPECS[name] for name in names
    }
    for model_name, groups in selected_specs.items():
        missing = [group for group in groups if not available_groups[group]]
        cols = _columns_for(groups)
        if missing:
            reason = f"unavailable: required feature group(s) {', '.join(missing)} have no stock-level net-flow data"
            result = {"status": "unavailable", "reason": reason, "groups": list(groups), "features": cols}
            model_results[model_name] = result
            comparison_rows.append({"model": model_name, "status": "unavailable", "reason": reason, "groups": "+".join(groups)})
            log(f"skip {model_name}: {reason}")
            continue

        log(f"training {model_name}: groups={'+'.join(groups)} features={len(cols)} seeds={cfg.random_seeds}")
        seeds = [_fit_predict_metrics(dataset, cols, cfg, seed, log) for seed in cfg.random_seeds]
        seed_frame = pd.DataFrame(seeds)
        seed_frame.to_csv(output / f"{model_name}_seed_metrics.csv", index=False, encoding="utf-8-sig")
        result = _aggregate_seed_metrics(seed_frame, cols, groups)
        model_results[model_name] = result
        comparison_rows.append({"model": model_name, "status": "ok", "reason": "", "groups": "+".join(groups), **result})
        log(
            f"finished {model_name}: RankIC={result['rank_ic_mean']:.4f} +/- {result['rank_ic_mean_std']:.4f}, "
            f"ICIR={result['rank_ic_ir']:.2f}, Top10={result['top10_mean_excess']:.4%}"
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output / "model_comparison.csv", index=False, encoding="utf-8-sig")
    summary = {
        "name": cfg.name,
        "data_version": version,
        "run_dir": str(output),
        "event_pool": f"touch_depth_atr >= {cfg.features.event_pool_threshold}",
        "rows_in_event_pool": int(len(dataset)),
        "features": feature_names,
        "available_feature_groups": available_groups,
        "random_seeds": cfg.random_seeds,
        "model_results": model_results,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    (output / "report.md").write_text(_report(cfg, version, factor_ic, comparison, available_groups), encoding="utf-8")
    log("wrote atomic_feature_ic.csv, touch_wick_2d_return.csv, model_comparison.csv, report.md")
    return summary


def _available_groups(dataset: pd.DataFrame) -> dict[str, bool]:
    return {group: bool(dataset[columns].notna().any().all()) for group, columns in FEATURE_GROUPS.items() if group != "all"}


def _columns_for(groups: tuple[str, ...]) -> list[str]:
    return [column for group in groups for column in FEATURE_GROUPS[group]]


def _fit_predict_metrics(
    dataset: pd.DataFrame, features: list[str], cfg: ATRReversionPipelineConfig, seed: int, log
) -> dict:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover - environment-specific dependency
        raise RuntimeError("lightgbm is required; install project optional dependency .[ml]") from exc

    train = _slice(dataset, cfg.segments.train.start, cfg.segments.train.end, features)
    valid = _slice(dataset, cfg.segments.valid.start, cfg.segments.valid.end, features)
    test = _slice(dataset, cfg.segments.test.start, cfg.segments.test.end, features)
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError(f"empty model slice train={len(train)} valid={len(valid)} test={len(test)}")
    params = cfg.model.model_dump()
    params.pop("objective", None)
    params.update(random_state=seed, verbosity=-1, force_col_wise=True)
    model = LGBMRegressor(objective="regression", **params)
    model.fit(
        train[features], train["label"],
        sample_weight=train["sample_weight"].fillna(1.0),
        eval_set=[(valid[features], valid["label"])],
    )
    joined = test[["datetime", "instrument", "label"]].copy()
    joined["score"] = model.predict(test[features])
    daily_ic = _daily_rank_ic(joined)
    top = _topn_excess(joined, [5, 10, 20])
    return {
        "seed": seed,
        "rank_ic_mean": float(daily_ic.mean()),
        "rank_ic_ir": _annualized_ir(daily_ic),
        "rank_ic_positive_ratio": float((daily_ic > 0).mean()),
        "n_days": int(len(daily_ic)),
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "n_test": int(len(test)),
        "top5_mean_excess": top[5],
        "top10_mean_excess": top[10],
        "top20_mean_excess": top[20],
    }


def _slice(dataset: pd.DataFrame, start: str, end: str, features: list[str]) -> pd.DataFrame:
    cols = ["datetime", "instrument", *features, "label", "sample_weight"]
    return dataset.loc[dataset["datetime"].between(pd.Timestamp(start), pd.Timestamp(end)), cols].dropna(subset=[*features, "label"])


def _daily_rank_ic(joined: pd.DataFrame, min_daily_n: int = 20) -> pd.Series:
    daily = joined.groupby("datetime").apply(
        lambda g: g["score"].corr(g["label"], method="spearman") if len(g) >= min_daily_n else np.nan,
        include_groups=False,
    ).dropna()
    daily.index = pd.to_datetime(daily.index)
    return daily


def _annualized_ir(daily: pd.Series) -> float:
    std = daily.std(ddof=1)
    return float(daily.mean() / std * np.sqrt(252)) if len(daily) > 1 and std > 0 else np.nan


def _topn_excess(joined: pd.DataFrame, top_ns: list[int]) -> dict[int, float]:
    universe = joined.groupby("datetime")["label"].mean()
    out: dict[int, float] = {}
    for top_n in top_ns:
        top = joined.sort_values(["datetime", "score"], ascending=[True, False]).groupby("datetime").head(top_n)
        out[top_n] = float((top.groupby("datetime")["label"].mean() - universe).mean())
    return out


def _aggregate_seed_metrics(frame: pd.DataFrame, features: list[str], groups: tuple[str, ...]) -> dict:
    metrics = {}
    for metric in ("rank_ic_mean", "rank_ic_ir", "rank_ic_positive_ratio", "top5_mean_excess", "top10_mean_excess", "top20_mean_excess"):
        metrics[metric] = float(frame[metric].mean())
        metrics[f"{metric}_std"] = float(frame[metric].std(ddof=0))
    metrics.update({
        "groups": list(groups), "features": features, "n_features": len(features), "n_seeds": int(len(frame)),
        "n_train": int(frame["n_train"].iloc[0]), "n_valid": int(frame["n_valid"].iloc[0]),
        "n_test": int(frame["n_test"].iloc[0]), "n_days": int(frame["n_days"].iloc[0]),
    })
    return metrics


def _touch_wick_heatmap(dataset: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    sub = dataset.loc[dataset["datetime"].between(pd.Timestamp(start), pd.Timestamp(end)), ["datetime", "touch_depth_atr", "lower_wick_share", "label"]].dropna().copy()
    if sub.empty:
        return pd.DataFrame()
    for column, name in (("touch_depth_atr", "touch_q"), ("lower_wick_share", "wick_q")):
        sub[name] = sub.groupby("datetime")[column].transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
    cells = sub.dropna().groupby(["datetime", "touch_q", "wick_q"])["label"].mean().reset_index()
    return cells.groupby(["touch_q", "wick_q"])["label"].mean().unstack("wick_q")


def _load_panel(cfg: ATRReversionPipelineConfig) -> tuple[str, pd.DataFrame]:
    project = load_project(cfg.project_config)
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, manifest = repo.load_manifest(cfg.data_version)
    if cfg.require_full_segment_coverage and (
        pd.Timestamp(manifest["start_date"]) > pd.Timestamp(cfg.segments.train.start)
        or pd.Timestamp(manifest["end_date"]) < pd.Timestamp(cfg.segments.test.end)
    ):
        raise ValueError(f"data version {version} does not cover configured train/test segments")
    # Reading the full 11m-row, 30+ column panel just to select a liquid subset
    # exceeds ordinary workstation memory.  Read two columns for the fixed universe,
    # then predicate-push the selected stock codes into the full feature read.
    path = repo.root / "versions" / version / "curated" / "stock_daily_panel.parquet"
    if cfg.universe_top_n:
        amount = pd.read_parquet(path, columns=["ts_code", "amount_cny"])
        keep = amount.groupby("ts_code")["amount_cny"].median().nlargest(cfg.universe_top_n).index.tolist()
        del amount
        import pyarrow.parquet as pq

        schema_names = set(pq.ParquetFile(path).schema_arrow.names)
        flow_candidates = {"net_flow_cny", "net_mf_amount", "main_net_inflow", "main_net_amount", "net_amount"}
        selected_columns = sorted(REQUIRED_PANEL_COLUMNS | (flow_candidates & schema_names))
        panel = pd.read_parquet(path, columns=selected_columns, filters=[("ts_code", "in", keep)])
    else:
        panel = pd.read_parquet(path, columns=sorted(REQUIRED_PANEL_COLUMNS))
    return version, panel


def _resolve_version(cfg: ATRReversionPipelineConfig) -> str:
    project = load_project(cfg.project_config)
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    return repo.resolve(cfg.data_version)


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"not serializable: {type(obj)}")


def _report(cfg, version: str, factor_ic: pd.DataFrame, comparison: pd.DataFrame, available_groups: dict[str, bool]) -> str:
    lines = [
        "# Bollinger Lower-Band Rejection: M0-M7 Frozen Comparison",
        "",
        f"- Data version: `{version}`",
        f"- Event pool: `touch_depth_atr >= {cfg.features.event_pool_threshold:g}`",
        f"- Label: next-open to +{cfg.label.primary_horizon}-day open, industry-neutral={cfg.label.industry_neutralize}",
        f"- Seeds: {', '.join(map(str, cfg.random_seeds))}",
        "- All executable models share identical time splits and LightGBM parameters.",
        "",
        "## Feature availability",
        "",
        "|Group|Available|Definition|",
        "|---|---|---|",
        "|S|yes|Bollinger touch/reclaim, wick geometry, close quality|",
        "|P|yes|3-day ATR-normalized price velocity|",
        "|V|yes|ATR level percentile and velocity|",
        f"|F|{'yes' if available_groups['F'] else 'no'}|net-flow intensity/change plus amount ratio|",
        "|A|yes|ATR and price acceleration|",
        "",
    ]
    if not available_groups["F"]:
        lines.extend([
            "> The published daily panel has no stock-level net-flow field. Flow variants are intentionally marked unavailable; amount is not substituted for net flow.",
            "",
        ])
    lines.extend([
        "## Atomic feature test Rank IC",
        "",
        factor_ic.sort_values("rank_ic_mean", ascending=False).round(5).to_markdown(),
        "",
        "## Model comparison",
        "",
        comparison.drop(columns=["features"], errors="ignore").round(6).to_markdown(index=False),
        "",
        "`topN_mean_excess` is mean five-day forward industry-neutral excess return of each daily top-N versus the event-pool universe. It is not annualized because five-day labels overlap.",
    ])
    return "\n".join(lines) + "\n"
