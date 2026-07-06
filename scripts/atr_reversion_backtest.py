"""Train ATR reversion LightGBM scores and run TopN rolling-hold backtests."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.backtest.engine import BacktestEngine
from factor_forge.config import CostModel, ExecutionConstraints, load_project
from factor_forge.data.repository import DataVersionRepository
from factor_forge.ml.atr_reversion_config import load_atr_reversion_config
from factor_forge.ml.atr_reversion_dataset import FEATURE_GROUPS, build_atr_reversion_dataset


TOP_NS = [50, 100, 200]
HOLDING_DAYS = [3, 5, 10]
COST_BPS = [0, 10, 20, 30]


def main(config_path: str = "configs/ml/atr_reversion_lightgbm_v1.yaml") -> None:
    cfg = load_atr_reversion_config(config_path)
    output = cfg.output_root / f"{cfg.name}_backtest_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"start config={config_path}")
    version, panel = _load_panel(cfg)
    log(f"loaded panel rows={len(panel):,} version={version}")
    if cfg.universe_top_n:
        keep = panel.groupby("ts_code")["amount_cny"].median().nlargest(cfg.universe_top_n).index
        panel = panel[panel["ts_code"].isin(keep)].copy()
        log(f"selected top{cfg.universe_top_n} rows={len(panel):,} stocks={panel['ts_code'].nunique():,}")

    dataset, features = _load_or_build_dataset(cfg, panel, log)
    predictions = _train_predict(dataset, features, cfg, log)
    pred_path = output / "predictions_all_features.parquet"
    predictions.to_parquet(pred_path, index=False)
    log(f"wrote predictions -> {pred_path}")

    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    test_panel = panel[
        panel["trade_date"].between(pd.Timestamp(cfg.segments.test.start), pd.Timestamp(cfg.segments.test.end))
    ].copy()
    log(f"test panel rows={len(test_panel):,}")

    rows = []
    engine = BacktestEngine()
    for holding in HOLDING_DAYS:
        for top_n in TOP_NS:
            for cost in COST_BPS:
                log(f"backtest top_n={top_n} holding={holding} cost_bps={cost}")
                result = engine.run(
                    test_panel,
                    predictions,
                    universe="tradeable",
                    top_n=top_n,
                    holding_days=holding,
                    initial_cash=10_000_000,
                    lot_size=100,
                    constraints=ExecutionConstraints(),
                    cost_model=CostModel(),
                    cost_scenario_bps=cost,
                )
                tag = f"top{top_n}_hold{holding}_cost{cost}"
                result.daily.to_parquet(output / f"daily_{tag}.parquet", index=False)
                metrics = {k: _json_value(v) for k, v in result.metrics.items()}
                metrics.update({
                    "top_n": top_n,
                    "holding_days": holding,
                    "cost_bps": cost,
                    "avg_daily_turnover": float(result.daily["portfolio_turnover"].mean()),
                    "avg_holding_count": float(result.daily["holding_count"].mean()),
                    "avg_cash_ratio": float(result.daily["cash_ratio"].mean()),
                })
                rows.append(metrics)
                log(
                    "done "
                    f"ann={metrics['annualized_return']:.2%} "
                    f"excess={metrics['annualized_excess_return']:.2%} "
                    f"sharpe={metrics['sharpe'] if metrics['sharpe'] is not None else np.nan:.2f} "
                    f"maxdd={metrics['max_drawdown']:.2%}"
                )

    metrics_df = pd.DataFrame(rows).sort_values(
        ["annualized_excess_return", "sharpe"], ascending=False
    )
    metrics_df.to_csv(output / "backtest_grid_metrics.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "name": cfg.name,
                "data_version": version,
                "run_dir": str(output),
                "prediction_path": str(pred_path),
                "best": metrics_df.head(10).to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_value,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(metrics_df, version), encoding="utf-8")
    log("wrote backtest_grid_metrics.csv, summary.json, report.md")
    log("done")
    print(f"run_dir={output}")


def _load_or_build_dataset(cfg, panel: pd.DataFrame, log) -> tuple[pd.DataFrame, list[str]]:
    cache = (
        cfg.output_root
        / f"atr_reversion_ic_dataset_top{cfg.universe_top_n or 'all'}_p{cfg.features.percentile_window}_h{cfg.label.primary_horizon}.parquet"
    )
    if cfg.cache_dataset and cache.exists():
        log(f"loading cached dataset {cache}")
        return pd.read_parquet(cache), FEATURE_GROUPS["all"]
    log("building ATR reversion dataset")
    dataset, features = build_atr_reversion_dataset(panel, cfg.features, cfg.label)
    if cfg.cache_dataset:
        cfg.output_root.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(cache, index=False)
        log(f"cached dataset -> {cache}")
    return dataset, features


def _train_predict(dataset: pd.DataFrame, features: list[str], cfg, log) -> pd.DataFrame:
    from lightgbm import LGBMRegressor

    cols = ["datetime", "instrument", *features, "label", "sample_weight"]
    train = dataset.loc[
        dataset["datetime"].between(pd.Timestamp(cfg.segments.train.start), pd.Timestamp(cfg.segments.train.end)),
        cols,
    ].dropna(subset=[*features, "label"])
    valid = dataset.loc[
        dataset["datetime"].between(pd.Timestamp(cfg.segments.valid.start), pd.Timestamp(cfg.segments.valid.end)),
        cols,
    ].dropna(subset=[*features, "label"])
    test = dataset.loc[
        dataset["datetime"].between(pd.Timestamp(cfg.segments.test.start), pd.Timestamp(cfg.segments.test.end)),
        cols,
    ].dropna(subset=features)
    log(f"model slices train={len(train):,} valid={len(valid):,} test_predictable={len(test):,}")

    params = cfg.model.model_dump()
    params.pop("objective", None)
    params.setdefault("verbosity", -1)
    params.setdefault("force_col_wise", True)
    model = LGBMRegressor(objective="regression", **params)
    fit_kwargs = {}
    if "sample_weight" in train:
        fit_kwargs["sample_weight"] = train["sample_weight"].fillna(1.0)
    eval_set = [(valid[features], valid["label"])] if len(valid) else None
    log("fitting LightGBM all_features")
    model.fit(train[features], train["label"], eval_set=eval_set, **fit_kwargs)
    log("predicting test scores")
    out = test[["datetime", "instrument"]].copy()
    out["factor_value"] = model.predict(test[features])
    return out.rename(columns={"datetime": "trade_date", "instrument": "ts_code"})


def _load_panel(cfg) -> tuple[str, pd.DataFrame]:
    project = load_project(cfg.project_config)
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest(cfg.data_version)
    _, panel = repo.load_panel(version)
    return version, panel


def _json_value(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _report(metrics: pd.DataFrame, version: str) -> str:
    show_cols = [
        "top_n",
        "holding_days",
        "cost_bps",
        "annualized_return",
        "benchmark_annualized_return",
        "annualized_excess_return",
        "sharpe",
        "max_drawdown",
        "avg_daily_turnover",
        "execution_rate",
    ]
    table = metrics[show_cols].head(20).copy()
    for col in [
        "annualized_return",
        "benchmark_annualized_return",
        "annualized_excess_return",
        "max_drawdown",
        "avg_daily_turnover",
        "execution_rate",
    ]:
        table[col] = table[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    table["sharpe"] = table["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    return "\n".join([
        "# ATR Reversion LightGBM Backtest",
        "",
        f"- data version: `{version}`",
        "- signal: all_features LightGBM score",
        "- execution: T close signal, T+1 open buy, overlapping equal-cash sleeves",
        "- benchmark: tradeable universe equal-weight open-to-open",
        "",
        "## Top Results",
        "",
        table.to_markdown(index=False),
        "",
    ])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/ml/atr_reversion_lightgbm_v1.yaml")

