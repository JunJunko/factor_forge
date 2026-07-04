from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.backtest.engine import BacktestEngine
from factor_forge.config import CostModel, ExecutionConstraints, load_project
from factor_forge.data.repository import DataVersionRepository
from .config import MLExperimentConfig, load_ml_config
from .dataset import build_dataset, to_qlib_frame


class MLExperimentRunner:
    def run(self, config_path: str | Path, *, estimator=None) -> dict:
        config_path = Path(config_path)
        cfg = load_ml_config(config_path)
        project = load_project(cfg.project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        version, manifest = repository.load_manifest(cfg.data_version)
        available_start = pd.Timestamp(manifest["start_date"])
        available_end = pd.Timestamp(manifest["end_date"])
        required_start = pd.Timestamp(cfg.segments.train.start)
        required_end = pd.Timestamp(cfg.segments.test.end)
        # Segment boundaries are calendar dates while manifests contain the
        # first/last trading date.  A short holiday/weekend gap is coverage,
        # not missing market data (e.g. 2016-01-01 -> 2016-01-04).
        boundary_tolerance = pd.Timedelta(days=7)
        if cfg.require_full_segment_coverage and (
            available_start > required_start + boundary_tolerance
            or available_end < required_end - boundary_tolerance
        ):
            raise ValueError(
                "data version does not fully cover configured segments: "
                f"required {required_start.date()}..{required_end.date()}, "
                f"available {available_start.date()}..{available_end.date()} "
                f"({version})"
            )
        _, panel = repository.load_panel(version)
        dataset, feature_names = build_dataset(panel, cfg.features, cfg.label)
        masks = {name: dataset["datetime"].between(pd.Timestamp(seg.start), pd.Timestamp(seg.end)) for name, seg in (("train", cfg.segments.train), ("valid", cfg.segments.valid), ("test", cfg.segments.test))}
        universe_column = f"is_{cfg.portfolio.universe}"
        if universe_column not in dataset:
            raise ValueError(f"dataset does not contain {universe_column}")
        usable = dataset[feature_names + ["label"]].notna().all(axis=1) & dataset[universe_column].eq(True)
        if not (masks["train"] & usable).any() or not (masks["test"] & usable).any():
            raise ValueError("train or test segment has no complete samples; check dates and lookbacks")
        model = estimator or self._lightgbm(cfg)
        train = masks["train"] & usable
        valid = masks["valid"] & usable
        fit_kwargs = {}
        if valid.any():
            fit_kwargs["eval_set"] = [(dataset.loc[valid, feature_names], dataset.loc[valid, "label"])]
        model.fit(dataset.loc[train, feature_names], dataset.loc[train, "label"], **fit_kwargs)
        score_mask = masks["test"] & dataset[feature_names].notna().all(axis=1)
        scored = dataset.loc[score_mask, ["datetime", "instrument", "label"]].copy()
        scored["score"] = model.predict(dataset.loc[score_mask, feature_names])
        predictions = scored.rename(columns={"datetime": "trade_date", "instrument": "ts_code", "score": "factor_value"})
        predictions["trade_date"] = pd.to_datetime(predictions["trade_date"])
        test_panel = panel[pd.to_datetime(panel["trade_date"]).between(pd.Timestamp(cfg.segments.test.start), pd.Timestamp(cfg.segments.test.end))].copy()
        result = BacktestEngine().run(test_panel, predictions, universe=cfg.portfolio.universe, top_n=cfg.portfolio.top_n, holding_days=cfg.portfolio.holding_days, initial_cash=cfg.portfolio.initial_cash, lot_size=cfg.portfolio.lot_size, constraints=ExecutionConstraints(), cost_model=CostModel(), cost_scenario_bps=cfg.portfolio.cost_bps)
        daily_ic = scored.groupby("datetime").apply(lambda x: x["score"].corr(x["label"], method="spearman") if len(x) >= 2 else np.nan, include_groups=False).rename("rank_ic")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + hashlib.sha256(config_path.read_bytes()).hexdigest()[:8]
        out = cfg.output_root / f"{cfg.name}_{run_id}"
        out.mkdir(parents=True, exist_ok=False)
        predictions.to_parquet(out / "predictions.parquet", index=False)
        to_qlib_frame(dataset.loc[masks["train"] | masks["valid"] | masks["test"]], feature_names).to_parquet(out / "qlib_dataset.parquet")
        daily_ic.to_frame().to_parquet(out / "daily_rank_ic.parquet")
        result.daily.to_parquet(out / "portfolio_daily.parquet", index=False)
        result.trades.to_parquet(out / "trades.parquet", index=False)
        summary = {**result.metrics, "rank_ic_mean": float(daily_ic.mean()), "rank_ic_ir": float(daily_ic.mean() / daily_ic.std()) if daily_ic.std() else None, "data_version": version, "features": feature_names, "run_dir": str(out)}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "config.yaml").write_bytes(config_path.read_bytes())
        return summary

    @staticmethod
    def _lightgbm(cfg: MLExperimentConfig):
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError('LightGBM is required. Install with: python -m pip install -e ".[ml]"') from exc
        return LGBMRegressor(**cfg.model.model_dump())
