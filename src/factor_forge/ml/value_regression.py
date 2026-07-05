from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import ConfigDict, BaseModel, Field, model_validator

from factor_forge.backtest.engine import BacktestEngine
from factor_forge.config import CostModel, ExecutionConstraints, load_project
from factor_forge.data.repository import DataVersionRepository
from .config import PortfolioConfig, Segment, Segments
from .value_dataset import (
    FUNDAMENTAL_FIELDS, VALUE_FEATURES, ValueFeatureParameters,
    attach_point_in_time_fundamentals, build_value_dataset, daily_feature_dependence,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ValueFeatureConfig(_StrictModel):
    fundamental_revision_window: int = Field(default=20, ge=1, le=60)
    momentum_formation: int = Field(default=100, ge=20, le=252)
    momentum_skip: int = Field(default=20, ge=5, le=60)
    pullback_window: int = Field(default=15, ge=5, le=60)
    pullback_skip: int = Field(default=5, ge=1, le=20)
    confirmation_window: int = Field(default=5, ge=2, le=20)
    liquidity_baseline: int = Field(default=20, ge=10, le=120)
    delay_window: int = Field(default=120, ge=60, le=252)
    delay_lags: int = Field(default=5, ge=1, le=10)
    delay_change: int = Field(default=20, ge=5, le=60)
    ridge_alpha: float = Field(default=5.0, ge=0)
    min_industry_size: int = Field(default=20, ge=10)
    mad_scale: float = Field(default=5.0, gt=0)
    minimum_non_null_features: int = Field(default=5, ge=1, le=len(VALUE_FEATURES))

    def parameters(self) -> ValueFeatureParameters:
        return ValueFeatureParameters(**self.model_dump(exclude={"minimum_non_null_features"}))


class ValueLabelConfig(_StrictModel):
    horizons: list[int] = Field(default_factory=lambda: [5, 10, 20])
    excess_over_universe: bool = True
    blend_weights: dict[int, float] = Field(default_factory=lambda: {5: 0.2, 10: 0.3, 20: 0.5})

    @model_validator(mode="after")
    def validate_horizons(self):
        if not self.horizons or len(set(self.horizons)) != len(self.horizons):
            raise ValueError("label horizons must be non-empty and unique")
        if any(item < 1 or item > 60 for item in self.horizons):
            raise ValueError("label horizons must be between 1 and 60")
        if set(self.blend_weights) != set(self.horizons):
            raise ValueError("blend_weights keys must exactly match horizons")
        if any(weight < 0 for weight in self.blend_weights.values()):
            raise ValueError("blend weights must be non-negative")
        total = sum(self.blend_weights.values())
        if total <= 0:
            raise ValueError("blend weights must have a positive sum")
        self.blend_weights = {key: value / total for key, value in self.blend_weights.items()}
        return self


class ValueModelConfig(_StrictModel):
    objective: str = "regression_l1"
    learning_rate: float = Field(default=0.03, gt=0)
    num_leaves: int = Field(default=15, ge=2)
    max_depth: int = Field(default=5, ge=-1)
    n_estimators: int = Field(default=1200, ge=1)
    min_child_samples: int = Field(default=500, ge=1)
    subsample: float = Field(default=0.8, gt=0, le=1)
    subsample_freq: int = Field(default=1, ge=0)
    colsample_bytree: float = Field(default=0.8, gt=0, le=1)
    reg_alpha: float = Field(default=0.2, ge=0)
    reg_lambda: float = Field(default=1.0, ge=0)
    random_state: int = 42
    n_jobs: int = -1
    early_stopping_rounds: int = Field(default=100, ge=0)


class IndependenceConfig(_StrictModel):
    enforce: bool = True
    max_median_abs_spearman: float = Field(default=0.15, ge=0, le=1)
    max_p90_abs_spearman: float = Field(default=0.35, ge=0, le=1)


class ValueRegressionConfig(_StrictModel):
    version: int = 1
    name: str = "value_regression_lightgbm_v1"
    project_config: Path = Path("configs/project.yaml")
    data_version: str = "latest"
    fundamentals_path: Path | None = None
    require_full_segment_coverage: bool = True
    segments: Segments
    features: ValueFeatureConfig = Field(default_factory=ValueFeatureConfig)
    labels: ValueLabelConfig = Field(default_factory=ValueLabelConfig)
    independence: IndependenceConfig = Field(default_factory=IndependenceConfig)
    model: ValueModelConfig = Field(default_factory=ValueModelConfig)
    portfolio: PortfolioConfig = Field(default_factory=lambda: PortfolioConfig(holding_days=20))
    output_root: Path = Path("artifacts/value_regression_runs")


def load_value_regression_config(path: str | Path) -> ValueRegressionConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return ValueRegressionConfig.model_validate(yaml.safe_load(handle) or {})


def _segment_mask(dates: pd.Series, segment: Segment) -> pd.Series:
    return dates.between(pd.Timestamp(segment.start), pd.Timestamp(segment.end))


def _purge_tail(mask: pd.Series, dates: pd.Series, horizon: int) -> pd.Series:
    unique = pd.Index(dates.loc[mask].drop_duplicates().sort_values())
    if len(unique) <= horizon:
        return pd.Series(False, index=mask.index)
    return mask & dates.le(unique[-(horizon + 1)])


def _daily_equal_weights(dates: pd.Series) -> pd.Series:
    return 1.0 / dates.groupby(dates).transform("size")


class ValueRegressionRunner:
    def run(self, config_path: str | Path) -> dict:
        config_path = Path(config_path)
        cfg = load_value_regression_config(config_path)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + hashlib.sha256(config_path.read_bytes()).hexdigest()[:8]
        output = cfg.output_root / f"{cfg.name}_{run_id}"
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.yaml").write_bytes(config_path.read_bytes())
        logger = self._run_logger(output, run_id)
        state = {"stage": "initializing"}
        started = time.perf_counter()

        def progress(stage: str, completed: int, total: int) -> None:
            state["stage"] = stage
            percent = completed / total if total else 1.0
            logger.info(
                "progress stage=%s completed=%d total=%d percent=%.1f elapsed_seconds=%.1f rss_gb=%s",
                stage, completed, total, percent * 100,
                time.perf_counter() - started, self._rss_gb(),
            )

        logger.info(
            "run_started pid=%d config=%s output=%s", os.getpid(), config_path, output
        )
        try:
            result = self._execute(config_path, cfg, output, logger, progress)
            logger.info(
                "run_completed elapsed_seconds=%.1f", time.perf_counter() - started
            )
            return result
        except Exception as exc:
            logger.exception(
                "run_failed stage=%s elapsed_seconds=%.1f",
                state["stage"], time.perf_counter() - started,
            )
            error = {
                "status": "FAILED",
                "stage": state["stage"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
            }
            (output / "error.json").write_text(
                json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            raise

    def _execute(self, config_path, cfg, output, logger, progress) -> dict:
        progress("load_project", 0, 1)
        project = load_project(cfg.project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        version, manifest = repository.load_manifest(cfg.data_version)
        self._check_coverage(cfg, manifest)
        progress("load_panel", 0, 1)
        _, panel = repository.load_panel(version)
        logger.info("panel_loaded rows=%d columns=%d rss_gb=%s", len(panel), len(panel.columns), self._rss_gb())
        progress("attach_pit_fundamentals", 0, 1)
        working_panel = self._attach_fundamentals(panel, cfg)
        logger.info(
            "fundamentals_attached rows=%d columns=%d rss_gb=%s",
            len(working_panel), len(working_panel.columns), self._rss_gb(),
        )
        progress("build_value_features", 0, 1)
        dataset, features, labels = build_value_dataset(
            working_panel, horizons=cfg.labels.horizons,
            parameters=cfg.features.parameters(),
            excess_over_universe=cfg.labels.excess_over_universe,
            progress=progress,
        )
        logger.info("dataset_built rows=%d rss_gb=%s", len(dataset), self._rss_gb())
        dates = dataset["trade_date"]
        masks = {
            "train": _segment_mask(dates, cfg.segments.train),
            "valid": _segment_mask(dates, cfg.segments.valid),
            "test": _segment_mask(dates, cfg.segments.test),
        }
        universe_column = f"is_{cfg.portfolio.universe}"
        if universe_column not in dataset:
            raise ValueError(f"dataset does not contain {universe_column}")
        eligible = dataset[universe_column].eq(True)
        enough_features = dataset[features].notna().sum(axis=1).ge(cfg.features.minimum_non_null_features)
        progress("feature_independence_audit", 0, 1)
        dependence = daily_feature_dependence(dataset.loc[masks["train"]], features)
        dependence.to_csv(output / "feature_dependence.csv", index=False, encoding="utf-8-sig")
        breaches = dependence.loc[
            dependence["median_abs_spearman"].gt(cfg.independence.max_median_abs_spearman)
            | dependence["p90_abs_spearman"].gt(cfg.independence.max_p90_abs_spearman)
        ]
        if cfg.independence.enforce and not breaches.empty:
            detail = breaches.sort_values("p90_abs_spearman", ascending=False).head(5)[
                ["feature_left", "feature_right", "median_abs_spearman", "p90_abs_spearman"]
            ].to_dict("records")
            raise ValueError(f"feature independence thresholds breached: {detail}")
        predictions = dataset.loc[masks["test"] & enough_features, ["trade_date", "ts_code", *labels]].copy()
        importance_rows: list[pd.DataFrame] = []
        horizon_metrics: dict[str, dict] = {}
        models = {}
        for horizon in cfg.labels.horizons:
            progress(f"lightgbm_{horizon}d", 0, 1)
            label = f"label_{horizon}d"
            train = _purge_tail(masks["train"], dates, horizon) & eligible & enough_features & dataset[label].notna()
            valid = _purge_tail(masks["valid"], dates, horizon) & eligible & enough_features & dataset[label].notna()
            if not train.any():
                raise ValueError(f"no usable training samples for {horizon}d label")
            model = self._lightgbm(cfg)
            fit_kwargs = {"sample_weight": _daily_equal_weights(dates.loc[train])}
            if valid.any():
                fit_kwargs["eval_set"] = [(dataset.loc[valid, features], dataset.loc[valid, label])]
                fit_kwargs["eval_sample_weight"] = [_daily_equal_weights(dates.loc[valid])]
                if cfg.model.early_stopping_rounds:
                    import lightgbm as lgb
                    fit_kwargs["callbacks"] = [lgb.early_stopping(cfg.model.early_stopping_rounds, verbose=False)]
            model.fit(dataset.loc[train, features], dataset.loc[train, label], **fit_kwargs)
            progress(f"lightgbm_{horizon}d", 1, 1)
            score_name = f"prediction_{horizon}d"
            predictions[score_name] = model.predict(
                dataset.loc[predictions.index, features],
                num_iteration=getattr(model, "best_iteration_", None),
            )
            importance_rows.append(pd.DataFrame({
                "horizon": horizon, "feature": features,
                "gain_importance": model.booster_.feature_importance(importance_type="gain"),
                "split_importance": model.booster_.feature_importance(importance_type="split"),
            }))
            sample = predictions[["trade_date", label, score_name]].dropna()
            daily_ic = sample.groupby("trade_date").apply(
                lambda frame: frame[score_name].corr(frame[label], method="spearman"),
                include_groups=False,
            ).dropna()
            horizon_metrics[f"{horizon}d"] = {
                "rank_ic_mean": float(daily_ic.mean()) if len(daily_ic) else None,
                "rank_ic_ir": float(daily_ic.mean() / daily_ic.std()) if len(daily_ic) and daily_ic.std() else None,
                "train_rows": int(train.sum()), "valid_rows": int(valid.sum()),
                "test_rows_with_label": int(len(sample)),
            }
            models[horizon] = model

        blend = pd.Series(0.0, index=predictions.index)
        for horizon, weight in cfg.labels.blend_weights.items():
            ranked = predictions.groupby("trade_date")[f"prediction_{horizon}d"].rank(pct=True)
            blend = blend.add(weight * ranked, fill_value=0)
        predictions["prediction_blend"] = blend
        signals = predictions[["trade_date", "ts_code", "prediction_blend"]].rename(columns={"prediction_blend": "factor_value"})
        progress("portfolio_backtest", 0, 1)
        test_panel = panel.loc[pd.to_datetime(panel["trade_date"]).between(
            pd.Timestamp(cfg.segments.test.start), pd.Timestamp(cfg.segments.test.end)
        )].copy()
        backtest = BacktestEngine().run(
            test_panel, signals, universe=cfg.portfolio.universe,
            top_n=cfg.portfolio.top_n, holding_days=cfg.portfolio.holding_days,
            initial_cash=cfg.portfolio.initial_cash, lot_size=cfg.portfolio.lot_size,
            constraints=ExecutionConstraints(), cost_model=CostModel(),
            cost_scenario_bps=cfg.portfolio.cost_bps,
        )
        progress("portfolio_backtest", 1, 1)
        progress("write_artifacts", 0, 1)
        predictions.reset_index(drop=True).to_parquet(output / "predictions.parquet", index=False)
        pd.concat(importance_rows, ignore_index=True).to_csv(output / "feature_importance.csv", index=False, encoding="utf-8-sig")
        backtest.daily.to_parquet(output / "portfolio_daily.parquet", index=False)
        backtest.trades.to_parquet(output / "trades.parquet", index=False)
        for horizon, model in models.items():
            model.booster_.save_model(str(output / f"model_{horizon}d.txt"))
        summary = {
            **backtest.metrics, "horizon_metrics": horizon_metrics,
            "data_version": version, "features": features,
            "blend_weights": cfg.labels.blend_weights,
            "max_median_abs_spearman": float(dependence["median_abs_spearman"].max()) if len(dependence) else None,
            "max_p90_abs_spearman": float(dependence["p90_abs_spearman"].max()) if len(dependence) else None,
            "run_dir": str(output),
        }
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        progress("write_artifacts", 1, 1)
        return summary

    @staticmethod
    def _run_logger(output: Path, run_id: str) -> logging.Logger:
        logger = logging.getLogger(f"factor_forge.value_regression.{run_id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
        )
        file_handler = logging.FileHandler(output / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.handlers[:] = [file_handler, stream_handler]
        return logger

    @staticmethod
    def _rss_gb() -> str:
        try:
            import psutil
            return f"{psutil.Process().memory_info().rss / (1024 ** 3):.2f}"
        except ImportError:
            return "unavailable"

    @staticmethod
    def _attach_fundamentals(panel: pd.DataFrame, cfg: ValueRegressionConfig) -> pd.DataFrame:
        if set(FUNDAMENTAL_FIELDS) <= set(panel.columns):
            return panel
        if cfg.fundamentals_path is None:
            raise ValueError(
                "current panel has no PIT fundamentals; set fundamentals_path to a parquet "
                "with ts_code, available_date and fields: " + ", ".join(FUNDAMENTAL_FIELDS)
            )
        if not cfg.fundamentals_path.exists():
            raise FileNotFoundError(f"fundamentals_path does not exist: {cfg.fundamentals_path}")
        return attach_point_in_time_fundamentals(panel, pd.read_parquet(cfg.fundamentals_path))

    @staticmethod
    def _check_coverage(cfg: ValueRegressionConfig, manifest: dict) -> None:
        if not cfg.require_full_segment_coverage:
            return
        available_start, available_end = pd.Timestamp(manifest["start_date"]), pd.Timestamp(manifest["end_date"])
        tolerance = pd.Timedelta(days=7)
        if (available_start > pd.Timestamp(cfg.segments.train.start) + tolerance
                or available_end < pd.Timestamp(cfg.segments.test.end) - tolerance):
            raise ValueError(
                "data version does not fully cover configured value-regression segments: "
                f"available {available_start.date()}..{available_end.date()}"
            )

    @staticmethod
    def _lightgbm(cfg: ValueRegressionConfig):
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError('LightGBM is required. Install with: python -m pip install -e ".[ml]"') from exc
        return LGBMRegressor(**cfg.model.model_dump(exclude={"early_stopping_rounds"}))
