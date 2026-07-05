from __future__ import annotations

import hashlib
import json
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from factor_forge.backtest.engine import BacktestEngine
from factor_forge.config import CostModel, ExecutionConstraints, load_project
from factor_forge.data.repository import DataVersionRepository
from .value_dataset import build_value_dataset
from .value_regression import (
    ValueRegressionRunner,
    _daily_equal_weights,
    _purge_tail,
    _segment_mask,
    load_value_regression_config,
)


PRICE_FEATURES = [
    "residual_price_dislocation_20_5",
    "industry_relative_strength_5d",
    "price_delay_improvement_20d",
    "residual_cross_sectional_momentum_120_20",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ValueDiagnosticsConfig(_StrictModel):
    version: int = 1
    name: str = "value_regression_diagnostics_v1"
    experiment_config: Path
    full_run_dir: Path
    top_n: list[int] = Field(default_factory=lambda: [5, 10, 20])
    holding_days: list[int] = Field(default_factory=lambda: [5, 10, 20])
    deciles: int = Field(default=10, ge=2, le=20)
    output_root: Path = Path("artifacts/value_regression_diagnostics")

    @model_validator(mode="after")
    def validate_grid(self):
        if not self.top_n or len(set(self.top_n)) != len(self.top_n) or min(self.top_n) < 1:
            raise ValueError("top_n must contain unique positive values")
        if (not self.holding_days or len(set(self.holding_days)) != len(self.holding_days)
                or min(self.holding_days) < 1):
            raise ValueError("holding_days must contain unique positive values")
        return self


def load_value_diagnostics_config(path: str | Path) -> ValueDiagnosticsConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return ValueDiagnosticsConfig.model_validate(yaml.safe_load(handle) or {})


class ValueDiagnosticsRunner:
    def run(self, config_path: str | Path) -> dict:
        config_path = Path(config_path)
        cfg = load_value_diagnostics_config(config_path)
        experiment = load_value_regression_config(cfg.experiment_config)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + hashlib.sha256(config_path.read_bytes()).hexdigest()[:8]
        output = cfg.output_root / f"{cfg.name}_{run_id}"
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.yaml").write_bytes(config_path.read_bytes())
        logger = ValueRegressionRunner._run_logger(output, run_id)
        started = time.perf_counter()
        state = {"stage": "initializing"}

        def stage(name: str, detail: str = "") -> None:
            state["stage"] = name
            logger.info(
                "progress stage=%s detail=%s elapsed_seconds=%.1f rss_gb=%s",
                name, detail, time.perf_counter() - started, ValueRegressionRunner._rss_gb(),
            )

        try:
            result = self._execute(cfg, experiment, output, logger, stage)
            logger.info("run_completed elapsed_seconds=%.1f", time.perf_counter() - started)
            return result
        except Exception as exc:
            logger.exception("run_failed stage=%s", state["stage"])
            (output / "error.json").write_text(json.dumps({
                "status": "FAILED", "stage": state["stage"],
                "error_type": type(exc).__name__, "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            raise

    def _execute(self, cfg, experiment, output, logger, stage) -> dict:
        stage("load_data")
        project = load_project(experiment.project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        version, panel = repository.load_panel(experiment.data_version)
        working_panel = ValueRegressionRunner._attach_fundamentals(panel, experiment)

        def factor_progress(name: str, completed: int, total: int) -> None:
            if completed == total or completed == 1 or completed % max(total // 10, 1) == 0:
                stage(f"features:{name}", f"{completed}/{total}")

        dataset, all_features, labels = build_value_dataset(
            working_panel,
            horizons=experiment.labels.horizons,
            parameters=experiment.features.parameters(),
            excess_over_universe=experiment.labels.excess_over_universe,
            progress=factor_progress,
        )
        dates = dataset["trade_date"]
        masks = {
            "train": _segment_mask(dates, experiment.segments.train),
            "valid": _segment_mask(dates, experiment.segments.valid),
            "test": _segment_mask(dates, experiment.segments.test),
        }
        universe_column = f"is_{experiment.portfolio.universe}"
        eligible = dataset[universe_column].eq(True)
        enough_full = dataset[all_features].notna().sum(axis=1).ge(
            experiment.features.minimum_non_null_features
        )
        price_complete = dataset[PRICE_FEATURES].notna().all(axis=1)

        stage("load_full_predictions")
        full_path = cfg.full_run_dir / "predictions.parquet"
        if not full_path.exists():
            raise FileNotFoundError(f"full model predictions do not exist: {full_path}")
        full_predictions = pd.read_parquet(full_path)
        full_columns = [
            "trade_date", "ts_code", "prediction_blend",
            *[f"prediction_{horizon}d" for horizon in experiment.labels.horizons],
        ]
        full_predictions = full_predictions[full_columns].rename(columns={
            column: column.replace("prediction", "full_prediction")
            for column in full_columns if column.startswith("prediction")
        })

        stage("train_price_model")
        common_test_mask = masks["test"] & eligible & enough_full & price_complete
        price_predictions = dataset.loc[
            common_test_mask, ["trade_date", "ts_code", *labels]
        ].copy()
        price_models = {}
        for horizon in experiment.labels.horizons:
            stage("train_price_model", f"horizon={horizon}")
            label = f"label_{horizon}d"
            train = (
                _purge_tail(masks["train"], dates, horizon)
                & eligible & enough_full & price_complete & dataset[label].notna()
            )
            valid = (
                _purge_tail(masks["valid"], dates, horizon)
                & eligible & enough_full & price_complete & dataset[label].notna()
            )
            model = ValueRegressionRunner._lightgbm(experiment)
            fit_kwargs = {"sample_weight": _daily_equal_weights(dates.loc[train])}
            if valid.any():
                fit_kwargs["eval_set"] = [(
                    dataset.loc[valid, PRICE_FEATURES], dataset.loc[valid, label]
                )]
                fit_kwargs["eval_sample_weight"] = [_daily_equal_weights(dates.loc[valid])]
                if experiment.model.early_stopping_rounds:
                    import lightgbm as lgb
                    fit_kwargs["callbacks"] = [lgb.early_stopping(
                        experiment.model.early_stopping_rounds, verbose=False
                    )]
            model.fit(dataset.loc[train, PRICE_FEATURES], dataset.loc[train, label], **fit_kwargs)
            price_predictions[f"price_prediction_{horizon}d"] = model.predict(
                dataset.loc[price_predictions.index, PRICE_FEATURES],
                num_iteration=getattr(model, "best_iteration_", None),
            )
            model.booster_.save_model(str(output / f"price_model_{horizon}d.txt"))
            price_models[horizon] = model
        blend = pd.Series(0.0, index=price_predictions.index)
        for horizon, weight in experiment.labels.blend_weights.items():
            blend += weight * price_predictions.groupby("trade_date")[
                f"price_prediction_{horizon}d"
            ].rank(pct=True)
        price_predictions["price_prediction_blend"] = blend

        stage("align_models")
        common = price_predictions.merge(full_predictions, on=["trade_date", "ts_code"], how="inner")
        if common.empty:
            raise ValueError("full and price predictions have no common test samples")
        common.to_parquet(output / "common_predictions.parquet", index=False)

        stage("rank_ic_comparison")
        ic = self._rank_ic_comparison(common, experiment.labels.horizons)
        ic.to_csv(output / "model_ic_comparison.csv", index=False, encoding="utf-8-sig")

        stage("portfolio_matrix")
        test_panel = panel.loc[pd.to_datetime(panel["trade_date"]).between(
            pd.Timestamp(experiment.segments.test.start),
            pd.Timestamp(experiment.segments.test.end),
        )].copy()
        matrix_rows = []
        score_columns = {
            "full": "full_prediction_blend",
            "price": "price_prediction_blend",
        }
        for model_name, score_column in score_columns.items():
            signals = common[["trade_date", "ts_code", score_column]].rename(
                columns={score_column: "factor_value"}
            )
            for top_n in cfg.top_n:
                for holding_days in cfg.holding_days:
                    stage("portfolio_matrix", f"model={model_name},top={top_n},hold={holding_days}")
                    result = BacktestEngine().run(
                        test_panel, signals,
                        universe=experiment.portfolio.universe,
                        top_n=top_n, holding_days=holding_days,
                        initial_cash=experiment.portfolio.initial_cash,
                        lot_size=experiment.portfolio.lot_size,
                        constraints=ExecutionConstraints(), cost_model=CostModel(),
                        cost_scenario_bps=experiment.portfolio.cost_bps,
                    )
                    matrix_rows.append({
                        "model": model_name, "top_n": top_n,
                        "holding_days": holding_days, **result.metrics,
                    })
        matrix = pd.DataFrame(matrix_rows)
        matrix.to_csv(output / "topn_holding_matrix.csv", index=False, encoding="utf-8-sig")

        stage("decile_curves")
        deciles = self._decile_returns(common, score_columns, experiment.labels.horizons, cfg.deciles)
        deciles.to_csv(output / "decile_returns.csv", index=False, encoding="utf-8-sig")
        self._plot_deciles(deciles, output / "decile_curves.png")

        summary = self._summary(matrix, ic, deciles, output, version)
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output / "report.md").write_text(
            self._report(matrix, ic, deciles, summary), encoding="utf-8"
        )
        return summary

    @staticmethod
    def _rank_ic_comparison(common: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
        rows = []
        for model in ["full", "price"]:
            for horizon in horizons:
                score = f"{model}_prediction_{horizon}d"
                label = f"label_{horizon}d"
                daily = common.groupby("trade_date").apply(
                    lambda frame: frame[score].corr(frame[label], method="spearman"),
                    include_groups=False,
                ).dropna()
                rows.append({
                    "model": model, "horizon": horizon,
                    "rank_ic_mean": float(daily.mean()),
                    "rank_ic_ir": float(daily.mean() / daily.std()) if daily.std() else np.nan,
                    "days": len(daily),
                })
        return pd.DataFrame(rows)

    @staticmethod
    def _decile_returns(common, score_columns, horizons, decile_count):
        rows = []
        for model, score in score_columns.items():
            ranked = common.groupby("trade_date")[score].rank(method="first", pct=True)
            decile = np.ceil(ranked * decile_count).clip(1, decile_count).astype(int)
            for horizon in horizons:
                label = f"label_{horizon}d"
                daily = common.assign(decile=decile).groupby(
                    ["trade_date", "decile"]
                )[label].mean().reset_index()
                aggregate = daily.groupby("decile")[label].agg(["mean", "std", "count"])
                for bucket, item in aggregate.iterrows():
                    rows.append({
                        "model": model, "horizon": horizon, "decile": int(bucket),
                        "mean_excess_return": float(item["mean"]),
                        "standard_error": float(item["std"] / np.sqrt(item["count"])),
                        "days": int(item["count"]),
                    })
        return pd.DataFrame(rows)

    @staticmethod
    def _plot_deciles(deciles: pd.DataFrame, path: Path) -> None:
        import matplotlib.pyplot as plt

        horizons = sorted(deciles["horizon"].unique())
        figure, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4), sharey=False)
        axes = np.atleast_1d(axes)
        for axis, horizon in zip(axes, horizons):
            for model, frame in deciles.loc[deciles["horizon"].eq(horizon)].groupby("model"):
                axis.plot(frame["decile"], frame["mean_excess_return"], marker="o", label=model)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(f"{horizon}d forward excess return")
            axis.set_xlabel("score decile (10=highest)")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("mean excess return")
        axes[-1].legend()
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)

    @staticmethod
    def _summary(matrix, ic, deciles, output, version):
        best = matrix.sort_values("annualized_excess_return", ascending=False).iloc[0]
        spreads = []
        for (model, horizon), frame in deciles.groupby(["model", "horizon"]):
            indexed = frame.set_index("decile")["mean_excess_return"]
            spreads.append({
                "model": model, "horizon": int(horizon),
                "d10_minus_d1": float(indexed.max() - indexed.min())
                if 1 not in indexed or indexed.index.max() not in indexed else
                float(indexed.loc[indexed.index.max()] - indexed.loc[1]),
            })
        return {
            "data_version": version,
            "best_matrix_cell": {
                "model": best["model"], "top_n": int(best["top_n"]),
                "holding_days": int(best["holding_days"]),
                "annualized_excess_return": float(best["annualized_excess_return"]),
                "annualized_return": float(best["annualized_return"]),
                "sharpe": float(best["sharpe"]),
                "max_drawdown": float(best["max_drawdown"]),
            },
            "rank_ic": ic.to_dict("records"),
            "decile_spreads": spreads,
            "run_dir": str(output),
        }

    @staticmethod
    def _report(matrix, ic, deciles, summary):
        best = pd.DataFrame([summary["best_matrix_cell"]]).to_markdown(index=False)
        ic_table = ic.to_markdown(index=False, floatfmt=".4f")
        pivot = matrix.pivot_table(
            index=["model", "top_n"], columns="holding_days",
            values="annualized_excess_return",
        ).to_markdown(floatfmt=".2%")
        spread = pd.DataFrame(summary["decile_spreads"]).to_markdown(index=False, floatfmt=".4f")
        return "\n".join([
            "# 价值回归模型诊断", "", "## 最佳组合单元", "", best, "",
            "## Rank IC 对比", "", ic_table, "", "## TopN × 持仓期年化超额", "",
            pivot, "", "## 十分位 D10-D1", "", spread, "",
        ])
