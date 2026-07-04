from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import Segments


RAW_FEATURES = (
    "range_compactness",
    "volatility_contraction",
    "trend_flatness",
    "approach_velocity",
    "pre_acceleration",
    "direction_persistence",
    "consolidation_age",
    "breakout_strength",
    "breakout_velocity",
    "breakout_acceleration",
    "relative_volume",
    "gap_atr",
    "continuous_move",
    "market_component_return",
    "market_trend_20",
    "market_volatility_20",
    "log_total_mv",
)

RANK_FEATURES = (
    "range_compactness",
    "volatility_contraction",
    "trend_flatness",
    "approach_velocity",
    "pre_acceleration",
    "direction_persistence",
    "breakout_strength",
    "breakout_velocity",
    "breakout_acceleration",
    "relative_volume",
    "continuous_move",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QlibLGBConfig(StrictModel):
    learning_rate: float = Field(default=0.03, gt=0)
    num_leaves: int = Field(default=31, ge=2)
    max_depth: int = -1
    num_boost_round: int = Field(default=800, ge=1)
    early_stopping_rounds: int = Field(default=60, ge=1)
    min_data_in_leaf: int = Field(default=80, ge=1)
    feature_fraction: float = Field(default=0.8, gt=0, le=1)
    bagging_fraction: float = Field(default=0.8, gt=0, le=1)
    bagging_freq: int = Field(default=1, ge=0)
    lambda_l1: float = Field(default=0.1, ge=0)
    lambda_l2: float = Field(default=1.0, ge=0)
    num_threads: int = -1
    seed: int = 42


class BreakoutQlibConfig(StrictModel):
    version: int = 1
    name: str = "breakout_qlib_lgb_v1"
    research_run: Path
    horizon: int = Field(default=10, ge=1, le=60)
    round_trip_cost_bps: float = Field(default=20.0, ge=0)
    segments: Segments
    targets: list[Literal["absolute", "event_excess", "cost_positive"]] = Field(
        default_factory=lambda: ["absolute", "event_excess", "cost_positive"]
    )
    add_daily_rank_features: bool = True
    top_n: list[int] = Field(default_factory=lambda: [5, 10, 20])
    model: QlibLGBConfig = Field(default_factory=QlibLGBConfig)
    output_root: Path = Path("artifacts/qlib_breakout_runs")

    @model_validator(mode="after")
    def validate_targets(self):
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("targets must be non-empty and unique")
        return self


def load_breakout_qlib_config(path: str | Path) -> BreakoutQlibConfig:
    return BreakoutQlibConfig.model_validate(
        yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    )


class DailyEqualWeight:
    """Qlib reweighter giving each event date equal aggregate training weight."""

    def reweight(self, data: pd.DataFrame) -> pd.Series:
        dates = data.index.get_level_values("datetime")
        counts = pd.Series(1.0, index=data.index).groupby(dates).transform("sum")
        weights = 1.0 / counts
        return weights / weights.mean()


@dataclass(frozen=True)
class TargetResult:
    target: str
    predictions: pd.DataFrame
    metrics: dict
    importance: pd.DataFrame
    evals_result: dict


class BreakoutQlibRunner:
    def run(self, config_path: str | Path) -> dict:
        config_path = Path(config_path)
        cfg = load_breakout_qlib_config(config_path)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = cfg.output_root / f"{cfg.name}_{run_id}"
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.yaml").write_bytes(config_path.read_bytes())
        manifest_path = output / "manifest.json"
        manifest = {
            "status": "RUNNING",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "research_run": str(cfg.research_run.resolve()),
            "output_path": str(output.resolve()),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        events, features = self._build_events(cfg)
        self._validate_segments(events, cfg)
        qlib_frame = self._to_qlib_frame(events, features, "label_absolute")
        qlib_frame.to_parquet(output / "qlib_event_dataset.parquet")
        self._initialize_qlib(output)

        target_results: list[TargetResult] = []
        for target in cfg.targets:
            result, model = self._fit_target(events, features, target, cfg)
            target_results.append(result)
            model.model.save_model(str(output / f"model_{target}.txt"))
            result.importance.to_csv(
                output / f"feature_importance_{target}.csv", index=False, encoding="utf-8-sig"
            )
            (output / f"training_curve_{target}.json").write_text(
                json.dumps(result.evals_result, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        predictions = self._merge_predictions(events, target_results, cfg)
        predictions.to_parquet(output / "predictions.parquet", index=False)
        metrics = pd.DataFrame([result.metrics for result in target_results])
        blend_metrics = self._prediction_metrics(
            predictions,
            "prediction_blend",
            cfg.top_n,
            cfg.round_trip_cost_bps,
        )
        blend_metrics["target"] = "blend"
        metrics = pd.concat([metrics, pd.DataFrame([blend_metrics])], ignore_index=True)
        metrics.to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
        report = self._report(metrics, cfg)
        (output / "report.md").write_text(report, encoding="utf-8")

        manifest.update(
            {
                "status": "COMPLETED",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "event_count": int(len(events)),
                "feature_count": len(features),
                "targets": cfg.targets,
                "best_test_target": str(
                    metrics.sort_values("test_top5_net_mean", ascending=False).iloc[0]["target"]
                ),
                "best_test_top5_net_mean": float(metrics["test_top5_net_mean"].max()),
            }
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self._end_qlib_experiment()
        return manifest

    @staticmethod
    def _build_events(cfg: BreakoutQlibConfig) -> tuple[pd.DataFrame, list[str]]:
        path = cfg.research_run / "events_with_scores.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing breakout event artifact: {path}")
        events = pd.read_parquet(path)
        events["trade_date"] = pd.to_datetime(events["trade_date"])
        if "continuous_move" not in events:
            events["continuous_move"] = -events["gap_atr"].abs()
        label_column = f"forward_return_{cfg.horizon}"
        if label_column not in events:
            raise ValueError(f"event artifact does not contain {label_column}")
        features = [feature for feature in RAW_FEATURES if feature in events]
        if cfg.add_daily_rank_features:
            for feature in RANK_FEATURES:
                if feature not in events:
                    continue
                name = f"rank_{feature}"
                events[name] = events.groupby("trade_date")[feature].rank(
                    method="average", pct=True
                )
                features.append(name)
        absolute = pd.to_numeric(events[label_column], errors="coerce")
        events["label_absolute"] = absolute
        events["label_event_excess"] = absolute - absolute.groupby(events["trade_date"]).transform(
            "mean"
        )
        threshold = cfg.round_trip_cost_bps / 10_000.0
        events["label_cost_positive"] = np.where(
            absolute.notna(), (absolute > threshold).astype(float), np.nan
        )
        events = events.replace([np.inf, -np.inf], np.nan)
        events = events.dropna(subset=["label_absolute"])
        if events.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError("Qlib event dataset has duplicate datetime/instrument keys")
        return events.sort_values(["trade_date", "ts_code"]).reset_index(drop=True), features

    @staticmethod
    def _validate_segments(events: pd.DataFrame, cfg: BreakoutQlibConfig) -> None:
        for name in ("train", "valid", "test"):
            segment = getattr(cfg.segments, name)
            mask = events["trade_date"].between(pd.Timestamp(segment.start), pd.Timestamp(segment.end))
            if not mask.any():
                raise ValueError(f"{name} segment contains no breakout events")

    @staticmethod
    def _to_qlib_frame(
        events: pd.DataFrame, features: list[str], label_column: str
    ) -> pd.DataFrame:
        frame = events.rename(columns={"trade_date": "datetime", "ts_code": "instrument"})
        frame = frame.set_index(["datetime", "instrument"])[features + [label_column]].sort_index()
        frame.columns = pd.MultiIndex.from_tuples(
            [("feature", name) for name in features] + [("label", "LABEL0")]
        )
        return frame

    @staticmethod
    def _initialize_qlib(output: Path) -> None:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import qlib

        database = (output / "qlib_mlflow.db").resolve().as_posix()
        tracking_uri = f"sqlite:///{database}"
        qlib.init(
            provider_uri="",
            exp_manager={
                "class": "MLflowExpManager",
                "module_path": "qlib.workflow.expm",
                "kwargs": {
                    "uri": tracking_uri,
                    "default_exp_name": "breakout_event_lightgbm",
                },
            },
        )

    def _fit_target(
        self,
        events: pd.DataFrame,
        features: list[str],
        target: str,
        cfg: BreakoutQlibConfig,
    ):
        from qlib.contrib.model.gbdt import LGBModel
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
        from qlib.data.dataset.loader import StaticDataLoader
        from qlib.data.dataset.weight import Reweighter

        class QlibDailyEqualWeight(Reweighter):
            def __init__(self) -> None:
                pass

            def reweight(self, data: pd.DataFrame) -> pd.Series:
                return DailyEqualWeight().reweight(data)

        label_column = f"label_{target}"
        frame = self._to_qlib_frame(events, features, label_column)
        handler = DataHandlerLP(
            data_loader=StaticDataLoader(frame), infer_processors=[], learn_processors=[]
        )
        segments = {
            name: (getattr(cfg.segments, name).start, getattr(cfg.segments, name).end)
            for name in ("train", "valid", "test")
        }
        dataset = DatasetH(handler=handler, segments=segments)
        params = cfg.model.model_dump(exclude={"num_boost_round", "early_stopping_rounds"})
        loss = "binary" if target == "cost_positive" else "mse"
        model = LGBModel(
            loss=loss,
            num_boost_round=cfg.model.num_boost_round,
            early_stopping_rounds=cfg.model.early_stopping_rounds,
            **params,
        )
        evals_result: dict = {}
        model.fit(
            dataset,
            verbose_eval=50,
            evals_result=evals_result,
            reweighter=QlibDailyEqualWeight(),
        )
        valid_prediction = model.predict(dataset, "valid").rename("prediction")
        test_prediction = model.predict(dataset, "test").rename("prediction")
        prediction = pd.concat(
            [
                valid_prediction.to_frame().assign(segment="valid"),
                test_prediction.to_frame().assign(segment="test"),
            ]
        ).reset_index()
        prediction = prediction.rename(columns={"datetime": "trade_date", "instrument": "ts_code"})
        evaluation = prediction.merge(
            events[
                [
                    "trade_date",
                    "ts_code",
                    "label_absolute",
                    "label_event_excess",
                    "label_cost_positive",
                ]
            ],
            on=["trade_date", "ts_code"],
            how="left",
            validate="one_to_one",
        )
        metrics = self._prediction_metrics(
            evaluation,
            "prediction",
            cfg.top_n,
            cfg.round_trip_cost_bps,
        )
        metrics["target"] = target
        importance = pd.DataFrame(
            {
                "feature": features,
                "gain": model.model.feature_importance(importance_type="gain"),
                "split": model.model.feature_importance(importance_type="split"),
            }
        ).sort_values("gain", ascending=False)
        return TargetResult(target, prediction, metrics, importance, evals_result), model

    @staticmethod
    def _prediction_metrics(
        prediction: pd.DataFrame,
        score_column: str,
        top_values: list[int],
        cost_bps: float,
    ) -> dict:
        result: dict[str, float | int | str | None] = {}
        threshold = cost_bps / 10_000.0
        for segment in ("valid", "test"):
            sample = prediction.loc[prediction["segment"] == segment].dropna(
                subset=[score_column, "label_absolute"]
            )
            result[f"{segment}_events"] = int(len(sample))
            if sample.empty:
                result[f"{segment}_rank_ic"] = np.nan
                result[f"{segment}_rank_ic_ir"] = np.nan
                for top_n in top_values:
                    result[f"{segment}_top{top_n}_gross_mean"] = np.nan
                    result[f"{segment}_top{top_n}_net_mean"] = np.nan
                    result[f"{segment}_top{top_n}_positive_ratio"] = np.nan
                continue
            daily_values = {}
            for trade_date, frame in sample.groupby("trade_date", sort=True):
                if (
                    len(frame) < 3
                    or frame[score_column].nunique(dropna=True) < 2
                    or frame["label_absolute"].nunique(dropna=True) < 2
                ):
                    daily_values[trade_date] = np.nan
                else:
                    daily_values[trade_date] = frame[score_column].corr(
                        frame["label_absolute"], method="spearman"
                    )
            daily_ic = pd.Series(daily_values, dtype=float)
            result[f"{segment}_rank_ic"] = float(daily_ic.mean())
            result[f"{segment}_rank_ic_ir"] = (
                float(daily_ic.mean() / daily_ic.std(ddof=1) * math.sqrt(252))
                if daily_ic.std(ddof=1) > 0
                else None
            )
            for top_n in top_values:
                selected = sample.sort_values(
                    ["trade_date", score_column], ascending=[True, False]
                ).groupby("trade_date").head(top_n)
                gross = float(selected["label_absolute"].mean())
                result[f"{segment}_top{top_n}_gross_mean"] = gross
                result[f"{segment}_top{top_n}_net_mean"] = gross - threshold
                result[f"{segment}_top{top_n}_positive_ratio"] = float(
                    (selected["label_absolute"] > threshold).mean()
                )
        return result

    @staticmethod
    def _merge_predictions(
        events: pd.DataFrame,
        results: list[TargetResult],
        cfg: BreakoutQlibConfig,
    ) -> pd.DataFrame:
        base = events[
            [
                "trade_date",
                "ts_code",
                "label_absolute",
                "label_event_excess",
                "label_cost_positive",
            ]
        ].copy()
        base["segment"] = np.select(
            [
                base["trade_date"].between(
                    pd.Timestamp(cfg.segments.valid.start), pd.Timestamp(cfg.segments.valid.end)
                ),
                base["trade_date"].between(
                    pd.Timestamp(cfg.segments.test.start), pd.Timestamp(cfg.segments.test.end)
                ),
            ],
            ["valid", "test"],
            default="other",
        )
        base = base.loc[base["segment"].isin(["valid", "test"])]
        for result in results:
            values = result.predictions.rename(columns={"prediction": f"prediction_{result.target}"})
            base = base.merge(
                values.drop(columns="segment"),
                on=["trade_date", "ts_code"],
                how="left",
                validate="one_to_one",
            )
        rank_columns = []
        for result in results:
            column = f"prediction_{result.target}"
            rank = f"rank_{column}"
            base[rank] = base.groupby("trade_date")[column].rank(method="average", pct=True)
            rank_columns.append(rank)
        base["prediction_blend"] = base[rank_columns].mean(axis=1, skipna=False)
        return base

    @staticmethod
    def _end_qlib_experiment() -> None:
        try:
            from qlib.workflow import R

            R.end_exp()
        except Exception:
            pass

    @staticmethod
    def _report(metrics: pd.DataFrame, cfg: BreakoutQlibConfig) -> str:
        ordered = metrics.sort_values("test_top5_net_mean", ascending=False)
        lines = [
            "# Qlib + LightGBM 突破事件训练报告",
            "",
            f"- 标签周期：{cfg.horizon}个交易日",
            f"- 交易成本门槛：{cfg.round_trip_cost_bps:.1f} bps",
            f"- Train：{cfg.segments.train.start} 至 {cfg.segments.train.end}",
            f"- Valid：{cfg.segments.valid.start} 至 {cfg.segments.valid.end}",
            f"- Test：{cfg.segments.test.start} 至 {cfg.segments.test.end}",
            "",
            "|目标|Test RankIC|Test Top5毛收益|Test Top5成本后收益|Top10成本后收益|Top20成本后收益|",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in ordered.itertuples():
            lines.append(
                f"|{row.target}|{row.test_rank_ic:.4f}|{row.test_top5_gross_mean:.2%}|"
                f"{row.test_top5_net_mean:.2%}|{row.test_top10_net_mean:.2%}|"
                f"{row.test_top20_net_mean:.2%}|"
            )
        return "\n".join(lines) + "\n"
