from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from .breakout_qlib import (
    BreakoutQlibConfig,
    BreakoutQlibRunner,
    QlibLGBConfig,
)
from .config import Segment, Segments


class WalkForwardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    name: str = "breakout_qlib_walkforward_v1"
    research_run: Path
    first_history_year: int = 2021
    first_test_year: int = 2023
    last_test_year: int = 2026
    history_years: int = Field(default=3, ge=2)
    horizon: int = Field(default=10, ge=1, le=60)
    round_trip_cost_bps: float = Field(default=20, ge=0)
    targets: list[str] = Field(
        default_factory=lambda: ["absolute", "event_excess", "cost_positive"]
    )
    add_daily_rank_features: bool = True
    top_n: list[int] = Field(default_factory=lambda: [5, 10, 20])
    model: QlibLGBConfig = Field(default_factory=QlibLGBConfig)
    output_root: Path = Path("artifacts/qlib_breakout_walkforward")
    project_config: Path = Path("configs/project.yaml")


def load_walkforward_config(path: str | Path) -> WalkForwardConfig:
    return WalkForwardConfig.model_validate(
        yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    )


class BreakoutQlibWalkForwardRunner:
    def run(self, config_path: str | Path) -> dict:
        config_path = Path(config_path)
        cfg = load_walkforward_config(config_path)
        output = cfg.output_root / (
            f"{cfg.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        )
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.yaml").write_bytes(config_path.read_bytes())
        manifest_path = output / "manifest.json"
        manifest = {
            "status": "RUNNING",
            "research_run": str(cfg.research_run.resolve()),
            "output_path": str(output.resolve()),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        helper = BreakoutQlibRunner()
        seed_segments = self._segments_for_year(cfg, cfg.first_test_year)
        seed_config = self._target_config(cfg, seed_segments, output)
        events, features = helper._build_events(seed_config)
        helper._to_qlib_frame(events, features, "label_absolute").to_parquet(
            output / "qlib_event_dataset.parquet"
        )
        helper._initialize_qlib(output)

        prediction_frames: list[pd.DataFrame] = []
        fold_metrics: list[dict] = []
        fold_manifest: list[dict] = []
        for test_year in range(cfg.first_test_year, cfg.last_test_year + 1):
            segments = self._segments_for_year(cfg, test_year)
            fold_config = self._target_config(cfg, segments, output)
            helper._validate_segments(events, fold_config)
            results = []
            for target in cfg.targets:
                result, model = helper._fit_target(events, features, target, fold_config)
                results.append(result)
                model.model.save_model(str(output / f"model_{test_year}_{target}.txt"))
                importance = result.importance.copy()
                importance.insert(0, "test_year", test_year)
                importance.to_csv(
                    output / f"feature_importance_{test_year}_{target}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                metrics = dict(result.metrics)
                metrics["test_year"] = test_year
                fold_metrics.append(metrics)

            fold_prediction = helper._merge_predictions(events, results, fold_config)
            fold_prediction = fold_prediction.loc[
                fold_prediction["segment"] == "test"
            ].copy()
            fold_prediction["test_year"] = test_year
            prediction_frames.append(fold_prediction)
            blend = helper._prediction_metrics(
                fold_prediction,
                "prediction_blend",
                cfg.top_n,
                cfg.round_trip_cost_bps,
            )
            blend.update({"target": "blend", "test_year": test_year})
            fold_metrics.append(blend)
            fold_manifest.append(
                {
                    "test_year": test_year,
                    "train": segments.train.model_dump(),
                    "valid": segments.valid.model_dump(),
                    "test": segments.test.model_dump(),
                    "prediction_count": int(len(fold_prediction)),
                }
            )

        predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
            ["trade_date", "ts_code"]
        )
        if predictions.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError("walk-forward predictions overlap across folds")
        predictions.to_parquet(output / "predictions.parquet", index=False)
        fold_metrics_frame = pd.DataFrame(fold_metrics)
        fold_metrics_frame.to_csv(
            output / "fold_metrics.csv", index=False, encoding="utf-8-sig"
        )

        overall_rows = []
        for target in [*cfg.targets, "blend"]:
            score = f"prediction_{target}" if target != "blend" else "prediction_blend"
            metrics = helper._prediction_metrics(
                predictions,
                score,
                cfg.top_n,
                cfg.round_trip_cost_bps,
            )
            metrics["target"] = target
            overall_rows.append(metrics)
        overall = pd.DataFrame(overall_rows).sort_values(
            "test_top5_net_mean", ascending=False
        )
        overall.to_csv(output / "overall_metrics.csv", index=False, encoding="utf-8-sig")
        (output / "report.md").write_text(
            self._report(fold_metrics_frame, overall, cfg), encoding="utf-8"
        )

        research_manifest = json.loads(
            (cfg.research_run / "manifest.json").read_text(encoding="utf-8")
        )
        manifest.update(
            {
                "status": "COMPLETED",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "data_version": research_manifest["data_version"],
                "feature_count": len(features),
                "folds": fold_manifest,
                "prediction_count": int(len(predictions)),
                "best_overall_target": str(overall.iloc[0]["target"]),
                "best_overall_top5_net_mean": float(overall.iloc[0]["test_top5_net_mean"]),
            }
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        helper._end_qlib_experiment()
        return manifest

    @staticmethod
    def _segments_for_year(cfg: WalkForwardConfig, test_year: int) -> Segments:
        history_start = max(cfg.first_history_year, test_year - cfg.history_years)
        valid_year = test_year - 1
        train_end = valid_year - 1
        if history_start > train_end:
            raise ValueError(f"not enough pre-test history for {test_year}")
        test_end = f"{test_year}-12-31"
        return Segments(
            train=Segment(start=f"{history_start}-01-01", end=f"{train_end}-12-31"),
            valid=Segment(start=f"{valid_year}-01-01", end=f"{valid_year}-12-31"),
            test=Segment(start=f"{test_year}-01-01", end=test_end),
        )

    @staticmethod
    def _target_config(
        cfg: WalkForwardConfig, segments: Segments, output: Path
    ) -> BreakoutQlibConfig:
        return BreakoutQlibConfig(
            name=cfg.name,
            research_run=cfg.research_run,
            horizon=cfg.horizon,
            round_trip_cost_bps=cfg.round_trip_cost_bps,
            segments=segments,
            targets=cfg.targets,
            add_daily_rank_features=cfg.add_daily_rank_features,
            top_n=cfg.top_n,
            model=cfg.model,
            output_root=output,
        )

    @staticmethod
    def _report(
        fold_metrics: pd.DataFrame,
        overall: pd.DataFrame,
        cfg: WalkForwardConfig,
    ) -> str:
        yearly = fold_metrics[[
            "test_year",
            "target",
            "test_rank_ic",
            "test_top5_net_mean",
            "test_top10_net_mean",
            "test_top20_net_mean",
        ]].sort_values(["test_year", "target"])
        lines = [
            "# 突破事件Qlib滚动走样本外报告",
            "",
            "每个测试年度只使用此前历史；历史窗口最后一年用于早停验证。",
            f"成本门槛：{cfg.round_trip_cost_bps:.1f} bps。",
            "",
            "## 分年度",
            "",
            "|年度|模型|RankIC|Top5净收益|Top10净收益|Top20净收益|",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for row in yearly.itertuples():
            lines.append(
                f"|{row.test_year}|{row.target}|{row.test_rank_ic:.4f}|"
                f"{row.test_top5_net_mean:.2%}|{row.test_top10_net_mean:.2%}|"
                f"{row.test_top20_net_mean:.2%}|"
            )
        lines.extend(
            [
                "",
                "## 拼接后的全部走样本外预测",
                "",
                "|模型|RankIC|Top5净收益|Top10净收益|Top20净收益|",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in overall.itertuples():
            lines.append(
                f"|{row.target}|{row.test_rank_ic:.4f}|{row.test_top5_net_mean:.2%}|"
                f"{row.test_top10_net_mean:.2%}|{row.test_top20_net_mean:.2%}|"
            )
        return "\n".join(lines) + "\n"
