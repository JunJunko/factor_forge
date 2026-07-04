from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from factor_forge.breakout_process.backtest import EventBacktestRunner
from factor_forge.config import load_project
from factor_forge.data import DataVersionRepository


class QlibPredictionBacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_run: Path
    project_config: Path = Path("configs/project.yaml")
    top_n: list[int] = Field(default_factory=lambda: [5, 10, 20])
    holding_days: int = Field(default=10, ge=1)
    initial_cash: float = Field(default=1_000_000, gt=0)
    lot_size: int = Field(default=100, ge=1)
    min_listing_days: int = Field(default=60, ge=0)
    cost_bps: float = Field(default=20, ge=0)


class QlibPredictionBacktestRunner:
    def run(self, config_path: str | Path) -> dict:
        config_path = Path(config_path)
        cfg = QlibPredictionBacktestConfig.model_validate(
            yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        )
        model_manifest = json.loads(
            (cfg.model_run / "manifest.json").read_text(encoding="utf-8")
        )
        research_run = Path(model_manifest["research_run"])
        research_manifest = json.loads(
            (research_run / "manifest.json").read_text(encoding="utf-8")
        )
        predictions = pd.read_parquet(cfg.model_run / "predictions.parquet")
        predictions["trade_date"] = pd.to_datetime(predictions["trade_date"])
        predictions = predictions.loc[predictions["segment"] == "test"].copy()
        score_columns = [
            column
            for column in predictions.columns
            if column.startswith("prediction_") and not column.startswith("prediction_rank_")
        ]

        project = load_project(cfg.project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        version = repository.resolve(research_manifest["data_version"])
        panel_path = (
            Path(project.paths.data_root)
            / "versions"
            / version
            / "curated"
            / "stock_daily_panel.parquet"
        )
        engine = EventBacktestRunner()
        panel = engine._load_market(panel_path, predictions["trade_date"].min())
        dates = list(pd.Index(panel["trade_date"].unique()).sort_values())
        by_date = {
            pd.Timestamp(date): frame.set_index("ts_code")
            for date, frame in panel.groupby("trade_date", sort=True)
        }

        rows: list[dict] = []
        daily_frames: list[pd.DataFrame] = []
        trade_frames: list[pd.DataFrame] = []
        for score in [*score_columns, "event_equal_weight"]:
            top_values = cfg.top_n if score != "event_equal_weight" else [0]
            for top_n in top_values:
                selections = self._selections(predictions, None if top_n == 0 else score, top_n)
                daily, trades, metrics = engine._simulate(
                    dates,
                    by_date,
                    selections,
                    holding_days=cfg.holding_days,
                    initial_cash=cfg.initial_cash,
                    lot_size=cfg.lot_size,
                    min_listing_days=cfg.min_listing_days,
                    cost_bps=cfg.cost_bps,
                    allocation_count=None if top_n == 0 else top_n,
                )
                run_key = f"{score}:top{top_n or 'all'}"
                rows.append(
                    {
                        "run_key": run_key,
                        "score": score,
                        "top_n": "all" if top_n == 0 else top_n,
                        **metrics,
                    }
                )
                daily.insert(0, "run_key", run_key)
                trades.insert(0, "run_key", run_key)
                daily_frames.append(daily)
                trade_frames.append(trades)

        summary = pd.DataFrame(rows)
        baseline_return = float(
            summary.loc[summary["score"] == "event_equal_weight", "annualized_return"].iloc[0]
        )
        summary["annualized_excess_vs_event_pool"] = (
            summary["annualized_return"] - baseline_return
        )
        summary = summary.sort_values("annualized_return", ascending=False).reset_index(drop=True)
        output = cfg.model_run / "backtests" / datetime.now(timezone.utc).strftime(
            "backtest_%Y%m%dT%H%M%SZ"
        )
        output.mkdir(parents=True, exist_ok=False)
        summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
        pd.concat(daily_frames, ignore_index=True).to_parquet(output / "daily.parquet", index=False)
        pd.concat(trade_frames, ignore_index=True).to_parquet(output / "trades.parquet", index=False)
        (output / "report.md").write_text(self._report(summary, cfg), encoding="utf-8")
        manifest = {
            "status": "COMPLETED",
            "model_run": str(cfg.model_run.resolve()),
            "data_version": version,
            "backtest_count": len(summary),
            "best_run": str(summary.iloc[0]["run_key"]),
            "best_annualized_return": float(summary.iloc[0]["annualized_return"]),
            "output_path": str(output.resolve()),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    @staticmethod
    def _selections(
        predictions: pd.DataFrame, score: str | None, top_n: int
    ) -> dict[pd.Timestamp, list[str]]:
        output: dict[pd.Timestamp, list[str]] = {}
        for trade_date, daily in predictions.groupby("trade_date", sort=True):
            if score is None:
                selected = daily.sort_values("ts_code")
            else:
                selected = daily.dropna(subset=[score]).sort_values(
                    [score, "ts_code"], ascending=[False, True]
                ).head(top_n)
            output[pd.Timestamp(trade_date)] = selected["ts_code"].tolist()
        return output

    @staticmethod
    def _report(summary: pd.DataFrame, cfg: QlibPredictionBacktestConfig) -> str:
        lines = [
            "# Qlib LightGBM测试期执行回测",
            "",
            f"总往返成本：{cfg.cost_bps:.1f} bps；持有期：{cfg.holding_days}日。",
            "",
            "|预测分数|TopN|年化收益|Sharpe|最大回撤|相对事件池年化超额|执行率|",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summary.itertuples():
            lines.append(
                f"|{row.score}|{row.top_n}|{row.annualized_return:.2%}|{row.sharpe:.2f}|"
                f"{row.max_drawdown:.2%}|{row.annualized_excess_vs_event_pool:.2%}|"
                f"{row.execution_rate:.2%}|"
            )
        return "\n".join(lines) + "\n"
