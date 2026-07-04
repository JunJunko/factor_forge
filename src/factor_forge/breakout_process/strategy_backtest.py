from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from factor_forge.backtest import BacktestEngine
from factor_forge.config import CostModel, ExecutionConstraints, load_project
from factor_forge.data import DataVersionRepository


class FrozenBreakoutStrategyRunner:
    """Run the immutable v1 event strategy through the platform execution engine."""

    def run(self, config_path: str | Path) -> dict:
        config_path = Path(config_path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        self._validate(raw)
        formula = raw["formula"]
        project = load_project(raw["project_config"])
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        data_version = repository.resolve(raw["data_version"])
        events_path = Path(raw["events_path"])
        events = pd.read_parquet(events_path)
        events["trade_date"] = pd.to_datetime(events["trade_date"])

        component_score = formula["score_column"]
        condition_column = formula["condition_column"]
        threshold = float(formula["condition_threshold"])
        signals = events.loc[
            events[condition_column] > threshold,
            ["trade_date", "ts_code", component_score],
        ].rename(columns={component_score: "factor_value"})
        signals = signals.dropna(subset=["factor_value"])
        if signals.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError("frozen strategy signals have duplicate date/security keys")

        panel_path = (
            Path(project.paths.data_root)
            / "versions"
            / data_version
            / "curated"
            / "stock_daily_panel.parquet"
        )
        columns = [
            "trade_date",
            "ts_code",
            "raw_open",
            "adj_open",
            "adj_close",
            "is_liquid",
            "is_suspended",
            "is_limit_up_open",
            "is_limit_down_open",
            "is_st",
            "is_delisting_period",
            "listing_trade_days",
        ]
        panel = pd.read_parquet(panel_path, columns=columns)
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        signal_filter = raw.get("signal_filter")
        if signal_filter:
            signals = self._apply_signal_filter(signals, panel, signal_filter)
        start = pd.Timestamp(raw["sample_start_date"])
        end = pd.Timestamp(raw["sample_end_date"]) if raw.get("sample_end_date") else panel["trade_date"].max()
        panel = panel.loc[(panel["trade_date"] >= start) & (panel["trade_date"] <= end)].copy()
        signals = signals.loc[(signals["trade_date"] >= start) & (signals["trade_date"] <= end)]

        formula_digest = hashlib.sha256(
            yaml.safe_dump(formula, sort_keys=True).encode("utf-8")
        ).hexdigest()
        strategy_digest = hashlib.sha256(
            yaml.safe_dump(
                {"formula": formula, "signal_filter": signal_filter}, sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
        run_digest = hashlib.sha256(
            yaml.safe_dump(raw, sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]
        run_id = f"frozen_breakout_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{run_digest}"
        output = Path(raw["output_root"]) / run_id
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.yaml").write_text(
            yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        signals.to_parquet(output / "frozen_signals.parquet", index=False)

        constraints = ExecutionConstraints.model_validate(raw["execution_constraints"])
        cost_model = CostModel.model_validate(raw["cost_model"])
        summary_rows: list[dict] = []
        for holding_days in raw["holding_periods"]:
            for top_n in raw["top_n"]:
                for cost_bps in raw["cost_scenarios_bps"]:
                    result = BacktestEngine().run(
                        panel,
                        signals,
                        universe="liquid",
                        top_n=int(top_n),
                        holding_days=int(holding_days),
                        initial_cash=float(raw["initial_cash"]),
                        lot_size=int(raw["lot_size"]),
                        constraints=constraints,
                        cost_model=cost_model,
                        cost_scenario_bps=float(cost_bps),
                    )
                    key = f"h{holding_days}_top{top_n}_cost{cost_bps}"
                    result.daily.to_parquet(output / f"daily_{key}.parquet", index=False)
                    result.trades.to_parquet(output / f"trades_{key}.parquet", index=False)
                    metrics = {
                        "holding_days": int(holding_days),
                        "top_n": int(top_n),
                        "cost_bps": float(cost_bps),
                        **result.metrics,
                    }
                    summary_rows.append(metrics)

        summary = pd.DataFrame(summary_rows)
        summary.to_csv(output / "backtest_summary.csv", index=False, encoding="utf-8-sig")
        report = self._report(
            summary, formula_digest, strategy_digest, signal_filter, len(signals), start, end
        )
        (output / "report.md").write_text(report, encoding="utf-8")
        manifest = {
            "run_id": run_id,
            "status": "COMPLETED",
            "data_version": data_version,
            "formula_sha256": formula_digest,
            "strategy_sha256": strategy_digest,
            "signal_filter": signal_filter,
            "signal_count": int(len(signals)),
            "sample_start": start.date().isoformat(),
            "sample_end": pd.Timestamp(end).date().isoformat(),
            "grid_runs": len(summary),
            "output_path": str(output.resolve()),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    @staticmethod
    def _apply_signal_filter(
        signals: pd.DataFrame, panel: pd.DataFrame, specification: dict
    ) -> pd.DataFrame:
        if specification.get("type") != "close_above_sma":
            raise ValueError("only close_above_sma signal_filter is supported")
        window = int(specification.get("window", 0))
        if window <= 1:
            raise ValueError("close_above_sma window must exceed 1")
        ordered = panel[["trade_date", "ts_code", "adj_close"]].sort_values(
            ["ts_code", "trade_date"], kind="stable"
        )
        moving_average = (
            ordered.groupby("ts_code", sort=False)["adj_close"]
            .rolling(window, min_periods=window)
            .mean()
            .droplevel(0)
            .reindex(ordered.index)
        )
        eligibility = ordered[["trade_date", "ts_code"]].copy()
        eligibility["signal_filter_eligible"] = ordered["adj_close"] >= moving_average
        filtered = signals.merge(
            eligibility, on=["trade_date", "ts_code"], how="left", validate="one_to_one"
        )
        return filtered.loc[filtered["signal_filter_eligible"].eq(True)].drop(
            columns="signal_filter_eligible"
        )

    @staticmethod
    def _validate(raw: dict) -> None:
        required = {
            "project_config",
            "data_version",
            "events_path",
            "sample_start_date",
            "formula",
            "holding_periods",
            "top_n",
            "cost_scenarios_bps",
            "initial_cash",
            "lot_size",
            "execution_constraints",
            "cost_model",
            "output_root",
        }
        missing = required - set(raw)
        if missing:
            raise ValueError(f"frozen backtest config missing fields: {sorted(missing)}")
        formula_required = {"score_column", "condition_column", "condition_threshold"}
        if not formula_required <= set(raw["formula"]):
            raise ValueError("frozen formula is incomplete")

    @staticmethod
    def _report(
        summary: pd.DataFrame,
        formula_digest: str,
        strategy_digest: str,
        signal_filter: dict | None,
        signal_count: int,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> str:
        lines = [
            "# 冻结盘整突破策略回测",
            "",
            f"- 公式 SHA-256：`{formula_digest}`",
            f"- 策略 SHA-256：`{strategy_digest}`",
            f"- 条件内信号：{signal_count:,}",
            f"- 样本：{start.date()} 至 {pd.Timestamp(end).date()}",
            "- 信号：突破日收盘；执行：下一交易日开盘",
            f"- 额外过滤：`{signal_filter or 'none'}`",
            "",
            "|持有期|TopN|成本bps|年化收益|基准年化|年化超额|Sharpe|最大回撤|执行率|",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summary.itertuples():
            lines.append(
                f"|{row.holding_days}|{row.top_n}|{row.cost_bps:.0f}|"
                f"{row.annualized_return:.2%}|{row.benchmark_annualized_return:.2%}|"
                f"{row.annualized_excess_return:.2%}|{row.sharpe:.2f}|"
                f"{row.max_drawdown:.2%}|{row.execution_rate:.2%}|"
            )
        return "\n".join(lines) + "\n"
