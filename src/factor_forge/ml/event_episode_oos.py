from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from factor_forge.backtest.engine import BacktestEngine
from factor_forge.config import CostModel, ExecutionConstraints, load_project
from factor_forge.experiments.artifacts import json_default


ARMS = ("e0_severity", "e1_raw", "e2_state", "e3_raw_state")


class EventEpisodeOOSBacktestRunner:
    def run(
        self,
        episode_run_dir: str | Path,
        *,
        project_config: str | Path = "configs/project.yaml",
        top_n: int = 10,
        holding_days: int = 5,
        cost_bps: float = 15.0,
        output_root: str | Path = "artifacts/event_episode_oos_backtests",
    ) -> dict:
        source = Path(episode_run_dir)
        source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        source_manifest = (source / "manifest.json").read_bytes()
        identity = hashlib.sha256(
            source_manifest + f"|top{top_n}|hold{holding_days}|cost{cost_bps}".encode()
        ).hexdigest()[:16]
        run_id = f"episode_oos_{identity}"
        output = Path(output_root) / run_id
        summary_path = output / "summary.json"
        if summary_path.exists():
            return {**json.loads(summary_path.read_text(encoding="utf-8")), "cached": True}

        predictions = {}
        for arm in ARMS:
            frame = pd.read_parquet(source / arm / "predictions.parquet")
            predictions[arm] = frame.rename(columns={
                "datetime": "trade_date", "instrument": "ts_code", "score": "factor_value",
            })[["trade_date", "ts_code", "factor_value"]]
            predictions[arm]["trade_date"] = pd.to_datetime(predictions[arm]["trade_date"])
        signal_start = min(frame["trade_date"].min() for frame in predictions.values())
        signal_end = max(frame["trade_date"].max() for frame in predictions.values())
        project = load_project(project_config)
        panel_path = (
            project.paths.data_root / "versions" / source_summary["data_version"]
            / "curated" / "stock_daily_panel.parquet"
        )
        columns = [
            "trade_date", "ts_code", "raw_open", "adj_open", "adj_close",
            "is_liquid", "is_tradeable", "is_suspended", "is_limit_up_open",
            "is_limit_down_open", "is_st", "is_delisting_period", "listing_trade_days",
            "industry_l1_code",
        ]
        panel = pd.read_parquet(
            panel_path, columns=columns,
            filters=[
                ("trade_date", ">=", signal_start),
                ("trade_date", "<=", pd.Timestamp(source_summary["as_of_date"])),
            ],
        )
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        membership = predictions["e0_severity"][["trade_date", "ts_code"]].copy()
        membership["selection_eligible"] = True
        membership["condition_quantile"] = 1

        output.mkdir(parents=True, exist_ok=False)
        rows, arm_summaries = [], {}
        for arm in ARMS:
            result = BacktestEngine().run(
                panel, predictions[arm], universe="liquid", top_n=top_n,
                holding_days=holding_days, initial_cash=10_000_000, lot_size=100,
                constraints=ExecutionConstraints(), cost_model=CostModel(),
                cost_scenario_bps=cost_bps, selection_membership=membership,
            )
            arm_output = output / arm
            arm_output.mkdir()
            result.daily.to_parquet(arm_output / "portfolio_daily.parquet", index=False)
            result.trades.to_parquet(arm_output / "trades.parquet", index=False)
            result.positions.to_parquet(arm_output / "positions.parquet", index=False)
            daily = result.daily.copy()
            drawdown = daily["nav"] / daily["nav"].cummax() - 1
            trough_index = drawdown.idxmin()
            peak_index = daily.loc[:trough_index, "nav"].idxmax()
            event_benchmark_total = float((1 + daily["benchmark_return"]).prod() - 1)
            universe_benchmark_total = float((1 + daily["universe_benchmark_return"]).prod() - 1)
            metrics = {
                **result.metrics,
                "oos_start": pd.Timestamp(daily["trade_date"].min()).strftime("%Y-%m-%d"),
                "oos_end": pd.Timestamp(daily["trade_date"].max()).strftime("%Y-%m-%d"),
                "trading_days": int(len(daily)),
                "event_pool_benchmark_total_return": event_benchmark_total,
                "universe_benchmark_total_return": universe_benchmark_total,
                "total_excess_vs_event_pool": float(result.metrics["total_return"] - event_benchmark_total),
                "total_excess_vs_universe": float(result.metrics["total_return"] - universe_benchmark_total),
                "max_drawdown_peak_date": pd.Timestamp(daily.loc[peak_index, "trade_date"]).strftime("%Y-%m-%d"),
                "max_drawdown_trough_date": pd.Timestamp(daily.loc[trough_index, "trade_date"]).strftime("%Y-%m-%d"),
            }
            (arm_output / "metrics.json").write_text(json.dumps(
                metrics, ensure_ascii=False, indent=2, default=json_default
            ), encoding="utf-8")
            arm_summaries[arm] = metrics
            rows.append({"arm": arm, **metrics})
        pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")
        summary = {
            "run_id": run_id, "source_run_id": source_summary["run_id"],
            "data_version": source_summary["data_version"],
            "oos_signal_start": pd.Timestamp(signal_start).strftime("%Y-%m-%d"),
            "oos_signal_end": pd.Timestamp(signal_end).strftime("%Y-%m-%d"),
            "top_n": top_n, "holding_days": holding_days, "round_trip_cost_bps": cost_bps,
            "arms": arm_summaries,
            "interpretation_boundary": (
                "Chronological held-out Event Episode OOS; this is a sparse-event TopN simulation, not forward live evidence."
            ),
            "run_dir": str(output.resolve()), "cached": False,
        }
        summary_path.write_text(json.dumps(
            summary, ensure_ascii=False, indent=2, default=json_default
        ), encoding="utf-8")
        (output / "manifest.json").write_text(json.dumps({
            "run_id": run_id, "runner_type": "event_episode_oos_backtests", "status": "COMPLETED",
            "source_run_id": source_summary["run_id"],
            "source_manifest_sha256": hashlib.sha256(source_manifest).hexdigest(),
            "data_version": source_summary["data_version"],
            "top_n": top_n, "holding_days": holding_days, "round_trip_cost_bps": cost_bps,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
