from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from factor_forge.config import load_project
from factor_forge.data import DataVersionRepository

from .backtest import EventBacktestRunner
from .research import BreakoutResearchRunner


@dataclass(frozen=True)
class ScenarioBacktestConfig:
    research_run: str
    project_config: str = "configs/project.yaml"
    top_scenarios: int = 30
    top_n: tuple[int, ...] = (5, 10, 20)
    cost_scenarios_bps: tuple[float, ...] = (0.0, 10.0, 20.0)
    initial_cash: float = 1_000_000.0
    lot_size: int = 100
    min_listing_days: int = 60
    output_subdir: str = "scenario_backtests"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScenarioBacktestConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if "top_n" in raw:
            raw["top_n"] = tuple(int(value) for value in raw["top_n"])
        if "cost_scenarios_bps" in raw:
            raw["cost_scenarios_bps"] = tuple(float(value) for value in raw["cost_scenarios_bps"])
        return cls(**raw)


class ScenarioBacktestRunner:
    def run(self, config_path: str | Path) -> dict:
        config_path = Path(config_path)
        config = ScenarioBacktestConfig.from_yaml(config_path)
        research_path = Path(config.research_run)
        research_manifest = json.loads((research_path / "manifest.json").read_text(encoding="utf-8"))
        events = pd.read_parquet(research_path / "events_with_scores.parquet")
        events["trade_date"] = pd.to_datetime(events["trade_date"])
        ic = pd.read_csv(research_path / "ic_results.csv")
        candidates = ic.loc[ic["ic_days"] >= 40].head(config.top_scenarios).copy()
        conditions = BreakoutResearchRunner._conditions(events)
        candidates, canonical_masks = self._deduplicate_conditions(candidates, conditions)

        project = load_project(config.project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        data_version = repository.resolve(research_manifest["data_version"])
        digest = hashlib.sha256(
            (config_path.read_text(encoding="utf-8") + data_version).encode("utf-8")
        ).hexdigest()[:10]
        output = research_path / config.output_subdir / f"scenario_checkpoint_{digest}"
        output.mkdir(parents=True, exist_ok=True)
        completed_manifest = output / "manifest.json"
        if completed_manifest.exists():
            manifest = json.loads(completed_manifest.read_text(encoding="utf-8"))
            if manifest.get("status") == "COMPLETED":
                return manifest
        (output / "config.yaml").write_text(
            config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        candidates.to_csv(output / "candidate_scenarios.csv", index=False, encoding="utf-8-sig")
        daily_parts = output / "daily_parts"
        trade_parts = output / "trade_parts"
        daily_parts.mkdir(exist_ok=True)
        trade_parts.mkdir(exist_ok=True)

        baseline_file = output / "baseline_checkpoint.csv"
        result_file = output / "results_checkpoint.csv"
        ablation_file = output / "ablations_checkpoint.csv"
        baseline_frame = pd.read_csv(baseline_file) if baseline_file.exists() else pd.DataFrame()
        results_frame = pd.read_csv(result_file) if result_file.exists() else pd.DataFrame()
        ablation_frame = pd.read_csv(ablation_file) if ablation_file.exists() else pd.DataFrame()
        baseline_cache: dict[tuple, dict] = {}
        if len(baseline_frame):
            for row in baseline_frame.to_dict("records"):
                key = (row.pop("condition"), int(row.pop("horizon")), float(row.pop("cost_bps")))
                baseline_cache[key] = row
        completed_runs = set(results_frame.get("run_key", pd.Series(dtype=str)).astype(str))
        completed_ablations = set(
            ablation_frame.get("ablation_key", pd.Series(dtype=str)).astype(str)
        )
        unique_baselines = candidates[["canonical_condition", "horizon"]].drop_duplicates()
        ablation_total = sum(
            1 + len(str(row.components).split("+"))
            for row in candidates.itertuples()
            if len(str(row.components).split("+")) > 1
        )
        total_steps = (
            len(candidates) * len(config.cost_scenarios_bps) * len(config.top_n)
            + len(unique_baselines) * len(config.cost_scenarios_bps)
            + ablation_total
        )
        completed_steps = len(baseline_cache) + len(completed_runs) + len(completed_ablations)
        self._write_progress(output, "loading_market", completed_steps, total_steps, None)
        panel_path = (
            Path(project.paths.data_root)
            / "versions"
            / data_version
            / "curated"
            / "stock_daily_panel.parquet"
        )
        engine = EventBacktestRunner()
        panel = engine._load_market(panel_path, events["trade_date"].min())
        dates = list(pd.Index(panel["trade_date"].unique()).sort_values())
        by_date = {
            pd.Timestamp(date): frame.set_index("ts_code")
            for date, frame in panel.groupby("trade_date", sort=True)
        }
        self._write_progress(output, "main_backtests", completed_steps, total_steps, None)
        for scenario in candidates.itertuples():
            mask = canonical_masks[scenario.canonical_condition]
            pool = events.loc[mask]
            scenario_id = self._scenario_id(scenario.score, scenario.canonical_condition, scenario.horizon)
            for cost in config.cost_scenarios_bps:
                baseline_key = (scenario.canonical_condition, int(scenario.horizon), float(cost))
                if baseline_key not in baseline_cache:
                    current = f"baseline:{scenario.canonical_condition}:h{scenario.horizon}:cost{cost:g}"
                    self._write_progress(output, "baseline", completed_steps, total_steps, current)
                    selections = self._select(pool, None, 0, ascending=False)
                    daily, _, metrics = engine._simulate(
                        dates,
                        by_date,
                        selections,
                        holding_days=int(scenario.horizon),
                        initial_cash=config.initial_cash,
                        lot_size=config.lot_size,
                        min_listing_days=config.min_listing_days,
                        cost_bps=float(cost),
                        allocation_count=None,
                    )
                    baseline_cache[baseline_key] = metrics
                    baseline_frame = pd.concat(
                        [
                            baseline_frame,
                            pd.DataFrame(
                                [
                                    {
                                        "condition": scenario.canonical_condition,
                                        "horizon": int(scenario.horizon),
                                        "cost_bps": float(cost),
                                        **metrics,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
                    baseline_frame.to_csv(baseline_file, index=False, encoding="utf-8-sig")
                    completed_steps += 1
                baseline = baseline_cache[baseline_key]

                for top_n in config.top_n:
                    run_key = f"{scenario_id}:top{top_n}:cost{cost:g}"
                    if run_key in completed_runs:
                        continue
                    self._write_progress(output, "main_backtest", completed_steps, total_steps, run_key)
                    selections = self._select(pool, scenario.score, top_n, ascending=False)
                    daily, trades, metrics = engine._simulate(
                        dates,
                        by_date,
                        selections,
                        holding_days=int(scenario.horizon),
                        initial_cash=config.initial_cash,
                        lot_size=config.lot_size,
                        min_listing_days=config.min_listing_days,
                        cost_bps=float(cost),
                        allocation_count=top_n,
                    )
                    row = {
                        "run_key": run_key,
                        "scenario_id": scenario_id,
                        "score": scenario.score,
                        "components": scenario.components,
                        "source_condition": scenario.condition,
                        "condition": scenario.canonical_condition,
                        "condition_aliases": scenario.condition_aliases,
                        "horizon": int(scenario.horizon),
                        "top_n": top_n,
                        "cost_bps": float(cost),
                        "source_rank_ic": scenario.rank_ic_mean,
                        "source_oos_ic": scenario.oos_rank_ic_mean,
                        **metrics,
                        "pool_annualized_return": baseline["annualized_return"],
                        "annualized_excess_vs_pool": metrics["annualized_return"]
                        - baseline["annualized_return"],
                    }
                    if float(cost) == 20.0:
                        daily.insert(0, "run_key", run_key)
                        trades.insert(0, "run_key", run_key)
                        part_name = run_key.replace(":", "_")
                        daily.to_parquet(daily_parts / f"{part_name}.parquet", index=False)
                        trades.to_parquet(trade_parts / f"{part_name}.parquet", index=False)
                    results_frame = pd.concat(
                        [results_frame, pd.DataFrame([row])], ignore_index=True
                    )
                    results_frame.to_csv(result_file, index=False, encoding="utf-8-sig")
                    completed_runs.add(run_key)
                    completed_steps += 1

        results = results_frame.copy()
        results = self._classify(results)
        ablation = self._run_ablations(
            candidates,
            canonical_masks,
            events,
            engine,
            dates,
            by_date,
            config,
            baseline_cache,
            checkpoint_file=ablation_file,
            existing=ablation_frame,
            output=output,
            completed_steps=completed_steps,
            total_steps=total_steps,
        )
        results.to_csv(output / "results.csv", index=False, encoding="utf-8-sig")
        ablation.to_csv(output / "ablations.csv", index=False, encoding="utf-8-sig")
        daily_files = sorted(daily_parts.glob("*.parquet"))
        trade_files = sorted(trade_parts.glob("*.parquet"))
        if daily_files:
            pd.concat((pd.read_parquet(path) for path in daily_files), ignore_index=True).to_parquet(
                output / "daily_20bps.parquet", index=False
            )
        if trade_files:
            pd.concat((pd.read_parquet(path) for path in trade_files), ignore_index=True).to_parquet(
                output / "trades_20bps.parquet", index=False
            )
        (output / "report.md").write_text(self._report(results, candidates, ablation), encoding="utf-8")
        manifest = {
            "status": "COMPLETED",
            "research_run": str(research_path.resolve()),
            "data_version": data_version,
            "source_top_scenarios": config.top_scenarios,
            "deduplicated_scenarios": int(len(candidates)),
            "backtest_count": int(len(results)),
            "classification_20bps": results.loc[results["cost_bps"] == 20, "classification"].value_counts().to_dict(),
            "output_path": str(output.resolve()),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._write_progress(output, "completed", total_steps, total_steps, None)
        return manifest

    @staticmethod
    def _deduplicate_conditions(
        candidates: pd.DataFrame, conditions: dict[str, pd.Series]
    ) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
        fingerprint_to_name: dict[str, str] = {}
        aliases: dict[str, list[str]] = {}
        canonical_masks: dict[str, pd.Series] = {}
        condition_to_canonical: dict[str, str] = {}
        for name in candidates["condition"].drop_duplicates():
            mask = conditions[name].fillna(False).to_numpy(dtype=np.uint8)
            fingerprint = hashlib.sha256(mask.tobytes()).hexdigest()
            canonical = fingerprint_to_name.setdefault(fingerprint, name)
            condition_to_canonical[name] = canonical
            canonical_masks[canonical] = conditions[canonical].fillna(False)
            aliases.setdefault(canonical, []).append(name)
        output = candidates.copy()
        output["canonical_condition"] = output["condition"].map(condition_to_canonical)
        output["condition_aliases"] = output["canonical_condition"].map(
            lambda value: "+".join(aliases[value])
        )
        output = output.sort_values("exploratory_score", ascending=False).drop_duplicates(
            ["score", "canonical_condition", "horizon"]
        )
        return output.reset_index(drop=True), canonical_masks

    @staticmethod
    def _scenario_id(score: str, condition: str, horizon: int) -> str:
        digest = hashlib.sha1(f"{score}|{condition}|{horizon}".encode()).hexdigest()[:8]
        return f"S_{digest}"

    @staticmethod
    def _select(
        pool: pd.DataFrame, score: str | None, top_n: int, *, ascending: bool
    ) -> dict[pd.Timestamp, list[str]]:
        selections: dict[pd.Timestamp, list[str]] = {}
        for trade_date, daily in pool.groupby("trade_date", sort=True):
            if score is None:
                selected = daily.sort_values("ts_code")
            else:
                selected = daily.dropna(subset=[score]).sort_values(
                    [score, "ts_code"], ascending=[ascending, True]
                ).head(top_n)
            selections[pd.Timestamp(trade_date)] = selected["ts_code"].tolist()
        return selections

    @staticmethod
    def _classify(results: pd.DataFrame) -> pd.DataFrame:
        results = results.copy()
        standalone = (
            (results["annualized_return"] > 0)
            & (results["sharpe"] >= 0.5)
            & (results["annualized_excess_vs_pool"] > 0.03)
            & (results["positive_year_ratio"] >= 0.60)
        )
        enhancer = (
            ~standalone
            & (results["pool_annualized_return"] > 0)
            & (results["annualized_excess_vs_pool"] > 0.03)
        )
        filter_only = (
            ~standalone
            & ~enhancer
            & (results["annualized_excess_vs_pool"] > 0.03)
            & (results["annualized_return"] > results["pool_annualized_return"])
        )
        results["classification"] = np.select(
            [standalone, enhancer, filter_only],
            ["standalone_entry", "ranking_enhancer", "risk_filter"],
            default="invalid",
        )
        return results.sort_values(
            ["cost_bps", "annualized_return"], ascending=[True, False]
        ).reset_index(drop=True)

    def _run_ablations(
        self,
        candidates: pd.DataFrame,
        masks: dict[str, pd.Series],
        events: pd.DataFrame,
        engine: EventBacktestRunner,
        dates: list,
        by_date: dict,
        config: ScenarioBacktestConfig,
        baseline_cache: dict,
        *,
        checkpoint_file: Path,
        existing: pd.DataFrame,
        output: Path,
        completed_steps: int,
        total_steps: int,
    ) -> pd.DataFrame:
        frame = existing.copy()
        completed = set(frame.get("ablation_key", pd.Series(dtype=str)).astype(str))
        for scenario in candidates.itertuples():
            components = str(scenario.components).split("+")
            if len(components) <= 1:
                continue
            pool = events.loc[masks[scenario.canonical_condition]]
            scores = [(scenario.score, "combined")]
            scores.extend((f"single:{component}", component) for component in components)
            for score, label in scores:
                scenario_id = self._scenario_id(
                    scenario.score, scenario.canonical_condition, scenario.horizon
                )
                ablation_key = f"{scenario_id}:{label}"
                if ablation_key in completed:
                    continue
                self._write_progress(
                    output, "ablation", completed_steps, total_steps, ablation_key
                )
                selections = self._select(pool, score, 10, ascending=False)
                _, _, metrics = engine._simulate(
                    dates,
                    by_date,
                    selections,
                    holding_days=int(scenario.horizon),
                    initial_cash=config.initial_cash,
                    lot_size=config.lot_size,
                    min_listing_days=config.min_listing_days,
                    cost_bps=20.0,
                    allocation_count=10,
                )
                frame = pd.concat(
                    [frame, pd.DataFrame([{
                        "ablation_key": ablation_key,
                        "scenario_id": scenario_id,
                        "condition": scenario.canonical_condition,
                        "horizon": int(scenario.horizon),
                        "variant": label,
                        "score": score,
                        "annualized_return_20bps": metrics["annualized_return"],
                        "sharpe_20bps": metrics["sharpe"],
                        "max_drawdown_20bps": metrics["max_drawdown"],
                    }])],
                    ignore_index=True,
                )
                frame.to_csv(checkpoint_file, index=False, encoding="utf-8-sig")
                completed.add(ablation_key)
                completed_steps += 1
        return frame

    @staticmethod
    def _write_progress(
        output: Path,
        stage: str,
        completed: int,
        total: int,
        current: str | None,
    ) -> None:
        payload = {
            "status": "COMPLETED" if stage == "completed" else "RUNNING",
            "stage": stage,
            "completed_steps": int(completed),
            "total_steps": int(total),
            "progress": float(completed / total) if total else 1.0,
            "current": current,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary = output / "progress.json.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output / "progress.json")

    @staticmethod
    def _report(results: pd.DataFrame, candidates: pd.DataFrame, ablation: pd.DataFrame) -> str:
        focus = results.loc[results["cost_bps"] == 20].sort_values(
            ["classification", "annualized_return"], ascending=[True, False]
        )
        lines = [
            "# Top30 IC 场景回测与研究分类",
            "",
            f"原始 Top30 经相同事件池去重后保留 {len(candidates)} 个场景。",
            "信号在 T 日收盘形成，T+1 开盘交易；持有期与 IC 场景的10/20日一致。",
            "分类以20 bps后绝对收益、条件池超额、Sharpe和年度稳定性共同判断。",
            "",
            "## 20 bps结果",
            "",
            "|场景|组合|条件|周期|TopN|年化|池基准年化|超额|Sharpe|最大回撤|年度胜率|分类|",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in focus.itertuples():
            lines.append(
                f"|{row.scenario_id}|{row.score}|{row.condition}|{row.horizon}|{row.top_n}|"
                f"{row.annualized_return:.2%}|{row.pool_annualized_return:.2%}|"
                f"{row.annualized_excess_vs_pool:.2%}|{row.sharpe:.2f}|{row.max_drawdown:.2%}|"
                f"{row.positive_year_ratio:.0%}|{row.classification}|"
            )
        return "\n".join(lines) + "\n"
