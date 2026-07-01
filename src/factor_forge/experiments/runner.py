from __future__ import annotations

import hashlib
import json
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from factor_forge.backtest import BacktestEngine
from factor_forge.config import load_experiment, load_factor, load_project, load_yaml
from factor_forge.data import DataVersionRepository
from factor_forge.evaluation import evaluate_factor_quality, evaluate_predictive_power
from factor_forge.evaluation.robustness import evaluate_robustness
from factor_forge.factors import FactorEngine
from factor_forge.scoring import AlphaScorer
from .artifacts import RunArtifacts


class ExperimentRunner:
    def run(self, experiment_path: str | Path) -> dict:
        experiment_path = Path(experiment_path)
        experiment_raw = load_yaml(experiment_path)
        experiment = load_experiment(experiment_path)
        project_path = Path(experiment.project_config)
        factor_path = Path(experiment.factor_config)
        scoring_path = Path(experiment.scoring_config)
        project_raw, factor_raw = load_yaml(project_path), load_yaml(factor_path)
        scoring_raw = load_yaml(scoring_path)
        project, factor = load_project(project_path), load_factor(factor_path)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        data_version, panel = repository.load_panel(experiment.data_version)
        market_benchmark = (
            repository.load_raw_dataset(data_version, "index_daily")
            if "market_index" in experiment.stage_l2.benchmarks else None
        )
        _, data_manifest = repository.load_manifest(data_version)
        coverage_blockers = self._coverage_blockers(data_manifest, factor, experiment)
        run_id = self._run_id(
            factor.factor.name, data_version, factor_raw, experiment_raw, project_raw, scoring_raw
        )
        artifacts = RunArtifacts(project.paths.artifacts_root, run_id)
        started = datetime.now(timezone.utc)
        manifest = {
            "run_id": run_id, "status": "RUNNING", "started_at": started.isoformat(),
            "factor_name": factor.factor.name, "factor_contract_version": factor.version,
            "experiment_profile": experiment.name, "data_version": data_version,
            "code_version": self._code_version(), "random_seed": 0,
            "selection_metadata": {"selected_by": "fixed_protocol", "requires_oos_confirmation": True},
            "data_coverage_blockers": coverage_blockers,
        }
        artifacts.yaml("inputs/factor.yaml", factor_raw)
        artifacts.yaml("inputs/experiment.yaml", experiment_raw)
        artifacts.yaml("inputs/project.yaml", project_raw)
        artifacts.yaml("inputs/scoring.yaml", scoring_raw)
        artifacts.json("manifest.json", manifest)
        scorer = AlphaScorer(scoring_raw)
        try:
            factor_values = FactorEngine().compute(panel, factor)
            artifacts.parquet("factor_values.parquet", factor_values)
            l0 = evaluate_factor_quality(panel, factor_values, factor, experiment.stage_l0)
            artifacts.json("l0_quality.json", l0)
            if not l0["passed"]:
                l1 = {"passed": False, "gate_paths": {}, "results": []}
                assessment = scorer.score(factor, l0, l1, [])
                assessment["classification"] = "INVALID"
                return self._finish(artifacts, manifest, assessment, l0, l1, [], started, "STOPPED_L0")
            if coverage_blockers:
                l1 = {"passed": False, "gate_paths": {}, "results": []}
                assessment = scorer.score(factor, l0, l1, [])
                assessment["classification"] = "INVALID"
                assessment["invalid_reason"] = "INVALID_DATA_COVERAGE"
                assessment["hard_flags"]["data_coverage"] = True
                return self._finish(artifacts, manifest, assessment, l0, l1, [], started, "INVALID_DATA_COVERAGE")
            l1 = evaluate_predictive_power(panel, factor_values, factor, experiment.stage_l1)
            artifacts.json("l1_predictive_power.json", l1)
            if factor.factor.direction == "unknown":
                assessment = scorer.score(factor, l0, l1, [])
                assessment["classification"] = "WATCHLIST"
                assessment["review_reason"] = "DIRECTION_UNFROZEN"
                return self._finish(artifacts, manifest, assessment, l0, l1, [], started, "STOPPED_DIRECTION_UNFROZEN")
            if not l1["passed"]:
                assessment = scorer.score(factor, l0, l1, [])
                return self._finish(artifacts, manifest, assessment, l0, l1, [], started, "STOPPED_L1")
            l2_rows = []
            engine = BacktestEngine()
            combinations = 0
            for universe in experiment.stage_l2.universes:
                for top_n in experiment.stage_l2.top_n:
                    for holding in experiment.stage_l2.holding_periods:
                        for cost in experiment.stage_l2.cost_scenarios_bps:
                            result = engine.run(
                                panel, factor_values, universe=universe, top_n=top_n,
                                holding_days=holding, initial_cash=experiment.stage_l2.initial_cash,
                                lot_size=experiment.stage_l2.lot_size,
                                constraints=experiment.stage_l2.execution_constraints,
                                cost_model=experiment.stage_l2.cost_model, cost_scenario_bps=cost,
                                market_benchmark=market_benchmark,
                            )
                            key = f"{universe}__top{top_n}__hold{holding}__cost{cost}"
                            base = f"l2/{key}"
                            artifacts.json(f"{base}/metrics.json", result.metrics)
                            if experiment.output.store_trade_details:
                                artifacts.parquet(f"{base}/trades.parquet", result.trades)
                            if experiment.output.store_daily_positions:
                                artifacts.parquet(f"{base}/positions.parquet", result.positions)
                            artifacts.parquet(f"{base}/daily.parquet", result.daily)
                            l2_rows.append({
                                "universe": universe, "top_n": top_n, "holding_days": holding,
                                "cost_bps": cost, "metrics": result.metrics, "daily": result.daily,
                                "positions": result.positions,
                            })
                            combinations += 1
            manifest["selection_metadata"]["total_combinations_tested"] = combinations
            manifest["stage_l3_ran"] = bool(experiment.stage_l3.enabled and any(
                row["cost_bps"] == 20 and row["metrics"]["annualized_excess_return"] > 0
                for row in l2_rows
            ))
            if manifest["stage_l3_ran"]:
                parameter_results = self._parameter_neighborhood(panel, factor, experiment)
                robustness = evaluate_robustness(panel, l2_rows, parameter_results)
                artifacts.json("l3_robustness.json", robustness)
            assessment = scorer.score(factor, l0, l1, l2_rows)
            compact_l2 = [{k: v for k, v in row.items() if k not in {"daily", "positions"}} for row in l2_rows]
            artifacts.json("l2_summary.json", compact_l2)
            return self._finish(artifacts, manifest, assessment, l0, l1, compact_l2, started, "SUCCESS")
        except Exception as exc:
            manifest.update({"status": "FAILED", "finished_at": datetime.now(timezone.utc).isoformat(), "error": str(exc)})
            artifacts.json("manifest.json", manifest)
            artifacts.text("error.log", traceback.format_exc())
            raise

    @staticmethod
    def _finish(artifacts, manifest, assessment, l0, l1, l2, started, status):
        finished = datetime.now(timezone.utc)
        manifest.update({"status": status, "finished_at": finished.isoformat(),
                         "elapsed_seconds": (finished - started).total_seconds()})
        artifacts.json("alpha_assessment.json", assessment)
        artifacts.json("manifest.json", manifest)
        report = (
            f"# Factor Forge 实验报告\n\n"
            f"- Run ID: `{manifest['run_id']}`\n"
            f"- 状态: **{status}**\n"
            f"- Alpha 分级: **{assessment['classification']}**\n"
            f"- 总分: **{assessment['total_score']} / 100**\n"
            f"- L0: {'通过' if l0['passed'] else '未通过'}\n"
            f"- L1: {'通过' if l1['passed'] else '未通过'}\n"
            f"- L2 组合数: {len(l2)}\n"
        )
        artifacts.text("report.md", report)
        return {"run_id": manifest["run_id"], "run_dir": str(artifacts.path),
                "status": status, "assessment": assessment}

    @staticmethod
    def _run_id(name: str, data_version: str, *configs: dict) -> str:
        payload = json.dumps([data_version, *configs], ensure_ascii=False, sort_keys=True).encode()
        suffix = hashlib.sha256(payload).hexdigest()[:8]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{name}__{timestamp}__{suffix}"

    @staticmethod
    def _code_version() -> str:
        try:
            return subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return "UNVERSIONED_WORKTREE"

    @staticmethod
    def _coverage_blockers(data_manifest, factor, experiment) -> list[str]:
        issues = {item["rule_name"] for item in data_manifest.get("quality_issues", [])
                  if item.get("severity") == "FEATURE_BLOCKING"}
        blockers = []
        uses_industry = factor.scope.cross_section == "industry" or "industry" in factor.data.required_fields
        if uses_industry and "industry_history_coverage" in issues:
            blockers.append("industry_history_coverage")
        uses_liquid = "liquid" in experiment.stage_l1.universes or "liquid" in experiment.stage_l2.universes
        if uses_liquid and "daily_basic_coverage" in issues:
            blockers.append("daily_basic_coverage")
        return blockers

    @staticmethod
    def _parameter_neighborhood(panel, factor, experiment) -> list[dict]:
        results = []
        for name, parameter in factor.calculation.parameters.items():
            for neighbor in parameter.robustness_neighbors:
                if neighbor == parameter.value:
                    continue
                variant = factor.model_copy(deep=True)
                variant.calculation.parameters[name].value = neighbor
                values = FactorEngine().compute(panel, variant)
                l1 = evaluate_predictive_power(panel, values, variant, experiment.stage_l1)
                means = [row["rank_ic"]["mean"] for row in l1["results"]
                         if row["rank_ic"]["mean"] is not None]
                results.append({"parameter": name, "value": neighbor,
                                "median_rank_ic": float(np.median(means)) if means else None,
                                "l1_passed": l1["passed"]})
        return results
