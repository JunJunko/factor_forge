from __future__ import annotations

import hashlib
import json
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.backtest import BacktestEngine, build_condition_membership
from factor_forge.config import factor_source_kind, load_experiment, load_factor, load_factor_combination, load_project, load_yaml
from factor_forge.combinations import FactorCombinationEngine
from factor_forge.combinations.diagnostics import CombinationDiagnostics
from factor_forge.data import DataVersionRepository
from factor_forge.evaluation import evaluate_conditional_ic, evaluate_factor_quality, evaluate_predictive_power
from factor_forge.evaluation.robustness import evaluate_robustness
from factor_forge.factors import FactorEngine
from factor_forge.scoring import AlphaScorer
from factor_forge.research.industry import IndustrySlicePipeline
from .artifacts import RunArtifacts


def industry_slice_enabled(experiment) -> bool:
    config = getattr(experiment, "industry_slice", None)
    return bool(config is not None and config.enabled)


class ExperimentRunner:
    def run(
        self,
        experiment_path: str | Path,
        factor_path: str | Path | None = None,
        position_multiplier: pd.Series | None = None,
        position_multiplier_source: str | None = None,
    ) -> dict:
        experiment_path = Path(experiment_path)
        experiment_raw = load_yaml(experiment_path)
        experiment = load_experiment(experiment_path)
        position_multiplier = self._normalize_position_multiplier(position_multiplier)
        position_metadata = self._position_multiplier_metadata(position_multiplier, position_multiplier_source)
        project_path = Path(experiment.project_config)
        factor_path = Path(factor_path) if factor_path is not None else Path(experiment.factor_config)
        scoring_path = Path(experiment.scoring_config)
        backtest_contract_path = Path(experiment.backtest_contract)
        project_raw, factor_raw = load_yaml(project_path), load_yaml(factor_path)
        scoring_raw = load_yaml(scoring_path)
        backtest_contract_raw = load_yaml(backtest_contract_path)
        conditional_config = experiment.stage_l1.conditional_ic
        conditioning_path = None
        conditioning_raw = None
        conditioning_source_kind = None
        if conditional_config.enabled and conditional_config.conditioning_factor != "main_factor":
            conditioning_path = Path(conditional_config.conditioning_factor)
            conditioning_raw = load_yaml(conditioning_path)
            conditioning_source_kind = factor_source_kind(conditioning_path)
        self._validate_backtest_contract(backtest_contract_raw, experiment)
        project = load_project(project_path)
        source_kind = factor_source_kind(factor_path)
        combination_spec = load_factor_combination(factor_path) if source_kind == "factor_combination" else None
        if combination_spec:
            first_source = combination_spec.factor_combination.components[0].source.path
            first_path = first_source if first_source.is_absolute() else factor_path.parent / first_source
            factor = load_factor(first_path)
            factor = factor.model_copy(deep=True)
            factor.factor.name = combination_spec.factor_combination.id
            factor.factor.label = combination_spec.factor_combination.name
            factor.factor.description = combination_spec.factor_combination.description or combination_spec.factor_combination.name
            factor.factor.hypothesis = combination_spec.factor_combination.description or "Fixed multi-factor combination"
            factor.factor.direction = "positive"
            factor.factor.expected_shape = "monotonic"
            factor.scope.universe = "default"
            factor.scope.cross_section = "market"
        else:
            factor = load_factor(factor_path)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        data_version, panel = repository.load_panel(experiment.data_version)
        panel_dates = pd.to_datetime(panel["trade_date"])
        sample_mask = pd.Series(True, index=panel.index)
        if experiment.sample_start_date:
            sample_mask &= panel_dates >= pd.Timestamp(experiment.sample_start_date)
        if experiment.sample_end_date:
            sample_mask &= panel_dates <= pd.Timestamp(experiment.sample_end_date)
        if not sample_mask.any():
            raise ValueError("Experiment sample date range contains no panel rows")
        market_benchmark = (
            repository.load_raw_dataset(data_version, "index_daily")
            if "market_index" in experiment.stage_l2.benchmarks else None
        )
        if market_benchmark is not None and (experiment.sample_start_date or experiment.sample_end_date):
            benchmark_dates = pd.to_datetime(market_benchmark["trade_date"])
            if experiment.sample_start_date:
                market_benchmark = market_benchmark.loc[benchmark_dates >= pd.Timestamp(experiment.sample_start_date)]
                benchmark_dates = pd.to_datetime(market_benchmark["trade_date"])
            if experiment.sample_end_date:
                market_benchmark = market_benchmark.loc[benchmark_dates <= pd.Timestamp(experiment.sample_end_date)]
        _, data_manifest = repository.load_manifest(data_version)
        coverage_blockers = self._coverage_blockers(data_manifest, factor, experiment)
        run_id = self._run_id(
            factor.factor.name, data_version, factor_raw, experiment_raw, project_raw, scoring_raw,
            backtest_contract_raw, conditioning_raw or {}, position_metadata,
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
        if position_metadata["enabled"]:
            manifest["position_multiplier"] = position_metadata
        artifacts.yaml("inputs/factor.yaml", factor_raw)
        artifacts.yaml("inputs/experiment.yaml", experiment_raw)
        artifacts.yaml("inputs/project.yaml", project_raw)
        artifacts.yaml("inputs/scoring.yaml", scoring_raw)
        artifacts.yaml("inputs/backtest_contract.yaml", backtest_contract_raw)
        if conditioning_raw is not None:
            artifacts.yaml("inputs/conditioning_factor.yaml", conditioning_raw)
        if position_multiplier is not None:
            stored_multiplier = position_multiplier.rename("position_multiplier").to_frame().reset_index()
            stored_multiplier = stored_multiplier.rename(columns={stored_multiplier.columns[0]: "trade_date"})
            artifacts.csv(
                "inputs/position_multiplier.csv",
                stored_multiplier,
            )
        artifacts.json("manifest.json", manifest)
        scorer = AlphaScorer(scoring_raw)
        try:
            combination_result = None
            if combination_spec:
                requested = set(experiment.stage_l1.universes) | set(experiment.stage_l2.universes)
                scope_mask = panel["is_factor_eligible"].fillna(False).astype(bool)
                if requested:
                    universe_masks = [panel[f"is_{name}"].fillna(False).astype(bool) for name in requested]
                    scope_mask &= np.logical_or.reduce(universe_masks)
                dates = pd.to_datetime(panel["trade_date"])
                combination_result = FactorCombinationEngine().run(
                    panel, factor_path, scope_mask=scope_mask,
                    cache_context={"data_version": data_version, "start_date": dates.min().date(),
                                   "end_date": dates.max().date(), "base_universe": "+".join(sorted(requested))},
                )
                factor_values = combination_result.factor_values
                artifacts.csv("factor_combination_summary.csv", combination_result.coverage)
                artifacts.csv("factor_component_coverage.csv", combination_result.component_coverage)
                artifacts.json("combination_cache_status.json", combination_result.cache_status)
                artifacts.json("combination_normalization_issues.json", combination_result.normalization_issues)
                manifest["factor_source_kind"] = "factor_combination"
            else:
                factor_values = FactorEngine().compute(panel, factor)
            if combination_spec:
                audit_compute = lambda prefix: FactorCombinationEngine().run(
                    prefix, factor_path
                ).factor_values
            else:
                audit_compute = lambda prefix: FactorEngine().compute(prefix, factor)
            temporal_audit = FactorEngine.audit_temporal_consistency(
                panel, factor_values, audit_compute
            )
            conditioning_values = None
            conditioning_name = None
            conditioning_audit = None
            if conditional_config.enabled:
                if conditioning_path is None:
                    conditioning_values = factor_values.copy()
                    conditioning_name = factor.factor.name
                    conditioning_audit = temporal_audit
                elif conditioning_source_kind == "factor_combination":
                    conditioning_combination = load_factor_combination(conditioning_path)
                    conditioning_name = conditioning_combination.factor_combination.id
                    conditioning_values = FactorCombinationEngine().run(
                        panel, conditioning_path,
                        cache_context={"data_version": data_version, "base_universe": "conditional_ic"},
                    ).factor_values
                    conditioning_audit = FactorEngine.audit_temporal_consistency(
                        panel, conditioning_values,
                        lambda prefix: FactorCombinationEngine().run(
                            prefix, conditioning_path
                        ).factor_values,
                    )
                else:
                    conditioning_spec = load_factor(conditioning_path)
                    conditioning_name = conditioning_spec.factor.name
                    conditioning_values = FactorEngine().compute(panel, conditioning_spec)
                    conditioning_audit = FactorEngine.audit_temporal_consistency(
                        panel, conditioning_values,
                        lambda prefix: FactorEngine().compute(prefix, conditioning_spec),
                    )
                if conditioning_audit.get("future_data_violations", 0) > 0:
                    raise ValueError(
                        "Conditional IC conditioning factor failed temporal consistency audit"
                    )
                manifest["conditional_ic"] = {
                    "enabled": True,
                    "conditioning_factor": conditioning_name,
                    "source": "main_factor" if conditioning_path is None else str(conditioning_path),
                    "source_kind": "main_factor" if conditioning_path is None else conditioning_source_kind,
                    "quantile_groups": conditional_config.quantile_groups,
                    "min_group_size": conditional_config.min_group_size,
                    "temporal_audit": conditioning_audit,
                }
            # Compute with all pre-sample history, then crop the evidence and backtest.
            # This preserves rolling/lag warm-up at sample_start_date.
            panel = panel.loc[sample_mask].reset_index(drop=True)
            factor_dates = pd.to_datetime(factor_values["trade_date"])
            factor_mask = pd.Series(True, index=factor_values.index)
            if experiment.sample_start_date:
                factor_mask &= factor_dates >= pd.Timestamp(experiment.sample_start_date)
            if experiment.sample_end_date:
                factor_mask &= factor_dates <= pd.Timestamp(experiment.sample_end_date)
            factor_values = factor_values.loc[factor_mask].reset_index(drop=True)
            if conditioning_values is not None:
                conditioning_dates = pd.to_datetime(conditioning_values["trade_date"])
                conditioning_mask = pd.Series(True, index=conditioning_values.index)
                if experiment.sample_start_date:
                    conditioning_mask &= conditioning_dates >= pd.Timestamp(experiment.sample_start_date)
                if experiment.sample_end_date:
                    conditioning_mask &= conditioning_dates <= pd.Timestamp(experiment.sample_end_date)
                conditioning_values = conditioning_values.loc[conditioning_mask].reset_index(drop=True)
                if conditioning_path is not None:
                    artifacts.parquet("conditioning_factor_values.parquet", conditioning_values)
            artifacts.parquet("factor_values.parquet", factor_values)
            l0 = evaluate_factor_quality(
                panel, factor_values, factor, experiment.stage_l0, temporal_audit=temporal_audit
            )
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
            if conditional_config.enabled:
                conditional_result, conditional_daily = evaluate_conditional_ic(
                    panel, factor_values, conditioning_values, factor, experiment.stage_l1,
                    conditioning_name,
                )
                l1["conditional_ic"] = conditional_result
                artifacts.csv(
                    "l1_conditional_ic_summary.csv",
                    pd.json_normalize(conditional_result["results"], sep="_"),
                )
                if conditional_config.store_daily_ic:
                    artifacts.parquet("l1_conditional_ic_daily.parquet", conditional_daily)
            artifacts.json("l1_predictive_power.json", l1)
            if combination_result:
                diagnostics = CombinationDiagnostics().run(
                    panel, factor, experiment, combination_result, main_l1=l1
                )
                for filename, frame in diagnostics.frames.items():
                    artifacts.csv(filename, frame)
                artifacts.text("factor_combination_report.md", diagnostics.report)
                manifest["leakage_checks"] = diagnostics.leakage_checks
                if not all(diagnostics.leakage_checks.values()):
                    assessment = scorer.score(factor, l0, l1, [])
                    assessment["classification"] = "INVALID"
                    assessment["invalid_reason"] = "FACTOR_COMBINATION_LEAKAGE_CHECK_FAILED"
                    return self._finish(artifacts, manifest, assessment, l0, l1, [], started, "INVALID")
            if industry_slice_enabled(experiment):
                industry = IndustrySlicePipeline().run(
                    panel, factor_values, factor, experiment.stage_l1, experiment.industry_slice,
                    factor_builder=(lambda mask: FactorCombinationEngine().run(
                        panel, factor_path, scope_mask=mask,
                        cache_context={"data_version": data_version, "base_universe": "industry_slice"}
                    ).factor_values) if combination_result else None,
                )
                artifacts.csv("industry_selector_summary.csv", industry.selector_summary)
                artifacts.csv("industry_selector_by_year.csv", industry.selector_by_year)
                artifacts.csv("industry_topn_future_returns.csv", industry.selector_summary)
                artifacts.csv("stock_factor_industry_slice_ic.csv", industry.stock_ic)
                artifacts.csv("stock_factor_industry_slice_by_year.csv", industry.stock_ic_by_year)
                artifacts.text("industry_slice_report.md", industry.report)
                artifacts.text("industry_slice_leakage_report.md", industry.leakage_report)
                if experiment.industry_slice.diagnostics.save_industry_intermediate:
                    artifacts.parquet("industry_daily_panel.parquet", industry.industry_panel)
                    artifacts.parquet("stock_industry_slice_panel.parquet", industry.stock_panel)
                manifest["industry_slice"] = {
                    "enabled": True,
                    "preset": experiment.industry_slice.selector.preset,
                    "effective_parameters": experiment.industry_slice.selector.overrides.model_dump(),
                }
            if factor.factor.direction == "unknown":
                assessment = scorer.score(factor, l0, l1, [])
                assessment["classification"] = "WATCHLIST"
                assessment["review_reason"] = "DIRECTION_UNFROZEN"
                return self._finish(artifacts, manifest, assessment, l0, l1, [], started, "STOPPED_DIRECTION_UNFROZEN")
            if not l1["passed"]:
                assessment = scorer.score(factor, l0, l1, [])
                return self._finish(artifacts, manifest, assessment, l0, l1, [], started, "STOPPED_L1")
            condition_filter = experiment.stage_l2.condition_filter
            condition_memberships: dict[str, pd.DataFrame] = {}
            if condition_filter.enabled:
                membership_frames = []
                coverage_rows = []
                for universe in experiment.stage_l2.universes:
                    membership = build_condition_membership(
                        panel,
                        conditioning_values,
                        universe=universe,
                        quantile_groups=conditional_config.quantile_groups,
                        include_quantiles=condition_filter.include_quantiles,
                        min_cross_section=condition_filter.min_cross_section,
                    )
                    membership = membership.loc[membership["selection_eligible"]].copy()
                    condition_memberships[universe] = membership
                    stored = membership.copy()
                    stored["universe"] = universe
                    membership_frames.append(stored)
                    daily_count = membership.groupby("trade_date").size()
                    coverage_rows.append({
                        "universe": universe,
                        "days": int(daily_count.size),
                        "mean_selected": float(daily_count.mean()) if len(daily_count) else 0.0,
                        "median_selected": float(daily_count.median()) if len(daily_count) else 0.0,
                        "min_selected": int(daily_count.min()) if len(daily_count) else 0,
                        "max_selected": int(daily_count.max()) if len(daily_count) else 0,
                    })
                membership_artifact = pd.concat(membership_frames, ignore_index=True)
                artifacts.parquet("l2_condition_membership.parquet", membership_artifact)
                artifacts.csv("l2_condition_filter_summary.csv", pd.DataFrame(coverage_rows))
                manifest["l2_condition_filter"] = {
                    "enabled": True,
                    "source": "stage_l1_conditional_ic",
                    "conditioning_factor": conditioning_name,
                    "quantile_groups": conditional_config.quantile_groups,
                    "include_quantiles": condition_filter.include_quantiles,
                    "min_cross_section": condition_filter.min_cross_section,
                    "benchmark": condition_filter.benchmark,
                }
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
                                selection_membership=condition_memberships.get(universe),
                                position_multiplier=position_multiplier,
                            )
                            condition_key = (
                                "__condition_q" + "_".join(map(str, condition_filter.include_quantiles))
                                if condition_filter.enabled else ""
                            )
                            key = f"{universe}{condition_key}__top{top_n}__hold{holding}__cost{cost}"
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
                                "position_multiplier": position_metadata["enabled"],
                                "condition_quantiles": condition_filter.include_quantiles
                                if condition_filter.enabled else None,
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
    def _normalize_position_multiplier(multiplier: pd.Series | None) -> pd.Series | None:
        if multiplier is None:
            return None
        if not isinstance(multiplier, pd.Series):
            raise TypeError("position_multiplier must be a pandas Series indexed by trade_date")
        result = multiplier.copy()
        result.index = pd.to_datetime(result.index)
        result = pd.to_numeric(result, errors="coerce").dropna().sort_index().clip(0.0, 1.0)
        result = result.groupby(result.index).last()
        if result.empty:
            raise ValueError("position_multiplier is empty after parsing numeric values")
        return result

    @staticmethod
    def _position_multiplier_metadata(multiplier: pd.Series | None, source: str | None) -> dict:
        if multiplier is None:
            return {"enabled": False}
        values = multiplier.to_numpy(float)
        payload = pd.DataFrame({
            "trade_date": multiplier.index.strftime("%Y-%m-%d"),
            "position_multiplier": values,
        }).to_csv(index=False).encode()
        return {
            "enabled": True,
            "source": source,
            "rows": int(len(multiplier)),
            "start_date": multiplier.index.min().date().isoformat(),
            "end_date": multiplier.index.max().date().isoformat(),
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

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
    def _validate_backtest_contract(contract: dict, experiment) -> None:
        """Fail fast when executable experiment settings drift from the V1 contract."""
        l2 = experiment.stage_l2
        expected = {
            "signal_time": (contract.get("signal_time"), "T_CLOSE"),
            "entry": (contract.get("entry"), "T_PLUS_1_OPEN"),
            "rebalance_frequency": (contract.get("rebalance_frequency"), l2.rebalance_frequency),
            "portfolio.model": (contract.get("portfolio", {}).get("model"), "independent_cash_sleeves"),
            "portfolio.weighting": (contract.get("portfolio", {}).get("weighting"), l2.weighting),
            "execution.lot_size": (contract.get("execution", {}).get("lot_size"), l2.lot_size),
            "execution.unfilled_buy": (contract.get("execution", {}).get("unfilled_buy"), l2.no_fill_policy),
        }
        aliases = {"1D": "1D", "equal": "equal", "keep_cash": "keep_cash"}
        mismatches = [name for name, (actual, wanted) in expected.items()
                      if aliases.get(actual, actual) != aliases.get(wanted, wanted)]
        constraints = l2.execution_constraints
        required_policies = {
            "execution.buy_limit_up": (contract.get("execution", {}).get("buy_limit_up") == "reject"
                                       and constraints.cannot_buy_limit_up),
            "execution.sell_limit_down": (contract.get("execution", {}).get("sell_limit_down") == "defer"
                                          and constraints.cannot_sell_limit_down),
            "execution.suspended": (contract.get("execution", {}).get("suspended") == "no_trade"
                                    and constraints.exclude_suspended),
        }
        mismatches.extend(name for name, valid in required_policies.items() if not valid)
        if mismatches:
            raise ValueError("Backtest settings drift from backtest_contract_v1: " + ", ".join(mismatches))

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
