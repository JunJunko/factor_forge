from __future__ import annotations

import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from factor_forge.backtest.engine import BacktestEngine, BacktestResult
from factor_forge.config import CostModel, ExecutionConstraints, load_project
from factor_forge.data.repository import DataVersionRepository
from .value_dataset import build_value_dataset
from .value_diagnostics import PRICE_FEATURES
from .value_regression import (
    ValueRegressionRunner,
    _daily_equal_weights,
    _purge_tail,
    _segment_mask,
    load_value_regression_config,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FixedValidationConfig(_StrictModel):
    version: int = 1
    name: str = "full_top5_hold10_fixed_validation_v1"
    experiment_config: Path
    output_root: Path = Path("artifacts/value_fixed_validations")


def load_fixed_validation_config(path: str | Path) -> FixedValidationConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return FixedValidationConfig.model_validate(yaml.safe_load(handle) or {})


class FixedValidationRunner:
    TOP_N = [3, 5, 8]
    HOLDING_DAYS = [8, 10, 12]
    SINGLE_SIDE_COSTS = [10, 20, 30, 50]
    CANDIDATE_TOP_N = 5
    CANDIDATE_HOLDING_DAYS = 10
    UNIFORM_PURGE_DAYS = 20

    def run(self, config_path: str | Path) -> dict:
        config_path = Path(config_path)
        cfg = load_fixed_validation_config(config_path)
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
            result = self._execute(experiment, output, logger, stage)
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

    def _execute(self, experiment, output, logger, stage) -> dict:
        stage("load_data")
        project = load_project(experiment.project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        version, panel = repository.load_panel(experiment.data_version)
        working = ValueRegressionRunner._attach_fundamentals(panel, experiment)

        def feature_progress(name: str, completed: int, total: int) -> None:
            if completed == total or completed == 1 or completed % max(total // 10, 1) == 0:
                stage(f"features:{name}", f"{completed}/{total}")

        dataset, full_features, labels = build_value_dataset(
            working,
            horizons=experiment.labels.horizons,
            parameters=experiment.features.parameters(),
            excess_over_universe=experiment.labels.excess_over_universe,
            progress=feature_progress,
        )
        dates = dataset["trade_date"]
        grouped = dataset.groupby("ts_code", sort=False)
        for horizon in experiment.labels.horizons:
            dataset[f"absolute_forward_{horizon}d"] = (
                grouped["adj_open"].shift(-(horizon + 1))
                / grouped["adj_open"].shift(-1) - 1
            )
        masks = {
            "train": _segment_mask(dates, experiment.segments.train),
            "valid": _segment_mask(dates, experiment.segments.valid),
            "test": _segment_mask(dates, experiment.segments.test),
        }
        universe_column = f"is_{experiment.portfolio.universe}"
        eligible = dataset[universe_column].eq(True)
        enough_full = dataset[full_features].notna().sum(axis=1).ge(
            experiment.features.minimum_non_null_features
        )
        price_complete = dataset[PRICE_FEATURES].notna().all(axis=1)

        stage("train_frozen_models")
        full_predictions, full_ic = self._fit_model_set(
            dataset, dates, masks, eligible, enough_full,
            full_features, "full", experiment, output, stage,
        )
        price_predictions, price_ic = self._fit_model_set(
            dataset, dates, masks, eligible, enough_full & price_complete,
            PRICE_FEATURES, "price", experiment, output, stage,
        )
        ic = pd.concat([full_ic, price_ic], ignore_index=True)
        ic.to_csv(output / "rank_ic.csv", index=False, encoding="utf-8-sig")

        full_signals = full_predictions[["trade_date", "ts_code", "full_prediction_blend"]].rename(
            columns={"full_prediction_blend": "factor_value"}
        )
        common = full_predictions.merge(
            price_predictions[["trade_date", "ts_code", "price_prediction_blend"]],
            on=["trade_date", "ts_code"], how="inner",
        )
        comparison_full_signals = common[["trade_date", "ts_code", "full_prediction_blend"]].rename(
            columns={"full_prediction_blend": "factor_value"}
        )
        price_signals = common[["trade_date", "ts_code", "price_prediction_blend"]].rename(
            columns={"price_prediction_blend": "factor_value"}
        )
        test_panel = panel.loc[pd.to_datetime(panel["trade_date"]).between(
            pd.Timestamp(experiment.segments.test.start),
            pd.Timestamp(experiment.segments.test.end),
        )].copy()

        # Preserve the candidate's original simplified 15 bps round-trip cost,
        # which the engine splits evenly into 7.5 bps per side.
        candidate_roundtrip_bps = float(experiment.portfolio.cost_bps)
        stage("candidate_backtest")
        candidate = self._backtest(
            test_panel, full_signals, experiment,
            self.CANDIDATE_TOP_N, self.CANDIDATE_HOLDING_DAYS, candidate_roundtrip_bps,
        )
        candidate_zero = self._backtest(
            test_panel, full_signals, experiment,
            self.CANDIDATE_TOP_N, self.CANDIDATE_HOLDING_DAYS, 0,
        )

        stage("parameter_neighborhood")
        neighborhood_rows = []
        neighborhood_results: dict[tuple[int, int], BacktestResult] = {}
        for top_n in self.TOP_N:
            for holding_days in self.HOLDING_DAYS:
                stage("parameter_neighborhood", f"top={top_n},hold={holding_days}")
                if top_n == self.CANDIDATE_TOP_N and holding_days == self.CANDIDATE_HOLDING_DAYS:
                    result = candidate
                else:
                    result = self._backtest(
                        test_panel, full_signals, experiment,
                        top_n, holding_days, candidate_roundtrip_bps,
                    )
                neighborhood_results[(top_n, holding_days)] = result
                neighborhood_rows.append(self._matrix_row(result, top_n, holding_days))
        neighborhood = pd.DataFrame(neighborhood_rows)
        neighborhood.to_csv(output / "parameter_neighborhood.csv", index=False, encoding="utf-8-sig")

        stage("full_vs_price")
        comparison_full = self._backtest(
            test_panel, comparison_full_signals, experiment, 5, 10, candidate_roundtrip_bps
        )
        comparison_price = self._backtest(
            test_panel, price_signals, experiment, 5, 10, candidate_roundtrip_bps
        )
        comparison = self._full_vs_price(comparison_full, comparison_price)
        comparison.to_csv(output / "full_vs_price_comparison.csv", index=False, encoding="utf-8-sig")

        stage("cost_sensitivity")
        cost_rows = []
        cost_results = {}
        for single_side in self.SINGLE_SIDE_COSTS:
            stage("cost_sensitivity", f"single_side_bps={single_side}")
            result = self._backtest(
                test_panel, full_signals, experiment, 5, 10, single_side * 2.0
            )
            cost_results[single_side] = result
            cost_rows.append({
                "cost_bps": single_side,
                "annualized_return": result.metrics["annualized_return"],
                "annualized_excess_return": result.metrics["annualized_excess_return"],
                "sharpe": result.metrics["sharpe"],
                "max_drawdown": result.metrics["max_drawdown"],
                "turnover": self._annualized_turnover(result),
                "cost_drag": (
                    candidate_zero.metrics["annualized_return"]
                    - result.metrics["annualized_return"]
                ),
            })
        cost_sensitivity = pd.DataFrame(cost_rows)
        cost_sensitivity.to_csv(output / "cost_sensitivity.csv", index=False, encoding="utf-8-sig")

        stage("yearly_performance")
        yearly = self._yearly_performance(candidate, candidate_zero)
        yearly.to_csv(output / "yearly_performance.csv", index=False, encoding="utf-8-sig")

        stage("holding_statistics")
        holding_daily, holding_summary = self._holding_statistics(candidate)
        holding_daily.to_csv(output / "holding_statistics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([holding_summary]).to_csv(
            output / "holding_statistics_summary.csv", index=False, encoding="utf-8-sig"
        )

        stage("decile_returns")
        deciles, decile_summary = self._decile_returns(full_predictions, experiment.labels.horizons)
        deciles.to_csv(output / "decile_returns.csv", index=False, encoding="utf-8-sig")

        summary_metrics = self._summary_metrics(
            candidate, candidate_zero, ic, holding_summary
        )
        answers = self._validation_answers(
            yearly, neighborhood, comparison, cost_sensitivity,
            decile_summary, holding_summary, summary_metrics,
        )
        summary = {
            "data_version": version,
            "test_start": experiment.segments.test.start,
            "test_end": experiment.segments.test.end,
            "uniform_purge_days": self.UNIFORM_PURGE_DAYS,
            "candidate": {"model": "full", "top_n": 5, "holding_days": 10},
            "metrics": summary_metrics,
            "answers": answers,
            "run_dir": str(output),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output / "validation_summary.md").write_text(
            self._validation_markdown(
                experiment, version, summary_metrics, ic, yearly, neighborhood,
                comparison, cost_sensitivity, decile_summary, holding_summary, answers,
            ),
            encoding="utf-8",
        )
        return summary

    def _fit_model_set(
        self, dataset, dates, masks, eligible, feature_ready, features,
        model_name, experiment, output, stage,
    ):
        predictions = dataset.loc[
            masks["test"] & eligible & feature_ready,
            ["trade_date", "ts_code", *[f"label_{h}d" for h in experiment.labels.horizons],
             *[f"absolute_forward_{h}d" for h in experiment.labels.horizons]],
        ].copy()
        ic_rows = []
        for horizon in experiment.labels.horizons:
            stage("train_frozen_models", f"model={model_name},horizon={horizon},purge=20")
            label = f"label_{horizon}d"
            # Fixed validation requirement: all targets use the maximum 20-day purge.
            train = (
                _purge_tail(masks["train"], dates, self.UNIFORM_PURGE_DAYS)
                & eligible & feature_ready & dataset[label].notna()
            )
            valid = (
                _purge_tail(masks["valid"], dates, self.UNIFORM_PURGE_DAYS)
                & eligible & feature_ready & dataset[label].notna()
            )
            model = ValueRegressionRunner._lightgbm(experiment)
            fit_kwargs = {"sample_weight": _daily_equal_weights(dates.loc[train])}
            if valid.any():
                fit_kwargs["eval_set"] = [(dataset.loc[valid, features], dataset.loc[valid, label])]
                fit_kwargs["eval_sample_weight"] = [_daily_equal_weights(dates.loc[valid])]
                if experiment.model.early_stopping_rounds:
                    import lightgbm as lgb
                    fit_kwargs["callbacks"] = [lgb.early_stopping(
                        experiment.model.early_stopping_rounds, verbose=False
                    )]
            model.fit(dataset.loc[train, features], dataset.loc[train, label], **fit_kwargs)
            score = f"{model_name}_prediction_{horizon}d"
            predictions[score] = model.predict(
                dataset.loc[predictions.index, features],
                num_iteration=getattr(model, "best_iteration_", None),
            )
            model.booster_.save_model(str(output / f"{model_name}_model_{horizon}d.txt"))
            daily = predictions[["trade_date", score, label]].dropna().groupby("trade_date").apply(
                lambda frame: frame[score].corr(frame[label], method="spearman"),
                include_groups=False,
            ).dropna()
            ic_rows.append({
                "model": model_name, "horizon": horizon,
                "rank_ic_mean": float(daily.mean()),
                "rank_ic_ir": float(daily.mean() / daily.std()) if daily.std() else np.nan,
                "days": len(daily), "purge_days": self.UNIFORM_PURGE_DAYS,
            })
        blend = pd.Series(0.0, index=predictions.index)
        for horizon, weight in experiment.labels.blend_weights.items():
            blend += weight * predictions.groupby("trade_date")[
                f"{model_name}_prediction_{horizon}d"
            ].rank(pct=True)
        predictions[f"{model_name}_prediction_blend"] = blend
        return predictions, pd.DataFrame(ic_rows)

    @staticmethod
    def _backtest(panel, signals, experiment, top_n, holding_days, roundtrip_bps):
        return BacktestEngine().run(
            panel, signals, universe=experiment.portfolio.universe,
            top_n=top_n, holding_days=holding_days,
            initial_cash=experiment.portfolio.initial_cash,
            lot_size=experiment.portfolio.lot_size,
            constraints=ExecutionConstraints(), cost_model=CostModel(),
            cost_scenario_bps=roundtrip_bps,
        )

    @staticmethod
    def _annualized_turnover(result: BacktestResult) -> float:
        return float(result.daily["portfolio_turnover"].mean() * 252)

    def _matrix_row(self, result, top_n, holding_days):
        return {
            "top_n": top_n, "holding_days": holding_days,
            "annualized_return": result.metrics["annualized_return"],
            "benchmark_return": result.metrics["benchmark_annualized_return"],
            "annualized_excess_return": result.metrics["annualized_excess_return"],
            "sharpe": result.metrics["sharpe"],
            "max_drawdown": result.metrics["max_drawdown"],
            "calmar": result.metrics["calmar"],
            "turnover": self._annualized_turnover(result),
            "average_holdings": float(result.daily["holding_count"].mean()),
        }

    @staticmethod
    def _year_slice(result: BacktestResult, year: int) -> dict:
        daily = result.daily.loc[pd.to_datetime(result.daily["trade_date"]).dt.year.eq(year)].copy()
        strategy_return = float((1 + daily["return"]).prod() - 1)
        benchmark_return = float((1 + daily["benchmark_return"]).prod() - 1)
        volatility = float(daily["return"].std(ddof=1) * np.sqrt(252))
        curve = (1 + daily["return"]).cumprod()
        drawdown = curve / curve.cummax() - 1
        max_drawdown = float(drawdown.min())
        trades = result.trades.loc[pd.to_datetime(result.trades["trade_date"]).dt.year.eq(year)]
        return {
            "strategy_return": strategy_return,
            "benchmark_return": benchmark_return,
            "excess_return": strategy_return - benchmark_return,
            "sharpe": strategy_return / volatility if volatility > 0 else np.nan,
            "max_drawdown": max_drawdown,
            "calmar": strategy_return / abs(max_drawdown) if max_drawdown < 0 else np.nan,
            "turnover": float(daily["portfolio_turnover"].sum()),
            "trade_count": int(len(trades)),
            "execution_rate": (
                float(daily["executed_buys"].sum() / daily["new_signals"].sum())
                if daily["new_signals"].sum() else 0.0
            ),
        }

    def _yearly_performance(self, candidate, zero):
        years = sorted(pd.to_datetime(candidate.daily["trade_date"]).dt.year.unique())
        rows = []
        for year in years:
            item = self._year_slice(candidate, int(year))
            zero_item = self._year_slice(zero, int(year))
            rows.append({"year": int(year), **item,
                         "cost_drag": zero_item["strategy_return"] - item["strategy_return"]})
        columns = [
            "year", "strategy_return", "benchmark_return", "excess_return", "sharpe",
            "max_drawdown", "calmar", "turnover", "cost_drag", "trade_count", "execution_rate",
        ]
        return pd.DataFrame(rows)[columns]

    def _full_vs_price(self, full, price):
        years = sorted(pd.to_datetime(full.daily["trade_date"]).dt.year.unique())
        rows = []
        for period in [*years, "ALL"]:
            if period == "ALL":
                full_item = self._overall_period(full)
                price_item = self._overall_period(price)
            else:
                full_item = self._year_slice(full, int(period))
                price_item = self._year_slice(price, int(period))
            rows.append({
                "year": period,
                "full_return": full_item["strategy_return"],
                "price_return": price_item["strategy_return"],
                "benchmark_return": full_item["benchmark_return"],
                "full_excess_return": full_item["excess_return"],
                "price_excess_return": price_item["excess_return"],
                "full_minus_price": full_item["strategy_return"] - price_item["strategy_return"],
                "full_max_drawdown": full_item["max_drawdown"],
                "price_max_drawdown": price_item["max_drawdown"],
                "full_sharpe": full_item["sharpe"],
                "price_sharpe": price_item["sharpe"],
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _overall_period(result):
        daily = result.daily
        strategy = float((1 + daily["return"]).prod() - 1)
        benchmark = float((1 + daily["benchmark_return"]).prod() - 1)
        volatility = float(daily["return"].std(ddof=1) * np.sqrt(252))
        curve = (1 + daily["return"]).cumprod()
        drawdown = curve / curve.cummax() - 1
        return {
            "strategy_return": strategy, "benchmark_return": benchmark,
            "excess_return": strategy - benchmark,
            "sharpe": result.metrics["annualized_return"] / volatility if volatility else np.nan,
            "max_drawdown": float(drawdown.min()),
        }

    @staticmethod
    def _holding_statistics(result):
        columns = [
            "trade_date", "new_signals", "executed_buys", "executed_sells",
            "holding_count", "unique_holding_count", "portfolio_turnover", "cash_ratio",
            "industry_count", "largest_position_weight",
        ]
        daily = result.daily[columns].copy()
        positions = result.positions.groupby(["trade_date", "ts_code"], as_index=False)["market_value"].sum()
        positions = positions.merge(result.daily[["trade_date", "nav"]], on="trade_date", how="left")
        position_weights = positions["market_value"] / positions["nav"]
        dates = pd.Index(pd.to_datetime(result.daily["trade_date"])).sort_values()
        date_position = {date: index for index, date in enumerate(dates)}
        sells = result.trades.loc[result.trades["side"].eq("SELL")]
        holding_days = [
            date_position.get(pd.Timestamp(row.trade_date), 0)
            - date_position.get(pd.Timestamp(row.entry_date), 0)
            for row in sells.itertuples()
        ]
        summary = {
            "average_holding_count": float(daily["holding_count"].mean()),
            "median_holding_count": float(daily["holding_count"].median()),
            "max_holding_count": int(daily["holding_count"].max()),
            "average_unique_holding_count": float(daily["unique_holding_count"].mean()),
            "median_unique_holding_count": float(daily["unique_holding_count"].median()),
            "max_unique_holding_count": int(daily["unique_holding_count"].max()),
            "average_position_weight": float(position_weights.mean()),
            "average_holding_days": float(np.mean(holding_days)) if holding_days else np.nan,
            "duplicate_signal_rate": (
                float(result.daily["duplicate_signals"].sum() / result.daily["new_signals"].sum())
                if result.daily["new_signals"].sum() else 0.0
            ),
        }
        return daily, summary

    @staticmethod
    def _decile_returns(predictions, horizons):
        score = "full_prediction_blend"
        ranked = predictions.groupby("trade_date")[score].rank(method="first", pct=True)
        decile = np.ceil(ranked * 10).clip(1, 10).astype(int)
        rows = []
        summaries = []
        frame = predictions.assign(decile=decile)
        for horizon in horizons:
            absolute = f"absolute_forward_{horizon}d"
            excess = f"label_{horizon}d"
            for bucket, sample in frame.groupby("decile"):
                daily_abs = sample.groupby("trade_date")[absolute].mean().dropna()
                daily_excess = sample.groupby("trade_date")[excess].mean().dropna()
                average_forward = float(daily_abs.mean())
                average_excess = float(daily_excess.mean())
                rows.append({
                    "model": "full", "horizon": horizon, "decile": int(bucket),
                    "annualized_return": (1 + average_forward) ** (252 / horizon) - 1,
                    "annualized_excess_return": (1 + average_excess) ** (252 / horizon) - 1,
                    "average_forward_return": average_forward,
                    "win_rate": float(sample[absolute].gt(0).mean()),
                    "sample_count": int(sample[absolute].notna().sum()),
                })
            horizon_frame = pd.DataFrame([row for row in rows if row["horizon"] == horizon]).set_index("decile")
            summaries.append({
                "horizon": horizon,
                "D10_return": float(horizon_frame.loc[10, "annualized_return"]),
                "D10_excess_return": float(horizon_frame.loc[10, "annualized_excess_return"]),
                "D9_return": float(horizon_frame.loc[9, "annualized_return"]),
                "D10_minus_D9": float(
                    horizon_frame.loc[10, "annualized_return"] - horizon_frame.loc[9, "annualized_return"]
                ),
                "D10_minus_D1": float(
                    horizon_frame.loc[10, "annualized_return"] - horizon_frame.loc[1, "annualized_return"]
                ),
            })
        result = pd.DataFrame(rows)
        summary = pd.DataFrame(summaries)
        result = result.merge(summary, on="horizon", how="left")
        return result, summary

    def _summary_metrics(self, candidate, zero, ic, holding_summary):
        full_ic = ic.loc[ic["model"].eq("full")].set_index("horizon")
        return {
            "annualized_return": candidate.metrics["annualized_return"],
            "benchmark_annualized_return": candidate.metrics["benchmark_annualized_return"],
            "annualized_excess_return": candidate.metrics["annualized_excess_return"],
            "sharpe": candidate.metrics["sharpe"],
            "max_drawdown": candidate.metrics["max_drawdown"],
            "calmar": candidate.metrics["calmar"],
            "annualized_volatility": candidate.metrics["annualized_volatility"],
            "annualized_turnover": self._annualized_turnover(candidate),
            "cost_drag": zero.metrics["annualized_return"] - candidate.metrics["annualized_return"],
            "execution_rate": candidate.metrics["execution_rate"],
            "average_holding_count": holding_summary["average_holding_count"],
            "max_holding_count": holding_summary["max_holding_count"],
            "rank_ic_5d": float(full_ic.loc[5, "rank_ic_mean"]),
            "rank_icir_5d": float(full_ic.loc[5, "rank_ic_ir"]),
            "rank_ic_10d": float(full_ic.loc[10, "rank_ic_mean"]),
            "rank_icir_10d": float(full_ic.loc[10, "rank_ic_ir"]),
            "rank_ic_20d": float(full_ic.loc[20, "rank_ic_mean"]),
            "rank_icir_20d": float(full_ic.loc[20, "rank_ic_ir"]),
        }

    @staticmethod
    def _validation_answers(yearly, neighborhood, comparison, cost, decile_summary, holdings, metrics):
        positive_years = int(yearly["excess_return"].gt(0).sum())
        positive_year_ratio = positive_years / len(yearly)
        positive_neighbors = int(neighborhood["annualized_excess_return"].gt(0).sum())
        all_row = comparison.loc[comparison["year"].astype(str).eq("ALL")].iloc[0]
        return_increase = float(all_row["full_minus_price"])
        drawdown_reduction = float(abs(all_row["price_max_drawdown"]) - abs(all_row["full_max_drawdown"]))
        cost30_positive = bool(
            cost.loc[cost["cost_bps"].eq(30), "annualized_excess_return"].iloc[0] > 0
        )
        d10_beats = bool((decile_summary["D10_excess_return"] > 0).all())
        holdings_match = bool(
            holdings["average_holding_count"] >= 40
            and holdings["average_holding_count"] <= 55
            and holdings["average_unique_holding_count"] >= 40
        )
        if (
            metrics["annualized_excess_return"] > 0
            and positive_year_ratio >= 2 / 3
            and positive_neighbors >= 6
            and cost30_positive and d10_beats and holdings_match
        ):
            conclusion = "通过验证，可以继续优化"
        elif metrics["annualized_excess_return"] > 0 and positive_neighbors >= 4:
            conclusion = "部分通过，需要修改交易规则"
        else:
            conclusion = "未通过，暂不继续增加因子"
        return {
            "positive_excess_years": positive_years,
            "positive_excess_year_ratio": positive_year_ratio,
            "positive_neighborhood_combinations": positive_neighbors,
            "full_minus_price_return": return_increase,
            "full_drawdown_reduction_vs_price": drawdown_reduction,
            "positive_excess_at_30bps_single_side": cost30_positive,
            "d10_beats_benchmark_all_horizons": d10_beats,
            "holdings_match_top5_design": holdings_match,
            "conclusion": conclusion,
        }

    @staticmethod
    def _md_table(frame: pd.DataFrame) -> str:
        return frame.to_markdown(index=False, floatfmt=".4f")

    def _validation_markdown(
        self, experiment, version, metrics, ic, yearly, neighborhood,
        comparison, cost, decile_summary, holdings, answers,
    ):
        metric_frame = pd.DataFrame([
            {"metric": key, "value": value} for key, value in metrics.items()
        ])
        rules = [
            "1. Top5表示每个交易日从当日收盘后可见的完整模型分数中形成最多5个新信号；下一交易日开盘尝试买入，不成交时不递补。",
            "2. 持仓10日表示每个成功买入批次计划持有10个交易日，在第10个交易日后的到期开盘卖出；无法卖出则顺延。",
            "3. 组合使用10个等额现金sleeve逐日轮换，理论稳态为50个持仓批次；受成交约束、现金取整和延期卖出影响，实际数量见holding_statistics.csv。",
            "4. 同一股票可在不同日期、不同sleeve重复入选并形成独立批次；同一sleeve若旧仓因约束未卖出，则不会再次买入该股票。",
            "5. T日收盘形成信号，T+1原始开盘价买入；计划到期日原始开盘价卖出。持仓估值使用复权开盘价，缺失时使用复权收盘价并向前填充。",
            "6. 股票池为liquid。停牌、开盘涨停、ST、退市整理期、上市不足60交易日或无开盘价时禁止买入且不替补；停牌、开盘跌停或无开盘价时卖出顺延。",
            f"7. 候选策略沿用原15bps往返成本场景，即买卖各{experiment.portfolio.cost_bps / 2:.1f}bps。成本敏感性中的10/20/30/50bps均为单边成本。年化换手率按每日双边成交额/前日NAV的均值×252计算；成本拖累为零成本与相应成本场景的年化收益之差。",
        ]
        answers_text = [
            f"1. 超额收益为正的年份占比：{answers['positive_excess_year_ratio']:.1%}（{answers['positive_excess_years']}/{len(yearly)}）。",
            f"2. 9个附近参数中，正超额组合数量：{answers['positive_neighborhood_combinations']}。",
            f"3. 完整模型相对价格模型总收益变化：{answers['full_minus_price_return']:.2%}；最大回撤降低：{answers['full_drawdown_reduction_vs_price']:.2%}。",
            f"4. 30bps单边成本下仍有正超额：{'是' if answers['positive_excess_at_30bps_single_side'] else '否'}。",
            f"5. D10在5/10/20日三个期限均真正跑赢基准：{'是' if answers['d10_beats_benchmark_all_horizons'] else '否'}。",
            f"6. 实际持仓符合Top5×10日设计：{'是' if answers['holdings_match_top5_design'] else '否'}；平均批次仓位{holdings['average_holding_count']:.2f}个，平均不同股票{holdings['average_unique_holding_count']:.2f}只。",
            f"7. 最终结论：{answers['conclusion']}。",
        ]
        return "\n".join([
            "# 完整模型 + Top5 + 持仓10日：固定参数验证", "",
            "> 本报告为冻结参数验证。未依据测试结果重新选择模型、特征、融合权重或候选参数。",
            "", "## 数据与防泄漏", "",
            f"- 数据版本：`{version}`", f"- 训练：{experiment.segments.train.start} 至 {experiment.segments.train.end}",
            f"- 验证：{experiment.segments.valid.start} 至 {experiment.segments.valid.end}",
            f"- 测试：{experiment.segments.test.start} 至 {experiment.segments.test.end}",
            f"- 所有5/10/20日模型统一使用{self.UNIFORM_PURGE_DAYS}个交易日purge。",
            "- 财务数据按available_date向后as-of合并，available_date为公告后的首个交易日。",
            "", "## 交易规则与持仓机制", "", *rules,
            "", "## 全时期指标", "", self._md_table(metric_frame),
            "", "## Rank IC", "", self._md_table(ic.loc[ic["model"].eq("full")]),
            "", "## 年度表现", "", self._md_table(yearly),
            "", "## 参数邻域（仅稳健性，不用于重新选型）", "", self._md_table(neighborhood),
            "", "## 完整模型与价格模型", "",
            "对比仅使用两模型均有有效预测的共同测试样本，Top5、持仓10日及其余交易规则完全一致。", "",
            self._md_table(comparison),
            "", "## 成本敏感性", "", self._md_table(cost),
            "", "## 十分位额外汇总", "", self._md_table(decile_summary),
            "", "## 持仓汇总", "", self._md_table(pd.DataFrame([holdings])),
            "", "## 明确回答", "", *answers_text, "",
        ])
