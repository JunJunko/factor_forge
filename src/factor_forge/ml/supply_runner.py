"""End-to-end runner: dataset -> Qlib LightGBM training -> Qlib backtest -> ablation.

Mirrors the data-flow discipline of :class:`factor_forge.ml.runner.MLExperimentRunner`
(immutable data version, segment-coverage gate) but trains through Qlib's native
:class:`LGBModel` / :class:`DatasetH` and backtests through Qlib's ``TopkDropoutStrategy``
+ ``SimulatorExecutor`` with the A-share :class:`AShareExchange`.  The headline output is
``incremental_alpha`` = Performance(Model B) - Performance(Model A) -- the only number that
tests whether the supply-contraction structure adds independent alpha over controls.

A project :class:`BacktestEngine` cross-check on Model B's predictions is run alongside as
an invariant safety net (Qlib's backtest is a first in this repo); it is reported but is
NOT the primary backtest.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import CostModel, ExecutionConstraints, load_project
from factor_forge.data.repository import DataVersionRepository

from .supply_config import SupplyPipelineConfig, load_supply_config
from .supply_dataset import build_supply_dataset, features_for_model, to_qlib_frame
from .supply_qlib_bin import MARKET_BENCHMARK_CODE, dump_supply_bin
from .supply_qlib_strategy import (
    AShareExchange,
    build_ashare_masks,
    ensure_topk_strategy_importable,
    load_topk_dropout_strategy,
)


class _DailyEqualWeight:
    """Qlib reweighter giving each signal date equal aggregate training weight."""

    def reweight(self, data: pd.DataFrame) -> pd.Series:
        return _daily_equal_weighted(data, None)


def _daily_equal_weighted(data: pd.DataFrame, sample_weight: pd.Series | None) -> pd.Series:
    """Per-sample weight = (1/date_count) [* sample_weight], normalized to mean 1.

    Combines the document's two reweighting goals: each trade date contributes equally
    (so a busy day does not dominate training), and within a day low-price-tick-noise /
    illiquid samples are down-weighted via ``sample_weight`` (sec. 9.6).
    """
    dates = data.index.get_level_values("datetime")
    counts = pd.Series(1.0, index=data.index).groupby(dates).transform("sum")
    weights = 1.0 / counts
    if sample_weight is not None:
        sw = sample_weight.reindex(data.index).fillna(1.0)
        weights = weights * sw
    return weights / weights.mean()


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"not serializable: {type(obj)}")


class SupplyContractionRunner:
    def run(self, config_path: str | Path) -> dict:
        try:
            from qlib.contrib.model.gbdt import LGBModel  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                'Qlib + LightGBM are required. Install with: python -m pip install -e ".[qlib,ml]"'
            ) from exc

        config_path = Path(config_path)
        cfg = load_supply_config(config_path)
        # Apply the cvxpy-bypass stub before any qlib.contrib.strategy import (TopkDropoutStrategy's
        # module unconditionally pulls in a cvxpy-backed optimizer that is broken against this numpy).
        ensure_topk_strategy_importable()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + hashlib.sha256(config_path.read_bytes()).hexdigest()[:8]
        output = cfg.output_root / f"{cfg.name}_{run_id}"
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.yaml").write_bytes(config_path.read_bytes())

        version, panel = self._load_panel(cfg)
        if cfg.universe_top_n:
            med = panel.groupby("ts_code")["amount_cny"].median()
            keep = med.nlargest(cfg.universe_top_n).index
            panel = panel[panel["ts_code"].isin(keep)].copy()
        index_daily = self._safe_load_index(cfg, version)

        qlib_dir, _source_hash, instrument_map = dump_supply_bin(
            panel, cfg.qlib_provider_root, version
        )
        self._initialize_qlib(qlib_dir, output, cfg.name)
        from factor_forge.ml.supply_qlib_bin import validate_dump

        sample_code = next(iter(instrument_map.values()))
        validate_dump(qlib_dir, sample_code)

        dataset, _feature_names = build_supply_dataset(
            panel, index_daily, cfg.features, cfg.label,
            sample_weight_train=(cfg.segments.train.start, cfg.segments.train.end)
            if cfg.features.use_sample_weight else None,
        )
        # Align instrument codes to the Qlib bin provider (ts_code -> path-safe code).
        dataset = dataset.copy()
        dataset["instrument"] = dataset["instrument"].map(instrument_map)
        masks = build_ashare_masks(panel, instrument_map, cfg.features.min_listing_days)

        # Per-sample training weight (price_weight * liquidity_weight); None disables it.
        sample_weight = None
        if cfg.features.use_sample_weight and "sample_weight" in dataset.columns:
            sw = dataset.dropna(subset=["sample_weight"]).set_index(["datetime", "instrument"])["sample_weight"]
            if not sw.empty:
                sample_weight = sw

        exchange = self._build_exchange(cfg, list(instrument_map.values()), masks)
        results = {}
        for model_spec in cfg.ablation.models:
            model_features = features_for_model(model_spec)
            res = self._train_and_backtest(
                dataset, model_features, cfg, exchange, model_spec.name, sample_weight
            )
            res["feature_groups"] = model_spec.feature_groups
            res["features"] = model_features
            results[model_spec.name] = res

        incremental_alpha = self._incremental_alpha(cfg, results)
        crosscheck = self._crosscheck(cfg, panel, instrument_map, results) if cfg.crosscheck.enabled else None

        summary = self._write_artifacts(
            output, cfg, version, results, incremental_alpha, crosscheck, panel, instrument_map
        )
        self._end_qlib_experiment()
        return summary

    # --------------------------------------------------------------- data ---
    @staticmethod
    def _load_panel(cfg: SupplyPipelineConfig) -> tuple[str, pd.DataFrame]:
        project = load_project(cfg.project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        version, manifest = repository.load_manifest(cfg.data_version)
        available_start = pd.Timestamp(manifest["start_date"])
        available_end = pd.Timestamp(manifest["end_date"])
        required_start = pd.Timestamp(cfg.segments.train.start)
        required_end = pd.Timestamp(cfg.segments.test.end)
        tolerance = pd.Timedelta(days=7)
        if cfg.require_full_segment_coverage and (
            available_start > required_start + tolerance
            or available_end < required_end - tolerance
        ):
            raise ValueError(
                "data version does not cover segments: required "
                f"{required_start.date()}..{required_end.date()}, available "
                f"{available_start.date()}..{available_end.date()} ({version})"
            )
        _, panel = repository.load_panel(version)
        return version, panel

    @staticmethod
    def _safe_load_index(cfg: SupplyPipelineConfig, version: str) -> pd.DataFrame | None:
        try:
            project = load_project(cfg.project_config)
            repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
            return repository.load_raw_dataset(version, "index_daily")
        except Exception:
            return None

    # --------------------------------------------------------------- qlib ---
    @staticmethod
    def _initialize_qlib(qlib_dir: Path, output: Path, name: str) -> None:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import qlib

        database = (output / "qlib_mlflow.db").resolve().as_posix()
        qlib.init(
            provider_uri=str(qlib_dir.resolve()),
            region="cn",
            exp_manager={
                "class": "MLflowExpManager",
                "module_path": "qlib.workflow.expm",
                "kwargs": {"uri": f"sqlite:///{database}", "default_exp_name": name},
            },
        )

    @staticmethod
    def _end_qlib_experiment() -> None:
        try:
            from qlib.workflow import R

            R.end_exp()
        except Exception:
            pass

    def _build_exchange(self, cfg, codes, masks):
        half = cfg.backtest.round_trip_cost_bps / 10_000.0 / 2.0
        return AShareExchange.make(
            freq="day",
            start_time=cfg.segments.test.start,
            end_time=cfg.segments.test.end,
            codes=codes,
            deal_price=("$open", "$open"),
            limit_threshold=0.095,
            open_cost=half,
            close_cost=half,
            min_cost=0.0,
            trade_unit=cfg.backtest.trade_unit,
            limit_buy_mask=masks[0],
            limit_sell_mask=masks[1],
        )

    # ------------------------------------------------------- train + bt ----
    def _train_and_backtest(self, dataset, model_features, cfg, exchange, model_name, sample_weight=None):
        from qlib.contrib.model.gbdt import LGBModel
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
        from qlib.data.dataset.loader import StaticDataLoader
        from qlib.data.dataset.weight import Reweighter

        class _Reweighter(Reweighter):
            def __init__(self, sample_weight_series=None) -> None:
                self._sw = sample_weight_series

            def reweight(self, data):
                return _daily_equal_weighted(data, self._sw)

        frame = to_qlib_frame(dataset, model_features)
        handler = DataHandlerLP(
            data_loader=StaticDataLoader(frame), infer_processors=[], learn_processors=[]
        )
        segments = {
            name: (getattr(cfg.segments, name).start, getattr(cfg.segments, name).end)
            for name in ("train", "valid", "test")
        }
        ds_obj = DatasetH(handler=handler, segments=segments)

        params = cfg.model.model_dump(exclude={"loss", "num_boost_round", "early_stopping_rounds"})
        model = LGBModel(
            loss=cfg.model.loss,
            num_boost_round=cfg.model.num_boost_round,
            early_stopping_rounds=cfg.model.early_stopping_rounds,
            **params,
        )
        evals_result: dict = {}
        model.fit(ds_obj, verbose_eval=50, evals_result=evals_result, reweighter=_Reweighter(sample_weight))
        pred = model.predict(ds_obj, "test").rename("score")

        port, _indicator = self._qlib_backtest(cfg, pred, exchange)
        metrics = self._metrics_from_port(port)
        metrics["rank_ic_mean"], metrics["rank_ic_ir"] = self._rank_ic(pred, dataset, cfg)
        report = port[next(iter(port))][0] if isinstance(port, dict) else port
        return {
            "prediction": pred.reset_index().rename(
                columns={"datetime": "trade_date", "instrument": "qlib_code", "score": "factor_value"}
            ),
            "portfolio_daily": report.reset_index() if hasattr(report, "reset_index") else pd.DataFrame(),
            "metrics": metrics,
            "evals_result": evals_result,
        }

    def _qlib_backtest(self, cfg, pred, exchange):
        from qlib.backtest import backtest

        strategy = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": pred,
                "topk": cfg.backtest.topk,
                "n_drop": cfg.backtest.n_drop,
                "hold_thresh": cfg.backtest.hold_thresh,
                "only_tradable": True,
                "forbid_all_trade_at_limit": False,
            },
        }
        executor = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        }
        return backtest(
            start_time=cfg.segments.test.start,
            end_time=cfg.segments.test.end,
            strategy=strategy,
            executor=executor,
            benchmark=MARKET_BENCHMARK_CODE,
            account=cfg.backtest.initial_cash,
            exchange_kwargs={"exchange": exchange},
        )

    @staticmethod
    def _metrics_from_port(port) -> dict:
        nan = {"annualized_return": np.nan, "annualized_return_gross": np.nan,
               "annualized_volatility": np.nan, "sharpe": np.nan, "max_drawdown": np.nan,
               "annualized_excess_return": np.nan, "avg_daily_turnover": np.nan,
               "avg_daily_cost_bps": np.nan}
        # qlib 0.9.7: backtest() returns {freq: (report_df, indicator_df)}; the daily
        # report's ``return`` is GROSS and ``cost`` is the daily cost fraction, so the
        # NET return (what the NAV actually realizes) is ``return - cost``.  Report NET as
        # the primary annualized_return; keep GROSS separately for diagnostics.
        if isinstance(port, dict):
            freq = next(iter(port))
            report = port[freq][0]
        else:
            report = port
        if report is None or not hasattr(report, "columns") or report.empty or "return" not in report.columns:
            return nan
        gross = report["return"].astype(float)
        cost = report["cost"].astype(float) if "cost" in report.columns else pd.Series(0.0, index=report.index)
        net = (gross - cost).dropna()
        gross_clean = gross.loc[net.index]
        if len(net) < 2:
            return nan

        def _ann(r: pd.Series) -> float:
            return float((1 + r).prod() ** (252.0 / len(r)) - 1.0)

        ann_vol = float(net.std(ddof=1) * np.sqrt(252.0))
        cum = (1 + net).cumprod()
        metrics = {
            "annualized_return": _ann(net),                 # NET (return - cost)
            "annualized_return_gross": _ann(gross_clean),
            "annualized_volatility": ann_vol,
            "sharpe": float(_ann(net) / ann_vol) if ann_vol > 0 else np.nan,
            "max_drawdown": float((cum.div(cum.cummax()) - 1).min()),
            "annualized_excess_return": np.nan,
            "avg_daily_turnover": float(report["turnover"].mean()) if "turnover" in report.columns else np.nan,
            "avg_daily_cost_bps": float(cost.mean() * 10_000),
        }
        if "bench" in report.columns:
            bench = report["bench"].astype(float).loc[net.index]
            excess = (net - bench).dropna()
            if len(excess) > 1:
                metrics["annualized_excess_return"] = _ann(excess)
        return metrics

    @staticmethod
    def _rank_ic(pred: pd.Series, dataset: pd.DataFrame, cfg) -> tuple[float, float | None]:
        label = dataset.set_index(["datetime", "instrument"])["label"]
        joined = pred.to_frame("score").join(label.rename("label"), how="inner").dropna()
        daily = (
            joined.groupby(level="datetime")
            .apply(lambda x: x["score"].corr(x["label"], method="spearman") if len(x) >= 2 else np.nan)
        ).dropna()
        mean = float(daily.mean()) if len(daily) else np.nan
        ir = float(daily.mean() / daily.std() * np.sqrt(252.0)) if len(daily) > 1 and daily.std() > 0 else None
        return mean, ir

    # ----------------------------------------------------------- analysis ---
    @staticmethod
    def _incremental_alpha(cfg, results) -> dict:
        names = [m.name for m in cfg.ablation.models]
        baseline, contender = names[0], names[1]
        out = {}
        for metric in ("annualized_return", "sharpe", "max_drawdown", "rank_ic_mean"):
            a = results[baseline]["metrics"].get(metric, np.nan)
            b = results[contender]["metrics"].get(metric, np.nan)
            out[metric] = float(b - a) if np.isfinite(a) and np.isfinite(b) else np.nan
        out["baseline"] = baseline
        out["contender"] = contender
        out["verdict"] = (
            "SUPPLY_ALPHA_PRESENT" if out["rank_ic_mean"] > 0 and out["annualized_return"] > 0
            else "NO_INCREMENTAL_ALPHA"
        )
        return out

    def _crosscheck(self, cfg, panel, instrument_map, results):
        """Run the project BacktestEngine on Model B's predictions (invariant sanity net)."""
        from factor_forge.backtest.engine import BacktestEngine

        names = [m.name for m in cfg.ablation.models]
        contender = names[1]
        pred_df = results[contender]["prediction"].copy()
        # Map qlib code back to panel ts_code.
        reverse = {v: k for k, v in instrument_map.items()}
        pred_df["ts_code"] = pred_df["qlib_code"].map(reverse)
        predictions = pred_df.rename(columns={"trade_date": "trade_date", "factor_value": "factor_value"})[
            ["trade_date", "ts_code", "factor_value"]
        ].dropna()
        predictions["trade_date"] = pd.to_datetime(predictions["trade_date"])
        test_panel = panel[
            pd.to_datetime(panel["trade_date"]).between(
                pd.Timestamp(cfg.segments.test.start), pd.Timestamp(cfg.segments.test.end)
            )
        ].copy()
        try:
            result = BacktestEngine().run(
                test_panel, predictions,
                universe=cfg.crosscheck.universe,
                top_n=cfg.crosscheck.top_n,
                holding_days=cfg.crosscheck.holding_days,
                initial_cash=cfg.backtest.initial_cash,
                lot_size=cfg.backtest.trade_unit,
                constraints=ExecutionConstraints(),
                cost_model=CostModel(),
                cost_scenario_bps=cfg.crosscheck.cost_bps,
            )
        except Exception as exc:  # cross-check is advisory; never fatal
            return {"error": f"BacktestEngine cross-check failed: {exc}"}
        return {
            "engine": "BacktestEngine",
            "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in result.metrics.items()},
        }

    # ----------------------------------------------------------- artifacts --
    def _write_artifacts(self, output, cfg, version, results, incremental_alpha, crosscheck, panel, instrument_map):
        for name, res in results.items():
            tag = name.replace(" ", "_")
            res["prediction"].to_parquet(output / f"predictions_{tag}.parquet", index=False)
            res["portfolio_daily"].to_parquet(output / f"portfolio_daily_{tag}.parquet", index=False)
            (output / f"metrics_{tag}.json").write_text(
                json.dumps(res["metrics"], ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
            )
        (output / "instrument_map.json").write_text(
            json.dumps(instrument_map, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report = self._report(cfg, version, results, incremental_alpha, crosscheck)
        (output / "report.md").write_text(report, encoding="utf-8")
        summary = {
            "name": cfg.name,
            "data_version": version,
            "run_dir": str(output),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "models": {name: res["metrics"] for name, res in results.items()},
            "model_feature_groups": {name: res["feature_groups"] for name, res in results.items()},
            "incremental_alpha": incremental_alpha,
            "crosscheck": crosscheck,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
        )
        return summary

    @staticmethod
    def _report(cfg, version, results, incremental_alpha, crosscheck) -> str:
        names = list(results.keys())
        lines = [
            "# 低量上涨供给收缩因子 — Qlib + LightGBM 报告",
            "",
            f"- 数据版本：`{version}`",
            f"- Train：{cfg.segments.train.start} ~ {cfg.segments.train.end}",
            f"- Valid：{cfg.segments.valid.start} ~ {cfg.segments.valid.end}",
            f"- Test ：{cfg.segments.test.start} ~ {cfg.segments.test.end}",
            f"- 标签：{cfg.label.label_method}，horizon={cfg.label.horizon}，行业中性={cfg.label.industry_neutralize}",
            f"- 回测：topk={cfg.backtest.topk}, n_drop={cfg.backtest.n_drop}, hold_thresh={cfg.backtest.hold_thresh}, "
            f"成本={cfg.backtest.round_trip_cost_bps:.0f}bps",
            "",
            "## A/B 消融结果",
            "",
            "|模型|特征组|Test RankIC|年化收益|Sharpe|最大回撤|",
            "|---|---|---:|---:|---:|---:|",
        ]
        for name in names:
            m = results[name]["metrics"]
            groups = "+".join(results[name]["feature_groups"])
            lines.append(
                f"|{name}|{groups}|{m.get('rank_ic_mean', float('nan')):.4f}|"
                f"{m.get('annualized_return', float('nan')):.2%}|"
                f"{m.get('sharpe', float('nan')):.2f}|"
                f"{m.get('max_drawdown', float('nan')):.2%}|"
            )
        lines += [
            "",
            "## 增量 Alpha（Model B − Model A）",
            "",
            f"- RankIC 增量：{incremental_alpha['rank_ic_mean']:+.4f}",
            f"- 年化收益增量：{incremental_alpha['annualized_return']:+.2%}",
            f"- Sharpe 增量：{incremental_alpha['sharpe']:+.2f}",
            f"- **判定：{incremental_alpha['verdict']}**",
            "",
            "> 若增量 Alpha 不稳定为正，则「低量上涨供给收缩」结构未提供可靠的独立 Alpha。",
        ]
        if crosscheck and "metrics" in crosscheck:
            cm = crosscheck["metrics"]
            lines += [
                "",
                "## BacktestEngine 交叉核对（Model B，仅作不变量 sanity check）",
                "",
                f"- 年化超额：{cm.get('annualized_excess_return', float('nan')):.2%}",
                f"- Sharpe：{cm.get('sharpe', float('nan')):.2f}",
                f"- 最大回撤：{cm.get('max_drawdown', float('nan')):.2%}",
                "",
                "Qlib 回测与项目 BacktestEngine 在同一信号上的差异若显著，需排查 Qlib A 股策略接线。",
            ]
        return "\n".join(lines) + "\n"
