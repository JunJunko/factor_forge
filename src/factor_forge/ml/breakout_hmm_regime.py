from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field

from factor_forge.breakout_process.backtest import EventBacktestRunner
from factor_forge.config import load_project
from factor_forge.data import DataVersionRepository


MARKET_FEATURES = (
    "market_return",
    "market_trend_20",
    "market_volatility_20",
    "breadth_up",
    "breadth_up_20",
    "return_dispersion",
    "amount_change_20",
)


class HMMRegimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "breakout_hmm_regime_v1"
    model_run: Path
    project_config: Path = Path("configs/project.yaml")
    states: int = Field(default=3, ge=2, le=6)
    history_days: int = Field(default=756, ge=252)
    refit_frequency: str = "QE"
    random_restarts: int = Field(default=4, ge=1, le=20)
    max_iterations: int = Field(default=300, ge=10)
    utility_shrink_events: float = Field(default=100.0, ge=0)
    horizon: int = Field(default=10, ge=1, le=60)
    cost_bps: float = Field(default=20.0, ge=0)
    scores: list[str] = Field(
        default_factory=lambda: ["prediction_blend", "prediction_cost_positive"]
    )
    top_n: list[int] = Field(default_factory=lambda: [5, 10])
    holding_days: int = Field(default=10, ge=1, le=60)
    initial_cash: float = Field(default=1_000_000, gt=0)
    lot_size: int = Field(default=100, ge=1)
    min_listing_days: int = Field(default=60, ge=0)
    output_root: Path = Path("artifacts/hmm_regime_runs")


class BreakoutHMMRegimeRunner:
    def run(self, config_path: str | Path) -> dict:
        config_path = Path(config_path)
        cfg = HMMRegimeConfig.model_validate(
            yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        )
        output = cfg.output_root / f"{cfg.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.yaml").write_bytes(config_path.read_bytes())

        model_manifest = json.loads((cfg.model_run / "manifest.json").read_text(encoding="utf-8"))
        research_run = Path(model_manifest["research_run"])
        research_manifest = json.loads((research_run / "manifest.json").read_text(encoding="utf-8"))
        predictions = pd.read_parquet(cfg.model_run / "predictions.parquet")
        predictions["trade_date"] = pd.to_datetime(predictions["trade_date"])
        predictions = predictions.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        events = pd.read_parquet(research_run / "events_with_scores.parquet")
        events["trade_date"] = pd.to_datetime(events["trade_date"])

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
        market = self._market_features(panel_path)
        regimes, fits = self._walk_forward_regimes(market, events, predictions, cfg)
        regimes.to_parquet(output / "regime_scores.parquet", index=False)
        fits.to_csv(output / "hmm_fits.csv", index=False, encoding="utf-8-sig")

        summary, daily, trades = self._backtest(
            predictions, regimes, panel_path, cfg
        )
        summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
        daily.to_parquet(output / "daily.parquet", index=False)
        trades.to_parquet(output / "trades.parquet", index=False)
        (output / "report.md").write_text(self._report(summary, fits), encoding="utf-8")
        manifest = {
            "status": "COMPLETED",
            "model_run": str(cfg.model_run.resolve()),
            "data_version": version,
            "regime_days": int(len(regimes)),
            "hmm_refits": int(len(fits)),
            "backtest_count": int(len(summary)),
            "best_run": str(summary.iloc[0]["run_key"]),
            "best_annualized_return": float(summary.iloc[0]["annualized_return"]),
            "output_path": str(output.resolve()),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    @staticmethod
    def _market_features(panel_path: Path) -> pd.DataFrame:
        available = set(pq.ParquetFile(panel_path).schema.names)
        columns = [
            "trade_date",
            "ts_code",
            "adj_close",
            "amount_cny",
            "is_factor_eligible",
        ]
        if "pct_change" in available:
            columns.append("pct_change")
        panel = pd.read_parquet(panel_path, columns=columns)
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        panel = panel.sort_values(["ts_code", "trade_date"], kind="stable")
        eligible = panel["is_factor_eligible"].fillna(False).astype(bool)
        if "pct_change" in panel:
            stock_return = panel["pct_change"] / 100.0
        else:
            stock_return = panel.groupby("ts_code", sort=False)["adj_close"].pct_change(
                fill_method=None
            )
        source = pd.DataFrame(
            {
                "trade_date": panel["trade_date"],
                "stock_return": stock_return,
                "amount_cny": panel["amount_cny"],
                "eligible": eligible,
            }
        ).loc[eligible]
        grouped = source.groupby("trade_date", sort=True)
        market = pd.DataFrame(
            {
                "market_return": grouped["stock_return"].mean(),
                "breadth_up": grouped["stock_return"].apply(
                    lambda value: float((value.dropna() > 0).mean())
                ),
                "return_dispersion": grouped["stock_return"].std(ddof=0),
                "total_amount": grouped["amount_cny"].sum(min_count=1),
            }
        ).sort_index()
        market["market_trend_20"] = market["market_return"].rolling(20).mean()
        market["market_volatility_20"] = market["market_return"].rolling(20).std(ddof=0)
        market["breadth_up_20"] = market["breadth_up"].rolling(20).mean()
        market["amount_change_20"] = np.log(market["total_amount"]).diff(20)
        return market.reset_index().dropna(subset=list(MARKET_FEATURES))

    def _walk_forward_regimes(
        self,
        market: pd.DataFrame,
        events: pd.DataFrame,
        predictions: pd.DataFrame,
        cfg: HMMRegimeConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        from hmmlearn.hmm import GaussianHMM

        market = market.sort_values("trade_date").reset_index(drop=True)
        prediction_start = predictions["trade_date"].min()
        prediction_end = predictions["trade_date"].max()
        test_market = market.loc[market["trade_date"].between(prediction_start, prediction_end)]
        periods = test_market["trade_date"].dt.to_period("Q").drop_duplicates()
        output: list[pd.DataFrame] = []
        fit_rows: list[dict] = []
        event_label = f"forward_return_{cfg.horizon}"
        for period in periods:
            period_mask = market["trade_date"].dt.to_period("Q") == period
            period_frame = market.loc[
                period_mask & market["trade_date"].between(prediction_start, prediction_end)
            ].copy()
            if period_frame.empty:
                continue
            cutoff = period_frame["trade_date"].min()
            train = market.loc[market["trade_date"] < cutoff].tail(cfg.history_days).copy()
            if len(train) < 252:
                continue
            mean = train[list(MARKET_FEATURES)].mean()
            std = train[list(MARKET_FEATURES)].std(ddof=0).replace(0, 1.0)
            x_train = ((train[list(MARKET_FEATURES)] - mean) / std).to_numpy(dtype=float)
            best_model = None
            best_likelihood = -np.inf
            for restart in range(cfg.random_restarts):
                model = GaussianHMM(
                    n_components=cfg.states,
                    covariance_type="diag",
                    n_iter=cfg.max_iterations,
                    tol=1e-4,
                    random_state=cfg.states * 1000 + restart,
                    min_covar=1e-4,
                )
                model.fit(x_train)
                likelihood = float(model.score(x_train))
                if likelihood > best_likelihood:
                    best_model, best_likelihood = model, likelihood
            train_probability = self._filtered_probabilities(best_model, x_train)
            utility, counts = self._state_utility(
                train,
                train_probability,
                events,
                event_label,
                cutoff,
                cfg,
            )
            train_score = train_probability @ utility
            q40, q60 = np.nanquantile(train_score, [0.4, 0.6])
            x_period = (
                (period_frame[list(MARKET_FEATURES)] - mean) / std
            ).to_numpy(dtype=float)
            period_probability = self._filtered_probabilities(
                best_model, x_period, initial_probability=train_probability[-1]
            )
            expected = period_probability @ utility
            result = period_frame[["trade_date"]].copy()
            for state in range(cfg.states):
                result[f"state_probability_{state}"] = period_probability[:, state]
                result[f"state_utility_{state}"] = utility[state]
            result["regime_score"] = expected
            cost_threshold = cfg.cost_bps / 10_000.0
            result["gate_absolute"] = (expected > cost_threshold).astype(float)
            result["gate_tiered"] = np.select(
                [expected > cost_threshold, expected > 0], [1.0, 0.5], default=0.0
            )
            result["gate_relative"] = np.select(
                [expected > q60, expected > q40], [1.0, 0.5], default=0.0
            )
            result["refit_period"] = str(period)
            output.append(result)
            fit_rows.append(
                {
                    "period": str(period),
                    "cutoff": cutoff,
                    "train_start": train["trade_date"].min(),
                    "train_end": train["trade_date"].max(),
                    "train_days": len(train),
                    "log_likelihood": best_likelihood,
                    "converged": bool(best_model.monitor_.converged),
                    "utility_min": float(np.min(utility)),
                    "utility_max": float(np.max(utility)),
                    "effective_events_min": float(np.min(counts)),
                    "score_q40": float(q40),
                    "score_q60": float(q60),
                }
            )
        regimes = pd.concat(output, ignore_index=True).sort_values("trade_date")
        return regimes, pd.DataFrame(fit_rows)

    @staticmethod
    def _filtered_probabilities(
        model,
        observations: np.ndarray,
        initial_probability: np.ndarray | None = None,
    ) -> np.ndarray:
        raw_variance = (
            getattr(model, "_covars_", model.covars_)
            if getattr(model, "covariance_type", "diag") == "diag"
            else model.covars_
        )
        variance = np.maximum(
            raw_variance,
            1e-8,
        )
        means = model.means_
        log_likelihood = -0.5 * (
            observations.shape[1] * math.log(2 * math.pi)
            + np.log(variance).sum(axis=1)[None, :]
            + (((observations[:, None, :] - means[None, :, :]) ** 2) / variance[None, :, :]).sum(axis=2)
        )
        likelihood = np.exp(log_likelihood - log_likelihood.max(axis=1, keepdims=True))
        output = np.empty((len(observations), model.n_components), dtype=float)
        prior = model.startprob_ if initial_probability is None else initial_probability @ model.transmat_
        for index in range(len(observations)):
            if index > 0:
                prior = output[index - 1] @ model.transmat_
            posterior = prior * likelihood[index]
            total = posterior.sum()
            output[index] = posterior / total if total > 0 else np.full(model.n_components, 1 / model.n_components)
        return output

    @staticmethod
    def _state_utility(
        train: pd.DataFrame,
        probability: np.ndarray,
        events: pd.DataFrame,
        label_column: str,
        cutoff: pd.Timestamp,
        cfg: HMMRegimeConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        # A horizon label is only known after its exit date; remove the final horizon sessions.
        available_end = train["trade_date"].iloc[-(cfg.horizon + 1)]
        sample = events.loc[
            events["trade_date"].between(train["trade_date"].min(), available_end),
            ["trade_date", label_column],
        ].dropna()
        probabilities = pd.DataFrame(
            probability,
            columns=[f"p{state}" for state in range(cfg.states)],
        )
        probabilities["trade_date"] = train["trade_date"].to_numpy()
        sample = sample.merge(probabilities, on="trade_date", how="inner")
        global_mean = float(sample[label_column].mean()) if len(sample) else 0.0
        utility = np.empty(cfg.states, dtype=float)
        counts = np.empty(cfg.states, dtype=float)
        for state in range(cfg.states):
            weight = sample[f"p{state}"].to_numpy(dtype=float)
            value = sample[label_column].to_numpy(dtype=float)
            counts[state] = weight.sum()
            numerator = float(np.dot(weight, value)) + cfg.utility_shrink_events * global_mean
            denominator = counts[state] + cfg.utility_shrink_events
            utility[state] = numerator / denominator if denominator > 0 else global_mean
        return utility, counts

    def _backtest(
        self,
        predictions: pd.DataFrame,
        regimes: pd.DataFrame,
        panel_path: Path,
        cfg: HMMRegimeConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        engine = EventBacktestRunner()
        panel = engine._load_market(panel_path, predictions["trade_date"].min())
        dates = list(pd.Index(panel["trade_date"].unique()).sort_values())
        by_date = {
            pd.Timestamp(date): frame.set_index("ts_code")
            for date, frame in panel.groupby("trade_date", sort=True)
        }
        policies = {
            "ungated": pd.Series(1.0, index=regimes.index),
            "hmm_absolute": regimes["gate_absolute"],
            "hmm_tiered": regimes["gate_tiered"],
            "hmm_relative": regimes["gate_relative"],
        }
        rows, daily_frames, trade_frames = [], [], []
        for score in cfg.scores:
            for top_n in cfg.top_n:
                selections = self._selections(predictions, score, top_n)
                for policy, values in policies.items():
                    exposure = dict(zip(pd.to_datetime(regimes["trade_date"]), values.astype(float)))
                    daily, trades, metrics = engine._simulate(
                        dates,
                        by_date,
                        selections,
                        holding_days=cfg.holding_days,
                        initial_cash=cfg.initial_cash,
                        lot_size=cfg.lot_size,
                        min_listing_days=cfg.min_listing_days,
                        cost_bps=cfg.cost_bps,
                        allocation_count=top_n,
                        exposure_by_signal_date=exposure,
                    )
                    run_key = f"{score}:top{top_n}:{policy}"
                    rows.append(
                        {
                            "run_key": run_key,
                            "score": score,
                            "top_n": top_n,
                            "policy": policy,
                            "mean_exposure": float(values.mean()),
                            "active_ratio": float((values > 0).mean()),
                            **metrics,
                        }
                    )
                    daily.insert(0, "run_key", run_key)
                    trades.insert(0, "run_key", run_key)
                    daily_frames.append(daily)
                    trade_frames.append(trades)
        summary = pd.DataFrame(rows).sort_values("annualized_return", ascending=False).reset_index(drop=True)
        return summary, pd.concat(daily_frames, ignore_index=True), pd.concat(trade_frames, ignore_index=True)

    @staticmethod
    def _selections(
        predictions: pd.DataFrame, score: str, top_n: int
    ) -> dict[pd.Timestamp, list[str]]:
        output = {}
        for trade_date, daily in predictions.groupby("trade_date", sort=True):
            selected = daily.dropna(subset=[score]).sort_values(
                [score, "ts_code"], ascending=[False, True]
            ).head(top_n)
            output[pd.Timestamp(trade_date)] = selected["ts_code"].tolist()
        return output

    @staticmethod
    def _report(summary: pd.DataFrame, fits: pd.DataFrame) -> str:
        lines = [
            "# HMM市场状态门控回测",
            "",
            "HMM使用季度滚动重训与在线过滤概率；LightGBM负责事件日个股排序。",
            f"HMM重训次数：{len(fits)}；全部收敛：{bool(fits['converged'].all())}。",
            "",
            "|分数|TopN|门控|平均仓位|启用比例|年化收益|Sharpe|最大回撤|",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
        for row in summary.itertuples():
            lines.append(
                f"|{row.score}|{row.top_n}|{row.policy}|{row.mean_exposure:.2%}|"
                f"{row.active_ratio:.2%}|{row.annualized_return:.2%}|{row.sharpe:.2f}|"
                f"{row.max_drawdown:.2%}|"
            )
        return "\n".join(lines) + "\n"
