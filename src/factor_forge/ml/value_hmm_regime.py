"""Leakage-safe HMM market-regime validation for the frozen value strategy.

Run with::
    python -m factor_forge.ml.value_hmm_regime configs/ml/value_hmm_regime_v1.yaml
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from factor_forge.backtest import BacktestEngine, BacktestResult
from factor_forge.config import CostModel, ExecutionConstraints, load_project
from factor_forge.data import DataVersionRepository
from factor_forge.ml.value_dataset import build_value_dataset
from factor_forge.ml.value_regression import ValueRegressionRunner, load_value_regression_config

FEATURES = [
    "market_return_20d", "market_return_slope_20d", "market_volatility_20d",
    "market_breadth_5d", "market_breadth_20d", "industry_breadth",
    "industry_dispersion", "market_turnover_change_5_20",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HMMSettings(_Strict):
    n_components: int = Field(3, ge=3, le=3)
    covariance_type: str = "diag"
    history_days: int = Field(756, ge=756)
    zscore_window: int = Field(756, ge=60)
    zscore_min_periods: int = Field(60, ge=20)
    refit_frequency: str = "monthly"
    seeds: list[int] = Field(default_factory=lambda: [11, 23, 37, 53, 71], min_length=5)
    max_iterations: int = 300
    tolerance: float = 1e-4
    min_covar: float = 1e-4

    @model_validator(mode="after")
    def fixed_contract(self):
        if self.covariance_type != "diag" or self.refit_frequency != "monthly":
            raise ValueError("v1 requires diag covariance and monthly refits")
        return self


class PortfolioSettings(_Strict):
    top_n: int = 5
    holding_days: int = 10
    roundtrip_cost_bps: float = 15.0
    probability_weights: list[float] = [1.0, 0.6, 0.2]
    probability_smoothing_days: int = 3


class ValueHMMConfig(_Strict):
    version: int = 1
    name: str = "value_hmm_regime_v1"
    experiment_config: Path
    frozen_model_run: Path
    data_version: str
    output_root: Path = Path("artifacts/value_hmm_regime_validations")
    hmm: HMMSettings = HMMSettings()
    portfolio: PortfolioSettings = PortfolioSettings()


@dataclass
class Inputs:
    experiment: Any
    version: str
    panel: pd.DataFrame
    signals: pd.DataFrame
    market: pd.DataFrame


class ValueHMMRegimeRunner:
    def run(self, config_path: str | Path) -> dict:
        path = Path(config_path)
        cfg = ValueHMMConfig.model_validate(yaml.safe_load(path.read_text("utf-8")) or {})
        suffix = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
        output = cfg.output_root / f"{cfg.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{suffix}"
        output.mkdir(parents=True, exist_ok=False)
        inputs = self._inputs(cfg)
        standardized = self._standardize_market(inputs.market, cfg)
        standardized.to_csv(output / "hmm_input_features.csv", index=False, encoding="utf-8-sig")

        seed_runs: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for seed in cfg.hmm.seeds:
            seed_runs[seed] = self._walk_forward(inputs.market, cfg, seed)
        selected_seed = self._choose_seed(seed_runs, inputs, cfg)
        states, fits = seed_runs[selected_seed]
        fits.to_csv(output / "hmm_model_fits.csv", index=False, encoding="utf-8-sig")
        ranks = self._validation_ranks(states, inputs, cfg)
        states = self._decorate_states(states, ranks, cfg, inputs.market)
        results = self._backtests(states, inputs, cfg)
        zero_cost_results = self._backtests(states, inputs, cfg, cost=0.0)
        test_states = states.loc[states.trade_date.ge(pd.Timestamp(inputs.experiment.segments.test.start))]
        cost10_result = self._engine(
            inputs, cfg, test_states.set_index("trade_date").position_multiplier,
            cost=20.0, period="test")
        cost10_metrics = self._period_metrics(cost10_result.daily)

        characteristics, transition = self._characteristics(states, inputs.market)
        regime = self._strategy_by_regime(states, results["baseline"])
        comparison = self._comparison(results, zero_cost_results, states, ranks)
        yearly = self._yearly(results)
        stability = self._stability(seed_runs, selected_seed, inputs, cfg)

        daily_columns = ["trade_date", "predicted_state", "state_name", "state_probability_0",
                         "state_probability_1", "state_probability_2", "position_multiplier",
                         "hmm_train_start", "hmm_train_end"]
        states[daily_columns].to_csv(output / "hmm_daily_states.csv", index=False, encoding="utf-8-sig")
        characteristics.to_csv(output / "hmm_state_characteristics.csv", index=False, encoding="utf-8-sig")
        regime.to_csv(output / "strategy_by_regime.csv", index=False, encoding="utf-8-sig")
        comparison.to_csv(output / "hmm_strategy_comparison.csv", index=False, encoding="utf-8-sig")
        yearly.to_csv(output / "hmm_yearly_comparison.csv", index=False, encoding="utf-8-sig")
        transition.to_csv(output / "hmm_transition_matrix.csv", index=True, encoding="utf-8-sig")
        stability.to_csv(output / "hmm_random_seed_stability.csv", index=False, encoding="utf-8-sig")
        (output / "hmm_regime_summary.md").write_text(
            self._report(cfg, inputs, selected_seed, fits, ranks, characteristics, transition,
                         regime, comparison, yearly, stability, cost10_metrics), encoding="utf-8")
        manifest = {"status": "COMPLETED", "output_path": str(output.resolve()),
                    "data_version": inputs.version, "frozen_model_run": str(cfg.frozen_model_run.resolve()),
                    "selected_seed": selected_seed, "state_rank": ranks}
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
        return manifest

    def _inputs(self, cfg: ValueHMMConfig) -> Inputs:
        experiment = load_value_regression_config(cfg.experiment_config)
        project = load_project(experiment.project_config)
        repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        version, panel = repo.load_panel(cfg.data_version)
        summary_path = cfg.frozen_model_run / "summary.json"
        if summary_path.exists():
            frozen_version = json.loads(summary_path.read_text("utf-8")).get("data_version")
            if frozen_version and frozen_version != version:
                raise ValueError(f"frozen model data version {frozen_version} != {version}")
        working = ValueRegressionRunner._attach_fundamentals(panel, experiment)
        dataset, feature_names, _ = build_value_dataset(
            working, horizons=experiment.labels.horizons,
            parameters=experiment.features.parameters(),
            excess_over_universe=experiment.labels.excess_over_universe)
        mask = pd.to_datetime(dataset["trade_date"]).between(
            pd.Timestamp(experiment.segments.valid.start), pd.Timestamp(experiment.segments.test.end))
        eligible = dataset[f"is_{experiment.portfolio.universe}"].eq(True)
        enough = dataset[feature_names].notna().sum(axis=1).ge(experiment.features.minimum_non_null_features)
        pred = dataset.loc[mask & eligible & enough, ["trade_date", "ts_code"]].copy()
        import lightgbm as lgb
        for horizon in experiment.labels.horizons:
            booster = lgb.Booster(model_file=str(cfg.frozen_model_run / f"full_model_{horizon}d.txt"))
            model_features = booster.feature_name()
            missing = set(model_features) - set(dataset.columns)
            if missing:
                raise ValueError(f"frozen model missing input features: {sorted(missing)}")
            pred[f"prediction_{horizon}d"] = booster.predict(dataset.loc[pred.index, model_features])
        blend = pd.Series(0.0, index=pred.index)
        for horizon, weight in experiment.labels.blend_weights.items():
            blend += weight * pred.groupby("trade_date")[f"prediction_{horizon}d"].rank(pct=True)
        pred["factor_value"] = blend
        market = self._market_features(panel)
        panel = panel.loc[pd.to_datetime(panel.trade_date).between(
            pd.Timestamp(experiment.segments.valid.start), pd.Timestamp(experiment.segments.test.end))].copy()
        return Inputs(experiment, version, panel, pred[["trade_date", "ts_code", "factor_value"]], market)

    @staticmethod
    def _market_features(panel: pd.DataFrame) -> pd.DataFrame:
        p = panel.copy()
        p["trade_date"] = pd.to_datetime(p["trade_date"])
        p = p.sort_values(["ts_code", "trade_date"])
        ret = (p["pct_change"] / 100 if "pct_change" in p else p.groupby("ts_code")["adj_close"].pct_change(fill_method=None))
        # HMM inputs describe the broad investable market, not the strategy's
        # narrower liquid selection universe.
        eligible = p.get("is_factor_eligible", p.get("is_tradeable", pd.Series(True, index=p.index))).fillna(False)
        s = pd.DataFrame({"trade_date": p.trade_date, "ts_code": p.ts_code, "ret": ret,
                          "amount": p.get("amount_cny", np.nan),
                          "industry": p.get("industry_l1_code", "UNKNOWN")}).loc[eligible.astype(bool)]
        g = s.groupby("trade_date", sort=True)
        daily = pd.DataFrame({"market_return": g.ret.mean(), "breadth": g.ret.apply(lambda x: (x.dropna() > 0).mean()),
                              "turnover": g.amount.sum(min_count=1)})
        ind = s.groupby(["trade_date", "industry"], observed=True).ret.mean().reset_index()
        ig = ind.groupby("trade_date").ret
        daily["industry_breadth"] = ig.apply(lambda x: (x.dropna() > 0).mean())
        daily["industry_dispersion"] = ig.std(ddof=0)
        daily["market_return_20d"] = (1 + daily.market_return).rolling(20).apply(np.prod, raw=True) - 1
        # OLS slope of cumulative log return, computed with current and previous observations only.
        x = np.arange(20, dtype=float); xc = x - x.mean(); denom = np.dot(xc, xc)
        logret = np.log1p(daily.market_return.clip(lower=-0.999))
        daily["market_return_slope_20d"] = logret.rolling(20).apply(lambda a: np.dot(xc, np.cumsum(a)) / denom, raw=True)
        daily["market_volatility_20d"] = daily.market_return.rolling(20).std(ddof=0)
        daily["market_breadth_5d"] = daily.breadth.rolling(5).mean()
        daily["market_breadth_20d"] = daily.breadth.rolling(20).mean()
        daily["market_turnover_change_5_20"] = daily.turnover.rolling(5).mean() / daily.turnover.rolling(20).mean() - 1
        return daily.reset_index()[["trade_date", *FEATURES]].replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _filtered(model, x: np.ndarray) -> np.ndarray:
        var = np.maximum(getattr(model, "_covars_", model.covars_), 1e-8)
        means = model.means_
        ll = -.5 * (x.shape[1] * math.log(2 * math.pi) + np.log(var).sum(1)[None, :] +
                    (((x[:, None] - means[None]) ** 2) / var[None]).sum(2))
        likelihood = np.exp(ll - ll.max(1, keepdims=True))
        out = np.empty((len(x), model.n_components)); prior = model.startprob_.copy()
        for i in range(len(x)):
            if i: prior = out[i - 1] @ model.transmat_
            post = prior * likelihood[i]; out[i] = post / post.sum() if post.sum() else 1 / model.n_components
        return out

    @staticmethod
    def _standardize_market(raw: pd.DataFrame, cfg: ValueHMMConfig) -> pd.DataFrame:
        f = raw.sort_values("trade_date").copy().reset_index(drop=True)
        for c in FEATURES:
            f[c] = f[c].ffill()
            past_mean = f[c].rolling(cfg.hmm.zscore_window,
                                     min_periods=cfg.hmm.zscore_min_periods).mean()
            past_std = f[c].rolling(cfg.hmm.zscore_window,
                                    min_periods=cfg.hmm.zscore_min_periods).std(ddof=0)
            f[c + "_z"] = (f[c] - past_mean) / past_std.replace(0, np.nan)
        return f

    def _walk_forward(self, raw: pd.DataFrame, cfg: ValueHMMConfig, seed: int):
        from hmmlearn.hmm import GaussianHMM
        from scipy.optimize import linear_sum_assignment
        # Past-only imputation and rolling standardization; no global moments.
        f = self._standardize_market(raw, cfg)
        zcols = [c + "_z" for c in FEATURES]
        f = f.dropna(subset=zcols).reset_index(drop=True)
        periods = f.trade_date.dt.to_period("M").drop_duplicates()
        outputs, fits, reference = [], [], None
        valid_start = pd.Timestamp(cfg and load_value_regression_config(cfg.experiment_config).segments.valid.start)
        test_end = pd.Timestamp(load_value_regression_config(cfg.experiment_config).segments.test.end)
        for period in periods:
            month = f.loc[f.trade_date.dt.to_period("M").eq(period) & f.trade_date.between(valid_start, test_end)].copy()
            if month.empty: continue
            cutoff = month.trade_date.min(); train = f.loc[f.trade_date.lt(cutoff)].tail(cfg.hmm.history_days)
            if len(train) < cfg.hmm.history_days: continue
            model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=cfg.hmm.max_iterations,
                                tol=cfg.hmm.tolerance, min_covar=cfg.hmm.min_covar, random_state=seed)
            model.fit(train[zcols].to_numpy(float))
            means = model.means_.copy()
            if reference is None:
                order = np.lexsort((means[:, 2], means[:, 0]))
            else:
                _, assigned = linear_sum_assignment(((reference[:, None] - means[None]) ** 2).mean(2))
                order = assigned
            inv = np.argsort(order); reference = means[order]
            prob = self._filtered(model, pd.concat([train, month])[zcols].to_numpy(float))[-len(month):, order]
            out = month[["trade_date"]].copy()
            for i in range(3): out[f"state_probability_{i}"] = prob[:, i]
            out["predicted_state"] = prob.argmax(1); out["hmm_train_start"] = train.trade_date.min()
            out["hmm_train_end"] = train.trade_date.max(); outputs.append(out)
            decoded = inv[model.predict(train[zcols].to_numpy(float))]
            fits.append({"period": str(period), "seed": seed, "train_start": train.trade_date.min(),
                         "train_end": train.trade_date.max(), "train_days": len(train),
                         "log_likelihood": model.score(train[zcols].to_numpy(float)),
                         "converged": bool(model.monitor_.converged),
                         **{f"state_{i}_samples": int((decoded == i).sum()) for i in range(3)},
                         **{f"state_{i}_{zcols[j]}_mean": float(means[order][i, j])
                            for i in range(3) for j in range(len(zcols))},
                         **{f"state_{i}_{zcols[j]}_variance": float(np.asarray(model._covars_)[order][i, j])
                            for i in range(3) for j in range(len(zcols))}})
        if not outputs: raise RuntimeError("no HMM predictions; data must provide >=756 prior sessions")
        return pd.concat(outputs, ignore_index=True), pd.DataFrame(fits)

    def _engine(self, inputs, cfg, multiplier=None, cost=None, period="all"):
        panel, signals = inputs.panel, inputs.signals
        if period == "valid":
            start, end = inputs.experiment.segments.valid.start, inputs.experiment.segments.valid.end
            panel = panel.loc[pd.to_datetime(panel.trade_date).between(pd.Timestamp(start), pd.Timestamp(end))]
            signals = signals.loc[pd.to_datetime(signals.trade_date).between(pd.Timestamp(start), pd.Timestamp(end))]
        elif period == "test":
            start, end = inputs.experiment.segments.test.start, inputs.experiment.segments.test.end
            panel = panel.loc[pd.to_datetime(panel.trade_date).between(pd.Timestamp(start), pd.Timestamp(end))]
            signals = signals.loc[pd.to_datetime(signals.trade_date).between(pd.Timestamp(start), pd.Timestamp(end))]
        return BacktestEngine().run(panel, signals, universe=inputs.experiment.portfolio.universe,
            top_n=cfg.portfolio.top_n, holding_days=cfg.portfolio.holding_days,
            initial_cash=inputs.experiment.portfolio.initial_cash, lot_size=inputs.experiment.portfolio.lot_size,
            constraints=ExecutionConstraints(), cost_model=CostModel(),
            cost_scenario_bps=cfg.portfolio.roundtrip_cost_bps if cost is None else cost,
            position_multiplier=multiplier)

    def _validation_ranks(self, states, inputs, cfg):
        baseline = self._engine(inputs, cfg, period="valid")
        d = baseline.daily.merge(states[["trade_date", "predicted_state"]], on="trade_date")
        excess = d.assign(excess=d["return"] - d["benchmark_return"]).groupby("predicted_state").excess.mean()
        ordered = list(excess.reindex(range(3)).fillna(-np.inf).sort_values(ascending=False).index)
        return {"best": int(ordered[0]), "neutral": int(ordered[1]), "worst": int(ordered[2])}

    def _decorate_states(self, states, ranks, cfg, market):
        s = states.copy(); names = self._state_names(s, market)
        s["state_name"] = s.predicted_state.map(names)
        p = s[[f"state_probability_{i}" for i in range(3)]].rolling(cfg.portfolio.probability_smoothing_days, min_periods=1).mean()
        weight = {ranks["best"]: cfg.portfolio.probability_weights[0], ranks["neutral"]: cfg.portfolio.probability_weights[1], ranks["worst"]: cfg.portfolio.probability_weights[2]}
        s["position_multiplier"] = sum(p[f"state_probability_{i}"] * weight[i] for i in range(3))
        return s

    @staticmethod
    def _state_names(states, market):
        # Rank only by observable market characteristics, never strategy PnL.
        x = states[["trade_date", "predicted_state"]].merge(market, on="trade_date")
        means = x.groupby("predicted_state")[FEATURES].mean().reindex(range(3))
        direction = pd.Series(0.0, index=means.index)
        for column in ["market_return_20d", "market_return_slope_20d",
                       "market_breadth_5d", "market_breadth_20d", "industry_breadth"]:
            scale = means[column].std(ddof=0)
            if np.isfinite(scale) and scale > 0:
                direction += (means[column] - means[column].mean()) / scale
        for column in ["market_volatility_20d", "industry_dispersion"]:
            scale = means[column].std(ddof=0)
            if np.isfinite(scale) and scale > 0:
                direction -= (means[column] - means[column].mean()) / scale
        order = direction.sort_values().index.tolist()
        labels = ["下跌收缩", "震荡轮动", "上涨扩散"]
        return {int(state): f"state_{int(state)}_{label}" for state, label in zip(order, labels)}

    def _backtests(self, states, inputs, cfg, cost=None):
        test_start = pd.Timestamp(inputs.experiment.segments.test.start)
        state_test = states.loc[states.trade_date.ge(test_start)]
        worst = self._validation_ranks(states, inputs, cfg)["worst"]
        hard = state_test.set_index("trade_date").predicted_state.ne(worst).astype(float)
        prob = state_test.set_index("trade_date").position_multiplier
        return {"baseline": self._engine(inputs, cfg, cost=cost, period="test"),
                "hard_gate": self._engine(inputs, cfg, hard, cost=cost, period="test"),
                "probability_position": self._engine(inputs, cfg, prob, cost=cost, period="test")}

    @staticmethod
    def _period_metrics(d):
        if d.empty: return {k: np.nan for k in ["strategy_return","benchmark_return","excess_return","annualized_return","annualized_excess_return","sharpe","max_drawdown","win_rate","turnover"]} | {"sample_days": 0}
        sr=(1+d["return"]).prod()-1; br=(1+d["benchmark_return"]).prod()-1; n=len(d)
        ann=(1+sr)**(252/n)-1; bann=(1+br)**(252/n)-1; vol=d["return"].std(ddof=1)*np.sqrt(252)
        curve=(1+d["return"]).cumprod(); dd=(curve/curve.cummax()-1).min()
        return {"strategy_return":sr,"benchmark_return":br,"excess_return":sr-br,"annualized_return":ann,
                "annualized_excess_return":ann-bann,"sharpe":ann/vol if vol else np.nan,"max_drawdown":dd,
                "win_rate":(d["return"]>0).mean(),"turnover":d.portfolio_turnover.mean()*252,"sample_days":n}

    def _strategy_by_regime(self, states, baseline):
        d=baseline.daily.merge(states[["trade_date","predicted_state","state_name"]],on="trade_date")
        trades = baseline.trades.merge(
            states[["trade_date", "predicted_state"]], on="trade_date", how="inner"
        ) if not baseline.trades.empty else pd.DataFrame(columns=["predicted_state"])
        rows=[]
        for (state,name), sf in d.groupby(["predicted_state","state_name"]):
            for year, y in [("ALL",sf), *[(str(v),g) for v,g in sf.groupby(sf.trade_date.dt.year)]]:
                m=self._period_metrics(y)
                state_trades = trades.loc[trades.predicted_state.eq(state)]
                if year != "ALL" and not state_trades.empty:
                    state_trades = state_trades.loc[state_trades.trade_date.dt.year.eq(int(year))]
                rows.append({"state":state,"state_name":name,"year":year,**m,
                             "trade_count":int(len(state_trades))})
        return pd.DataFrame(rows)

    def _comparison(self, results, zero_cost_results=None, states=None, ranks=None):
        rows=[]
        for version,r in results.items():
            m=self._period_metrics(r.daily); vol=r.daily["return"].std()*np.sqrt(252)
            if zero_cost_results is not None:
                zero_ann = self._period_metrics(zero_cost_results[version].daily)["annualized_return"]
                cost_drag = zero_ann - m["annualized_return"]
            else:
                cost_drag = np.nan
            if states is None:
                average_multiplier = np.nan
            else:
                test_states = states.loc[states.trade_date.ge(pd.Timestamp(r.daily.trade_date.min()))]
            if states is not None and version == "baseline":
                average_multiplier = 1.0
            elif states is not None and version == "hard_gate":
                average_multiplier = float(test_states.predicted_state.ne(ranks["worst"]).mean())
            elif states is not None:
                average_multiplier = float(test_states.position_multiplier.mean())
            rows.append({"version":version,"annualized_return":m["annualized_return"],"benchmark_return":r.metrics["benchmark_annualized_return"],
                         "annualized_excess_return":m["annualized_excess_return"],"sharpe":m["sharpe"],"max_drawdown":m["max_drawdown"],
                         "calmar":m["annualized_return"]/abs(m["max_drawdown"]) if m["max_drawdown"]<0 else np.nan,
                         "annualized_volatility":vol,"turnover":m["turnover"],"cost_drag":cost_drag,
                         "average_position_multiplier":average_multiplier,"average_cash_ratio":r.daily.cash_ratio.mean()})
        return pd.DataFrame(rows)

    def _yearly(self, results):
        rows=[]
        for version,r in results.items():
            for year,d in r.daily.groupby(r.daily.trade_date.dt.year):
                m=self._period_metrics(d); rows.append({"year":year,"version":version,"strategy_return":m["strategy_return"],
                    "benchmark_return":m["benchmark_return"],"excess_return":m["excess_return"],"sharpe":m["sharpe"],
                    "max_drawdown":m["max_drawdown"],"turnover":m["turnover"],"cost_drag":d.transaction_cost.sum()/d.nav.iloc[0]})
        return pd.DataFrame(rows)

    def _characteristics(self, states, market):
        x=states.merge(market,on="trade_date"); seq=x.predicted_state.to_numpy(); runs=[]
        for state in range(3):
            lengths=[]; n=0
            for value in seq:
                if value==state:n+=1
                elif n:lengths.append(n);n=0
            if n:lengths.append(n)
            names = self._state_names(states, market)
            sf=x.loc[x.predicted_state.eq(state)]; row={"state":state,"state_name":names[state],
                "sample_days":len(sf),"sample_ratio":len(sf)/len(x),"average_duration":np.mean(lengths) if lengths else 0,
                "max_duration":max(lengths,default=0),
                "state_switch_count":max(len(lengths)-1, 0)}
            row.update({f"{c}_mean":sf[c].mean() for c in FEATURES}); runs.append(row)
            row.update({f"{c}_variance":sf[c].var(ddof=0) for c in FEATURES})
        transition=pd.crosstab(pd.Series(seq[:-1],name="from_state"),pd.Series(seq[1:],name="to_state"),normalize="index").reindex(index=range(3),columns=range(3),fill_value=0)
        transition.columns=[f"to_state_{i}" for i in range(3)]; transition.index=[f"from_state_{i}" for i in range(3)]
        return pd.DataFrame(runs),transition

    def _choose_seed(self, seed_runs, inputs, cfg):
        # Selection is frozen at validation end; no test-month likelihood is inspected.
        valid_end = pd.Timestamp(inputs.experiment.segments.valid.end).to_period("M")
        def score(seed):
            fits = seed_runs[seed][1]
            eligible = fits.loc[pd.PeriodIndex(fits["period"], freq="M") <= valid_end]
            return eligible.log_likelihood.mean()
        return max(seed_runs, key=score)

    def _stability(self, seed_runs, selected, inputs, cfg):
        from scipy.optimize import linear_sum_assignment
        ref=seed_runs[selected][0]
        ref_profile = ref.merge(inputs.market, on="trade_date").groupby("predicted_state")[FEATURES].mean().reindex(range(3))
        rows=[]
        for seed,(states,_) in seed_runs.items():
            profile = states.merge(inputs.market, on="trade_date").groupby("predicted_state")[FEATURES].mean().reindex(range(3))
            scale = pd.concat([ref_profile, profile]).std(ddof=0).replace(0, 1.0)
            cost = (((ref_profile.to_numpy()[:,None,:]-profile.to_numpy()[None,:,:]) /
                     scale.to_numpy()[None,None,:])**2).mean(2)
            ref_idx, seed_idx = linear_sum_assignment(cost)
            mapping = {int(s): int(r) for r,s in zip(ref_idx, seed_idx)}
            states = states.copy()
            probability = states[[f"state_probability_{i}" for i in range(3)]].copy()
            states["predicted_state"] = states.predicted_state.map(mapping)
            for old,new in mapping.items():
                states[f"state_probability_{new}"] = probability[f"state_probability_{old}"]
            ranks=self._validation_ranks(states,inputs,cfg); decorated=self._decorate_states(states,ranks,cfg,inputs.market)
            results=self._backtests(decorated,inputs,cfg); comp=self._comparison(results).set_index("version")
            merged=ref.merge(states,on="trade_date",suffixes=("_ref","_seed")); similarity=(merged.predicted_state_ref==merged.predicted_state_seed).mean()
            durations=[]
            for _,g in states.groupby((states.predicted_state!=states.predicted_state.shift()).cumsum()): durations.append(len(g))
            rows.append({"seed":seed,"state_characteristic_similarity":similarity,"state_duration":np.mean(durations),
                "baseline_return":comp.loc["baseline","annualized_return"],"hard_gate_return":comp.loc["hard_gate","annualized_return"],
                "probability_position_return":comp.loc["probability_position","annualized_return"],
                "probability_position_excess_return":comp.loc["probability_position","annualized_excess_return"],
                "probability_position_sharpe":comp.loc["probability_position","sharpe"],
                "probability_position_max_drawdown":comp.loc["probability_position","max_drawdown"]})
        return pd.DataFrame(rows)

    @staticmethod
    def _table(df): return df.to_markdown(index=False, floatfmt=".4f")

    def _report(self,cfg,inputs,seed,fits,ranks,char,transition,regime,comparison,yearly,stability,cost10):
        base=comparison.set_index("version").loc["baseline"]; hard=comparison.set_index("version").loc["hard_gate"]
        prob=comparison.set_index("version").loc["probability_position"]
        criteria=[prob.annualized_excess_return-base.annualized_excess_return>=.02,
                  abs(base.max_drawdown)-abs(prob.max_drawdown)>=.05,prob.sharpe-base.sharpe>=.15,
                  prob.calmar>base.calmar,
                  cost10["annualized_excess_return"] > 0]
        stable=int((stability.probability_position_excess_return>0).sum())>=4
        year_table = yearly.set_index(["year", "version"])
        weak_answers = []
        for year in (2024, 2026):
            b = year_table.loc[(year, "baseline"), "strategy_return"]
            p = year_table.loc[(year, "probability_position"), "strategy_return"]
            weak_answers.append(f"{year}{'改善' if p > b else '恶化'}（{b:.2%}→{p:.2%}）")
        if sum(criteria)>=2 and stable: conclusion="HMM Regime有效，可以进入下一轮优化"
        elif abs(prob.max_drawdown)<abs(base.max_drawdown) and prob.annualized_return<=base.annualized_return: conclusion="HMM主要降低回撤，但不能提高收益"
        elif stability.state_characteristic_similarity.mean()<.6: conclusion="HMM状态不稳定，需要修改输入变量"
        else: conclusion="HMM没有明显增量，不建议继续"
        return "\n".join(["# 价值回归 HMM Regime 验证","",f"- 冻结模型：`{cfg.frozen_model_run}`",f"- 数据版本：`{inputs.version}`",
            f"- GaussianHMM：3状态、diag协方差、每月重训、756日窗口；选中种子：{seed}",
            "- 防泄漏：滚动标准化只使用截至当日的数据；每月模型只使用当月首日前756个交易日；概率为前向过滤概率；状态优劣仅由验证集排序。",
            "- HMM只缩放新开仓资金，已有仓位仍按原到期规则卖出。","","## 状态市场特征",self._table(char),"",
            "## 状态转移矩阵",transition.to_markdown(floatfmt=".4f"),"","## 原策略按状态表现",self._table(regime),"",
            "## 三方案总体比较",self._table(comparison),"","## 分年度比较",self._table(yearly),"","## 随机种子稳定性",self._table(stability),"",
            "## 验收回答",f"1. 状态间收益存在差异：{'是' if regime.loc[regime.year.eq('ALL'),'annualized_excess_return'].max()-regime.loc[regime.year.eq('ALL'),'annualized_excess_return'].min()>.02 else '否'}。",
            f"2. 稳定负Alpha状态：{'是' if (regime.loc[regime.year.ne('ALL')].groupby('state').excess_return.max()<0).any() else '否'}。",
            f"3. 硬开关改善超额或回撤：{'是' if hard.annualized_excess_return>base.annualized_excess_return or abs(hard.max_drawdown)<abs(base.max_drawdown) else '否'}。",
            f"4. 概率仓位提高Sharpe和Calmar：{'是' if prob.sharpe>base.sharpe and prob.calmar>base.calmar else '否'}。",
            f"5. 弱阶段损失：{'；'.join(weak_answers)}。",f"6. 换手/成本明显增加：{'是' if prob.turnover>base.turnover*1.05 else '否'}。",
            f"7. 随机种子稳定：{'是' if stable else '否'}。","8. 若收益下降而回撤改善，改善主要来自降低市场Beta；本实验不改变个股排序，不能归因为选股Alpha改善。","",
            f"- 单边10bps成本下概率仓位年化超额：{cost10['annualized_excess_return']:.2%}。","",
            f"## 最终结论\n\n**{conclusion}**",""])


if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("usage: python -m factor_forge.ml.value_hmm_regime CONFIG.yaml")
    print(json.dumps(ValueHMMRegimeRunner().run(sys.argv[1]), ensure_ascii=False, indent=2))
