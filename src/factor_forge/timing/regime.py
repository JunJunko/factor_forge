from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegimeModelConfig(StrictModel):
    method: Literal["hmm", "gmm"] = "hmm"
    n_components: int = Field(default=3, ge=2, le=5)
    covariance_type: Literal["diag", "full"] = "diag"
    history_days: int = Field(default=252, ge=120, le=1260)
    refit_frequency: Literal["monthly", "quarterly"] = "monthly"
    zscore_window: int = Field(default=252, ge=60, le=1260)
    zscore_min_periods: int = Field(default=60, ge=20)
    random_state: int = 42
    max_iterations: int = Field(default=300, ge=50)
    tolerance: float = Field(default=1e-4, gt=0)
    min_covar: float = Field(default=1e-4, gt=0)


class FactorDiagnosticsConfig(StrictModel):
    min_coverage: float = Field(default=0.55, ge=0, le=1)
    quantiles: int = Field(default=5, ge=2, le=10)
    top_curve_factors: int = Field(default=30, ge=1, le=200)
    exclude_patterns: list[str] = Field(default_factory=lambda: [
        "_low_", "_high_", "label_", "state_probability_",
    ])


class InteractionModelConfig(StrictModel):
    enabled: bool = True
    model_type: Literal["ridge", "elasticnet"] = "ridge"
    alpha: float = Field(default=1.0, ge=0)
    l1_ratio: float = Field(default=0.2, ge=0, le=1)
    train_end: str = "2025-06-30"
    test_start: str = "2025-07-01"
    max_features: int = Field(default=20, ge=5, le=300)

    @model_validator(mode="after")
    def ordered(self):
        if self.train_end >= self.test_start:
            raise ValueError("train_end must be before test_start")
        return self


class TimingRegimeConfig(StrictModel):
    version: int = 1
    name: str = "timing_regime_diagnostics_v1"
    dataset_path: Path
    feature_names_path: Path | None = None
    output_root: Path = Path("artifacts/timing_regime")
    start_date: str = "2023-04-18"
    label_column: str = "label_20d_excess_return"
    label_columns: list[str] | None = None
    regime_features: list[str] = Field(default_factory=lambda: [
        "index_ret_20d",
        "index_ret_60d",
        "index_vol_20d",
        "index_drawdown_60d",
        "up_ratio",
        "rzmre_ratio",
        "put_call_log",
        "iv_atm",
        "fut_near_basis_ann",
        "fut_ls_log",
        "pmi",
        "epu_log",
    ])
    regime: RegimeModelConfig = Field(default_factory=RegimeModelConfig)
    diagnostics: FactorDiagnosticsConfig = Field(default_factory=FactorDiagnosticsConfig)
    interaction_model: InteractionModelConfig = Field(default_factory=InteractionModelConfig)


class TimingRegimeGridConfig(TimingRegimeConfig):
    name: str = "timing_regime_grid_v1"
    output_root: Path = Path("artifacts/timing_regime_grid")
    label_column: str = "label_10d_excess_return"
    label_columns: list[str] | None = Field(default_factory=lambda: ["label_10d_excess_return"])
    methods: list[Literal["hmm", "gmm"]] = Field(default_factory=lambda: ["hmm", "gmm"])
    n_components_grid: list[int] = Field(default_factory=lambda: [2, 3, 4])
    history_days_grid: list[int] = Field(default_factory=lambda: [126, 252, 504])
    random_states: list[int] = Field(default_factory=lambda: [11, 42, 73])


@dataclass(frozen=True)
class TimingRegimeResult:
    output_path: Path
    daily_states: pd.DataFrame
    factor_ic: pd.DataFrame
    interaction_metrics: dict | None


def load_timing_regime_config(path: str | Path) -> TimingRegimeConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return TimingRegimeConfig.model_validate(yaml.safe_load(handle) or {})


def load_timing_regime_grid_config(path: str | Path) -> TimingRegimeGridConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return TimingRegimeGridConfig.model_validate(yaml.safe_load(handle) or {})


class TimingRegimeRunner:
    def run(self, config_path: str | Path) -> dict:
        config_path = Path(config_path)
        cfg = load_timing_regime_config(config_path)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + hashlib.sha256(config_path.read_bytes()).hexdigest()[:8]
        output = cfg.output_root / f"{cfg.name}_{run_id}"
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.yaml").write_bytes(config_path.read_bytes())

        dataset, feature_names = self._load_dataset(cfg)
        label_columns = cfg.label_columns or [cfg.label_column]
        regime_features = [column for column in cfg.regime_features if column in dataset.columns]
        missing_regime = sorted(set(cfg.regime_features) - set(regime_features))
        if len(regime_features) < 3:
            raise ValueError(f"too few regime features found: {regime_features}; missing={missing_regime}")
        states = self._walk_forward_regimes(dataset, regime_features, cfg)
        states.to_csv(output / "regime_daily_states.csv", index=False, encoding="utf-8-sig")

        factors = self._diagnostic_features(dataset, feature_names, cfg, regime_features)
        ic_frames = [
            self._factor_regime_ic(dataset, states, factors, label)
            for label in label_columns
        ]
        factor_ic = pd.concat(ic_frames, ignore_index=True)
        factor_ic.to_csv(output / "factor_regime_ic.csv", index=False, encoding="utf-8-sig")
        quantiles = pd.concat([
            self._factor_regime_quantiles(dataset, states, factors, cfg, label)
            for label in label_columns
        ], ignore_index=True)
        quantiles.to_csv(output / "factor_regime_quantiles.csv", index=False, encoding="utf-8-sig")
        curves = pd.concat([
            self._top_factor_curves(dataset, states, factor_ic.loc[factor_ic["label"].eq(label)], cfg, label)
            for label in label_columns
        ], ignore_index=True)
        curves.to_csv(output / "top_factor_regime_curves.csv", index=False, encoding="utf-8-sig")
        regime_summary = pd.concat([
            self._regime_summary(dataset, states, label)
            for label in label_columns
        ], ignore_index=True)
        regime_summary.to_csv(output / "regime_summary.csv", index=False, encoding="utf-8-sig")

        interaction_metrics = None
        if cfg.interaction_model.enabled:
            interaction_outputs = []
            coefficient_outputs = []
            interaction_metrics = {}
            for label in label_columns:
                metrics, predictions, coefficients = self._interaction_model(
                    dataset, states, factor_ic.loc[factor_ic["label"].eq(label)], cfg, label
                )
                predictions["label"] = label
                coefficients["label"] = label
                interaction_metrics[label] = metrics
                interaction_outputs.append(predictions)
                coefficient_outputs.append(coefficients)
            predictions = pd.concat(interaction_outputs, ignore_index=True)
            coefficients = pd.concat(coefficient_outputs, ignore_index=True)
            predictions.to_csv(output / "interaction_model_predictions.csv", index=False, encoding="utf-8-sig")
            coefficients.to_csv(output / "interaction_model_coefficients.csv", index=False, encoding="utf-8-sig")

        summary = {
            "status": "SUCCESS",
            "run_dir": str(output),
            "dataset_path": str(cfg.dataset_path),
            "rows": int(len(dataset)),
            "start_date": cfg.start_date,
            "label_column": cfg.label_column,
            "label_columns": label_columns,
            "regime_method": cfg.regime.method,
            "regime_features": regime_features,
            "missing_regime_features": missing_regime,
            "diagnostic_feature_count": int(len(factors)),
            "state_counts": states["predicted_state"].value_counts().sort_index().to_dict(),
            "interaction_model": interaction_metrics,
        }
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        (output / "report.md").write_text(
            self._report(summary, regime_summary, factor_ic, quantiles, interaction_metrics),
            encoding="utf-8",
        )
        return summary

    @staticmethod
    def _load_dataset(cfg: TimingRegimeConfig) -> tuple[pd.DataFrame, list[str]]:
        if not cfg.dataset_path.exists():
            raise FileNotFoundError(f"dataset_path does not exist: {cfg.dataset_path}")
        data = pd.read_parquet(cfg.dataset_path)
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        data = data.loc[data["trade_date"].ge(pd.Timestamp(cfg.start_date))].sort_values("trade_date").reset_index(drop=True)
        labels = cfg.label_columns or [cfg.label_column]
        missing_labels = [label for label in labels if label not in data]
        if missing_labels:
            raise ValueError(f"label columns not found: {missing_labels}")
        if cfg.feature_names_path and cfg.feature_names_path.exists():
            feature_names = json.loads(cfg.feature_names_path.read_text(encoding="utf-8"))
        else:
            feature_names = [
                column for column in data.columns
                if column not in {"trade_date", "index_close", cfg.label_column}
                and not column.startswith("label_")
            ]
        return data, [column for column in feature_names if column in data.columns]

    @staticmethod
    def _standardize(data: pd.DataFrame, features: list[str], cfg: TimingRegimeConfig) -> pd.DataFrame:
        result = data[["trade_date", *features]].copy()
        for column in features:
            value = pd.to_numeric(result[column], errors="coerce").ffill()
            mean = value.rolling(cfg.regime.zscore_window, min_periods=cfg.regime.zscore_min_periods).mean()
            std = value.rolling(cfg.regime.zscore_window, min_periods=cfg.regime.zscore_min_periods).std(ddof=0)
            result[f"{column}_z"] = ((value - mean) / std.replace(0, np.nan)).clip(-5, 5)
        return result

    def _walk_forward_regimes(
        self,
        dataset: pd.DataFrame,
        regime_features: list[str],
        cfg: TimingRegimeConfig,
    ) -> pd.DataFrame:
        standardized = self._standardize(dataset, regime_features, cfg)
        zcols = [f"{column}_z" for column in regime_features]
        usable = standardized.dropna(subset=zcols).reset_index(drop=True)
        if len(usable) < cfg.regime.history_days + 20:
            raise ValueError("not enough usable rows for walk-forward regime fitting")
        period_freq = "M" if cfg.regime.refit_frequency == "monthly" else "Q"
        periods = usable["trade_date"].dt.to_period(period_freq).drop_duplicates()
        outputs: list[pd.DataFrame] = []
        reference: np.ndarray | None = None
        for period in periods:
            month = usable.loc[usable["trade_date"].dt.to_period(period_freq).eq(period)].copy()
            cutoff = month["trade_date"].min()
            train = usable.loc[usable["trade_date"].lt(cutoff)].tail(cfg.regime.history_days)
            if len(train) < cfg.regime.history_days:
                continue
            model = self._fit_regime_model(train[zcols].to_numpy(float), cfg)
            means = self._model_means(model)
            order = _align_state_order(means, reference)
            reference = means[order]
            probabilities = self._predict_probabilities(model, pd.concat([train, month])[zcols].to_numpy(float), cfg)
            probabilities = probabilities[-len(month):, order]
            out = month[["trade_date"]].copy()
            for state in range(cfg.regime.n_components):
                out[f"state_probability_{state}"] = probabilities[:, state]
            out["predicted_state"] = probabilities.argmax(axis=1)
            out["regime_train_start"] = train["trade_date"].min()
            out["regime_train_end"] = train["trade_date"].max()
            outputs.append(out)
        if not outputs:
            raise RuntimeError("no walk-forward regime predictions were produced")
        states = pd.concat(outputs, ignore_index=True)
        states = self._name_states(states, dataset, regime_features)
        return states

    @staticmethod
    def _fit_regime_model(x: np.ndarray, cfg: TimingRegimeConfig):
        if cfg.regime.method == "hmm":
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            try:
                from hmmlearn.hmm import GaussianHMM
            except ImportError as exc:
                raise RuntimeError('hmmlearn is required for method="hmm". Install with: python -m pip install -e ".[ml]"') from exc
            model = GaussianHMM(
                n_components=cfg.regime.n_components,
                covariance_type=cfg.regime.covariance_type,
                n_iter=cfg.regime.max_iterations,
                tol=cfg.regime.tolerance,
                min_covar=cfg.regime.min_covar,
                random_state=cfg.regime.random_state,
            )
        else:
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            from sklearn.mixture import GaussianMixture
            model = GaussianMixture(
                n_components=cfg.regime.n_components,
                covariance_type=cfg.regime.covariance_type,
                max_iter=cfg.regime.max_iterations,
                tol=cfg.regime.tolerance,
                reg_covar=cfg.regime.min_covar,
                random_state=cfg.regime.random_state,
            )
        model.fit(x)
        return model

    @staticmethod
    def _model_means(model) -> np.ndarray:
        return np.asarray(model.means_, dtype=float)

    @staticmethod
    def _predict_probabilities(model, x: np.ndarray, cfg: TimingRegimeConfig) -> np.ndarray:
        if cfg.regime.method == "hmm":
            return _hmm_filtered_probabilities(model, x)
        return model.predict_proba(x)

    @staticmethod
    def _name_states(states: pd.DataFrame, dataset: pd.DataFrame, regime_features: list[str]) -> pd.DataFrame:
        merged = states[["trade_date", "predicted_state"]].merge(
            dataset[["trade_date", *regime_features]], on="trade_date", how="left"
        )
        profile = merged.groupby("predicted_state")[regime_features].mean()
        score = pd.Series(0.0, index=profile.index)
        for column in ["index_ret_20d", "index_ret_60d", "up_ratio"]:
            if column in profile:
                std = profile[column].std(ddof=0)
                if np.isfinite(std) and std > 0:
                    score += (profile[column] - profile[column].mean()) / std
        for column in ["index_vol_20d", "index_drawdown_60d", "iv_atm"]:
            if column in profile:
                std = profile[column].std(ddof=0)
                if np.isfinite(std) and std > 0:
                    sign = -1.0 if column != "index_drawdown_60d" else 1.0
                    score += sign * (profile[column] - profile[column].mean()) / std
        ordered = score.sort_values().index.tolist()
        labels = ["stress", "neutral", "risk_on", "extended", "euphoric"]
        mapping = {int(state): f"state_{int(state)}_{labels[pos]}" for pos, state in enumerate(ordered)}
        result = states.copy()
        result["state_name"] = result["predicted_state"].map(mapping)
        return result

    @staticmethod
    def _diagnostic_features(
        dataset: pd.DataFrame,
        feature_names: list[str],
        cfg: TimingRegimeConfig,
        regime_features: list[str],
    ) -> list[str]:
        excluded = set(regime_features)
        result = []
        for column in feature_names:
            if column in excluded or column not in dataset:
                continue
            if any(pattern in column for pattern in cfg.diagnostics.exclude_patterns):
                continue
            coverage = dataset[column].notna().mean()
            if coverage >= cfg.diagnostics.min_coverage:
                result.append(column)
        return result

    @staticmethod
    def _factor_regime_ic(
        dataset: pd.DataFrame,
        states: pd.DataFrame,
        factors: list[str],
        label: str,
    ) -> pd.DataFrame:
        merged = states[["trade_date", "predicted_state", "state_name"]].merge(
            dataset[["trade_date", label, *factors]], on="trade_date", how="left"
        )
        rows: list[dict] = []
        for factor in factors:
            for (state, state_name), frame in merged.groupby(["predicted_state", "state_name"], sort=True):
                sample = frame[[factor, label]].dropna()
                if len(sample) < 20:
                    ic = rank_ic = np.nan
                else:
                    ic = _safe_corr(sample[factor], sample[label], method="pearson")
                    rank_ic = _safe_corr(sample[factor], sample[label], method="spearman")
                rows.append({
                    "factor": factor,
                    "label": label,
                    "state": int(state),
                    "state_name": state_name,
                    "observations": int(len(sample)),
                    "ic": float(ic) if np.isfinite(ic) else np.nan,
                    "rank_ic": float(rank_ic) if np.isfinite(rank_ic) else np.nan,
                    "mean_factor": float(sample[factor].mean()) if len(sample) else np.nan,
                    "mean_forward_return": float(sample[label].mean()) if len(sample) else np.nan,
                })
            sample = merged[[factor, label]].dropna()
            rows.append({
                "factor": factor,
                "label": label,
                "state": -1,
                "state_name": "ALL",
                "observations": int(len(sample)),
                "ic": _safe_corr(sample[factor], sample[label], method="pearson") if len(sample) >= 20 else np.nan,
                "rank_ic": _safe_corr(sample[factor], sample[label], method="spearman") if len(sample) >= 20 else np.nan,
                "mean_factor": float(sample[factor].mean()) if len(sample) else np.nan,
                "mean_forward_return": float(sample[label].mean()) if len(sample) else np.nan,
            })
        return pd.DataFrame(rows).sort_values(["state", "rank_ic"], key=lambda item: item.abs() if item.name == "rank_ic" else item, ascending=False)

    @staticmethod
    def _factor_regime_quantiles(
        dataset: pd.DataFrame,
        states: pd.DataFrame,
        factors: list[str],
        cfg: TimingRegimeConfig,
        label: str,
    ) -> pd.DataFrame:
        merged = states[["trade_date", "predicted_state", "state_name"]].merge(
            dataset[["trade_date", label, *factors]], on="trade_date", how="left"
        )
        rows: list[dict] = []
        for factor in factors:
            for (state, state_name), frame in merged.groupby(["predicted_state", "state_name"], sort=True):
                sample = frame[["trade_date", factor, label]].dropna()
                if len(sample) < cfg.diagnostics.quantiles * 8 or sample[factor].nunique() < cfg.diagnostics.quantiles:
                    continue
                bucket = pd.qcut(sample[factor].rank(method="first"), cfg.diagnostics.quantiles, labels=False) + 1
                sample = sample.assign(quantile=bucket)
                for quantile, local in sample.groupby("quantile"):
                    returns = local[label]
                    rows.append({
                        "factor": factor,
                        "label": label,
                        "state": int(state),
                        "state_name": state_name,
                        "quantile": int(quantile),
                        "observations": int(len(local)),
                        "mean_forward_return": float(returns.mean()),
                        "hit_rate": float(returns.gt(0).mean()),
                        "cumulative_return": float((1 + returns).prod() - 1),
                    })
        return pd.DataFrame(rows)

    @staticmethod
    def _top_factor_curves(
        dataset: pd.DataFrame,
        states: pd.DataFrame,
        factor_ic: pd.DataFrame,
        cfg: TimingRegimeConfig,
        label: str,
    ) -> pd.DataFrame:
        candidates = factor_ic.loc[factor_ic["state"].ge(0)].copy()
        top = (
            candidates.assign(abs_rank_ic=candidates["rank_ic"].abs())
            .sort_values("abs_rank_ic", ascending=False)
            .drop_duplicates(["factor", "state"])
            .head(cfg.diagnostics.top_curve_factors)
        )
        merged = states[["trade_date", "predicted_state", "state_name"]].merge(dataset, on="trade_date", how="left")
        rows: list[dict] = []
        for item in top.itertuples(index=False):
            frame = merged.loc[merged["predicted_state"].eq(item.state), ["trade_date", item.factor, label]].dropna()
            if len(frame) < 20:
                continue
            median = frame[item.factor].median()
            direction = 1.0 if item.rank_ic >= 0 else -1.0
            signal = np.where(frame[item.factor].ge(median), direction, -direction)
            strategy_return = signal * frame[label]
            curve = (1 + strategy_return).cumprod()
            for date, daily_return, nav in zip(frame["trade_date"], strategy_return, curve):
                rows.append({
                    "trade_date": date,
                    "factor": item.factor,
                    "label": label,
                    "state": int(item.state),
                    "state_name": item.state_name,
                    "daily_strategy_return": float(daily_return),
                    "cumulative_nav": float(nav),
                })
        return pd.DataFrame(rows)

    @staticmethod
    def _regime_summary(dataset: pd.DataFrame, states: pd.DataFrame, label: str) -> pd.DataFrame:
        merged = states.merge(dataset[["trade_date", label]], on="trade_date", how="left")
        rows = []
        for (state, state_name), frame in merged.groupby(["predicted_state", "state_name"], sort=True):
            returns = frame[label].dropna()
            rows.append({
                "state": int(state),
                "state_name": state_name,
                "label": label,
                "days": int(len(frame)),
                "days_with_label": int(len(returns)),
                "mean_forward_return": float(returns.mean()) if len(returns) else np.nan,
                "median_forward_return": float(returns.median()) if len(returns) else np.nan,
                "hit_rate": float(returns.gt(0).mean()) if len(returns) else np.nan,
                "return_volatility": float(returns.std(ddof=1)) if len(returns) > 1 else np.nan,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _interaction_model(
        dataset: pd.DataFrame,
        states: pd.DataFrame,
        factor_ic: pd.DataFrame,
        cfg: TimingRegimeConfig,
        label: str,
    ) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
        all_ic = factor_ic.loc[factor_ic["state_name"].eq("ALL")].copy()
        selected = (
            all_ic.assign(abs_rank_ic=all_ic["rank_ic"].abs())
            .sort_values("abs_rank_ic", ascending=False)
            .head(cfg.interaction_model.max_features)["factor"]
            .tolist()
        )
        merged = states.merge(dataset[["trade_date", label, *selected]], on="trade_date", how="left")
        state_cols = [column for column in states.columns if column.startswith("state_probability_")]
        interaction_columns = {
            f"{factor}__x__{state_col}": merged[factor] * merged[state_col]
            for factor in selected
            for state_col in state_cols
        }
        if interaction_columns:
            merged = pd.concat([merged, pd.DataFrame(interaction_columns, index=merged.index)], axis=1)
        model_features = selected + list(interaction_columns)
        train = merged["trade_date"].le(pd.Timestamp(cfg.interaction_model.train_end))
        test = merged["trade_date"].ge(pd.Timestamp(cfg.interaction_model.test_start))
        usable = merged[model_features + [label]].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        train &= usable
        test &= usable
        if train.sum() < 80 or test.sum() < 20:
            raise ValueError("not enough samples for interaction model train/test split")
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if cfg.interaction_model.model_type == "ridge":
            from sklearn.linear_model import Ridge
            estimator = Ridge(alpha=cfg.interaction_model.alpha)
        else:
            from sklearn.linear_model import ElasticNet
            estimator = ElasticNet(alpha=cfg.interaction_model.alpha, l1_ratio=cfg.interaction_model.l1_ratio, max_iter=10000)
        model = make_pipeline(StandardScaler(), estimator)
        model.fit(merged.loc[train, model_features], merged.loc[train, label])
        predictions = merged.loc[train | test, ["trade_date", label, "predicted_state", "state_name"]].copy()
        predictions = predictions.rename(columns={label: "forward_return"})
        predictions["sample"] = np.where(predictions["trade_date"].le(pd.Timestamp(cfg.interaction_model.train_end)), "train", "test")
        predictions["prediction"] = model.predict(merged.loc[predictions.index, model_features])
        metrics = {}
        for name, local in predictions.groupby("sample"):
            corr = _safe_corr(local["prediction"], local["forward_return"], method="spearman")
            metrics[name] = {
                "rows": int(len(local)),
                "rank_ic": float(corr) if np.isfinite(corr) else np.nan,
                "top_half_mean_return": float(local.loc[local["prediction"].ge(local["prediction"].median()), "forward_return"].mean()),
                "bottom_half_mean_return": float(local.loc[local["prediction"].lt(local["prediction"].median()), "forward_return"].mean()),
            }
        estimator_step = model.steps[-1][1]
        coefficients = pd.DataFrame({
            "feature": model_features,
            "coefficient": getattr(estimator_step, "coef_", np.repeat(np.nan, len(model_features))),
        }).sort_values("coefficient", key=lambda item: item.abs(), ascending=False)
        return {
            "model_type": cfg.interaction_model.model_type,
            "selected_feature_count": len(selected),
            "expanded_feature_count": len(model_features),
            "metrics": metrics,
        }, predictions, coefficients

    @staticmethod
    def _report(summary, regime_summary, factor_ic, quantiles, interaction_metrics) -> str:
        top_ic = (
            factor_ic.loc[factor_ic["state"].ge(0)]
            .assign(abs_rank_ic=lambda frame: frame["rank_ic"].abs())
            .sort_values("abs_rank_ic", ascending=False)
            .head(20)
            [["label", "factor", "state_name", "observations", "rank_ic", "ic"]]
        )
        if quantiles.empty:
            top_spreads = pd.DataFrame()
        else:
            spreads = []
            for (label, factor, state_name), frame in quantiles.groupby(["label", "factor", "state_name"]):
                pivot = frame.set_index("quantile")["mean_forward_return"]
                if len(pivot) >= 2:
                    spreads.append({
                        "label": label,
                        "factor": factor,
                        "state_name": state_name,
                        "q_high_minus_low": float(pivot.loc[pivot.index.max()] - pivot.loc[pivot.index.min()]),
                    })
            top_spreads = pd.DataFrame(spreads).sort_values("q_high_minus_low", key=lambda item: item.abs(), ascending=False).head(20) if spreads else pd.DataFrame()
        sections = [
            "# Timing Regime Diagnostics",
            "",
            f"- Dataset: `{summary['dataset_path']}`",
            f"- Start date: `{summary['start_date']}`",
            f"- Method: `{summary['regime_method']}`",
            f"- Diagnostic features: `{summary['diagnostic_feature_count']}`",
            "",
            "## Regime Summary",
            regime_summary.to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Top Conditional IC",
            top_ic.to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Top Quantile Spreads",
            top_spreads.to_markdown(index=False, floatfmt=".4f") if not top_spreads.empty else "No quantile spreads available.",
        ]
        if interaction_metrics is not None:
            sections.extend([
                "",
                "## Regime Interaction Model",
                "```json",
                json.dumps(interaction_metrics, ensure_ascii=False, indent=2),
                "```",
            ])
        return "\n".join(sections) + "\n"


class TimingRegimeGridRunner:
    def run(self, config_path: str | Path) -> dict:
        config_path = Path(config_path)
        cfg = load_timing_regime_grid_config(config_path)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + hashlib.sha256(config_path.read_bytes()).hexdigest()[:8]
        output = cfg.output_root / f"{cfg.name}_{run_id}"
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.yaml").write_bytes(config_path.read_bytes())

        helper = TimingRegimeRunner()
        dataset, feature_names = helper._load_dataset(cfg)
        label = cfg.label_column
        regime_features = [column for column in cfg.regime_features if column in dataset.columns]
        factors = helper._diagnostic_features(dataset, feature_names, cfg, regime_features)

        rows: list[dict] = []
        state_rows: list[pd.DataFrame] = []
        ic_rows: list[pd.DataFrame] = []
        state_paths: dict[tuple, pd.DataFrame] = {}
        total = len(cfg.methods) * len(cfg.n_components_grid) * len(cfg.history_days_grid) * len(cfg.random_states)
        completed = 0
        for method in cfg.methods:
            for n_components in cfg.n_components_grid:
                for history_days in cfg.history_days_grid:
                    for seed in cfg.random_states:
                        completed += 1
                        run_cfg = self._variant_config(cfg, method, n_components, history_days, seed)
                        combo = {
                            "method": method,
                            "n_components": n_components,
                            "history_days": history_days,
                            "random_state": seed,
                        }
                        try:
                            states = helper._walk_forward_regimes(dataset, regime_features, run_cfg)
                            regime_summary = helper._regime_summary(dataset, states, label)
                            factor_ic = helper._factor_regime_ic(dataset, states, factors, label)
                            interaction_metrics = None
                            interaction_error = None
                            if run_cfg.interaction_model.enabled:
                                try:
                                    interaction_metrics, _, _ = helper._interaction_model(
                                        dataset, states, factor_ic, run_cfg, label
                                    )
                                except Exception as exc:
                                    interaction_error = f"{type(exc).__name__}: {exc}"
                            row = self._summarize_combo(combo, states, regime_summary, factor_ic, interaction_metrics)
                            row["interaction_error"] = interaction_error
                            rows.append({"status": "SUCCESS", **row})
                            state_paths[(method, n_components, history_days, seed)] = states[["trade_date", "predicted_state"]].copy()
                            regime_summary.insert(0, "random_state", seed)
                            regime_summary.insert(0, "history_days", history_days)
                            regime_summary.insert(0, "n_components", n_components)
                            regime_summary.insert(0, "method", method)
                            state_rows.append(regime_summary)
                            top_ic = (
                                factor_ic.loc[factor_ic["state"].ge(0)]
                                .assign(abs_rank_ic=lambda frame: frame["rank_ic"].abs())
                                .sort_values("abs_rank_ic", ascending=False)
                                .head(30)
                            )
                            top_ic.insert(0, "random_state", seed)
                            top_ic.insert(0, "history_days", history_days)
                            top_ic.insert(0, "n_components", n_components)
                            top_ic.insert(0, "method", method)
                            ic_rows.append(top_ic)
                        except Exception as exc:
                            rows.append({
                                "status": "FAILED",
                                **combo,
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                            })
                        print(f"timing-regime-grid {completed}/{total} {method} k={n_components} history={history_days} seed={seed}")

        grid = pd.DataFrame(rows)
        stability = self._seed_stability(state_paths)
        if not stability.empty:
            grid = grid.merge(stability, on=["method", "n_components", "history_days"], how="left")
        grid = grid.sort_values(
            ["status", "test_rank_ic", "mean_top20_abs_rank_ic"],
            ascending=[False, False, False],
            na_position="last",
        )
        grid.to_csv(output / "grid_results.csv", index=False, encoding="utf-8-sig")
        if state_rows:
            pd.concat(state_rows, ignore_index=True).to_csv(output / "grid_regime_summary.csv", index=False, encoding="utf-8-sig")
        if ic_rows:
            pd.concat(ic_rows, ignore_index=True).to_csv(output / "grid_top_factor_ic.csv", index=False, encoding="utf-8-sig")
        stability.to_csv(output / "grid_seed_stability.csv", index=False, encoding="utf-8-sig")

        summary = {
            "status": "SUCCESS",
            "run_dir": str(output),
            "dataset_path": str(cfg.dataset_path),
            "label_column": label,
            "total_combinations": total,
            "successful_combinations": int(grid["status"].eq("SUCCESS").sum()) if "status" in grid else 0,
            "best_combinations": grid.head(10).to_dict("records"),
        }
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        (output / "report.md").write_text(self._report(summary, grid, stability), encoding="utf-8")
        return summary

    @staticmethod
    def _variant_config(
        cfg: TimingRegimeGridConfig,
        method: str,
        n_components: int,
        history_days: int,
        random_state: int,
    ) -> TimingRegimeConfig:
        regime = cfg.regime.model_copy(update={
            "method": method,
            "n_components": n_components,
            "history_days": history_days,
            "random_state": random_state,
        })
        return TimingRegimeConfig.model_validate({
            **cfg.model_dump(exclude={
                "methods", "n_components_grid", "history_days_grid", "random_states"
            }),
            "regime": regime.model_dump(),
            "label_columns": [cfg.label_column],
        })

    @staticmethod
    def _summarize_combo(combo, states, regime_summary, factor_ic, interaction_metrics) -> dict:
        successful_ic = factor_ic.loc[factor_ic["state"].ge(0) & factor_ic["rank_ic"].notna()]
        top_abs = (
            successful_ic.assign(abs_rank_ic=successful_ic["rank_ic"].abs())
            .sort_values("abs_rank_ic", ascending=False)
            .head(20)["abs_rank_ic"]
        )
        state_days = states["predicted_state"].value_counts(normalize=True)
        state_return_spread = regime_summary["mean_forward_return"].max() - regime_summary["mean_forward_return"].min()
        row = {
            **combo,
            "predicted_days": int(len(states)),
            "min_state_ratio": float(state_days.min()) if len(state_days) else np.nan,
            "max_state_ratio": float(state_days.max()) if len(state_days) else np.nan,
            "state_return_spread": float(state_return_spread) if np.isfinite(state_return_spread) else np.nan,
            "mean_top20_abs_rank_ic": float(top_abs.mean()) if len(top_abs) else np.nan,
            "median_top20_abs_rank_ic": float(top_abs.median()) if len(top_abs) else np.nan,
        }
        if interaction_metrics:
            test = interaction_metrics["metrics"].get("test", {})
            train = interaction_metrics["metrics"].get("train", {})
            row.update({
                "train_rank_ic": train.get("rank_ic"),
                "test_rank_ic": test.get("rank_ic"),
                "test_top_half_mean_return": test.get("top_half_mean_return"),
                "test_bottom_half_mean_return": test.get("bottom_half_mean_return"),
                "test_top_minus_bottom": (
                    test.get("top_half_mean_return", np.nan)
                    - test.get("bottom_half_mean_return", np.nan)
                ),
            })
        return row

    @staticmethod
    def _seed_stability(state_paths: dict[tuple, pd.DataFrame]) -> pd.DataFrame:
        from sklearn.metrics import adjusted_rand_score
        rows = []
        groups: dict[tuple, list[tuple[int, pd.DataFrame]]] = {}
        for (method, n_components, history_days, seed), states in state_paths.items():
            groups.setdefault((method, n_components, history_days), []).append((seed, states))
        for (method, n_components, history_days), items in groups.items():
            scores = []
            for left_pos in range(len(items)):
                for right_pos in range(left_pos + 1, len(items)):
                    left_seed, left = items[left_pos]
                    right_seed, right = items[right_pos]
                    merged = left.merge(right, on="trade_date", suffixes=("_left", "_right"))
                    if len(merged) >= 20:
                        scores.append(adjusted_rand_score(
                            merged["predicted_state_left"],
                            merged["predicted_state_right"],
                        ))
            rows.append({
                "method": method,
                "n_components": n_components,
                "history_days": history_days,
                "seed_pair_count": len(scores),
                "mean_adjusted_rand": float(np.mean(scores)) if scores else np.nan,
                "min_adjusted_rand": float(np.min(scores)) if scores else np.nan,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _report(summary, grid, stability) -> str:
        successful = grid.loc[grid["status"].eq("SUCCESS")].copy()
        top = successful.head(15)
        return "\n".join([
            "# Timing Regime Grid Stability",
            "",
            f"- Dataset: `{summary['dataset_path']}`",
            f"- Label: `{summary['label_column']}`",
            f"- Successful combinations: `{summary['successful_combinations']}/{summary['total_combinations']}`",
            "",
            "## Top Combinations",
            top.to_markdown(index=False, floatfmt=".4f") if not top.empty else "No successful combinations.",
            "",
            "## Seed Stability",
            stability.sort_values("mean_adjusted_rand", ascending=False).to_markdown(index=False, floatfmt=".4f") if not stability.empty else "No stability rows.",
            "",
        ]) + "\n"


def _align_state_order(means: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    if reference is None:
        # Low first-feature mean is treated as the more defensive state. The
        # exact economic label is assigned later from realized state profiles.
        return np.argsort(means[:, 0])
    from scipy.optimize import linear_sum_assignment
    _, assigned = linear_sum_assignment(((reference[:, None] - means[None]) ** 2).mean(axis=2))
    return assigned


def _hmm_filtered_probabilities(model, x: np.ndarray) -> np.ndarray:
    means = np.asarray(model.means_, dtype=float)
    covars = np.asarray(getattr(model, "_covars_", model.covars_), dtype=float)
    if covars.ndim == 3:
        var = np.diagonal(covars, axis1=1, axis2=2)
    else:
        var = covars
    var = np.maximum(var, 1e-8)
    ll = -0.5 * (
        x.shape[1] * math.log(2 * math.pi)
        + np.log(var).sum(axis=1)[None, :]
        + (((x[:, None, :] - means[None, :, :]) ** 2) / var[None, :, :]).sum(axis=2)
    )
    likelihood = np.exp(ll - ll.max(axis=1, keepdims=True))
    out = np.empty((len(x), model.n_components), dtype=float)
    prior = np.asarray(model.startprob_, dtype=float).copy()
    for i in range(len(x)):
        if i:
            prior = out[i - 1] @ np.asarray(model.transmat_, dtype=float)
        posterior = prior * likelihood[i]
        total = posterior.sum()
        out[i] = posterior / total if total else np.repeat(1.0 / model.n_components, model.n_components)
    return out


def _safe_corr(left: pd.Series, right: pd.Series, *, method: str) -> float:
    sample = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(sample) < 2 or sample["left"].nunique() < 2 or sample["right"].nunique() < 2:
        return np.nan
    value = sample["left"].corr(sample["right"], method=method)
    return float(value) if np.isfinite(value) else np.nan


def _json_default(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
