from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .regime import TimingRegimeConfig, TimingRegimeRunner


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PositionMappingConfig(StrictModel):
    quantiles: list[float] = Field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    positions: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    max_daily_position_change: float = Field(default=0.25, ge=0.0, le=1.0)
    transaction_cost_bps: float = Field(default=3.0, ge=0.0)

    @model_validator(mode="after")
    def validate_mapping(self):
        if sorted(self.quantiles) != self.quantiles:
            raise ValueError("quantiles must be sorted ascending")
        if any(item <= 0 or item >= 1 for item in self.quantiles):
            raise ValueError("quantiles must be strictly between 0 and 1")
        if len(self.positions) != len(self.quantiles) + 1:
            raise ValueError("positions length must equal len(quantiles) + 1")
        if any(item < 0 or item > 1 for item in self.positions):
            raise ValueError("positions must be in [0, 1]")
        return self


class PositionModelConfig(StrictModel):
    model_type: Literal["ridge", "elasticnet"] = "ridge"
    alpha: float = Field(default=1.0, ge=0.0)
    l1_ratio: float = Field(default=0.2, ge=0.0, le=1.0)


class TimingPositionModelConfig(StrictModel):
    version: int = 1
    name: str = "timing_position_model_v1"
    dataset_path: Path
    stable_factors_path: Path
    feature_names_path: Path | None = None
    output_root: Path = Path("artifacts/timing_position_models")
    start_date: str = "2023-04-18"
    label_column: str = "label_10d_excess_return"
    horizon_days: int = Field(default=10, ge=1, le=120)
    train_end: str = "2025-06-30"
    test_start: str = "2025-07-01"
    model: PositionModelConfig = Field(default_factory=PositionModelConfig)
    mapping: PositionMappingConfig = Field(default_factory=PositionMappingConfig)
    regime_method: Literal["hmm", "gmm"] = "hmm"
    n_components: int = Field(default=3, ge=2, le=5)
    history_days: int = Field(default=252, ge=120, le=1260)
    random_state: int = 42
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

    @model_validator(mode="after")
    def validate_dates(self):
        if self.train_end >= self.test_start:
            raise ValueError("train_end must be before test_start")
        return self


def load_timing_position_model_config(path: str | Path) -> TimingPositionModelConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return TimingPositionModelConfig.model_validate(yaml.safe_load(handle) or {})


class TimingPositionModelRunner:
    def run(self, config_path: str | Path) -> dict:
        config_path = Path(config_path)
        cfg = load_timing_position_model_config(config_path)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + hashlib.sha256(config_path.read_bytes()).hexdigest()[:8]
        output = cfg.output_root / f"{cfg.name}_{run_id}"
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.yaml").write_bytes(config_path.read_bytes())

        dataset, selected_factors, states = self._load_inputs(cfg)
        merged, model_features = self._build_model_frame(dataset, states, selected_factors, cfg)
        predictions, coefficients, thresholds = self._fit_predict(merged, model_features, cfg)
        daily = self._backtest_positions(predictions, cfg)
        metrics = {
            "train": self._sample_metrics(predictions, daily, "train"),
            "test": self._sample_metrics(predictions, daily, "test"),
        }

        predictions.to_csv(output / "timing_position_predictions.csv", index=False, encoding="utf-8-sig")
        daily.to_csv(output / "timing_position_daily.csv", index=False, encoding="utf-8-sig")
        coefficients.to_csv(output / "timing_position_coefficients.csv", index=False, encoding="utf-8-sig")
        states.to_csv(output / "regime_daily_states.csv", index=False, encoding="utf-8-sig")

        summary = {
            "status": "SUCCESS",
            "run_dir": str(output),
            "dataset_path": str(cfg.dataset_path),
            "stable_factors_path": str(cfg.stable_factors_path),
            "label_column": cfg.label_column,
            "horizon_days": cfg.horizon_days,
            "train_end": cfg.train_end,
            "test_start": cfg.test_start,
            "selected_factor_count": int(len(selected_factors)),
            "expanded_feature_count": int(len(model_features)),
            "regime": {
                "method": cfg.regime_method,
                "n_components": cfg.n_components,
                "history_days": cfg.history_days,
                "random_state": cfg.random_state,
            },
            "position_thresholds": {str(key): float(value) for key, value in thresholds.items()},
            "metrics": metrics,
        }
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        (output / "report.md").write_text(self._report(summary, coefficients), encoding="utf-8")
        return summary

    def _load_inputs(
        self,
        cfg: TimingPositionModelConfig,
    ) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
        helper = TimingRegimeRunner()
        regime_cfg = self._regime_config(cfg)
        dataset, _ = helper._load_dataset(regime_cfg)
        if not cfg.stable_factors_path.exists():
            raise FileNotFoundError(f"stable_factors_path does not exist: {cfg.stable_factors_path}")
        stable = pd.read_csv(cfg.stable_factors_path)
        if "factor" not in stable.columns:
            raise ValueError("stable_factors_path must contain a factor column")
        selected_factors = [factor for factor in stable["factor"].dropna().astype(str).tolist() if factor in dataset.columns]
        missing = sorted(set(stable["factor"].dropna().astype(str)) - set(selected_factors))
        if len(selected_factors) < 5:
            raise ValueError(f"too few selected factors found in dataset; missing={missing}")
        regime_features = [column for column in cfg.regime_features if column in dataset.columns]
        if len(regime_features) < 3:
            raise ValueError(f"too few regime features found: {regime_features}")
        states = helper._walk_forward_regimes(dataset, regime_features, regime_cfg)
        return dataset, selected_factors, states

    @staticmethod
    def _regime_config(cfg: TimingPositionModelConfig) -> TimingRegimeConfig:
        return TimingRegimeConfig.model_validate({
            "name": cfg.name,
            "dataset_path": cfg.dataset_path,
            "feature_names_path": cfg.feature_names_path,
            "output_root": cfg.output_root,
            "start_date": cfg.start_date,
            "label_column": cfg.label_column,
            "label_columns": [cfg.label_column],
            "regime_features": cfg.regime_features,
            "regime": {
                "method": cfg.regime_method,
                "n_components": cfg.n_components,
                "history_days": cfg.history_days,
                "random_state": cfg.random_state,
                "covariance_type": "diag",
                "refit_frequency": "monthly",
                "zscore_window": 252,
                "zscore_min_periods": 60,
                "max_iterations": 300,
                "tolerance": 0.0001,
                "min_covar": 0.0001,
            },
            "diagnostics": {
                "min_coverage": 0.55,
                "quantiles": 5,
                "top_curve_factors": 30,
                "exclude_patterns": ["_low_", "_high_", "label_", "state_probability_"],
            },
            "interaction_model": {"enabled": False},
        })

    @staticmethod
    def _build_model_frame(
        dataset: pd.DataFrame,
        states: pd.DataFrame,
        selected_factors: list[str],
        cfg: TimingPositionModelConfig,
    ) -> tuple[pd.DataFrame, list[str]]:
        base_columns = ["trade_date", "index_close", cfg.label_column, *selected_factors]
        merged = states.merge(dataset[base_columns], on="trade_date", how="left")
        state_cols = [column for column in states.columns if column.startswith("state_probability_")]
        interactions = {
            f"{factor}__x__{state_col}": merged[factor] * merged[state_col]
            for factor in selected_factors
            for state_col in state_cols
        }
        if interactions:
            merged = pd.concat([merged, pd.DataFrame(interactions, index=merged.index)], axis=1)
        model_features = selected_factors + state_cols + list(interactions)
        merged = merged.replace([np.inf, -np.inf], np.nan).sort_values("trade_date").reset_index(drop=True)
        return merged, model_features

    @staticmethod
    def _fit_predict(
        merged: pd.DataFrame,
        model_features: list[str],
        cfg: TimingPositionModelConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[float, float]]:
        train_end = pd.Timestamp(cfg.train_end)
        test_start = pd.Timestamp(cfg.test_start)
        feature_usable = merged[model_features + ["index_close"]].notna().all(axis=1)
        label_usable = merged[cfg.label_column].notna()
        train = merged["trade_date"].le(train_end) & feature_usable & label_usable
        train_dates = merged.loc[train, "trade_date"].drop_duplicates().sort_values().tolist()
        if len(train_dates) > cfg.horizon_days:
            purge_start = train_dates[-cfg.horizon_days]
            train &= merged["trade_date"].lt(purge_start)
        test = merged["trade_date"].ge(test_start) & feature_usable
        if train.sum() < 80 or test.sum() < 20:
            raise ValueError(f"not enough train/test samples: train={int(train.sum())}, test={int(test.sum())}")

        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if cfg.model.model_type == "ridge":
            from sklearn.linear_model import Ridge
            estimator = Ridge(alpha=cfg.model.alpha)
        else:
            from sklearn.linear_model import ElasticNet
            estimator = ElasticNet(alpha=cfg.model.alpha, l1_ratio=cfg.model.l1_ratio, max_iter=10000)
        model = make_pipeline(StandardScaler(), estimator)
        model.fit(merged.loc[train, model_features], merged.loc[train, cfg.label_column])

        output = merged.loc[train | test, [
            "trade_date",
            "index_close",
            cfg.label_column,
            "predicted_state",
            "state_name",
        ]].copy()
        output = output.rename(columns={cfg.label_column: "forward_return"})
        output["sample"] = np.where(output["trade_date"].le(train_end), "train", "test")
        output["prediction"] = model.predict(merged.loc[output.index, model_features])
        train_prediction = output.loc[output["sample"].eq("train"), "prediction"]
        thresholds = train_prediction.quantile(cfg.mapping.quantiles).to_dict()
        output["raw_position"] = map_predictions_to_positions(output["prediction"], thresholds, cfg.mapping.positions)
        output["target_position"] = smooth_positions(output["raw_position"], cfg.mapping.max_daily_position_change)

        estimator_step = model.steps[-1][1]
        coefficients = pd.DataFrame({
            "feature": model_features,
            "coefficient": getattr(estimator_step, "coef_", np.repeat(np.nan, len(model_features))),
        }).sort_values("coefficient", key=lambda item: item.abs(), ascending=False)
        return output, coefficients, thresholds

    @staticmethod
    def _backtest_positions(predictions: pd.DataFrame, cfg: TimingPositionModelConfig) -> pd.DataFrame:
        daily = predictions.sort_values("trade_date").reset_index(drop=True).copy()
        daily["index_return"] = daily["index_close"].pct_change().fillna(0.0)
        daily["executed_position"] = daily["target_position"].shift(1).fillna(0.0).clip(0.0, 1.0)
        daily["turnover"] = daily["executed_position"].diff().abs().fillna(daily["executed_position"].abs())
        daily["transaction_cost"] = daily["turnover"] * cfg.mapping.transaction_cost_bps / 10000.0
        daily["strategy_return"] = daily["executed_position"] * daily["index_return"] - daily["transaction_cost"]
        daily["benchmark_return"] = daily["index_return"]
        daily["strategy_nav"] = (1.0 + daily["strategy_return"]).cumprod()
        daily["benchmark_nav"] = (1.0 + daily["benchmark_return"]).cumprod()
        return daily

    @staticmethod
    def _sample_metrics(predictions: pd.DataFrame, daily: pd.DataFrame, sample: str) -> dict:
        pred = predictions.loc[predictions["sample"].eq(sample)].copy()
        day = daily.loc[daily["sample"].eq(sample)].copy()
        rank_ic = _safe_corr(pred["prediction"], pred["forward_return"], method="spearman")
        top_half = pred["prediction"].ge(pred["prediction"].median())
        return {
            "rows": int(len(pred)),
            "rank_ic": float(rank_ic) if np.isfinite(rank_ic) else np.nan,
            "top_half_mean_forward_return": float(pred.loc[top_half, "forward_return"].mean()) if len(pred) else np.nan,
            "bottom_half_mean_forward_return": float(pred.loc[~top_half, "forward_return"].mean()) if len(pred) else np.nan,
            **performance_metrics(day["strategy_return"], day["benchmark_return"], day["turnover"], day["executed_position"]),
        }

    @staticmethod
    def _report(summary: dict, coefficients: pd.DataFrame) -> str:
        top_coefficients = coefficients.head(20).copy()
        train = summary["metrics"]["train"]
        test = summary["metrics"]["test"]
        sections = [
            "# Timing Position Model",
            "",
            f"- Dataset: `{summary['dataset_path']}`",
            f"- Stable factors: `{summary['selected_factor_count']}`",
            f"- Expanded features: `{summary['expanded_feature_count']}`",
            f"- Label: `{summary['label_column']}`",
            f"- Split: train <= `{summary['train_end']}`, test >= `{summary['test_start']}`",
            "",
            "## Metrics",
            pd.DataFrame([{"sample": "train", **train}, {"sample": "test", **test}]).to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Top Coefficients",
            top_coefficients.to_markdown(index=False, floatfmt=".6f"),
        ]
        return "\n".join(sections) + "\n"


def map_predictions_to_positions(
    prediction: pd.Series,
    thresholds: dict[float, float],
    positions: list[float],
) -> pd.Series:
    ordered_thresholds = [thresholds[key] for key in sorted(thresholds)]
    buckets = np.searchsorted(ordered_thresholds, prediction.to_numpy(float), side="right")
    return pd.Series([positions[int(bucket)] for bucket in buckets], index=prediction.index, dtype=float)


def smooth_positions(position: pd.Series, max_daily_change: float) -> pd.Series:
    values = position.to_numpy(float)
    if len(values) == 0 or max_daily_change >= 1:
        return pd.Series(values, index=position.index, dtype=float)
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for idx in range(1, len(values)):
        delta = np.clip(values[idx] - smoothed[idx - 1], -max_daily_change, max_daily_change)
        smoothed[idx] = smoothed[idx - 1] + delta
    return pd.Series(smoothed, index=position.index, dtype=float).clip(0.0, 1.0)


def performance_metrics(
    strategy_return: pd.Series,
    benchmark_return: pd.Series,
    turnover: pd.Series,
    position: pd.Series,
) -> dict:
    returns = pd.to_numeric(strategy_return, errors="coerce").fillna(0.0)
    benchmark = pd.to_numeric(benchmark_return, errors="coerce").fillna(0.0)
    nav = (1.0 + returns).cumprod()
    benchmark_nav = (1.0 + benchmark).cumprod()
    days = len(returns)
    if days == 0:
        return {
            "strategy_total_return": np.nan,
            "benchmark_total_return": np.nan,
            "excess_total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "benchmark_annualized_return": np.nan,
            "benchmark_annualized_volatility": np.nan,
            "benchmark_sharpe": np.nan,
            "benchmark_max_drawdown": np.nan,
            "average_position": np.nan,
            "average_daily_turnover": np.nan,
        }
    annual = 252.0 / days
    total_return = float(nav.iloc[-1] - 1.0)
    benchmark_total = float(benchmark_nav.iloc[-1] - 1.0)
    vol = float(returns.std(ddof=1) * np.sqrt(252.0)) if days > 1 else np.nan
    benchmark_vol = float(benchmark.std(ddof=1) * np.sqrt(252.0)) if days > 1 else np.nan
    annualized = float(nav.iloc[-1] ** annual - 1.0) if nav.iloc[-1] > 0 else np.nan
    benchmark_annualized = float(benchmark_nav.iloc[-1] ** annual - 1.0) if benchmark_nav.iloc[-1] > 0 else np.nan
    return {
        "strategy_total_return": total_return,
        "benchmark_total_return": benchmark_total,
        "excess_total_return": total_return - benchmark_total,
        "annualized_return": annualized,
        "annualized_volatility": vol,
        "sharpe": float(annualized / vol) if np.isfinite(vol) and vol > 0 else np.nan,
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()),
        "benchmark_annualized_return": benchmark_annualized,
        "benchmark_annualized_volatility": benchmark_vol,
        "benchmark_sharpe": (
            float(benchmark_annualized / benchmark_vol)
            if np.isfinite(benchmark_annualized) and np.isfinite(benchmark_vol) and benchmark_vol > 0
            else np.nan
        ),
        "benchmark_max_drawdown": float((benchmark_nav / benchmark_nav.cummax() - 1.0).min()),
        "average_position": float(pd.to_numeric(position, errors="coerce").mean()),
        "average_daily_turnover": float(pd.to_numeric(turnover, errors="coerce").mean()),
    }


def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    sample = pd.concat([left, right], axis=1).dropna()
    if len(sample) < 3 or sample.iloc[:, 0].nunique() <= 1 or sample.iloc[:, 1].nunique() <= 1:
        return np.nan
    return float(sample.iloc[:, 0].corr(sample.iloc[:, 1], method=method))


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
