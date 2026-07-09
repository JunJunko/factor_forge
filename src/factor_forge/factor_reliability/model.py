from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .dataset import ReliabilitySplitConfig, split_dataset, target_column


MODEL_NAMES = ["historical_mean", "ridge", "lightgbm_shallow"]


@dataclass(frozen=True)
class ReliabilityModelConfig:
    horizons: tuple[int, ...] = (5, 10, 20)
    historical_mean_window: int = 20
    random_state: int = 42


def run_reliability_regression(
    dataset: pd.DataFrame,
    features: list[str],
    *,
    split_config: ReliabilitySplitConfig | None = None,
    model_config: ReliabilityModelConfig | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = model_config or ReliabilityModelConfig()
    prediction_frames = []
    metric_frames = []
    bucket_frames = []
    calibration_frames = []
    stability_frames = []
    importance_frames = []
    for horizon in cfg.horizons:
        target = target_column(horizon)
        splits = split_dataset(dataset, target=target, features=features, config=split_config)
        predictions, importances = _fit_predict_horizon(splits, features, target, horizon, cfg)
        prediction_frames.append(predictions)
        if not importances.empty:
            importance_frames.append(importances)
        metric_frames.append(model_metrics(predictions, horizon))
        bucket_frames.append(bucket_test(predictions, horizon))
        calibration_frames.append(calibration_table(predictions, horizon))
        stability_frames.append(stability_by_year(predictions, horizon))
    return {
        "predictions": pd.concat(prediction_frames, ignore_index=True),
        "metrics": pd.concat(metric_frames, ignore_index=True),
        "bucket_test": pd.concat(bucket_frames, ignore_index=True),
        "calibration": pd.concat(calibration_frames, ignore_index=True),
        "stability": pd.concat(stability_frames, ignore_index=True),
        "feature_importance": pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame(),
    }


def _fit_predict_horizon(
    splits: dict[str, pd.DataFrame],
    features: list[str],
    target: str,
    horizon: int,
    cfg: ReliabilityModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = splits["train"].copy()
    valid = splits["valid"].copy()
    train_fit = pd.concat([train, valid], ignore_index=True)
    medians = train[features].median(numeric_only=True).reindex(features).fillna(0.0)
    frames = []
    importances = []
    for model_name in MODEL_NAMES:
        model = _fit_model(model_name, train, train_fit, features, target, medians, cfg)
        if model_name == "lightgbm_shallow" and model is None:
            continue
        for sample, frame in splits.items():
            pred = _predict_model(model_name, model, frame, features, target, medians, cfg)
            pred["sample"] = sample
            pred["model"] = model_name
            pred["horizon"] = horizon
            frames.append(pred)
        imp = _feature_importance(model_name, model, features, horizon)
        if not imp.empty:
            importances.append(imp)
    return pd.concat(frames, ignore_index=True), pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()


def _fit_model(
    model_name: str,
    train: pd.DataFrame,
    train_fit: pd.DataFrame,
    features: list[str],
    target: str,
    medians: pd.Series,
    cfg: ReliabilityModelConfig,
) -> Any:
    if model_name == "historical_mean":
        return None
    x = train_fit[features].fillna(medians)
    y = train_fit[target].astype(float)
    if model_name == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=cfg.random_state))
        model.fit(x, y)
        return model
    if model_name == "lightgbm_shallow":
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            return None
        model = LGBMRegressor(
            objective="regression",
            max_depth=3,
            num_leaves=8,
            learning_rate=0.05,
            n_estimators=240,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            random_state=cfg.random_state,
            verbosity=-1,
        )
        model.fit(x, y)
        return model
    raise ValueError(f"unknown model: {model_name}")


def _predict_model(
    model_name: str,
    model: Any,
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    medians: pd.Series,
    cfg: ReliabilityModelConfig,
) -> pd.DataFrame:
    out = frame[["date", "factor_name", target]].rename(columns={target: "actual_future_spread"}).copy()
    if model_name == "historical_mean":
        out["predicted_spread"] = frame["spread_20"] if "spread_20" in frame.columns else frame[target].expanding().mean().shift(1)
    else:
        out["predicted_spread"] = model.predict(frame[features].fillna(medians))
    return out


def _feature_importance(model_name: str, model: Any, features: list[str], horizon: int) -> pd.DataFrame:
    if model_name == "ridge" and model is not None:
        ridge = model.named_steps["ridge"]
        return pd.DataFrame(
            {"horizon": horizon, "model": model_name, "feature": features, "importance": np.abs(ridge.coef_)}
        ).sort_values("importance", ascending=False)
    if model_name == "lightgbm_shallow" and model is not None:
        return pd.DataFrame(
            {
                "horizon": horizon,
                "model": model_name,
                "feature": features,
                "importance": model.booster_.feature_importance(importance_type="gain"),
            }
        ).sort_values("importance", ascending=False)
    return pd.DataFrame()


def model_metrics(predictions: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows = []
    for (model, sample), group in predictions.groupby(["model", "sample"], sort=False):
        g = group.dropna(subset=["predicted_spread", "actual_future_spread"])
        rows.append(
            {
                "horizon": horizon,
                "model": model,
                "sample": sample,
                "rows": int(len(g)),
                "rank_ic": _safe_corr(g["predicted_spread"], g["actual_future_spread"], method="spearman"),
                "pearson": _safe_corr(g["predicted_spread"], g["actual_future_spread"], method="pearson"),
                "mae": float((g["predicted_spread"] - g["actual_future_spread"]).abs().mean()) if len(g) else np.nan,
                "rmse": float(np.sqrt(((g["predicted_spread"] - g["actual_future_spread"]) ** 2).mean())) if len(g) else np.nan,
                "q5_gt_q1": _q5_gt_q1(g),
            }
        )
    return pd.DataFrame(rows)


def bucket_test(predictions: pd.DataFrame, horizon: int, buckets: int = 5) -> pd.DataFrame:
    rows = []
    labels = [f"Q{i}" for i in range(1, buckets + 1)]
    for (model, sample), group in predictions.groupby(["model", "sample"], sort=False):
        g = group.dropna(subset=["predicted_spread", "actual_future_spread"]).copy()
        if len(g) < buckets:
            continue
        try:
            g["bucket"] = pd.qcut(g["predicted_spread"].rank(method="first"), buckets, labels=labels)
        except ValueError:
            continue
        for bucket, item in g.groupby("bucket", observed=True):
            rows.append(
                {
                    "horizon": horizon,
                    "model": model,
                    "sample": sample,
                    "bucket": str(bucket),
                    "mean_predicted_spread": float(item["predicted_spread"].mean()),
                    "actual_future_spread": float(item["actual_future_spread"].mean()),
                    "win_ratio": float(item["actual_future_spread"].gt(0.002).mean()),
                    "rows": int(len(item)),
                }
            )
    return pd.DataFrame(rows)


def calibration_table(predictions: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows = []
    for (model, sample), group in predictions.groupby(["model", "sample"], sort=False):
        g = group.dropna(subset=["predicted_spread", "actual_future_spread"]).copy()
        if g.empty:
            continue
        g["predicted_bucket"] = pd.cut(
            g["predicted_spread"],
            bins=[-np.inf, -0.02, 0.0, 0.01, 0.02, np.inf],
            labels=["<-2%", "-2%~0", "0~1%", "1%~2%", ">2%"],
        )
        for bucket, item in g.groupby("predicted_bucket", observed=True):
            rows.append(
                {
                    "horizon": horizon,
                    "model": model,
                    "sample": sample,
                    "predicted_bucket": str(bucket),
                    "mean_predicted_spread": float(item["predicted_spread"].mean()),
                    "actual_future_spread": float(item["actual_future_spread"].mean()),
                    "rows": int(len(item)),
                }
            )
    return pd.DataFrame(rows)


def stability_by_year(predictions: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frame = predictions.copy()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    rows = []
    for (model, sample, year), group in frame.groupby(["model", "sample", "year"], sort=False):
        rows.append(
            {
                "horizon": horizon,
                "model": model,
                "sample": sample,
                "year": int(year),
                "rows": int(len(group)),
                "rank_ic": _safe_corr(group["predicted_spread"], group["actual_future_spread"], method="spearman"),
                "mean_actual_spread": float(group["actual_future_spread"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return np.nan
    value = x[mask].corr(y[mask], method=method)
    return float(value) if pd.notna(value) else np.nan


def _q5_gt_q1(group: pd.DataFrame) -> bool | None:
    if len(group) < 5:
        return None
    try:
        bucket = pd.qcut(group["predicted_spread"].rank(method="first"), 5, labels=False) + 1
    except ValueError:
        return None
    means = group.assign(bucket=bucket).groupby("bucket", observed=True)["actual_future_spread"].mean()
    return bool(means.get(5, np.nan) > means.get(1, np.nan)) if 1 in means.index and 5 in means.index else None
