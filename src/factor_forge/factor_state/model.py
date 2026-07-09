from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import available_model_features
from .label import STATE_NAMES, FactorState


@dataclass(frozen=True)
class FactorStateModelConfig:
    train_start: str = "20180101"
    train_end: str = "20221231"
    valid_start: str = "20230101"
    valid_end: str = "20231231"
    test_start: str = "20240101"
    test_end: str = "20261231"
    max_features: int = 30
    probability_threshold: float = 0.5
    random_state: int = 42
    feature_candidates: tuple[str, ...] = field(default_factory=tuple)


def run_factor_state_model(
    labeled: pd.DataFrame,
    *,
    output_dir: str | Path,
    config: FactorStateModelConfig | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = config or FactorStateModelConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = _prepare_labeled(labeled)
    features = available_model_features(
        data,
        cfg.feature_candidates if cfg.feature_candidates else None,
        max_features=cfg.max_features,
    )
    if not features:
        raise ValueError("no eligible factor-state features found")
    splits = _time_splits(data, cfg)
    if splits["train"].empty or splits["test"].empty:
        splits = _fallback_chronological_splits(data)
    results: dict[str, pd.DataFrame] = {"feature_list": pd.DataFrame({"feature": features})}
    predictions = []
    metric_rows = []
    for model_name in ["logistic_regression", "lightgbm_shallow"]:
        try:
            model_result = _fit_predict_model(model_name, splits, features, cfg)
        except Exception:
            model_result = None
        if model_result is None:
            continue
        pred = model_result["predictions"]
        predictions.append(pred)
        metric_rows.extend(_accuracy_rows(pred, model_name))
        results[f"{model_name}_feature_importance"] = model_result["feature_importance"]
    if not predictions:
        raise ValueError("no state model could be trained; check label class diversity")

    prediction_frame = pd.concat(predictions, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    confusion = _confusion_matrix(prediction_frame)
    calibration = _calibration_table(prediction_frame)
    transition = _state_transition_matrix(data)
    warning = _early_warning_metrics(prediction_frame, cfg.probability_threshold)

    results.update(
        {
            "predictions": prediction_frame,
            "metrics": metrics,
            "confusion_matrix": confusion,
            "calibration": calibration,
            "transition_matrix": transition,
            "early_warning": warning,
        }
    )
    _write_outputs(output, results)
    return results


def _prepare_labeled(labeled: pd.DataFrame) -> pd.DataFrame:
    data = labeled.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.dropna(subset=["state"]).copy()
    data["state"] = data["state"].astype(int)
    data["state_name"] = data["state"].map(STATE_NAMES)
    return data.sort_values(["date", "factor_name"]).reset_index(drop=True)


def _time_splits(data: pd.DataFrame, cfg: FactorStateModelConfig) -> dict[str, pd.DataFrame]:
    date = data["date"]
    return {
        "train": data.loc[date.between(pd.Timestamp(cfg.train_start), pd.Timestamp(cfg.train_end))].copy(),
        "valid": data.loc[date.between(pd.Timestamp(cfg.valid_start), pd.Timestamp(cfg.valid_end))].copy(),
        "test": data.loc[date.between(pd.Timestamp(cfg.test_start), pd.Timestamp(cfg.test_end))].copy(),
    }


def _fallback_chronological_splits(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dates = pd.Series(data["date"].drop_duplicates().sort_values().to_numpy())
    if len(dates) < 30:
        raise ValueError("not enough dates for chronological factor-state training")
    train_end = dates.iloc[max(0, int(len(dates) * 0.6) - 1)]
    valid_end = dates.iloc[max(0, int(len(dates) * 0.8) - 1)]
    return {
        "train": data.loc[data["date"].le(train_end)].copy(),
        "valid": data.loc[data["date"].gt(train_end) & data["date"].le(valid_end)].copy(),
        "test": data.loc[data["date"].gt(valid_end)].copy(),
    }


def _fit_predict_model(
    model_name: str,
    splits: dict[str, pd.DataFrame],
    features: list[str],
    cfg: FactorStateModelConfig,
) -> dict[str, pd.DataFrame] | None:
    train = splits["train"].copy()
    valid = splits["valid"].copy()
    test = splits["test"].copy()
    train_fit = pd.concat([train, valid], ignore_index=True) if not valid.empty else train
    classes = sorted(train_fit["state"].dropna().astype(int).unique().tolist())
    if len(classes) < 2:
        return None
    medians = train_fit[features].median(numeric_only=True).reindex(features).fillna(0.0)
    x_train = train_fit[features].replace([np.inf, -np.inf], np.nan).fillna(medians)
    y_train = train_fit["state"].astype(int)

    class_labels = None
    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cfg.random_state),
        )
        model.fit(x_train, y_train)
        class_labels = model.classes_
        importance = _logistic_importance(model, features)
    elif model_name == "lightgbm_shallow":
        try:
            from lightgbm import LGBMClassifier
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            return None
        encoder = LabelEncoder()
        y_fit = encoder.fit_transform(y_train)
        class_labels = encoder.classes_
        model = LGBMClassifier(
            max_depth=3,
            num_leaves=8,
            n_estimators=120,
            learning_rate=0.05,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=cfg.random_state,
            class_weight="balanced",
            verbosity=-1,
        )
        model.fit(x_train, y_fit)
        importance = pd.DataFrame(
            {
                "model": model_name,
                "feature": features,
                "importance": model.booster_.feature_importance(importance_type="gain"),
            }
        ).sort_values("importance", ascending=False)
    else:
        raise ValueError(f"unknown model: {model_name}")

    preds = []
    for sample, frame in splits.items():
        if frame.empty:
            continue
        x = frame[features].replace([np.inf, -np.inf], np.nan).fillna(medians)
        proba = model.predict_proba(x)
        pred = frame[["date", "factor_name", "state", "state_name"]].copy()
        pred["sample"] = sample
        pred["model"] = model_name
        for idx, klass in enumerate(class_labels):
            pred[f"p_{STATE_NAMES[int(klass)].lower()}"] = proba[:, idx]
        for state in FactorState:
            col = f"p_{state.name.lower()}"
            if col not in pred.columns:
                pred[col] = 0.0
        prob_cols = [f"p_{state.name.lower()}" for state in FactorState]
        pred["predicted_state"] = pred[prob_cols].to_numpy().argmax(axis=1)
        pred["predicted_state_name"] = pred["predicted_state"].map(STATE_NAMES)
        preds.append(pred)
    return {"predictions": pd.concat(preds, ignore_index=True), "feature_importance": importance}


def _logistic_importance(model: Any, features: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["logisticregression"]
    coef = np.abs(estimator.coef_).mean(axis=0)
    return pd.DataFrame({"model": "logistic_regression", "feature": features, "importance": coef}).sort_values(
        "importance", ascending=False
    )


def _accuracy_rows(pred: pd.DataFrame, model_name: str) -> list[dict[str, Any]]:
    rows = []
    for sample, group in pred.groupby("sample", sort=False):
        rows.append(
            {
                "model": model_name,
                "sample": sample,
                "rows": int(len(group)),
                "accuracy": float((group["state"] == group["predicted_state"]).mean()),
                "broken_recall": _recall(group, FactorState.Broken.value),
                "weakening_recall": _recall(group, FactorState.Weakening.value),
            }
        )
    return rows


def _recall(group: pd.DataFrame, state: int) -> float:
    actual = group["state"].eq(state)
    if not actual.any():
        return np.nan
    return float(group.loc[actual, "predicted_state"].eq(state).mean())


def _confusion_matrix(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, sample), group in pred.groupby(["model", "sample"], sort=False):
        matrix = pd.crosstab(group["state_name"], group["predicted_state_name"], normalize="index")
        for actual in STATE_NAMES.values():
            for predicted in STATE_NAMES.values():
                rows.append(
                    {
                        "model": model,
                        "sample": sample,
                        "actual_state": actual,
                        "predicted_state": predicted,
                        "ratio": float(matrix.loc[actual, predicted]) if actual in matrix.index and predicted in matrix.columns else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _calibration_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, model_group in pred.groupby("model", sort=False):
        for state in FactorState:
            col = f"p_{state.name.lower()}"
            if col not in model_group:
                continue
            values = model_group[[col, "state"]].copy()
            try:
                values["bucket"] = pd.qcut(values[col].rank(method="first"), 5, labels=False) + 1
            except ValueError:
                continue
            for bucket, group in values.groupby("bucket", observed=True):
                rows.append(
                    {
                        "model": model,
                        "state": state.name,
                        "probability_bucket": int(bucket),
                        "mean_probability": float(group[col].mean()),
                        "realized_frequency": float(group["state"].eq(state.value).mean()),
                        "rows": int(len(group)),
                    }
                )
    return pd.DataFrame(rows)


def _state_transition_matrix(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor_name, group in data.sort_values("date").groupby("factor_name", sort=False):
        g = group[["state", "state_name"]].copy()
        g["next_state"] = g["state"].shift(-1)
        g["next_state_name"] = g["next_state"].map(STATE_NAMES)
        matrix = pd.crosstab(g["state_name"], g["next_state_name"], normalize="index")
        for current in STATE_NAMES.values():
            for next_state in STATE_NAMES.values():
                rows.append(
                    {
                        "factor_name": factor_name,
                        "current_state": current,
                        "next_state": next_state,
                        "transition_probability": float(matrix.loc[current, next_state])
                        if current in matrix.index and next_state in matrix.columns
                        else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _early_warning_metrics(pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for (model, factor_name, sample), group in pred.sort_values("date").groupby(["model", "factor_name", "sample"], sort=False):
        broken_dates = group.loc[group["state"].eq(FactorState.Broken.value), "date"].tolist()
        for horizon in [30, 60]:
            hits = 0
            eligible = 0
            for date in broken_dates:
                window = group.loc[group["date"].lt(date) & group["date"].ge(date - pd.Timedelta(days=horizon * 2))]
                window = window.tail(horizon)
                if window.empty:
                    continue
                eligible += 1
                risk_prob = window["p_weakening"].fillna(0.0) + window["p_broken"].fillna(0.0)
                if risk_prob.max() >= threshold or window["predicted_state"].isin([FactorState.Weakening.value, FactorState.Broken.value]).any():
                    hits += 1
            rows.append(
                {
                    "model": model,
                    "factor_name": factor_name,
                    "sample": sample,
                    "warning_horizon_days": horizon,
                    "broken_events": int(len(broken_dates)),
                    "eligible_events": int(eligible),
                    "warning_hit_rate": float(hits / eligible) if eligible else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _write_outputs(output: Path, results: dict[str, pd.DataFrame]) -> None:
    for name, frame in results.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
