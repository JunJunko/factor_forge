from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, mean_pinball_loss

from factor_forge.research.concept_etf_exit_ml import (
    EXIT_FEATURES,
    ExitMLFold,
    ExitMLRules,
    build_exit_folds,
)


def fit_tail_walk_forward(
    states: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    start: str,
    end: str,
    feature_columns: list[str] | None = None,
    rules: ExitMLRules = ExitMLRules(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[ExitMLFold]]:
    features = list(feature_columns or EXIT_FEATURES)
    missing = sorted(set(features) - set(states.columns))
    if missing:
        raise ValueError(f"missing tail-exit features: {missing}")
    calendar = sorted(panel.loc[
        panel["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end)), "trade_date"
    ].unique())
    folds = build_exit_folds(calendar, rules=rules)
    prediction_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    pinball_scorer = make_scorer(
        mean_pinball_loss, greater_is_better=False, alpha=rules.tail_quantile,
    )
    for fold in folds:
        train = states.loc[
            states["state_date"].between(fold.train_start, fold.train_end)
            & states["tail_label_available_date"].lt(fold.valid_start)
        ].copy()
        valid = states.loc[
            states["state_date"].between(fold.valid_start, fold.valid_end)
            & states["tail_label_available_date"].lt(fold.test_start)
        ].copy()
        if len(train) < rules.minimum_train_rows or len(valid) < rules.minimum_validation_rows:
            continue
        if train["tail_label_available_date"].ge(fold.valid_start).any():
            raise ValueError(f"fold {fold.fold} has immature tail training labels")
        if valid["tail_label_available_date"].ge(fold.test_start).any():
            raise ValueError(f"fold {fold.fold} has immature tail validation labels")
        model = _tail_regressor(rules.seed + fold.fold, rules.tail_quantile)
        model.fit(train[features], train["tail_worst_open_return"])
        placebo_train = train.copy()
        rng = np.random.default_rng(rules.seed + 10_000 + fold.fold)
        placebo_train["tail_worst_open_return"] = rng.permutation(
            placebo_train["tail_worst_open_return"].to_numpy()
        )
        placebo = _tail_regressor(rules.seed + 20_000 + fold.fold, rules.tail_quantile)
        placebo.fit(placebo_train[features], placebo_train["tail_worst_open_return"])

        test = panel.loc[panel["trade_date"].between(fold.test_start, fold.test_end)].copy()
        age_parts = []
        for age in range(1, rules.holding_days):
            part = test.copy()
            part["holding_age"] = age
            part["days_remaining"] = rules.holding_days - age
            age_parts.append(part)
        grid = pd.concat(age_parts, ignore_index=True)
        output = grid[["trade_date", "ts_code", "holding_age", "days_remaining"]].rename(
            columns={"trade_date": "state_date"}
        )
        output["predicted_tail_return"] = model.predict(grid[features])
        output["placebo_tail_return"] = placebo.predict(grid[features])
        output["fold"] = fold.fold
        prediction_parts.append(output)

        permutation = permutation_importance(
            model, valid[features], valid["tail_worst_open_return"],
            scoring=pinball_scorer, n_repeats=5,
            random_state=rules.seed + 30_000 + fold.fold,
        )
        importance_parts.append(pd.DataFrame({
            "fold": fold.fold,
            "feature": features,
            "importance": np.clip(permutation.importances_mean, 0, None),
        }))
        audit_rows.append({
            "fold": fold.fold,
            "train_start": fold.train_start, "train_end": fold.train_end,
            "valid_start": fold.valid_start, "valid_end": fold.valid_end,
            "test_start": fold.test_start, "test_end": fold.test_end,
            "train_rows": len(train), "valid_rows": len(valid), "test_grid_rows": len(grid),
            "train_label_available_max": train["tail_label_available_date"].max(),
            "valid_label_available_max": valid["tail_label_available_date"].max(),
            "iterations": model.n_iter_,
        })
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    importance = pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame()
    return predictions, importance, pd.DataFrame(audit_rows), folds


def evaluate_tail_predictions(
    states: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    loss_threshold: float,
) -> tuple[dict, pd.DataFrame]:
    actual = states[[
        "state_date", "ts_code", "holding_age", "tail_worst_open_return", "tail_event",
    ]]
    joined = actual.merge(
        predictions, on=["state_date", "ts_code", "holding_age"],
        how="inner", validate="many_to_one",
    )
    joined["predicted_tail_event"] = joined["predicted_tail_return"].le(loss_threshold)
    joined["placebo_tail_event"] = joined["placebo_tail_return"].le(loss_threshold)
    actual_event = joined["tail_event"].astype(bool)
    predicted_event = joined["predicted_tail_event"]
    placebo_event = joined["placebo_tail_event"]
    audit = {
        "oof_held_state_rows": len(joined),
        "tail_event_base_rate": float(actual_event.mean()),
        "predicted_event_rate": float(predicted_event.mean()),
        "tail_precision": _safe_ratio((predicted_event & actual_event).sum(), predicted_event.sum()),
        "tail_recall": _safe_ratio((predicted_event & actual_event).sum(), actual_event.sum()),
        "placebo_precision": _safe_ratio((placebo_event & actual_event).sum(), placebo_event.sum()),
        "placebo_recall": _safe_ratio((placebo_event & actual_event).sum(), actual_event.sum()),
        "tail_prediction_correlation": float(
            joined["predicted_tail_return"].corr(joined["tail_worst_open_return"])
        ),
        "placebo_prediction_correlation": float(
            joined["placebo_tail_return"].corr(joined["tail_worst_open_return"])
        ),
    }
    return audit, joined


def summarize_tail_importance(importance: pd.DataFrame) -> pd.DataFrame:
    result = importance.groupby("feature", as_index=False).agg(
        mean_importance=("importance", "mean"),
        median_importance=("importance", "median"),
        positive_folds=("importance", lambda values: int(values.gt(0).sum())),
        folds=("fold", "nunique"),
    )
    total = result["mean_importance"].sum()
    result["importance_share"] = result["mean_importance"] / total if total > 0 else 0.0
    return result.sort_values("mean_importance", ascending=False).reset_index(drop=True)


def _tail_regressor(seed: int, quantile: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=quantile,
        learning_rate=0.05, max_iter=180,
        max_leaf_nodes=7, max_depth=3, min_samples_leaf=30,
        l2_regularization=5.0, early_stopping=False,
        random_state=seed,
    )


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0
