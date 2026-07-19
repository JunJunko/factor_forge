from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from factor_forge.research.concept_etf_exit_ml import ExitMLRules, build_exit_folds
from factor_forge.research.concept_first_rotation import (
    CONCEPT_FEATURES,
    HORIZONS,
    HORIZON_WEIGHTS,
)


STATE_RESIDUAL_POLICIES = {
    "R1_within_linear_5d": "score_R1_within_linear_5d",
    "R2_within_nonlinear_5d": "score_R2_within_nonlinear_5d",
    "R3_within_multihorizon": "score_R3_within_multihorizon",
    "R4_two_stage": "score_R4_two_stage",
    "R5_within_state_placebo": "score_R5_within_state_placebo",
}


@dataclass(frozen=True)
class StateResidualRules:
    ridge_alpha: float = 20.0
    hgb_learning_rate: float = 0.05
    hgb_max_iter: int = 60
    hgb_max_depth: int = 3
    hgb_l2_regularization: float = 10.0
    minimum_train_days: int = 100
    validation_days: int = 20
    test_days: int = 20
    embargo_days: int = 11
    minimum_train_rows: int = 20_000
    state_prior_weight: float = 0.20
    concept_overlay_weight: float = 0.20
    seed: int = 42


def fit_state_residual_walk_forward(
    concepts: pd.DataFrame,
    *,
    start: str,
    end: str,
    rules: StateResidualRules = StateResidualRules(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result = concepts.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    calendar = pd.DatetimeIndex(sorted(result.loc[
        result["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end)),
        "trade_date",
    ].unique()))
    if len(calendar) == 0:
        raise ValueError("state-residual walk-forward calendar is empty")
    result = _attach_label_availability(result, calendar)
    folds = build_exit_folds(calendar, rules=ExitMLRules(
        minimum_train_days=rules.minimum_train_days,
        validation_days=rules.validation_days,
        test_days=rules.test_days,
        embargo_days=rules.embargo_days,
    ))
    eligible = result["eligible_concept"].astype("boolean").fillna(False).astype(bool)
    predictions: list[pd.DataFrame] = []
    coefficient_rows: list[dict] = []
    state_prior_rows: list[dict] = []
    audit_rows: list[dict] = []
    for fold in folds:
        test = result.loc[
            eligible & result["trade_date"].between(fold.test_start, fold.test_end)
        ].copy()
        if test.empty:
            continue
        output = test[["trade_date", "concept_code", "rrg_quadrant"]].copy()
        output["fold"] = fold.fold
        train_counts: dict[int, int] = {}
        maturity_max: dict[int, pd.Timestamp] = {}
        for horizon in HORIZONS:
            label = f"forward_excess_{horizon}d"
            availability = f"label_available_date_{horizon}d"
            train = result.loc[
                eligible
                & result["trade_date"].between(pd.Timestamp(start), fold.valid_end)
                & result[availability].lt(fold.test_start)
            ].dropna(subset=[label]).copy()
            if len(train) < rules.minimum_train_rows:
                break
            if train[availability].ge(fold.test_start).any():
                raise ValueError(f"fold {fold.fold} has immature {horizon}d state labels")
            train["state_date_mean"] = train.groupby(
                ["trade_date", "rrg_quadrant"], observed=True,
            )[label].transform("mean")
            train["within_state_label"] = train[label] - train["state_date_mean"]
            state_prior = train.groupby("rrg_quadrant", observed=True)[label].mean()
            global_prior = float(train[label].mean())
            output[f"state_prior_{horizon}d"] = (
                test["rrg_quadrant"].map(state_prior).fillna(global_prior)
            )
            for state, value in state_prior.items():
                state_prior_rows.append({
                    "fold": fold.fold,
                    "horizon": horizon,
                    "rrg_quadrant": state,
                    "state_prior": float(value),
                })
            linear = Ridge(alpha=rules.ridge_alpha)
            linear.fit(train[CONCEPT_FEATURES], train["within_state_label"])
            output[f"within_linear_prediction_{horizon}d"] = linear.predict(
                test[CONCEPT_FEATURES]
            )
            for feature, coefficient in zip(CONCEPT_FEATURES, linear.coef_):
                coefficient_rows.append({
                    "fold": fold.fold,
                    "horizon": horizon,
                    "feature": feature,
                    "coefficient": float(coefficient),
                })
            if horizon == 5:
                nonlinear = HistGradientBoostingRegressor(
                    learning_rate=rules.hgb_learning_rate,
                    max_iter=rules.hgb_max_iter,
                    max_depth=rules.hgb_max_depth,
                    l2_regularization=rules.hgb_l2_regularization,
                    random_state=rules.seed + fold.fold,
                )
                nonlinear.fit(train[CONCEPT_FEATURES], train["within_state_label"])
                output["within_nonlinear_prediction_5d"] = nonlinear.predict(
                    test[CONCEPT_FEATURES]
                )
            train_counts[horizon] = len(train)
            maturity_max[horizon] = pd.Timestamp(train[availability].max())
        if len(train_counts) != len(HORIZONS):
            continue
        predictions.append(output)
        audit_rows.append({
            "fold": fold.fold,
            "train_start": pd.Timestamp(start),
            "train_end": fold.valid_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            **{f"train_rows_{horizon}d": train_counts[horizon] for horizon in HORIZONS},
            **{
                f"train_label_available_max_{horizon}d": maturity_max[horizon]
                for horizon in HORIZONS
            },
            "test_rows": len(test),
        })
    if not predictions:
        return (
            pd.DataFrame(), pd.DataFrame(coefficient_rows),
            pd.DataFrame(state_prior_rows), pd.DataFrame(audit_rows),
        )
    scored = pd.concat(predictions, ignore_index=True)
    for horizon in HORIZONS:
        prediction = f"within_linear_prediction_{horizon}d"
        scored[f"within_linear_rank_{horizon}d"] = scored.groupby(
            ["trade_date", "rrg_quadrant"], observed=True,
        )[prediction].rank(pct=True)
    scored["score_R1_within_linear_5d"] = _state_date_zscore(
        scored, "within_linear_prediction_5d",
    )
    scored["score_R2_within_nonlinear_5d"] = _state_date_zscore(
        scored, "within_nonlinear_prediction_5d",
    )
    scored["score_R3_within_multihorizon"] = sum(
        HORIZON_WEIGHTS[horizon] * scored[f"within_linear_rank_{horizon}d"]
        for horizon in HORIZONS
    )
    scored["score_R3_within_multihorizon"] = _state_date_zscore(
        scored, "score_R3_within_multihorizon",
    )
    state_prior = sum(
        HORIZON_WEIGHTS[horizon] * scored[f"state_prior_{horizon}d"]
        for horizon in HORIZONS
    )
    scored["state_prior_multihorizon_z"] = state_prior.groupby(
        scored["trade_date"], sort=False,
    ).transform(_zscore)
    scored["score_R4_two_stage"] = (
        (1 - rules.state_prior_weight) * scored["score_R3_within_multihorizon"]
        + rules.state_prior_weight * scored["state_prior_multihorizon_z"]
    )
    scored["score_R5_within_state_placebo"] = _within_state_placebo(
        scored, "score_R3_within_multihorizon", seed=rules.seed,
    )
    return (
        scored,
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(state_prior_rows),
        pd.DataFrame(audit_rows),
    )


def attach_state_residual_scores_to_etfs(
    panel: pd.DataFrame,
    concept_scores: pd.DataFrame,
    *,
    concept_overlay_weight: float = 0.20,
) -> pd.DataFrame:
    if not 0 <= concept_overlay_weight <= 1:
        raise ValueError("concept_overlay_weight must be in [0, 1]")
    result = panel.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    columns = ["trade_date", "concept_code", "fold", *STATE_RESIDUAL_POLICIES.values()]
    result = result.merge(
        concept_scores[columns],
        on=["trade_date", "concept_code"], how="left", validate="many_to_one",
    )
    result["price_momentum_z"] = result.groupby(
        "trade_date", sort=False,
    )["score_etf_momentum"].transform(_zscore)
    result["score_S0_etf_r4"] = result["price_momentum_z"]
    overlay_mapping = {
        "score_S1_linear_overlay": "score_R1_within_linear_5d",
        "score_S2_nonlinear_overlay": "score_R2_within_nonlinear_5d",
        "score_S3_multihorizon_overlay": "score_R3_within_multihorizon",
        "score_S4_two_stage_overlay": "score_R4_two_stage",
        "score_S5_state_placebo_overlay": "score_R5_within_state_placebo",
    }
    for output, concept_column in overlay_mapping.items():
        result[output] = (
            (1 - concept_overlay_weight) * result["price_momentum_z"]
            + concept_overlay_weight * result[concept_column]
        )
    return result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def within_state_oof_diagnostics(
    concepts: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    policies: dict[str, str] = STATE_RESIDUAL_POLICIES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = concepts[[
        "trade_date", "concept_code", "rrg_quadrant",
        *[f"forward_excess_{horizon}d" for horizon in HORIZONS],
    ]].copy()
    sample = scores.merge(
        labels.drop(columns="rrg_quadrant"),
        on=["trade_date", "concept_code"], how="left", validate="one_to_one",
    )
    ic_rows: list[dict] = []
    bucket_rows: list[dict] = []
    for policy, score_column in policies.items():
        for horizon in HORIZONS:
            label = f"forward_excess_{horizon}d"
            frame = sample.dropna(subset=[score_column, label]).copy()
            frame["within_state_label"] = frame[label] - frame.groupby(
                ["trade_date", "rrg_quadrant"], observed=True,
            )[label].transform("mean")
            group_ic = frame.groupby(
                ["trade_date", "rrg_quadrant"], observed=True,
            ).apply(
                lambda group: (
                    group[score_column].corr(group["within_state_label"], method="spearman")
                    if len(group) >= 5 else np.nan
                ),
                include_groups=False,
            ).dropna()
            ic_rows.append({
                "policy": policy,
                "horizon": horizon,
                "dates": int(group_ic.index.get_level_values(0).nunique()),
                "state_date_groups": len(group_ic),
                "mean_within_state_rank_ic": float(group_ic.mean()),
                "positive_group_rate": float(group_ic.gt(0).mean()),
            })
            frame["bucket"] = frame.groupby(
                ["trade_date", "rrg_quadrant"], observed=True,
            )[score_column].transform(_five_bucket)
            grouped = frame.dropna(subset=["bucket"]).groupby(
                "bucket", observed=True,
            )["within_state_label"].agg(["mean", "size"])
            for bucket, row in grouped.iterrows():
                bucket_rows.append({
                    "policy": policy,
                    "horizon": horizon,
                    "bucket": int(bucket),
                    "mean_within_state_excess": float(row["mean"]),
                    "observations": int(row["size"]),
                })
    return pd.DataFrame(ic_rows), pd.DataFrame(bucket_rows)


def state_residual_coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    if coefficients.empty:
        return pd.DataFrame()
    return coefficients.groupby(["horizon", "feature"], as_index=False).agg(
        mean_coefficient=("coefficient", "mean"),
        median_coefficient=("coefficient", "median"),
        positive_fraction=("coefficient", lambda values: float(values.gt(0).mean())),
        negative_fraction=("coefficient", lambda values: float(values.lt(0).mean())),
        folds=("fold", "nunique"),
    )


def fit_frozen_s2_model(
    concepts: pd.DataFrame,
    *,
    training_cutoff: str,
    rules: StateResidualRules = StateResidualRules(),
) -> tuple[HistGradientBoostingRegressor, dict]:
    """Fit the preregistered S2 model using only labels mature by the cutoff."""
    result = concepts.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    cutoff = pd.Timestamp(training_cutoff)
    calendar = pd.DatetimeIndex(sorted(result.loc[
        result["trade_date"].le(cutoff), "trade_date",
    ].unique()))
    result = _attach_label_availability(result, calendar)
    eligible = result["eligible_concept"].astype("boolean").fillna(False).astype(bool)
    label = "forward_excess_5d"
    availability = "label_available_date_5d"
    train = result.loc[
        eligible
        & result["trade_date"].le(cutoff)
        & result[availability].le(cutoff)
    ].dropna(subset=[label]).copy()
    if len(train) < rules.minimum_train_rows:
        raise ValueError(
            f"frozen S2 requires {rules.minimum_train_rows} mature rows; got {len(train)}"
        )
    train["state_date_mean"] = train.groupby(
        ["trade_date", "rrg_quadrant"], observed=True,
    )[label].transform("mean")
    train["within_state_label"] = train[label] - train["state_date_mean"]
    model = HistGradientBoostingRegressor(
        learning_rate=rules.hgb_learning_rate,
        max_iter=rules.hgb_max_iter,
        max_depth=rules.hgb_max_depth,
        l2_regularization=rules.hgb_l2_regularization,
        random_state=rules.seed,
    )
    model.fit(train[CONCEPT_FEATURES], train["within_state_label"])
    audit = {
        "model": "HistGradientBoostingRegressor",
        "policy": "S2_nonlinear_overlay",
        "training_cutoff": str(cutoff.date()),
        "train_start": str(train["trade_date"].min().date()),
        "train_end": str(train["trade_date"].max().date()),
        "label_available_max": str(train[availability].max().date()),
        "mature_train_rows": int(len(train)),
        "train_days": int(train["trade_date"].nunique()),
        "train_concepts": int(train["concept_code"].nunique()),
        "feature_columns": list(CONCEPT_FEATURES),
        "hyperparameters": {
            "learning_rate": rules.hgb_learning_rate,
            "max_iter": rules.hgb_max_iter,
            "max_depth": rules.hgb_max_depth,
            "l2_regularization": rules.hgb_l2_regularization,
            "random_state": rules.seed,
            "concept_overlay_weight": rules.concept_overlay_weight,
        },
    }
    return model, audit


def score_frozen_s2_model(
    concepts: pd.DataFrame,
    model: HistGradientBoostingRegressor,
) -> pd.DataFrame:
    eligible = concepts["eligible_concept"].astype("boolean").fillna(False).astype(bool)
    result = concepts.loc[
        eligible, ["trade_date", "concept_code", "rrg_quadrant", *CONCEPT_FEATURES],
    ].copy()
    result["within_nonlinear_prediction_5d"] = model.predict(result[CONCEPT_FEATURES])
    result["score_R2_within_nonlinear_5d"] = _state_date_zscore(
        result, "within_nonlinear_prediction_5d",
    )
    return result[[
        "trade_date", "concept_code", "rrg_quadrant",
        "within_nonlinear_prediction_5d", "score_R2_within_nonlinear_5d",
    ]].sort_values(["trade_date", "concept_code"]).reset_index(drop=True)


def s2_fold_train_test_diagnostics(
    concepts: pd.DataFrame,
    *,
    start: str,
    end: str,
    rules: StateResidualRules = StateResidualRules(),
) -> pd.DataFrame:
    """Compare in-sample and OOF S2 rank direction without changing the model."""
    result = concepts.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    calendar = pd.DatetimeIndex(sorted(result.loc[
        result["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end)), "trade_date",
    ].unique()))
    result = _attach_label_availability(result, calendar)
    folds = build_exit_folds(calendar, rules=ExitMLRules(
        minimum_train_days=rules.minimum_train_days,
        validation_days=rules.validation_days,
        test_days=rules.test_days,
        embargo_days=rules.embargo_days,
    ))
    eligible = result["eligible_concept"].astype("boolean").fillna(False).astype(bool)
    rows = []
    for fold in folds:
        train = result.loc[
            eligible
            & result["trade_date"].between(pd.Timestamp(start), fold.valid_end)
            & result["label_available_date_5d"].lt(fold.test_start)
        ].dropna(subset=["forward_excess_5d"]).copy()
        test = result.loc[
            eligible & result["trade_date"].between(fold.test_start, fold.test_end)
        ].dropna(subset=["forward_excess_5d"]).copy()
        if len(train) < rules.minimum_train_rows or test.empty:
            continue
        train["within_state_label"] = train["forward_excess_5d"] - train.groupby(
            ["trade_date", "rrg_quadrant"], observed=True,
        )["forward_excess_5d"].transform("mean")
        test["within_state_label"] = test["forward_excess_5d"] - test.groupby(
            ["trade_date", "rrg_quadrant"], observed=True,
        )["forward_excess_5d"].transform("mean")
        model = HistGradientBoostingRegressor(
            learning_rate=rules.hgb_learning_rate,
            max_iter=rules.hgb_max_iter,
            max_depth=rules.hgb_max_depth,
            l2_regularization=rules.hgb_l2_regularization,
            random_state=rules.seed + fold.fold,
        )
        model.fit(train[CONCEPT_FEATURES], train["within_state_label"])
        for sample_name, sample in (("train_in_sample", train), ("test_oof", test)):
            evaluated = sample.copy()
            evaluated["prediction"] = model.predict(evaluated[CONCEPT_FEATURES])
            group_ic = evaluated.groupby(
                ["trade_date", "rrg_quadrant"], observed=True,
            ).apply(
                lambda group: (
                    group["prediction"].corr(group["within_state_label"], method="spearman")
                    if len(group) >= 5 else np.nan
                ),
                include_groups=False,
            ).dropna()
            rows.append({
                "fold": fold.fold, "sample": sample_name,
                "train_end": fold.valid_end, "test_start": fold.test_start,
                "test_end": fold.test_end, "rows": len(evaluated),
                "state_date_groups": len(group_ic),
                "mean_within_state_rank_ic": float(group_ic.mean()),
                "positive_group_rate": float(group_ic.gt(0).mean()),
            })
    return pd.DataFrame(rows)


def _attach_label_availability(
    frame: pd.DataFrame, calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    result = frame.copy()
    positions = {pd.Timestamp(date): index for index, date in enumerate(calendar)}
    for horizon in HORIZONS:
        mapping = {
            date: pd.Timestamp(calendar[index + horizon + 1])
            for date, index in positions.items()
            if index + horizon + 1 < len(calendar)
        }
        result[f"label_available_date_{horizon}d"] = result["trade_date"].map(mapping)
    return result


def _state_date_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby(
        ["trade_date", "rrg_quadrant"], observed=True,
    )[column].transform(_zscore)


def _within_state_placebo(
    frame: pd.DataFrame, column: str, *, seed: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for (date, state), indices in frame.groupby(
        ["trade_date", "rrg_quadrant"], sort=True,
    ).groups.items():
        values = frame.loc[indices, column]
        valid = values.notna()
        salt = sum(ord(character) for character in str(state))
        rng = np.random.default_rng(seed + int(pd.Timestamp(date).strftime("%Y%m%d")) + salt)
        result.loc[values.index[valid]] = rng.permutation(values.loc[valid].to_numpy())
    return result


def _five_bucket(values: pd.Series) -> pd.Series:
    valid = values.notna()
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.sum() < 5:
        return result
    ranks = values.loc[valid].rank(method="first", pct=True)
    result.loc[valid] = np.ceil(ranks * 5).clip(1, 5)
    return result


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = numeric.std(ddof=0)
    if not np.isfinite(std) or std <= 0:
        return pd.Series(0.0, index=values.index)
    return (numeric - numeric.mean()) / std
