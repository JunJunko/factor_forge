from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from factor_forge.research.concept_etf_exit_ml import ExitMLRules, build_exit_folds


HORIZONS = (3, 5, 10)
HORIZON_WEIGHTS = {3: 0.30, 5: 0.40, 10: 0.30}

CONCEPT_FEATURES = [
    "residual_momentum_5d_z",
    "residual_momentum_20d_z",
    "residual_momentum_60d_z",
    "momentum_acceleration_z",
    "rs_momentum_5d_z",
    "diffusion_level_z",
    "diffusion_acceleration_z",
    "diffusion_persistence_z",
    "breadth_price_divergence_z",
    "breadth_equal_minus_float_z",
    "amount_participation_change_z",
    "concept_volatility_20d_z",
    "membership_churn_5d_z",
    "member_match_coverage_z",
    "rrg_improving_transition",
    "rrg_leading",
    "rrg_weakening",
]


@dataclass(frozen=True)
class ConceptFirstRules:
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
    seed: int = 42


def build_concept_first_features(concepts: pd.DataFrame) -> pd.DataFrame:
    result = concepts.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    result = result.sort_values(["concept_code", "trade_date"]).reset_index(drop=True)
    grouped = result.groupby("concept_code", sort=False)

    for horizon in (5, 20, 60):
        concept_return = pd.to_numeric(result[f"concept_return_{horizon}d"], errors="coerce")
        market_return = pd.to_numeric(result[f"market_return_{horizon}d"], errors="coerce")
        result[f"residual_momentum_{horizon}d"] = (
            (1 + concept_return) / (1 + market_return) - 1
        )
    result["momentum_acceleration"] = (
        result["residual_momentum_5d"] - result["residual_momentum_20d"] / 4
    )
    result["diffusion_level"] = pd.to_numeric(
        result["common_breadth_delta_smooth5"], errors="coerce",
    )
    result["diffusion_acceleration"] = (
        result["diffusion_level"] - grouped["diffusion_level"].shift(5)
    )
    positive = result["diffusion_level"].gt(0).astype(float)
    result["diffusion_persistence"] = positive.groupby(
        result["concept_code"], sort=False,
    ).transform(lambda values: values.rolling(3, min_periods=2).mean())
    result["breadth_price_divergence"] = (
        _cross_sectional_z(result, "diffusion_level")
        - _cross_sectional_z(result, "rs_momentum_5d")
    )
    result["breadth_equal_minus_float"] = (
        pd.to_numeric(result["breadth_equal_raw"], errors="coerce")
        - pd.to_numeric(result["breadth_float_raw"], errors="coerce")
    )
    log_amount = np.log1p(pd.to_numeric(result["concept_amount"], errors="coerce").clip(lower=0))
    result["amount_participation_change"] = log_amount - log_amount.groupby(
        result["concept_code"], sort=False,
    ).shift(5)
    result["concept_volatility_20d"] = grouped["concept_return_1d"].transform(
        lambda values: values.rolling(20, min_periods=15).std(ddof=0)
    )
    previous_rrg = grouped["rrg_quadrant"].shift(1)
    result["rrg_improving_transition"] = (
        result["rrg_quadrant"].eq("improving") & previous_rrg.ne("improving")
    ).astype(float)
    result["rrg_leading"] = result["rrg_quadrant"].eq("leading").astype(float)
    result["rrg_weakening"] = result["rrg_quadrant"].eq("weakening").astype(float)

    raw_to_feature = {
        "residual_momentum_5d": "residual_momentum_5d_z",
        "residual_momentum_20d": "residual_momentum_20d_z",
        "residual_momentum_60d": "residual_momentum_60d_z",
        "momentum_acceleration": "momentum_acceleration_z",
        "rs_momentum_5d": "rs_momentum_5d_z",
        "diffusion_level": "diffusion_level_z",
        "diffusion_acceleration": "diffusion_acceleration_z",
        "diffusion_persistence": "diffusion_persistence_z",
        "breadth_price_divergence": "breadth_price_divergence_z",
        "breadth_equal_minus_float": "breadth_equal_minus_float_z",
        "amount_participation_change": "amount_participation_change_z",
        "concept_volatility_20d": "concept_volatility_20d_z",
        "membership_churn_5d": "membership_churn_5d_z",
        "member_match_coverage": "member_match_coverage_z",
    }
    for raw, feature in raw_to_feature.items():
        result[feature] = _cross_sectional_z(result, raw).fillna(0.0)
    result[CONCEPT_FEATURES] = result[CONCEPT_FEATURES].replace(
        [np.inf, -np.inf], np.nan,
    ).fillna(0.0)
    return result.sort_values(["trade_date", "concept_code"]).reset_index(drop=True)


def fit_concept_first_walk_forward(
    concepts: pd.DataFrame,
    *,
    start: str,
    end: str,
    rules: ConceptFirstRules = ConceptFirstRules(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result = concepts.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    calendar = pd.DatetimeIndex(sorted(result.loc[
        result["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end)),
        "trade_date",
    ].unique()))
    if len(calendar) == 0:
        raise ValueError("concept walk-forward calendar is empty")
    result = _attach_label_availability(result, calendar)
    folds = build_exit_folds(calendar, rules=ExitMLRules(
        minimum_train_days=rules.minimum_train_days,
        validation_days=rules.validation_days,
        test_days=rules.test_days,
        embargo_days=rules.embargo_days,
    ))
    predictions: list[pd.DataFrame] = []
    coefficient_rows: list[dict] = []
    audit_rows: list[dict] = []
    eligible = result["eligible_concept"].astype("boolean").fillna(False).astype(bool)
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
            ].dropna(subset=[label])
            if len(train) < rules.minimum_train_rows:
                break
            if train[availability].ge(fold.test_start).any():
                raise ValueError(f"fold {fold.fold} has immature {horizon}d labels")
            linear = Ridge(alpha=rules.ridge_alpha)
            linear.fit(train[CONCEPT_FEATURES], train[label])
            output[f"linear_prediction_{horizon}d"] = linear.predict(test[CONCEPT_FEATURES])
            for feature, coefficient in zip(CONCEPT_FEATURES, linear.coef_):
                coefficient_rows.append({
                    "fold": fold.fold,
                    "horizon": horizon,
                    "feature": feature,
                    "coefficient": float(coefficient),
                })
            train_counts[horizon] = len(train)
            maturity_max[horizon] = pd.Timestamp(train[availability].max())
            if horizon == 5:
                nonlinear = HistGradientBoostingRegressor(
                    learning_rate=rules.hgb_learning_rate,
                    max_iter=rules.hgb_max_iter,
                    max_depth=rules.hgb_max_depth,
                    l2_regularization=rules.hgb_l2_regularization,
                    random_state=rules.seed + fold.fold,
                )
                nonlinear.fit(train[CONCEPT_FEATURES], train[label])
                output["nonlinear_prediction_5d"] = nonlinear.predict(test[CONCEPT_FEATURES])
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
        return pd.DataFrame(), pd.DataFrame(coefficient_rows), pd.DataFrame(audit_rows)
    scored = pd.concat(predictions, ignore_index=True)
    for horizon in HORIZONS:
        column = f"linear_prediction_{horizon}d"
        scored[f"linear_rank_{horizon}d"] = scored.groupby(
            "trade_date", sort=False,
        )[column].rank(pct=True)
    scored["score_C1_linear_5d"] = _date_zscore(scored, "linear_prediction_5d")
    scored["score_C2_nonlinear_5d"] = _date_zscore(scored, "nonlinear_prediction_5d")
    scored["score_C3_multihorizon"] = sum(
        HORIZON_WEIGHTS[horizon] * scored[f"linear_rank_{horizon}d"]
        for horizon in HORIZONS
    )
    scored["score_C3_multihorizon"] = _date_zscore(scored, "score_C3_multihorizon")
    scored["score_C5_state_placebo"] = _state_preserving_placebo(
        scored, "score_C3_multihorizon", seed=rules.seed,
    )
    return scored, pd.DataFrame(coefficient_rows), pd.DataFrame(audit_rows)


def attach_concept_scores_to_etfs(
    panel: pd.DataFrame,
    concept_scores: pd.DataFrame,
) -> pd.DataFrame:
    result = panel.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    columns = [
        "trade_date", "concept_code", "score_C1_linear_5d",
        "score_C2_nonlinear_5d", "score_C3_multihorizon",
        "score_C5_state_placebo", "fold",
    ]
    result = result.merge(
        concept_scores[columns],
        on=["trade_date", "concept_code"], how="left", validate="many_to_one",
    )
    result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    result["mapping_correlation_60d"] = result.groupby(
        "ts_code", sort=False,
    ).apply(
        lambda group: group["etf_return_1d"].rolling(60, min_periods=40).corr(
            group["concept_return_1d"]
        ),
        include_groups=False,
    ).reset_index(level=0, drop=True).sort_index()
    tracking_residual = result["etf_return_1d"] - result["concept_return_1d"]
    result["tracking_residual_volatility_20d"] = tracking_residual.groupby(
        result["ts_code"], sort=False,
    ).transform(lambda values: values.rolling(20, min_periods=15).std(ddof=0))
    result["mapping_quality_score"] = (
        0.50 * _cross_sectional_z(result, "mapping_correlation_60d")
        - 0.25 * _cross_sectional_z(result, "tracking_residual_volatility_20d")
        + 0.125 * _cross_sectional_log_z(result, "amount_cny")
        + 0.125 * _cross_sectional_log_z(result, "aum_cny")
    )
    result["score_C0_etf_r4"] = result["score_etf_momentum"]
    result["score_C4_mapping_quality"] = (
        0.80 * result["score_C3_multihorizon"]
        + 0.20 * result["mapping_quality_score"]
    )
    return result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def concept_oof_diagnostics(
    concepts: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    policies: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = concepts[[
        "trade_date", "concept_code", *[f"forward_excess_{horizon}d" for horizon in HORIZONS]
    ]]
    sample = scores.merge(
        labels, on=["trade_date", "concept_code"], how="left", validate="one_to_one",
    )
    ic_rows: list[dict] = []
    bucket_rows: list[dict] = []
    for policy, score_column in policies.items():
        for horizon in HORIZONS:
            label = f"forward_excess_{horizon}d"
            frame = sample.dropna(subset=[score_column, label]).copy()
            daily_ic = frame.groupby("trade_date", observed=True).apply(
                lambda day: day[score_column].corr(day[label], method="spearman"),
                include_groups=False,
            ).dropna()
            ic_rows.append({
                "policy": policy,
                "horizon": horizon,
                "days": len(daily_ic),
                "mean_rank_ic": float(daily_ic.mean()),
                "positive_rank_ic_rate": float(daily_ic.gt(0).mean()),
            })
            frame["bucket"] = frame.groupby("trade_date", observed=True)[score_column].transform(
                _five_bucket,
            )
            grouped = frame.groupby("bucket", observed=True)[label].agg(["mean", "size"])
            for bucket, row in grouped.iterrows():
                bucket_rows.append({
                    "policy": policy,
                    "horizon": horizon,
                    "bucket": int(bucket),
                    "mean_forward_excess": float(row["mean"]),
                    "observations": int(row["size"]),
                })
    return pd.DataFrame(ic_rows), pd.DataFrame(bucket_rows)


def coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    if coefficients.empty:
        return pd.DataFrame()
    return coefficients.groupby(["horizon", "feature"], as_index=False).agg(
        mean_coefficient=("coefficient", "mean"),
        median_coefficient=("coefficient", "median"),
        positive_fraction=("coefficient", lambda values: float(values.gt(0).mean())),
        negative_fraction=("coefficient", lambda values: float(values.lt(0).mean())),
        folds=("fold", "nunique"),
    )


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


def _state_preserving_placebo(
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


def _cross_sectional_z(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.groupby(frame["trade_date"], sort=False).transform(_zscore)


def _cross_sectional_log_z(frame: pd.DataFrame, column: str) -> pd.Series:
    values = np.log1p(pd.to_numeric(frame[column], errors="coerce").clip(lower=0))
    return values.groupby(frame["trade_date"], sort=False).transform(_zscore)


def _date_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("trade_date", sort=False)[column].transform(_zscore)


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
