from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from factor_forge.research.concept_etf_exit_ml import ExitMLRules, build_exit_folds


DIFFUSION_FEATURES = [
    "price_momentum_z",
    "diffusion_rank_z",
    "breadth_acceleration_rank_z",
    "breadth_persistence_rank_z",
]


@dataclass(frozen=True)
class DiffusionBlendRules:
    fixed_diffusion_weight: float = 0.20
    maximum_learned_diffusion_weight: float = 0.30
    ridge_alpha: float = 10.0
    horizon: int = 5
    minimum_train_days: int = 60
    validation_days: int = 15
    test_days: int = 15
    embargo_days: int = 6
    minimum_train_rows: int = 400
    seed: int = 42


def attach_positive_diffusion_scores(
    panel: pd.DataFrame,
    *,
    fixed_diffusion_weight: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    if not 0 <= fixed_diffusion_weight <= 1:
        raise ValueError("fixed_diffusion_weight must be in [0, 1]")
    result = panel.sort_values(["ts_code", "trade_date"]).copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    grouped = result.groupby("ts_code", sort=False)["common_breadth_delta_smooth5"]
    result["breadth_acceleration_5d"] = result["common_breadth_delta_smooth5"] - grouped.shift(5)
    positive = result["common_breadth_delta_smooth5"].gt(0).astype(float)
    result["breadth_positive_fraction_3d"] = positive.groupby(result["ts_code"], sort=False).transform(
        lambda values: values.rolling(3, min_periods=2).mean()
    )
    by_date = result.groupby("trade_date", sort=False)
    result["price_momentum_z"] = by_date["score_etf_momentum"].transform(_zscore)
    result["diffusion_rank_z"] = by_date["common_delta_rank"].transform(_zscore)
    acceleration_rank = by_date["breadth_acceleration_5d"].rank(pct=True)
    persistence_rank = by_date["breadth_positive_fraction_3d"].rank(pct=True)
    result["breadth_acceleration_rank_z"] = acceleration_rank.groupby(result["trade_date"]).transform(_zscore)
    result["breadth_persistence_rank_z"] = persistence_rank.groupby(result["trade_date"]).transform(_zscore)
    result["positive_diffusion_score"] = (
        0.50 * result["diffusion_rank_z"]
        + 0.30 * result["breadth_acceleration_rank_z"]
        + 0.20 * result["breadth_persistence_rank_z"]
    )
    result["score_B0_price"] = result["price_momentum_z"]
    result["score_B1_fixed_diffusion"] = (
        (1 - fixed_diffusion_weight) * result["price_momentum_z"]
        + fixed_diffusion_weight * result["positive_diffusion_score"]
    )
    result["placebo_diffusion_score"] = np.nan
    for date, indices in result.groupby("trade_date", sort=True).groups.items():
        values = result.loc[indices, "positive_diffusion_score"]
        valid = values.notna()
        rng = np.random.default_rng(seed + int(pd.Timestamp(date).strftime("%Y%m%d")))
        result.loc[values.index[valid], "placebo_diffusion_score"] = rng.permutation(
            values.loc[valid].to_numpy()
        )
    result["score_B3_placebo"] = (
        (1 - fixed_diffusion_weight) * result["price_momentum_z"]
        + fixed_diffusion_weight * result["placebo_diffusion_score"]
    )
    return result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def fit_positive_diffusion_walk_forward(
    panel: pd.DataFrame,
    *,
    start: str,
    end: str,
    rules: DiffusionBlendRules = DiffusionBlendRules(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result = panel.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    calendar = pd.DatetimeIndex(sorted(result.loc[
        result["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end)), "trade_date"
    ].unique()))
    positions = {pd.Timestamp(date): index for index, date in enumerate(calendar)}
    label_dates = {
        date: pd.Timestamp(calendar[index + rules.horizon + 1])
        for date, index in positions.items() if index + rules.horizon + 1 < len(calendar)
    }
    result["label_available_date"] = result["trade_date"].map(label_dates)
    result["forward_excess_5d"] = result["forward_open_5d"] - result.groupby(
        "trade_date", sort=False
    )["forward_open_5d"].transform("mean")
    fold_rules = ExitMLRules(
        minimum_train_days=rules.minimum_train_days,
        validation_days=rules.validation_days,
        test_days=rules.test_days,
        embargo_days=rules.embargo_days,
    )
    folds = build_exit_folds(calendar, rules=fold_rules)
    predictions: list[pd.DataFrame] = []
    weight_rows: list[dict] = []
    audit_rows: list[dict] = []
    for fold in folds:
        train = result.loc[
            result["trade_date"].between(pd.Timestamp(start), fold.valid_end)
            & result["label_available_date"].lt(fold.test_start)
        ].dropna(subset=DIFFUSION_FEATURES + ["forward_excess_5d"])
        test = result.loc[result["trade_date"].between(fold.test_start, fold.test_end)].copy()
        if len(train) < rules.minimum_train_rows or test.empty:
            continue
        if train["label_available_date"].ge(fold.test_start).any():
            raise ValueError(f"fold {fold.fold} has immature diffusion labels")
        model = Ridge(alpha=rules.ridge_alpha, positive=True)
        model.fit(train[DIFFUSION_FEATURES], train["forward_excess_5d"])
        weights = constrained_blend_weights(
            model.coef_, maximum_diffusion_weight=rules.maximum_learned_diffusion_weight,
        )
        output = test[["trade_date", "ts_code"]].copy()
        output["score_B2_learned_diffusion"] = sum(
            weights[column] * test[column] for column in DIFFUSION_FEATURES
        )
        output["fold"] = fold.fold
        predictions.append(output)
        weight_rows.append({"fold": fold.fold, **weights})
        audit_rows.append({
            "fold": fold.fold,
            "train_start": train["trade_date"].min(),
            "train_end": train["trade_date"].max(),
            "test_start": fold.test_start, "test_end": fold.test_end,
            "train_rows": len(train), "test_rows": len(test),
            "train_label_available_max": train["label_available_date"].max(),
        })
    oof = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    weights = pd.DataFrame(weight_rows)
    audit = pd.DataFrame(audit_rows)
    return oof, weights, audit


def attach_learned_oof_score(panel: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    result = panel.merge(
        predictions[["trade_date", "ts_code", "score_B2_learned_diffusion", "fold"]],
        on=["trade_date", "ts_code"], how="left", validate="one_to_one",
    )
    result["score_B2_learned_diffusion"] = result["score_B2_learned_diffusion"].where(
        result["score_B2_learned_diffusion"].notna(), result["score_B0_price"]
    )
    return result


def constrained_blend_weights(
    coefficients: np.ndarray,
    *,
    maximum_diffusion_weight: float,
) -> dict[str, float]:
    values = np.clip(np.asarray(coefficients, dtype=float), 0, None)
    price = values[0]
    diffusion = values[1:]
    diffusion_sum = float(diffusion.sum())
    if diffusion_sum <= 0:
        return dict(zip(DIFFUSION_FEATURES, [1.0, 0.0, 0.0, 0.0]))
    raw_share = diffusion_sum / (price + diffusion_sum) if price + diffusion_sum > 0 else 1.0
    diffusion_share = min(raw_share, maximum_diffusion_weight)
    diffusion_weights = diffusion / diffusion_sum * diffusion_share
    return dict(zip(
        DIFFUSION_FEATURES,
        np.r_[1 - diffusion_share, diffusion_weights].tolist(),
    ))


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = numeric.std(ddof=0)
    if not np.isfinite(std) or std <= 0:
        return pd.Series(0.0, index=values.index)
    return (numeric - numeric.mean()) / std
