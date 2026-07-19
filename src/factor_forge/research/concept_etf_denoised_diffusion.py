from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from factor_forge.research.concept_etf_exit_ml import ExitMLRules, build_exit_folds


DENOISED_ML_FEATURES = [
    "price_momentum_z",
    "denoised_diffusion_z",
    "diffusion_persistence_z",
    "price_diffusion_interaction_z",
]

AUXILIARY_CONTROLS = [
    "breadth_float",
    "concept_amount",
    "matched_member_count",
    "membership_churn_5d",
]


@dataclass(frozen=True)
class DenoisedDiffusionRules:
    maximum_diffusion_weight: float = 0.10
    ridge_alpha: float = 10.0
    control_ridge_alpha: float = 5.0
    horizon: int = 5
    minimum_train_days: int = 60
    validation_days: int = 15
    test_days: int = 15
    embargo_days: int = 6
    minimum_train_rows: int = 400
    stability_blocks: int = 5
    minimum_stability_fraction: float = 0.60
    seed: int = 42


def attach_denoised_diffusion_scores(
    panel: pd.DataFrame,
    concept_features: pd.DataFrame | None = None,
    *,
    maximum_diffusion_weight: float = 0.10,
    control_ridge_alpha: float = 5.0,
    confirmation_percentile: float = 0.30,
    seed: int = 42,
) -> pd.DataFrame:
    """Build causal denoised diffusion scores known at signal-date close."""
    if not 0 <= maximum_diffusion_weight <= 1:
        raise ValueError("maximum_diffusion_weight must be in [0, 1]")
    if not 0 <= confirmation_percentile <= 1:
        raise ValueError("confirmation_percentile must be in [0, 1]")
    result = panel.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    if concept_features is not None:
        result = _attach_auxiliary_controls(result, concept_features)
    result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    by_date = result.groupby("trade_date", sort=False)
    result["price_momentum_z"] = by_date["score_etf_momentum"].transform(_zscore)
    raw = pd.to_numeric(result["common_breadth_delta_smooth5"], errors="coerce")
    lower = raw.groupby(result["trade_date"], sort=False).transform(lambda x: x.quantile(0.05))
    upper = raw.groupby(result["trade_date"], sort=False).transform(lambda x: x.quantile(0.95))
    result["diffusion_winsorized"] = raw.clip(lower=lower, upper=upper)
    result["diffusion_temporal_filter"] = result.groupby(
        "ts_code", sort=False
    )["diffusion_winsorized"].transform(
        lambda x: x.rolling(3, min_periods=2).median().ewm(span=3, adjust=False).mean()
    )

    controls = ["price_momentum_z"]
    for column in AUXILIARY_CONTROLS:
        if column in result and pd.to_numeric(result[column], errors="coerce").notna().any():
            control_name = f"control_{column}_z"
            if column == "concept_amount":
                values = np.log1p(pd.to_numeric(result[column], errors="coerce").clip(lower=0))
            else:
                values = pd.to_numeric(result[column], errors="coerce")
            result[control_name] = values.groupby(result["trade_date"], sort=False).transform(_zscore)
            controls.append(control_name)
    result["denoised_diffusion_residual"] = _daily_ridge_residual(
        result,
        target="diffusion_temporal_filter",
        controls=controls,
        alpha=control_ridge_alpha,
    )
    by_date = result.groupby("trade_date", sort=False)
    result["denoised_diffusion_z"] = by_date["denoised_diffusion_residual"].transform(_zscore)
    positive = result["denoised_diffusion_residual"].gt(0).astype(float)
    result["diffusion_positive_fraction_3d"] = positive.groupby(
        result["ts_code"], sort=False
    ).transform(lambda x: x.rolling(3, min_periods=2).mean())
    result["diffusion_persistence_z"] = result.groupby(
        "trade_date", sort=False
    )["diffusion_positive_fraction_3d"].transform(_zscore)
    interaction = (
        result["price_momentum_z"].clip(lower=0)
        * result["diffusion_persistence_z"].clip(lower=0)
    )
    result["price_diffusion_interaction_z"] = interaction.groupby(
        result["trade_date"], sort=False
    ).transform(_zscore)
    result["denoised_diffusion_score"] = (
        0.70 * result["denoised_diffusion_z"]
        + 0.30 * result["diffusion_persistence_z"]
    )

    residual_rank = result.groupby("trade_date", sort=False)["denoised_diffusion_z"].rank(pct=True)
    persistence_rank = result.groupby(
        "trade_date", sort=False
    )["diffusion_persistence_z"].rank(pct=True)
    result["diffusion_confirmation_gate"] = ~(
        residual_rank.lt(confirmation_percentile)
        & persistence_rank.lt(confirmation_percentile)
    )
    result["score_D0_price"] = result["price_momentum_z"]
    result["score_D1_confirmation"] = result["price_momentum_z"].where(
        result["diffusion_confirmation_gate"]
    )
    result["score_D2_ten_percent_boost"] = (
        (1 - maximum_diffusion_weight) * result["price_momentum_z"]
        + maximum_diffusion_weight * result["denoised_diffusion_score"]
    )
    result["placebo_denoised_diffusion"] = _daily_permutation(
        result, "denoised_diffusion_score", seed=seed,
    )
    result["score_D4_placebo"] = (
        (1 - maximum_diffusion_weight) * result["price_momentum_z"]
        + maximum_diffusion_weight * result["placebo_denoised_diffusion"]
    )
    return result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def fit_denoised_diffusion_walk_forward(
    panel: pd.DataFrame,
    *,
    start: str,
    end: str,
    rules: DenoisedDiffusionRules = DenoisedDiffusionRules(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result = panel.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    calendar = pd.DatetimeIndex(sorted(result.loc[
        result["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end)), "trade_date"
    ].unique()))
    positions = {pd.Timestamp(date): index for index, date in enumerate(calendar)}
    label_dates = {
        date: pd.Timestamp(calendar[index + rules.horizon + 1])
        for date, index in positions.items()
        if index + rules.horizon + 1 < len(calendar)
    }
    result["label_available_date"] = result["trade_date"].map(label_dates)
    result["forward_excess_5d"] = result["forward_open_5d"] - result.groupby(
        "trade_date", sort=False
    )["forward_open_5d"].transform("mean")
    folds = build_exit_folds(calendar, rules=ExitMLRules(
        minimum_train_days=rules.minimum_train_days,
        validation_days=rules.validation_days,
        test_days=rules.test_days,
        embargo_days=rules.embargo_days,
    ))
    predictions: list[pd.DataFrame] = []
    weight_rows: list[dict] = []
    stability_rows: list[dict] = []
    audit_rows: list[dict] = []
    for fold in folds:
        train = result.loc[
            result["trade_date"].between(pd.Timestamp(start), fold.valid_end)
            & result["label_available_date"].lt(fold.test_start)
        ].dropna(subset=DENOISED_ML_FEATURES + ["forward_excess_5d"])
        test = result.loc[result["trade_date"].between(fold.test_start, fold.test_end)].copy()
        if len(train) < rules.minimum_train_rows or test.empty:
            continue
        if train["label_available_date"].ge(fold.test_start).any():
            raise ValueError(f"fold {fold.fold} has immature diffusion labels")
        stability = _blocked_stability(train, rules)
        selected = {
            feature
            for feature in DENOISED_ML_FEATURES[1:]
            if stability[feature] >= rules.minimum_stability_fraction
        }
        model = Ridge(alpha=rules.ridge_alpha, positive=True)
        model.fit(train[DENOISED_ML_FEATURES], train["forward_excess_5d"])
        coefficients = np.asarray(model.coef_, dtype=float)
        for index, feature in enumerate(DENOISED_ML_FEATURES[1:], start=1):
            if feature not in selected:
                coefficients[index] = 0.0
        weights = constrained_denoised_weights(
            coefficients,
            maximum_diffusion_weight=rules.maximum_diffusion_weight,
        )
        output = test[["trade_date", "ts_code"]].copy()
        output["score_D3_learned"] = sum(
            weights[column] * test[column] for column in DENOISED_ML_FEATURES
        )
        output["fold"] = fold.fold
        predictions.append(output)
        weight_rows.append({"fold": fold.fold, **weights})
        stability_rows.append({"fold": fold.fold, **stability})
        audit_rows.append({
            "fold": fold.fold,
            "train_start": train["trade_date"].min(),
            "train_end": train["trade_date"].max(),
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_label_available_max": train["label_available_date"].max(),
            "selected_diffusion_features": ",".join(sorted(selected)),
        })
    return (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.DataFrame(weight_rows),
        pd.DataFrame(stability_rows),
        pd.DataFrame(audit_rows),
    )


def attach_learned_denoised_score(panel: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    result = panel.merge(
        predictions[["trade_date", "ts_code", "score_D3_learned", "fold"]],
        on=["trade_date", "ts_code"], how="left", validate="one_to_one",
    )
    result["score_D3_learned"] = result["score_D3_learned"].where(
        result["score_D3_learned"].notna(), result["score_D0_price"]
    )
    return result


def attach_forward_open_returns(
    panel: pd.DataFrame, horizons: tuple[int, ...] = (1, 3, 5, 10),
) -> pd.DataFrame:
    result = panel.sort_values(["ts_code", "trade_date"]).copy()
    grouped = result.groupby("ts_code", sort=False)["adj_open"]
    entry = grouped.shift(-1)
    for horizon in horizons:
        column = f"forward_open_{horizon}d"
        if column not in result:
            result[column] = grouped.shift(-(horizon + 1)) / entry - 1
    return result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def diffusion_signal_diagnostics(
    panel: pd.DataFrame,
    *,
    start: str,
    end: str,
    horizons: tuple[int, ...] = (1, 3, 5, 10),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = attach_forward_open_returns(panel, horizons)
    frame = frame.loc[frame["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    ic_rows: list[dict] = []
    bucket_rows: list[dict] = []
    for horizon in horizons:
        label = f"forward_open_{horizon}d"
        sample = frame.dropna(subset=["denoised_diffusion_score", label]).copy()
        sample["forward_excess"] = sample[label] - sample.groupby(
            "trade_date", sort=False
        )[label].transform("mean")
        daily_ic = sample.groupby("trade_date", observed=True).apply(
            lambda day: day["denoised_diffusion_score"].corr(
                day["forward_excess"], method="spearman"
            ),
            include_groups=False,
        ).dropna()
        ic_rows.append({
            "horizon": horizon,
            "days": len(daily_ic),
            "mean_rank_ic": float(daily_ic.mean()),
            "positive_rank_ic_rate": float(daily_ic.gt(0).mean()),
        })
        sample["bucket"] = sample.groupby("trade_date", observed=True)[
            "denoised_diffusion_score"
        ].transform(_five_bucket)
        grouped = sample.dropna(subset=["bucket"]).groupby("bucket", observed=True)[
            "forward_excess"
        ].agg(["mean", "size"])
        for bucket, row in grouped.iterrows():
            bucket_rows.append({
                "horizon": horizon,
                "bucket": int(bucket),
                "mean_forward_excess": float(row["mean"]),
                "observations": int(row["size"]),
            })
    return pd.DataFrame(ic_rows), pd.DataFrame(bucket_rows)


def constrained_denoised_weights(
    coefficients: np.ndarray,
    *,
    maximum_diffusion_weight: float,
) -> dict[str, float]:
    values = np.clip(np.asarray(coefficients, dtype=float), 0, None)
    diffusion = values[1:]
    diffusion_sum = float(diffusion.sum())
    if diffusion_sum <= 0:
        return dict(zip(DENOISED_ML_FEATURES, [1.0, 0.0, 0.0, 0.0]))
    price = float(values[0])
    raw_share = diffusion_sum / (price + diffusion_sum) if price + diffusion_sum > 0 else 1.0
    diffusion_share = min(raw_share, maximum_diffusion_weight)
    diffusion_weights = diffusion / diffusion_sum * diffusion_share
    return dict(zip(
        DENOISED_ML_FEATURES,
        np.r_[1 - diffusion_share, diffusion_weights].tolist(),
    ))


def _attach_auxiliary_controls(panel: pd.DataFrame, concepts: pd.DataFrame) -> pd.DataFrame:
    features = concepts.copy()
    features["trade_date"] = pd.to_datetime(features["trade_date"])
    columns = ["trade_date", "concept_code", *[
        column for column in AUXILIARY_CONTROLS if column in features
    ]]
    duplicate = [column for column in columns[2:] if column in panel]
    base = panel.drop(columns=duplicate)
    return base.merge(
        features[columns].drop_duplicates(["trade_date", "concept_code"]),
        on=["trade_date", "concept_code"], how="left", validate="many_to_one",
    )


def _daily_ridge_residual(
    frame: pd.DataFrame,
    *,
    target: str,
    controls: list[str],
    alpha: float,
) -> pd.Series:
    residual = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, day in frame.groupby("trade_date", sort=False):
        valid = day[[target, *controls]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < max(5, len(controls) + 2):
            residual.loc[valid.index] = valid[target] - valid[target].mean()
            continue
        model = Ridge(alpha=alpha)
        model.fit(valid[controls], valid[target])
        residual.loc[valid.index] = valid[target] - model.predict(valid[controls])
    return residual


def _blocked_stability(train: pd.DataFrame, rules: DenoisedDiffusionRules) -> dict[str, float]:
    dates = np.asarray(sorted(train["trade_date"].unique()))
    blocks = [block for block in np.array_split(dates, rules.stability_blocks) if len(block)]
    positive = dict.fromkeys(DENOISED_ML_FEATURES, 0)
    fitted = 0
    for dates_in_block in blocks:
        sample = train.loc[train["trade_date"].isin(dates_in_block)]
        if len(sample) < max(50, len(DENOISED_ML_FEATURES) * 10):
            continue
        model = Ridge(alpha=rules.ridge_alpha, positive=True)
        model.fit(sample[DENOISED_ML_FEATURES], sample["forward_excess_5d"])
        for feature, coefficient in zip(DENOISED_ML_FEATURES, model.coef_):
            positive[feature] += int(float(coefficient) > 1e-10)
        fitted += 1
    return {
        feature: positive[feature] / fitted if fitted else 0.0
        for feature in DENOISED_ML_FEATURES
    }


def _daily_permutation(frame: pd.DataFrame, column: str, *, seed: int) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for date, indices in frame.groupby("trade_date", sort=True).groups.items():
        values = frame.loc[indices, column]
        valid = values.notna()
        rng = np.random.default_rng(seed + int(pd.Timestamp(date).strftime("%Y%m%d")))
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
