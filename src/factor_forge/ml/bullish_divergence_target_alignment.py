from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from factor_forge.ml.bullish_divergence_conditional_ml import (
    ModelConfig,
    TestPeriod,
    daily_equal_sample_weight,
    feature_columns_by_block,
    make_folds,
    shuffle_labels_within_date,
)


TARGET_OBJECTIVES = (
    "lgb_regression",
    "lgb_lambdarank",
    "lgb_top_decile",
    "logit_top_decile",
)

DT_SCORE_FIELDS = (
    "structure__double_divergence_present",
    "structure__double_divergence_trend_score",
    "structure__triple_history_available",
)


@dataclass(frozen=True)
class TargetAlignmentConfig:
    relevance_grades: int = 10
    top_fraction: float = 0.10
    top_ns: tuple[int, ...] = (5, 10, 20)
    quantile_fractions: tuple[float, ...] = (0.10, 0.20)
    costs_bps: tuple[int, ...] = (20, 40, 60)
    primary_cost_bps: int = 40
    block_bootstrap_days: int = 10
    bootstrap_iterations: int = 500
    placebo_repeats: int = 20
    logistic_c: float = 0.10


def dt_score_feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    blocks = feature_columns_by_block(frame)
    base = sorted(set(blocks["X"] + blocks["D"] + blocks["T"]))
    missing = sorted(set(DT_SCORE_FIELDS) - set(frame.columns))
    if missing:
        raise ValueError(f"DT_SCORE fields are missing: {missing}")
    return {
        "DT_BASE": base,
        "DT_SCORE": sorted(set(base) | set(DT_SCORE_FIELDS)),
    }


def within_date_relevance(
    labels: pd.Series,
    dates: pd.Series,
    *,
    grades: int = 10,
) -> pd.Series:
    """Map continuous returns to causal training-only within-date relevance grades."""
    if grades < 2:
        raise ValueError("grades must be at least 2")
    ranks = labels.groupby(dates, sort=False).rank(method="first")
    sizes = dates.groupby(dates, sort=False).transform("size")
    relevance = np.floor((ranks - 1) * grades / sizes).clip(upper=grades - 1)
    return relevance.astype("int16")


def within_date_top_fraction(
    labels: pd.Series,
    dates: pd.Series,
    *,
    fraction: float = 0.10,
) -> pd.Series:
    """Mark exactly ceil(fraction * n) highest-return events within each date."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")
    descending_rank = labels.groupby(dates, sort=False).rank(
        method="first", ascending=False
    )
    sizes = dates.groupby(dates, sort=False).transform("size")
    cutoff = np.ceil(sizes * fraction).clip(lower=1)
    return descending_rank.le(cutoff).astype("int8")


def balanced_daily_class_weight(target: pd.Series, dates: pd.Series) -> np.ndarray:
    """Give every date equal total weight and positives/negatives equal weight per date."""
    output = pd.Series(index=target.index, dtype="float64")
    for _, index in dates.groupby(dates, sort=False).groups.items():
        positions = list(index)
        values = target.loc[positions]
        positive = values.eq(1)
        negative = ~positive
        if positive.any() and negative.any():
            output.loc[values.index[positive]] = 0.5 / int(positive.sum())
            output.loc[values.index[negative]] = 0.5 / int(negative.sum())
        else:
            output.loc[positions] = 1.0 / len(positions)
    return (output / output.mean()).to_numpy(dtype="float64")


def _safe_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.loc[:, columns].copy()
    for column in columns:
        if pd.api.types.is_bool_dtype(out[column]):
            out[column] = out[column].astype("float32")
        elif not pd.api.types.is_numeric_dtype(out[column]):
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _lgb_common(config: ModelConfig, seed: int) -> dict[str, Any]:
    return {
        "n_estimators": config.lgb_num_boost_round,
        "learning_rate": config.lgb_learning_rate,
        "num_leaves": config.lgb_num_leaves,
        "max_depth": config.lgb_max_depth,
        "min_child_samples": config.lgb_min_child_samples,
        "colsample_bytree": 1.0,
        "subsample": config.lgb_bagging_fraction,
        "subsample_freq": 1,
        "reg_lambda": config.lgb_reg_lambda,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }


def _fit_objective(
    objective: str,
    train_x: pd.DataFrame,
    train_returns: pd.Series,
    train_dates: pd.Series,
    test_x: pd.DataFrame,
    *,
    model_config: ModelConfig,
    alignment_config: TargetAlignmentConfig,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    common = _lgb_common(model_config, seed)
    if objective == "lgb_regression":
        model = lgb.LGBMRegressor(objective="regression_l2", **common)
        weight = daily_equal_sample_weight(train_dates)
        model.fit(train_x, train_returns, sample_weight=weight)
        prediction = model.predict(test_x)
        importance = model.booster_.feature_importance(importance_type="gain")
        return prediction, pd.DataFrame({"feature": train_x.columns, "importance": importance})

    if objective == "lgb_lambdarank":
        order = np.argsort(train_dates.to_numpy(), kind="stable")
        ordered_x = train_x.iloc[order]
        ordered_returns = train_returns.iloc[order]
        ordered_dates = train_dates.iloc[order]
        relevance = within_date_relevance(
            ordered_returns,
            ordered_dates,
            grades=alignment_config.relevance_grades,
        )
        group = ordered_dates.groupby(ordered_dates, sort=False).size().to_numpy()
        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            label_gain=list(range(alignment_config.relevance_grades)),
            lambdarank_truncation_level=max(alignment_config.top_ns),
            **common,
        )
        model.fit(
            ordered_x,
            relevance,
            group=group,
            sample_weight=daily_equal_sample_weight(ordered_dates),
        )
        prediction = model.predict(test_x)
        importance = model.booster_.feature_importance(importance_type="gain")
        return prediction, pd.DataFrame({"feature": train_x.columns, "importance": importance})

    target = within_date_top_fraction(
        train_returns,
        train_dates,
        fraction=alignment_config.top_fraction,
    )
    weight = balanced_daily_class_weight(target, train_dates)
    if objective == "lgb_top_decile":
        model = lgb.LGBMClassifier(objective="binary", **common)
        model.fit(train_x, target, sample_weight=weight)
        prediction = model.predict_proba(test_x)[:, 1]
        importance = model.booster_.feature_importance(importance_type="gain")
        return prediction, pd.DataFrame({"feature": train_x.columns, "importance": importance})
    if objective == "logit_top_decile":
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=alignment_config.logistic_c,
                    max_iter=2_000,
                    solver="lbfgs",
                ),
            ),
        ])
        pipeline.fit(train_x, target, model__sample_weight=weight)
        prediction = pipeline.predict_proba(test_x)[:, 1]
        names = pipeline.named_steps["imputer"].get_feature_names_out(train_x.columns)
        coefficients = pipeline.named_steps["model"].coef_[0]
        return prediction, pd.DataFrame({"feature": names, "importance": coefficients})
    raise KeyError(f"Unknown target objective: {objective}")


def run_target_alignment_oof(
    events: pd.DataFrame,
    *,
    scope: str,
    periods: Sequence[TestPeriod],
    model_config: ModelConfig,
    alignment_config: TargetAlignmentConfig,
    objectives: Sequence[str] = TARGET_OBJECTIVES,
    run_placebo: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = events.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.normalize()
    data["label_available_date"] = pd.to_datetime(
        data["label_available_date"]
    ).dt.normalize()
    feature_sets = dt_score_feature_sets(data)
    folds = make_folds(data, periods)
    predictions: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []

    for period in periods:
        test_start = pd.Timestamp(period.test_start)
        train_mask = (
            data["trade_date"].lt(test_start)
            & data["label_available_date"].notna()
            & data["label_available_date"].lt(test_start)
        )
        test_mask = data["trade_date"].between(period.test_start, period.test_end)
        if not train_mask.any() or not test_mask.any():
            continue
        train = data.loc[train_mask].copy()
        test = data.loc[test_mask].copy()
        true_returns = train["label__industry_excess_10d"].astype("float64")
        jobs: list[tuple[str, int | None]] = [(objective, None) for objective in objectives]
        if run_placebo:
            jobs.extend(
                (objective, repeat)
                for objective in objectives
                for repeat in range(alignment_config.placebo_repeats)
            )
        for objective, repeat in jobs:
            train_returns = (
                shuffle_labels_within_date(
                    true_returns,
                    train["trade_date"],
                    seed=20260718 + 100 * int(repeat) + period.fold_id,
                )
                if repeat is not None
                else true_returns
            )
            for arm, features in feature_sets.items():
                observed = [column for column in features if train[column].notna().any()]
                train_x = _safe_numeric(train, observed)
                test_x = _safe_numeric(test, observed)
                prediction, importance = _fit_objective(
                    objective,
                    train_x,
                    train_returns,
                    train["trade_date"],
                    test_x,
                    model_config=model_config,
                    alignment_config=alignment_config,
                    seed=model_config.random_seed + 1_000 * period.fold_id,
                )
                output = test.loc[:, [
                    "event_id",
                    "trade_date",
                    "ts_code",
                    "industry_l1_code",
                    "label__industry_excess_10d",
                ]].copy()
                output["scope"] = scope
                output["fold_id"] = period.fold_id
                output["objective"] = objective
                output["arm"] = arm
                output["placebo_repeat"] = repeat
                output["is_placebo"] = repeat is not None
                output["score"] = prediction
                predictions.append(output)

                importance = importance.copy()
                importance["scope"] = scope
                importance["fold_id"] = period.fold_id
                importance["objective"] = objective
                importance["arm"] = arm
                importance["placebo_repeat"] = repeat
                importance["is_placebo"] = repeat is not None
                importances.append(importance)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    duplicate = prediction_frame.duplicated([
        "scope",
        "fold_id",
        "objective",
        "arm",
        "placebo_repeat",
        "event_id",
    ]).sum()
    if duplicate:
        raise AssertionError(f"Duplicate target-alignment OOF predictions: {duplicate}")
    return prediction_frame, pd.concat(importances, ignore_index=True), folds


def _rank_ic(frame: pd.DataFrame) -> float:
    clean = frame.loc[:, ["score", "label__industry_excess_10d"]].dropna()
    if len(clean) < 5 or clean["score"].nunique() < 2:
        return np.nan
    return float(spearmanr(clean["score"], clean["label__industry_excess_10d"]).statistic)


def _portfolio_record(
    *,
    portfolio: str,
    gross: float,
    all_mean: float,
    selected_n: float,
    cost_bps: int,
    cost_legs: int = 1,
) -> dict[str, Any]:
    cost = cost_legs * cost_bps / 10_000.0
    return {
        "portfolio": portfolio,
        "selected_n": selected_n,
        "cost_bps": cost_bps,
        "gross": gross,
        "net": gross - cost,
        "minus_all": gross - all_mean if cost_legs == 1 else gross,
    }


def build_target_aligned_daily_evaluation(
    predictions: pd.DataFrame,
    *,
    config: TargetAlignmentConfig = TargetAlignmentConfig(),
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    keys = [
        "scope",
        "objective",
        "arm",
        "is_placebo",
        "placebo_repeat",
        "fold_id",
        "trade_date",
    ]
    for values, group in predictions.groupby(keys, sort=False, dropna=False):
        daily = group.dropna(subset=["score", "label__industry_excess_10d"]).sort_values(
            "score", ascending=False
        )
        if daily.empty:
            continue
        y = daily["label__industry_excess_10d"].to_numpy(float)
        all_mean = float(np.mean(y))
        rank_ic = _rank_ic(daily)
        portfolio_rows: list[dict[str, Any]] = []
        for top_n in config.top_ns:
            n = min(top_n, len(daily))
            portfolio_rows.extend(
                _portfolio_record(
                    portfolio=f"top_{top_n}",
                    gross=float(np.mean(y[:n])),
                    all_mean=all_mean,
                    selected_n=n,
                    cost_bps=cost,
                )
                for cost in config.costs_bps
            )
        for fraction in config.quantile_fractions:
            n = min(max(1, int(np.ceil(len(daily) * fraction))), len(daily))
            name = int(round(100 * fraction))
            portfolio_rows.extend(
                _portfolio_record(
                    portfolio=f"top_{name}pct",
                    gross=float(np.mean(y[:n])),
                    all_mean=all_mean,
                    selected_n=n,
                    cost_bps=cost,
                )
                for cost in config.costs_bps
            )

        ranks = daily["score"].rank(method="average", pct=True).to_numpy(float)
        long_weights = ranks / ranks.sum()
        rank_long = float(np.dot(long_weights, y))
        centered = ranks - ranks.mean()
        positive = np.clip(centered, 0, None)
        negative = np.clip(-centered, 0, None)
        if positive.sum() > 0 and negative.sum() > 0:
            long_short = float(
                np.dot(positive / positive.sum(), y)
                - np.dot(negative / negative.sum(), y)
            )
        else:
            long_short = np.nan
        for cost in config.costs_bps:
            portfolio_rows.append(_portfolio_record(
                portfolio="rank_weighted_long",
                gross=rank_long,
                all_mean=all_mean,
                selected_n=float(len(daily)),
                cost_bps=cost,
            ))
            portfolio_rows.append(_portfolio_record(
                portfolio="rank_weighted_ls",
                gross=long_short,
                all_mean=all_mean,
                selected_n=float(len(daily)),
                cost_bps=cost,
                cost_legs=2,
            ))

        identity = dict(zip(keys, values))
        for row in portfolio_rows:
            records.append({
                **identity,
                "event_count": int(len(daily)),
                "rank_ic": rank_ic,
                "all_gross": all_mean,
                **row,
            })
    return pd.DataFrame(records)


def _block_bootstrap_mean(
    values: pd.Series,
    *,
    block_days: int,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    clean = values.dropna().to_numpy(float)
    if len(clean) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    block = min(max(1, block_days), len(clean))
    starts = np.arange(0, len(clean) - block + 1)
    count = int(np.ceil(len(clean) / block))
    means = np.empty(iterations)
    for iteration in range(iterations):
        sample_starts = rng.choice(starts, size=count, replace=True)
        sample = np.concatenate([clean[start : start + block] for start in sample_starts])
        means[iteration] = sample[: len(clean)].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_target_alignment(
    daily: pd.DataFrame,
    *,
    config: TargetAlignmentConfig = TargetAlignmentConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = daily.loc[daily["cost_bps"].eq(config.primary_cost_bps)].copy()
    group_keys = ["scope", "objective", "arm", "portfolio", "is_placebo", "placebo_repeat"]
    summary = (
        primary.groupby(group_keys, as_index=False, dropna=False)
        .agg(
            day_count=("trade_date", "nunique"),
            fold_count=("fold_id", "nunique"),
            rank_ic_mean=("rank_ic", "mean"),
            gross_mean=("gross", "mean"),
            net_mean=("net", "mean"),
            minus_all_mean=("minus_all", "mean"),
        )
    )

    pair_records: list[dict[str, Any]] = []
    metrics = ("rank_ic", "gross", "net", "minus_all")
    pair_keys = ["fold_id", "trade_date"]
    compare_keys = ["scope", "objective", "portfolio", "is_placebo", "placebo_repeat"]
    for values, group in primary.groupby(compare_keys, sort=False, dropna=False):
        left = group.loc[group["arm"].eq("DT_SCORE")].set_index(pair_keys)
        right = group.loc[group["arm"].eq("DT_BASE")].set_index(pair_keys)
        common = left.index.intersection(right.index)
        if common.empty:
            continue
        identity = dict(zip(compare_keys, values))
        for metric in metrics:
            delta = (left.loc[common, metric] - right.loc[common, metric]).dropna()
            low, high = (np.nan, np.nan)
            if not bool(identity["is_placebo"]):
                low, high = _block_bootstrap_mean(
                    delta.reset_index(drop=True),
                    block_days=config.block_bootstrap_days,
                    iterations=config.bootstrap_iterations,
                    seed=20260718 + len(pair_records),
                )
            pair_records.append({
                **identity,
                "metric": metric,
                "mean_delta": float(delta.mean()),
                "ci_low": low,
                "ci_high": high,
                "day_count": int(len(delta)),
                "positive_day_ratio": float(delta.gt(0).mean()),
            })
    comparisons = pd.DataFrame(pair_records)
    placebo_records: list[dict[str, Any]] = []
    placebo_keys = ["scope", "objective", "portfolio", "metric"]
    for values, group in comparisons.groupby(placebo_keys, sort=False):
        actual = group.loc[~group["is_placebo"]]
        placebo = group.loc[group["is_placebo"], "mean_delta"].dropna().to_numpy(float)
        if actual.empty or not len(placebo):
            continue
        actual_row = actual.iloc[0]
        actual_delta = float(actual_row["mean_delta"])
        placebo_records.append({
            **dict(zip(placebo_keys, values)),
            "actual_delta": actual_delta,
            "actual_ci_low": float(actual_row["ci_low"]),
            "actual_ci_high": float(actual_row["ci_high"]),
            "placebo_count": int(len(placebo)),
            "placebo_mean": float(placebo.mean()),
            "placebo_std": float(placebo.std(ddof=1)) if len(placebo) > 1 else 0.0,
            "placebo_min": float(placebo.min()),
            "placebo_max": float(placebo.max()),
            "empirical_percentile": float(
                (1 + np.sum(placebo <= actual_delta)) / (len(placebo) + 1)
            ),
            "one_sided_p_positive": float(
                (1 + np.sum(placebo >= actual_delta)) / (len(placebo) + 1)
            ),
            "two_sided_p": float(
                (1 + np.sum(np.abs(placebo) >= abs(actual_delta)))
                / (len(placebo) + 1)
            ),
        })
    return summary, pd.DataFrame(placebo_records)


def aggregate_target_importance(importance: pd.DataFrame) -> pd.DataFrame:
    actual = importance.loc[~importance["is_placebo"]].copy()
    return (
        actual.groupby(["scope", "objective", "arm", "feature"], as_index=False)
        .agg(
            importance_mean=("importance", "mean"),
            importance_abs_mean=("importance", lambda values: float(np.abs(values).mean())),
            folds=("fold_id", "nunique"),
        )
        .sort_values(
            ["scope", "objective", "arm", "importance_abs_mean"],
            ascending=[True, True, True, False],
        )
        .reset_index(drop=True)
    )
