from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def attach_label_available_date(
    episodes: pd.DataFrame,
    calendar: Iterable[pd.Timestamp],
    *,
    horizon: int,
) -> pd.DataFrame:
    """Attach the first date on which a T+1-open to T+h+1-open label is known."""
    result = episodes.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(calendar)).unique()))
    ordinal = pd.Series(np.arange(len(dates), dtype=int), index=dates)
    event_ordinal = result["trade_date"].map(ordinal)
    available_ordinal = event_ordinal + horizon + 1
    result["label_available_date"] = pd.NaT
    valid = event_ordinal.notna() & available_ordinal.lt(len(dates))
    result.loc[valid, "label_available_date"] = dates[
        available_ordinal.loc[valid].astype(int).to_numpy()
    ].to_numpy()
    return result


def build_pit_recent_structure_features(
    episodes: pd.DataFrame,
    *,
    calendar: Iterable[pd.Timestamp],
    windows: list[int],
    target_col: str = "target",
    factor_columns: list[str] | None = None,
    minimum_mature_events: int = 20,
) -> tuple[pd.DataFrame, list[str]]:
    """Build template efficacy features using only labels mature by each signal date.

    The returned table has one row per input episode. A row's own target can never
    enter its features unless that label was already available at the signal time,
    which is impossible under the T+1-open label contract.
    """
    required = {"trade_date", "template_id", "label_available_date", target_col}
    missing = required - set(episodes.columns)
    if missing:
        raise ValueError(f"recent structure input is missing columns: {sorted(missing)}")
    if sorted(set(windows)) != windows:
        raise ValueError("windows must be unique and increasing")
    factors = list(factor_columns or [])
    unknown = set(factors) - set(episodes.columns)
    if unknown:
        raise ValueError(f"unknown factor columns: {sorted(unknown)}")

    data = episodes.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data["label_available_date"] = pd.to_datetime(data["label_available_date"])
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(calendar)).unique()))
    feature_names: list[str] = []
    for window in windows:
        feature_names.extend([
            f"template_mature_count_{window}",
            f"template_target_mean_{window}",
            f"template_target_std_{window}",
            f"template_hit_rate_{window}",
        ])
        feature_names.extend(f"factor_ic_{name}_{window}" for name in factors)
    feature_names.extend(["template_target_velocity", "template_direction_stability"])

    mature = data.loc[
        data["label_available_date"].notna()
        & pd.to_numeric(data[target_col], errors="coerce").notna()
    ].copy()
    mature[target_col] = pd.to_numeric(mature[target_col], errors="coerce")
    states = []
    for template in data["template_id"].astype(str).drop_duplicates():
        history = mature.loc[mature["template_id"].astype(str).eq(template)].copy()
        history["_target_sq"] = history[target_col].pow(2)
        history["_positive"] = history[target_col].gt(0).astype(float)
        daily = history.groupby("label_available_date").agg(
            count=(target_col, "count"), target_sum=(target_col, "sum"),
            target_sq_sum=("_target_sq", "sum"), positive_sum=("_positive", "sum"),
        ).reindex(dates, fill_value=0.0)
        for factor in factors:
            values = []
            for date, group in history.groupby("label_available_date", sort=False):
                paired = group[[factor, target_col]].apply(pd.to_numeric, errors="coerce").dropna()
                correlation = (
                    paired[factor].corr(paired[target_col], method="spearman")
                    if len(paired) >= 3 and paired[factor].nunique() > 1
                    and paired[target_col].nunique() > 1 else np.nan
                )
                values.append((pd.Timestamp(date), correlation))
            daily[f"_daily_ic_{factor}"] = pd.Series(dict(values)).reindex(dates)
        state = pd.DataFrame({"trade_date": dates, "template_id": template})
        means = {}
        for window in windows:
            count = daily["count"].rolling(window, min_periods=1).sum()
            total = daily["target_sum"].rolling(window, min_periods=1).sum()
            total_sq = daily["target_sq_sum"].rolling(window, min_periods=1).sum()
            positive = daily["positive_sum"].rolling(window, min_periods=1).sum()
            enough = count.ge(minimum_mature_events)
            mean = (total / count.replace(0, np.nan)).where(enough)
            variance = ((total_sq - total.pow(2) / count.replace(0, np.nan))
                        / (count - 1).replace(0, np.nan))
            state[f"template_mature_count_{window}"] = count.to_numpy(float)
            state[f"template_target_mean_{window}"] = mean.to_numpy(float)
            state[f"template_target_std_{window}"] = np.sqrt(variance.clip(lower=0)).where(enough).to_numpy(float)
            state[f"template_hit_rate_{window}"] = (positive / count.replace(0, np.nan)).where(enough).to_numpy(float)
            means[window] = mean
            for factor in factors:
                rolling_ic = daily[f"_daily_ic_{factor}"].rolling(window, min_periods=1).mean()
                state[f"factor_ic_{factor}_{window}"] = rolling_ic.where(enough).to_numpy(float)
        short, long = windows[0], windows[-1]
        state["template_target_velocity"] = (means[short] - means[long]).to_numpy(float)
        signs = pd.concat([means[window].apply(np.sign) for window in windows], axis=1)
        state["template_direction_stability"] = (
            signs.sum(axis=1).abs() / signs.notna().sum(axis=1).replace(0, np.nan)
        ).to_numpy(float)
        states.append(state)
    structure = pd.concat(states, ignore_index=True) if states else pd.DataFrame(
        columns=["trade_date", "template_id", *feature_names]
    )
    output = data.merge(
        structure, on=["trade_date", "template_id"], how="left", validate="many_to_one"
    )
    return output, feature_names


def build_walk_forward_folds(
    calendar: Iterable[pd.Timestamp],
    *,
    training_days: int,
    validation_days: int,
    test_days: int,
    step_days: int,
    horizon: int,
) -> list[WalkForwardFold]:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(calendar)).unique()))
    embargo = horizon + 1
    first_test = training_days + validation_days + 2 * embargo
    folds: list[WalkForwardFold] = []
    test_start_index = first_test
    fold = 0
    while test_start_index + test_days <= len(dates):
        valid_end_index = test_start_index - embargo - 1
        valid_start_index = valid_end_index - validation_days + 1
        train_end_index = valid_start_index - embargo - 1
        train_start_index = train_end_index - training_days + 1
        if train_start_index < 0:
            test_start_index += step_days
            continue
        folds.append(WalkForwardFold(
            fold=fold,
            train_start=dates[train_start_index], train_end=dates[train_end_index],
            valid_start=dates[valid_start_index], valid_end=dates[valid_end_index],
            test_start=dates[test_start_index], test_end=dates[test_start_index + test_days - 1],
        ))
        fold += 1
        test_start_index += step_days
    return folds


def assert_fold_label_maturity(episodes: pd.DataFrame, fold: WalkForwardFold) -> None:
    """Fail closed when a training or validation label was unavailable at use time."""
    data = episodes.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data["label_available_date"] = pd.to_datetime(data["label_available_date"])
    train = data["trade_date"].between(fold.train_start, fold.train_end)
    valid = data["trade_date"].between(fold.valid_start, fold.valid_end)
    if data.loc[train, "label_available_date"].gt(fold.valid_start).any():
        raise ValueError("training label matures after validation starts")
    if data.loc[valid, "label_available_date"].gt(fold.test_start).any():
        raise ValueError("validation label matures after test starts")
