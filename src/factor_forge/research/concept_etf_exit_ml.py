from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from factor_forge.research.concept_etf_shadow import (
    CASH,
    _aggregate_sleeve_nav,
    _serialize_weights,
    _turnover,
    simulate_staggered_sleeves,
    staggered_target_weights,
)


EXIT_FEATURES = [
    "etf_return_1d",
    "etf_return_3d",
    "etf_return_5d",
    "etf_momentum_20d",
    "etf_momentum_60d",
    "etf_momentum_acceleration",
    "volatility_20d",
    "amount_ratio_5_20",
    "concept_return_1d",
    "concept_return_5d",
    "concept_return_20d",
    "breadth_float",
    "breadth_equal_minus_float",
    "common_breadth_delta_smooth5",
    "breadth_acceleration_5d",
    "breadth_negative_fraction_3d",
    "common_delta_rank",
    "rs_momentum_5d",
    "membership_churn_5d",
    "rrg_weakening",
    "rrg_lagging",
    "holding_age",
    "days_remaining",
]


@dataclass(frozen=True)
class ExitMLRules:
    holding_days: int = 5
    minimum_train_days: int = 60
    validation_days: int = 15
    test_days: int = 15
    embargo_days: int = 6
    minimum_train_rows: int = 250
    minimum_validation_rows: int = 40
    consecutive_negative_days: int = 2
    early_exit_cost_bps: float = 20.0
    tail_horizon_days: int = 2
    tail_loss_threshold: float = -0.03
    tail_quantile: float = 0.10
    seed: int = 42


@dataclass(frozen=True)
class ExitMLFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def enrich_exit_features(
    panel: pd.DataFrame,
    concept_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result = panel.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    if concept_features is not None:
        concepts = concept_features.copy()
        concepts["trade_date"] = pd.to_datetime(concepts["trade_date"])
        wanted = [
            "trade_date", "concept_code", "concept_return_5d", "concept_return_20d",
            "breadth_equal_raw", "breadth_float_raw", "membership_churn_5d",
        ]
        available = [column for column in wanted if column in concepts]
        additions = [column for column in available if column not in result or column in {"trade_date", "concept_code"}]
        result = result.merge(
            concepts[additions].drop_duplicates(["trade_date", "concept_code"]),
            on=["trade_date", "concept_code"], how="left", validate="many_to_one",
        )
    result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    grouped = result.groupby("ts_code", sort=False)
    result["etf_return_3d"] = grouped["adj_close"].pct_change(3, fill_method=None)
    result["etf_return_5d"] = grouped["adj_close"].pct_change(5, fill_method=None)
    result["etf_momentum_acceleration"] = result["etf_return_5d"] - result["etf_momentum_20d"] / 4
    amount_5 = grouped["amount_cny"].transform(lambda values: values.rolling(5, min_periods=5).mean())
    amount_20 = grouped["amount_cny"].transform(lambda values: values.rolling(20, min_periods=18).mean())
    result["amount_ratio_5_20"] = amount_5 / amount_20 - 1
    breadth_grouped = result.groupby("ts_code", sort=False)["common_breadth_delta_smooth5"]
    result["breadth_acceleration_5d"] = (
        result["common_breadth_delta_smooth5"] - breadth_grouped.shift(5)
    )
    negative = result["common_breadth_delta_smooth5"].lt(0).astype(float)
    result["breadth_negative_fraction_3d"] = negative.groupby(result["ts_code"], sort=False).transform(
        lambda values: values.rolling(3, min_periods=2).mean()
    )
    result["breadth_negative_2d"] = (
        result["common_breadth_delta_smooth5"].lt(0)
        & breadth_grouped.shift(1).lt(0)
    )
    if {"breadth_equal_raw", "breadth_float_raw"}.issubset(result):
        result["breadth_equal_minus_float"] = result["breadth_equal_raw"] - result["breadth_float_raw"]
    else:
        result["breadth_equal_minus_float"] = np.nan
    if "membership_churn_5d" not in result:
        result["membership_churn_5d"] = np.nan
    if "concept_return_5d" not in result:
        result["concept_return_5d"] = grouped["concept_return_1d"].transform(
            lambda values: (1 + values).rolling(5, min_periods=5).apply(np.prod, raw=True) - 1
        )
    if "concept_return_20d" not in result:
        result["concept_return_20d"] = grouped["concept_return_1d"].transform(
            lambda values: (1 + values).rolling(20, min_periods=18).apply(np.prod, raw=True) - 1
        )
    result["rrg_weakening"] = result["rrg_quadrant"].eq("weakening").astype(int)
    result["rrg_lagging"] = result["rrg_quadrant"].eq("lagging").astype(int)
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def build_fixed_r4_held_states(
    panel: pd.DataFrame,
    *,
    start: str,
    end: str,
    rules: ExitMLRules = ExitMLRules(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, sleeves, _ = simulate_staggered_sleeves(
        panel, "R4_rank_buffer", start=start, end=end,
        roundtrip_cost_bps=rules.early_exit_cost_bps,
    )
    prices = panel.pivot(index="trade_date", columns="ts_code", values="adj_open").sort_index()
    price_calendar = pd.Index(prices.index)
    price_positions = {pd.Timestamp(date): index for index, date in enumerate(price_calendar)}
    lookup = panel.set_index(["trade_date", "ts_code"])
    rows: list[dict] = []
    for sleeve_id, sleeve in sleeves.groupby("sleeve", observed=True):
        ordered = sleeve.sort_values("holding_date").copy()
        ordered["period_id"] = ordered["is_rebalance"].cumsum()
        for period_id, period in ordered.loc[ordered["period_id"].gt(0)].groupby("period_id"):
            period = period.sort_values("holding_date")
            if len(period) != rules.holding_days or not bool(period.iloc[0]["is_rebalance"]):
                continue
            scheduled_exit = pd.Timestamp(period.iloc[-1]["return_date"])
            for position, item in enumerate(period.iloc[:-1].itertuples(index=False)):
                state_date = pd.Timestamp(item.holding_date)
                next_open = pd.Timestamp(item.return_date)
                holding_age = position + 1
                weights = parse_weights(item.target_weights)
                for code, weight in weights.items():
                    if code == CASH or weight <= 0 or (state_date, code) not in lookup.index:
                        continue
                    next_price = prices.at[next_open, code]
                    exit_price = prices.at[scheduled_exit, code]
                    if not np.isfinite(next_price) or not np.isfinite(exit_price):
                        continue
                    remaining_return = float(exit_price / next_price - 1)
                    next_position = price_positions[next_open]
                    scheduled_position = price_positions[scheduled_exit]
                    tail_end_position = min(
                        next_position + rules.tail_horizon_days, scheduled_position,
                    )
                    tail_dates = price_calendar[next_position + 1:tail_end_position + 1]
                    if len(tail_dates) == 0:
                        continue
                    tail_returns = prices.loc[tail_dates, code] / next_price - 1
                    tail_worst_return = float(tail_returns.min())
                    tail_label_available_date = pd.Timestamp(tail_dates[-1])
                    row = lookup.loc[(state_date, code)].to_dict()
                    row.update({
                        "state_date": state_date,
                        "ts_code": code,
                        "sleeve": int(sleeve_id),
                        "period_id": int(period_id),
                        "holding_age": holding_age,
                        "days_remaining": rules.holding_days - holding_age,
                        "position_weight": float(weight),
                        "next_open_date": next_open,
                        "label_available_date": scheduled_exit,
                        "remaining_open_return": remaining_return,
                        "exit_advantage": -remaining_return - rules.early_exit_cost_bps / 10_000,
                        "tail_worst_open_return": tail_worst_return,
                        "tail_event": tail_worst_return <= rules.tail_loss_threshold,
                        "tail_label_available_date": tail_label_available_date,
                    })
                    rows.append(row)
    states = pd.DataFrame(rows)
    return states.sort_values(["state_date", "sleeve", "ts_code"]).reset_index(drop=True), sleeves


def build_exit_folds(
    dates: Iterable[pd.Timestamp],
    *,
    rules: ExitMLRules = ExitMLRules(),
) -> list[ExitMLFold]:
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(list(dates)).unique()))
    first_test = rules.minimum_train_days + rules.validation_days + 2 * rules.embargo_days
    folds: list[ExitMLFold] = []
    test_start = first_test
    fold_id = 0
    while test_start < len(calendar):
        test_end = min(test_start + rules.test_days - 1, len(calendar) - 1)
        valid_end = test_start - rules.embargo_days - 1
        valid_start = valid_end - rules.validation_days + 1
        train_end = valid_start - rules.embargo_days - 1
        if train_end + 1 >= rules.minimum_train_days:
            folds.append(ExitMLFold(
                fold=fold_id,
                train_start=calendar[0],
                train_end=calendar[train_end],
                valid_start=calendar[valid_start],
                valid_end=calendar[valid_end],
                test_start=calendar[test_start],
                test_end=calendar[test_end],
            ))
            fold_id += 1
        test_start += rules.test_days
    return folds


def fit_exit_walk_forward(
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
        raise ValueError(f"missing exit features: {missing}")
    calendar = sorted(panel.loc[panel["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end)), "trade_date"].unique())
    folds = build_exit_folds(calendar, rules=rules)
    prediction_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    for fold in folds:
        train = states.loc[
            states["state_date"].between(fold.train_start, fold.train_end)
            & states["label_available_date"].lt(fold.valid_start)
        ].copy()
        valid = states.loc[
            states["state_date"].between(fold.valid_start, fold.valid_end)
            & states["label_available_date"].lt(fold.test_start)
        ].copy()
        if len(train) < rules.minimum_train_rows or len(valid) < rules.minimum_validation_rows:
            continue
        if train["label_available_date"].ge(fold.valid_start).any():
            raise ValueError(f"fold {fold.fold} has immature training labels")
        if valid["label_available_date"].ge(fold.test_start).any():
            raise ValueError(f"fold {fold.fold} has immature validation labels")
        model = _exit_regressor(rules.seed + fold.fold)
        model.fit(train[features], train["exit_advantage"])
        placebo_train = train.copy()
        rng = np.random.default_rng(rules.seed + 10_000 + fold.fold)
        placebo_train["exit_advantage"] = rng.permutation(placebo_train["exit_advantage"].to_numpy())
        placebo = _exit_regressor(rules.seed + 20_000 + fold.fold)
        placebo.fit(placebo_train[features], placebo_train["exit_advantage"])
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
        output["predicted_exit_advantage"] = model.predict(grid[features])
        output["placebo_exit_advantage"] = placebo.predict(grid[features])
        output["fold"] = fold.fold
        prediction_parts.append(output)
        permutation = permutation_importance(
            model, valid[features], valid["exit_advantage"],
            scoring="neg_mean_absolute_error", n_repeats=5,
            random_state=rules.seed + 30_000 + fold.fold,
        )
        importance_parts.append(pd.DataFrame({
            "fold": fold.fold,
            "feature": features,
            "gain": np.clip(permutation.importances_mean, 0, None),
            "split": np.nan,
        }))
        audit_rows.append({
            "fold": fold.fold,
            "train_start": fold.train_start, "train_end": fold.train_end,
            "valid_start": fold.valid_start, "valid_end": fold.valid_end,
            "test_start": fold.test_start, "test_end": fold.test_end,
            "train_rows": len(train), "valid_rows": len(valid), "test_grid_rows": len(grid),
            "train_label_available_max": train["label_available_date"].max(),
            "valid_label_available_max": valid["label_available_date"].max(),
            "best_iteration": model.n_iter_,
        })
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    importance = pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame()
    return predictions, importance, pd.DataFrame(audit_rows), folds


def simulate_r4_exit_sleeves(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    policy: str,
    start: str,
    end: str,
    roundtrip_cost_bps: float,
    rules: ExitMLRules = ExitMLRules(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if policy not in {
        "X0_fixed", "X1_rule", "X2_ml", "X3_placebo",
        "T0_fixed", "T1_tail_ml", "T2_tail_placebo",
    }:
        raise KeyError(policy)
    frame = panel.sort_values(["ts_code", "trade_date"]).copy()
    full_calendar = pd.Index(sorted(frame["trade_date"].unique()))
    active = pd.Index([date for date in full_calendar if pd.Timestamp(start) <= date <= pd.Timestamp(end)])
    next_date = {
        pd.Timestamp(full_calendar[index]): pd.Timestamp(full_calendar[index + 1])
        for index in range(len(full_calendar) - 1)
    }
    previous_date = {
        pd.Timestamp(full_calendar[index]): pd.Timestamp(full_calendar[index - 1])
        for index in range(1, len(full_calendar))
    }
    prices = frame.pivot(index="trade_date", columns="ts_code", values="adj_open").sort_index()
    feature_lookup = frame.set_index(["trade_date", "ts_code"])
    prediction_lookup = (
        predictions.set_index(["state_date", "ts_code", "holding_age"])
        if not predictions.empty else pd.DataFrame()
    )
    sleeve_parts: list[pd.DataFrame] = []
    exit_rows: list[dict] = []
    for sleeve_id in range(rules.holding_days):
        signals = active[sleeve_id::rules.holding_days]
        executions = {
            pd.Timestamp(next_date[pd.Timestamp(signal)]): pd.Timestamp(signal)
            for signal in signals
            if pd.Timestamp(signal) in next_date and next_date[pd.Timestamp(signal)] <= pd.Timestamp(end)
        }
        execution_dates = sorted(executions)
        following_execution = {
            date: execution_dates[index + 1]
            for index, date in enumerate(execution_dates[:-1])
        }
        pretrade = {CASH: 1.0}
        sleeve_nav = 1.0
        holding_age = 0
        negative_streaks: dict[str, int] = {}
        scheduled_exit: pd.Timestamp | None = None
        rows = []
        for date_value in active:
            date = pd.Timestamp(date_value)
            if date not in next_date or next_date[date] > pd.Timestamp(end):
                continue
            signal_date = executions.get(date)
            target = dict(pretrade)
            turnover = 0.0
            exited_codes: list[str] = []
            exit_signal_date: pd.Timestamp | None = None
            if signal_date is not None:
                previous_holdings = {
                    code for code, weight in pretrade.items() if code != CASH and weight > 0
                }
                signal_day = frame.loc[frame["trade_date"].eq(signal_date)]
                target = staggered_target_weights(
                    signal_day, "R4_rank_buffer", previous_holdings=previous_holdings,
                )
                turnover = _turnover(pretrade, target)
                holding_age = 0
                negative_streaks = {}
                scheduled_exit = following_execution.get(date)
            elif holding_age > 0 and policy not in {"X0_fixed", "T0_fixed"}:
                exit_signal_date = previous_date.get(date)
                if exit_signal_date is not None:
                    for code, weight in list(target.items()):
                        if code == CASH or weight <= 0:
                            continue
                        negative = _negative_exit_signal(
                            policy, code, exit_signal_date, holding_age,
                            feature_lookup, prediction_lookup, rules.tail_loss_threshold,
                        )
                        if policy in {"X2_ml", "X3_placebo"}:
                            negative_streaks[code] = negative_streaks.get(code, 0) + 1 if negative else 0
                            should_exit = negative_streaks[code] >= rules.consecutive_negative_days
                        else:
                            should_exit = negative
                        if should_exit:
                            exited_codes.append(code)
                    if exited_codes:
                        before_exit = dict(target)
                        cash = target.get(CASH, 0.0)
                        for code in exited_codes:
                            cash += target.pop(code, 0.0)
                        target[CASH] = cash
                        turnover = _turnover(before_exit, target)
                        for code in exited_codes:
                            remaining_return = np.nan
                            if scheduled_exit is not None and scheduled_exit in prices.index:
                                remaining_return = float(prices.at[scheduled_exit, code] / prices.at[date, code] - 1)
                            tail_worst = _realized_tail_return(
                                prices, full_calendar, date, code, scheduled_exit,
                                rules.tail_horizon_days,
                            )
                            exit_rows.append({
                                "policy": policy, "sleeve": sleeve_id,
                                "exit_date": date, "exit_signal_date": exit_signal_date,
                                "scheduled_exit_date": scheduled_exit, "ts_code": code,
                                "holding_age": holding_age,
                                "remaining_return_after_exit": remaining_return,
                                "false_exit": bool(np.isfinite(remaining_return) and remaining_return > 0),
                                "tail_worst_return_after_exit": tail_worst,
                                "tail_event_after_exit": bool(
                                    np.isfinite(tail_worst) and tail_worst <= rules.tail_loss_threshold
                                ),
                            })
            following = next_date[date]
            returns = (prices.loc[following] / prices.loc[date] - 1).to_dict()
            missing_prices = [
                code for code, weight in target.items()
                if code != CASH and weight > 0 and not np.isfinite(returns.get(code, np.nan))
            ]
            if missing_prices:
                raise ValueError(f"missing ETF open return on {date.date()}: {missing_prices}")
            gross = sum(
                weight * returns.get(code, 0.0)
                for code, weight in target.items() if code != CASH and weight > 0
            )
            cost = turnover * roundtrip_cost_bps / 10_000
            net = gross - cost
            nav_before = sleeve_nav
            sleeve_nav *= 1 + net
            rows.append({
                "return_date": following, "holding_date": date, "signal_date": signal_date,
                "portfolio": policy, "variant": policy, "sleeve": sleeve_id,
                "gross_return": gross, "net_return": net, "net_nav": sleeve_nav,
                "nav_before": nav_before, "turnover": turnover, "cost_drag": cost,
                "cash_weight": target.get(CASH, 0.0), "is_rebalance": signal_date is not None,
                "is_early_exit": bool(exited_codes), "exited_codes": ",".join(exited_codes),
                "exit_signal_date": exit_signal_date,
                "target_weights": _serialize_weights(target),
            })
            denominator = 1 + gross
            pretrade = {
                code: weight * (1 + returns.get(code, 0.0)) / denominator
                for code, weight in target.items()
            }
            holding_age += 1
        sleeve_parts.append(pd.DataFrame(rows))
    sleeve_daily = pd.concat(sleeve_parts, ignore_index=True)
    aggregate = _aggregate_sleeve_nav(sleeve_daily, policy)
    exits = pd.DataFrame(exit_rows)
    return aggregate, sleeve_daily, exits


def parse_weights(value: str) -> dict[str, float]:
    return {
        item.split(":", 1)[0]: float(item.split(":", 1)[1])
        for item in str(value).split(";") if item
    }


def _exit_regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.05, max_iter=180,
        max_leaf_nodes=7, max_depth=3, min_samples_leaf=30,
        l2_regularization=5.0, early_stopping=False,
        random_state=seed,
    )


def _negative_exit_signal(
    policy: str,
    code: str,
    signal_date: pd.Timestamp,
    holding_age: int,
    feature_lookup: pd.DataFrame,
    prediction_lookup: pd.DataFrame,
    tail_loss_threshold: float,
) -> bool:
    if policy == "X1_rule":
        key = (signal_date, code)
        return bool(key in feature_lookup.index and feature_lookup.loc[key, "breadth_negative_2d"])
    if prediction_lookup.empty:
        return False
    key = (signal_date, code, holding_age)
    if key not in prediction_lookup.index:
        return False
    if policy in {"T1_tail_ml", "T2_tail_placebo"}:
        column = "predicted_tail_return" if policy == "T1_tail_ml" else "placebo_tail_return"
        value = prediction_lookup.loc[key, column]
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        return bool(np.isfinite(value) and value <= tail_loss_threshold)
    column = "predicted_exit_advantage" if policy == "X2_ml" else "placebo_exit_advantage"
    value = prediction_lookup.loc[key, column]
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    return bool(np.isfinite(value) and value > 0)


def _realized_tail_return(
    prices: pd.DataFrame,
    calendar: pd.Index,
    exit_date: pd.Timestamp,
    code: str,
    scheduled_exit: pd.Timestamp | None,
    horizon: int,
) -> float:
    positions = {pd.Timestamp(date): index for index, date in enumerate(calendar)}
    if exit_date not in positions:
        return np.nan
    start = positions[exit_date]
    end = min(start + horizon, len(calendar) - 1)
    if scheduled_exit is not None and scheduled_exit in positions:
        end = min(end, positions[scheduled_exit])
    if end <= start:
        return np.nan
    future = calendar[start + 1:end + 1]
    returns = prices.loc[future, code] / prices.at[exit_date, code] - 1
    return float(returns.min())
