from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from factor_forge.research.concept_portfolio_backtest import TargetBuildResult, cap_and_redistribute
from factor_forge.research.narrative_rotation import (
    NarrativeRules,
    _core_stocks,
    _deduplicate,
    attach_rotation_signals,
)
from factor_forge.research.concept_rotation_alpha import repair_partial_member_snapshots


FEATURE_COLUMNS = [
    "concept_return_1d", "concept_return_5d", "concept_return_20d", "concept_return_60d",
    "return_acceleration_5_20", "return_reversal_1_5", "rs_20d", "rs_momentum_5d",
    "breadth_float", "breadth_delta_5d", "common_breadth_delta_5d",
    "common_breadth_delta_smooth5", "breadth_delta_rank", "common_delta_rank",
    "membership_churn_5d", "member_match_coverage", "log_member_count",
    "log_concept_amount", "concept_amount_ratio", "rotation_amount_rank",
    "concept_age_days", "leading_age", "market_return_5d", "market_return_20d",
    "market_return_60d", "market_breadth", "market_breadth_delta_5d",
    "small_minus_large_10d", "core_return20_mean", "core_return20_std", "core_log_adv20",
    "rrg_leading", "rrg_improving", "rrg_weakening", "rrg_lagging",
    "regime_repair", "regime_overheat", "regime_retreat", "regime_divergence",
]


@dataclass(frozen=True)
class ConceptMLRules:
    horizon: int = 5
    candidate_concepts: int = 20
    selected_concepts: int = 3
    stocks_per_concept: int = 3
    minimum_core_stocks: int = 2
    label_roundtrip_cost_bps: float = 20.0
    minimum_train_days: int = 60
    validation_days: int = 15
    test_days: int = 15
    blend_model_weight: float = 0.20
    maximum_stock_weight: float = 0.10
    minimum_adv20_cny: float = 20_000_000.0
    jaccard_limit: float = 0.80
    minimum_daily_candidates: int = 3


@dataclass(frozen=True)
class ConceptMLFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def load_dc_snapshot_roots(
    roots: Iterable[str | Path], *, trade_dates: Iterable[pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and de-duplicate adjacent PIT snapshot roots.

    Reading monthly files individually tolerates empty endpoint months whose Parquet
    null schema cannot be unified by Arrow's directory reader.
    """
    allowed = None if trade_dates is None else {
        pd.Timestamp(date).strftime("%Y%m%d") for date in trade_dates
    }
    index_frames: list[pd.DataFrame] = []
    member_frames: list[pd.DataFrame] = []
    for raw_root in roots:
        root = Path(raw_root)
        monthly = []
        for path in sorted((root / "index_monthly").glob("*.parquet")):
            frame = pd.read_parquet(path)
            if not frame.empty:
                monthly.append(frame)
        if not monthly:
            continue
        index = pd.concat(monthly, ignore_index=True)
        concept_type = index.groupby("idx_type", observed=True)["ts_code"].nunique().idxmax()
        index = index.loc[index["idx_type"].eq(concept_type)].copy()
        if allowed is not None:
            index = index.loc[index["trade_date"].astype(str).isin(allowed)]
        active_dates = index["trade_date"].astype(str).unique().tolist()
        concept_codes = frozenset(index["ts_code"].dropna().astype(str).unique())
        if not active_dates:
            continue
        table = pads.dataset(root / "members_by_concept", format="parquet").to_table(
            columns=["trade_date", "ts_code", "con_code"],
            filter=pads.field("trade_date").isin(active_dates),
        )
        members = table.to_pandas(categories=["ts_code", "con_code"])
        members = members.loc[members["ts_code"].astype(str).isin(concept_codes)]
        index_frames.append(index)
        member_frames.append(members)
    if not index_frames or not member_frames:
        raise ValueError("snapshot roots produced no concept data")
    index = pd.concat(index_frames, ignore_index=True).drop_duplicates(
        ["trade_date", "ts_code"], keep="last"
    ).rename(columns={"ts_code": "concept_code", "name": "concept_name"})
    concept_categories = sorted({
        str(value) for frame in member_frames for value in frame["ts_code"].cat.categories
    })
    stock_categories = sorted({
        str(value) for frame in member_frames for value in frame["con_code"].cat.categories
    })
    for frame in member_frames:
        frame["ts_code"] = frame["ts_code"].astype(
            pd.CategoricalDtype(concept_categories)
        )
        frame["con_code"] = frame["con_code"].astype(
            pd.CategoricalDtype(stock_categories)
        )
    members = pd.concat(member_frames, ignore_index=True).drop_duplicates(
        ["trade_date", "ts_code", "con_code"], keep="last"
    ).rename(columns={"ts_code": "concept_code", "con_code": "ts_code"})
    index["trade_date"] = pd.to_datetime(index["trade_date"].astype(str))
    members["trade_date"] = pd.to_datetime(members["trade_date"].astype(str))
    members, repaired_dates = repair_partial_member_snapshots(members)
    names = index.sort_values("trade_date").drop_duplicates("concept_code", keep="last")[[
        "concept_code", "concept_name",
    ]]
    index = members[["trade_date", "concept_code"]].drop_duplicates().merge(
        names, on="concept_code", how="left", validate="many_to_one"
    )
    index.attrs["repaired_member_dates"] = repaired_dates
    members.attrs["repaired_member_dates"] = repaired_dates
    return (
        index.sort_values(["trade_date", "concept_code"]).reset_index(drop=True),
        members.sort_values(["trade_date", "concept_code", "ts_code"]).reset_index(drop=True),
    )


def attach_concept_ml_features(features: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    result = attach_rotation_signals(features)
    result = result.sort_values(["concept_code", "trade_date"]).copy()
    result["return_acceleration_5_20"] = result["concept_return_5d"] - result["concept_return_20d"] / 4
    result["return_reversal_1_5"] = result["concept_return_1d"] - result["concept_return_5d"] / 5
    result["log_member_count"] = np.log1p(result["matched_member_count"].clip(lower=0))
    result["log_concept_amount"] = np.log1p(result["concept_amount"].clip(lower=0))
    result["leading_age"] = result.groupby("concept_code", sort=False)["rrg_quadrant"].transform(
        _consecutive_leading_age
    )
    regime_columns = [
        "trade_date", "regime", "market_breadth", "breadth_delta_5d", "small_minus_large_10d",
    ]
    available = [column for column in regime_columns if column in regimes]
    market = regimes[available].drop_duplicates("trade_date").rename(
        columns={"breadth_delta_5d": "market_breadth_delta_5d"}
    )
    result = result.merge(market, on="trade_date", how="left", validate="many_to_one")
    for quadrant in ("leading", "improving", "weakening", "lagging"):
        result[f"rrg_{quadrant}"] = result["rrg_quadrant"].eq(quadrant).astype(np.int8)
    for regime in ("repair", "overheat", "retreat", "divergence"):
        result[f"regime_{regime}"] = result.get("regime", pd.Series(index=result.index, dtype=str)).eq(regime).astype(np.int8)
    return result.sort_values(["trade_date", "concept_code"]).reset_index(drop=True)


def build_concept_ml_dataset(
    features: pd.DataFrame,
    members: pd.DataFrame,
    stock_panel: pd.DataFrame,
    regimes: pd.DataFrame,
    *,
    candidate_mode: str = "momentum",
    rules: ConceptMLRules = ConceptMLRules(),
) -> pd.DataFrame:
    data = attach_concept_ml_features(features, regimes)
    eligible = concept_candidate_mask(data, candidate_mode)
    candidates = data.loc[eligible].sort_values(
        ["trade_date", "rotation_momentum_score", "concept_code"],
        ascending=[True, False, True],
    ).groupby("trade_date", observed=True).head(rules.candidate_concepts).copy()

    stocks = stock_panel.sort_values(["ts_code", "trade_date"]).copy()
    grouped = stocks.groupby("ts_code", sort=False)
    stocks["ml_entry_open"] = grouped["adj_open"].shift(-1)
    stocks["ml_exit_open"] = grouped["adj_open"].shift(-(rules.horizon + 1))
    stocks["ml_forward_return"] = stocks["ml_exit_open"] / stocks["ml_entry_open"] - 1
    stock_days = {
        pd.Timestamp(date): day.set_index("ts_code")
        for date, day in stocks.loc[stocks["trade_date"].isin(candidates["trade_date"].unique())].groupby("trade_date", observed=True)
    }
    member_days = {
        pd.Timestamp(date): day[["concept_code", "ts_code"]]
        for date, day in members.loc[members["trade_date"].isin(candidates["trade_date"].unique())].groupby("trade_date", observed=True)
    }
    narrative_rules = NarrativeRules(
        holding_days=rules.horizon,
        concepts=rules.selected_concepts,
        stocks_per_concept=rules.stocks_per_concept,
        maximum_stock_weight=rules.maximum_stock_weight,
        minimum_adv20_cny=rules.minimum_adv20_cny,
        jaccard_limit=rules.jaccard_limit,
    )
    basket_rows: list[dict] = []
    for row in candidates.itertuples(index=False):
        date = pd.Timestamp(row.trade_date)
        day_stocks = stock_days.get(date)
        day_members = member_days.get(date)
        if day_stocks is None or day_members is None:
            continue
        concept = pd.Series(row._asdict())
        picks = _core_stocks(str(row.concept_code), concept, day_members, day_stocks, narrative_rules)
        picked = day_stocks.loc[day_stocks.index.intersection(picks)].copy()
        returns = pd.to_numeric(picked.get("ml_forward_return"), errors="coerce").dropna()
        return20 = pd.to_numeric(picked.get("stock_return_20d"), errors="coerce")
        adv20 = pd.to_numeric(picked.get("amount_ma20"), errors="coerce")
        basket_rows.append({
            "trade_date": date,
            "concept_code": str(row.concept_code),
            "basket_stocks": "|".join(picks),
            "basket_stock_count": len(picks),
            "core_return20_mean": float(return20.mean()) if return20.notna().any() else np.nan,
            "core_return20_std": float(return20.std(ddof=0)) if return20.notna().any() else np.nan,
            "core_log_adv20": float(np.log1p(adv20.clip(lower=0)).mean()) if adv20.notna().any() else np.nan,
            "basket_forward_gross_5d": float(returns.mean()) if len(returns) >= rules.minimum_core_stocks else np.nan,
        })
    baskets = pd.DataFrame(basket_rows)
    result = candidates.merge(
        baskets, on=["trade_date", "concept_code"], how="left", validate="one_to_one"
    )
    result["basket_forward_net_5d"] = (
        result["basket_forward_gross_5d"] - rules.label_roundtrip_cost_bps / 10_000
    )
    result["label_available_date"] = _label_available_dates(
        result["trade_date"], stocks["trade_date"].unique(), rules.horizon
    )
    result["relevance"] = result.groupby("trade_date", observed=True)["basket_forward_net_5d"].transform(
        _relevance_labels
    )
    result["candidate_mode"] = candidate_mode
    usable = result["basket_stock_count"].ge(rules.minimum_core_stocks)
    usable_counts = usable.groupby(result["trade_date"], observed=True).transform("sum")
    result = result.loc[usable_counts.ge(rules.minimum_daily_candidates)].copy()
    return result.sort_values(["trade_date", "concept_code"]).reset_index(drop=True)


def concept_candidate_mask(data: pd.DataFrame, candidate_mode: str) -> pd.Series:
    modes = {"momentum", "leading", "leading_breadth"}
    if candidate_mode not in modes:
        raise KeyError(f"unknown candidate mode: {candidate_mode}")
    eligible = data["eligible_concept"].fillna(False) & data["rotation_momentum_score"].notna()
    if candidate_mode in {"leading", "leading_breadth"}:
        eligible &= data["rrg_quadrant"].eq("leading") & data["rotation_rs_rank"].ge(0.85)
    if candidate_mode == "leading_breadth":
        eligible &= data["breadth_float"].gt(0.50) & data["common_delta_rank"].ge(0.70)
    return eligible


def build_expanding_ml_folds(
    dates: Iterable[pd.Timestamp], *, rules: ConceptMLRules = ConceptMLRules(),
) -> list[ConceptMLFold]:
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(list(dates)).unique()))
    embargo = rules.horizon + 1
    first_test = rules.minimum_train_days + rules.validation_days + 2 * embargo
    folds: list[ConceptMLFold] = []
    test_start = first_test
    fold = 0
    while test_start < len(calendar):
        test_end = min(test_start + rules.test_days - 1, len(calendar) - 1)
        valid_end = test_start - embargo - 1
        valid_start = valid_end - rules.validation_days + 1
        train_end = valid_start - embargo - 1
        if train_end + 1 >= rules.minimum_train_days:
            folds.append(ConceptMLFold(
                fold=fold, train_start=calendar[0], train_end=calendar[train_end],
                valid_start=calendar[valid_start], valid_end=calendar[valid_end],
                test_start=calendar[test_start], test_end=calendar[test_end],
            ))
            fold += 1
        test_start += rules.test_days
    return folds


def fit_walk_forward_rankers(
    dataset: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    rules: ConceptMLRules = ConceptMLRules(),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[ConceptMLFold]]:
    features = list(feature_columns or FEATURE_COLUMNS)
    missing = sorted(set(features) - set(dataset.columns))
    if missing:
        raise ValueError(f"missing ML features: {missing}")
    dates = dataset.loc[dataset["basket_stock_count"].ge(rules.minimum_core_stocks), "trade_date"].unique()
    folds = build_expanding_ml_folds(dates, rules=rules)
    predictions: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    for fold in folds:
        train = _model_sample(dataset, fold.train_start, fold.train_end, rules.minimum_core_stocks)
        valid = _model_sample(dataset, fold.valid_start, fold.valid_end, rules.minimum_core_stocks)
        test = dataset.loc[dataset["trade_date"].between(fold.test_start, fold.test_end)].copy()
        if train.empty or valid.empty or test.empty:
            continue
        if train["label_available_date"].ge(fold.valid_start).any():
            raise ValueError(f"fold {fold.fold} has immature training labels")
        if valid["label_available_date"].ge(fold.test_start).any():
            raise ValueError(f"fold {fold.fold} has immature validation labels")
        model = _ranker(seed + fold.fold)
        model.fit(
            train[features], train["relevance"].astype(int), group=_groups(train),
            eval_set=[(valid[features], valid["relevance"].astype(int))],
            eval_group=[_groups(valid)], eval_at=[3],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        placebo_train = train.copy()
        placebo_valid = valid.copy()
        placebo_train["relevance"] = _shuffle_within_date(placebo_train, seed + 10_000 + fold.fold)
        placebo_valid["relevance"] = _shuffle_within_date(placebo_valid, seed + 20_000 + fold.fold)
        placebo = _ranker(seed + 30_000 + fold.fold)
        placebo.fit(
            placebo_train[features], placebo_train["relevance"].astype(int), group=_groups(placebo_train),
            eval_set=[(placebo_valid[features], placebo_valid["relevance"].astype(int))],
            eval_group=[_groups(placebo_valid)], eval_at=[3],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        output = test[[
            "trade_date", "concept_code", "concept_name", "basket_stocks", "basket_stock_count",
            "rotation_momentum_score", "basket_forward_gross_5d", "basket_forward_net_5d",
        ]].copy()
        output["lgb_score"] = model.predict(test[features], num_iteration=model.best_iteration_)
        output["placebo_score"] = placebo.predict(test[features], num_iteration=placebo.best_iteration_)
        output["fold"] = fold.fold
        output["model_rank"] = output.groupby("trade_date", observed=True)["lgb_score"].rank(pct=True)
        output["placebo_rank"] = output.groupby("trade_date", observed=True)["placebo_score"].rank(pct=True)
        output["momentum_rank"] = output.groupby("trade_date", observed=True)["rotation_momentum_score"].rank(pct=True)
        output["blend_rank"] = (
            (1 - rules.blend_model_weight) * output["momentum_rank"]
            + rules.blend_model_weight * output["model_rank"]
        )
        predictions.append(output)
        importances.append(pd.DataFrame({
            "fold": fold.fold, "feature": features,
            "gain": model.booster_.feature_importance(importance_type="gain"),
            "split": model.booster_.feature_importance(importance_type="split"),
        }))
    return (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.concat(importances, ignore_index=True) if importances else pd.DataFrame(),
        folds,
    )


def build_ml_rotation_targets(
    predictions: pd.DataFrame,
    members: pd.DataFrame,
    regimes: pd.DataFrame,
    stock_panel: pd.DataFrame,
    *,
    variant: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    offset: int,
    rules: ConceptMLRules = ConceptMLRules(),
) -> TargetBuildResult:
    score_columns = {
        "momentum_baseline": "momentum_rank",
        "lgbm_direct": "model_rank",
        "lgbm_blend_20": "blend_rank",
        "label_shuffle_placebo": "placebo_rank",
        "M0_momentum": "momentum_rank",
        "M1_leading_momentum": "momentum_rank",
        "M2_leading_lgbm": "blend_rank",
        "M3_leading_breadth_momentum": "momentum_rank",
        "M4_leading_breadth_lgbm": "blend_rank",
    }
    if variant not in score_columns:
        raise KeyError(variant)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    calendar = pd.Index(sorted(stock_panel.loc[stock_panel["trade_date"].between(start, end), "trade_date"].unique()))
    if not 0 <= offset < rules.horizon:
        raise ValueError("offset outside holding period")
    days = {pd.Timestamp(date): day for date, day in predictions.groupby("trade_date", observed=True)}
    member_days = {
        pd.Timestamp(date): day[["concept_code", "ts_code"]]
        for date, day in members.loc[members["trade_date"].isin(predictions["trade_date"].unique())].groupby("trade_date", observed=True)
    }
    regime_map = regimes.set_index("trade_date")["regime"].to_dict()
    rows: list[dict] = []
    selections: list[dict] = []
    entry_dates: list[pd.Timestamp] = []
    exit_dates: list[pd.Timestamp] = []
    narrative_rules = NarrativeRules(
        holding_days=rules.horizon, concepts=rules.selected_concepts,
        stocks_per_concept=rules.stocks_per_concept,
        maximum_stock_weight=rules.maximum_stock_weight,
        minimum_adv20_cny=rules.minimum_adv20_cny, jaccard_limit=rules.jaccard_limit,
    )
    score_column = score_columns[variant]
    for position in range(offset, len(calendar), rules.horizon):
        entry_position = position + 1
        exit_position = entry_position + rules.horizon
        if exit_position >= len(calendar):
            break
        date = pd.Timestamp(calendar[position])
        entry_date, exit_date = pd.Timestamp(calendar[entry_position]), pd.Timestamp(calendar[exit_position])
        entry_dates.append(entry_date)
        exit_dates.append(exit_date)
        day = days.get(date)
        if day is None:
            continue
        day_members = member_days.get(date, pd.DataFrame(columns=["concept_code", "ts_code"]))
        candidates = [
            (str(row.concept_code), "ml", float(getattr(row, score_column)))
            for row in day.sort_values(score_column, ascending=False).itertuples(index=False)
            if np.isfinite(getattr(row, score_column))
        ]
        selected = _deduplicate(candidates, day_members, narrative_rules)
        regime = str(regime_map.get(date, "neutral"))
        exposure = 0.60 if regime == "overheat" else 0.30 if regime == "retreat" else 0.90
        contributions: list[dict] = []
        total_score = sum(max(score, 0.01) for _, _, score in selected)
        concept_counts = day_members.groupby("ts_code", observed=True)["concept_code"].nunique()
        for code, _, score in selected:
            item = day.loc[day["concept_code"].astype(str).eq(code)].iloc[0]
            picks = [stock for stock in str(item["basket_stocks"]).split("|") if stock and stock != "nan"]
            if not picks:
                continue
            concept_weight = max(score, 0.01) / total_score
            for stock in picks:
                contributions.append({
                    "ts_code": stock, "concept_code": code,
                    "raw_weight": concept_weight / len(picks) / math.sqrt(max(concept_counts.get(stock, 1), 1)),
                })
            selections.append({
                "variant": variant, "signal_date": date, "entry_date": entry_date,
                "concept_code": code, "concept_name": item["concept_name"],
                "score": score, "regime": regime, "stocks": len(picks), "fold": int(item["fold"]),
            })
        if contributions:
            target = pd.DataFrame(contributions).groupby("ts_code", observed=True).agg(
                raw_weight=("raw_weight", "sum"), supporting_concepts=("concept_code", "nunique")
            ).reset_index()
            target["target_weight"] = cap_and_redistribute(
                target["raw_weight"], min(rules.maximum_stock_weight / exposure, 1.0)
            ).to_numpy() * exposure
            for target_row in target.itertuples(index=False):
                rows.append({
                    "variant": variant, "signal_date": date, "entry_date": entry_date,
                    "ts_code": target_row.ts_code, "target_weight": target_row.target_weight,
                    "supporting_concepts": target_row.supporting_concepts, "regime": regime,
                })
    targets = pd.DataFrame(rows)
    if targets.empty:
        targets = pd.DataFrame(columns=[
            "variant", "signal_date", "entry_date", "ts_code", "target_weight",
            "supporting_concepts", "regime",
        ])
    return TargetBuildResult(targets, pd.DataFrame(selections), sorted(set(entry_dates + exit_dates)), sorted(set(entry_dates)))


def prediction_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score in ("momentum_rank", "model_rank", "blend_rank", "placebo_rank"):
        sample = predictions[["trade_date", score, "basket_forward_net_5d"]].dropna()
        daily_ic = sample.groupby("trade_date", observed=True).apply(
            lambda day: day[score].corr(day["basket_forward_net_5d"], method="spearman")
            if len(day) >= 10 and day[score].nunique() > 1 else np.nan,
            include_groups=False,
        )
        top = sample.sort_values(["trade_date", score], ascending=[True, False]).groupby("trade_date", observed=True).head(3)
        rows.append({
            "score": score, "dates": int(sample["trade_date"].nunique()),
            "rank_ic": float(daily_ic.mean()), "top3_mean_net_label": float(top["basket_forward_net_5d"].mean()),
            "positive_ic_share": float(daily_ic.gt(0).mean()),
        })
    return pd.DataFrame(rows)


def _ranker(seed: int) -> lgb.LGBMRanker:
    return lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=600, learning_rate=0.03,
        num_leaves=7, max_depth=3, min_child_samples=50, subsample=0.80,
        subsample_freq=1, colsample_bytree=0.80, reg_alpha=1.0, reg_lambda=5.0,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )


def _model_sample(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, minimum_stocks: int) -> pd.DataFrame:
    return data.loc[
        data["trade_date"].between(start, end)
        & data["basket_stock_count"].ge(minimum_stocks)
        & data["relevance"].notna()
        & data["label_available_date"].notna()
    ].sort_values(["trade_date", "concept_code"]).copy()


def _groups(frame: pd.DataFrame) -> list[int]:
    return frame.groupby("trade_date", sort=True, observed=True).size().astype(int).tolist()


def _shuffle_within_date(frame: pd.DataFrame, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    output = pd.Series(index=frame.index, dtype=float)
    for _, indices in frame.groupby("trade_date", sort=True).groups.items():
        output.loc[indices] = rng.permutation(frame.loc[indices, "relevance"].to_numpy())
    return output


def _label_available_dates(signal_dates: pd.Series, calendar: Iterable[pd.Timestamp], horizon: int) -> pd.Series:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(calendar)).unique()))
    ordinal = pd.Series(np.arange(len(dates)), index=dates)
    positions = pd.to_datetime(signal_dates).map(ordinal) + horizon + 1
    output = pd.Series(pd.NaT, index=signal_dates.index, dtype="datetime64[ns]")
    valid = positions.notna() & positions.lt(len(dates))
    output.loc[valid] = dates[positions.loc[valid].astype(int)].to_numpy()
    return output


def _relevance_labels(values: pd.Series) -> pd.Series:
    valid = values.notna()
    output = pd.Series(np.nan, index=values.index)
    if valid.sum() >= 5:
        percentile = values.loc[valid].rank(method="first", pct=True)
        output.loc[valid] = np.minimum((percentile * 5).apply(np.ceil).astype(int) - 1, 4)
    return output


def _consecutive_leading_age(values: pd.Series) -> pd.Series:
    leading = values.eq("leading")
    groups = (~leading).cumsum()
    return leading.groupby(groups).cumsum().astype(float)
