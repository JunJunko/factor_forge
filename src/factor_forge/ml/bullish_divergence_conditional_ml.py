from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from factor_forge.research.concept_rotation_ml import load_dc_snapshot_roots


ARM_BLOCKS: dict[str, tuple[str, ...]] = {
    "M0": ("X",),
    "M1": ("X", "D", "T"),
    "M2": ("X", "R", "C", "M"),
    "M3": ("X", "D", "T", "R"),
    "M4": ("X", "D", "T", "C"),
    "M5": ("X", "D", "T", "M"),
    "M6": ("X", "D", "T", "R", "C", "M"),
    "M7": ("X", "D", "T", "R", "C", "M", "I"),
}

STRUCTURE_ARM_BLOCKS: dict[str, tuple[str, ...]] = {
    "H0": ("X",),
    "H1": ("X", "D", "T"),
    "H2": ("X", "S"),
    "H3": ("X", "D", "T", "S"),
    "H4": ("X", "D", "T", "R", "C", "M"),
    "H5": ("X", "D", "T", "S", "R", "C", "M"),
}

ALL_ARM_BLOCKS = {**ARM_BLOCKS, **STRUCTURE_ARM_BLOCKS}

BLOCK_PREFIXES: dict[str, tuple[str, ...]] = {
    "X": ("control__",),
    "D": ("div__", "div_v2__"),
    "T": ("touch__",),
    "S": ("structure__",),
    "R": ("regime__",),
    "C": ("industry_rotation__", "concept__"),
    "M": ("momentum__",),
    "I": ("interaction__",),
}

IDENTITY_COLUMNS = [
    "event_id",
    "trade_date",
    "ts_code",
    "industry_l1_code",
    "label__industry_excess_10d",
    "label_available_date",
]


def read_parquet_compat(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    return pd.read_parquet(Path(path), columns=columns)


@dataclass(frozen=True)
class TestPeriod:
    __test__ = False

    fold_id: int
    test_start: str
    test_end: str


@dataclass(frozen=True)
class ModelConfig:
    ridge_alpha: float = 100.0
    lgb_num_boost_round: int = 160
    lgb_learning_rate: float = 0.03
    lgb_num_leaves: int = 15
    lgb_max_depth: int = 4
    lgb_min_child_samples: int = 100
    lgb_feature_fraction: float = 0.8
    lgb_bagging_fraction: float = 0.8
    lgb_reg_lambda: float = 5.0
    random_seed: int = 20260716
    placebo_repeats: int = 5


@dataclass(frozen=True)
class EvaluationConfig:
    top_ns: tuple[int, ...] = (5, 10, 20)
    costs_bps: tuple[int, ...] = (20, 40, 60)
    primary_top_n: int = 10
    primary_cost_bps: int = 40
    block_bootstrap_days: int = 10
    bootstrap_iterations: int = 500


def _ensure_datetime(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce").dt.normalize()
    return out


def _safe_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.loc[:, columns].copy()
    for column in columns:
        if pd.api.types.is_bool_dtype(out[column]):
            out[column] = out[column].astype("float32")
        elif not pd.api.types.is_numeric_dtype(out[column]):
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def feature_columns_by_block(frame: pd.DataFrame) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    for block, prefixes in BLOCK_PREFIXES.items():
        blocks[block] = sorted(
            column
            for column in frame.columns
            if any(column.startswith(prefix) for prefix in prefixes)
        )
    return blocks


def arm_feature_columns(
    frame: pd.DataFrame,
    arm: str,
    *,
    feature_blocks: Mapping[str, Sequence[str]] | None = None,
    arm_blocks: Mapping[str, Sequence[str]] | None = None,
) -> list[str]:
    active_arm_blocks = arm_blocks or ALL_ARM_BLOCKS
    if arm not in active_arm_blocks:
        raise KeyError(f"Unknown arm: {arm}")
    blocks = feature_blocks or feature_columns_by_block(frame)
    columns: list[str] = []
    for block in active_arm_blocks[arm]:
        columns.extend(blocks.get(block, ()))
    # Canonical ordering is part of the model specification.  LightGBM's
    # feature subsampling draws column indices, so identical feature sets in a
    # different block order must not silently produce different seeded fits.
    return sorted(set(columns))


def _select_episode_features(episodes: pd.DataFrame) -> pd.DataFrame:
    keep_prefixes = ("div__", "div_v2__", "touch__", "structure__")
    excluded = {
        "div__a_low",
        "div__b_low",
        "div__a_date",
        "div__b_date",
        "div_v2__q_price",
        "div_v2__q_osc",
        "div_v2__q_structure",
        "div_v2__q_micro",
    }
    keep = [
        column
        for column in episodes.columns
        if column in {"event_id", "trade_date", "ts_code"}
        or (
            column.startswith(keep_prefixes)
            and column not in excluded
            and not column.endswith("_date")
            and not column.endswith("_price")
        )
    ]
    return episodes.loc[:, list(dict.fromkeys(keep))]


def attach_structure_features(
    events: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["event_id", "trade_date", "ts_code"]
    base = _ensure_datetime(events, ["trade_date"])
    source = _ensure_datetime(episodes, ["trade_date"])
    missing_keys = set(keys) - set(source.columns)
    if missing_keys:
        raise ValueError(f"structure episodes are missing keys: {sorted(missing_keys)}")
    structure_columns = sorted(
        column
        for column in source.columns
        if column.startswith("structure__") and not column.endswith("_date")
    )
    if not structure_columns:
        raise ValueError("structure episodes contain no structure__ features")
    source = source.loc[:, [*keys, *structure_columns]].copy()
    if source.duplicated(keys).any():
        raise ValueError("structure episodes contain duplicate event keys")
    existing = [column for column in structure_columns if column in base.columns]
    if existing:
        base = base.drop(columns=existing)
    merged = base.merge(
        source,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator="_structure_merge",
    )
    unmatched = merged["_structure_merge"].ne("both")
    if unmatched.any():
        raise ValueError(
            f"structure episodes failed to match {int(unmatched.sum())} conditional events"
        )
    return merged.drop(columns="_structure_merge").sort_values(
        ["trade_date", "ts_code", "event_id"]
    ).reset_index(drop=True)


def load_base_events(
    episodes_path: str | Path,
    labeled_events_path: str | Path,
) -> pd.DataFrame:
    episodes = _ensure_datetime(
        read_parquet_compat(Path(episodes_path)),
        ["trade_date"],
    )
    labels = _ensure_datetime(
        read_parquet_compat(Path(labeled_events_path)),
        ["trade_date"],
    )
    episode_features = _select_episode_features(episodes)
    control_columns = sorted(
        column for column in labels.columns if column.startswith("control__")
    )
    label_keep = [
        "event_id",
        "trade_date",
        "ts_code",
        "industry_l1_code",
        "label__industry_excess_10d",
        *control_columns,
    ]
    merged = episode_features.merge(
        labels.loc[:, label_keep],
        on=["event_id", "trade_date", "ts_code"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged.dropna(subset=["label__industry_excess_10d"]).copy()
    merged["event_id"] = merged["event_id"].astype(str)
    merged["ts_code"] = merged["ts_code"].astype(str)
    return merged.sort_values(["trade_date", "ts_code", "event_id"]).reset_index(drop=True)


def add_label_availability(
    events: pd.DataFrame,
    trading_dates: Sequence[pd.Timestamp],
    *,
    forward_trading_days: int = 11,
) -> pd.DataFrame:
    calendar = pd.DatetimeIndex(pd.to_datetime(list(trading_dates))).normalize().unique().sort_values()
    positions = pd.Series(np.arange(len(calendar), dtype="int32"), index=calendar)
    out = events.copy()
    ordinal = out["trade_date"].map(positions)
    available_ordinal = ordinal + int(forward_trading_days)
    valid = available_ordinal.notna() & (available_ordinal < len(calendar))
    out["label_available_date"] = pd.NaT
    if valid.any():
        out.loc[valid, "label_available_date"] = calendar[
            available_ordinal.loc[valid].astype(int)
        ].to_numpy()
    out["label_available_date"] = pd.to_datetime(out["label_available_date"]).dt.normalize()
    return out


def build_stock_momentum_features(
    panel_path: str | Path,
    *,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    columns = [
        "trade_date",
        "ts_code",
        "adj_close",
        "adj_high",
        "adj_low",
        "amount_cny",
        "turnover_rate",
        "industry_l1_code",
    ]
    panel = _ensure_datetime(
        read_parquet_compat(Path(panel_path), columns=columns),
        ["trade_date"],
    )
    warmup_start = pd.Timestamp(event_start) - pd.Timedelta(days=140)
    panel = panel.loc[
        panel["trade_date"].between(warmup_start, pd.Timestamp(event_end))
    ].copy()
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    all_trading_dates = pd.DatetimeIndex(panel["trade_date"].dropna().unique()).sort_values()

    grouped = panel.groupby("ts_code", sort=False, group_keys=False)
    close = panel["adj_close"].astype("float64")
    high = panel["adj_high"].astype("float64")
    low = panel["adj_low"].astype("float64")
    prev_close = grouped["adj_close"].shift(1).astype("float64")
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    panel["_tr"] = true_range
    panel["_ret1"] = grouped["adj_close"].pct_change(fill_method=None)

    def pct_change(periods: int) -> pd.Series:
        return grouped["adj_close"].pct_change(periods, fill_method=None)

    features = panel.loc[:, ["trade_date", "ts_code"]].copy()
    for horizon in (1, 3, 5, 10, 20, 60):
        features[f"momentum__ret_{horizon}d"] = pct_change(horizon)
    features["momentum__acceleration_5_20"] = (
        features["momentum__ret_5d"] - features["momentum__ret_20d"] / 4.0
    )
    features["momentum__acceleration_10_60"] = (
        features["momentum__ret_10d"] - features["momentum__ret_60d"] / 6.0
    )

    atr20 = (
        panel.groupby("ts_code", sort=False)["_tr"]
        .rolling(20, min_periods=10)
        .mean()
        .reset_index(level=0, drop=True)
    )
    for window in (5, 20, 60):
        ma = (
            panel.groupby("ts_code", sort=False)["adj_close"]
            .rolling(window, min_periods=max(3, window // 2))
            .mean()
            .reset_index(level=0, drop=True)
        )
        features[f"momentum__distance_ma{window}_atr"] = (close - ma) / atr20.replace(0.0, np.nan)
    rolling_high60 = (
        panel.groupby("ts_code", sort=False)["adj_high"]
        .rolling(60, min_periods=20)
        .max()
        .reset_index(level=0, drop=True)
    )
    features["momentum__distance_high60_atr"] = (
        close - rolling_high60
    ) / atr20.replace(0.0, np.nan)
    features["momentum__drawdown_60d"] = close / rolling_high60.replace(0.0, np.nan) - 1.0

    amount5 = (
        panel.groupby("ts_code", sort=False)["amount_cny"]
        .rolling(5, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    amount20 = (
        panel.groupby("ts_code", sort=False)["amount_cny"]
        .rolling(20, min_periods=10)
        .mean()
        .reset_index(level=0, drop=True)
    )
    turnover5 = (
        panel.groupby("ts_code", sort=False)["turnover_rate"]
        .rolling(5, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    turnover20 = (
        panel.groupby("ts_code", sort=False)["turnover_rate"]
        .rolling(20, min_periods=10)
        .mean()
        .reset_index(level=0, drop=True)
    )
    features["momentum__amount_ratio_5_20"] = amount5 / amount20.replace(0.0, np.nan)
    features["momentum__turnover_delta_5_20"] = turnover5 - turnover20

    industry_keys = panel.loc[:, ["trade_date", "industry_l1_code"]].copy()
    industry_keys["_ret5"] = features["momentum__ret_5d"]
    industry_keys["_ret20"] = features["momentum__ret_20d"]
    industry_mean = industry_keys.groupby(
        ["trade_date", "industry_l1_code"], dropna=False
    )[["_ret5", "_ret20"]].transform("mean")
    features["momentum__industry_excess_5d"] = (
        features["momentum__ret_5d"] - industry_mean["_ret5"]
    )
    features["momentum__industry_excess_20d"] = (
        features["momentum__ret_20d"] - industry_mean["_ret20"]
    )
    features = features.loc[
        features["trade_date"].between(pd.Timestamp(event_start), pd.Timestamp(event_end))
    ].copy()
    return features, all_trading_dates


DEFAULT_REGIME_COLUMNS = [
    "index_ret_20d",
    "index_ret_60d",
    "index_vol_20d",
    "index_drawdown_60d",
    "index_above_ma20",
    "index_above_ma60",
    "up_ratio",
    "up_ratio_ma5",
    "up_ratio_ma20",
    "up_ratio_chg_5d",
    "breadth_thrust",
    "rzmre_ratio",
    "rzye_chg_20d",
    "main_net_ratio",
    "main_net_ratio_ma5",
    "main_net_ratio_sum_20d",
    "erp",
    "pmi",
    "macro_growth_score",
    "macro_uncertainty_score",
]


def load_regime_features(
    timing_path: str | Path,
    *,
    columns: Sequence[str] = DEFAULT_REGIME_COLUMNS,
) -> pd.DataFrame:
    timing = _ensure_datetime(read_parquet_compat(Path(timing_path)), ["trade_date"])
    available = [column for column in columns if column in timing.columns]
    out = timing.loc[:, ["trade_date", *available]].copy()
    return out.rename(columns={column: f"regime__{column}" for column in available})


DEFAULT_INDUSTRY_COLUMNS = [
    "industry_return_1d",
    "industry_return_5d",
    "industry_return_20d",
    "industry_return_60d",
    "breadth_float",
    "breadth_delta_5d",
    "rs_20d",
    "rs_momentum_5d",
    "relative_strength_velocity",
    "industry_amount_share",
    "industry_amount_share_delta_5d",
    "member_count",
    "rotation_momentum_score",
    "leading_age",
    "weakening_age",
    "lagging_age",
    "improving_age",
]


def load_industry_rotation_features(
    industry_path: str | Path,
    *,
    columns: Sequence[str] = DEFAULT_INDUSTRY_COLUMNS,
) -> pd.DataFrame:
    industry = _ensure_datetime(read_parquet_compat(Path(industry_path)), ["trade_date"])
    available = [column for column in columns if column in industry.columns]
    out = industry.loc[:, ["trade_date", "industry_code", *available]].copy()
    out = out.rename(
        columns={
            "industry_code": "industry_l1_code",
            **{column: f"industry_rotation__{column}" for column in available},
        }
    )
    out["industry_l1_code"] = out["industry_l1_code"].astype(str)
    return out


DEFAULT_CONCEPT_COLUMNS = [
    "rs_20d",
    "rs_momentum_5d",
    "relative_strength_velocity",
    "breadth_float",
    "breadth_delta_5d",
    "common_breadth_delta_smooth5",
    "membership_churn_5d",
    "rotation_amount_rank",
    "rotation_momentum_score",
    "leading_age",
    "weakening_age",
    "lagging_age",
    "improving_age",
]


def build_pit_concept_stock_features(
    concept_ml_dataset_path: str | Path,
    snapshot_roots: Sequence[str | Path],
    *,
    event_dates: Sequence[pd.Timestamp],
    columns: Sequence[str] = DEFAULT_CONCEPT_COLUMNS,
) -> pd.DataFrame:
    concept_daily = _ensure_datetime(
        read_parquet_compat(Path(concept_ml_dataset_path)),
        ["trade_date"],
    )
    event_calendar = pd.DatetimeIndex(pd.to_datetime(list(event_dates))).normalize().unique()
    concept_daily = concept_daily.loc[
        concept_daily["trade_date"].isin(event_calendar)
    ].copy()
    if concept_daily.empty:
        return pd.DataFrame(columns=["trade_date", "ts_code", "concept__snapshot_available"])

    concept_id_column = "concept_code" if "concept_code" in concept_daily.columns else "concept_id"
    available = [column for column in columns if column in concept_daily.columns]
    active = concept_daily.loc[:, ["trade_date", concept_id_column, *available]].copy()
    active = active.rename(columns={concept_id_column: "concept_code"})
    active["concept_code"] = active["concept_code"].astype(str)
    active = active.drop_duplicates(["trade_date", "concept_code"], keep="last")

    _, members = load_dc_snapshot_roots(
        [Path(root) for root in snapshot_roots],
        trade_dates=event_calendar,
    )
    members = _ensure_datetime(members, ["trade_date"])
    member_code = "stock_code" if "stock_code" in members.columns else "ts_code"
    members = members.rename(columns={member_code: "ts_code"})
    members["concept_code"] = members["concept_code"].astype(str)
    members["ts_code"] = members["ts_code"].astype(str)
    joined = members.loc[:, ["trade_date", "concept_code", "ts_code"]].merge(
        active,
        on=["trade_date", "concept_code"],
        how="inner",
        validate="many_to_one",
    )
    if joined.empty:
        return pd.DataFrame(columns=["trade_date", "ts_code", "concept__snapshot_available"])

    keys = ["trade_date", "ts_code"]
    counts = joined.groupby(keys, sort=False).size().rename("concept__active_membership_count")
    best = joined.groupby(keys, sort=False)[available].max().add_prefix("concept__best_")
    if "rotation_momentum_score" in joined.columns:
        top3_source = joined.sort_values(
            [*keys, "rotation_momentum_score"],
            ascending=[True, True, False],
        ).groupby(keys, sort=False).head(3)
    else:
        top3_source = joined.groupby(keys, sort=False).head(3)
    top3 = top3_source.groupby(keys, sort=False)[available].mean().add_prefix("concept__top3_")
    aggregated = pd.concat([counts, best, top3], axis=1).reset_index()
    aggregated["concept__snapshot_available"] = 1.0
    return aggregated


def merge_asof_regime(events: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    left = events.sort_values("trade_date").copy()
    right = regime.sort_values("trade_date").copy()
    merged = pd.merge_asof(
        left,
        right,
        on="trade_date",
        direction="backward",
        tolerance=pd.Timedelta(days=7),
    )
    return merged.sort_values(["trade_date", "ts_code", "event_id"]).reset_index(drop=True)


def add_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    def product(name: str, columns: Sequence[str], *, scale: float = 1.0) -> None:
        if all(column in out.columns for column in columns):
            value = pd.Series(scale, index=out.index, dtype="float64")
            for column in columns:
                value = value * pd.to_numeric(out[column], errors="coerce")
            out[name] = value

    score = "div_v2__score" if "div_v2__score" in out.columns else "div__indicator_agreement_count"
    product(
        "interaction__div_score_x_market_drawdown",
        [score, "regime__index_drawdown_60d"],
        scale=0.01,
    )
    product(
        "interaction__div_score_x_breadth_change",
        [score, "regime__up_ratio_chg_5d"],
        scale=0.01,
    )
    product(
        "interaction__div_score_x_industry_rs_momentum",
        [score, "industry_rotation__rs_momentum_5d"],
        scale=0.01,
    )
    product(
        "interaction__div_score_x_industry_breadth",
        [score, "industry_rotation__breadth_delta_5d"],
        scale=0.01,
    )
    product(
        "interaction__div_score_x_stock_acceleration",
        [score, "momentum__acceleration_5_20"],
        scale=0.01,
    )
    product(
        "interaction__touch_support_x_stock_acceleration",
        ["touch__pre_b_count", "momentum__acceleration_5_20"],
    )
    product(
        "interaction__touch_reclaim_x_market_breadth",
        ["touch__last_reclaim_strength_atr", "regime__up_ratio_chg_5d"],
    )
    product(
        "interaction__stock_momentum_x_industry_momentum",
        ["momentum__industry_excess_5d", "industry_rotation__rs_momentum_5d"],
    )
    product(
        "interaction__volume_activation_x_industry_breadth",
        ["momentum__amount_ratio_5_20", "industry_rotation__breadth_delta_5d"],
    )
    concept_momentum = "concept__best_rs_momentum_5d"
    if concept_momentum in out.columns:
        product(
            "interaction__stock_momentum_x_concept_momentum",
            ["momentum__industry_excess_5d", concept_momentum],
        )
        product(
            "interaction__div_score_x_concept_momentum",
            [score, concept_momentum],
            scale=0.01,
        )
    return out


def assemble_conditional_dataset(
    *,
    episodes_path: str | Path,
    labeled_events_path: str | Path,
    panel_path: str | Path,
    timing_path: str | Path,
    industry_path: str | Path,
    concept_ml_dataset_path: str | Path | None = None,
    concept_snapshot_roots: Sequence[str | Path] = (),
) -> pd.DataFrame:
    events = load_base_events(episodes_path, labeled_events_path)
    start = events["trade_date"].min()
    end = events["trade_date"].max()
    momentum, calendar = build_stock_momentum_features(
        panel_path,
        event_start=start,
        event_end=end,
    )
    events = add_label_availability(events, calendar)
    events = events.merge(
        momentum,
        on=["trade_date", "ts_code"],
        how="left",
        validate="many_to_one",
    )
    events = merge_asof_regime(events, load_regime_features(timing_path))
    industry = load_industry_rotation_features(industry_path)
    events["industry_l1_code"] = events["industry_l1_code"].astype(str)
    events = events.merge(
        industry,
        on=["trade_date", "industry_l1_code"],
        how="left",
        validate="many_to_one",
    )
    events["industry_rotation__taxonomy_valid"] = (
        events["trade_date"] >= pd.Timestamp("2021-11-01")
    ).astype("float32")
    industry_feature_columns = [
        column
        for column in events.columns
        if column.startswith("industry_rotation__") and column != "industry_rotation__taxonomy_valid"
    ]
    invalid_industry = events["industry_rotation__taxonomy_valid"].eq(0.0)
    events.loc[invalid_industry, industry_feature_columns] = np.nan

    if concept_ml_dataset_path is not None and concept_snapshot_roots:
        concept_features = build_pit_concept_stock_features(
            concept_ml_dataset_path,
            concept_snapshot_roots,
            event_dates=events["trade_date"].unique(),
        )
        events = events.merge(
            concept_features,
            on=["trade_date", "ts_code"],
            how="left",
            validate="many_to_one",
        )
        concept_calendar = _ensure_datetime(
            read_parquet_compat(
                Path(concept_ml_dataset_path),
                columns=["trade_date"],
            ),
            ["trade_date"],
        )["trade_date"].dropna().unique()
        events["concept__snapshot_available"] = (
            events["trade_date"].isin(concept_calendar)
        ).astype("float32")
        if "concept__active_membership_count" in events.columns:
            available_mask = events["concept__snapshot_available"].eq(1.0)
            events.loc[
                available_mask & events["concept__active_membership_count"].isna(),
                "concept__active_membership_count",
            ] = 0.0
    events = add_interactions(events)
    return events.sort_values(["trade_date", "ts_code", "event_id"]).reset_index(drop=True)


def make_folds(
    events: pd.DataFrame,
    periods: Sequence[TestPeriod],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for period in periods:
        test_start = pd.Timestamp(period.test_start)
        test_end = pd.Timestamp(period.test_end)
        train_mask = (
            events["trade_date"].lt(test_start)
            & events["label_available_date"].notna()
            & events["label_available_date"].lt(test_start)
        )
        test_mask = events["trade_date"].between(test_start, test_end)
        records.append(
            {
                "fold_id": int(period.fold_id),
                "test_start": test_start,
                "test_end": test_end,
                "train_count": int(train_mask.sum()),
                "test_count": int(test_mask.sum()),
                "train_start": events.loc[train_mask, "trade_date"].min(),
                "train_end": events.loc[train_mask, "trade_date"].max(),
                "max_train_label_available_date": events.loc[
                    train_mask, "label_available_date"
                ].max(),
            }
        )
    return pd.DataFrame(records)


def daily_equal_sample_weight(dates: pd.Series) -> np.ndarray:
    counts = dates.groupby(dates).transform("size").astype("float64")
    weights = 1.0 / counts
    return (weights / weights.mean()).to_numpy(dtype="float64")


def shuffle_labels_within_date(
    labels: pd.Series,
    dates: pd.Series,
    *,
    seed: int,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    shuffled = labels.copy()
    for _, index in dates.groupby(dates).groups.items():
        positions = np.asarray(list(index))
        shuffled.loc[positions] = rng.permutation(labels.loc[positions].to_numpy())
    return shuffled


def _fit_ridge(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    train_weight: np.ndarray,
    test_x: pd.DataFrame,
    *,
    alpha: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )
    pipeline.fit(train_x, train_y, model__sample_weight=train_weight)
    prediction = pipeline.predict(test_x)
    feature_names = pipeline.named_steps["imputer"].get_feature_names_out(train_x.columns)
    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": pipeline.named_steps["model"].coef_,
        }
    )
    return prediction, coefficients


def _fit_lightgbm(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    train_weight: np.ndarray,
    test_x: pd.DataFrame,
    *,
    config: ModelConfig,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    model = lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=config.lgb_num_boost_round,
        learning_rate=config.lgb_learning_rate,
        num_leaves=config.lgb_num_leaves,
        max_depth=config.lgb_max_depth,
        min_child_samples=config.lgb_min_child_samples,
        colsample_bytree=config.lgb_feature_fraction,
        subsample=config.lgb_bagging_fraction,
        subsample_freq=1,
        reg_lambda=config.lgb_reg_lambda,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train_x, train_y, sample_weight=train_weight)
    prediction = model.predict(test_x)
    importance = pd.DataFrame(
        {
            "feature": train_x.columns,
            "importance": model.booster_.feature_importance(importance_type="gain"),
        }
    )
    return prediction, importance


def run_oof_predictions(
    events: pd.DataFrame,
    *,
    scope: str,
    periods: Sequence[TestPeriod],
    arms: Sequence[str] = tuple(ARM_BLOCKS),
    algorithms: Sequence[str] = ("ridge", "lightgbm"),
    model_config: ModelConfig = ModelConfig(),
    run_shuffle_placebo: bool = True,
    placebo_arms: Sequence[str] = ("M6", "M7"),
    feature_blocks: Mapping[str, Sequence[str]] | None = None,
    arm_blocks: Mapping[str, Sequence[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blocks = feature_blocks or feature_columns_by_block(events)
    fold_table = make_folds(events, periods)
    predictions: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []

    for period in periods:
        test_start = pd.Timestamp(period.test_start)
        test_end = pd.Timestamp(period.test_end)
        train_mask = (
            events["trade_date"].lt(test_start)
            & events["label_available_date"].notna()
            & events["label_available_date"].lt(test_start)
        )
        test_mask = events["trade_date"].between(test_start, test_end)
        if not train_mask.any() or not test_mask.any():
            continue
        train = events.loc[train_mask].copy()
        test = events.loc[test_mask].copy()
        train_weight = daily_equal_sample_weight(train["trade_date"])
        true_train_y = train["label__industry_excess_10d"].astype("float64")

        jobs: list[tuple[str, str, int | None]] = [
            (arm, algorithm, None) for arm in arms for algorithm in algorithms
        ]
        if run_shuffle_placebo:
            active_placebo_arms = [arm for arm in placebo_arms if arm in arms]
            jobs.extend(
                (placebo_arm, algorithm, repeat)
                for placebo_arm in active_placebo_arms
                for algorithm in algorithms
                for repeat in range(model_config.placebo_repeats)
            )

        for arm, algorithm, placebo_repeat in jobs:
            features = arm_feature_columns(
                events,
                arm,
                feature_blocks=blocks,
                arm_blocks=arm_blocks,
            )
            features = [column for column in features if train[column].notna().any()]
            if not features:
                raise ValueError(
                    f"No observed training features for scope={scope}, fold={period.fold_id}, arm={arm}"
                )
            train_x = _safe_numeric(train, features)
            test_x = _safe_numeric(test, features)
            train_y = (
                shuffle_labels_within_date(
                    true_train_y,
                    train["trade_date"],
                    seed=(
                        model_config.random_seed
                        + 100 * int(placebo_repeat)
                        + period.fold_id
                    ),
                )
                if placebo_repeat is not None
                else true_train_y
            )
            # Keep the stochastic fit identical across ablation arms.  Making
            # the seed depend on feature count confounds incremental features
            # with a different bagging/feature-sampling draw.
            seed = model_config.random_seed + 1000 * period.fold_id
            if algorithm == "ridge":
                prediction, importance = _fit_ridge(
                    train_x,
                    train_y,
                    train_weight,
                    test_x,
                    alpha=model_config.ridge_alpha,
                )
            elif algorithm == "lightgbm":
                prediction, importance = _fit_lightgbm(
                    train_x,
                    train_y,
                    train_weight,
                    test_x,
                    config=model_config,
                    seed=seed,
                )
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            model_name = (
                f"{arm}_shuffle_{int(placebo_repeat):02d}"
                if placebo_repeat is not None
                else arm
            )
            output = test.loc[
                :,
                [
                    "event_id",
                    "trade_date",
                    "ts_code",
                    "industry_l1_code",
                    "label__industry_excess_10d",
                ],
            ].copy()
            output["scope"] = scope
            output["fold_id"] = period.fold_id
            output["arm"] = model_name
            output["algorithm"] = algorithm
            output["score"] = prediction
            predictions.append(output)

            importance = importance.copy()
            importance["scope"] = scope
            importance["fold_id"] = period.fold_id
            importance["arm"] = model_name
            importance["algorithm"] = algorithm
            importances.append(importance)

    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    importance_frame = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    if not prediction_frame.empty:
        duplicate_count = prediction_frame.duplicated(
            ["scope", "fold_id", "arm", "algorithm", "event_id"]
        ).sum()
        if duplicate_count:
            raise AssertionError(f"Duplicate OOF predictions: {duplicate_count}")
    return prediction_frame, importance_frame, fold_table


def _daily_rank_ic(frame: pd.DataFrame) -> float:
    clean = frame.loc[:, ["score", "label__industry_excess_10d"]].dropna()
    if len(clean) < 5 or clean["score"].nunique() < 2:
        return np.nan
    return float(spearmanr(clean["score"], clean["label__industry_excess_10d"]).statistic)


def build_daily_evaluation(
    predictions: pd.DataFrame,
    *,
    config: EvaluationConfig = EvaluationConfig(),
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    group_columns = ["scope", "algorithm", "arm", "fold_id", "trade_date"]
    for keys, daily in predictions.groupby(group_columns, sort=False):
        scope, algorithm, arm, fold_id, trade_date = keys
        daily = daily.dropna(subset=["score", "label__industry_excess_10d"]).sort_values(
            "score", ascending=False
        )
        if daily.empty:
            continue
        all_mean = float(daily["label__industry_excess_10d"].mean())
        rank_ic = _daily_rank_ic(daily)
        for top_n in config.top_ns:
            n = min(int(top_n), len(daily))
            top = daily.head(n)["label__industry_excess_10d"]
            bottom = daily.tail(n)["label__industry_excess_10d"]
            top_gross = float(top.mean())
            bottom_gross = float(bottom.mean())
            for cost_bps in config.costs_bps:
                cost = float(cost_bps) / 10_000.0
                records.append(
                    {
                        "scope": scope,
                        "algorithm": algorithm,
                        "arm": arm,
                        "fold_id": int(fold_id),
                        "trade_date": trade_date,
                        "event_count": len(daily),
                        "top_n": int(top_n),
                        "actual_top_n": n,
                        "cost_bps": int(cost_bps),
                        "rank_ic": rank_ic,
                        "all_gross": all_mean,
                        "top_gross": top_gross,
                        "top_net": top_gross - cost,
                        "top_minus_all": top_gross - all_mean,
                        "top_minus_bottom": top_gross - bottom_gross,
                    }
                )
    return pd.DataFrame(records)


def _block_bootstrap_mean(
    values: pd.Series,
    *,
    block_days: int,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    clean = values.dropna().to_numpy(dtype="float64")
    if len(clean) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(clean)
    block = min(max(1, block_days), n)
    starts = np.arange(0, n - block + 1)
    means = np.empty(iterations, dtype="float64")
    required_blocks = int(np.ceil(n / block))
    for iteration in range(iterations):
        sampled_starts = rng.choice(starts, size=required_blocks, replace=True)
        sample = np.concatenate([clean[start : start + block] for start in sampled_starts])[:n]
        means[iteration] = sample.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_evaluation(
    daily: pd.DataFrame,
    *,
    config: EvaluationConfig = EvaluationConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = [
        "rank_ic",
        "all_gross",
        "top_gross",
        "top_net",
        "top_minus_all",
        "top_minus_bottom",
    ]
    group_columns = ["scope", "algorithm", "arm", "top_n", "cost_bps"]
    records: list[dict[str, Any]] = []
    for keys, group in daily.groupby(group_columns, sort=False):
        record = dict(zip(group_columns, keys))
        record["day_count"] = int(group["trade_date"].nunique())
        record["fold_count"] = int(group["fold_id"].nunique())
        for metric in metric_columns:
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_std"] = float(group[metric].std(ddof=1))
        if (
            record["top_n"] == config.primary_top_n
            and record["cost_bps"] == config.primary_cost_bps
        ):
            for metric in ("rank_ic", "top_net", "top_minus_all", "top_minus_bottom"):
                low, high = _block_bootstrap_mean(
                    group.sort_values("trade_date")[metric],
                    block_days=config.block_bootstrap_days,
                    iterations=config.bootstrap_iterations,
                    seed=20260716 + len(records),
                )
                record[f"{metric}_ci_low"] = low
                record[f"{metric}_ci_high"] = high
        records.append(record)
    summary = pd.DataFrame(records)

    primary = daily.loc[
        daily["top_n"].eq(config.primary_top_n)
        & daily["cost_bps"].eq(config.primary_cost_bps)
    ].copy()
    fold_summary = (
        primary.groupby(["scope", "algorithm", "arm", "fold_id"], as_index=False)
        .agg(
            day_count=("trade_date", "nunique"),
            rank_ic=("rank_ic", "mean"),
            top_net=("top_net", "mean"),
            top_minus_all=("top_minus_all", "mean"),
            top_minus_bottom=("top_minus_bottom", "mean"),
        )
    )
    return summary, fold_summary


def compare_arms(
    daily: pd.DataFrame,
    *,
    config: EvaluationConfig = EvaluationConfig(),
) -> pd.DataFrame:
    primary = daily.loc[
        daily["top_n"].eq(config.primary_top_n)
        & daily["cost_bps"].eq(config.primary_cost_bps)
        & ~daily["arm"].str.contains("_shuffle_", regex=False)
    ].copy()
    comparisons = [
        ("M1", "M0"),
        ("M2", "M0"),
        ("M3", "M1"),
        ("M4", "M1"),
        ("M5", "M1"),
        ("M6", "M1"),
        ("M6", "M2"),
        ("M6", "M0"),
        ("M7", "M6"),
        ("H1", "H0"),
        ("H2", "H0"),
        ("H3", "H1"),
        ("H3", "H2"),
        ("H5", "H4"),
        ("H5", "H3"),
    ]
    records: list[dict[str, Any]] = []
    keys = ["scope", "algorithm", "fold_id", "trade_date"]
    for challenger, baseline in comparisons:
        left = primary.loc[primary["arm"].eq(challenger)].set_index(keys)
        right = primary.loc[primary["arm"].eq(baseline)].set_index(keys)
        common = left.index.intersection(right.index)
        if common.empty:
            continue
        for metric in ("rank_ic", "top_net", "top_minus_all", "top_minus_bottom"):
            delta = left.loc[common, metric] - right.loc[common, metric]
            low, high = _block_bootstrap_mean(
                delta.reset_index(drop=True),
                block_days=config.block_bootstrap_days,
                iterations=config.bootstrap_iterations,
                seed=20260716 + len(records),
            )
            records.append(
                {
                    "challenger": challenger,
                    "baseline": baseline,
                    "metric": metric,
                    "mean_delta": float(delta.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "day_count": int(len(delta)),
                    "positive_day_ratio": float((delta > 0).mean()),
                }
            )
    if not records:
        return pd.DataFrame()
    comparison = pd.DataFrame(records)
    scope_algorithm = primary.loc[:, ["scope", "algorithm"]].drop_duplicates()
    if len(scope_algorithm) == 1:
        comparison.insert(0, "algorithm", scope_algorithm.iloc[0]["algorithm"])
        comparison.insert(0, "scope", scope_algorithm.iloc[0]["scope"])
        return comparison

    all_records: list[pd.DataFrame] = []
    for (scope, algorithm), group in primary.groupby(["scope", "algorithm"], sort=False):
        subset_records: list[dict[str, Any]] = []
        for challenger, baseline in comparisons:
            left = group.loc[group["arm"].eq(challenger)].set_index(["fold_id", "trade_date"])
            right = group.loc[group["arm"].eq(baseline)].set_index(["fold_id", "trade_date"])
            common = left.index.intersection(right.index)
            for metric in ("rank_ic", "top_net", "top_minus_all", "top_minus_bottom"):
                if common.empty:
                    continue
                delta = left.loc[common, metric] - right.loc[common, metric]
                low, high = _block_bootstrap_mean(
                    delta.reset_index(drop=True),
                    block_days=config.block_bootstrap_days,
                    iterations=config.bootstrap_iterations,
                    seed=20260716 + len(subset_records),
                )
                subset_records.append(
                    {
                        "scope": scope,
                        "algorithm": algorithm,
                        "challenger": challenger,
                        "baseline": baseline,
                        "metric": metric,
                        "mean_delta": float(delta.mean()),
                        "ci_low": low,
                        "ci_high": high,
                        "day_count": int(len(delta)),
                        "positive_day_ratio": float((delta > 0).mean()),
                    }
                )
        all_records.append(pd.DataFrame(subset_records))
    return pd.concat(all_records, ignore_index=True)


def compare_paired_placebo_arms(
    daily: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    *,
    config: EvaluationConfig = EvaluationConfig(),
) -> pd.DataFrame:
    """Compare named arm pairs for actual and identically shuffled-label fits.

    A placebo delta uses ``challenger_shuffle_k - baseline_shuffle_k``.  Pairing
    the same repeat removes the baseline model's shuffled-label noise and tests
    the incremental feature block rather than the whole challenger model.
    """
    primary = daily.loc[
        daily["top_n"].eq(config.primary_top_n)
        & daily["cost_bps"].eq(config.primary_cost_bps)
    ].copy()
    metrics = ("rank_ic", "top_net", "top_minus_all", "top_minus_bottom")
    records: list[dict[str, Any]] = []
    for (scope, algorithm), group in primary.groupby(
        ["scope", "algorithm"], sort=False
    ):
        for challenger, baseline in pairs:
            arm_pairs: list[tuple[str, str, int | None]] = [(challenger, baseline, None)]
            challenger_prefix = f"{challenger}_shuffle_"
            baseline_prefix = f"{baseline}_shuffle_"
            challenger_repeats = {
                int(arm.removeprefix(challenger_prefix))
                for arm in group["arm"].astype(str)
                if arm.startswith(challenger_prefix)
            }
            baseline_repeats = {
                int(arm.removeprefix(baseline_prefix))
                for arm in group["arm"].astype(str)
                if arm.startswith(baseline_prefix)
            }
            for repeat in sorted(challenger_repeats & baseline_repeats):
                arm_pairs.append((
                    f"{challenger_prefix}{repeat:02d}",
                    f"{baseline_prefix}{repeat:02d}",
                    repeat,
                ))

            for challenger_arm, baseline_arm, repeat in arm_pairs:
                keys = ["fold_id", "trade_date"]
                left = group.loc[group["arm"].eq(challenger_arm)].set_index(keys)
                right = group.loc[group["arm"].eq(baseline_arm)].set_index(keys)
                common = left.index.intersection(right.index)
                if common.empty:
                    continue
                for metric in metrics:
                    delta = (left.loc[common, metric] - right.loc[common, metric]).dropna()
                    if delta.empty:
                        continue
                    low, high = (np.nan, np.nan)
                    if repeat is None:
                        low, high = _block_bootstrap_mean(
                            delta.reset_index(drop=True),
                            block_days=config.block_bootstrap_days,
                            iterations=config.bootstrap_iterations,
                            seed=20260717 + len(records),
                        )
                    records.append({
                        "scope": scope,
                        "algorithm": algorithm,
                        "challenger": challenger,
                        "baseline": baseline,
                        "metric": metric,
                        "placebo_repeat": repeat,
                        "is_placebo": repeat is not None,
                        "mean_delta": float(delta.mean()),
                        "ci_low": low,
                        "ci_high": high,
                        "day_count": int(len(delta)),
                        "positive_day_ratio": float(delta.gt(0).mean()),
                    })
    return pd.DataFrame(records)


def summarize_paired_placebos(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Summarize the actual incremental delta against paired placebo deltas."""
    if comparisons.empty:
        return comparisons.copy()
    records: list[dict[str, Any]] = []
    keys = ["scope", "algorithm", "challenger", "baseline", "metric"]
    for values, group in comparisons.groupby(keys, sort=False):
        actual = group.loc[~group["is_placebo"]]
        placebo = group.loc[group["is_placebo"], "mean_delta"].dropna().to_numpy(float)
        if actual.empty or not len(placebo):
            continue
        actual_value = float(actual.iloc[0]["mean_delta"])
        records.append({
            **dict(zip(keys, values)),
            "actual_delta": actual_value,
            "actual_ci_low": float(actual.iloc[0]["ci_low"]),
            "actual_ci_high": float(actual.iloc[0]["ci_high"]),
            "placebo_count": int(len(placebo)),
            "placebo_mean": float(placebo.mean()),
            "placebo_std": float(placebo.std(ddof=1)) if len(placebo) > 1 else 0.0,
            "placebo_min": float(placebo.min()),
            "placebo_max": float(placebo.max()),
            "empirical_percentile": float(
                (1 + np.sum(placebo <= actual_value)) / (len(placebo) + 1)
            ),
            "one_sided_p_positive": float(
                (1 + np.sum(placebo >= actual_value)) / (len(placebo) + 1)
            ),
            "two_sided_p": float(
                (1 + np.sum(np.abs(placebo) >= abs(actual_value)))
                / (len(placebo) + 1)
            ),
        })
    return pd.DataFrame(records)


def aggregate_importance(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return importance
    output = (
        importance.groupby(["scope", "algorithm", "arm", "feature"], as_index=False)
        .agg(
            importance_mean=("importance", "mean"),
            importance_abs_mean=("importance", lambda values: float(np.abs(values).mean())),
            folds=("fold_id", "nunique"),
        )
    )
    return output.sort_values(
        ["scope", "algorithm", "arm", "importance_abs_mean"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def summarize_placebos(
    summary: pd.DataFrame,
    *,
    config: EvaluationConfig = EvaluationConfig(),
) -> pd.DataFrame:
    primary = summary.loc[
        summary["top_n"].eq(config.primary_top_n)
        & summary["cost_bps"].eq(config.primary_cost_bps)
    ].copy()
    records: list[dict[str, Any]] = []
    metrics = ("rank_ic_mean", "top_net_mean", "top_minus_all_mean", "top_minus_bottom_mean")
    for (scope, algorithm), group in primary.groupby(["scope", "algorithm"], sort=False):
        placebo_arm_names = sorted({
            arm.split("_shuffle_", maxsplit=1)[0]
            for arm in group["arm"].astype(str)
            if "_shuffle_" in arm
        })
        for arm in placebo_arm_names:
            actual = group.loc[group["arm"].eq(arm)]
            placebo = group.loc[
                group["arm"].str.startswith(f"{arm}_shuffle_")
            ]
            if actual.empty or placebo.empty:
                continue
            for metric in metrics:
                actual_value = float(actual.iloc[0][metric])
                placebo_values = placebo[metric].dropna().to_numpy(dtype="float64")
                records.append(
                    {
                        "scope": scope,
                        "algorithm": algorithm,
                        "arm": arm,
                        "metric": metric,
                        "actual": actual_value,
                        "placebo_count": int(len(placebo_values)),
                        "placebo_mean": float(placebo_values.mean()),
                        "placebo_std": float(placebo_values.std(ddof=1)),
                        "placebo_max": float(placebo_values.max()),
                        "empirical_percentile": float(
                            (1 + np.sum(placebo_values <= actual_value))
                            / (len(placebo_values) + 1)
                        ),
                    }
                )
    return pd.DataFrame(records)


def persist_experiment(
    output_dir: str | Path,
    *,
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    folds: pd.DataFrame,
    daily_evaluation: pd.DataFrame,
    summary: pd.DataFrame,
    fold_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    model_config: ModelConfig,
    evaluation_config: EvaluationConfig,
    metadata: Mapping[str, Any],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output / "conditional_event_dataset.parquet", index=False)
    predictions.to_parquet(output / "oof_predictions.parquet", index=False)
    importance.to_parquet(output / "feature_importance_by_fold.parquet", index=False)
    aggregate_importance(importance).to_parquet(
        output / "feature_importance_aggregate.parquet",
        index=False,
    )
    folds.to_csv(output / "walk_forward_folds.csv", index=False)
    daily_evaluation.to_parquet(output / "daily_oof_evaluation.parquet", index=False)
    summary.to_csv(output / "oof_summary.csv", index=False)
    fold_summary.to_csv(output / "oof_fold_summary.csv", index=False)
    comparisons.to_csv(output / "oof_arm_comparisons.csv", index=False)
    summarize_placebos(summary, config=evaluation_config).to_csv(
        output / "oof_placebo_summary.csv",
        index=False,
    )
    feature_blocks = feature_columns_by_block(dataset)
    manifest = {
        "metadata": dict(metadata),
        "model_config": asdict(model_config),
        "evaluation_config": asdict(evaluation_config),
        "arm_blocks": {key: list(value) for key, value in ALL_ARM_BLOCKS.items()},
        "feature_blocks": feature_blocks,
        "row_count": int(len(dataset)),
        "event_start": str(dataset["trade_date"].min().date()),
        "event_end": str(dataset["trade_date"].max().date()),
        "prediction_rows": int(len(predictions)),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
