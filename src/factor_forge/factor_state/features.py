from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_MARKET_CONTEXT_COLUMNS = [
    "market_ret_20",
    "market_ret_60",
    "market_vol_20",
    "market_breadth_20",
    "market_xsec_vol_20",
    "market_turnover_change",
    "market_turnover_chg_5_20",
]


@dataclass(frozen=True)
class FactorHealthConfig:
    date_col: str = "trade_date"
    asset_col: str = "ts_code"
    factor_col: str = "factor_score"
    return_col: str = "future_return"
    factor_name: str = "band_score"
    windows: tuple[int, ...] = (20, 60, 120)
    deciles: int = 10
    min_obs_per_day: int = 30
    market_context_columns: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_MARKET_CONTEXT_COLUMNS))


def build_factor_health_daily(frame: pd.DataFrame, config: FactorHealthConfig | None = None) -> pd.DataFrame:
    """Build daily factor lifecycle health features from cross-sectional factor observations.

    Required input columns are date, asset id, factor score and forward return. Market context
    columns are optional and copied as daily means when present.
    """
    cfg = config or FactorHealthConfig()
    data = _normalize_input(frame, cfg)
    observations = _daily_observations(data, cfg)
    health = _rolling_health_features(observations, cfg)
    return health.sort_values(["date", "factor_name"]).reset_index(drop=True)


def _normalize_input(frame: pd.DataFrame, cfg: FactorHealthConfig) -> pd.DataFrame:
    required = [cfg.date_col, cfg.asset_col, cfg.factor_col, cfg.return_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    keep = list(dict.fromkeys([*required, *cfg.market_context_columns]))
    data = frame[[col for col in keep if col in frame.columns]].copy()
    data = data.rename(
        columns={
            cfg.date_col: "date",
            cfg.asset_col: "asset",
            cfg.factor_col: "factor_score",
            cfg.return_col: "future_return",
        }
    )
    data["date"] = pd.to_datetime(data["date"])
    data["factor_name"] = cfg.factor_name
    data["factor_score"] = pd.to_numeric(data["factor_score"], errors="coerce")
    data["future_return"] = pd.to_numeric(data["future_return"], errors="coerce")
    return data.dropna(subset=["date", "asset", "factor_score", "future_return"])


def _daily_observations(data: pd.DataFrame, cfg: FactorHealthConfig) -> pd.DataFrame:
    rows: list[dict] = []
    previous_top: set[str] | None = None
    for date, group in data.sort_values(["date", "asset"]).groupby("date", sort=True):
        if len(group) < cfg.min_obs_per_day:
            continue
        factor = group["factor_score"]
        returns = group["future_return"]
        rank_ic = factor.corr(returns, method="spearman")
        pearson_ic = factor.corr(returns, method="pearson")
        decile_table = _decile_returns(group, cfg.deciles)
        top_return = decile_table.get(cfg.deciles, np.nan)
        bottom_return = decile_table.get(1, np.nan)
        spread = top_return - bottom_return if pd.notna(top_return) and pd.notna(bottom_return) else np.nan
        monotonicity = _decile_monotonicity(decile_table)
        score_rank = factor.rank(method="first", pct=True)
        top_mask = score_rank > (1.0 - 1.0 / cfg.deciles)
        bottom_mask = score_rank <= (1.0 / cfg.deciles)
        top_assets = set(group.loc[top_mask, "asset"].astype(str))
        overlap = _set_overlap(previous_top, top_assets)
        previous_top = top_assets
        top_scores = factor.loc[top_mask]
        abs_top = top_scores.abs()
        top_concentration = float(abs_top.max() / abs_top.sum()) if abs_top.sum() > 0 else np.nan
        row = {
            "date": date,
            "factor_name": cfg.factor_name,
            "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
            "pearson_ic": float(pearson_ic) if pd.notna(pearson_ic) else np.nan,
            "decile_monotonicity": monotonicity,
            "top_decile_return": float(top_return) if pd.notna(top_return) else np.nan,
            "bottom_decile_return": float(bottom_return) if pd.notna(bottom_return) else np.nan,
            "decile_spread": float(spread) if pd.notna(spread) else np.nan,
            "score_dispersion": float(factor.std(ddof=1)),
            "top_bottom_score_gap": _score_gap(factor, top_mask, bottom_mask),
            "top_decile_turnover": np.nan if overlap is None else 1.0 - overlap,
            "holding_overlap": overlap,
            "top_stock_concentration": top_concentration,
            "observation_count": int(len(group)),
        }
        for decile, value in decile_table.items():
            row[f"decile_{decile}_return"] = value
        for col in cfg.market_context_columns:
            if col in group.columns:
                row[_market_output_name(col)] = float(pd.to_numeric(group[col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _decile_returns(group: pd.DataFrame, deciles: int) -> dict[int, float]:
    ranked = group["factor_score"].rank(method="first")
    try:
        buckets = pd.qcut(ranked, deciles, labels=False) + 1
    except ValueError:
        return {}
    values = group.assign(_decile=buckets).groupby("_decile", observed=True)["future_return"].mean()
    return {int(decile): float(value) for decile, value in values.items()}


def _decile_monotonicity(decile_table: dict[int, float]) -> float:
    if len(decile_table) < 3:
        return np.nan
    deciles = pd.Series(decile_table).sort_index()
    return float(pd.Series(deciles.index, index=deciles.index).corr(deciles, method="spearman"))


def _score_gap(factor: pd.Series, top_mask: pd.Series, bottom_mask: pd.Series) -> float:
    if not top_mask.any() or not bottom_mask.any():
        return np.nan
    return float(factor.loc[top_mask].mean() - factor.loc[bottom_mask].mean())


def _set_overlap(previous: set[str] | None, current: set[str]) -> float | None:
    if previous is None or not previous or not current:
        return None
    return float(len(previous & current) / len(previous | current))


def _rolling_health_features(observations: pd.DataFrame, cfg: FactorHealthConfig) -> pd.DataFrame:
    if observations.empty:
        return observations
    frames = []
    for factor_name, group in observations.groupby("factor_name", sort=False):
        g = group.sort_values("date").copy()
        for window in cfg.windows:
            min_periods = max(5, min(window, window // 2))
            g[f"rolling_rank_ic_{window}"] = g["rank_ic"].rolling(window, min_periods=min_periods).mean()
            ic_std = g["rank_ic"].rolling(window, min_periods=min_periods).std(ddof=1)
            g[f"icir_{window}"] = g[f"rolling_rank_ic_{window}"] / ic_std.replace(0, np.nan)
            g[f"spread_{window}"] = g["decile_spread"].rolling(window, min_periods=min_periods).mean()
        g["spread_velocity_20_60"] = g["spread_20"] - g["spread_60"]
        g["spread_velocity_60_120"] = g["spread_60"] - g["spread_120"]
        g["ic_velocity_20_60"] = g["rolling_rank_ic_20"] - g["rolling_rank_ic_60"]
        g["ic_velocity_60_120"] = g["rolling_rank_ic_60"] - g["rolling_rank_ic_120"]
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def _market_output_name(column: str) -> str:
    if column == "market_turnover_chg_5_20":
        return "market_turnover_change"
    return column


def available_model_features(frame: pd.DataFrame, candidates: Iterable[str] | None = None, max_features: int = 30) -> list[str]:
    base = list(
        candidates
        or [
            "rolling_rank_ic_20",
            "rolling_rank_ic_60",
            "rolling_rank_ic_120",
            "icir_20",
            "icir_60",
            "icir_120",
            "spread_20",
            "spread_60",
            "spread_120",
            "spread_velocity_20_60",
            "spread_velocity_60_120",
            "ic_velocity_20_60",
            "ic_velocity_60_120",
            "decile_monotonicity",
            "score_dispersion",
            "top_bottom_score_gap",
            "top_decile_turnover",
            "holding_overlap",
            "top_stock_concentration",
            "market_ret_20",
            "market_ret_60",
            "market_vol_20",
            "market_breadth_20",
            "market_xsec_vol_20",
            "market_turnover_change",
        ]
    )
    return [col for col in base if col in frame.columns][:max_features]
