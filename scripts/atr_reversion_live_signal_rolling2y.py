"""Generate a live-style ATR lower-shadow signal.

Signal convention:
- Features use data available at signal-date close.
- Trades are intended for the next trading day open, subject to fillability.
- Model training uses rolling two calendar years of rows with realized labels.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository
from factor_forge.ml.atr_reversion_config import load_atr_reversion_config
from factor_forge.ml.atr_reversion_dataset import FEATURE_GROUPS, build_atr_reversion_dataset

from atr_reversion_pit_hmm_calibrated_backtest import _market_features, _tiered_weight, _walk_forward_hmm
from atr_reversion_pit_liquidity_backtest import _attach_and_preprocess_pit, _pit_liquidity_flags
from atr_reversion_regime_aware_cluster_matrix import CLUSTER_COLS, _add_clusters
from atr_reversion_strategy_regime_mining import _build_regime_features
from atr_reversion_walk_forward import REBALANCE_DAYS


SIGNAL_DATE: str | None = None
TRAIN_YEARS = 2
TOP_N = 5
FIT_QUALITY_LOOKBACK = 40
FIT_QUALITY_MIN_OBS = 15
STRATEGY_PAYOFF_LOOKBACK_DAYS = 240
STRATEGY_PAYOFF_MIN_EFFECTIVE_OBS = 12.0
STRATEGY_PAYOFF_Z = 1.0
STRATEGY_PAYOFF_COST_BPS = 20
EXCLUDED_BOARDS = ["STAR 688/689.SH", "ChiNext 300/301/302.SZ", "Beijing *.BJ"]
OUTPUT_ROOT = Path("artifacts/atr_reversion_live_signals")
CONFIG_PATH = "configs/ml/atr_reversion_lightgbm_v1.yaml"
FIT_QUALITY_SOURCE_RUN = Path(
    "artifacts/atr_reversion_runs/"
    "atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/"
    "hmm_window_comparison_20260707T002239Z"
)
HEALTH_SCORE_PATH = Path(
    "artifacts/atr_reversion_runs/"
    "atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/"
    "three_layer_gate_20260706T113826Z/"
    "gate_scores_test_2026h1_cost10.csv"
)
EVENT_FROZEN_RUN = Path(
    "artifacts/atr_reversion_runs/"
    "atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/"
    "event_badtrade_iteration_20260707T111609Z"
)
EVENT_MODEL_VERSION = "event_alpha_payoff_gate_top5_frozen_20260707"
FIT_QUALITY_SOURCE_RUN = EVENT_FROZEN_RUN
HMM_RANK_PATH = EVENT_FROZEN_RUN / "test_2026h1" / "state_validation_perf_event_cost20.csv"


def _json_default(obj):
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _load_panel(version_name: str = "latest") -> tuple[str, pd.DataFrame]:
    project = load_project("configs/project.yaml")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, manifest = repo.load_manifest(version_name)
    _, panel = repo.load_panel(version)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    return version, panel


def _log_factory(output: Path):
    log_path = output / "run.log"

    def log(message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return log


def _hmm_ranks(path: Path) -> dict[str, int]:
    perf = pd.read_csv(path)
    ordered = perf.sort_values("mean_excess", ascending=False)["predicted_state"].astype(int).tolist()
    return {"best": ordered[0], "neutral": ordered[1], "worst": ordered[2]}


def _latest_health_value(path: Path, signal_date: pd.Timestamp) -> dict[str, object]:
    if not path.exists():
        return {
            "top5_excess_5round": np.nan,
            "health_source_date": None,
            "health_source_path": None,
        }
    scores = pd.read_csv(path, parse_dates=["trade_date"]).sort_values("trade_date")
    scores = scores[scores["trade_date"] <= signal_date]
    if scores.empty or "top5_excess_5round" not in scores:
        return {
            "top5_excess_5round": np.nan,
            "health_source_date": None,
            "health_source_path": str(path),
        }
    scores = scores[scores["top5_excess_5round"].notna()]
    if scores.empty:
        return {
            "top5_excess_5round": np.nan,
            "health_source_date": None,
            "health_source_path": str(path),
        }
    row = scores.iloc[-1]
    return {
        "top5_excess_5round": float(row["top5_excess_5round"]),
        "health_source_date": row["trade_date"],
        "health_source_path": str(path),
    }


def _stock_name_map() -> pd.DataFrame:
    latest_raw = Path("data/versions/data_v1_20260706T135559Z_b6ec0d6c/raw/tushare/stock_basic.parquet")
    fallback_raw = Path("data/versions/data_v1_20260704T074315Z_88d001e2/raw/tushare/stock_basic.parquet")
    for path in (latest_raw, fallback_raw):
        if path.exists():
            names = pd.read_parquet(path, columns=["ts_code", "name"])
            return names.drop_duplicates("ts_code")
    return pd.DataFrame({"ts_code": [], "name": []})


def _load_fit_quality_predictions(signal_date: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for path in sorted(FIT_QUALITY_SOURCE_RUN.glob("test_*/predictions_event_alpha_bad.parquet")):
        pred = pd.read_parquet(path)
        if "factor_value" not in pred.columns and "alpha_pred" in pred.columns:
            pred["factor_value"] = pred["alpha_pred"]
        pred = pred[["trade_date", "ts_code", "factor_value"]].copy()
        pred["trade_date"] = pd.to_datetime(pred["trade_date"])
        pred = pred[pred["ts_code"].map(_permission_eligible)]
        frames.append(pred[pred["trade_date"].le(signal_date)])
    if not frames:
        return pd.DataFrame(columns=["trade_date", "ts_code", "factor_value"])
    out = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["trade_date", "ts_code"])
        .drop_duplicates(["trade_date", "ts_code"], keep="last")
    )
    return out[out["trade_date"].le(signal_date)].copy()


def _fit_quality_for_signal(
    panel: pd.DataFrame,
    pit: pd.DataFrame,
    signal_date: pd.Timestamp,
    log,
) -> dict[str, object]:
    pred = _load_fit_quality_predictions(signal_date)
    if pred.empty:
        log("fit-quality: no historical prediction file found; using normal score direction")
        return _empty_fit_quality(signal_date, "missing_prediction_history")

    start = pred["trade_date"].min()
    if "permission_eligible" not in panel.columns:
        panel = panel.copy()
        panel["permission_eligible"] = panel["ts_code"].map(_permission_eligible).astype(bool)
    p = panel.loc[
        panel["trade_date"].between(start, signal_date) & panel["permission_eligible"]
    ].merge(
        pit[["trade_date", "ts_code", "pit_top1000"]],
        on=["trade_date", "ts_code"],
        how="left",
    )
    p["pit_top1000"] = p["pit_top1000"].fillna(False).astype(bool)
    data = p.merge(pred, on=["trade_date", "ts_code"], how="left").sort_values(["ts_code", "trade_date"])
    data["next_open"] = data.groupby("ts_code")["adj_open"].shift(-1)
    data["exit_open"] = data.groupby("ts_code")["adj_open"].shift(-(REBALANCE_DAYS + 1))
    data["exit_date"] = data.groupby("ts_code")["trade_date"].shift(-(REBALANCE_DAYS + 1))
    data["fwd_ret"] = data["exit_open"] / data["next_open"] - 1.0
    eligible = (
        data["pit_top1000"]
        & data["factor_value"].notna()
        & data["fwd_ret"].replace([np.inf, -np.inf], np.nan).notna()
        & data["exit_date"].notna()
        & pd.to_datetime(data["exit_date"]).le(signal_date)
    )
    daily = data.loc[eligible].copy()
    rows = []
    for date, g in daily.groupby("trade_date", sort=True):
        if len(g) < 100:
            continue
        ranked = g["factor_value"].rank(method="first")
        decile = pd.qcut(ranked, 10, labels=False, duplicates="drop")
        g = g.assign(decile=decile)
        dec = g.groupby("decile", observed=True)["fwd_ret"].mean()
        spread = float(dec.loc[dec.index.max()] - dec.loc[dec.index.min()]) if len(dec) >= 2 else np.nan
        top = g.nlargest(TOP_N, "factor_value")
        rows.append(
            {
                "trade_date": date,
                "known_date": pd.to_datetime(g["exit_date"].max()),
                "rank_ic": float(g["factor_value"].corr(g["fwd_ret"], method="spearman")),
                "decile_spread": spread,
                "top5_excess_forward_return": float(top["fwd_ret"].mean() - g["fwd_ret"].mean()),
                "top5_hit_rate": float((top["fwd_ret"] > 0.0).mean()),
            }
        )
    fit = pd.DataFrame(rows)
    if fit.empty:
        log("fit-quality: no completed historical outcomes; using normal score direction")
        return _empty_fit_quality(signal_date, "missing_completed_outcomes")

    hist = fit.sort_values("known_date")
    hist = hist[hist["known_date"].le(signal_date)].tail(FIT_QUALITY_LOOKBACK)
    rank_ic = float(hist["rank_ic"].mean()) if len(hist) else np.nan
    spread = float(hist["decile_spread"].mean()) if len(hist) else np.nan
    top5_excess = float(hist["top5_excess_forward_return"].mean()) if len(hist) else np.nan
    top5_hit = float(hist["top5_hit_rate"].mean()) if len(hist) else np.nan
    flip = len(hist) >= FIT_QUALITY_MIN_OBS and rank_ic < 0.0 and spread < 0.0
    direction = -1.0 if flip else 1.0
    log(
        "fit-quality frozen rule "
        f"lookback={FIT_QUALITY_LOOKBACK} min_obs={FIT_QUALITY_MIN_OBS} "
        f"fit_obs={len(hist)} rank_ic={rank_ic:.4f} decile_spread={spread:.4f} "
        f"score_direction={direction:+.0f}"
    )
    return {
        "policy": "fit_quality_flip_only_frozen",
        "lookback": FIT_QUALITY_LOOKBACK,
        "min_obs": FIT_QUALITY_MIN_OBS,
        "fit_obs": int(len(hist)),
        "rank_ic_rolling": rank_ic,
        "decile_spread_rolling": spread,
        "top5_excess_rolling": top5_excess,
        "top5_hit_rolling": top5_hit,
        "score_direction": direction,
        "flipped": bool(flip),
        "latest_completed_signal_date": hist["trade_date"].max() if len(hist) else None,
        "latest_known_date": hist["known_date"].max() if len(hist) else None,
        "source_run": str(FIT_QUALITY_SOURCE_RUN),
    }


def _empty_fit_quality(signal_date: pd.Timestamp, reason: str) -> dict[str, object]:
    return {
        "policy": "fit_quality_flip_only_frozen",
        "lookback": FIT_QUALITY_LOOKBACK,
        "min_obs": FIT_QUALITY_MIN_OBS,
        "fit_obs": 0,
        "rank_ic_rolling": np.nan,
        "decile_spread_rolling": np.nan,
        "top5_excess_rolling": np.nan,
        "top5_hit_rolling": np.nan,
        "score_direction": 1.0,
        "flipped": False,
        "latest_completed_signal_date": None,
        "latest_known_date": None,
        "source_run": str(FIT_QUALITY_SOURCE_RUN),
        "fallback_reason": reason,
        "asof": signal_date,
    }


def _permission_eligible(ts_code: str) -> bool:
    code = str(ts_code)
    if code.endswith(".BJ"):
        return False
    if code.endswith(".SH") and code[:3] in {"688", "689"}:
        return False
    if code.endswith(".SZ") and code[:3] in {"300", "301", "302"}:
        return False
    return True


def _event_trigger(d: pd.DataFrame) -> pd.Series:
    lower = d["lower_shadow_pct"].ge(0.60) | d["lower_shadow_atr"].ge(0.80)
    downside = d["down_deviation_pct"].ge(0.50) | d["down_deviation_atr"].ge(0.0)
    confirm = d["core_signal"].notna() & d["intraday_repair"].notna()
    no_extreme_upper = d["upper_shadow_pct"].fillna(1.0).le(0.95)
    return lower & downside & confirm & no_extreme_upper


def _future_low_10(s: pd.Series) -> pd.Series:
    return s.shift(-1).iloc[::-1].rolling(REBALANCE_DAYS, min_periods=1).min().iloc[::-1]


def _build_forward_outcomes(panel: pd.DataFrame, pit: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "trade_date",
        "ts_code",
        "adj_open",
        "adj_low",
        "industry_l1_code",
        "is_tradeable",
        "is_suspended",
        "is_st",
        "is_delisting_period",
        "listing_trade_days",
    ]
    d = panel[cols].merge(pit[["trade_date", "ts_code", "pit_top1000"]], on=["trade_date", "ts_code"], how="left")
    d["pit_top1000"] = d["pit_top1000"].fillna(False).astype(bool)
    d = d[d["ts_code"].map(_permission_eligible)].sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    stocks = d["ts_code"]
    entry_open = d["adj_open"].groupby(stocks, sort=False).shift(-1)
    exit_open = d["adj_open"].groupby(stocks, sort=False).shift(-(REBALANCE_DAYS + 1))
    exit_date = d["trade_date"].groupby(stocks, sort=False).shift(-(REBALANCE_DAYS + 1))
    fwd_ret = exit_open / entry_open - 1.0
    future_low = d["adj_low"].groupby(stocks, sort=False).transform(_future_low_10)
    fwd_dd = future_low / entry_open - 1.0
    valid = (
        d["pit_top1000"]
        & d["is_tradeable"].fillna(False).astype(bool)
        & ~d["is_suspended"].fillna(True).astype(bool)
        & ~d["is_st"].fillna(True).astype(bool)
        & ~d["is_delisting_period"].fillna(True).astype(bool)
        & d["listing_trade_days"].ge(60)
    )
    industry_mean = fwd_ret.where(valid).groupby([d["trade_date"], d["industry_l1_code"]], sort=False).transform("mean")
    out = d[["trade_date", "ts_code", "industry_l1_code"]].copy()
    out["fwd_ret_10"] = fwd_ret
    out["fwd_industry_excess_10"] = fwd_ret - industry_mean
    out["fwd_drawdown_10"] = fwd_dd
    out["bad_trade_10"] = ((fwd_ret <= -0.05) | (fwd_dd <= -0.08)).where(fwd_ret.notna() & fwd_dd.notna())
    out["exit_date"] = pd.to_datetime(exit_date)
    return out


def _prepare_event_model_dataset(
    dataset: pd.DataFrame,
    outcomes: pd.DataFrame,
    signal_date: pd.Timestamp,
    log,
) -> pd.DataFrame:
    d = _add_clusters(dataset).rename(columns={"datetime": "trade_date", "instrument": "ts_code"}).copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    d = d.merge(outcomes, on=["trade_date", "ts_code"], how="left")
    d["event_trigger"] = _event_trigger(d)
    for col in CLUSTER_COLS:
        d[col] = d[col].replace([np.inf, -np.inf], np.nan)
    usable = d["pit_top1000"].fillna(False).astype(bool) & d["event_trigger"].fillna(False).astype(bool)
    d.loc[~usable, ["fwd_industry_excess_10", "bad_trade_10"]] = np.nan
    event_rows = int(d.loc[d["trade_date"].eq(signal_date), "event_trigger"].fillna(False).sum())
    pit_event_rows = int((d["trade_date"].eq(signal_date) & usable).sum())
    log(
        f"prepared event model dataset rows={len(d):,} "
        f"signal_event_rows={event_rows:,} signal_pit_event_rows={pit_event_rows:,}"
    )
    return d


def _train_event_alpha_model(dataset: pd.DataFrame, train_start: pd.Timestamp, signal_date: pd.Timestamp, log):
    from lightgbm import LGBMRegressor

    cols = [
        "trade_date",
        "ts_code",
        *CLUSTER_COLS,
        "sample_weight",
        "pit_top1000",
        "event_trigger",
        "fwd_industry_excess_10",
        "exit_date",
    ]
    eligible = dataset.loc[
        dataset["trade_date"].between(train_start, signal_date - pd.Timedelta(days=1))
        & dataset["pit_top1000"].fillna(False).astype(bool)
        & dataset["event_trigger"].fillna(False).astype(bool)
        & dataset["exit_date"].notna()
        & pd.to_datetime(dataset["exit_date"]).le(signal_date),
        cols,
    ].dropna(subset=[*CLUSTER_COLS, "fwd_industry_excess_10"])
    if eligible.empty:
        raise ValueError("No realized event-alpha training rows after rolling window, PIT, and label filters.")

    log(
        "training frozen event-alpha LightGBM "
        f"rows={len(eligible):,} "
        f"dates={eligible['trade_date'].min().date()}..{eligible['trade_date'].max().date()}"
    )
    model = LGBMRegressor(
        objective="regression",
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        n_estimators=450,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )
    model.fit(
        eligible[CLUSTER_COLS],
        eligible["fwd_industry_excess_10"],
        sample_weight=eligible["sample_weight"].fillna(1.0),
    )
    return model, eligible


def _historical_tercile_bucket(s: pd.Series, min_obs: int = 120) -> pd.Series:
    values = s.replace([np.inf, -np.inf], np.nan).astype(float)
    low = values.expanding(min_periods=min_obs).quantile(1 / 3).shift(1)
    high = values.expanding(min_periods=min_obs).quantile(2 / 3).shift(1)
    bucket = pd.Series(pd.NA, index=s.index, dtype="object")
    bucket = bucket.mask(values.le(low), "low")
    bucket = bucket.mask(values.gt(low) & values.lt(high), "mid")
    bucket = bucket.mask(values.ge(high), "high")
    return bucket


def _payoff_gate_for_signal(regime: pd.DataFrame, signal_date: pd.Timestamp) -> dict[str, object]:
    cols = ["market_breadth_20", "reversal_strength_20", "market_vol_20", "turnover_chg_5_20"]
    r = regime.loc[regime["trade_date"].le(signal_date), ["trade_date", *cols]].sort_values("trade_date").copy()
    for col in cols:
        r[f"{col}_bucket"] = _historical_tercile_bucket(r[col])
    if r.empty or not r["trade_date"].eq(signal_date).any():
        return {
            "payoff_gate": 1.0,
            "payoff_gate_score": np.nan,
            "fallback_reason": "missing_regime_row",
        }
    row = r.loc[r["trade_date"].eq(signal_date)].iloc[-1]
    score = 0.0
    score += 1.0 if row.get("market_breadth_20_bucket") == "low" else 0.0
    score += -1.0 if row.get("market_breadth_20_bucket") == "mid" else 0.0
    score += 1.0 if row.get("reversal_strength_20_bucket") in {"low", "mid"} else 0.0
    score += -1.0 if row.get("reversal_strength_20_bucket") == "high" else 0.0
    score += 1.0 if row.get("market_vol_20_bucket") == "high" else 0.0
    score += -1.0 if row.get("market_vol_20_bucket") == "mid" else 0.0
    score += 0.5 if row.get("turnover_chg_5_20_bucket") == "high" else 0.0
    missing = pd.isna(row.get("market_breadth_20_bucket")) or pd.isna(row.get("reversal_strength_20_bucket")) or pd.isna(row.get("market_vol_20_bucket"))
    gate = 1.0 if score >= 1.0 else (0.5 if score >= 0.0 else 0.0)
    if missing:
        gate = 1.0
        score = np.nan
    out = row.to_dict()
    out["payoff_gate_score"] = float(score) if np.isfinite(score) else np.nan
    out["payoff_gate"] = float(gate)
    return out


def _strategy_payoff_gate_for_signal(
    panel: pd.DataFrame,
    pit: pd.DataFrame,
    signal_date: pd.Timestamp,
    score_direction: float,
    log,
) -> tuple[dict[str, object], pd.DataFrame]:
    pred = _load_fit_quality_predictions(signal_date)
    if pred.empty:
        log("strategy-payoff gate: no historical prediction file found; gate defaults to 1.0")
        return _empty_strategy_payoff(signal_date, "missing_prediction_history"), pd.DataFrame()

    start = pred["trade_date"].min()
    p = panel.loc[panel["trade_date"].between(start, signal_date)].copy()
    if "permission_eligible" not in p.columns:
        p["permission_eligible"] = p["ts_code"].map(_permission_eligible).astype(bool)
    p = p[p["permission_eligible"]].merge(
        pit[["trade_date", "ts_code", "pit_top1000"]],
        on=["trade_date", "ts_code"],
        how="left",
    )
    p["pit_top1000"] = p["pit_top1000"].fillna(False).astype(bool)
    data = p.merge(pred, on=["trade_date", "ts_code"], how="left").sort_values(["ts_code", "trade_date"])
    by_stock = data.groupby("ts_code", sort=False)
    data["entry_open"] = by_stock["adj_open"].shift(-1)
    data["exit_open"] = by_stock["adj_open"].shift(-(REBALANCE_DAYS + 1))
    data["exit_date"] = by_stock["trade_date"].shift(-(REBALANCE_DAYS + 1))
    data["entry_is_tradeable"] = by_stock["is_tradeable"].shift(-1)
    data["entry_is_suspended"] = by_stock["is_suspended"].shift(-1)
    data["entry_is_st"] = by_stock["is_st"].shift(-1)
    data["entry_is_delisting_period"] = by_stock["is_delisting_period"].shift(-1)
    data["entry_listing_trade_days"] = by_stock["listing_trade_days"].shift(-1)
    data["entry_is_limit_up_open"] = by_stock["is_limit_up_open"].shift(-1)
    data["fwd_ret"] = data["exit_open"] / data["entry_open"] - 1.0
    data["live_score"] = data["factor_value"] * score_direction

    eligible = (
        data["pit_top1000"]
        & data["factor_value"].notna()
        & data["fwd_ret"].replace([np.inf, -np.inf], np.nan).notna()
        & data["exit_date"].notna()
        & pd.to_datetime(data["exit_date"]).le(signal_date)
        & data["entry_is_tradeable"].fillna(False).astype(bool)
        & ~data["entry_is_suspended"].fillna(True).astype(bool)
        & ~data["entry_is_st"].fillna(True).astype(bool)
        & ~data["entry_is_delisting_period"].fillna(True).astype(bool)
        & data["entry_listing_trade_days"].ge(60)
        & ~data["entry_is_limit_up_open"].fillna(False).astype(bool)
    )
    completed = data.loc[eligible].copy()
    if completed.empty:
        log("strategy-payoff gate: no completed fillable Top5 outcomes; gate defaults to 1.0")
        return _empty_strategy_payoff(signal_date, "missing_completed_outcomes"), pd.DataFrame()

    cost = STRATEGY_PAYOFF_COST_BPS / 10_000.0
    rows = []
    for date, g in completed.groupby("trade_date", sort=True):
        if len(g) < TOP_N:
            continue
        top = g.nlargest(TOP_N, "live_score")
        rows.append(
            {
                "trade_date": date,
                "known_date": pd.to_datetime(top["exit_date"].max()),
                "selected_count": int(len(top)),
                "candidate_count": int(len(g)),
                "gross_return_10d": float(top["fwd_ret"].mean()),
                "net_return_10d": float(top["fwd_ret"].mean() - cost),
                "hit_rate": float((top["fwd_ret"] > cost).mean()),
            }
        )
    history = pd.DataFrame(rows)
    if history.empty:
        log("strategy-payoff gate: completed history has no full Top5 cohorts; gate defaults to 1.0")
        return _empty_strategy_payoff(signal_date, "no_full_top5_cohorts"), history

    cutoff = signal_date - pd.Timedelta(days=STRATEGY_PAYOFF_LOOKBACK_DAYS)
    hist = history[
        history["known_date"].le(signal_date)
        & history["trade_date"].ge(cutoff)
        & history["net_return_10d"].replace([np.inf, -np.inf], np.nan).notna()
    ].sort_values("known_date")
    n = int(len(hist))
    n_eff = n / float(REBALANCE_DAYS)
    if n == 0 or n_eff < STRATEGY_PAYOFF_MIN_EFFECTIVE_OBS:
        log(
            "strategy-payoff gate: insufficient effective observations "
            f"n={n} n_eff={n_eff:.1f}; gate defaults to 1.0"
        )
        info = _empty_strategy_payoff(signal_date, "insufficient_effective_observations")
        info.update({"payoff_obs": n, "payoff_effective_obs": n_eff})
        return info, history

    mean = float(hist["net_return_10d"].mean())
    std = float(hist["net_return_10d"].std(ddof=1))
    stderr = std / np.sqrt(n_eff) if np.isfinite(std) and n_eff > 0 else 0.0
    lcb = mean - STRATEGY_PAYOFF_Z * stderr
    if not np.isfinite(stderr) or stderr <= 0.0:
        gate = 1.0 if mean > 0.0 else 0.0
    else:
        gate = float(np.clip(mean / (STRATEGY_PAYOFF_Z * stderr), 0.0, 1.0))
    if mean <= 0.0:
        gate = 0.0
    state = "PASS" if gate >= 1.0 else ("REDUCE" if gate > 0.0 else "RISK_OFF")
    log(
        "strategy-payoff gate "
        f"lookback_days={STRATEGY_PAYOFF_LOOKBACK_DAYS} n={n} n_eff={n_eff:.1f} "
        f"mean={mean:.4%} stderr={stderr:.4%} lcb={lcb:.4%} gate={gate:.2f}"
    )
    return (
        {
            "gate_type": "strategy_realized_payoff_gate",
            "payoff_gate": gate,
            "state": state,
            "lookback_days": STRATEGY_PAYOFF_LOOKBACK_DAYS,
            "cost_bps": STRATEGY_PAYOFF_COST_BPS,
            "z": STRATEGY_PAYOFF_Z,
            "min_effective_obs": STRATEGY_PAYOFF_MIN_EFFECTIVE_OBS,
            "payoff_obs": n,
            "payoff_effective_obs": n_eff,
            "payoff_mean_net_10d": mean,
            "payoff_std_net_10d": std,
            "payoff_stderr_net_10d": stderr,
            "payoff_lcb_net_10d": lcb,
            "payoff_hit_rate": float(hist["hit_rate"].mean()),
            "latest_completed_signal_date": hist["trade_date"].max(),
            "latest_known_date": hist["known_date"].max(),
            "score_direction_used": score_direction,
            "source_run": str(FIT_QUALITY_SOURCE_RUN),
        },
        history,
    )


def _empty_strategy_payoff(signal_date: pd.Timestamp, reason: str) -> dict[str, object]:
    return {
        "gate_type": "strategy_realized_payoff_gate",
        "payoff_gate": 1.0,
        "state": "UNKNOWN",
        "lookback_days": STRATEGY_PAYOFF_LOOKBACK_DAYS,
        "cost_bps": STRATEGY_PAYOFF_COST_BPS,
        "z": STRATEGY_PAYOFF_Z,
        "min_effective_obs": STRATEGY_PAYOFF_MIN_EFFECTIVE_OBS,
        "payoff_obs": 0,
        "payoff_effective_obs": 0.0,
        "payoff_mean_net_10d": np.nan,
        "payoff_std_net_10d": np.nan,
        "payoff_stderr_net_10d": np.nan,
        "payoff_lcb_net_10d": np.nan,
        "payoff_hit_rate": np.nan,
        "latest_completed_signal_date": None,
        "latest_known_date": None,
        "score_direction_used": np.nan,
        "source_run": str(FIT_QUALITY_SOURCE_RUN),
        "fallback_reason": reason,
        "asof": signal_date,
    }


def main(signal_date: str | None = SIGNAL_DATE, top_n: int = TOP_N) -> None:
    preloaded: tuple[str, pd.DataFrame] | None = None
    if signal_date is None:
        preloaded = _load_panel("latest")
        signal_ts = pd.Timestamp(preloaded[1]["trade_date"].max()).normalize()
    else:
        signal_ts = pd.Timestamp(signal_date)
    signal_date_text = f"{signal_ts:%Y-%m-%d}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = OUTPUT_ROOT / f"event_payoff_gate_signal_{signal_ts:%Y%m%d}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    log = _log_factory(output)

    log(f"loading config={CONFIG_PATH}")
    cfg0 = load_atr_reversion_config(CONFIG_PATH)
    cfg = cfg0.model_copy(
        update={
            "universe_top_n": None,
            "features": cfg0.features.model_copy(update={"cross_sectional_zscore": False, "winsor_quantile": 0.0}),
            "label": cfg0.label.model_copy(update={"cross_sectional_rank_label": False}),
        }
    )
    features = FEATURE_GROUPS["all"]

    version, panel = preloaded if preloaded is not None else _load_panel("latest")
    panel["permission_eligible"] = panel["ts_code"].map(_permission_eligible).astype(bool)
    manifest_end = panel["trade_date"].max()
    if signal_ts > manifest_end:
        raise ValueError(f"signal_date {signal_ts.date()} is after latest panel end {manifest_end.date()}")
    log(f"data_version={version} panel_dates={panel['trade_date'].min().date()}..{manifest_end.date()} rows={len(panel):,}")

    train_start = signal_ts - pd.DateOffset(years=TRAIN_YEARS)
    model_panel_start = train_start - pd.DateOffset(days=420)
    model_panel = panel.loc[panel["trade_date"].between(model_panel_start, signal_ts)].copy()
    regime_panel_start = signal_ts - pd.DateOffset(years=5)
    regime_panel = panel.loc[panel["trade_date"].between(regime_panel_start, signal_ts)].copy()
    log(
        "using sliced panels "
        f"model={model_panel['trade_date'].min().date()}..{model_panel['trade_date'].max().date()} rows={len(model_panel):,}; "
        f"regime={regime_panel['trade_date'].min().date()}..{regime_panel['trade_date'].max().date()} rows={len(regime_panel):,}"
    )

    log("building PIT top1000 flags for model/regime windows")
    pit = _pit_liquidity_flags(model_panel)
    regime_pit = _pit_liquidity_flags(regime_panel)

    log("building ATR reversion dataset")
    dataset_raw, _ = build_atr_reversion_dataset(model_panel, cfg.features, cfg.label)
    dataset = _attach_and_preprocess_pit(dataset_raw, pit, features, cfg0.features.winsor_quantile)

    log("building event clusters and realized 10d industry-excess labels")
    outcomes = _build_forward_outcomes(model_panel, pit)
    model_dataset = _prepare_event_model_dataset(dataset, outcomes, signal_ts, log)
    model, train = _train_event_alpha_model(model_dataset, train_start, signal_ts, log)

    predict_rows = model_dataset.loc[
        model_dataset["trade_date"].eq(signal_ts)
        & model_dataset["pit_top1000"].fillna(False).astype(bool)
        & model_dataset["event_trigger"].fillna(False).astype(bool),
        ["trade_date", "ts_code", *CLUSTER_COLS, "pit_top1000"],
    ].dropna(subset=CLUSTER_COLS)
    predict_rows = predict_rows[predict_rows["ts_code"].map(_permission_eligible)].copy()
    log(f"predicting signal_date={signal_ts.date()} candidates={len(predict_rows):,}")
    if predict_rows.empty:
        pred = pd.DataFrame(columns=["trade_date", "ts_code", "factor_value"])
    else:
        pred = predict_rows[["trade_date", "ts_code"]].copy()
        pred["factor_value"] = model.predict(predict_rows[CLUSTER_COLS])

    log("computing frozen fit-quality score direction")
    live_pit = _pit_liquidity_flags(panel.loc[panel["trade_date"].le(signal_ts)].copy())
    fit_quality = _fit_quality_for_signal(panel, live_pit, signal_ts, log)
    score_direction = float(fit_quality["score_direction"])
    pred["raw_factor_value"] = pred["factor_value"]
    pred["score_direction"] = score_direction
    pred["factor_value"] = pred["raw_factor_value"] * score_direction
    pred = pred.sort_values(["factor_value", "ts_code"], ascending=[False, True]).reset_index(drop=True)

    log("computing HMM state and tiered exposure")
    market = _market_features(regime_panel)
    if not market["trade_date"].eq(signal_ts).any():
        available = "none" if market.empty else f"{market['trade_date'].min().date()}..{market['trade_date'].max().date()}"
        today_rows = int(regime_panel["trade_date"].eq(signal_ts).sum())
        tradeable_rows = (
            int(regime_panel.loc[regime_panel["trade_date"].eq(signal_ts), "is_tradeable"].fillna(False).astype(bool).sum())
            if "is_tradeable" in regime_panel
            else 0
        )
        raise RuntimeError(
            f"signal_date {signal_ts.date()} has no computable market features for HMM; "
            f"market_feature_range={available}; panel_rows={today_rows}; tradeable_rows={tradeable_rows}"
        )
    states = _walk_forward_hmm(market, signal_date_text, signal_date_text, log)
    state_row = states.loc[states["trade_date"].eq(signal_ts)].iloc[-1]
    ranks = _hmm_ranks(HMM_RANK_PATH)
    hmm_exposure = float(_tiered_weight(state_row, ranks))

    log("computing frozen market payoff gate for audit")
    regime = _build_regime_features(regime_panel, regime_pit, dataset)
    market_payoff_info = _payoff_gate_for_signal(regime, signal_ts)
    market_payoff_gate = float(market_payoff_info.get("payoff_gate", 1.0))

    log("computing strategy realized payoff gate")
    strategy_payoff_info, strategy_payoff_history = _strategy_payoff_gate_for_signal(
        panel,
        live_pit,
        signal_ts,
        score_direction,
        log,
    )
    strategy_payoff_gate = float(strategy_payoff_info.get("payoff_gate", 1.0))
    risk_gate = strategy_payoff_gate
    final_exposure = hmm_exposure * strategy_payoff_gate

    top = pred.head(top_n).copy()
    top["rank"] = np.arange(1, len(top) + 1)
    top["target_weight"] = final_exposure / len(top) if len(top) and final_exposure > 0 else 0.0
    keep_columns = [
        "trade_date",
        "ts_code",
        "industry_l1_name",
        "raw_close",
        "amount_cny",
        "is_tradeable",
        "is_st",
        "is_delisting_period",
        "is_suspended",
        "listing_trade_days",
        "is_limit_up_open",
        "is_limit_down_open",
    ]
    if "name" in panel.columns:
        keep_columns.insert(2, "name")
    keep_panel = panel.loc[panel["trade_date"].eq(signal_ts), keep_columns].copy()
    if "name" not in keep_panel.columns:
        keep_panel = keep_panel.merge(_stock_name_map(), on="ts_code", how="left")
        keep_panel["name"] = keep_panel["name"].fillna("")
    top = top.merge(keep_panel, on=["trade_date", "ts_code"], how="left")
    cols = [
        "rank",
        "trade_date",
        "ts_code",
        "name",
        "industry_l1_name",
        "factor_value",
        "raw_factor_value",
        "score_direction",
        "target_weight",
        "raw_close",
        "amount_cny",
        "listing_trade_days",
        "is_tradeable",
        "is_st",
        "is_delisting_period",
        "is_suspended",
        "is_limit_up_open",
        "is_limit_down_open",
    ]
    for col in cols:
        if col not in top.columns:
            top[col] = np.nan
    top[cols].to_csv(output / "top_recommendations.csv", index=False, encoding="utf-8-sig")
    pred.head(100).merge(keep_panel, on=["trade_date", "ts_code"], how="left").to_csv(
        output / "top100_candidates.csv", index=False, encoding="utf-8-sig"
    )
    if not strategy_payoff_history.empty:
        strategy_payoff_history.to_csv(output / "strategy_payoff_history.csv", index=False, encoding="utf-8-sig")
    states.to_csv(output / "hmm_state.csv", index=False, encoding="utf-8-sig")

    summary = {
        "signal_date": signal_ts,
        "intended_execution": "next_trade_day_open",
        "data_version": version,
        "model": "LightGBM event_alpha rolling_2y",
        "signal_algorithm": EVENT_MODEL_VERSION,
        "frozen_model_version": EVENT_MODEL_VERSION,
        "permission_filter": {
            "enabled": True,
            "excluded_boards": EXCLUDED_BOARDS,
        },
        "event_trigger": {
            "lower_shadow_pct_min": 0.60,
            "lower_shadow_atr_min": 0.80,
            "down_deviation_pct_min": 0.50,
            "down_deviation_atr_min": 0.0,
            "upper_shadow_pct_max": 0.95,
            "requires_core_signal_and_intraday_repair": True,
        },
        "cluster_features": CLUSTER_COLS,
        "train_start_requested": train_start,
        "train_start_actual": train["trade_date"].min(),
        "train_end_actual": train["trade_date"].max(),
        "train_rows": int(len(train)),
        "predictable_candidates": int(len(predict_rows)),
        "pit_universe": "rolling amount top1000, point-in-time",
        "hmm_ranks_source": str(HMM_RANK_PATH),
        "hmm_ranks": ranks,
        "hmm_predicted_state": int(state_row["predicted_state"]),
        "hmm_state_probability_0": float(state_row["state_probability_0"]),
        "hmm_state_probability_1": float(state_row["state_probability_1"]),
        "hmm_state_probability_2": float(state_row["state_probability_2"]),
        "hmm_exposure": hmm_exposure,
        "risk_gate": risk_gate,
        "payoff_gate": strategy_payoff_gate,
        "market_payoff_gate_audit": market_payoff_gate,
        "final_exposure": final_exposure,
        "fit_quality": fit_quality,
        "risk_gate_inputs": {
            **strategy_payoff_info,
            "market_gate_audit": {
                "gate_type": "market_payoff_gate",
                "payoff_gate": market_payoff_gate,
                "payoff_gate_score": market_payoff_info.get("payoff_gate_score"),
                "market_breadth_20": market_payoff_info.get("market_breadth_20"),
                "market_breadth_20_bucket": market_payoff_info.get("market_breadth_20_bucket"),
                "reversal_strength_20": market_payoff_info.get("reversal_strength_20"),
                "reversal_strength_20_bucket": market_payoff_info.get("reversal_strength_20_bucket"),
                "market_vol_20": market_payoff_info.get("market_vol_20"),
                "market_vol_20_bucket": market_payoff_info.get("market_vol_20_bucket"),
                "turnover_chg_5_20": market_payoff_info.get("turnover_chg_5_20"),
                "turnover_chg_5_20_bucket": market_payoff_info.get("turnover_chg_5_20_bucket"),
            },
        },
        "next_day_fillability_note": "Cannot verify next-day suspension/limit-up/open fill until next trading day data is available.",
    }
    (output / "signal_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    log(
        f"final_exposure={final_exposure:.2f} hmm_exposure={hmm_exposure:.2f} "
        f"strategy_payoff_gate={strategy_payoff_gate:.2f} "
        f"market_payoff_gate_audit={market_payoff_gate:.2f} score_direction={score_direction:+.0f}"
    )
    log(f"wrote output={output}")
    print(f"run_dir={output}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0] if args else None)
