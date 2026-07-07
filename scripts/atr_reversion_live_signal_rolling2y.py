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

from atr_reversion_defensive_gate import _risk_kill_only_gate
from atr_reversion_pit_hmm_calibrated_backtest import _market_features, _tiered_weight, _walk_forward_hmm
from atr_reversion_pit_liquidity_backtest import _attach_and_preprocess_pit, _pit_liquidity_flags
from atr_reversion_strategy_regime_mining import _build_regime_features
from atr_reversion_walk_forward import REBALANCE_DAYS


SIGNAL_DATE: str | None = None
TRAIN_YEARS = 2
TOP_N = 5
FIT_QUALITY_LOOKBACK = 40
FIT_QUALITY_MIN_OBS = 15
EXCLUDED_BOARDS = ["STAR 688/689.SH", "ChiNext 300/301/302.SZ", "Beijing *.BJ"]
OUTPUT_ROOT = Path("artifacts/atr_reversion_live_signals")
CONFIG_PATH = "configs/ml/atr_reversion_lightgbm_v1.yaml"
FIT_QUALITY_SOURCE_RUN = Path(
    "artifacts/atr_reversion_runs/"
    "atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/"
    "hmm_window_comparison_20260707T002239Z"
)
HMM_RANK_PATH = Path(
    "artifacts/atr_reversion_runs/"
    "atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/"
    "training_window_experiment_20260706T130710Z/"
    "rolling_2y/test_2026h1/state_validation_perf_cost10.csv"
)
HEALTH_SCORE_PATH = Path(
    "artifacts/atr_reversion_runs/"
    "atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/"
    "three_layer_gate_20260706T113826Z/"
    "gate_scores_test_2026h1_cost10.csv"
)


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
    for path in sorted(FIT_QUALITY_SOURCE_RUN.glob("test_*/predictions_valid_test_rolling_2y.parquet")):
        pred = pd.read_parquet(path, columns=["trade_date", "ts_code", "factor_value"])
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


def _train_model(dataset: pd.DataFrame, features: list[str], train_start: pd.Timestamp, signal_date: pd.Timestamp, cfg, log):
    from lightgbm import LGBMRegressor

    cols = ["datetime", "instrument", *features, "label", "sample_weight", "pit_top1000"]
    eligible = dataset.loc[
        dataset["datetime"].between(train_start, signal_date - pd.Timedelta(days=1))
        & dataset["pit_top1000"],
        cols,
    ].dropna(subset=[*features, "label"])
    if eligible.empty:
        raise ValueError("No realized training rows after applying rolling window and PIT filters.")

    params = cfg.model.model_dump()
    params.pop("objective", None)
    params.setdefault("verbosity", -1)
    params.setdefault("force_col_wise", True)
    model = LGBMRegressor(objective="regression", **params)
    fit_kwargs = {"sample_weight": eligible["sample_weight"].fillna(1.0)}
    log(
        "training LightGBM "
        f"rows={len(eligible):,} "
        f"dates={eligible['datetime'].min().date()}..{eligible['datetime'].max().date()}"
    )
    model.fit(eligible[features], eligible["label"], **fit_kwargs)
    return model, eligible


def main(signal_date: str | None = SIGNAL_DATE, top_n: int = TOP_N) -> None:
    preloaded: tuple[str, pd.DataFrame] | None = None
    if signal_date is None:
        preloaded = _load_panel("latest")
        signal_ts = pd.Timestamp(preloaded[1]["trade_date"].max()).normalize()
    else:
        signal_ts = pd.Timestamp(signal_date)
    signal_date_text = f"{signal_ts:%Y-%m-%d}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = OUTPUT_ROOT / f"rolling2y_risk_kill_signal_{signal_ts:%Y%m%d}_{stamp}"
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

    model, train = _train_model(dataset, features, train_start, signal_ts, cfg, log)

    predict_rows = dataset.loc[
        dataset["datetime"].eq(signal_ts) & dataset["pit_top1000"],
        ["datetime", "instrument", *features, "pit_top1000"],
    ].dropna(subset=features)
    predict_rows = predict_rows[predict_rows["instrument"].map(_permission_eligible)].copy()
    if predict_rows.empty:
        raise ValueError(f"No predictable PIT rows on {signal_ts.date()}.")
    log(f"predicting signal_date={signal_ts.date()} candidates={len(predict_rows):,}")
    pred = predict_rows[["datetime", "instrument"]].rename(columns={"datetime": "trade_date", "instrument": "ts_code"})
    pred["factor_value"] = model.predict(predict_rows[features])

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
    states = _walk_forward_hmm(market, signal_date_text, signal_date_text, log)
    state_row = states.loc[states["trade_date"].eq(signal_ts)].iloc[-1]
    ranks = _hmm_ranks(HMM_RANK_PATH)
    hmm_exposure = float(_tiered_weight(state_row, ranks))

    log("computing risk-kill gate inputs")
    regime = _build_regime_features(regime_panel, regime_pit, dataset)
    regime_row = regime.loc[regime["trade_date"].eq(signal_ts)].iloc[-1].to_dict()
    health = _latest_health_value(HEALTH_SCORE_PATH, signal_ts)
    gate_row = pd.Series({**regime_row, **health})
    risk_gate = float(_risk_kill_only_gate(gate_row))
    final_exposure = hmm_exposure * risk_gate

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
    top[cols].to_csv(output / "top_recommendations.csv", index=False, encoding="utf-8-sig")
    pred.head(100).merge(keep_panel, on=["trade_date", "ts_code"], how="left").to_csv(
        output / "top100_candidates.csv", index=False, encoding="utf-8-sig"
    )
    states.to_csv(output / "hmm_state.csv", index=False, encoding="utf-8-sig")

    summary = {
        "signal_date": signal_ts,
        "intended_execution": "next_trade_day_open",
        "data_version": version,
        "model": "LightGBM rolling_2y",
        "signal_algorithm": "rolling_2y + HMM rolling_3y PIT + risk_kill + frozen fit-quality flip",
        "permission_filter": {
            "enabled": True,
            "excluded_boards": EXCLUDED_BOARDS,
        },
        "train_start_requested": train_start,
        "train_start_actual": train["datetime"].min(),
        "train_end_actual": train["datetime"].max(),
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
        "final_exposure": final_exposure,
        "fit_quality": fit_quality,
        "risk_gate_inputs": {
            "xsec_vol_20": float(gate_row.get("xsec_vol_20", np.nan)),
            "market_ret_20": float(gate_row.get("market_ret_20", np.nan)),
            "market_ret_60": float(gate_row.get("market_ret_60", np.nan)),
            "momentum_minus_reversal_20": float(gate_row.get("momentum_minus_reversal_20", np.nan)),
            "top5_excess_5round": float(gate_row.get("top5_excess_5round", np.nan)),
            "health_source_date": gate_row.get("health_source_date"),
            "health_source_path": gate_row.get("health_source_path"),
        },
        "next_day_fillability_note": "Cannot verify next-day suspension/limit-up/open fill until next trading day data is available.",
    }
    (output / "signal_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    log(
        f"final_exposure={final_exposure:.2f} hmm_exposure={hmm_exposure:.2f} "
        f"risk_gate={risk_gate:.2f} score_direction={score_direction:+.0f}"
    )
    log(f"wrote output={output}")
    print(f"run_dir={output}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0] if args else None)
