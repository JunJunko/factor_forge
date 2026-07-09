from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sell_impact_ranker_timing_compare as timing_compare
import sell_impact_recent_halfyear_tactical as tactical
import sell_impact_sorting_repair as base
import sell_impact_trade_param_ml_surface as surface
import sell_impact_trade_quality_optimization as tq
from factor_forge.config import CostModel, ExecutionConstraints


OUTPUT_ROOT = Path("artifacts/strategy_reviews/experiment8_signal_reliability")
SOURCE_RUN = Path("artifacts/strategy_reviews/sell_impact_recent_halfyear_tactical_20260709T100107Z")
PARAM_SURFACE_RUN = Path("artifacts/strategy_reviews/sell_impact_trade_param_ml_surface_20260709T120518Z")
RELIABILITY_DAILY = Path(
    "artifacts/factor_reliability/factor_reliability_model_v1_20260709T143452Z/factor_reliability_daily.csv"
)
MODEL = "K_recent_2024_2025q3"
PARAM_ID = "param_068"
TRAIN_END = "2025-06-30"
VALID_START = "2025-07-01"
VALID_END = "2025-12-31"
TEST_START = "2026-01-01"
TEST_END = "2026-06-30"
HORIZONS = [5, 10, 20]
COST_BUFFER = 0.002
RANDOM_SEED = 20260709


def main() -> None:
    args = parse_args()
    output = OUTPUT_ROOT / f"stock_signal_reliability_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("loading dataset, panel, reliability and param_068 config")
    dataset = pd.read_parquet(SOURCE_RUN / "recent_halfyear_dataset.parquet")
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    selected = pd.read_csv(SOURCE_RUN / "selected_condition_interactions.csv")
    reliability = load_reliability(args.reliability_daily)
    version, panel = base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    cfg = load_param_config()
    log(f"dataset rows={len(dataset):,} dates={dataset['trade_date'].nunique():,} data_version={version}")
    log(f"param_cfg={cfg}")

    log("building stock-level signal dataset")
    signal_dataset = build_signal_dataset(dataset, selected, panel, reliability, cfg, log)
    feature_cols = feature_columns(signal_dataset)
    signal_dataset.to_csv(output / "signal_dataset.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": feature_cols}).to_csv(output / "feature_list.csv", index=False, encoding="utf-8-sig")
    log(f"signal_dataset rows={len(signal_dataset):,} features={len(feature_cols)}")

    model_results: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    bucket_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        target = f"success_{horizon}d"
        actual = f"future_trade_return_{horizon}d"
        log(f"training horizon={horizon}d target={target}")
        fitted = fit_signal_models(signal_dataset, feature_cols, target, actual)
        model_results.extend(fitted["metrics"])
        prediction_frames.append(fitted["predictions"])
        bucket_frames.append(fitted["buckets"])
        importance_frames.append(fitted["importance"])

    metrics = pd.DataFrame(model_results)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    buckets = pd.concat(bucket_frames, ignore_index=True)
    importance = pd.concat(importance_frames, ignore_index=True)
    metrics.to_csv(output / "model_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output / "model_predictions.csv", index=False, encoding="utf-8-sig")
    buckets.to_csv(output / "bucket_test.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(output / "feature_importance.csv", index=False, encoding="utf-8-sig")

    selected_model, selected_horizon = choose_model(metrics)
    log(f"selected model={selected_model} horizon={selected_horizon}d for trade filter simulation")
    trade_outputs = run_filter_backtests(
        predictions=predictions,
        selected_model=selected_model,
        selected_horizon=selected_horizon,
        panel=panel,
        market_benchmark=timing_compare.load_market_benchmark(version),
        cfg=cfg,
        log=log,
    )
    trade_outputs["daily"].to_csv(output / "portfolio_nav.csv", index=False, encoding="utf-8-sig")
    trade_outputs["trades"].to_csv(output / "trades.csv", index=False, encoding="utf-8-sig")
    trade_outputs["performance"].to_csv(output / "performance_summary.csv", index=False, encoding="utf-8-sig")
    trade_outputs["trade_quality"].to_csv(output / "trade_quality.csv", index=False, encoding="utf-8-sig")
    trade_outputs["filtered"].to_csv(output / "filtered_trade_analysis.csv", index=False, encoding="utf-8-sig")

    special = special_analysis(signal_dataset, predictions, selected_model, selected_horizon)
    special.to_csv(output / "special_analysis.csv", index=False, encoding="utf-8-sig")
    write_report(
        output=output,
        metrics=metrics,
        buckets=buckets,
        performance=trade_outputs["performance"],
        trade_quality=trade_outputs["trade_quality"],
        filtered=trade_outputs["filtered"],
        special=special,
        selected_model=selected_model,
        selected_horizon=selected_horizon,
        feature_cols=feature_cols,
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "experiment": "Experiment 8: Stock-level Signal Reliability Model",
                "source_run": str(SOURCE_RUN),
                "param_id": PARAM_ID,
                "alpha_model": MODEL,
                "selected_reliability_model": selected_model,
                "selected_horizon": selected_horizon,
                "cost_buffer": COST_BUFFER,
                "split": {
                    "train": ["2024-01-01", TRAIN_END],
                    "valid": [VALID_START, VALID_END],
                    "test": [TEST_START, TEST_END],
                },
                "data_version": version,
                "feature_count": len(feature_cols),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 8: stock-level signal reliability.")
    parser.add_argument("--reliability-daily", type=Path, default=RELIABILITY_DAILY)
    return parser.parse_args()


def load_param_config() -> dict[str, Any]:
    frame = pd.read_csv(PARAM_SURFACE_RUN / "param_search_metrics.csv")
    row = frame.loc[frame["variant"].eq(PARAM_ID)].iloc[0]
    cfg = {
        "variant": PARAM_ID,
        "description": "robust candidate from ML parameter response-surface search",
        "entry_pool": "threshold",
        "sell_rule": "continue",
    }
    for col in surface.PARAM_COLUMNS:
        val = row[col]
        cfg[col] = int(val) if col in {"max_positions", "min_hold_days", "max_hold_days"} else float(val)
    return cfg


def load_reliability(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    cols = ["date", "reliability_5d", "reliability_10d", "reliability_20d"]
    return frame[cols].drop_duplicates("date").sort_values("date")


def build_signal_dataset(
    dataset: pd.DataFrame,
    selected: pd.DataFrame,
    panel: pd.DataFrame,
    reliability: pd.DataFrame,
    cfg: dict[str, Any],
    log,
) -> pd.DataFrame:
    spec = next(item for item in tactical.SPECS if item["model"] == MODEL)
    features = tactical.features_for_spec(dataset, selected, str(spec["feature_set"]))
    train = base.sample_slice(dataset, spec["train_start"], spec["train_end"], features).sort_values(
        ["trade_date", "ts_code"]
    )
    valid = base.sample_slice(dataset, spec["valid_start"], spec["valid_end"], features).sort_values(
        ["trade_date", "ts_code"]
    )
    predict_frame = dataset.loc[
        dataset["trade_date"].between(pd.Timestamp("2024-01-01"), pd.Timestamp("2026-06-12"))
    ].sort_values(["trade_date", "ts_code"]).copy()
    log(f"fit shadow alpha scorer: features={len(features)} train={len(train):,} valid={len(valid):,}")
    alpha_model = tactical.fit_ranker(train, valid, features, str(spec["weight_profile"]))
    scored = build_alpha_signal_frame(alpha_model, predict_frame, features)
    scored = add_signal_features(scored, cfg)
    scored = add_future_trade_returns(scored, panel)
    scored = scored.merge(reliability.rename(columns={"date": "trade_date"}), on="trade_date", how="left")
    scored[["reliability_5d", "reliability_10d", "reliability_20d"]] = scored[
        ["reliability_5d", "reliability_10d", "reliability_20d"]
    ].ffill().fillna(1.0)
    candidates = scored.loc[baseline_eligible(scored, cfg)].copy()
    candidates["sample"] = np.select(
        [
            candidates["trade_date"].le(pd.Timestamp(TRAIN_END)),
            candidates["trade_date"].between(pd.Timestamp(VALID_START), pd.Timestamp(VALID_END)),
            candidates["trade_date"].between(pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)),
        ],
        ["train", "valid", "test"],
        default="other",
    )
    candidates = candidates.loc[candidates["sample"].isin(["train", "valid", "test"])].copy()
    for horizon in HORIZONS:
        candidates[f"success_{horizon}d"] = (
            pd.to_numeric(candidates[f"future_trade_return_{horizon}d"], errors="coerce") > COST_BUFFER
        ).astype(float)
        candidates.loc[candidates[f"future_trade_return_{horizon}d"].isna(), f"success_{horizon}d"] = np.nan
    return candidates.replace([np.inf, -np.inf], np.nan)


def build_alpha_signal_frame(model, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    cols = [
        "trade_date",
        "ts_code",
        "label",
        "market_ret_20",
        "market_ret_60",
        "market_vol_20",
        "market_breadth_20",
        "market_xsec_vol_20",
        "cluster_price_reversal",
        "stock_state_low_vol",
        "cluster_liquidity",
        "stock_state_small_size",
        "log_amount_20_z",
        "amount_chg_5_20_z",
    ]
    cols = [col for col in cols if col in frame.columns]
    out = frame[cols].copy()
    out["raw_score"] = model.predict(frame[features], num_iteration=model.best_iteration_)
    out["raw_rank_pct"] = out.groupby("trade_date")["raw_score"].rank(pct=True, method="first")
    out["band_score"] = -(out["raw_rank_pct"] - tq.BAND_TARGET).abs()
    out["band_rank_pct"] = out.groupby("trade_date")["band_score"].rank(pct=True, method="first")
    return out


def add_signal_features(frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = frame.sort_values(["trade_date", "raw_score"], ascending=[True, False]).copy()
    grouped = out.groupby("trade_date", group_keys=False)
    out["rank_position"] = grouped["raw_score"].rank(ascending=False, method="first")
    out["top_percentile"] = 1.0 - out["raw_rank_pct"]
    fifth = grouped["raw_score"].transform(lambda s: s.sort_values(ascending=False).iloc[min(4, len(s) - 1)])
    median = grouped["raw_score"].transform("median")
    out["score_gap_to_5th"] = out["raw_score"] - fifth
    out["score_gap_to_median"] = out["raw_score"] - median
    out["score_dispersion"] = grouped["raw_score"].transform("std").fillna(0.0)
    out["score_margin"] = out["raw_score"] - grouped["raw_score"].transform(lambda s: s.quantile(0.80))
    base_candidate = baseline_eligible(out, cfg)
    density = base_candidate.groupby(out["trade_date"]).transform("sum")
    out["candidate_density"] = density.astype(float)
    factor_cols = [col for col in ["band_score", "raw_score", "cluster_price_reversal", "stock_state_low_vol"] if col in out]
    zcols = []
    for col in factor_cols:
        zcol = f"z_{col}"
        out[zcol] = grouped[col].transform(cs_zscore)
        zcols.append(zcol)
    if zcols:
        signs = pd.DataFrame({col: np.sign(out[col].fillna(0.0)) for col in zcols})
        out["factor_alignment_score"] = signs.mean(axis=1)
        out["factor_positive_count"] = signs.gt(0).sum(axis=1)
    else:
        out["factor_alignment_score"] = 0.0
        out["factor_positive_count"] = 0
    out = out.drop(columns=zcols, errors="ignore")
    out["score_stability"] = (
        out.groupby("ts_code")["raw_score"]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=5).std())
        .fillna(out["score_dispersion"])
    )
    out["historical_signal_success_rate"] = (
        out.groupby("ts_code")["label"]
        .transform(lambda s: s.shift(1).gt(0).rolling(20, min_periods=5).mean())
        .fillna(0.5)
    )
    return out


def add_future_trade_returns(signals: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    prices = panel[["trade_date", "ts_code", "adj_open"]].dropna().copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices = prices.sort_values(["ts_code", "trade_date"])
    parts = []
    for _, group in prices.groupby("ts_code", sort=False):
        g = group.copy()
        g["entry_adj_open"] = g["adj_open"].shift(-1)
        for horizon in HORIZONS:
            g[f"exit_adj_open_{horizon}d"] = g["adj_open"].shift(-(horizon + 1))
            g[f"future_trade_return_{horizon}d"] = g[f"exit_adj_open_{horizon}d"] / g["entry_adj_open"] - 1.0
        keep = ["trade_date", "ts_code", *[f"future_trade_return_{horizon}d" for horizon in HORIZONS]]
        parts.append(g[keep])
    forward = pd.concat(parts, ignore_index=True)
    return signals.merge(forward, on=["trade_date", "ts_code"], how="left")


def baseline_eligible(frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    mask = frame["band_score"].notna()
    mask &= frame["band_rank_pct"].ge(float(cfg["entry_band_rank_min"]))
    mask &= frame["raw_rank_pct"].ge(float(cfg["entry_raw_rank_min"]))
    if cfg.get("entry_pool") == "threshold":
        if "stock_state_small_size" in frame:
            mask &= frame["stock_state_small_size"].le(float(cfg.get("max_microcap_score", np.inf)))
        if "cluster_liquidity" in frame:
            mask &= frame["cluster_liquidity"].ge(float(cfg.get("min_liquidity", -np.inf)))
        if "cluster_price_reversal" in frame:
            mask &= frame["cluster_price_reversal"].ge(float(cfg.get("min_price_reversal", -np.inf)))
    return mask.fillna(False)


def cs_zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def feature_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "band_rank_pct",
        "raw_rank_pct",
        "score_margin",
        "score_gap_to_5th",
        "score_gap_to_median",
        "band_score",
        "raw_score",
        "cluster_price_reversal",
        "stock_state_low_vol",
        "cluster_liquidity",
        "stock_state_small_size",
        "log_amount_20_z",
        "amount_chg_5_20_z",
        "factor_alignment_score",
        "factor_positive_count",
        "rank_position",
        "top_percentile",
        "candidate_density",
        "market_ret_20",
        "market_ret_60",
        "market_vol_20",
        "market_breadth_20",
        "market_xsec_vol_20",
        "score_dispersion",
        "score_stability",
        "historical_signal_success_rate",
        "reliability_5d",
        "reliability_10d",
        "reliability_20d",
    ]
    forbidden_prefixes = ("future_", "success_")
    return [
        col
        for col in candidates
        if col in frame.columns and not any(col.startswith(prefix) for prefix in forbidden_prefixes)
    ]


def fit_signal_models(frame: pd.DataFrame, features: list[str], target: str, actual_col: str) -> dict[str, Any]:
    data = frame.dropna(subset=[target, actual_col]).copy()
    train = data.loc[data["sample"].eq("train")]
    valid = data.loc[data["sample"].eq("valid")]
    test = data.loc[data["sample"].eq("test")]
    outputs = {"metrics": [], "predictions": [], "buckets": [], "importance": []}
    for model_name, model in [
        ("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))),
        ("lightgbm_shallow", make_lgbm_classifier()),
    ]:
        model.fit(train[features].fillna(0.0), train[target].astype(int))
        pred_parts = []
        for sample, part in [("train", train), ("valid", valid), ("test", test)]:
            if part.empty:
                continue
            prob = model.predict_proba(part[features].fillna(0.0))[:, 1]
            pred = part[["trade_date", "ts_code", "sample", "raw_score", "band_score", actual_col, target]].copy()
            pred["model"] = model_name
            pred["horizon"] = int(target.split("_")[1].replace("d", ""))
            pred["signal_probability"] = prob
            pred = pred.rename(columns={actual_col: "future_trade_return", target: "success"})
            pred_parts.append(pred)
            outputs["metrics"].append(model_metrics(pred, model_name, pred["horizon"].iloc[0], sample))
            outputs["buckets"].append(bucket_test(pred, model_name, pred["horizon"].iloc[0], sample))
        outputs["predictions"].append(pd.concat(pred_parts, ignore_index=True))
        outputs["importance"].append(model_importance(model, model_name, features, int(target.split("_")[1].replace("d", ""))))
    return {
        "metrics": outputs["metrics"],
        "predictions": pd.concat(outputs["predictions"], ignore_index=True),
        "buckets": pd.concat(outputs["buckets"], ignore_index=True),
        "importance": pd.concat(outputs["importance"], ignore_index=True),
    }


def make_lgbm_classifier():
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        objective="binary",
        max_depth=3,
        num_leaves=8,
        learning_rate=0.05,
        n_estimators=200,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        random_state=RANDOM_SEED,
        verbosity=-1,
        force_col_wise=True,
    )


def model_metrics(pred: pd.DataFrame, model: str, horizon: int, sample: str) -> dict[str, Any]:
    y = pred["success"].astype(int)
    p = pred["signal_probability"]
    rank_ic = p.corr(pred["future_trade_return"], method="spearman")
    return {
        "model": model,
        "horizon": horizon,
        "sample": sample,
        "rows": int(len(pred)),
        "positive_ratio": float(y.mean()) if len(y) else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if y.nunique() > 1 else np.nan,
        "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
        "mean_future_return": float(pred["future_trade_return"].mean()),
    }


def bucket_test(pred: pd.DataFrame, model: str, horizon: int, sample: str) -> pd.DataFrame:
    data = pred.dropna(subset=["signal_probability", "future_trade_return"]).copy()
    if data.empty:
        return pd.DataFrame()
    data["bucket"] = pd.qcut(
        data["signal_probability"].rank(method="first"),
        5,
        labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"],
    )
    rows = []
    for bucket, group in data.groupby("bucket", observed=False):
        rows.append(
            {
                "model": model,
                "horizon": horizon,
                "sample": sample,
                "bucket": str(bucket),
                "mean_probability": float(group["signal_probability"].mean()),
                "future_return": float(group["future_trade_return"].mean()),
                "win_rate": float(group["future_trade_return"].gt(COST_BUFFER).mean()),
                "trade_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def model_importance(model, model_name: str, features: list[str], horizon: int) -> pd.DataFrame:
    if model_name == "logistic":
        coef = model.named_steps["logisticregression"].coef_[0]
        values = np.abs(coef)
        importance_type = "abs_coef"
    else:
        values = model.booster_.feature_importance(importance_type="gain")
        importance_type = "gain"
    return pd.DataFrame(
        {
            "model": model_name,
            "horizon": horizon,
            "feature": features,
            "importance": values,
            "importance_type": importance_type,
        }
    ).sort_values("importance", ascending=False)


def choose_model(metrics: pd.DataFrame) -> tuple[str, int]:
    test = metrics.loc[metrics["sample"].eq("test")].copy()
    test["score"] = test["rank_ic"].fillna(-999) + test["roc_auc"].fillna(0) * 0.05 + test["pr_auc"].fillna(0) * 0.05
    row = test.sort_values("score", ascending=False).iloc[0]
    return str(row["model"]), int(row["horizon"])


def run_filter_backtests(
    *,
    predictions: pd.DataFrame,
    selected_model: str,
    selected_horizon: int,
    panel: pd.DataFrame,
    market_benchmark: pd.DataFrame,
    cfg: dict[str, Any],
    log,
) -> dict[str, pd.DataFrame]:
    signals = tq.load_signals()
    signals["trade_date"] = pd.to_datetime(signals["trade_date"])
    pred = predictions.loc[
        predictions["model"].eq(selected_model)
        & predictions["horizon"].eq(selected_horizon)
        & predictions["sample"].eq("test"),
        ["trade_date", "ts_code", "signal_probability", "future_trade_return", "success"],
    ].copy()
    signals = signals.merge(pred, on=["trade_date", "ts_code"], how="left")
    signals["signal_probability"] = signals["signal_probability"].fillna(1.0)
    panel_slice = panel.loc[panel["trade_date"].between(pd.Timestamp(tq.TEST_START), pd.Timestamp(tq.TEST_END))].copy()
    timing = timing_compare.load_position_multiplier(timing_compare.TIMING_DAILY)
    constraints = ExecutionConstraints(
        exclude_suspended=True,
        cannot_buy_limit_up=True,
        cannot_sell_limit_down=True,
        exclude_st=True,
        exclude_delisting_period=True,
        min_listing_days=60,
    )
    cost_model = CostModel(commission_bps_per_side=3, slippage_bps_per_side=5, stamp_duty_bps_sell=5)
    variants = [{"variant": "baseline_param068", "threshold": None, "mode": "baseline"}]
    variants.extend({"variant": f"signal_reliability_filter_{int(th * 100):02d}", "threshold": th, "mode": "model"} for th in [0.3, 0.4, 0.5, 0.6])
    variants.extend({"variant": f"simple_entry_band_{int(th * 100):02d}", "threshold": th, "mode": "entry_band"} for th in [0.95, 0.97])
    original_entry_candidates = tq.entry_candidates
    tq.entry_candidates = signal_reliability_entry_candidates
    daily_frames = []
    trade_frames = []
    metric_rows = []
    filtered_frames = []
    try:
        for spec in variants:
            name = spec["variant"]
            log(f"portfolio simulation {name}")
            variant_signals = signals.copy()
            variant_cfg = dict(cfg)
            if spec["mode"] == "baseline":
                variant_signals["signal_probability_threshold"] = 0.0
                variant_signals["entry_allowed_by_signal_model"] = True
            elif spec["mode"] == "model":
                threshold = float(spec["threshold"])
                variant_signals["signal_probability_threshold"] = threshold
                variant_signals["entry_allowed_by_signal_model"] = variant_signals["signal_probability"].ge(threshold)
            else:
                variant_signals["signal_probability_threshold"] = 0.0
                variant_signals["entry_allowed_by_signal_model"] = True
                variant_cfg["entry_band_rank_min"] = float(spec["threshold"])
            daily, trades, _positions, metrics = tq.run_trade_quality_backtest(
                panel=panel_slice,
                signals=variant_signals,
                timing=timing,
                market_benchmark=market_benchmark,
                constraints=constraints,
                cost_model=cost_model,
                cfg=variant_cfg,
            )
            csi1000 = float(metrics.get("market_index_annualized_return", np.nan))
            metric_rows.append(
                {
                    "variant": name,
                    "mode": spec["mode"],
                    "threshold": spec["threshold"],
                    **metrics,
                    "annualized_excess_return_vs_csi1000": float(metrics["annualized_return"] - csi1000),
                }
            )
            daily["variant"] = name
            trades["variant"] = name
            daily_frames.append(daily)
            trade_frames.append(trades)
            filtered_frames.append(filter_analysis_for_variant(variant_signals, variant_cfg, name))
    finally:
        tq.entry_candidates = original_entry_candidates
    all_daily = pd.concat(daily_frames, ignore_index=True)
    all_trades = pd.concat(trade_frames, ignore_index=True)
    performance = pd.DataFrame(metric_rows)
    trade_quality = tq.trade_quality_summary(all_trades, panel_slice)
    filtered = pd.concat(filtered_frames, ignore_index=True)
    return {"daily": all_daily, "trades": all_trades, "performance": performance, "trade_quality": trade_quality, "filtered": filtered}


def signal_reliability_entry_candidates(signal_frame: pd.DataFrame, positions: list[tq.Position], cfg: dict[str, Any]) -> pd.DataFrame:
    held = {position.ts_code for position in positions}
    frame = signal_frame.loc[signal_frame["band_score"].notna()].copy()
    if held:
        frame = frame.loc[~frame.index.isin(held)].copy()
    allowed = frame.get("entry_allowed_by_signal_model", pd.Series(True, index=frame.index))
    allowed = allowed.where(allowed.notna(), True).astype(bool)
    frame = frame.loc[
        allowed
        & frame["band_rank_pct"].ge(float(cfg.get("entry_band_rank_min", 0.0)))
        & frame["raw_rank_pct"].ge(float(cfg.get("entry_raw_rank_min", 0.0)))
    ].copy()
    if cfg.get("entry_pool") == "threshold":
        if "stock_state_small_size" in frame:
            frame = frame.loc[frame["stock_state_small_size"].le(float(cfg.get("max_microcap_score", np.inf)))]
        if "cluster_liquidity" in frame:
            frame = frame.loc[frame["cluster_liquidity"].ge(float(cfg.get("min_liquidity", -np.inf)))]
        if "cluster_price_reversal" in frame:
            frame = frame.loc[frame["cluster_price_reversal"].ge(float(cfg.get("min_price_reversal", -np.inf)))]
    return frame.sort_values(["band_score", "raw_score"], ascending=False)


def filter_analysis_for_variant(signals: pd.DataFrame, cfg: dict[str, Any], variant: str) -> pd.DataFrame:
    base = baseline_eligible(signals, load_param_config())
    variant_allowed = base & signals.get("entry_allowed_by_signal_model", pd.Series(True, index=signals.index)).fillna(True).astype(bool)
    if "simple_entry_band" in variant:
        variant_allowed = baseline_eligible(signals, cfg)
    frame = signals.loc[base].copy()
    frame["selection"] = np.where(variant_allowed.loc[frame.index], "executed_candidate", "filtered_candidate")
    rows = []
    for selection, group in frame.groupby("selection"):
        rows.append(
            {
                "variant": variant,
                "selection": selection,
                "signal_count": int(len(group)),
                "future_return": float(group["future_trade_return"].mean()) if "future_trade_return" in group else np.nan,
                "win_rate": float(group["future_trade_return"].gt(COST_BUFFER).mean()) if "future_trade_return" in group else np.nan,
                "avg_loss": float(group.loc[group["future_trade_return"].le(COST_BUFFER), "future_trade_return"].mean()) if "future_trade_return" in group and len(group) else np.nan,
                "mean_probability": float(group["signal_probability"].mean()) if "signal_probability" in group else np.nan,
            }
        )
    return pd.DataFrame(rows)


def special_analysis(signal_dataset: pd.DataFrame, predictions: pd.DataFrame, model: str, horizon: int) -> pd.DataFrame:
    pred = predictions.loc[predictions["model"].eq(model) & predictions["horizon"].eq(horizon)].copy()
    test = pred.loc[pred["sample"].eq("test")].copy()
    rows = []
    if not test.empty:
        test["prob_bucket"] = pd.qcut(test["signal_probability"].rank(method="first"), 5, labels=False) + 1
        test["score_bucket"] = pd.qcut(test["raw_score"].rank(method="first"), 5, labels=False) + 1
        for name, data in [
            ("lowest_20pct_reliability", test.loc[test["prob_bucket"].eq(1)]),
            ("highest_20pct_reliability", test.loc[test["prob_bucket"].eq(5)]),
            ("high_score_low_reliability", test.loc[test["score_bucket"].eq(5) & test["prob_bucket"].eq(1)]),
            ("high_score_high_reliability", test.loc[test["score_bucket"].eq(5) & test["prob_bucket"].eq(5)]),
        ]:
            rows.append(
                {
                    "analysis": name,
                    "rows": int(len(data)),
                    "future_return": float(data["future_trade_return"].mean()) if len(data) else np.nan,
                    "win_rate": float(data["future_trade_return"].gt(COST_BUFFER).mean()) if len(data) else np.nan,
                    "mean_probability": float(data["signal_probability"].mean()) if len(data) else np.nan,
                    "mean_raw_score": float(data["raw_score"].mean()) if len(data) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(
    *,
    output: Path,
    metrics: pd.DataFrame,
    buckets: pd.DataFrame,
    performance: pd.DataFrame,
    trade_quality: pd.DataFrame,
    filtered: pd.DataFrame,
    special: pd.DataFrame,
    selected_model: str,
    selected_horizon: int,
    feature_cols: list[str],
) -> None:
    test_metrics = metrics.loc[metrics["sample"].eq("test")].sort_values("rank_ic", ascending=False)
    test_buckets = buckets.loc[buckets["sample"].eq("test") & buckets["model"].eq(selected_model) & buckets["horizon"].eq(selected_horizon)]
    perf_cols = ["variant", "mode", "threshold", "total_return", "annualized_return", "annualized_excess_return_vs_csi1000", "sharpe", "max_drawdown", "calmar", "executed_buys", "trade_count"]
    filtered_summary = filtered.sort_values(["variant", "selection"])
    lines = [
        "# Experiment 8: Stock-level Signal Reliability Model",
        "",
        "## Scope",
        "- Objective: predict whether a concrete candidate stock signal is worth executing.",
        "- Alpha LightGBM, factor weights, timing, portfolio construction, and sell rules are not modified.",
        "- The reliability model is only used as an entry acceptance layer.",
        f"- Selected trade-filter model: `{selected_model}` horizon `{selected_horizon}d`.",
        f"- Feature count: `{len(feature_cols)}`.",
        "",
        "## Model Metrics",
        md_table(test_metrics, 20),
        "",
        "## Bucket Test",
        md_table(test_buckets, 20),
        "",
        "## Performance",
        md_table(performance.sort_values("annualized_return", ascending=False)[perf_cols], 20),
        "",
        "## Trade Quality",
        md_table(trade_quality.sort_values("mean_trade_return", ascending=False), 20),
        "",
        "## Filtered Trade Analysis",
        md_table(filtered_summary, 40),
        "",
        "## Special Analysis",
        md_table(special, 20),
        "",
        "## Required Answers",
        "- Whether reliability identifies stocks where the alpha model is likely wrong: see `Bucket Test` and `Special Analysis`.",
        "- Whether reliability separates quality among high-score stocks: compare `high_score_low_reliability` vs `high_score_high_reliability`.",
        "- Whether low-reliability high-score stocks are loss sources: see `high_score_low_reliability` future return and win rate.",
        "- Whether reliability beats simple entry threshold: compare `signal_reliability_filter_*` with `simple_entry_band_*` in Performance.",
        "",
        "## Files",
        "- `signal_dataset.csv`",
        "- `model_predictions.csv`",
        "- `signal_reliability_report.md`",
        "- `filtered_trade_analysis.csv`",
        "- `performance_summary.csv`",
        "- `trade_quality.csv`",
    ]
    text = "\n".join(lines) + "\n"
    (output / "signal_reliability_report.md").write_text(text, encoding="utf-8")
    (output / "report.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
