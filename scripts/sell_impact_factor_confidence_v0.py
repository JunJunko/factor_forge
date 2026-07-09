from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sell_impact_recent_halfyear_tactical as tactical
import sell_impact_ranker_timing_compare as timing_compare
import sell_impact_sorting_repair as base
import sell_impact_trade_param_ml_surface as surface
import sell_impact_trade_quality_optimization as tq
from factor_forge.config import CostModel, ExecutionConstraints


SOURCE_RUN = Path("artifacts/strategy_reviews/sell_impact_recent_halfyear_tactical_20260709T100107Z")
PARAM_SURFACE_RUN = Path("artifacts/strategy_reviews/sell_impact_trade_param_ml_surface_20260709T120518Z")
OUTPUT_ROOT = Path("artifacts/strategy_reviews")
MODEL = "K_recent_2024_2025q3"
PARAM_ID = "param_068"
TEST_START = "20260101"
TEST_END = "20260623"
BAND_TARGET = 0.95
LABEL_DELAY_TRADING_DAYS = 10
PRIOR_MIN_OBS = 35
RECENT_WINDOW = 20
RECENT_STABILITY_WINDOW = 60
PRIOR_STRENGTH_K = 40


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_factor_confidence_v0_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"source_run={SOURCE_RUN}")
    dataset = pd.read_parquet(SOURCE_RUN / "recent_halfyear_dataset.parquet")
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    selected = pd.read_csv(SOURCE_RUN / "selected_condition_interactions.csv")
    log(f"dataset rows={len(dataset):,} dates={dataset['trade_date'].nunique():,}")

    scored = build_band_scores(dataset, selected, log)
    daily_metrics = daily_factor_metrics(scored)
    health = build_factor_health_daily(daily_metrics)
    validation = confidence_bucket_validation(health)
    health.to_csv(output / "factor_health_daily.csv", index=False, encoding="utf-8-sig")
    daily_metrics.to_csv(output / "band_score_daily_payoff.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(output / "confidence_bucket_validation.csv", index=False, encoding="utf-8-sig")
    log(f"health rows={len(health):,} validation rows={len(validation):,}")

    version, panel = base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel_slice = panel.loc[panel["trade_date"].between(pd.Timestamp(TEST_START), pd.Timestamp(TEST_END))].copy()
    timing = timing_compare.load_position_multiplier(timing_compare.TIMING_DAILY)
    market_benchmark = timing_compare.load_market_benchmark(version)
    constraints = ExecutionConstraints(
        exclude_suspended=True,
        cannot_buy_limit_up=True,
        cannot_sell_limit_down=True,
        exclude_st=True,
        exclude_delisting_period=True,
        min_listing_days=60,
    )
    cost_model = CostModel(commission_bps_per_side=3, slippage_bps_per_side=5, stamp_duty_bps_sell=5)
    cfg = load_param_config()
    signals = tq.load_signals()

    log("running baseline param_068")
    daily_base, trades_base, positions_base, metrics_base = tq.run_trade_quality_backtest(
        panel=panel_slice,
        signals=signals,
        timing=timing,
        market_benchmark=market_benchmark,
        constraints=constraints,
        cost_model=cost_model,
        cfg=cfg,
    )

    confidence_signal = health[["trade_date", "confidence", "entry_band_rank_min_dynamic"]].copy()
    confidence_signal["trade_date"] = pd.to_datetime(confidence_signal["trade_date"])
    signals_dynamic = signals.merge(confidence_signal, on="trade_date", how="left")
    signals_dynamic[["confidence", "entry_band_rank_min_dynamic"]] = (
        signals_dynamic.groupby("ts_code")[["confidence", "entry_band_rank_min_dynamic"]].ffill()
    )
    signals_dynamic["confidence"] = signals_dynamic["confidence"].fillna(0.50)
    signals_dynamic["entry_band_rank_min_dynamic"] = signals_dynamic["entry_band_rank_min_dynamic"].fillna(0.95)

    confidence_variants = [
        ("confidence_v0_dynamic_entry", {}),
        ("confidence_gate_min065", {"min_entry_confidence": 0.65}),
        ("confidence_gate_min070", {"min_entry_confidence": 0.70}),
        ("confidence_gate_min075", {"min_entry_confidence": 0.75}),
    ]
    confidence_results: list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]] = []
    original_entry_candidates = tq.entry_candidates
    tq.entry_candidates = dynamic_entry_candidates
    try:
        for variant, extra_cfg in confidence_variants:
            log(f"running {variant}")
            daily_conf, trades_conf, positions_conf, metrics_conf = tq.run_trade_quality_backtest(
                panel=panel_slice,
                signals=signals_dynamic,
                timing=timing,
                market_benchmark=market_benchmark,
                constraints=constraints,
                cost_model=cost_model,
                cfg={**cfg, **extra_cfg, "variant": variant},
            )
            confidence_results.append((variant, daily_conf, trades_conf, positions_conf, metrics_conf))
    finally:
        tq.entry_candidates = original_entry_candidates

    daily_base["variant"] = "baseline_param068"
    trades_base["variant"] = "baseline_param068"
    positions_base["variant"] = "baseline_param068"
    for variant, daily_conf, trades_conf, positions_conf, _metrics_conf in confidence_results:
        daily_conf["variant"] = variant
        trades_conf["variant"] = variant
        positions_conf["variant"] = variant

    metrics = pd.DataFrame(
        [
            {"variant": "baseline_param068", **metrics_base},
            *[
                {"variant": variant, **metrics_conf}
                for variant, _daily_conf, _trades_conf, _positions_conf, metrics_conf in confidence_results
            ],
        ]
    )
    metrics["annualized_excess_return_vs_csi1000"] = metrics["annualized_return"] - metrics[
        "market_index_annualized_return"
    ]
    monthly = pd.concat(
        [
            tq.period_breakdown(daily_base, trades_base, "baseline_param068", "M"),
            *[
                tq.period_breakdown(daily_conf, trades_conf, variant, "M")
                for variant, daily_conf, trades_conf, _positions_conf, _metrics_conf in confidence_results
            ],
        ],
        ignore_index=True,
    )
    all_daily = pd.concat([daily_base, *[item[1] for item in confidence_results]], ignore_index=True)
    all_trades = pd.concat([trades_base, *[item[2] for item in confidence_results]], ignore_index=True)
    all_positions = pd.concat([positions_base, *[item[3] for item in confidence_results]], ignore_index=True)
    trade_quality = tq.trade_quality_summary(all_trades, panel_slice)
    filtered = pd.concat(
        [
            filtered_trade_audit(trades_base, trades_conf, variant)
            for variant, _daily_conf, trades_conf, _positions_conf, _metrics_conf in confidence_results
        ],
        ignore_index=True,
    )

    metrics.to_csv(output / "confidence_backtest_metrics.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "confidence_monthly.csv", index=False, encoding="utf-8-sig")
    trade_quality.to_csv(output / "confidence_trade_quality.csv", index=False, encoding="utf-8-sig")
    filtered.to_csv(output / "confidence_filtered_trade_audit.csv", index=False, encoding="utf-8-sig")
    all_daily.to_parquet(output / "confidence_daily.parquet", index=False)
    all_trades.to_csv(output / "confidence_trades.csv", index=False, encoding="utf-8-sig")
    all_positions.to_parquet(output / "confidence_positions.parquet", index=False)

    write_report(output, metrics, monthly, trade_quality, validation, filtered, cfg)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "source_run": str(SOURCE_RUN),
                "param_surface_run": str(PARAM_SURFACE_RUN),
                "model": MODEL,
                "param_id": PARAM_ID,
                "confidence_model": "band_score_confidence_v0",
                "confidence_formula": "regime prior + delayed rolling IC/spread evidence + dispersion diagnostics; dynamic entry threshold = 0.98 - 0.06 * confidence",
                "label_delay_trading_days": LABEL_DELAY_TRADING_DAYS,
                "prior_min_obs": PRIOR_MIN_OBS,
                "recent_window": RECENT_WINDOW,
                "recent_stability_window": RECENT_STABILITY_WINDOW,
                "prior_strength_k": PRIOR_STRENGTH_K,
                "test_window": [TEST_START, TEST_END],
                "data_version": version,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


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


def build_band_scores(dataset: pd.DataFrame, selected: pd.DataFrame, log) -> pd.DataFrame:
    spec = next(item for item in tactical.SPECS if item["model"] == MODEL)
    features = tactical.features_for_spec(dataset, selected, str(spec["feature_set"]))
    train = base.sample_slice(dataset, spec["train_start"], spec["train_end"], features).sort_values(
        ["trade_date", "ts_code"]
    )
    valid = base.sample_slice(dataset, spec["valid_start"], spec["valid_end"], features).sort_values(
        ["trade_date", "ts_code"]
    )
    log(f"fit shadow scorer for confidence: features={len(features)} train={len(train):,} valid={len(valid):,}")
    model = tactical.fit_ranker(train, valid, features, str(spec["weight_profile"]))
    frame = dataset[
        [
            "trade_date",
            "ts_code",
            "label",
            "market_ret_20",
            "market_ret_60",
            "market_vol_20",
            "market_breadth_20",
            "market_xsec_vol_20",
        ]
    ].copy()
    frame["raw_score"] = model.predict(dataset[features], num_iteration=model.best_iteration_)
    frame["raw_rank_pct"] = frame.groupby("trade_date")["raw_score"].rank(pct=True, method="first")
    frame["band_score"] = -(frame["raw_rank_pct"] - BAND_TARGET).abs()
    frame["band_rank_pct"] = frame.groupby("trade_date")["band_score"].rank(pct=True, method="first")
    return frame.replace([np.inf, -np.inf], np.nan)


def daily_factor_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    state = (
        scored.groupby("trade_date", as_index=False)
        .agg(
            market_ret_20=("market_ret_20", "median"),
            market_ret_60=("market_ret_60", "median"),
            market_vol_20=("market_vol_20", "median"),
            market_breadth_20=("market_breadth_20", "median"),
            market_xsec_vol_20=("market_xsec_vol_20", "median"),
        )
        .sort_values("trade_date")
    )
    state = add_regime_buckets(state)

    rows = []
    for date, group in scored.groupby("trade_date", sort=True):
        data = group[["label", "band_score", "band_rank_pct"]].dropna()
        row: dict[str, Any] = {"trade_date": date, "factor": "band_score", "n_stocks": int(len(data))}
        if len(data) < 50:
            rows.append(row)
            continue
        row["rank_ic"] = float(data["band_score"].corr(data["label"], method="spearman"))
        row["pearson_ic"] = float(data["band_score"].corr(data["label"], method="pearson"))
        row["score_dispersion"] = float(data["band_score"].std(ddof=1))
        top = data.loc[data["band_rank_pct"].ge(0.90), "band_score"].mean()
        mid = data.loc[data["band_rank_pct"].between(0.25, 0.75), "band_score"].mean()
        row["top_score_gap"] = float(top - mid) if pd.notna(top) and pd.notna(mid) else np.nan
        try:
            data = data.copy()
            data["decile"] = pd.qcut(data["band_score"].rank(method="first"), 10, labels=False) + 1
            avg = data.groupby("decile")["label"].mean()
            row["top_decile_return"] = float(avg.get(10, np.nan))
            row["bottom_decile_return"] = float(avg.get(1, np.nan))
            row["top_bottom_spread"] = float(avg.get(10, np.nan) - avg.get(1, np.nan))
            row["decile_monotonicity"] = float(pd.Series(avg.index.astype(float)).corr(pd.Series(avg.values)))
        except ValueError:
            row["top_decile_return"] = np.nan
            row["bottom_decile_return"] = np.nan
            row["top_bottom_spread"] = np.nan
            row["decile_monotonicity"] = np.nan
        rows.append(row)
    metrics = pd.DataFrame(rows).merge(state, on="trade_date", how="left")
    return metrics.sort_values("trade_date").reset_index(drop=True)


def add_regime_buckets(state: pd.DataFrame) -> pd.DataFrame:
    out = state.copy().sort_values("trade_date")
    out["trend_bucket"] = np.where(out["market_ret_20"].ge(0), "trend_up", "trend_down")
    for source, target in [
        ("market_vol_20", "vol_pct_pit"),
        ("market_breadth_20", "breadth_pct_pit"),
        ("market_xsec_vol_20", "dispersion_pct_pit"),
    ]:
        out[target] = expanding_percentile(out[source])
    out["vol_bucket"] = np.select(
        [out["vol_pct_pit"].le(0.33), out["vol_pct_pit"].ge(0.67)],
        ["low_vol", "high_vol"],
        default="mid_vol",
    )
    out["breadth_bucket"] = np.select(
        [out["breadth_pct_pit"].le(0.33), out["breadth_pct_pit"].ge(0.67)],
        ["weak_breadth", "broad_breadth"],
        default="neutral_breadth",
    )
    out["dispersion_bucket"] = np.where(out["dispersion_pct_pit"].ge(0.50), "high_dispersion", "low_dispersion")
    out["regime"] = out["trend_bucket"] + "__" + out["vol_bucket"] + "__" + out["breadth_bucket"]
    out["coarse_regime"] = out["trend_bucket"] + "__" + out["vol_bucket"]
    return out


def expanding_percentile(series: pd.Series, min_obs: int = 60) -> pd.Series:
    values = series.astype(float).to_numpy()
    out = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        hist = values[:i]
        hist = hist[np.isfinite(hist)]
        if not np.isfinite(value) or len(hist) < min_obs:
            continue
        out[i] = float((hist <= value).mean())
    fallback = pd.Series(values).rank(pct=True).to_numpy()
    return pd.Series(np.where(np.isfinite(out), out, fallback), index=series.index)


def build_factor_health_daily(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    m = metrics.sort_values("trade_date").reset_index(drop=True)
    for i, row in m.iterrows():
        available = m.iloc[: max(0, i - LABEL_DELAY_TRADING_DAYS)].copy()
        recent20 = available.tail(RECENT_WINDOW)
        recent60 = available.tail(RECENT_STABILITY_WINDOW)
        prior = select_prior(available, str(row.get("regime")), str(row.get("coarse_regime")))
        recent_mu = float(recent20["rank_ic"].mean()) if len(recent20) else np.nan
        recent_std = float(recent60["rank_ic"].std(ddof=1)) if len(recent60) > 1 else np.nan
        recent_spread = float(recent20["top_bottom_spread"].mean()) if len(recent20) else np.nan
        recent_spread60 = float(recent60["top_bottom_spread"].mean()) if len(recent60) else np.nan
        positive_ic_ratio60 = float(recent60["rank_ic"].gt(0).mean()) if len(recent60) else np.nan
        monotonicity60 = float(recent60["decile_monotonicity"].mean()) if len(recent60) else np.nan
        prior_mu = prior["prior_rank_ic_mean"]
        prior_std = prior["prior_rank_ic_std"]
        if not np.isfinite(prior_std) or prior_std <= 1e-6:
            prior_std = float(available["rank_ic"].std(ddof=1)) if len(available) > 1 else 0.10
        if not np.isfinite(recent_std) or recent_std <= 1e-6:
            recent_std = prior_std
        n = int(recent20["rank_ic"].notna().sum())
        w = n / (n + PRIOR_STRENGTH_K) if n > 0 else 0.0
        posterior_mu = w * nz(recent_mu, prior_mu) + (1.0 - w) * nz(prior_mu, 0.0)
        posterior_std = np.sqrt(max(1e-8, w * nz(recent_std, prior_std) ** 2 + (1.0 - w) * nz(prior_std, 0.10) ** 2))
        posterior_z = posterior_mu / posterior_std if posterior_std > 0 else 0.0
        confidence_raw = float(np.clip(sigmoid(1.5 * posterior_z), 0.20, 1.0))
        rows.append(
            {
                "trade_date": row["trade_date"],
                "factor": "band_score",
                "regime": row.get("regime"),
                "coarse_regime": row.get("coarse_regime"),
                "prior_scope": prior["prior_scope"],
                "prior_obs": prior["prior_obs"],
                "prior_rank_ic_mean": prior_mu,
                "prior_rank_ic_std": prior_std,
                "prior_spread_mean": prior["prior_spread_mean"],
                "recent_rank_ic_20": recent_mu,
                "recent_rank_ic_std_60": recent_std,
                "recent_rank_icir_60": recent_mu / recent_std if np.isfinite(recent_mu) and recent_std > 0 else np.nan,
                "positive_ic_ratio_60": positive_ic_ratio60,
                "recent_spread_20": recent_spread,
                "recent_spread_60": recent_spread60,
                "decile_monotonicity_60": monotonicity60,
                "score_dispersion": row.get("score_dispersion"),
                "top_score_gap": row.get("top_score_gap"),
                "posterior_rank_ic": posterior_mu,
                "posterior_rank_ic_std": posterior_std,
                "posterior_z": posterior_z,
                "confidence_raw": confidence_raw,
                "future_rank_ic": row.get("rank_ic"),
                "future_spread": row.get("top_bottom_spread"),
                "future_monotonicity": row.get("decile_monotonicity"),
                "market_ret_20": row.get("market_ret_20"),
                "market_ret_60": row.get("market_ret_60"),
                "market_vol_20": row.get("market_vol_20"),
                "market_breadth_20": row.get("market_breadth_20"),
                "market_xsec_vol_20": row.get("market_xsec_vol_20"),
            }
        )
    health = pd.DataFrame(rows)
    health["confidence"] = health["confidence_raw"].ewm(span=5, adjust=False).mean().clip(0.20, 1.0)
    health["entry_band_rank_min_dynamic"] = (0.98 - 0.06 * health["confidence"]).clip(0.92, 0.98)
    return health


def select_prior(available: pd.DataFrame, regime: str, coarse_regime: str) -> dict[str, Any]:
    for scope, mask in [
        ("regime", available["regime"].eq(regime) if "regime" in available else pd.Series(False, index=available.index)),
        (
            "coarse_regime",
            available["coarse_regime"].eq(coarse_regime)
            if "coarse_regime" in available
            else pd.Series(False, index=available.index),
        ),
        ("global", pd.Series(True, index=available.index)),
    ]:
        group = available.loc[mask].dropna(subset=["rank_ic"])
        if len(group) >= PRIOR_MIN_OBS or scope == "global":
            return {
                "prior_scope": scope,
                "prior_obs": int(len(group)),
                "prior_rank_ic_mean": float(group["rank_ic"].mean()) if len(group) else 0.0,
                "prior_rank_ic_std": float(group["rank_ic"].std(ddof=1)) if len(group) > 1 else 0.10,
                "prior_spread_mean": float(group["top_bottom_spread"].mean()) if len(group) else 0.0,
            }
    raise RuntimeError("unreachable prior fallback")


def dynamic_entry_candidates(signal_frame: pd.DataFrame, positions: list[tq.Position], cfg: dict[str, Any]) -> pd.DataFrame:
    held = {position.ts_code for position in positions}
    frame = signal_frame.loc[signal_frame["band_score"].notna()].copy()
    if held:
        frame = frame.loc[~frame.index.isin(held)].copy()
    static_threshold = float(cfg.get("entry_band_rank_min", 0.0))
    threshold = frame.get("entry_band_rank_min_dynamic", pd.Series(static_threshold, index=frame.index)).fillna(
        static_threshold
    )
    frame = frame.loc[
        frame["band_rank_pct"].ge(threshold)
        & frame["raw_rank_pct"].ge(float(cfg.get("entry_raw_rank_min", 0.0)))
    ].copy()
    if "min_entry_confidence" in cfg and "confidence" in frame:
        frame = frame.loc[frame["confidence"].ge(float(cfg["min_entry_confidence"]))].copy()
    if cfg.get("entry_pool") == "threshold":
        if "stock_state_small_size" in frame:
            frame = frame.loc[frame["stock_state_small_size"].le(float(cfg.get("max_microcap_score", np.inf)))]
        if "cluster_liquidity" in frame:
            frame = frame.loc[frame["cluster_liquidity"].ge(float(cfg.get("min_liquidity", -np.inf)))]
        if "cluster_price_reversal" in frame:
            frame = frame.loc[frame["cluster_price_reversal"].ge(float(cfg.get("min_price_reversal", -np.inf)))]
    return frame.sort_values(["band_score", "raw_score"], ascending=False)


def confidence_bucket_validation(health: pd.DataFrame) -> pd.DataFrame:
    frame = health.loc[health["trade_date"].between(pd.Timestamp(TEST_START), pd.Timestamp("2026-06-12"))].copy()
    frame = frame.dropna(subset=["confidence", "future_rank_ic", "future_spread"])
    if frame.empty:
        return pd.DataFrame()
    frame["bucket"] = pd.qcut(frame["confidence"].rank(method="first"), 5, labels=["q1_low", "q2", "q3", "q4", "q5_high"])
    return (
        frame.groupby("bucket", observed=True)
        .agg(
            days=("trade_date", "count"),
            confidence_mean=("confidence", "mean"),
            future_rank_ic_mean=("future_rank_ic", "mean"),
            future_spread_mean=("future_spread", "mean"),
            positive_future_ic_ratio=("future_rank_ic", lambda s: float((s > 0).mean())),
            positive_future_spread_ratio=("future_spread", lambda s: float((s > 0).mean())),
            entry_threshold_mean=("entry_band_rank_min_dynamic", "mean"),
        )
        .reset_index()
    )


def filtered_trade_audit(baseline_trades: pd.DataFrame, confidence_trades: pd.DataFrame, variant: str) -> pd.DataFrame:
    buys_base = baseline_trades.loc[baseline_trades["side"].eq("BUY")].copy()
    sells_base = baseline_trades.loc[baseline_trades["side"].eq("SELL")].copy()
    if buys_base.empty:
        return pd.DataFrame()
    base_pairs = buys_base.merge(sells_base, on=["ts_code", "entry_date"], suffixes=("_buy", "_sell"), how="left")
    conf_buys = confidence_trades.loc[confidence_trades["side"].eq("BUY"), ["ts_code", "entry_date"]].copy()
    conf_buys["kept_by_confidence"] = True
    merged = base_pairs.merge(conf_buys, on=["ts_code", "entry_date"], how="left")
    merged["kept_by_confidence"] = merged["kept_by_confidence"].fillna(False)
    merged["baseline_trade_return_net"] = (
        (merged["gross_value_sell"] - merged["cost_sell"]) / (merged["gross_value_buy"] + merged["cost_buy"]) - 1.0
    )
    rows = []
    for kept, group in merged.groupby("kept_by_confidence"):
        rows.append(
            {
                "variant": variant,
                "group": "kept_by_confidence" if kept else "filtered_by_confidence",
                "round_trips": int(len(group)),
                "mean_baseline_trade_return": float(group["baseline_trade_return_net"].mean()),
                "median_baseline_trade_return": float(group["baseline_trade_return_net"].median()),
                "win_rate": float(group["baseline_trade_return_net"].gt(0).mean()),
                "avg_entry_band_rank_pct": float(group["entry_band_rank_pct_buy"].mean()),
                "avg_entry_raw_rank_pct": float(group["entry_raw_rank_pct_buy"].mean()),
            }
        )
    return pd.DataFrame(rows)


def nz(value: float, fallback: float) -> float:
    return float(value) if np.isfinite(value) else float(fallback)


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(
    output: Path,
    metrics: pd.DataFrame,
    monthly: pd.DataFrame,
    trade_quality: pd.DataFrame,
    validation: pd.DataFrame,
    filtered: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    lines = [
        "# Band Score Factor Confidence v0",
        "",
        "## Design",
        "- Scope: only `band_score` confidence; no new production LightGBM confidence model.",
        "- Confidence uses delayed factor evidence: daily IC/spread observations are lagged by 10 trading days before they can enter rolling evidence or regime prior.",
        "- Regime prior uses PIT market state buckets and falls back from `trend/vol/breadth` regime to `trend/vol` coarse regime to global prior.",
        "- Integration: only dynamic entry threshold, no timing change and no position multiplier change.",
        f"- Dynamic entry rule: `entry_band_rank_min = 0.98 - 0.06 * confidence`, clipped to `[0.92, 0.98]`; baseline param config `{json.dumps(cfg, ensure_ascii=False)}`.",
        "",
        "## Backtest Metrics",
        md_table(metrics, 20),
        "",
        "## Monthly",
        md_table(monthly, 40),
        "",
        "## Trade Quality",
        md_table(trade_quality, 20),
        "",
        "## Confidence Bucket Validation",
        md_table(validation, 10),
        "",
        "## Filtered Baseline Trades",
        md_table(filtered, 10),
        "",
        "## Files",
        "- `factor_health_daily.csv`",
        "- `band_score_daily_payoff.csv`",
        "- `confidence_bucket_validation.csv`",
        "- `confidence_backtest_metrics.csv`",
        "- `confidence_monthly.csv`",
        "- `confidence_trade_quality.csv`",
        "- `confidence_filtered_trade_audit.csv`",
        "- `confidence_trades.csv`",
        "- `confidence_daily.parquet`",
        "- `confidence_positions.parquet`",
    ]
    (output / "factor_confidence_v0_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
