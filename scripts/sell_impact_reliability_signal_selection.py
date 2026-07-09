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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sell_impact_ranker_timing_compare as timing_compare
import sell_impact_sorting_repair as base
import sell_impact_trade_param_ml_surface as surface
import sell_impact_trade_quality_optimization as tq
from factor_forge.config import CostModel, ExecutionConstraints


OUTPUT_ROOT = Path("artifacts/strategy_reviews/experiment7_reliability_selection")
PARAM_SURFACE_RUN = Path("artifacts/strategy_reviews/sell_impact_trade_param_ml_surface_20260709T120518Z")
DEFAULT_RELIABILITY = Path(
    "artifacts/factor_reliability/factor_reliability_model_v1_20260709T143452Z/factor_reliability_daily.csv"
)
PARAM_ID = "param_068"
RANDOM_SEED = 20260709


def main() -> None:
    args = parse_args()
    output = OUTPUT_ROOT / f"reliability_signal_selection_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("loading signals, panel, timing, benchmark and reliability")
    signals = tq.load_signals()
    signals["trade_date"] = pd.to_datetime(signals["trade_date"])
    reliability = load_reliability(args.reliability_daily)
    version, panel = base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel_slice = panel.loc[panel["trade_date"].between(pd.Timestamp(tq.TEST_START), pd.Timestamp(tq.TEST_END))].copy()
    timing = timing_compare.load_position_multiplier(timing_compare.TIMING_DAILY)
    market_benchmark = timing_compare.load_market_benchmark(version)
    cfg = load_param_config()
    constraints = ExecutionConstraints(
        exclude_suspended=True,
        cannot_buy_limit_up=True,
        cannot_sell_limit_down=True,
        exclude_st=True,
        exclude_delisting_period=True,
        min_listing_days=60,
    )
    cost_model = CostModel(commission_bps_per_side=3, slippage_bps_per_side=5, stamp_duty_bps_sell=5)
    log(f"param_cfg={cfg}")

    variants = build_variant_specs()
    metric_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    signal_daily_frames: list[pd.DataFrame] = []
    filtered_frames: list[pd.DataFrame] = []

    original_entry_candidates = tq.entry_candidates
    tq.entry_candidates = reliability_entry_candidates
    try:
        for spec in variants:
            name = spec["variant"]
            log(f"running {name}: {spec['description']}")
            variant_signals = build_variant_signals(signals, reliability, spec, cfg)
            signal_daily, filtered = signal_selection_analysis(variant_signals, spec, cfg)
            daily, trades, _positions, metrics = tq.run_trade_quality_backtest(
                panel=panel_slice,
                signals=variant_signals,
                timing=timing,
                market_benchmark=market_benchmark,
                constraints=constraints,
                cost_model=cost_model,
                cfg=cfg,
            )
            csi1000 = float(metrics.get("market_index_annualized_return", np.nan))
            metric_rows.append(
                {
                    "variant": name,
                    "experiment": spec["experiment"],
                    "horizon": spec.get("horizon"),
                    "parameter": spec.get("parameter"),
                    "description": spec["description"],
                    **metrics,
                    "annualized_excess_return_vs_csi1000": float(metrics["annualized_return"] - csi1000),
                }
            )
            alpha_rows.append(alpha_quality(variant_signals, name))
            daily["variant"] = name
            trades["variant"] = name
            signal_daily["variant"] = name
            filtered["variant"] = name
            daily_frames.append(daily)
            trade_frames.append(trades)
            signal_daily_frames.append(signal_daily)
            filtered_frames.append(filtered)
            log(
                f"{name}: ann={metrics['annualized_return']:.2%} sharpe={metrics['sharpe']:.2f} "
                f"mdd={metrics['max_drawdown']:.2%} buys={metrics['executed_buys']}"
            )
    finally:
        tq.entry_candidates = original_entry_candidates

    daily_all = pd.concat(daily_frames, ignore_index=True)
    trades_all = pd.concat(trade_frames, ignore_index=True)
    signal_daily_all = pd.concat(signal_daily_frames, ignore_index=True)
    filtered_all = pd.concat(filtered_frames, ignore_index=True)
    performance = pd.DataFrame(metric_rows)
    alpha = pd.DataFrame(alpha_rows)
    trade_quality = tq.trade_quality_summary(trades_all, panel_slice)
    signal_efficiency = build_signal_efficiency(trade_quality, performance)
    low_high = reliability_bucket_trade_quality(signals, reliability, cfg)

    daily_all.to_csv(output / "portfolio_nav.csv", index=False, encoding="utf-8-sig")
    trades_all.to_csv(output / "trades.csv", index=False, encoding="utf-8-sig")
    signal_daily_all.to_csv(output / "signal_selection_daily.csv", index=False, encoding="utf-8-sig")
    filtered_all.to_csv(output / "filtered_trade_analysis.csv", index=False, encoding="utf-8-sig")
    performance.to_csv(output / "performance_summary.csv", index=False, encoding="utf-8-sig")
    alpha.to_csv(output / "alpha_quality.csv", index=False, encoding="utf-8-sig")
    trade_quality.to_csv(output / "trade_quality.csv", index=False, encoding="utf-8-sig")
    signal_efficiency.to_csv(output / "signal_efficiency.csv", index=False, encoding="utf-8-sig")
    low_high.to_csv(output / "reliability_low_high_signal_quality.csv", index=False, encoding="utf-8-sig")

    write_report(output, performance, alpha, trade_quality, signal_efficiency, filtered_all, signal_daily_all, low_high)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "experiment": "Experiment 7: Reliability-aware Signal Selection Framework",
                "param_id": PARAM_ID,
                "source_signals": str(tq.SOURCE_RUN),
                "param_surface_run": str(PARAM_SURFACE_RUN),
                "reliability_daily": str(args.reliability_daily),
                "data_version": version,
                "test_window": [tq.TEST_START, tq.TEST_END],
                "variants": variants,
                "note": "Production code is not modified; only signal acceptance fields are added to research signals.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 7: reliability-aware signal selection.")
    parser.add_argument("--reliability-daily", type=Path, default=DEFAULT_RELIABILITY)
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
    keep = ["date", "reliability_5d", "reliability_10d", "reliability_20d"]
    return frame[keep].drop_duplicates("date").sort_values("date").replace([np.inf, -np.inf], np.nan)


def build_variant_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "variant": "baseline_param068",
            "experiment": "baseline",
            "description": "Original param_068 entry thresholds.",
            "mode": "baseline",
            "horizon": None,
            "parameter": None,
        }
    ]
    for horizon in [5, 10, 20]:
        for adjustment in [0.03, 0.05, 0.07]:
            specs.append(
                {
                    "variant": f"A_threshold_r{horizon}_adj{int(adjustment * 100):02d}",
                    "experiment": "A_dynamic_entry_threshold",
                    "description": f"entry_band_rank_min = base + {adjustment:.2f} * (1 - reliability_{horizon}d).",
                    "mode": "dynamic_band_threshold",
                    "horizon": horizon,
                    "parameter": adjustment,
                }
            )
        specs.append(
            {
                "variant": f"B_pool_step_r{horizon}",
                "experiment": "B_dynamic_entry_pool_size",
                "description": f"raw candidate pool: high reliability top5%, middle top2%, low top1% using reliability_{horizon}d.",
                "mode": "dynamic_raw_pool",
                "horizon": horizon,
                "parameter": "0.95/0.98/0.99",
            }
        )
        for threshold in [0.2, 0.3, 0.4, 0.5]:
            specs.append(
                {
                    "variant": f"C_filter_r{horizon}_min{int(threshold * 100):02d}",
                    "experiment": "C_reliability_trading_filter",
                    "description": f"No new entries when reliability_{horizon}d < {threshold:.2f}.",
                    "mode": "hard_filter",
                    "horizon": horizon,
                    "parameter": threshold,
                }
            )
    specs.extend(
        [
            {
                "variant": "D_placebo_threshold_adj05",
                "experiment": "D_random_placebo",
                "description": "Random reliability placebo for dynamic band threshold, adjustment=0.05.",
                "mode": "dynamic_band_threshold",
                "horizon": 5,
                "parameter": 0.05,
                "placebo": True,
            },
            {
                "variant": "D_placebo_filter_min30",
                "experiment": "D_random_placebo",
                "description": "Random reliability placebo hard filter, threshold=0.30.",
                "mode": "hard_filter",
                "horizon": 5,
                "parameter": 0.30,
                "placebo": True,
            },
        ]
    )
    return specs


def build_variant_signals(
    signals: pd.DataFrame,
    reliability: pd.DataFrame,
    spec: dict[str, Any],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    out = signals.copy()
    out["entry_band_rank_min_dynamic"] = float(cfg["entry_band_rank_min"])
    out["entry_raw_rank_min_dynamic"] = float(cfg["entry_raw_rank_min"])
    out["entry_allowed"] = True
    out["selection_reliability"] = 1.0
    out["selection_mode"] = spec["mode"]
    if spec["mode"] == "baseline":
        return out

    rel_values = reliability_series(out["trade_date"], reliability, int(spec.get("horizon") or 5), bool(spec.get("placebo", False)))
    out["selection_reliability"] = rel_values.to_numpy()
    mode = spec["mode"]
    if mode == "dynamic_band_threshold":
        adjustment = float(spec["parameter"])
        out["entry_band_rank_min_dynamic"] = (
            float(cfg["entry_band_rank_min"]) + adjustment * (1.0 - out["selection_reliability"])
        ).clip(upper=0.999)
    elif mode == "dynamic_raw_pool":
        rel = out["selection_reliability"]
        out["entry_raw_rank_min_dynamic"] = np.select(
            [rel.ge(0.67), rel.ge(0.33)],
            [0.95, 0.98],
            default=0.99,
        )
    elif mode == "hard_filter":
        threshold = float(spec["parameter"])
        out["entry_allowed"] = out["selection_reliability"].ge(threshold)
    else:
        raise ValueError(f"unknown selection mode: {mode}")
    return out.replace([np.inf, -np.inf], np.nan)


def reliability_series(dates: pd.Series, reliability: pd.DataFrame, horizon: int, placebo: bool) -> pd.Series:
    unique = pd.DataFrame({"trade_date": sorted(pd.to_datetime(dates).unique())})
    if placebo:
        rng = np.random.default_rng(RANDOM_SEED + horizon)
        unique["selection_reliability"] = rng.uniform(0.0, 1.0, size=len(unique))
    else:
        col = f"reliability_{horizon}d"
        values = reliability[["date", col]].rename(columns={"date": "trade_date", col: "selection_reliability"})
        unique = unique.merge(values, on="trade_date", how="left")
        unique["selection_reliability"] = unique["selection_reliability"].ffill().fillna(1.0).clip(0.0, 1.0)
    mapped = pd.Series(pd.to_datetime(dates)).map(unique.set_index("trade_date")["selection_reliability"])
    return mapped.fillna(1.0).astype(float)


def reliability_entry_candidates(signal_frame: pd.DataFrame, positions: list[tq.Position], cfg: dict[str, Any]) -> pd.DataFrame:
    held = {position.ts_code for position in positions}
    frame = signal_frame.loc[signal_frame["band_score"].notna()].copy()
    if held:
        frame = frame.loc[~frame.index.isin(held)].copy()
    if "entry_allowed" in frame:
        allowed = frame["entry_allowed"].where(frame["entry_allowed"].notna(), True).astype(bool)
    else:
        allowed = pd.Series(True, index=frame.index)
    band_min = pd.to_numeric(
        frame.get("entry_band_rank_min_dynamic", float(cfg.get("entry_band_rank_min", 0.0))),
        errors="coerce",
    ).fillna(float(cfg.get("entry_band_rank_min", 0.0)))
    raw_min = pd.to_numeric(
        frame.get("entry_raw_rank_min_dynamic", float(cfg.get("entry_raw_rank_min", 0.0))),
        errors="coerce",
    ).fillna(float(cfg.get("entry_raw_rank_min", 0.0)))
    frame = frame.loc[allowed & frame["band_rank_pct"].ge(band_min) & frame["raw_rank_pct"].ge(raw_min)].copy()
    if cfg.get("entry_pool") == "threshold":
        if "stock_state_small_size" in frame:
            frame = frame.loc[frame["stock_state_small_size"].le(float(cfg.get("max_microcap_score", np.inf)))]
        if "cluster_liquidity" in frame:
            frame = frame.loc[frame["cluster_liquidity"].ge(float(cfg.get("min_liquidity", -np.inf)))]
        if "cluster_price_reversal" in frame:
            frame = frame.loc[frame["cluster_price_reversal"].ge(float(cfg.get("min_price_reversal", -np.inf)))]
    return frame.sort_values(["band_score", "raw_score"], ascending=False)


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


def variant_eligible(frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    band_min = pd.to_numeric(frame["entry_band_rank_min_dynamic"], errors="coerce").fillna(float(cfg["entry_band_rank_min"]))
    raw_min = pd.to_numeric(frame["entry_raw_rank_min_dynamic"], errors="coerce").fillna(float(cfg["entry_raw_rank_min"]))
    allowed = frame["entry_allowed"].fillna(True).astype(bool)
    mask = frame["band_score"].notna() & allowed
    mask &= frame["band_rank_pct"].ge(band_min)
    mask &= frame["raw_rank_pct"].ge(raw_min)
    if cfg.get("entry_pool") == "threshold":
        if "stock_state_small_size" in frame:
            mask &= frame["stock_state_small_size"].le(float(cfg.get("max_microcap_score", np.inf)))
        if "cluster_liquidity" in frame:
            mask &= frame["cluster_liquidity"].ge(float(cfg.get("min_liquidity", -np.inf)))
        if "cluster_price_reversal" in frame:
            mask &= frame["cluster_price_reversal"].ge(float(cfg.get("min_price_reversal", -np.inf)))
    return mask.fillna(False)


def signal_selection_analysis(signals: pd.DataFrame, spec: dict[str, Any], cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = signals.copy()
    frame["baseline_eligible"] = baseline_eligible(frame, cfg)
    frame["variant_eligible"] = variant_eligible(frame, cfg)
    frame["filtered_by_reliability"] = frame["baseline_eligible"] & ~frame["variant_eligible"]
    frame["retained_by_reliability"] = frame["baseline_eligible"] & frame["variant_eligible"]
    daily_rows = []
    filtered_rows = []
    for date, group in frame.groupby("trade_date", sort=True):
        base = group.loc[group["baseline_eligible"]]
        retained = group.loc[group["retained_by_reliability"]]
        filtered = group.loc[group["filtered_by_reliability"]]
        daily_rows.append(
            {
                "trade_date": date,
                "experiment": spec["experiment"],
                "horizon": spec.get("horizon"),
                "parameter": spec.get("parameter"),
                "reliability": float(group["selection_reliability"].median()),
                "entry_band_rank_min_dynamic": float(group["entry_band_rank_min_dynamic"].median()),
                "entry_raw_rank_min_dynamic": float(group["entry_raw_rank_min_dynamic"].median()),
                "entry_allowed": bool(group["entry_allowed"].iloc[0]),
                "baseline_candidate_count": int(len(base)),
                "retained_candidate_count": int(len(retained)),
                "filtered_candidate_count": int(len(filtered)),
                "retention_rate": float(len(retained) / len(base)) if len(base) else np.nan,
                "baseline_future_return": float(base["label"].mean()) if len(base) else np.nan,
                "retained_future_return": float(retained["label"].mean()) if len(retained) else np.nan,
                "filtered_future_return": float(filtered["label"].mean()) if len(filtered) else np.nan,
                "filtered_win_rate": float(filtered["label"].gt(0).mean()) if len(filtered) else np.nan,
                "retained_win_rate": float(retained["label"].gt(0).mean()) if len(retained) else np.nan,
            }
        )
        for bucket_name, data in [("retained", retained), ("filtered", filtered)]:
            filtered_rows.append(
                {
                    "experiment": spec["experiment"],
                    "horizon": spec.get("horizon"),
                    "parameter": spec.get("parameter"),
                    "trade_date": date,
                    "selection": bucket_name,
                    "signal_count": int(len(data)),
                    "future_return": float(data["label"].mean()) if len(data) else np.nan,
                    "win_rate": float(data["label"].gt(0).mean()) if len(data) else np.nan,
                    "mean_reliability": float(data["selection_reliability"].mean()) if len(data) else np.nan,
                }
            )
    return pd.DataFrame(daily_rows), pd.DataFrame(filtered_rows)


def alpha_quality(signals: pd.DataFrame, variant: str) -> dict[str, Any]:
    frame = signals.loc[variant_eligible(signals, {"entry_pool": "threshold", **load_param_config()})].copy()
    values = []
    for _, group in frame.groupby("trade_date"):
        data = group.dropna(subset=["band_score", "label"])
        if len(data) < 5 or data["band_score"].nunique() < 2 or data["label"].nunique() < 2:
            continue
        value = data["band_score"].corr(data["label"], method="spearman")
        if pd.notna(value):
            values.append(float(value))
    series = pd.Series(values, dtype=float)
    mean = float(series.mean()) if len(series) else np.nan
    std = float(series.std(ddof=1)) if len(series) > 1 else np.nan
    return {
        "variant": variant,
        "days": int(len(series)),
        "rank_ic_mean": mean,
        "rank_ic_std": std,
        "icir": float(mean / std * np.sqrt(252)) if std and np.isfinite(std) and std > 0 else np.nan,
        "positive_ic_ratio": float((series > 0).mean()) if len(series) else np.nan,
        "alpha_decay": float(series.tail(20).mean() - series.head(20).mean()) if len(series) >= 40 else np.nan,
    }


def build_signal_efficiency(trade_quality: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    frame = performance[["variant", "generated_signals", "executed_buys", "execution_rate", "annualized_return", "max_drawdown"]].copy()
    if trade_quality.empty:
        return frame
    return frame.merge(
        trade_quality[
            [
                "variant",
                "round_trips",
                "mean_trade_return",
                "median_trade_return",
                "win_rate",
                "avg_win",
                "avg_loss",
                "payoff_ratio",
            ]
        ],
        on="variant",
        how="left",
    )


def reliability_bucket_trade_quality(signals: pd.DataFrame, reliability: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out_rows = []
    frame = build_variant_signals(
        signals,
        reliability,
        {"variant": "bucket_probe", "experiment": "bucket_probe", "mode": "dynamic_band_threshold", "horizon": 5, "parameter": 0.0, "description": ""},
        cfg,
    )
    frame = frame.loc[baseline_eligible(frame, cfg)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["reliability_bucket"] = pd.qcut(
        frame["selection_reliability"].rank(method="first"),
        5,
        labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"],
    )
    for bucket, group in frame.groupby("reliability_bucket", observed=False):
        out_rows.append(
            {
                "bucket": str(bucket),
                "signal_count": int(len(group)),
                "mean_reliability": float(group["selection_reliability"].mean()),
                "future_return": float(group["label"].mean()),
                "win_rate": float(group["label"].gt(0).mean()),
                "avg_band_rank_pct": float(group["band_rank_pct"].mean()),
                "avg_raw_rank_pct": float(group["raw_rank_pct"].mean()),
            }
        )
    return pd.DataFrame(out_rows)


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(
    output: Path,
    performance: pd.DataFrame,
    alpha: pd.DataFrame,
    trade_quality: pd.DataFrame,
    signal_efficiency: pd.DataFrame,
    filtered: pd.DataFrame,
    signal_daily: pd.DataFrame,
    low_high: pd.DataFrame,
) -> None:
    perf_cols = [
        "variant",
        "experiment",
        "horizon",
        "parameter",
        "total_return",
        "annualized_return",
        "annualized_excess_return_vs_csi1000",
        "sharpe",
        "max_drawdown",
        "calmar",
        "executed_buys",
        "trade_count",
    ]
    filtered_summary = (
        filtered.groupby(["variant", "selection"], as_index=False)
        .agg(
            signal_count=("signal_count", "sum"),
            future_return=("future_return", "mean"),
            win_rate=("win_rate", "mean"),
            mean_reliability=("mean_reliability", "mean"),
        )
        .sort_values(["variant", "selection"])
    )
    daily_summary = (
        signal_daily.groupby("variant", as_index=False)
        .agg(
            avg_reliability=("reliability", "mean"),
            baseline_candidates=("baseline_candidate_count", "sum"),
            retained_candidates=("retained_candidate_count", "sum"),
            filtered_candidates=("filtered_candidate_count", "sum"),
            retention_rate=("retention_rate", "mean"),
            retained_future_return=("retained_future_return", "mean"),
            filtered_future_return=("filtered_future_return", "mean"),
        )
        .sort_values("filtered_future_return")
    )
    best = performance.sort_values(["annualized_return", "sharpe"], ascending=False).head(12)
    lines = [
        "# Experiment 7: Reliability-aware Signal Selection Framework",
        "",
        "## Scope",
        "- Baseline is original `param_068`.",
        "- LightGBM score, factor weights, timing model, account exposure, holding count, sell rules, and execution constraints are unchanged.",
        "- Reliability only affects whether a baseline signal can enter the candidate pool.",
        "",
        "## Performance",
        md_table(best[perf_cols], 20),
        "",
        "Full performance table is in `performance_summary.csv`.",
        "",
        "## Alpha Quality",
        md_table(alpha.sort_values("icir", ascending=False), 30),
        "",
        "## Trade Quality",
        md_table(trade_quality.sort_values("mean_trade_return", ascending=False), 30),
        "",
        "## Signal Efficiency",
        md_table(signal_efficiency.sort_values("mean_trade_return", ascending=False), 30),
        "",
        "## Selection Analysis",
        "Filtered signals should be worse than retained signals if reliability is useful as an entry layer.",
        md_table(filtered_summary, 80),
        "",
        "Daily selection summary:",
        md_table(daily_summary, 40),
        "",
        "## Reliability Low/High Signal Quality",
        md_table(low_high, 20),
        "",
        "## Files",
        "- `portfolio_nav.csv`",
        "- `trades.csv`",
        "- `signal_selection_daily.csv`",
        "- `filtered_trade_analysis.csv`",
        "- `performance_summary.csv`",
        "- `alpha_quality.csv`",
        "- `trade_quality.csv`",
        "- `signal_efficiency.csv`",
        "- `reliability_low_high_signal_quality.csv`",
    ]
    text = "\n".join(lines) + "\n"
    (output / "reliability_signal_selection_report.md").write_text(text, encoding="utf-8")
    (output / "report.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
