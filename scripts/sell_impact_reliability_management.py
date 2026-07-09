from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
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


OUTPUT_ROOT = Path("artifacts/strategy_reviews/experiment9_reliability_management")
PREDICTIONS = Path(
    "artifacts/strategy_reviews/experiment8_signal_reliability/"
    "stock_signal_reliability_20260709T151758Z/model_predictions.csv"
)
PARAM_SURFACE_RUN = Path("artifacts/strategy_reviews/sell_impact_trade_param_ml_surface_20260709T120518Z")
PARAM_ID = "param_068"
RELIABILITY_MODEL = "lightgbm_shallow"
RELIABILITY_HORIZON = 10
PROB_COL = "signal_probability"


@dataclass
class Variant:
    variant: str
    experiment: str
    description: str
    ranking_mode: str = "baseline"
    ranking_param: float | None = None
    exit_threshold: float | None = None
    exit_streak_days: int | None = None
    extension_days: int | None = None


ORIGINAL_ENTRY_CANDIDATES = None
ORIGINAL_SELL_DECISION = None
CURRENT_VARIANT: Variant | None = None
LOW_RELIABILITY_STREAK: dict[tuple[str, pd.Timestamp], int] = {}
HIGH_RELIABILITY_STREAK: dict[tuple[str, pd.Timestamp], int] = {}


def main() -> None:
    args = parse_args()
    output = OUTPUT_ROOT / f"reliability_management_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("loading signals, reliability predictions, panel and param_068 config")
    signals = tq.load_signals()
    signals["trade_date"] = pd.to_datetime(signals["trade_date"])
    predictions = load_signal_reliability(args.predictions)
    signals = attach_signal_reliability(signals, predictions)
    cfg = load_param_config()
    version, panel = base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel_slice = panel.loc[panel["trade_date"].between(pd.Timestamp(tq.TEST_START), pd.Timestamp(tq.TEST_END))].copy()
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
    log(f"signals rows={len(signals):,} dates={signals['trade_date'].nunique():,}")
    log(f"param_cfg={cfg}")

    variants = build_variants()
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    performance_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    selection_frames: list[pd.DataFrame] = []

    global ORIGINAL_ENTRY_CANDIDATES, ORIGINAL_SELL_DECISION, CURRENT_VARIANT, LOW_RELIABILITY_STREAK, HIGH_RELIABILITY_STREAK
    ORIGINAL_ENTRY_CANDIDATES = tq.entry_candidates
    ORIGINAL_SELL_DECISION = tq.sell_decision
    tq.entry_candidates = reliability_entry_candidates
    tq.sell_decision = reliability_sell_decision
    try:
        for variant in variants:
            CURRENT_VARIANT = variant
            LOW_RELIABILITY_STREAK = {}
            HIGH_RELIABILITY_STREAK = {}
            log(f"running {variant.variant}: {variant.description}")
            selection_frames.append(selection_quality(signals, cfg, variant))
            alpha_rows.append(alpha_quality(signals, cfg, variant))
            daily, trades, _positions, metrics = tq.run_trade_quality_backtest(
                panel=panel_slice,
                signals=signals,
                timing=timing,
                market_benchmark=market_benchmark,
                constraints=constraints,
                cost_model=cost_model,
                cfg=cfg,
            )
            csi1000 = float(metrics.get("market_index_annualized_return", np.nan))
            performance_rows.append(
                {
                    "variant": variant.variant,
                    "experiment": variant.experiment,
                    "ranking_mode": variant.ranking_mode,
                    "ranking_param": variant.ranking_param,
                    "exit_threshold": variant.exit_threshold,
                    "exit_streak_days": variant.exit_streak_days,
                    "extension_days": variant.extension_days,
                    **metrics,
                    "annualized_excess_return_vs_csi1000": float(metrics["annualized_return"] - csi1000),
                }
            )
            daily["variant"] = variant.variant
            trades["variant"] = variant.variant
            daily_frames.append(daily)
            trade_frames.append(trades)
            log(
                f"{variant.variant}: ann={metrics['annualized_return']:.2%} "
                f"sharpe={metrics['sharpe']:.2f} mdd={metrics['max_drawdown']:.2%} buys={metrics['executed_buys']}"
            )
    finally:
        tq.entry_candidates = ORIGINAL_ENTRY_CANDIDATES
        tq.sell_decision = ORIGINAL_SELL_DECISION
        CURRENT_VARIANT = None

    portfolio_nav = pd.concat(daily_frames, ignore_index=True)
    trades = pd.concat(trade_frames, ignore_index=True)
    performance = pd.DataFrame(performance_rows)
    alpha = pd.DataFrame(alpha_rows)
    trade_quality = tq.trade_quality_summary(trades, panel_slice)
    selection = pd.concat(selection_frames, ignore_index=True)
    holding = holding_decay_analysis(trades)
    ranking_results = performance.loc[performance["experiment"].isin(["baseline", "ranking", "combined"])].copy()
    holding_results = performance.loc[performance["experiment"].isin(["baseline", "holding_exit", "hold_extension", "combined"])].copy()

    ranking_results.to_csv(output / "ranking_results.csv", index=False, encoding="utf-8-sig")
    holding_results.to_csv(output / "holding_results.csv", index=False, encoding="utf-8-sig")
    portfolio_nav.to_csv(output / "portfolio_nav.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(output / "trades.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(output / "selection_analysis.csv", index=False, encoding="utf-8-sig")
    holding.to_csv(output / "holding_decay_analysis.csv", index=False, encoding="utf-8-sig")
    performance.to_csv(output / "performance_summary.csv", index=False, encoding="utf-8-sig")
    alpha.to_csv(output / "alpha_quality.csv", index=False, encoding="utf-8-sig")
    trade_quality.to_csv(output / "trade_quality.csv", index=False, encoding="utf-8-sig")
    write_report(output, performance, alpha, trade_quality, selection, holding)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "experiment": "Experiment 9: Reliability-aware Ranking and Holding Management Framework",
                "predictions": str(args.predictions),
                "reliability_model": RELIABILITY_MODEL,
                "reliability_horizon": RELIABILITY_HORIZON,
                "param_id": PARAM_ID,
                "source_signals": str(tq.SOURCE_RUN),
                "data_version": version,
                "variants": [variant.__dict__ for variant in variants],
                "note": "Production code is not modified; monkey patches are scoped to this research run.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 9: reliability-aware ranking and holding management.")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
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


def load_signal_reliability(path: Path) -> pd.DataFrame:
    pred = pd.read_csv(path, parse_dates=["trade_date"])
    pred = pred.loc[
        pred["model"].eq(RELIABILITY_MODEL)
        & pred["horizon"].eq(RELIABILITY_HORIZON)
        & pred["sample"].eq("test")
    ].copy()
    return pred[["trade_date", "ts_code", "signal_probability", "future_trade_return", "success"]]


def attach_signal_reliability(signals: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    out = signals.merge(pred, on=["trade_date", "ts_code"], how="left")
    out[PROB_COL] = pd.to_numeric(out[PROB_COL], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    out["future_trade_return"] = pd.to_numeric(out["future_trade_return"], errors="coerce")
    out["success"] = pd.to_numeric(out["success"], errors="coerce")
    out["reliability_z"] = out.groupby("trade_date")[PROB_COL].transform(cs_zscore)
    return out.replace([np.inf, -np.inf], np.nan)


def build_variants() -> list[Variant]:
    variants = [
        Variant("baseline_param068", "baseline", "Original param_068 ranking and sell rule."),
        Variant("rank_v1_multiply", "ranking", "score_v1 = raw_score * (0.5 + 0.5 * reliability).", "multiply"),
    ]
    for lam in [0.05, 0.10, 0.20]:
        variants.append(Variant(f"rank_log_lambda_{int(lam * 100):03d}", "ranking", f"raw_score + {lam} * zscore(reliability).", "log_adjust", lam))
    for threshold in [0.01, 0.02, 0.05]:
        variants.append(Variant(f"rank_tiebreak_{int(threshold * 100):02d}", "ranking", f"Reliability tie-breaker within raw_score bucket {threshold}.", "tie_breaker", threshold))
    for threshold in [0.3, 0.4, 0.5]:
        for days in [3, 5, 10]:
            variants.append(
                Variant(
                    f"hold_exit_p{int(threshold * 100):02d}_n{days}",
                    "holding_exit",
                    f"Early exit after reliability < {threshold} for {days} consecutive signal days.",
                    exit_threshold=threshold,
                    exit_streak_days=days,
                )
            )
    for extra in [5, 10]:
        variants.append(
            Variant(
                f"hold_extend_high80_plus{extra}",
                "hold_extension",
                f"Allow max_hold_days + {extra} after three high-reliability days > 0.8.",
                extension_days=extra,
            )
        )
    variants.extend(
        [
            Variant(
                "combined_log010_exit50_n3",
                "combined",
                "Ranking log lambda=0.10 plus early exit reliability < 0.50 for 3 days.",
                "log_adjust",
                0.10,
                0.5,
                3,
            ),
            Variant(
                "combined_tie02_exit50_n3",
                "combined",
                "Tie-break threshold=0.02 plus early exit reliability < 0.50 for 3 days.",
                "tie_breaker",
                0.02,
                0.5,
                3,
            ),
            Variant(
                "combined_log010_extend5",
                "combined",
                "Ranking log lambda=0.10 plus high-reliability hold extension +5.",
                "log_adjust",
                0.10,
                extension_days=5,
            ),
        ]
    )
    return variants


def reliability_entry_candidates(signal_frame: pd.DataFrame, positions: list[tq.Position], cfg: dict[str, Any]) -> pd.DataFrame:
    frame = baseline_candidate_pool(signal_frame, positions, cfg)
    if frame.empty:
        return frame
    frame["reliability_sort_score"] = ranking_sort_score(frame, CURRENT_VARIANT)
    if CURRENT_VARIANT and CURRENT_VARIANT.ranking_mode == "tie_breaker":
        threshold = float(CURRENT_VARIANT.ranking_param or 0.02)
        frame["alpha_bucket"] = np.floor(pd.to_numeric(frame["raw_score"], errors="coerce") / threshold)
        return frame.sort_values(["alpha_bucket", PROB_COL, "raw_score"], ascending=False)
    if CURRENT_VARIANT and CURRENT_VARIANT.ranking_mode != "baseline":
        return frame.sort_values(["reliability_sort_score", "band_score", "raw_score"], ascending=False)
    return frame.sort_values(["band_score", "raw_score"], ascending=False)


def baseline_candidate_pool(signal_frame: pd.DataFrame, positions: list[tq.Position], cfg: dict[str, Any]) -> pd.DataFrame:
    held = {position.ts_code for position in positions}
    frame = signal_frame.loc[signal_frame["band_score"].notna()].copy()
    if held:
        frame = frame.loc[~frame.index.isin(held)].copy()
    frame = frame.loc[
        frame["band_rank_pct"].ge(float(cfg.get("entry_band_rank_min", 0.0)))
        & frame["raw_rank_pct"].ge(float(cfg.get("entry_raw_rank_min", 0.0)))
    ].copy()
    if cfg.get("entry_pool") == "threshold":
        if "stock_state_small_size" in frame:
            frame = frame.loc[frame["stock_state_small_size"].le(float(cfg.get("max_microcap_score", np.inf)))]
        if "cluster_liquidity" in frame:
            frame = frame.loc[frame["cluster_liquidity"].ge(float(cfg.get("min_liquidity", -np.inf)))]
        if "cluster_price_reversal" in frame:
            frame = frame.loc[frame["cluster_price_reversal"].ge(float(cfg.get("min_price_reversal", -np.inf)))]
    return frame


def ranking_sort_score(frame: pd.DataFrame, variant: Variant | None) -> pd.Series:
    alpha = pd.to_numeric(frame["raw_score"], errors="coerce").fillna(0.0)
    reliability = pd.to_numeric(frame[PROB_COL], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    if variant is None or variant.ranking_mode == "baseline":
        return pd.to_numeric(frame["band_score"], errors="coerce").fillna(-999)
    if variant.ranking_mode == "multiply":
        return alpha * (0.5 + 0.5 * reliability)
    if variant.ranking_mode == "log_adjust":
        lam = float(variant.ranking_param or 0.1)
        z = frame.get("reliability_z", pd.Series(0.0, index=frame.index)).fillna(0.0)
        return alpha + lam * z
    if variant.ranking_mode == "tie_breaker":
        return alpha
    return alpha


def reliability_sell_decision(position: tq.Position, date_index: int, signal_frame: pd.DataFrame | None, cfg: dict[str, Any]) -> str | None:
    holding_days = date_index - position.entry_index
    key = (position.ts_code, pd.Timestamp(position.entry_date))
    reliability = current_position_reliability(position, signal_frame)
    if reliability is not None:
        if CURRENT_VARIANT and CURRENT_VARIANT.exit_threshold is not None and reliability < CURRENT_VARIANT.exit_threshold:
            LOW_RELIABILITY_STREAK[key] = LOW_RELIABILITY_STREAK.get(key, 0) + 1
        else:
            LOW_RELIABILITY_STREAK[key] = 0
        if reliability > 0.8:
            HIGH_RELIABILITY_STREAK[key] = HIGH_RELIABILITY_STREAK.get(key, 0) + 1
        else:
            HIGH_RELIABILITY_STREAK[key] = 0

    if CURRENT_VARIANT and CURRENT_VARIANT.exit_threshold is not None and CURRENT_VARIANT.exit_streak_days is not None:
        if holding_days >= int(cfg["min_hold_days"]) and LOW_RELIABILITY_STREAK.get(key, 0) >= CURRENT_VARIANT.exit_streak_days:
            return "reliability_decay_exit"

    max_hold = int(cfg["max_hold_days"])
    if CURRENT_VARIANT and CURRENT_VARIANT.extension_days:
        extension_ok = HIGH_RELIABILITY_STREAK.get(key, 0) >= 3
        if holding_days >= max_hold + int(CURRENT_VARIANT.extension_days):
            return "max_hold_extended"
        if holding_days >= max_hold and not extension_ok:
            return "max_hold"
    elif holding_days >= max_hold:
        return "max_hold"

    if str(cfg["sell_rule"]) == "fixed":
        return None
    if holding_days < int(cfg["min_hold_days"]):
        return None
    if signal_frame is None or position.ts_code not in signal_frame.index:
        return None
    row = signal_frame.loc[position.ts_code]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    band_rank = float(row.get("band_rank_pct", np.nan))
    raw_rank = float(row.get("raw_rank_pct", np.nan))
    keep = (
        np.isfinite(band_rank)
        and band_rank >= float(cfg.get("continue_band_rank_min", 0.80))
    ) or (
        np.isfinite(raw_rank)
        and raw_rank >= float(cfg.get("continue_raw_rank_min", 0.50))
    )
    return None if keep else "signal_deteriorated"


def current_position_reliability(position: tq.Position, signal_frame: pd.DataFrame | None) -> float | None:
    if signal_frame is None or position.ts_code not in signal_frame.index:
        return None
    row = signal_frame.loc[position.ts_code]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    value = pd.to_numeric(row.get(PROB_COL, np.nan), errors="coerce")
    return float(value) if np.isfinite(value) else None


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


def selection_quality(signals: pd.DataFrame, cfg: dict[str, Any], variant: Variant) -> pd.DataFrame:
    rows = []
    for date, group in signals.groupby("trade_date", sort=True):
        pool = group.loc[baseline_eligible(group, cfg)].copy()
        if pool.empty:
            continue
        baseline_top = pool.sort_values(["band_score", "raw_score"], ascending=False).head(5)
        pool["variant_score"] = ranking_sort_score(pool, variant)
        if variant.ranking_mode == "tie_breaker":
            threshold = float(variant.ranking_param or 0.02)
            pool["alpha_bucket"] = np.floor(pd.to_numeric(pool["raw_score"], errors="coerce") / threshold)
            variant_top = pool.sort_values(["alpha_bucket", PROB_COL, "raw_score"], ascending=False).head(5)
        elif variant.ranking_mode == "baseline":
            variant_top = baseline_top
        else:
            variant_top = pool.sort_values(["variant_score", "band_score", "raw_score"], ascending=False).head(5)
        for selection, data in [("baseline_top5", baseline_top), ("reliability_top5", variant_top)]:
            rows.append(
                {
                    "variant": variant.variant,
                    "trade_date": date,
                    "selection": selection,
                    "avg_alpha_score": float(data["raw_score"].mean()),
                    "avg_band_score": float(data["band_score"].mean()),
                    "avg_reliability": float(data[PROB_COL].mean()),
                    "future_return": float(data["future_trade_return"].mean()),
                    "win_rate": float(data["future_trade_return"].gt(0.002).mean()),
                    "count": int(len(data)),
                }
            )
    return pd.DataFrame(rows)


def alpha_quality(signals: pd.DataFrame, cfg: dict[str, Any], variant: Variant) -> dict[str, Any]:
    values = []
    for _, group in signals.groupby("trade_date", sort=True):
        pool = group.loc[baseline_eligible(group, cfg)].copy()
        if len(pool) < 10 or pool["future_trade_return"].nunique() < 2:
            continue
        score = ranking_sort_score(pool, variant)
        if pd.Series(score).nunique(dropna=True) < 2:
            continue
        ic = score.corr(pool["future_trade_return"], method="spearman")
        if pd.notna(ic):
            values.append(float(ic))
    series = pd.Series(values, dtype=float)
    mean = float(series.mean()) if len(series) else np.nan
    std = float(series.std(ddof=1)) if len(series) > 1 else np.nan
    return {
        "variant": variant.variant,
        "experiment": variant.experiment,
        "days": int(len(series)),
        "rank_ic": mean,
        "rank_ic_std": std,
        "icir": float(mean / std * np.sqrt(252)) if std and np.isfinite(std) and std > 0 else np.nan,
        "positive_ic_ratio": float((series > 0).mean()) if len(series) else np.nan,
        "alpha_decay": float(series.tail(20).mean() - series.head(20).mean()) if len(series) >= 40 else np.nan,
    }


def holding_decay_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    buys = trades.loc[trades["side"].eq("BUY")].copy()
    sells = trades.loc[trades["side"].eq("SELL")].copy()
    if buys.empty or sells.empty:
        return pd.DataFrame()
    pairs = buys.merge(sells, on=["variant", "ts_code", "entry_date"], suffixes=("_buy", "_sell"), how="inner")
    if pairs.empty:
        return pd.DataFrame()
    pairs["trade_return_net"] = (
        (pairs["gross_value_sell"] - pairs["cost_sell"]) / (pairs["gross_value_buy"] + pairs["cost_buy"]) - 1.0
    )
    rows = []
    for (variant, reason), group in pairs.groupby(["variant", "reason_sell"]):
        losses = group.loc[group["trade_return_net"].le(0), "trade_return_net"]
        wins = group.loc[group["trade_return_net"].gt(0), "trade_return_net"]
        rows.append(
            {
                "variant": variant,
                "sell_reason": reason,
                "round_trips": int(len(group)),
                "mean_trade_return": float(group["trade_return_net"].mean()),
                "win_rate": float(group["trade_return_net"].gt(0).mean()),
                "avg_win": float(wins.mean()) if len(wins) else np.nan,
                "avg_loss": float(losses.mean()) if len(losses) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["variant", "mean_trade_return"])


def cs_zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(
    output: Path,
    performance: pd.DataFrame,
    alpha: pd.DataFrame,
    trade_quality: pd.DataFrame,
    selection: pd.DataFrame,
    holding: pd.DataFrame,
) -> None:
    perf_cols = [
        "variant",
        "experiment",
        "total_return",
        "annualized_return",
        "annualized_excess_return_vs_csi1000",
        "sharpe",
        "max_drawdown",
        "calmar",
        "executed_buys",
        "trade_count",
    ]
    selection_summary = (
        selection.groupby(["variant", "selection"], as_index=False)
        .agg(
            avg_alpha_score=("avg_alpha_score", "mean"),
            avg_reliability=("avg_reliability", "mean"),
            future_return=("future_return", "mean"),
            win_rate=("win_rate", "mean"),
            days=("trade_date", "nunique"),
        )
        .sort_values(["variant", "selection"])
    )
    high_alpha_low_rel = high_alpha_low_reliability_summary(selection)
    lines = [
        "# Experiment 9: Reliability-aware Ranking and Holding Management Framework",
        "",
        "## Scope",
        "- Alpha LightGBM, factor generation, timing model, stock pool, and production code are not modified.",
        "- Research variants only change candidate ranking order or add reliability decision logic around holding.",
        f"- Stock-level reliability input: `{PREDICTIONS}`, model `{RELIABILITY_MODEL}`, horizon `{RELIABILITY_HORIZON}d`.",
        "",
        "## Portfolio Performance",
        md_table(performance.sort_values("annualized_return", ascending=False)[perf_cols], 40),
        "",
        "## Alpha Quality",
        md_table(alpha.sort_values("icir", ascending=False), 40),
        "",
        "## Trade Quality",
        md_table(trade_quality.sort_values("mean_trade_return", ascending=False), 40),
        "",
        "## Selection Quality",
        md_table(selection_summary, 80),
        "",
        "## Holding Decay Analysis",
        md_table(holding, 80),
        "",
        "## High Alpha / Low Reliability Check",
        md_table(high_alpha_low_rel, 20),
        "",
        "## Required Answers",
        "- Ranking improvement: compare `baseline_top5` and `reliability_top5` in `selection_analysis.csv`.",
        "- High alpha / low reliability loss source: see `High Alpha / Low Reliability Check`.",
        "- Holding reliability decay: see `holding_decay_analysis.csv`, especially `reliability_decay_exit` rows.",
        "- Alpha decay reduction: compare `alpha_decay` in `alpha_quality.csv` across baseline/ranking/holding variants.",
        "",
        "## Files",
        "- `ranking_results.csv`",
        "- `holding_results.csv`",
        "- `portfolio_nav.csv`",
        "- `trades.csv`",
        "- `selection_analysis.csv`",
        "- `holding_decay_analysis.csv`",
        "- `performance_summary.csv`",
    ]
    text = "\n".join(lines) + "\n"
    (output / "report.md").write_text(text, encoding="utf-8")
    (output / "reliability_management_report.md").write_text(text, encoding="utf-8")


def high_alpha_low_reliability_summary(selection: pd.DataFrame) -> pd.DataFrame:
    rel = selection.loc[selection["selection"].eq("reliability_top5")].copy()
    if rel.empty:
        return pd.DataFrame()
    rows = []
    for variant, group in rel.groupby("variant"):
        rows.append(
            {
                "variant": variant,
                "avg_reliability": float(group["avg_reliability"].mean()),
                "avg_top5_future_return": float(group["future_return"].mean()),
                "bad_top5_day_ratio": float(group["future_return"].lt(0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_top5_future_return")


if __name__ == "__main__":
    main()
