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

import sell_impact_candidate_robustness_audit as audit
import sell_impact_low_vol_regime_experiment as low_vol
import sell_impact_model_weight_attribution_experiment as attr
import sell_impact_ranker_timing_compare as timing_compare
import sell_impact_score_band_walkforward as wf
import sell_impact_sorting_repair as base


SOURCE_RUN = Path("artifacts/strategy_reviews/sell_impact_topk_blend_postprocess_20260709T072928Z")
OUTPUT_ROOT = Path("artifacts/strategy_reviews")
BASE_VARIANT = "raw_90_top_decile_10"
REPAIR_VARIANT = "raw_70_positive_tail_30"
RAW_VARIANT = "raw_model"
LABEL_LAG_DAYS = low_vol.HOLDING_DAYS + 1
LOOKBACKS = [60, 120]
SIMILARITY_LOOKBACK = 252
SIMILARITY_NEIGHBORS = 40
REGIME_COLS = ["market_ret_60", "market_vol_20", "market_breadth_20", "market_turnover_chg_5_20"]


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_pit_regime_blend_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"source_run={SOURCE_RUN}")
    predictions = pd.read_parquet(SOURCE_RUN / "topk_blend_predictions.parquet")
    predictions["trade_date"] = pd.to_datetime(predictions["trade_date"])
    predictions = predictions.loc[predictions["variant"].isin([RAW_VARIANT, BASE_VARIANT, REPAIR_VARIANT])].copy()
    dataset = low_vol.load_dataset(log)
    market_context = build_market_context(dataset)
    version, panel = base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    market_benchmark = timing_compare.load_market_benchmark(version)
    position_multiplier = timing_compare.load_position_multiplier(timing_compare.TIMING_DAILY)

    pred_rows: list[pd.DataFrame] = []
    weight_rows: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []
    concentration_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    style_rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []

    for fold in wf.FOLDS:
        fold_name = fold["fold"]
        fold_pred = predictions.loc[predictions["fold"].eq(fold_name)].copy()
        wide = build_wide_scores(fold_pred)
        payoff = precompute_top5_payoff(wide)
        log(f"{fold_name}: rows={len(wide):,} dates={wide['trade_date'].nunique():,}")

        static_variants = {
            "raw_model": wide["raw_model_z"],
            "base_raw90_top_decile10": wide["base_z"],
            "repair_raw70_positive_tail30": wide["repair_z"],
        }
        for variant_name, score in static_variants.items():
            pred = wide[["trade_date", "ts_code", "label", "sample"]].copy()
            pred["score"] = score
            pred["variant"] = variant_name
            pred["fold"] = fold_name
            pred_rows.append(pred)

        for lookback in LOOKBACKS:
            variant_name = f"pit_regime_blend_lb{lookback}"
            pred, weights = build_pit_regime_blend(wide, payoff, market_context, lookback)
            pred["variant"] = variant_name
            pred["fold"] = fold_name
            for row in weights:
                row["variant"] = variant_name
                row["fold"] = fold_name
            pred_rows.append(pred)
            weight_rows.extend(weights)

        variant_name = f"pit_regime_similarity_lb{SIMILARITY_LOOKBACK}_nn{SIMILARITY_NEIGHBORS}"
        pred, weights = build_pit_regime_similarity_blend(
            wide,
            payoff,
            market_context,
            SIMILARITY_LOOKBACK,
            SIMILARITY_NEIGHBORS,
        )
        pred["variant"] = variant_name
        pred["fold"] = fold_name
        for row in weights:
            row["variant"] = variant_name
            row["fold"] = fold_name
        pred_rows.append(pred)
        weight_rows.extend(weights)

        for pred in [frame for frame in pred_rows if frame["fold"].iloc[0] == fold_name]:
            variant_name = str(pred["variant"].iloc[0])
            test_pred = pred.loc[pred["sample"].eq("test")].copy()
            if test_pred.empty:
                continue
            ic_rows.append(low_vol.ic_summary(low_vol.daily_rank_ic(test_pred), variant_name, fold_name, "test"))
            decile_rows.extend(attr.decile_spread(test_pred, variant_name, fold_name))
            topk_rows.extend(topk_payoff(test_pred, variant_name, fold_name))
            result = audit.run_full_backtest(
                panel=panel,
                dataset=dataset,
                pred=test_pred,
                fold=fold,
                market_benchmark=market_benchmark,
                position_multiplier=position_multiplier,
            )
            csi1000 = float(result.metrics.get("market_index_annualized_return", np.nan))
            portfolio_rows.append(
                {
                    "variant": variant_name,
                    "fold": fold_name,
                    **result.metrics,
                    "annualized_excess_return_vs_csi1000": float(result.metrics["annualized_return"] - csi1000),
                    "annualized_turnover": float(result.daily["portfolio_turnover"].mean() * 252),
                }
            )
            trades = result.trades.copy()
            trades["variant"] = variant_name
            trades["fold"] = fold_name
            daily = result.daily.copy()
            daily["variant"] = variant_name
            daily["fold"] = fold_name
            daily_frames.append(daily)
            trade_frames.append(trades)
            concentration_rows.append(audit.trade_concentration(trades, variant_name, fold_name))
            monthly_rows.extend(audit.monthly_breakdown(daily, trades, variant_name, fold_name))
            style_rows.append(audit.style_exposure(dataset, test_pred, variant_name, fold_name))
            log(
                f"{variant_name} {fold_name}: ann={portfolio_rows[-1]['annualized_return']:.2%} "
                f"excess={portfolio_rows[-1]['annualized_excess_return_vs_csi1000']:.2%} "
                f"mdd={portfolio_rows[-1]['max_drawdown']:.2%}"
            )

    pred_all = pd.concat(pred_rows, ignore_index=True)
    weights = pd.DataFrame(weight_rows)
    ic = pd.DataFrame(ic_rows)
    deciles = pd.DataFrame(decile_rows)
    topk = pd.DataFrame(topk_rows)
    portfolio = pd.DataFrame(portfolio_rows)
    concentration = pd.DataFrame(concentration_rows)
    monthly = pd.DataFrame(monthly_rows)
    style = pd.DataFrame(style_rows)
    daily_all = pd.concat(daily_frames, ignore_index=True)
    trades_all = pd.concat(trade_frames, ignore_index=True)
    comparison = build_comparison(ic, portfolio, deciles, topk, concentration, monthly, style)
    weight_summary = summarize_weights(weights)

    pred_all.to_parquet(output / "pit_regime_blend_predictions.parquet", index=False)
    daily_all.to_parquet(output / "pit_regime_blend_daily.parquet", index=False)
    trades_all.to_parquet(output / "pit_regime_blend_trades.parquet", index=False)
    weights.to_csv(output / "pit_regime_blend_daily_weights.csv", index=False, encoding="utf-8-sig")
    weight_summary.to_csv(output / "pit_regime_blend_weight_summary.csv", index=False, encoding="utf-8-sig")
    ic.to_csv(output / "pit_regime_blend_rank_ic.csv", index=False, encoding="utf-8-sig")
    deciles.to_csv(output / "pit_regime_blend_decile_spread.csv", index=False, encoding="utf-8-sig")
    topk.to_csv(output / "pit_regime_blend_topk_payoff.csv", index=False, encoding="utf-8-sig")
    portfolio.to_csv(output / "pit_regime_blend_portfolio_metrics.csv", index=False, encoding="utf-8-sig")
    concentration.to_csv(output / "pit_regime_blend_trade_concentration.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "pit_regime_blend_monthly_breakdown.csv", index=False, encoding="utf-8-sig")
    style.to_csv(output / "pit_regime_blend_style_exposure.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "pit_regime_blend_comparison.csv", index=False, encoding="utf-8-sig")
    write_report(output, comparison, portfolio, ic, topk, style, concentration, monthly, weight_summary)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "source_run": str(SOURCE_RUN),
                "data_version": version,
                "base_variant": BASE_VARIANT,
                "repair_variant": REPAIR_VARIANT,
                "label_lag_days": LABEL_LAG_DAYS,
                "lookbacks": LOOKBACKS,
                "similarity_lookback": SIMILARITY_LOOKBACK,
                "similarity_neighbors": SIMILARITY_NEIGHBORS,
                "rule": (
                    "PIT regime-aware blend. Each signal date uses only labels with trade_date <= current_date - "
                    "holding_days - 1 and current market regime features. Repair weight is restricted to 0/10/20/30%."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def build_market_context(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset[["trade_date", *REGIME_COLS]].drop_duplicates("trade_date").copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame.sort_values("trade_date").reset_index(drop=True)


def build_wide_scores(fold_pred: pd.DataFrame) -> pd.DataFrame:
    key = ["trade_date", "ts_code", "label", "sample"]
    parts = []
    mapping = {
        RAW_VARIANT: "raw_model",
        BASE_VARIANT: "base",
        REPAIR_VARIANT: "repair",
    }
    for variant, name in mapping.items():
        part = fold_pred.loc[fold_pred["variant"].eq(variant), [*key, "score"]].rename(columns={"score": f"{name}_score"})
        parts.append(part)
    wide = parts[0]
    for part in parts[1:]:
        wide = wide.merge(part, on=key, how="inner")
    for name in mapping.values():
        wide[f"{name}_z"] = wide.groupby("trade_date")[f"{name}_score"].transform(attr.cs_zscore_series)
    return wide.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def precompute_top5_payoff(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in wide.groupby("trade_date"):
        q20 = group["label"].quantile(0.20)
        for model in ["base", "repair"]:
            top = group.nlargest(low_vol.TOP_N, f"{model}_z")
            rows.append(
                {
                    "trade_date": date,
                    "model": model,
                    "top5_mean_label": float(top["label"].mean()),
                    "top5_bad_bottom20_ratio": float(top["label"].le(q20).mean()),
                    "top5_positive_ratio": float((top["label"] > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def build_pit_regime_blend(
    wide: pd.DataFrame,
    payoff: pd.DataFrame,
    market_context: pd.DataFrame,
    lookback: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out_parts = []
    weight_rows = []
    dates = list(pd.Index(wide["trade_date"].unique()).sort_values())
    test_dates = list(pd.Index(wide.loc[wide["sample"].eq("test"), "trade_date"].unique()).sort_values())
    date_pos = {date: i for i, date in enumerate(dates)}
    payoff_wide = payoff.pivot(index="trade_date", columns="model", values=["top5_mean_label", "top5_bad_bottom20_ratio"])
    payoff_wide.columns = [f"{metric}_{model}" for metric, model in payoff_wide.columns]
    payoff_wide = payoff_wide.reset_index().sort_values("trade_date")
    for date in dates:
        day = wide.loc[wide["trade_date"].eq(date)].copy()
        if day["sample"].iloc[0] != "test":
            day["score"] = day["base_z"]
            out_parts.append(day[["trade_date", "ts_code", "label", "sample", "score"]])
            continue
        pos = date_pos[date]
        cutoff = dates[max(0, pos - LABEL_LAG_DAYS)]
        history = payoff_wide.loc[payoff_wide["trade_date"].le(cutoff)].tail(lookback)
        stats = payoff_stats(history)
        regime = regime_state(market_context, date, cutoff)
        weight, reason = decide_repair_weight(stats, regime)
        day["score"] = (1.0 - weight) * day["base_z"].fillna(0.0) + weight * day["repair_z"].fillna(0.0)
        out_parts.append(day[["trade_date", "ts_code", "label", "sample", "score"]])
        weight_rows.append(
            {
                "trade_date": date,
                "lookback": lookback,
                "repair_weight": weight,
                "reason": reason,
                **stats,
                **regime,
                "history_start": history["trade_date"].min() if not history.empty else pd.NaT,
                "history_end": history["trade_date"].max() if not history.empty else pd.NaT,
                "history_days": int(history["trade_date"].nunique()),
            }
        )
    return pd.concat(out_parts, ignore_index=True), weight_rows


def build_pit_regime_similarity_blend(
    wide: pd.DataFrame,
    payoff: pd.DataFrame,
    market_context: pd.DataFrame,
    lookback: int,
    neighbors: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out_parts = []
    weight_rows = []
    dates = list(pd.Index(wide["trade_date"].unique()).sort_values())
    date_pos = {date: i for i, date in enumerate(dates)}
    payoff_wide = payoff.pivot(index="trade_date", columns="model", values=["top5_mean_label", "top5_bad_bottom20_ratio"])
    payoff_wide.columns = [f"{metric}_{model}" for metric, model in payoff_wide.columns]
    payoff_wide = payoff_wide.reset_index().sort_values("trade_date")
    market = market_context[["trade_date", *REGIME_COLS]].copy()
    payoff_regime = payoff_wide.merge(market, on="trade_date", how="left")
    for date in dates:
        day = wide.loc[wide["trade_date"].eq(date)].copy()
        if day["sample"].iloc[0] != "test":
            day["score"] = day["base_z"]
            out_parts.append(day[["trade_date", "ts_code", "label", "sample", "score"]])
            continue
        pos = date_pos[date]
        cutoff = dates[max(0, pos - LABEL_LAG_DAYS)]
        history = payoff_regime.loc[payoff_regime["trade_date"].le(cutoff)].tail(lookback).dropna(subset=REGIME_COLS)
        today = market.loc[market["trade_date"].eq(date)]
        similar = nearest_regime_history(history, today, neighbors)
        stats = payoff_stats(similar)
        regime = regime_state(market_context, date, cutoff)
        weight, reason = decide_similarity_weight(stats, regime, len(similar))
        day["score"] = (1.0 - weight) * day["base_z"].fillna(0.0) + weight * day["repair_z"].fillna(0.0)
        out_parts.append(day[["trade_date", "ts_code", "label", "sample", "score"]])
        weight_rows.append(
            {
                "trade_date": date,
                "lookback": lookback,
                "repair_weight": weight,
                "reason": reason,
                **stats,
                **regime,
                "history_start": similar["trade_date"].min() if not similar.empty else pd.NaT,
                "history_end": similar["trade_date"].max() if not similar.empty else pd.NaT,
                "history_days": int(similar["trade_date"].nunique()),
            }
        )
    return pd.concat(out_parts, ignore_index=True), weight_rows


def nearest_regime_history(history: pd.DataFrame, today: pd.DataFrame, neighbors: int) -> pd.DataFrame:
    if history.empty or today.empty or len(history) < 20:
        return history.iloc[0:0]
    center = history[REGIME_COLS].median()
    scale = history[REGIME_COLS].std(ddof=0).replace(0, np.nan)
    current = today.iloc[0][REGIME_COLS]
    z = (history[REGIME_COLS] - center) / scale
    current_z = (current - center) / scale
    distance = ((z - current_z) ** 2).sum(axis=1)
    out = history.assign(_distance=distance).sort_values("_distance").head(neighbors)
    return out.drop(columns=["_distance"])


def payoff_stats(history: pd.DataFrame) -> dict[str, float]:
    if history.empty:
        return {
            "base_top5_mean": np.nan,
            "repair_top5_mean": np.nan,
            "repair_advantage": np.nan,
            "base_bad_bottom20": np.nan,
            "repair_bad_bottom20": np.nan,
        }
    base_mean = float(history["top5_mean_label_base"].mean())
    repair_mean = float(history["top5_mean_label_repair"].mean())
    return {
        "base_top5_mean": base_mean,
        "repair_top5_mean": repair_mean,
        "repair_advantage": repair_mean - base_mean,
        "base_bad_bottom20": float(history["top5_bad_bottom20_ratio_base"].mean()),
        "repair_bad_bottom20": float(history["top5_bad_bottom20_ratio_repair"].mean()),
    }


def regime_state(market_context: pd.DataFrame, date: pd.Timestamp, cutoff: pd.Timestamp) -> dict[str, Any]:
    today = market_context.loc[market_context["trade_date"].eq(date)]
    hist = market_context.loc[market_context["trade_date"].le(cutoff)]
    if today.empty or len(hist) < 60:
        return {
            "market_stress": False,
            "market_ret_60": np.nan,
            "market_vol_20": np.nan,
            "market_breadth_20": np.nan,
            "market_turnover_chg_5_20": np.nan,
            "ret60_low": False,
            "vol20_high": False,
            "breadth20_low": False,
            "turnover_weak": False,
        }
    row = today.iloc[0]
    ret60_low = bool(row["market_ret_60"] <= hist["market_ret_60"].quantile(0.35))
    vol20_high = bool(row["market_vol_20"] >= hist["market_vol_20"].quantile(0.65))
    breadth20_low = bool(row["market_breadth_20"] <= hist["market_breadth_20"].quantile(0.35))
    turnover_weak = bool(row["market_turnover_chg_5_20"] <= hist["market_turnover_chg_5_20"].quantile(0.35))
    market_stress = (ret60_low and (vol20_high or breadth20_low)) or (breadth20_low and turnover_weak)
    return {
        "market_stress": market_stress,
        "market_ret_60": float(row["market_ret_60"]),
        "market_vol_20": float(row["market_vol_20"]),
        "market_breadth_20": float(row["market_breadth_20"]),
        "market_turnover_chg_5_20": float(row["market_turnover_chg_5_20"]),
        "ret60_low": ret60_low,
        "vol20_high": vol20_high,
        "breadth20_low": breadth20_low,
        "turnover_weak": turnover_weak,
    }


def decide_repair_weight(stats: dict[str, float], regime: dict[str, Any]) -> tuple[float, str]:
    advantage = stats.get("repair_advantage", np.nan)
    base_mean = stats.get("base_top5_mean", np.nan)
    base_bad = stats.get("base_bad_bottom20", np.nan)
    if not np.isfinite(advantage):
        return 0.0, "no_history"
    base_weak = (np.isfinite(base_mean) and base_mean < 0.006) or (np.isfinite(base_bad) and base_bad > 0.20)
    stress = bool(regime.get("market_stress", False))
    if advantage > 0.005 and stress:
        return 0.30, "repair_advantage_and_market_stress"
    if advantage > 0.002 and base_weak:
        return 0.20, "repair_advantage_and_base_weak"
    if advantage > 0.0 and (stress or base_weak):
        return 0.10, "mild_repair_advantage"
    return 0.0, "base_only"


def decide_similarity_weight(stats: dict[str, float], regime: dict[str, Any], history_days: int) -> tuple[float, str]:
    advantage = stats.get("repair_advantage", np.nan)
    if history_days < 20 or not np.isfinite(advantage):
        return 0.0, "insufficient_similar_history"
    if advantage > 0.006:
        return 0.30, "similar_regime_repair_strong"
    if advantage > 0.003:
        return 0.20, "similar_regime_repair_positive"
    if advantage > 0.0 and bool(regime.get("market_stress", False)):
        return 0.10, "similar_regime_mild_repair_stress"
    return 0.0, "similar_regime_base_only"


def topk_payoff(pred: pd.DataFrame, variant: str, fold: str) -> list[dict[str, Any]]:
    rows = []
    for date, group in pred.groupby("trade_date"):
        data = group.dropna(subset=["score", "label"]).copy()
        if len(data) < 50:
            continue
        q80 = data["label"].quantile(0.80)
        q20 = data["label"].quantile(0.20)
        for k in [5, 10, 20]:
            top = data.nlargest(k, "score")
            rows.append(
                {
                    "variant": variant,
                    "fold": fold,
                    "trade_date": date,
                    "k": k,
                    "topk_mean_label": float(top["label"].mean()),
                    "topk_positive_ratio": float((top["label"] > 0).mean()),
                    "topk_hit_top20_ratio": float(top["label"].ge(q80).mean()),
                    "topk_bad_bottom20_ratio": float(top["label"].le(q20).mean()),
                }
            )
    return rows


def build_comparison(
    ic: pd.DataFrame,
    portfolio: pd.DataFrame,
    deciles: pd.DataFrame,
    topk: pd.DataFrame,
    concentration: pd.DataFrame,
    monthly: pd.DataFrame,
    style: pd.DataFrame,
) -> pd.DataFrame:
    ic_summary = ic.groupby("variant", as_index=False).agg(
        mean_test_rank_ic=("rank_ic_mean", "mean"),
        min_test_rank_ic=("rank_ic_mean", "min"),
        mean_test_icir=("icir", "mean"),
    )
    port = portfolio.groupby("variant", as_index=False).agg(
        mean_annualized_return=("annualized_return", "mean"),
        min_annualized_return=("annualized_return", "min"),
        mean_excess_vs_csi1000=("annualized_excess_return_vs_csi1000", "mean"),
        mean_sharpe=("sharpe", "mean"),
        worst_mdd=("max_drawdown", "min"),
    )
    decile = deciles.groupby("score_variant", as_index=False).agg(
        mean_decile_spread=("decile_spread", "mean"),
        positive_spread_ratio=("decile_spread", lambda s: float((s > 0).mean())),
    ).rename(columns={"score_variant": "variant"})
    top5 = topk.loc[topk["k"].eq(5)].groupby("variant", as_index=False).agg(
        mean_top5_label=("topk_mean_label", "mean"),
        top5_positive_ratio=("topk_positive_ratio", "mean"),
        top5_bad_bottom20_ratio=("topk_bad_bottom20_ratio", "mean"),
    )
    conc = concentration.groupby("variant", as_index=False).agg(
        max_top5_stock_buy_share=("top5_stock_buy_share", "max"),
        mean_top5_stock_buy_share=("top5_stock_buy_share", "mean"),
        max_buy_hhi=("buy_hhi", "max"),
    )
    pos = monthly.loc[monthly["month_return"].gt(0)].groupby("variant")["month_return"].sum()
    top_pos = monthly.loc[monthly["month_return"].gt(0)].groupby("variant")["month_return"].max()
    month_conc = (top_pos / pos).rename("top_positive_month_return_share").reset_index()
    style_summary = style.groupby("variant", as_index=False).agg(
        max_top5_microcap_risk_share=("top5_microcap_risk_share", "max"),
        mean_top5_microcap_risk_share=("top5_microcap_risk_share", "mean"),
        mean_score_small_size_rank_corr=("score_small_size_rank_corr", "mean"),
    )
    return (
        port.merge(ic_summary, on="variant", how="left")
        .merge(decile, on="variant", how="left")
        .merge(top5, on="variant", how="left")
        .merge(conc, on="variant", how="left")
        .merge(month_conc, on="variant", how="left")
        .merge(style_summary, on="variant", how="left")
        .sort_values("mean_excess_vs_csi1000", ascending=False)
    )


def summarize_weights(weights: pd.DataFrame) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame()
    return weights.groupby(["variant", "fold"], as_index=False).agg(
        avg_repair_weight=("repair_weight", "mean"),
        max_repair_weight=("repair_weight", "max"),
        active_weight_ratio=("repair_weight", lambda s: float((s > 0).mean())),
        stress_ratio=("market_stress", "mean"),
        avg_repair_advantage=("repair_advantage", "mean"),
        avg_base_top5_mean=("base_top5_mean", "mean"),
        avg_repair_top5_mean=("repair_top5_mean", "mean"),
    )


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(
    output: Path,
    comparison: pd.DataFrame,
    portfolio: pd.DataFrame,
    ic: pd.DataFrame,
    topk: pd.DataFrame,
    style: pd.DataFrame,
    concentration: pd.DataFrame,
    monthly: pd.DataFrame,
    weight_summary: pd.DataFrame,
) -> None:
    top5 = topk.loc[topk["k"].eq(5)].copy()
    top5["year"] = pd.to_datetime(top5["trade_date"]).dt.year
    top5_year = top5.groupby(["variant", "year"], as_index=False).agg(
        top5_mean_label=("topk_mean_label", "mean"),
        top5_positive_ratio=("topk_positive_ratio", "mean"),
        top5_bad_bottom20_ratio=("topk_bad_bottom20_ratio", "mean"),
    )
    lines = [
        "# PIT Regime-aware Blend Experiment",
        "",
        "## Rule",
        "- Base model: `raw_90_top_decile_10`.",
        "- Repair model: `raw_70_positive_tail_30`.",
        "- Repair weight is decided per signal date using only matured historical labels and current market regime.",
        "- Weight choices are frozen at `0/0.1/0.2/0.3`.",
        "",
        "## Comparison",
        md_table(comparison, 40),
        "",
        "## Portfolio",
        md_table(portfolio.sort_values(["variant", "fold"]), 80),
        "",
        "## Weight Summary",
        md_table(weight_summary, 40),
        "",
        "## Test RankIC",
        md_table(ic.sort_values(["variant", "fold"]), 80),
        "",
        "## Top5 Payoff By Year",
        md_table(top5_year.sort_values(["variant", "year"]), 80),
        "",
        "## Style Exposure",
        md_table(style.sort_values(["variant", "fold"]), 80),
        "",
        "## Trade Concentration",
        md_table(concentration.sort_values(["variant", "fold"]), 80),
        "",
        "## Monthly Concentration Source",
        md_table(monthly.sort_values(["variant", "month_return"], ascending=[True, False]).groupby("variant").head(5), 80),
    ]
    (output / "pit_regime_blend_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
