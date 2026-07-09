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

import sell_impact_low_vol_regime_experiment as low_vol
import sell_impact_model_weight_attribution_experiment as attr
import sell_impact_ranker_timing_compare as timing_compare
import sell_impact_score_band_walkforward as wf
import sell_impact_sorting_repair as base


SOURCE_RUN = Path("artifacts/strategy_reviews/sell_impact_topk_aware_ranker_20260709T072358Z")
OUTPUT_ROOT = Path("artifacts/strategy_reviews")
BLENDS = {
    "raw_model": ("baseline_b_ranker", None, 0.0),
    "raw_90_positive_tail_10": ("baseline_b_ranker", "positive_top_tail_weighted", 0.10),
    "raw_80_positive_tail_20": ("baseline_b_ranker", "positive_top_tail_weighted", 0.20),
    "raw_70_positive_tail_30": ("baseline_b_ranker", "positive_top_tail_weighted", 0.30),
    "raw_90_top_decile_10": ("baseline_b_ranker", "top_decile_ndcg", 0.10),
    "raw_80_top_decile_20": ("baseline_b_ranker", "top_decile_ndcg", 0.20),
}


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_topk_blend_postprocess_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"loading predictions from {SOURCE_RUN}")
    predictions = pd.read_parquet(SOURCE_RUN / "topk_ranker_predictions.parquet")
    predictions["trade_date"] = pd.to_datetime(predictions["trade_date"])
    dataset = low_vol.load_dataset(log)
    version, panel = base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    market_benchmark = timing_compare.load_market_benchmark(version)
    position_multiplier = timing_compare.load_position_multiplier(timing_compare.TIMING_DAILY)

    pred_rows: list[pd.DataFrame] = []
    ic_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []

    for fold in wf.FOLDS:
        fold_name = fold["fold"]
        fold_pred = predictions.loc[predictions["fold"].eq(fold_name)].copy()
        for blend_name, (base_variant, overlay_variant, overlay_weight) in BLENDS.items():
            blended = build_blend(fold_pred, base_variant, overlay_variant, overlay_weight)
            blended["variant"] = blend_name
            blended["fold"] = fold_name
            pred_rows.append(blended)
            for sample, sample_frame in blended.groupby("sample"):
                ic_rows.append(low_vol.ic_summary(low_vol.daily_rank_ic(sample_frame), blend_name, fold_name, sample))
            test_pred = blended.loc[blended["sample"].eq("test")].copy()
            decile_rows.extend(attr.decile_spread(test_pred, blend_name, fold_name))
            topk_rows.extend(topk_payoff(test_pred, blend_name, fold_name))
            portfolio_rows.append(
                low_vol.run_backtest(
                    panel=panel,
                    dataset=dataset,
                    pred=test_pred,
                    fold=fold,
                    variant=blend_name,
                    market_benchmark=market_benchmark,
                    position_multiplier=position_multiplier,
                    log=log,
                )
            )

    pred_all = pd.concat(pred_rows, ignore_index=True)
    ic = pd.DataFrame(ic_rows)
    deciles = pd.DataFrame(decile_rows)
    topk = pd.DataFrame(topk_rows)
    portfolio = pd.DataFrame(portfolio_rows)
    comparison = build_comparison(ic, portfolio, deciles, topk)

    pred_all.to_parquet(output / "topk_blend_predictions.parquet", index=False)
    ic.to_csv(output / "topk_blend_train_valid_test_ic.csv", index=False, encoding="utf-8-sig")
    deciles.to_csv(output / "topk_blend_decile_spread.csv", index=False, encoding="utf-8-sig")
    topk.to_csv(output / "topk_blend_topk_payoff.csv", index=False, encoding="utf-8-sig")
    portfolio.to_csv(output / "topk_blend_portfolio_metrics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "topk_blend_comparison.csv", index=False, encoding="utf-8-sig")
    write_report(output, comparison, portfolio, ic, topk)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "source_run": str(SOURCE_RUN),
                "data_version": version,
                "blends": BLENDS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def build_blend(
    fold_pred: pd.DataFrame,
    base_variant: str,
    overlay_variant: str | None,
    overlay_weight: float,
) -> pd.DataFrame:
    base_frame = fold_pred.loc[fold_pred["variant"].eq(base_variant)].copy()
    base_frame = base_frame[["trade_date", "ts_code", "label", "sample", "score"]].rename(columns={"score": "base_score"})
    base_frame["base_z"] = base_frame.groupby("trade_date")["base_score"].transform(attr.cs_zscore_series)
    if overlay_variant is None or overlay_weight <= 0:
        out = base_frame.rename(columns={"base_score": "score"})
        return out[["trade_date", "ts_code", "label", "sample", "score"]]
    overlay = fold_pred.loc[fold_pred["variant"].eq(overlay_variant)].copy()
    overlay = overlay[["trade_date", "ts_code", "score"]].rename(columns={"score": "overlay_score"})
    overlay["overlay_z"] = overlay.groupby("trade_date")["overlay_score"].transform(attr.cs_zscore_series)
    merged = base_frame.merge(overlay[["trade_date", "ts_code", "overlay_z"]], on=["trade_date", "ts_code"], how="left")
    merged["score"] = (1.0 - overlay_weight) * merged["base_z"].fillna(0.0) + overlay_weight * merged["overlay_z"].fillna(0.0)
    return merged[["trade_date", "ts_code", "label", "sample", "score"]]


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
) -> pd.DataFrame:
    test_ic = (
        ic.loc[ic["sample"].eq("test")]
        .groupby("variant", as_index=False)
        .agg(
            mean_test_rank_ic=("rank_ic_mean", "mean"),
            min_test_rank_ic=("rank_ic_mean", "min"),
            mean_test_icir=("icir", "mean"),
        )
    )
    port = (
        portfolio.groupby("variant", as_index=False)
        .agg(
            mean_annualized_return=("annualized_return", "mean"),
            mean_excess_vs_csi1000=("annualized_excess_return_vs_csi1000", "mean"),
            mean_sharpe=("sharpe", "mean"),
            worst_mdd=("max_drawdown", "min"),
        )
    )
    decile = (
        deciles.groupby("score_variant", as_index=False)
        .agg(mean_decile_spread=("decile_spread", "mean"), positive_spread_ratio=("decile_spread", lambda s: float((s > 0).mean())))
        .rename(columns={"score_variant": "variant"})
    )
    top5 = (
        topk.loc[topk["k"].eq(5)]
        .groupby("variant", as_index=False)
        .agg(
            mean_top5_label=("topk_mean_label", "mean"),
            top5_positive_ratio=("topk_positive_ratio", "mean"),
            top5_bad_bottom20_ratio=("topk_bad_bottom20_ratio", "mean"),
        )
    )
    return port.merge(test_ic, on="variant", how="left").merge(decile, on="variant", how="left").merge(top5, on="variant", how="left").sort_values(
        "mean_excess_vs_csi1000", ascending=False
    )


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(output: Path, comparison: pd.DataFrame, portfolio: pd.DataFrame, ic: pd.DataFrame, topk: pd.DataFrame) -> None:
    top5 = topk.loc[topk["k"].eq(5)].copy()
    top5["year"] = pd.to_datetime(top5["trade_date"]).dt.year
    top5_year = top5.groupby(["variant", "year"], as_index=False).agg(
        top5_mean_label=("topk_mean_label", "mean"),
        top5_positive_ratio=("topk_positive_ratio", "mean"),
        top5_bad_bottom20_ratio=("topk_bad_bottom20_ratio", "mean"),
    )
    lines = [
        "# Top-k Blend Postprocess",
        "",
        "## Comparison",
        md_table(comparison, 20),
        "",
        "## Portfolio",
        md_table(portfolio.sort_values(["variant", "fold"]), 80),
        "",
        "## Test RankIC",
        md_table(ic.loc[ic["sample"].eq("test")].sort_values(["variant", "fold"]), 80),
        "",
        "## Top5 Payoff By Year",
        md_table(top5_year.sort_values(["variant", "year"]), 80),
    ]
    (output / "topk_blend_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
