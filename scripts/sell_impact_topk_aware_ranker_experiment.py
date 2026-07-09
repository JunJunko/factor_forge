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
import sell_impact_ranker_timing_compare as timing_compare
import sell_impact_score_band_walkforward as wf
import sell_impact_sorting_repair as base


OUTPUT_ROOT = Path("artifacts/strategy_reviews")
SOURCE_VARIANT = "B_cluster_plus_low_vol"
TOPK_VARIANTS = {
    "baseline_b_ranker": "Original B LightGBM LambdaRank labels and parameters.",
    "top_decile_ndcg": "Use top-decile-heavy relevance labels and NDCG@5/10.",
    "top_decile_tail_weighted": "Top-decile-heavy relevance plus larger sample weights for top winners and bottom losers.",
    "positive_top_tail_weighted": "Reward positive top tail and strongly weight bottom tail avoidance.",
    "top5_bucket_ndcg": "Use very steep top-5-stock-per-day relevance labels.",
}


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_topk_aware_ranker_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("loading dataset, panel, timing and benchmark")
    dataset = low_vol.load_dataset(log)
    version, panel = base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    market_benchmark = timing_compare.load_market_benchmark(version)
    position_multiplier = timing_compare.load_position_multiplier(timing_compare.TIMING_DAILY)
    features = low_vol.features_for_variant(dataset, SOURCE_VARIANT)
    fmap = low_vol.build_feature_map(features, SOURCE_VARIANT)
    log(f"features={len(features)} data_version={version}")

    prediction_frames: list[pd.DataFrame] = []
    training_rows: list[dict[str, Any]] = []
    importance_rows: list[pd.DataFrame] = []
    score_ic_rows: list[dict[str, Any]] = []
    yearly_ic_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []

    for fold in wf.FOLDS:
        fold_name = fold["fold"]
        train = base.sample_slice(dataset, fold["train_start"], fold["train_end"], features).sort_values(
            ["trade_date", "ts_code"]
        )
        valid = base.sample_slice(dataset, fold["valid_start"], fold["valid_end"], features).sort_values(
            ["trade_date", "ts_code"]
        )
        test = base.sample_slice(dataset, fold["test_start"], fold["test_end"], features).sort_values(
            ["trade_date", "ts_code"]
        )
        log(f"{fold_name}: train={len(train):,} valid={len(valid):,} test={len(test):,}")
        for variant in TOPK_VARIANTS:
            log(f"fit {variant} {fold_name}")
            model = fit_topk_ranker(train, valid, features, variant)
            training_rows.append(
                {
                    "variant": variant,
                    "fold": fold_name,
                    "feature_count": len(features),
                    "train_rows": len(train),
                    "valid_rows": len(valid),
                    "test_rows": len(test),
                    "best_iteration": int(model.best_iteration_ or model.n_estimators),
                }
            )
            importance = low_vol.feature_importance(model, features, fmap, variant, fold_name)
            importance_rows.append(importance)

            parts = []
            for sample_name, frame in [("train", train), ("valid", valid), ("test", test)]:
                pred = low_vol.predict_scores(model, frame, features)
                pred["sample"] = sample_name
                pred["fold"] = fold_name
                pred["variant"] = variant
                parts.append(pred)
                score_ic_rows.append(low_vol.ic_summary(low_vol.daily_rank_ic(pred), variant, fold_name, sample_name))
            pred_all = pd.concat(parts, ignore_index=True)
            prediction_frames.append(pred_all)
            test_pred = pred_all.loc[pred_all["sample"].eq("test")].copy()
            yearly_ic_rows.extend(low_vol.yearly_model_ic(test_pred, variant, fold_name))
            decile_rows.extend(decile_spread(test_pred, variant, fold_name))
            topk_rows.extend(topk_payoff(test_pred, variant, fold_name))
            exposure_rows.append(low_vol.style_exposure(dataset, test_pred, variant, fold_name))
            portfolio_rows.append(
                low_vol.run_backtest(
                    panel=panel,
                    dataset=dataset,
                    pred=test_pred,
                    fold=fold,
                    variant=variant,
                    market_benchmark=market_benchmark,
                    position_multiplier=position_multiplier,
                    log=log,
                )
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    training = pd.DataFrame(training_rows)
    importance = pd.concat(importance_rows, ignore_index=True)
    score_ic = pd.DataFrame(score_ic_rows)
    yearly_ic = pd.DataFrame(yearly_ic_rows)
    deciles = pd.DataFrame(decile_rows)
    topk = pd.DataFrame(topk_rows)
    portfolio = pd.DataFrame(portfolio_rows)
    exposure = pd.DataFrame(exposure_rows)
    comparison = build_comparison(score_ic, portfolio, deciles, topk, exposure)
    group_importance = summarize_group_importance(importance)

    predictions.to_parquet(output / "topk_ranker_predictions.parquet", index=False)
    training.to_csv(output / "topk_ranker_training.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(output / "topk_ranker_feature_importance_gain.csv", index=False, encoding="utf-8-sig")
    group_importance.to_csv(output / "topk_ranker_group_importance.csv", index=False, encoding="utf-8-sig")
    score_ic.to_csv(output / "topk_ranker_train_valid_test_ic.csv", index=False, encoding="utf-8-sig")
    yearly_ic.to_csv(output / "topk_ranker_yearly_ic.csv", index=False, encoding="utf-8-sig")
    deciles.to_csv(output / "topk_ranker_decile_spread.csv", index=False, encoding="utf-8-sig")
    topk.to_csv(output / "topk_ranker_topk_payoff.csv", index=False, encoding="utf-8-sig")
    portfolio.to_csv(output / "topk_ranker_portfolio_metrics.csv", index=False, encoding="utf-8-sig")
    exposure.to_csv(output / "topk_ranker_style_exposure.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "topk_ranker_comparison.csv", index=False, encoding="utf-8-sig")
    write_report(output, comparison, score_ic, portfolio, topk, group_importance, exposure)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "source_variant": SOURCE_VARIANT,
                "data_version": version,
                "variants": TOPK_VARIANTS,
                "portfolio": {
                    "top_n": low_vol.TOP_N,
                    "holding_days": low_vol.HOLDING_DAYS,
                    "cost_bps": low_vol.COST_BPS,
                    "timing_daily": str(timing_compare.TIMING_DAILY),
                },
                "purpose": "Shift LightGBM training toward top-bucket payoff instead of broad cross-sectional RankIC.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def fit_topk_ranker(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], variant: str):
    if variant == "baseline_b_ranker":
        return low_vol.fit_ranker(train, valid, features)

    import lightgbm as lgb

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=250,
        learning_rate=0.035,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        random_state=42,
        verbosity=-1,
        force_col_wise=True,
        eval_at=[5, 10],
    )
    train_y = relevance_for_variant(train, variant)
    valid_y = relevance_for_variant(valid, variant)
    train_weight = sample_weight_for_variant(train, variant)
    valid_weight = sample_weight_for_variant(valid, variant)
    fit_kwargs: dict[str, Any] = {
        "X": train[features],
        "y": train_y,
        "group": train.groupby("trade_date").size().to_list(),
        "eval_set": [(valid[features], valid_y)],
        "eval_group": [valid.groupby("trade_date").size().to_list()],
        "callbacks": [lgb.early_stopping(35, verbose=False)],
    }
    if train_weight is not None:
        fit_kwargs["sample_weight"] = train_weight
        fit_kwargs["eval_sample_weight"] = [valid_weight]
    model.fit(**fit_kwargs)
    return model


def relevance_for_variant(frame: pd.DataFrame, variant: str) -> pd.Series:
    if variant == "baseline_b_ranker":
        return base.relevance_labels(frame)
    if variant == "top_decile_ndcg":
        return label_by_rank_bins(frame, [(0.95, 12), (0.90, 8), (0.80, 4), (0.60, 1)], default=0)
    if variant == "top_decile_tail_weighted":
        return label_by_rank_bins(frame, [(0.95, 12), (0.90, 8), (0.80, 4), (0.50, 1)], default=0)
    if variant == "positive_top_tail_weighted":
        return positive_top_tail_labels(frame)
    if variant == "top5_bucket_ndcg":
        return top_n_bucket_labels(frame, top_n=5)
    raise ValueError(f"unknown variant: {variant}")


def sample_weight_for_variant(frame: pd.DataFrame, variant: str) -> pd.Series | None:
    if variant in {"baseline_b_ranker", "top_decile_ndcg", "top5_bucket_ndcg"}:
        return None
    ranks = frame.groupby("trade_date")["label"].rank(pct=True, method="first")
    weights = pd.Series(1.0, index=frame.index, dtype=float)
    if variant == "top_decile_tail_weighted":
        weights.loc[ranks.ge(0.90)] = 4.0
        weights.loc[ranks.le(0.20)] = 3.0
        weights.loc[ranks.ge(0.95)] = 6.0
    elif variant == "positive_top_tail_weighted":
        weights.loc[ranks.ge(0.90) & frame["label"].gt(0)] = 5.0
        weights.loc[ranks.le(0.20)] = 4.0
        weights.loc[frame["label"].lt(-0.06)] = 5.0
    else:
        return None
    return weights


def label_by_rank_bins(frame: pd.DataFrame, bins: list[tuple[float, int]], default: int = 0) -> pd.Series:
    ranks = frame.groupby("trade_date")["label"].rank(pct=True, method="first")
    out = pd.Series(default, index=frame.index, dtype=int)
    for threshold, value in bins:
        out.loc[ranks.ge(threshold)] = value
    return out


def positive_top_tail_labels(frame: pd.DataFrame) -> pd.Series:
    ranks = frame.groupby("trade_date")["label"].rank(pct=True, method="first")
    out = pd.Series(0, index=frame.index, dtype=int)
    out.loc[ranks.gt(0.30)] = 1
    out.loc[ranks.ge(0.80) & frame["label"].gt(0)] = 4
    out.loc[ranks.ge(0.90) & frame["label"].gt(0)] = 9
    out.loc[ranks.ge(0.95) & frame["label"].gt(0)] = 14
    return out


def top_n_bucket_labels(frame: pd.DataFrame, top_n: int) -> pd.Series:
    labels = pd.Series(0, index=frame.index, dtype=int)
    for _, group in frame.groupby("trade_date"):
        ordered = group["label"].rank(method="first", ascending=False)
        labels.loc[group.index[ordered.le(top_n)]] = 15
        labels.loc[group.index[(ordered.gt(top_n)) & (ordered.le(top_n * 2))]] = 8
        labels.loc[group.index[(ordered.gt(top_n * 2)) & (ordered.le(top_n * 4))]] = 3
        labels.loc[group.index[(ordered.gt(top_n * 4)) & (ordered.le(top_n * 8))]] = 1
    return labels


def decile_spread(pred: pd.DataFrame, variant: str, fold: str) -> list[dict[str, Any]]:
    rows = []
    for date, group in pred.groupby("trade_date"):
        data = group.dropna(subset=["score", "label"]).copy()
        if len(data) < 50 or data["score"].nunique() < 10:
            continue
        data["decile"] = pd.qcut(data["score"].rank(method="first"), 10, labels=False) + 1
        avg = data.groupby("decile")["label"].mean()
        rows.append(
            {
                "variant": variant,
                "fold": fold,
                "trade_date": date,
                "top_decile_return": float(avg.get(10, np.nan)),
                "bottom_decile_return": float(avg.get(1, np.nan)),
                "decile_spread": float(avg.get(10, np.nan) - avg.get(1, np.nan)),
                "is_monotonic": bool(avg.is_monotonic_increasing),
            }
        )
    return rows


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
                    "topk_median_label": float(top["label"].median()),
                    "topk_positive_ratio": float((top["label"] > 0).mean()),
                    "topk_hit_top20_ratio": float(top["label"].ge(q80).mean()),
                    "topk_bad_bottom20_ratio": float(top["label"].le(q20).mean()),
                }
            )
    return rows


def build_comparison(
    score_ic: pd.DataFrame,
    portfolio: pd.DataFrame,
    deciles: pd.DataFrame,
    topk: pd.DataFrame,
    exposure: pd.DataFrame,
) -> pd.DataFrame:
    test_ic = (
        score_ic.loc[score_ic["sample"].eq("test")]
        .groupby("variant", as_index=False)
        .agg(
            mean_test_rank_ic=("rank_ic_mean", "mean"),
            min_test_rank_ic=("rank_ic_mean", "min"),
            mean_test_icir=("icir", "mean"),
            positive_fold_count=("rank_ic_mean", lambda s: int((s > 0).sum())),
        )
    )
    port = (
        portfolio.groupby("variant", as_index=False)
        .agg(
            mean_annualized_return=("annualized_return", "mean"),
            mean_excess_vs_csi1000=("annualized_excess_return_vs_csi1000", "mean"),
            mean_sharpe=("sharpe", "mean"),
            worst_mdd=("max_drawdown", "min"),
            mean_turnover=("annualized_turnover", "mean"),
        )
    )
    decile = (
        deciles.groupby("variant", as_index=False)
        .agg(
            mean_decile_spread=("decile_spread", "mean"),
            positive_spread_ratio=("decile_spread", lambda s: float((s > 0).mean())),
            monotonic_ratio=("is_monotonic", "mean"),
        )
    )
    top5 = (
        topk.loc[topk["k"].eq(5)]
        .groupby("variant", as_index=False)
        .agg(
            mean_top5_label=("topk_mean_label", "mean"),
            top5_positive_ratio=("topk_positive_ratio", "mean"),
            top5_hit_top20_ratio=("topk_hit_top20_ratio", "mean"),
            top5_bad_bottom20_ratio=("topk_bad_bottom20_ratio", "mean"),
        )
    )
    expo = (
        exposure.groupby("variant", as_index=False)
        .agg(
            mean_score_small_size_rank_corr=("score_small_size_rank_corr", "mean"),
            mean_top5_microcap_risk_share=("top5_microcap_risk_share", "mean"),
            mean_score_low_vol_rank_corr=("score_low_vol_rank_corr", "mean"),
        )
    )
    return (
        port.merge(test_ic, on="variant", how="left")
        .merge(decile, on="variant", how="left")
        .merge(top5, on="variant", how="left")
        .merge(expo, on="variant", how="left")
        .sort_values("mean_excess_vs_csi1000", ascending=False)
    )


def summarize_group_importance(importance: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        importance.groupby(["variant", "fold", "feature_group"], as_index=False)
        .agg(gain_importance=("gain_importance", "sum"), split_importance=("split_importance", "sum"))
    )
    totals = grouped.groupby(["variant", "fold"])["gain_importance"].transform("sum").replace(0, np.nan)
    grouped["gain_share"] = grouped["gain_importance"] / totals
    return grouped.sort_values(["variant", "fold", "gain_share"], ascending=[True, True, False])


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(
    output: Path,
    comparison: pd.DataFrame,
    score_ic: pd.DataFrame,
    portfolio: pd.DataFrame,
    topk: pd.DataFrame,
    group_importance: pd.DataFrame,
    exposure: pd.DataFrame,
) -> None:
    top5_year = topk.loc[topk["k"].eq(5)].copy()
    top5_year["year"] = pd.to_datetime(top5_year["trade_date"]).dt.year
    top5_year_summary = (
        top5_year.groupby(["variant", "year"], as_index=False)
        .agg(
            top5_mean_label=("topk_mean_label", "mean"),
            top5_positive_ratio=("topk_positive_ratio", "mean"),
            top5_hit_top20_ratio=("topk_hit_top20_ratio", "mean"),
            top5_bad_bottom20_ratio=("topk_bad_bottom20_ratio", "mean"),
        )
    )
    lines = [
        "# Top-k Aware Ranker Experiment",
        "",
        "## Scope",
        "- Keep B feature set: `cluster_stock_state + stock_state_low_vol`.",
        "- Same data, stock pool, labels, train/valid/test split, timing overlay and CSI1000 benchmark.",
        "- Change only LightGBM relevance labels / sample weights to target top-bucket payoff.",
        "",
        "## Variant Comparison",
        md_table(comparison, 20),
        "",
        "## Test RankIC",
        md_table(score_ic.loc[score_ic["sample"].eq("test")].sort_values(["variant", "fold"]), 80),
        "",
        "## Portfolio Metrics",
        md_table(
            portfolio[
                [
                    "variant",
                    "fold",
                    "annualized_return",
                    "annualized_excess_return_vs_csi1000",
                    "sharpe",
                    "max_drawdown",
                    "annualized_turnover",
                    "execution_rate",
                ]
            ].sort_values(["variant", "fold"]),
            80,
        ),
        "",
        "## Top5 Payoff By Year",
        md_table(top5_year_summary.sort_values(["variant", "year"]), 80),
        "",
        "## Group Gain Importance",
        md_table(group_importance, 120),
        "",
        "## Style Exposure",
        md_table(exposure.sort_values(["variant", "fold"]), 80),
    ]
    (output / "topk_aware_ranker_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
