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
POST_SCORE_VARIANTS = {
    "raw_model": "Original LightGBM score.",
    "stable_group_weighted": "Reweight SHAP factor-cluster contributions by train/valid stable payoff.",
    "anti_valid_chase_group_weighted": "Same as stable, but penalize groups whose valid IC jumps far above train IC.",
}


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_model_weight_attribution_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
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
    feature_to_group = fmap.set_index("feature")["feature_group"].to_dict()
    groups = sorted({feature_to_group.get(feature, "other") for feature in features})
    log(f"features={len(features)} groups={len(groups)} data_version={version}")

    all_predictions: list[pd.DataFrame] = []
    group_ic_rows: list[dict[str, Any]] = []
    group_weight_rows: list[dict[str, Any]] = []
    group_contrib_rows: list[dict[str, Any]] = []
    importance_rows: list[pd.DataFrame] = []
    score_ic_rows: list[dict[str, Any]] = []
    yearly_ic_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []

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
        log(f"fit {fold_name}: train={len(train):,} valid={len(valid):,} test={len(test):,}")
        model = low_vol.fit_ranker(train, valid, features)
        importance_rows.append(low_vol.feature_importance(model, features, fmap, SOURCE_VARIANT, fold_name))

        sample_parts: list[pd.DataFrame] = []
        for sample_name, frame in [("train", train), ("valid", valid), ("test", test)]:
            log(f"shap contributions {fold_name} {sample_name}: rows={len(frame):,}")
            part = build_group_contribution_frame(model, frame, features, feature_to_group, groups)
            part["sample"] = sample_name
            part["fold"] = fold_name
            sample_parts.append(part)
            group_contrib_rows.extend(group_contribution_summary(part, groups, fold_name, sample_name))
        fold_contrib = pd.concat(sample_parts, ignore_index=True)

        group_payoff = build_group_payoff(fold_contrib, groups, fold_name)
        group_ic_rows.extend(group_payoff)
        weights = build_group_weights(group_payoff, groups, fold_name)
        group_weight_rows.extend(weights)
        weight_frame = pd.DataFrame(weights)

        scored = add_post_scores(fold_contrib, groups, weight_frame)
        all_predictions.append(scored)
        for score_variant in POST_SCORE_VARIANTS:
            renamed = scored[["trade_date", "ts_code", "label", "sample", "fold", score_variant]].rename(
                columns={score_variant: "score"}
            )
            renamed["score_variant"] = score_variant
            for sample_name, frame in renamed.groupby("sample"):
                score_ic_rows.append(ic_summary(low_vol.daily_rank_ic(frame), score_variant, fold_name, sample_name))
            test_pred = renamed.loc[renamed["sample"].eq("test")].copy()
            yearly_ic_rows.extend(yearly_ic(test_pred, score_variant, fold_name))
            decile_rows.extend(decile_spread(test_pred, score_variant, fold_name))
            portfolio_rows.append(
                run_portfolio(
                    panel=panel,
                    dataset=dataset,
                    pred=test_pred,
                    fold=fold,
                    score_variant=score_variant,
                    market_benchmark=market_benchmark,
                    position_multiplier=position_multiplier,
                    log=log,
                )
            )

    predictions = pd.concat(all_predictions, ignore_index=True)
    group_ic = pd.DataFrame(group_ic_rows)
    group_weights = pd.DataFrame(group_weight_rows)
    group_contrib = pd.DataFrame(group_contrib_rows)
    importance = pd.concat(importance_rows, ignore_index=True)
    score_ic = pd.DataFrame(score_ic_rows)
    yearly = pd.DataFrame(yearly_ic_rows)
    deciles = pd.DataFrame(decile_rows)
    portfolio = pd.DataFrame(portfolio_rows)
    comparison = build_comparison(score_ic, portfolio, deciles)
    feature_group_importance = summarize_importance(importance)

    predictions.to_parquet(output / "score_group_contributions.parquet", index=False)
    group_ic.to_csv(output / "factor_cluster_payoff_ic.csv", index=False, encoding="utf-8-sig")
    group_weights.to_csv(output / "factor_cluster_postscore_weights.csv", index=False, encoding="utf-8-sig")
    group_contrib.to_csv(output / "factor_cluster_shap_contribution_summary.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(output / "feature_importance_gain.csv", index=False, encoding="utf-8-sig")
    feature_group_importance.to_csv(output / "factor_cluster_gain_importance.csv", index=False, encoding="utf-8-sig")
    score_ic.to_csv(output / "score_variant_train_valid_test_ic.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "score_variant_yearly_ic.csv", index=False, encoding="utf-8-sig")
    deciles.to_csv(output / "score_variant_decile_spread.csv", index=False, encoding="utf-8-sig")
    portfolio.to_csv(output / "score_variant_portfolio_metrics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "score_variant_comparison.csv", index=False, encoding="utf-8-sig")
    write_report(output, comparison, group_ic, group_weights, group_contrib, feature_group_importance, score_ic, portfolio)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "source_variant": SOURCE_VARIANT,
                "data_version": version,
                "features": features,
                "post_score_variants": POST_SCORE_VARIANTS,
                "timing_daily": str(timing_compare.TIMING_DAILY),
                "purpose": (
                    "Decompose LightGBM score into factor-cluster SHAP contributions, "
                    "test whether validation-year dominant weights reduce OOS sorting, "
                    "and evaluate group-reweighted model scores."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def build_group_contribution_frame(
    model,
    frame: pd.DataFrame,
    features: list[str],
    feature_to_group: dict[str, str],
    groups: list[str],
) -> pd.DataFrame:
    contrib = model.booster_.predict(frame[features], pred_contrib=True)
    feature_contrib = pd.DataFrame(contrib[:, :-1], columns=features)
    out = frame[["trade_date", "ts_code", "label"]].copy().reset_index(drop=True)
    out["raw_model"] = np.asarray(model.predict(frame[features]), dtype=float)
    out["base_value"] = contrib[:, -1]
    for group in groups:
        cols = [feature for feature in features if feature_to_group.get(feature, "other") == group]
        out[f"group__{group}"] = feature_contrib[cols].sum(axis=1).to_numpy() if cols else 0.0
    return out


def group_contribution_summary(frame: pd.DataFrame, groups: list[str], fold: str, sample: str) -> list[dict[str, Any]]:
    rows = []
    total_abs = sum(float(frame[f"group__{group}"].abs().mean()) for group in groups)
    for group in groups:
        values = frame[f"group__{group}"]
        mean_abs = float(values.abs().mean())
        rows.append(
            {
                "fold": fold,
                "sample": sample,
                "feature_group": group,
                "mean_abs_shap": mean_abs,
                "shap_abs_share": mean_abs / total_abs if total_abs > 0 else np.nan,
                "mean_shap": float(values.mean()),
                "positive_contribution_ratio": float((values > 0).mean()),
            }
        )
    return rows


def build_group_payoff(frame: pd.DataFrame, groups: list[str], fold: str) -> list[dict[str, Any]]:
    rows = []
    for sample, sample_frame in frame.groupby("sample"):
        for group in groups:
            pred = sample_frame[["trade_date", "ts_code", "label", f"group__{group}"]].rename(
                columns={f"group__{group}": "score"}
            )
            row = ic_summary(low_vol.daily_rank_ic(pred), group, fold, sample)
            row["feature_group"] = group
            rows.append(row)
    return rows


def build_group_weights(payoff_rows: list[dict[str, Any]], groups: list[str], fold: str) -> list[dict[str, Any]]:
    payoff = pd.DataFrame(payoff_rows)
    pivot = payoff.pivot_table(index="feature_group", columns="sample", values="rank_ic_mean", aggfunc="mean")
    rows = []
    stable_scores = {}
    anti_chase_scores = {}
    for group in groups:
        train_ic = float(pivot.get("train", pd.Series(dtype=float)).get(group, np.nan))
        valid_ic = float(pivot.get("valid", pd.Series(dtype=float)).get(group, np.nan))
        train_pos = max(train_ic, 0.0) if np.isfinite(train_ic) else 0.0
        valid_pos = max(valid_ic, 0.0) if np.isfinite(valid_ic) else 0.0
        stable = min(train_pos, valid_pos)
        valid_jump = max(valid_pos - train_pos, 0.0)
        anti_chase = stable / (1.0 + 20.0 * valid_jump)
        stable_scores[group] = stable
        anti_chase_scores[group] = anti_chase

    stable_weights = normalize_weights(stable_scores, groups)
    anti_chase_weights = normalize_weights(anti_chase_scores, groups)
    for group in groups:
        rows.append(
            {
                "fold": fold,
                "score_variant": "stable_group_weighted",
                "feature_group": group,
                "train_rank_ic": float(pivot.get("train", pd.Series(dtype=float)).get(group, np.nan)),
                "valid_rank_ic": float(pivot.get("valid", pd.Series(dtype=float)).get(group, np.nan)),
                "raw_weight_score": stable_scores[group],
                "normalized_weight": stable_weights[group],
            }
        )
        rows.append(
            {
                "fold": fold,
                "score_variant": "anti_valid_chase_group_weighted",
                "feature_group": group,
                "train_rank_ic": float(pivot.get("train", pd.Series(dtype=float)).get(group, np.nan)),
                "valid_rank_ic": float(pivot.get("valid", pd.Series(dtype=float)).get(group, np.nan)),
                "raw_weight_score": anti_chase_scores[group],
                "normalized_weight": anti_chase_weights[group],
            }
        )
    return rows


def normalize_weights(scores: dict[str, float], groups: list[str]) -> dict[str, float]:
    positive_sum = sum(value for value in scores.values() if np.isfinite(value) and value > 0)
    if positive_sum <= 0:
        return {group: 1.0 / len(groups) for group in groups}
    return {
        group: (scores[group] / positive_sum if np.isfinite(scores[group]) and scores[group] > 0 else 0.0)
        for group in groups
    }


def add_post_scores(frame: pd.DataFrame, groups: list[str], weight_frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    z_cols = []
    for group in groups:
        col = f"group__{group}"
        z_col = f"z__{group}"
        out[z_col] = out.groupby("trade_date")[col].transform(cs_zscore_series)
        z_cols.append(z_col)
    for score_variant in ["stable_group_weighted", "anti_valid_chase_group_weighted"]:
        weights = (
            weight_frame.loc[weight_frame["score_variant"].eq(score_variant)]
            .set_index("feature_group")["normalized_weight"]
            .to_dict()
        )
        out[score_variant] = 0.0
        for group in groups:
            out[score_variant] += out[f"z__{group}"].fillna(0.0) * float(weights.get(group, 0.0))
    return out.drop(columns=z_cols)


def cs_zscore_series(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def ic_summary(values: pd.Series, variant: str, fold: Any, sample: Any) -> dict[str, Any]:
    row = low_vol.ic_summary(values, variant, fold, sample)
    row["score_variant"] = variant
    return row


def yearly_ic(pred: pd.DataFrame, score_variant: str, fold: str) -> list[dict[str, Any]]:
    rows = []
    frame = pred.copy()
    frame["year"] = frame["trade_date"].dt.year
    for year, group in frame.groupby("year"):
        row = ic_summary(low_vol.daily_rank_ic(group), score_variant, fold, int(year))
        rows.append(row)
    return rows


def decile_spread(pred: pd.DataFrame, score_variant: str, fold: str) -> list[dict[str, Any]]:
    rows = []
    for date, group in pred.groupby("trade_date"):
        data = group.dropna(subset=["score", "label"]).copy()
        if len(data) < 50 or data["score"].nunique() < 10:
            continue
        data["decile"] = pd.qcut(data["score"].rank(method="first"), 10, labels=False) + 1
        avg = data.groupby("decile")["label"].mean()
        rows.append(
            {
                "trade_date": date,
                "fold": fold,
                "score_variant": score_variant,
                "top_decile_return": float(avg.get(10, np.nan)),
                "bottom_decile_return": float(avg.get(1, np.nan)),
                "decile_spread": float(avg.get(10, np.nan) - avg.get(1, np.nan)),
                "is_monotonic": bool(avg.is_monotonic_increasing),
            }
        )
    return rows


def run_portfolio(
    *,
    panel: pd.DataFrame,
    dataset: pd.DataFrame,
    pred: pd.DataFrame,
    fold: dict[str, str],
    score_variant: str,
    market_benchmark: pd.DataFrame,
    position_multiplier: pd.Series,
    log,
) -> dict[str, Any]:
    row = low_vol.run_backtest(
        panel=panel,
        dataset=dataset,
        pred=pred,
        fold=fold,
        variant=score_variant,
        market_benchmark=market_benchmark,
        position_multiplier=position_multiplier,
        log=log,
    )
    row["score_variant"] = score_variant
    return row


def summarize_importance(importance: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        importance.groupby(["fold", "feature_group"], as_index=False)
        .agg(gain_importance=("gain_importance", "sum"), split_importance=("split_importance", "sum"))
    )
    totals = grouped.groupby("fold")["gain_importance"].transform("sum").replace(0, np.nan)
    grouped["gain_share"] = grouped["gain_importance"] / totals
    return grouped.sort_values(["fold", "gain_share"], ascending=[True, False])


def build_comparison(score_ic: pd.DataFrame, portfolio: pd.DataFrame, deciles: pd.DataFrame) -> pd.DataFrame:
    test_ic = (
        score_ic.loc[score_ic["sample"].eq("test")]
        .groupby("score_variant", as_index=False)
        .agg(
            mean_test_rank_ic=("rank_ic_mean", "mean"),
            min_test_rank_ic=("rank_ic_mean", "min"),
            mean_test_icir=("icir", "mean"),
            positive_fold_count=("rank_ic_mean", lambda s: int((s > 0).sum())),
        )
    )
    port = (
        portfolio.groupby("score_variant", as_index=False)
        .agg(
            mean_annualized_return=("annualized_return", "mean"),
            mean_excess_vs_csi1000=("annualized_excess_return_vs_csi1000", "mean"),
            mean_sharpe=("sharpe", "mean"),
            worst_mdd=("max_drawdown", "min"),
            mean_turnover=("annualized_turnover", "mean"),
        )
    )
    decile = (
        deciles.groupby("score_variant", as_index=False)
        .agg(
            mean_decile_spread=("decile_spread", "mean"),
            positive_spread_ratio=("decile_spread", lambda s: float((s > 0).mean())),
            monotonic_ratio=("is_monotonic", "mean"),
        )
    )
    return port.merge(test_ic, on="score_variant", how="left").merge(decile, on="score_variant", how="left").sort_values(
        "mean_test_rank_ic", ascending=False
    )


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(
    output: Path,
    comparison: pd.DataFrame,
    group_ic: pd.DataFrame,
    group_weights: pd.DataFrame,
    group_contrib: pd.DataFrame,
    feature_group_importance: pd.DataFrame,
    score_ic: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> None:
    test_2026_groups = group_ic.loc[
        group_ic["fold"].eq("test_2026") & group_ic["sample"].isin(["train", "valid", "test"])
    ][["sample", "feature_group", "rank_ic_mean", "icir", "positive_ratio"]].sort_values(
        ["sample", "rank_ic_mean"], ascending=[True, False]
    )
    lines = [
        "# Model Weight Attribution And Group Reweighting",
        "",
        "## Question",
        "Could the ranker be over-relying on factors that were unusually strong in the validation year, "
        "causing lower sorting power later?",
        "",
        "## Method",
        "- Train the same B model: `cluster_stock_state + stock_state_low_vol`.",
        "- Decompose each prediction into factor-cluster SHAP contributions.",
        "- Compute each cluster contribution's own RankIC on train, valid and test.",
        "- Build two post-score variants from the same learned model contributions.",
        "- Backtest raw and post-score variants with the same Top5/10D/20bps/timing setup.",
        "",
        "## Score Variant Comparison",
        md_table(comparison, 20),
        "",
        "## 2026 Factor-Cluster Payoff IC",
        md_table(test_2026_groups, 80),
        "",
        "## 2026 Post-score Weights",
        md_table(
            group_weights.loc[group_weights["fold"].eq("test_2026")].sort_values(
                ["score_variant", "normalized_weight"], ascending=[True, False]
            ),
            80,
        ),
        "",
        "## Factor-Cluster SHAP Share",
        md_table(
            group_contrib.sort_values(["fold", "sample", "shap_abs_share"], ascending=[True, True, False]),
            120,
        ),
        "",
        "## Factor-Cluster Gain Share",
        md_table(feature_group_importance, 120),
        "",
        "## Train/Valid/Test RankIC",
        md_table(score_ic.sort_values(["fold", "score_variant", "sample"]), 80),
        "",
        "## Portfolio Metrics",
        md_table(
            portfolio[
                [
                    "score_variant",
                    "fold",
                    "annualized_return",
                    "annualized_excess_return_vs_csi1000",
                    "sharpe",
                    "max_drawdown",
                    "annualized_turnover",
                    "execution_rate",
                ]
            ].sort_values(["score_variant", "fold"]),
            80,
        ),
        "",
        "## Files",
        "- `score_group_contributions.parquet`",
        "- `factor_cluster_payoff_ic.csv`",
        "- `factor_cluster_postscore_weights.csv`",
        "- `factor_cluster_shap_contribution_summary.csv`",
        "- `factor_cluster_gain_importance.csv`",
        "- `score_variant_train_valid_test_ic.csv`",
        "- `score_variant_decile_spread.csv`",
        "- `score_variant_portfolio_metrics.csv`",
        "- `score_variant_comparison.csv`",
    ]
    (output / "model_weight_attribution_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
