from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sell_impact_ranker_timing_compare as timing_compare
import sell_impact_score_band_walkforward as wf
import sell_impact_sorting_repair as base


SOURCE_RUN = Path("artifacts/strategy_reviews/sell_impact_score_band_walkforward_20260708T091419Z")
OUTPUT_ROOT = Path("artifacts/strategy_reviews")
TOP_N = 5
HOLDING_DAYS = 10
COST_BPS = 20
INITIAL_CASH = 1_000_000
LOT_SIZE = 100

VARIANTS = {
    "A_original_cluster_stock_state": "original cluster_stock_state feature set",
    "B_cluster_plus_low_vol": "cluster_stock_state + stock_state_low_vol",
    "H_cluster_low_vol_x_ret60": "cluster_stock_state + low_vol + low_vol x market_ret_60",
    "I_cluster_low_vol_x_vol20": "cluster_stock_state + low_vol + low_vol x market_vol_20",
    "J_cluster_low_vol_all_regime": "cluster_stock_state + low_vol + all low_vol regime interactions",
}


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_low_vol_regime_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("loading source walk-forward dataset and panel")
    dataset = load_dataset(log)
    version, panel = base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    market_benchmark = timing_compare.load_market_benchmark(version)
    position_multiplier = timing_compare.load_position_multiplier(timing_compare.TIMING_DAILY)

    rows: dict[str, list] = {
        "training": [],
        "model_ic": [],
        "yearly_ic": [],
        "portfolio": [],
        "importance": [],
        "contribution": [],
        "feature_map": [],
        "style_exposure": [],
        "predictions": [],
    }

    for variant in VARIANTS:
        features = features_for_variant(dataset, variant)
        fmap = build_feature_map(features, variant)
        rows["feature_map"].append(fmap)
        log(f"variant={variant} features={len(features)}")
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
            log(f"fit {variant} {fold_name}: train={len(train):,} valid={len(valid):,} test={len(test):,}")
            model = fit_ranker(train, valid, features)
            rows["training"].append(
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

            fold_predictions = []
            for sample_name, frame in [("train", train), ("valid", valid), ("test", test)]:
                pred = predict_scores(model, frame, features)
                pred["sample"] = sample_name
                pred["fold"] = fold_name
                pred["variant"] = variant
                fold_predictions.append(pred)
                rows["model_ic"].append(ic_summary(daily_rank_ic(pred), variant, fold_name, sample_name))

            pred_all = pd.concat(fold_predictions, ignore_index=True)
            rows["predictions"].append(pred_all)
            test_pred = pred_all.loc[pred_all["sample"].eq("test")].copy()
            rows["yearly_ic"].extend(yearly_model_ic(test_pred, variant, fold_name))
            rows["importance"].append(feature_importance(model, features, fmap, variant, fold_name))
            rows["contribution"].append(shap_contribution(model, test, features, fmap, variant, fold_name))
            rows["style_exposure"].append(style_exposure(dataset, test_pred, variant, fold_name))
            rows["portfolio"].append(
                run_backtest(
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

    result = materialize(rows)
    comparison = build_variant_comparison(result["portfolio"], result["model_ic"], result["style_exposure"])
    group_importance = summarize_group_importance(result["importance"], result["contribution"])
    low_vol_shap = summarize_low_vol_shap(group_importance)
    verdict = build_verdict(comparison, low_vol_shap)

    result["feature_map"].to_csv(output / "feature_map.csv", index=False, encoding="utf-8-sig")
    result["training"].to_csv(output / "lightgbm_training_result.csv", index=False, encoding="utf-8-sig")
    result["model_ic"].to_csv(output / "model_train_valid_test_ic.csv", index=False, encoding="utf-8-sig")
    result["yearly_ic"].to_csv(output / "model_yearly_ic.csv", index=False, encoding="utf-8-sig")
    result["portfolio"].to_csv(output / "portfolio_metrics.csv", index=False, encoding="utf-8-sig")
    result["importance"].to_csv(output / "feature_importance_gain.csv", index=False, encoding="utf-8-sig")
    result["contribution"].to_csv(output / "shap_contribution_summary.csv", index=False, encoding="utf-8-sig")
    group_importance.to_csv(output / "stock_state_group_contribution.csv", index=False, encoding="utf-8-sig")
    low_vol_shap.to_csv(output / "low_vol_shap_summary.csv", index=False, encoding="utf-8-sig")
    result["style_exposure"].to_csv(output / "style_exposure.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "variant_comparison_summary.csv", index=False, encoding="utf-8-sig")
    verdict.to_csv(output / "final_verdict.csv", index=False, encoding="utf-8-sig")
    result["predictions"].to_parquet(output / "predictions.parquet", index=False)
    write_report(output, comparison, result, group_importance, low_vol_shap, verdict)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "source_run": str(SOURCE_RUN),
                "data_version": version,
                "stock_pool": "main-board permission pool",
                "top_n": TOP_N,
                "holding_days": HOLDING_DAYS,
                "cost_bps": COST_BPS,
                "timing_daily": str(timing_compare.TIMING_DAILY),
                "variants": VARIANTS,
                "excluded_predictive_features": ["stock_state_small_size", "stock_state_microcap_risk"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def load_dataset(log) -> pd.DataFrame:
    dataset = pd.read_parquet(SOURCE_RUN / "walkforward_dataset.parquet")
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    before = len(dataset)
    dataset = dataset.loc[dataset["ts_code"].map(permission_eligible)].copy()
    log(f"permission filter: {before:,} -> {len(dataset):,} rows")
    dataset["stock_state_low_vol"] = -pd.to_numeric(dataset["volatility_20_z"], errors="coerce")
    dataset["stock_state_small_size"] = -pd.to_numeric(dataset["log_circ_mv_z"], errors="coerce")
    for regime in base.REGIME_COLS:
        dataset[f"stock_state_low_vol__x__regime_{regime}"] = dataset["stock_state_low_vol"] * dataset[regime]
    return dataset.replace([np.inf, -np.inf], np.nan)


def permission_eligible(ts_code: str) -> bool:
    code = str(ts_code)
    if code.endswith(".BJ"):
        return False
    if code.endswith(".SH") and code[:3] in {"688", "689"}:
        return False
    if code.endswith(".SZ") and code[:3] in {"300", "301", "302"}:
        return False
    return True


def features_for_variant(dataset: pd.DataFrame, variant: str) -> list[str]:
    original_interactions = [
        c
        for c in dataset.columns
        if "__x__regime_" in c and any(c.startswith(f"{cluster}__x__") for cluster in base.CLUSTER_COLS)
    ]
    original = [*base.CLUSTER_COLS, *base.REGIME_COLS, *original_interactions]
    if variant == "A_original_cluster_stock_state":
        return original
    if variant == "B_cluster_plus_low_vol":
        return [*original, "stock_state_low_vol"]
    if variant == "H_cluster_low_vol_x_ret60":
        return [*original, "stock_state_low_vol", "stock_state_low_vol__x__regime_market_ret_60"]
    if variant == "I_cluster_low_vol_x_vol20":
        return [*original, "stock_state_low_vol", "stock_state_low_vol__x__regime_market_vol_20"]
    if variant == "J_cluster_low_vol_all_regime":
        low_vol_interactions = [f"stock_state_low_vol__x__regime_{regime}" for regime in base.REGIME_COLS]
        return [*original, "stock_state_low_vol", *low_vol_interactions]
    raise ValueError(f"unknown variant: {variant}")


def build_feature_map(features: list[str], variant: str) -> pd.DataFrame:
    rows = []
    for feature in features:
        group = feature_group(feature)
        rows.append(
            {
                "variant": variant,
                "feature": feature,
                "feature_group": group,
                "is_stock_state_related": group.startswith(("stock_state", "old_cluster_stock_state")),
            }
        )
    return pd.DataFrame(rows)


def feature_group(feature: str) -> str:
    if feature == "cluster_stock_state":
        return "old_cluster_stock_state_direct"
    if feature.startswith("cluster_stock_state__x__"):
        return "old_cluster_stock_state_regime_interaction"
    if feature == "stock_state_low_vol":
        return "stock_state_low_vol_direct"
    if feature.startswith("stock_state_low_vol__x__"):
        return "stock_state_low_vol_regime_interaction"
    if feature in base.REGIME_COLS:
        return "market_regime_raw"
    for cluster in base.CLUSTER_COLS:
        if feature == cluster or feature.startswith(f"{cluster}__x__"):
            return cluster
    return "other"


def fit_ranker(train: pd.DataFrame, valid: pd.DataFrame, features: list[str]):
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
    )
    model.fit(
        train[features],
        base.relevance_labels(train),
        group=train.groupby("trade_date").size().to_list(),
        eval_set=[(valid[features], base.relevance_labels(valid))],
        eval_group=[valid.groupby("trade_date").size().to_list()],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model


def predict_scores(model, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = frame[["trade_date", "ts_code", "label"]].copy()
    out["score"] = model.predict(frame[features])
    return out


def feature_importance(model, features: list[str], fmap: pd.DataFrame, variant: str, fold: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "feature": features,
            "gain_importance": model.booster_.feature_importance(importance_type="gain"),
            "split_importance": model.booster_.feature_importance(importance_type="split"),
            "variant": variant,
            "fold": fold,
        }
    )
    return out.merge(fmap[["feature", "feature_group"]], on="feature", how="left")


def shap_contribution(model, test: pd.DataFrame, features: list[str], fmap: pd.DataFrame, variant: str, fold: str) -> pd.DataFrame:
    contrib = model.booster_.predict(test[features], pred_contrib=True)
    data = pd.DataFrame(contrib[:, :-1], columns=features)
    mapping = fmap.set_index("feature")["feature_group"].to_dict()
    rows = []
    for feature in features:
        values = pd.to_numeric(data[feature], errors="coerce")
        rows.append(
            {
                "variant": variant,
                "fold": fold,
                "feature": feature,
                "feature_group": mapping.get(feature, "other"),
                "mean_abs_shap": float(values.abs().mean()),
                "mean_shap": float(values.mean()),
                "positive_contribution_ratio": float((values > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def daily_rank_ic(pred: pd.DataFrame) -> pd.Series:
    values = []
    for _, group in pred.groupby("trade_date"):
        if len(group) < 30 or group["score"].nunique() < 2 or group["label"].nunique() < 2:
            continue
        value = group["score"].corr(group["label"], method="spearman")
        if pd.notna(value):
            values.append(float(value))
    return pd.Series(values, dtype=float)


def ic_summary(values: pd.Series, variant: str, fold: Any, sample: Any) -> dict[str, Any]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    mean = float(values.mean()) if len(values) else np.nan
    std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    return {
        "variant": variant,
        "fold": fold,
        "sample": sample,
        "days": int(len(values)),
        "rank_ic_mean": mean,
        "rank_ic_std": std,
        "icir": float(mean / std * np.sqrt(252)) if std and np.isfinite(std) and std > 0 else np.nan,
        "positive_ratio": float((values > 0).mean()) if len(values) else np.nan,
    }


def yearly_model_ic(pred: pd.DataFrame, variant: str, fold: str) -> list[dict[str, Any]]:
    out = []
    frame = pred.copy()
    frame["year"] = frame["trade_date"].dt.year
    for year, group in frame.groupby("year"):
        out.append(ic_summary(daily_rank_ic(group), variant, fold, int(year)))
    return out


def run_backtest(
    *,
    panel: pd.DataFrame,
    dataset: pd.DataFrame,
    pred: pd.DataFrame,
    fold: dict[str, str],
    variant: str,
    market_benchmark: pd.DataFrame,
    position_multiplier: pd.Series,
    log,
) -> dict[str, Any]:
    member = dataset.loc[
        dataset["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"])),
        ["trade_date", "ts_code", "condition_quantile"],
    ].copy()
    member["selection_eligible"] = True
    factor_values = pred[["trade_date", "ts_code", "score"]].rename(columns={"score": "factor_value"})
    panel_slice = panel.loc[panel["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))].copy()
    result = base.BacktestEngine().run(
        panel_slice,
        factor_values,
        universe="liquid",
        top_n=TOP_N,
        holding_days=HOLDING_DAYS,
        initial_cash=INITIAL_CASH,
        lot_size=LOT_SIZE,
        constraints=base.ExecutionConstraints(
            exclude_suspended=True,
            cannot_buy_limit_up=True,
            cannot_sell_limit_down=True,
            exclude_st=True,
            exclude_delisting_period=True,
            min_listing_days=60,
        ),
        cost_model=base.CostModel(commission_bps_per_side=3, slippage_bps_per_side=5, stamp_duty_bps_sell=5),
        cost_scenario_bps=COST_BPS,
        selection_membership=member,
        position_multiplier=position_multiplier,
        market_benchmark=market_benchmark,
    )
    csi1000 = float(result.metrics.get("market_index_annualized_return", np.nan))
    row = {
        "variant": variant,
        "fold": fold["fold"],
        "top_n": TOP_N,
        "holding_days": HOLDING_DAYS,
        "cost_bps": COST_BPS,
        **result.metrics,
        "csi1000_annualized_return": csi1000,
        "annualized_excess_return_vs_csi1000": float(result.metrics["annualized_return"] - csi1000),
        "annualized_turnover": float(result.daily["portfolio_turnover"].mean() * 252),
    }
    log(
        f"bt {variant} {fold['fold']} ann={row['annualized_return']:.2%} "
        f"excess_csi1000={row['annualized_excess_return_vs_csi1000']:.2%} "
        f"mdd={row['max_drawdown']:.2%}"
    )
    return row


def style_exposure(dataset: pd.DataFrame, pred: pd.DataFrame, variant: str, fold: str) -> dict[str, Any]:
    frame = pred.merge(
        dataset[["trade_date", "ts_code", "stock_state_small_size", "log_circ_mv_z", "stock_state_low_vol"]],
        on=["trade_date", "ts_code"],
        how="left",
    )
    score_size_corr = []
    score_low_vol_corr = []
    top_rows = []
    for _, group in frame.groupby("trade_date"):
        data = group.dropna(subset=["score", "stock_state_small_size", "stock_state_low_vol"])
        if len(data) >= 30 and data["score"].nunique() > 2:
            if data["stock_state_small_size"].nunique() > 2:
                score_size_corr.append(float(data["score"].corr(data["stock_state_small_size"], method="spearman")))
            if data["stock_state_low_vol"].nunique() > 2:
                score_low_vol_corr.append(float(data["score"].corr(data["stock_state_low_vol"], method="spearman")))
        top_rows.append(data.nlargest(TOP_N, "score"))
    top = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()
    return {
        "variant": variant,
        "fold": fold,
        "score_small_size_rank_corr": float(pd.Series(score_size_corr).mean()) if score_size_corr else np.nan,
        "score_low_vol_rank_corr": float(pd.Series(score_low_vol_corr).mean()) if score_low_vol_corr else np.nan,
        "top5_mean_stock_state_small_size": float(top["stock_state_small_size"].mean()) if not top.empty else np.nan,
        "top5_microcap_risk_share": float(top["stock_state_small_size"].gt(1.0).mean()) if not top.empty else np.nan,
        "top5_large_size_share": float(top["stock_state_small_size"].lt(-1.0).mean()) if not top.empty else np.nan,
        "top5_mean_stock_state_low_vol": float(top["stock_state_low_vol"].mean()) if not top.empty else np.nan,
        "top5_mean_log_circ_mv_z": float(top["log_circ_mv_z"].mean()) if not top.empty else np.nan,
    }


def materialize(rows: dict[str, list]) -> dict[str, pd.DataFrame]:
    return {
        "training": pd.DataFrame(rows["training"]),
        "model_ic": pd.DataFrame(rows["model_ic"]),
        "yearly_ic": pd.DataFrame(rows["yearly_ic"]),
        "portfolio": pd.DataFrame(rows["portfolio"]),
        "importance": pd.concat(rows["importance"], ignore_index=True),
        "contribution": pd.concat(rows["contribution"], ignore_index=True),
        "feature_map": pd.concat(rows["feature_map"], ignore_index=True),
        "style_exposure": pd.DataFrame(rows["style_exposure"]),
        "predictions": pd.concat(rows["predictions"], ignore_index=True),
    }


def summarize_group_importance(importance: pd.DataFrame, contribution: pd.DataFrame) -> pd.DataFrame:
    gain = (
        importance.groupby(["variant", "fold", "feature_group"], as_index=False)
        .agg(gain_importance=("gain_importance", "sum"), split_importance=("split_importance", "sum"))
    )
    shap = (
        contribution.groupby(["variant", "fold", "feature_group"], as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "sum"), mean_shap=("mean_shap", "sum"))
    )
    merged = gain.merge(shap, on=["variant", "fold", "feature_group"], how="outer").fillna(0.0)
    totals = merged.groupby(["variant", "fold"])[["gain_importance", "mean_abs_shap"]].transform("sum").replace(0, np.nan)
    merged["gain_share"] = merged["gain_importance"] / totals["gain_importance"]
    merged["shap_abs_share"] = merged["mean_abs_shap"] / totals["mean_abs_shap"]
    return merged.sort_values(["variant", "fold", "shap_abs_share"], ascending=[True, True, False])


def summarize_low_vol_shap(group_importance: pd.DataFrame) -> pd.DataFrame:
    groups = {"stock_state_low_vol_direct", "stock_state_low_vol_regime_interaction"}
    out = (
        group_importance.loc[group_importance["feature_group"].isin(groups)]
        .groupby(["variant", "fold"], as_index=False)
        .agg(low_vol_shap_share=("shap_abs_share", "sum"), low_vol_gain_share=("gain_share", "sum"))
    )
    return out.sort_values(["variant", "fold"])


def build_variant_comparison(portfolio: pd.DataFrame, model_ic: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    test_ic = model_ic.loc[model_ic["sample"].eq("test")].copy()
    ic = (
        test_ic.groupby("variant", as_index=False)
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
            mean_annualized_turnover=("annualized_turnover", "mean"),
            positive_excess_folds=("annualized_excess_return_vs_csi1000", lambda s: int((s > 0).sum())),
        )
    )
    expo = (
        exposure.groupby("variant", as_index=False)
        .agg(
            mean_score_small_size_rank_corr=("score_small_size_rank_corr", "mean"),
            mean_score_low_vol_rank_corr=("score_low_vol_rank_corr", "mean"),
            mean_top5_microcap_risk_share=("top5_microcap_risk_share", "mean"),
            mean_top5_stock_state_small_size=("top5_mean_stock_state_small_size", "mean"),
            mean_top5_stock_state_low_vol=("top5_mean_stock_state_low_vol", "mean"),
        )
    )
    return port.merge(ic, on="variant", how="left").merge(expo, on="variant", how="left").sort_values(
        "mean_excess_vs_csi1000", ascending=False
    )


def build_verdict(comparison: pd.DataFrame, low_vol_shap: pd.DataFrame) -> pd.DataFrame:
    base_row = comparison.loc[comparison["variant"].eq("A_original_cluster_stock_state")]
    base_excess = float(base_row["mean_excess_vs_csi1000"].iloc[0]) if len(base_row) else np.nan
    base_microcap = float(base_row["mean_top5_microcap_risk_share"].iloc[0]) if len(base_row) else np.nan
    rows = []
    low_vol_mean = low_vol_shap.groupby("variant")["low_vol_shap_share"].mean().to_dict() if not low_vol_shap.empty else {}
    for row in comparison.itertuples(index=False):
        excess_improved = bool(pd.notna(base_excess) and row.mean_excess_vs_csi1000 > base_excess)
        microcap_reduced = bool(pd.notna(base_microcap) and row.mean_top5_microcap_risk_share <= base_microcap)
        rows.append(
            {
                "variant": row.variant,
                "excess_vs_A_improved": excess_improved,
                "microcap_dependency_reduced_vs_A": microcap_reduced,
                "low_vol_shap_share_mean": low_vol_mean.get(row.variant, np.nan),
                "evidence": (
                    f"excess={row.mean_excess_vs_csi1000:.2%}; "
                    f"rank_ic={row.mean_test_rank_ic:.4f}; "
                    f"microcap={row.mean_top5_microcap_risk_share:.2%}"
                ),
            }
        )
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(
    output: Path,
    comparison: pd.DataFrame,
    result: dict[str, pd.DataFrame],
    group_importance: pd.DataFrame,
    low_vol_shap: pd.DataFrame,
    verdict: pd.DataFrame,
) -> None:
    stock_groups = group_importance.loc[
        group_importance["feature_group"].str.startswith("stock_state")
        | group_importance["feature_group"].str.startswith("old_cluster_stock_state")
    ].copy()
    lines = [
        "# Low Vol Regime Enhancement Experiment",
        "",
        "## Scope",
        "- Keep `cluster_stock_state` in all variants.",
        "- Do not include `stock_state_small_size` or microcap risk as predictive features.",
        "- Same source walk-forward dataset, label, split, main-board permission pool, Top5, 10-day holding, 20bps cost, timing overlay and CSI1000 benchmark.",
        "",
        "## Variants",
        *[f"- `{key}`: {value}" for key, value in VARIANTS.items()],
        "",
        "## Summary",
        md_table(comparison, 20),
        "",
        "## Test RankIC",
        md_table(
            result["model_ic"].loc[result["model_ic"]["sample"].eq("test")][
                ["variant", "fold", "rank_ic_mean", "icir", "positive_ratio"]
            ].sort_values(["variant", "fold"]),
            80,
        ),
        "",
        "## Portfolio By Year",
        md_table(
            result["portfolio"][
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
        "## Low Vol SHAP",
        md_table(low_vol_shap, 80),
        "",
        "## Style Exposure",
        md_table(result["style_exposure"].sort_values(["variant", "fold"]), 80),
        "",
        "## Stock-state SHAP Contribution",
        md_table(stock_groups.sort_values(["variant", "fold", "shap_abs_share"], ascending=[True, True, False]), 120),
        "",
        "## Verdict",
        md_table(verdict, 20),
        "",
        "## Files",
        "- `variant_comparison_summary.csv`",
        "- `model_train_valid_test_ic.csv`",
        "- `portfolio_metrics.csv`",
        "- `low_vol_shap_summary.csv`",
        "- `style_exposure.csv`",
        "- `stock_state_group_contribution.csv`",
        "- `final_verdict.csv`",
    ]
    (output / "low_vol_regime_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
