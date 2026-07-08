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
from factor_forge.backtest import BacktestEngine
from factor_forge.config import CostModel, ExecutionConstraints


SOURCE_RUN = Path("artifacts/strategy_reviews/sell_impact_score_band_walkforward_20260708T091419Z")
OUTPUT_ROOT = Path("artifacts/strategy_reviews")
MODEL_VARIANT = "regime_aware_cluster_ranker"
TOP_N = 5
HOLDING_DAYS = 10
COST_BPS = 20
INITIAL_CASH = 1_000_000
LOT_SIZE = 100


VARIANTS = {
    "A_original_cluster_stock_state": "原cluster_stock_state",
    "B_low_vol_only": "只拆出low_vol",
    "C_low_vol_plus_size": "low_vol + size",
    "D_low_vol_size_regime_interaction": "low_vol + size + regime interaction",
}


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_stock_state_refactor_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("loading dataset/panel")
    dataset = load_dataset(log)
    version, panel = base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    market_benchmark = timing_compare.load_market_benchmark(version)
    position_multiplier = timing_compare.load_position_multiplier(timing_compare.TIMING_DAILY)

    log("validating new factors")
    factor_validation = validate_new_factors(dataset, output)

    feature_map_rows: list[pd.DataFrame] = []
    training_rows: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []
    yearly_ic_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []
    importance_rows: list[pd.DataFrame] = []
    contribution_rows: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for variant in VARIANTS:
        features = features_for_variant(dataset, variant)
        fmap = build_feature_map(features, variant)
        feature_map_rows.append(fmap)
        log(f"variant={variant} features={len(features)}")
        for fold in wf.FOLDS:
            fold_name = fold["fold"]
            train = base.sample_slice(dataset, fold["train_start"], fold["train_end"], features).sort_values(["trade_date", "ts_code"])
            valid = base.sample_slice(dataset, fold["valid_start"], fold["valid_end"], features).sort_values(["trade_date", "ts_code"])
            test = base.sample_slice(dataset, fold["test_start"], fold["test_end"], features).sort_values(["trade_date", "ts_code"])
            log(f"fit {variant} {fold_name}: train={len(train):,} valid={len(valid):,} test={len(test):,}")
            model = fit_ranker(train, valid, features)
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

            pred_frames = []
            for sample_name, frame in [("train", train), ("valid", valid), ("test", test)]:
                pred = predict_scores(model, frame, features)
                pred["sample"] = sample_name
                pred["fold"] = fold_name
                pred["variant"] = variant
                pred_frames.append(pred)
                ic_rows.append({**ic_summary(daily_rank_ic(pred), variant, fold_name, sample_name)})
            pred_all = pd.concat(pred_frames, ignore_index=True)
            prediction_frames.append(pred_all)
            test_pred = pred_all.loc[pred_all["sample"].eq("test")].copy()
            yearly_ic_rows.extend(yearly_model_ic(test_pred, variant, fold_name))

            gain = feature_importance(model, features, fmap, variant, fold_name)
            importance_rows.append(gain)
            contribution_rows.append(shap_contribution(model, test, features, fmap, variant, fold_name))
            portfolio_rows.extend(
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

    feature_map = pd.concat(feature_map_rows, ignore_index=True)
    training = pd.DataFrame(training_rows)
    model_ic = pd.DataFrame(ic_rows)
    yearly_model_ic_df = pd.DataFrame(yearly_ic_rows)
    portfolio = pd.DataFrame(portfolio_rows)
    importance = pd.concat(importance_rows, ignore_index=True)
    contribution = pd.concat(contribution_rows, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    feature_group_importance = summarize_group_importance(importance, contribution)
    comparison = build_variant_comparison(portfolio, model_ic)
    recommendation = build_recommendation(comparison, feature_group_importance, factor_validation["factor_ic"])

    feature_map.to_csv(output / "feature_map.csv", index=False, encoding="utf-8-sig")
    training.to_csv(output / "lightgbm_training_result.csv", index=False, encoding="utf-8-sig")
    model_ic.to_csv(output / "model_train_valid_test_ic.csv", index=False, encoding="utf-8-sig")
    yearly_model_ic_df.to_csv(output / "model_yearly_ic.csv", index=False, encoding="utf-8-sig")
    portfolio.to_csv(output / "portfolio_ablation_metrics.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(output / "feature_importance_gain.csv", index=False, encoding="utf-8-sig")
    contribution.to_csv(output / "shap_contribution_summary.csv", index=False, encoding="utf-8-sig")
    feature_group_importance.to_csv(output / "stock_state_contribution_comparison.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "variant_comparison_summary.csv", index=False, encoding="utf-8-sig")
    predictions.to_parquet(output / "predictions.parquet", index=False)
    recommendation.to_csv(output / "final_recommendation.csv", index=False, encoding="utf-8-sig")

    write_report(
        output=output,
        factor_validation=factor_validation,
        training=training,
        model_ic=model_ic,
        yearly_model_ic=yearly_model_ic_df,
        portfolio=portfolio,
        group_importance=feature_group_importance,
        comparison=comparison,
        recommendation=recommendation,
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "source_run": str(SOURCE_RUN),
                "data_version": version,
                "stock_pool": "current production main-board permission pool",
                "label_and_splits": wf.FOLDS,
                "variants": VARIANTS,
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
        dataset[f"stock_state_small_size__x__regime_{regime}"] = dataset["stock_state_small_size"] * dataset[regime]
    return dataset


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
        if "__x__regime_" in c
        and any(c.startswith(f"{cluster}__x__") for cluster in base.CLUSTER_COLS)
    ]
    original = [*base.CLUSTER_COLS, *base.REGIME_COLS, *original_interactions]
    stock_state_removed = [f for f in original if f != "cluster_stock_state" and not f.startswith("cluster_stock_state__x__")]
    if variant == "A_original_cluster_stock_state":
        return original
    if variant == "B_low_vol_only":
        return [*stock_state_removed, "stock_state_low_vol"]
    if variant == "C_low_vol_plus_size":
        return [*stock_state_removed, "stock_state_low_vol", "stock_state_small_size"]
    if variant == "D_low_vol_size_regime_interaction":
        low_vol_interactions = [f"stock_state_low_vol__x__regime_{regime}" for regime in base.REGIME_COLS]
        size_interactions = [f"stock_state_small_size__x__regime_{regime}" for regime in base.REGIME_COLS]
        return [*stock_state_removed, "stock_state_low_vol", "stock_state_small_size", *low_vol_interactions, *size_interactions]
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
                "is_stock_state_related": group.startswith("stock_state") or group.startswith("old_cluster_stock_state"),
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
    if feature == "stock_state_small_size":
        return "stock_state_small_size_direct"
    if feature.startswith("stock_state_low_vol__x__"):
        return "stock_state_low_vol_regime_interaction"
    if feature.startswith("stock_state_small_size__x__"):
        return "stock_state_small_size_regime_interaction"
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
    rows = []
    mapping = fmap.set_index("feature")["feature_group"].to_dict()
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


def validate_new_factors(dataset: pd.DataFrame, output: Path) -> dict[str, pd.DataFrame]:
    factor_dir = output / "factor_validation"
    factor_dir.mkdir()
    factor_cols = [
        "stock_state_low_vol",
        "stock_state_small_size",
        *[f"stock_state_low_vol__x__regime_{regime}" for regime in base.REGIME_COLS],
        *[f"stock_state_small_size__x__regime_{regime}" for regime in base.REGIME_COLS],
    ]
    factor_rows = []
    yearly_rows = []
    decile_rows = []
    for factor in factor_cols:
        daily = daily_factor_ic(dataset, factor)
        factor_rows.append(factor_ic_summary(daily["rank_ic"], factor, "2024_2026"))
        if not daily.empty:
            daily["year"] = daily["trade_date"].dt.year
            for year, frame in daily.groupby("year"):
                yearly_rows.append(factor_ic_summary(frame["rank_ic"], factor, int(year)))
        dec = decile_return(dataset, factor)
        dec["factor"] = factor
        decile_rows.append(dec)
    factor_ic = pd.DataFrame(factor_rows).sort_values("rank_ic_mean", ascending=False)
    yearly_ic = pd.DataFrame(yearly_rows).sort_values(["factor", "window"])
    deciles = pd.concat(decile_rows, ignore_index=True)
    factor_ic.to_csv(factor_dir / "new_factor_ic.csv", index=False, encoding="utf-8-sig")
    yearly_ic.to_csv(factor_dir / "new_factor_yearly_ic.csv", index=False, encoding="utf-8-sig")
    deciles.to_csv(factor_dir / "new_factor_decile_returns.csv", index=False, encoding="utf-8-sig")
    return {"factor_ic": factor_ic, "yearly_ic": yearly_ic, "deciles": deciles}


def daily_factor_ic(frame: pd.DataFrame, factor: str) -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("trade_date"):
        data = group[[factor, "label"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 30 or data[factor].nunique() < 3 or data["label"].nunique() < 3:
            continue
        rows.append(
            {
                "trade_date": pd.Timestamp(date),
                "rank_ic": float(data[factor].corr(data["label"], method="spearman")),
                "pearson_ic": float(data[factor].corr(data["label"], method="pearson")),
                "n": int(len(data)),
            }
        )
    return pd.DataFrame(rows, columns=["trade_date", "rank_ic", "pearson_ic", "n"])


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


def factor_ic_summary(values: pd.Series, factor: str, window: Any) -> dict[str, Any]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    mean = float(values.mean()) if len(values) else np.nan
    std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    return {
        "factor": factor,
        "window": window,
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


def decile_return(frame: pd.DataFrame, factor: str, bins: int = 10) -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("trade_date"):
        data = group[[factor, "label"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < bins * 10 or data[factor].nunique() < bins:
            continue
        data = data.assign(decile=pd.qcut(data[factor].rank(method="first"), bins, labels=False) + 1)
        for decile, local in data.groupby("decile"):
            rows.append(
                {
                    "trade_date": pd.Timestamp(date),
                    "decile": int(decile),
                    "mean_forward_return": float(local["label"].mean()),
                    "count": int(len(local)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["decile", "mean_forward_return", "count", "days"])
    data = pd.DataFrame(rows)
    return (
        data.groupby("decile", as_index=False)
        .agg(mean_forward_return=("mean_forward_return", "mean"), count=("count", "mean"), days=("trade_date", "nunique"))
    )


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
) -> list[dict[str, Any]]:
    member = dataset.loc[
        dataset["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"])),
        ["trade_date", "ts_code", "condition_quantile"],
    ].copy()
    member["selection_eligible"] = True
    factor_values = pred[["trade_date", "ts_code", "score"]].rename(columns={"score": "factor_value"})
    panel_slice = panel.loc[panel["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))].copy()
    result = BacktestEngine().run(
        panel_slice,
        factor_values,
        universe="liquid",
        top_n=TOP_N,
        holding_days=HOLDING_DAYS,
        initial_cash=INITIAL_CASH,
        lot_size=LOT_SIZE,
        constraints=production_constraints(),
        cost_model=CostModel(commission_bps_per_side=3, slippage_bps_per_side=5, stamp_duty_bps_sell=5),
        cost_scenario_bps=COST_BPS,
        selection_membership=member,
        position_multiplier=position_multiplier,
        market_benchmark=market_benchmark,
    )
    annualized_turnover = float(result.daily["portfolio_turnover"].mean() * 252)
    row = {
        "variant": variant,
        "fold": fold["fold"],
        "top_n": TOP_N,
        "holding_days": HOLDING_DAYS,
        "cost_bps": COST_BPS,
        **result.metrics,
        "csi1000_annualized_return": result.metrics.get("market_index_annualized_return"),
        "annualized_excess_return_vs_csi1000": (
            result.metrics["annualized_return"] - result.metrics.get("market_index_annualized_return", np.nan)
        ),
        "annualized_turnover": annualized_turnover,
    }
    log(
        f"bt {variant} {fold['fold']} ann={row['annualized_return']:.2%} "
        f"excess_csi1000={row['annualized_excess_return_vs_csi1000']:.2%} "
        f"mdd={row['max_drawdown']:.2%}"
    )
    return [row]


def production_constraints() -> ExecutionConstraints:
    return ExecutionConstraints(
        exclude_suspended=True,
        cannot_buy_limit_up=True,
        cannot_sell_limit_down=True,
        exclude_st=True,
        exclude_delisting_period=True,
        min_listing_days=60,
    )


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


def build_variant_comparison(portfolio: pd.DataFrame, model_ic: pd.DataFrame) -> pd.DataFrame:
    test_ic = model_ic.loc[model_ic["sample"].eq("test")].copy()
    ic_summary_df = (
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
    return port.merge(ic_summary_df, on="variant", how="left").sort_values("mean_excess_vs_csi1000", ascending=False)


def build_recommendation(
    comparison: pd.DataFrame,
    group_importance: pd.DataFrame,
    factor_ic: pd.DataFrame,
) -> pd.DataFrame:
    best_variant = comparison.iloc[0]["variant"] if not comparison.empty else None
    old = comparison.loc[comparison["variant"].eq("A_original_cluster_stock_state")]
    best = comparison.loc[comparison["variant"].eq(best_variant)] if best_variant is not None else pd.DataFrame()
    low_vol_ic = factor_ic.loc[factor_ic["factor"].eq("stock_state_low_vol"), "rank_ic_mean"]
    size_ic = factor_ic.loc[factor_ic["factor"].eq("stock_state_small_size"), "rank_ic_mean"]
    d = group_importance.loc[group_importance["variant"].eq("D_low_vol_size_regime_interaction")]
    stock_related = d.loc[d["feature_group"].str.startswith("stock_state")]
    interaction_share = float(
        stock_related.loc[stock_related["feature_group"].str.contains("regime_interaction"), "shap_abs_share"].mean()
    ) if not stock_related.empty else np.nan
    best_has_all_positive_excess = bool(len(best) and int(best["positive_excess_folds"].iloc[0]) == 3)
    best_excess = float(best["mean_excess_vs_csi1000"].iloc[0]) if len(best) else np.nan
    old_excess = float(old["mean_excess_vs_csi1000"].iloc[0]) if len(old) else np.nan
    regime_answer = (
        "YES" if best_variant == "D_low_vol_size_regime_interaction" and best_has_all_positive_excess
        else ("PARTIAL" if best_variant == "D_low_vol_size_regime_interaction" else "NOT_CONFIRMED")
    )
    rows = [
        {
            "question": "cluster_stock_state是否应该删除",
            "answer": "YES_REPLACE_WITH_SPLIT_FEATURES" if best_variant != "A_original_cluster_stock_state" else "NO",
            "evidence": f"best_variant={best_variant}; mean_excess_old={old_excess:.4f}; mean_excess_best={best_excess:.4f}",
        },
        {
            "question": "low_vol是否应该成为核心Alpha",
            "answer": "YES" if len(low_vol_ic) and float(low_vol_ic.iloc[0]) > 0 else "NO",
            "evidence": f"stock_state_low_vol_rank_ic={float(low_vol_ic.iloc[0]) if len(low_vol_ic) else np.nan:.4f}",
        },
        {
            "question": "size是否应该降级为风险控制变量",
            "answer": "YES" if len(size_ic) and float(size_ic.iloc[0]) <= 0 else "WATCH",
            "evidence": f"stock_state_small_size_rank_ic={float(size_ic.iloc[0]) if len(size_ic) else np.nan:.4f}",
        },
        {
            "question": "regime interaction是否提升稳定性",
            "answer": regime_answer,
            "evidence": (
                f"D_interaction_mean_shap_share={interaction_share:.4f}; "
                f"positive_excess_folds={int(best['positive_excess_folds'].iloc[0]) if len(best) else 'NA'}/3"
            ),
        },
    ]
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(
    *,
    output: Path,
    factor_validation: dict[str, pd.DataFrame],
    training: pd.DataFrame,
    model_ic: pd.DataFrame,
    yearly_model_ic: pd.DataFrame,
    portfolio: pd.DataFrame,
    group_importance: pd.DataFrame,
    comparison: pd.DataFrame,
    recommendation: pd.DataFrame,
) -> None:
    stock_groups = group_importance.loc[
        group_importance["feature_group"].str.startswith("stock_state")
        | group_importance["feature_group"].str.startswith("old_cluster_stock_state")
    ].copy()
    lines = [
        "# cluster_stock_state Refactor Experiment",
        "",
        "## Scope",
        "- Same walk-forward dataset, labels, splits, timing overlay, Top5, holding-days and cost assumptions.",
        "- Stock pool: current production main-board permission pool.",
        "- Only feature engineering around `cluster_stock_state` is changed.",
        "",
        "## Variants",
        *[f"- `{key}`: {value}" for key, value in VARIANTS.items()],
        "",
        "## New Factor Validation",
        md_table(factor_validation["factor_ic"], 30),
        "",
        "## LightGBM Train / Valid / Test IC",
        md_table(model_ic.sort_values(["variant", "fold", "sample"]), 60),
        "",
        "## Portfolio Ablation Summary",
        md_table(comparison),
        "",
        "## Portfolio Metrics By Fold",
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
            60,
        ),
        "",
        "## Stock-state Contribution Comparison",
        md_table(stock_groups.sort_values(["variant", "fold", "shap_abs_share"], ascending=[True, True, False]), 80),
        "",
        "## Final Recommendation",
        md_table(recommendation),
        "",
        "## Files",
        "- `factor_validation/new_factor_ic.csv`",
        "- `factor_validation/new_factor_yearly_ic.csv`",
        "- `factor_validation/new_factor_decile_returns.csv`",
        "- `model_train_valid_test_ic.csv`",
        "- `model_yearly_ic.csv`",
        "- `portfolio_ablation_metrics.csv`",
        "- `feature_importance_gain.csv`",
        "- `shap_contribution_summary.csv`",
        "- `stock_state_contribution_comparison.csv`",
        "- `variant_comparison_summary.csv`",
        "- `final_recommendation.csv`",
    ]
    (output / "stock_state_refactor_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
