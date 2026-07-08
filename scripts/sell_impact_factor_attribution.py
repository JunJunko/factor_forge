from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sell_impact_score_band_walkforward as wf
import sell_impact_sorting_repair as base


SOURCE_RUN = Path("artifacts/strategy_reviews/sell_impact_score_band_walkforward_20260708T091419Z")
OUTPUT_ROOT = Path("artifacts/strategy_reviews")
MODEL_VARIANT = "regime_aware_cluster_ranker"
TOP_N = 5

FAMILY_NAMES = {
    "cluster_sell_impact": "卖压冲击修复",
    "cluster_condition_deviation": "异常偏离",
    "cluster_price_reversal": "价格反转",
    "cluster_liquidity": "流动性结构",
    "cluster_stock_state": "个股状态",
    "cluster_industry_context": "行业相对强弱",
    "cluster_market_context": "市场环境",
    "market_regime": "市场状态",
    "other": "其他",
}

FAMILY_INTERPRETATION = {
    "cluster_sell_impact": "下跌与卖压冲击后的修复主逻辑，解释事件本身是否有 alpha。",
    "cluster_condition_deviation": "当前冲击是否偏离个股自身历史常态，解释事件稀缺性。",
    "cluster_price_reversal": "短中期价格回撤后的反转弹性，解释是否买到反转 payoff。",
    "cluster_liquidity": "成交额、换手与放量结构，解释修复是否被流动性约束或拥挤。",
    "cluster_stock_state": "低波动、小市值等个股状态，解释风格暴露对排序的影响。",
    "cluster_industry_context": "个股相对行业强弱，解释主线/行业是否支持修复。",
    "cluster_market_context": "市场涨跌、波动、广度等环境本身的贡献。",
    "market_regime": "只来自 regime 原始特征，不绑定具体 alpha 事件。",
    "other": "未归入明确经济族的残余特征。",
}


def permission_eligible(ts_code: str) -> bool:
    code = str(ts_code)
    if code.endswith(".BJ"):
        return False
    if code.endswith(".SH") and code[:3] in {"688", "689"}:
        return False
    if code.endswith(".SZ") and code[:3] in {"300", "301", "302"}:
        return False
    return True


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_factor_attribution_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("loading walk-forward dataset")
    dataset = pd.read_parquet(SOURCE_RUN / "walkforward_dataset.parquet")
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    before_rows = len(dataset)
    dataset = dataset.loc[dataset["ts_code"].map(permission_eligible)].copy()
    log(f"permission filter: {before_rows:,} -> {len(dataset):,} rows")
    features = base.features_for_variant(MODEL_VARIANT, dataset)
    feature_map = build_feature_map(features)
    feature_map.to_csv(output / "factor_family_map.csv", index=False, encoding="utf-8-sig")

    all_pred_rows = []
    all_contrib_rows = []
    all_pick_rows = []
    family_shap_rows = []
    ablation_rows = []

    for fold in wf.FOLDS:
        fold_name = fold["fold"]
        log(f"fit full model {fold_name}")
        train = base.sample_slice(dataset, fold["train_start"], fold["train_end"], features).sort_values(["trade_date", "ts_code"])
        valid = base.sample_slice(dataset, fold["valid_start"], fold["valid_end"], features).sort_values(["trade_date", "ts_code"])
        test = base.sample_slice(dataset, fold["test_start"], fold["test_end"], features).sort_values(["trade_date", "ts_code"])
        model = fit_ranker(train, valid, features)

        pred, contrib, family_contrib = predict_with_family_contrib(model, test, features, feature_map)
        pred["fold"] = fold_name
        contrib["fold"] = fold_name
        family_contrib["fold"] = fold_name
        all_pred_rows.append(pred)
        all_contrib_rows.append(contrib)

        family_shap_rows.extend(family_shap_summary(family_contrib, fold_name))
        picks = top_pick_attribution(pred, family_contrib, fold_name)
        all_pick_rows.append(picks)

        full_metrics = rank_metrics(pred, fold_name, "FULL", "full")
        ablation_rows.extend(full_metrics)
        for family in ordered_families(feature_map):
            drop_features = feature_map.loc[feature_map["family"].ne(family), "feature"].tolist()
            if len(drop_features) == len(features) or len(drop_features) < 5:
                continue
            log(f"ablation {fold_name} drop={family} features={len(drop_features)}")
            drop_model = fit_ranker(train, valid, drop_features)
            drop_pred = predict_scores(drop_model, test, drop_features)
            ablation_rows.extend(rank_metrics(drop_pred, fold_name, family, "drop_family"))

    predictions = pd.concat(all_pred_rows, ignore_index=True)
    row_contrib = pd.concat(all_contrib_rows, ignore_index=True)
    top_picks = pd.concat(all_pick_rows, ignore_index=True)
    family_shap = pd.DataFrame(family_shap_rows)
    ablation = pd.DataFrame(ablation_rows)
    ablation_delta = build_ablation_delta(ablation)
    payoff = build_payoff_attribution(top_picks)
    yearly = build_yearly_family_summary(top_picks)

    predictions.to_parquet(output / "prediction_scores.parquet", index=False)
    row_contrib.to_parquet(output / "row_family_contribution.parquet", index=False)
    top_picks.to_csv(output / "top5_pick_attribution.csv", index=False, encoding="utf-8-sig")
    family_shap.to_csv(output / "family_shap_summary.csv", index=False, encoding="utf-8-sig")
    payoff.to_csv(output / "family_payoff_attribution.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_family_attribution.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(output / "family_ablation_rank_ic.csv", index=False, encoding="utf-8-sig")
    ablation_delta.to_csv(output / "family_ablation_delta.csv", index=False, encoding="utf-8-sig")
    write_report(output, feature_map, family_shap, payoff, yearly, ablation_delta)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "source_run": str(SOURCE_RUN),
                "model_variant": MODEL_VARIANT,
                "top_n": TOP_N,
                "folds": wf.FOLDS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def build_feature_map(features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        family = feature_family(feature)
        component = "interaction" if "__x__regime_" in feature else ("regime" if feature in base.REGIME_COLS else "direct")
        rows.append(
            {
                "feature": feature,
                "family": family,
                "family_name": FAMILY_NAMES.get(family, family),
                "component_type": component,
                "economic_interpretation": FAMILY_INTERPRETATION.get(family, ""),
            }
        )
    return pd.DataFrame(rows)


def feature_family(feature: str) -> str:
    for cluster in base.CLUSTER_COLS:
        if feature == cluster or feature.startswith(f"{cluster}__x__"):
            return cluster
    if feature in base.REGIME_COLS:
        return "market_regime"
    return "other"


def ordered_families(feature_map: pd.DataFrame) -> list[str]:
    preferred = [*base.CLUSTER_COLS, "market_regime", "other"]
    existing = set(feature_map["family"])
    return [item for item in preferred if item in existing]


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


def predict_with_family_contrib(
    model,
    frame: pd.DataFrame,
    features: list[str],
    feature_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = predict_scores(model, frame, features)
    contrib = model.booster_.predict(frame[features], pred_contrib=True)
    feature_contrib = pd.DataFrame(contrib[:, :-1], columns=features)
    feature_contrib.insert(0, "ts_code", frame["ts_code"].to_numpy())
    feature_contrib.insert(0, "trade_date", frame["trade_date"].to_numpy())
    feature_contrib["bias"] = contrib[:, -1]
    family = feature_map.set_index("feature")["family"].to_dict()
    family_values = feature_contrib[["trade_date", "ts_code"]].copy()
    for fam in ordered_families(feature_map):
        cols = [feature for feature in features if family.get(feature) == fam]
        family_values[fam] = feature_contrib[cols].sum(axis=1) if cols else 0.0
    family_values["bias"] = feature_contrib["bias"]
    family_values["score_from_contrib"] = family_values[[f for f in ordered_families(feature_map)]].sum(axis=1) + family_values["bias"]
    return pred, feature_contrib, family_values


def family_shap_summary(family_contrib: pd.DataFrame, fold: str) -> list[dict[str, Any]]:
    rows = []
    families = [c for c in family_contrib.columns if c.startswith("cluster_") or c in {"market_regime", "other"}]
    total_abs = family_contrib[families].abs().sum(axis=1).replace(0, np.nan)
    for family in families:
        values = pd.to_numeric(family_contrib[family], errors="coerce")
        rows.append(
            {
                "fold": fold,
                "family": family,
                "family_name": FAMILY_NAMES.get(family, family),
                "mean_abs_contribution": float(values.abs().mean()),
                "mean_signed_contribution": float(values.mean()),
                "mean_abs_share": float((values.abs() / total_abs).mean()),
                "positive_contribution_ratio": float((values > 0).mean()),
            }
        )
    return rows


def top_pick_attribution(pred: pd.DataFrame, family_contrib: pd.DataFrame, fold: str) -> pd.DataFrame:
    families = [c for c in family_contrib.columns if c.startswith("cluster_") or c in {"market_regime", "other"}]
    merged = pred.merge(family_contrib[["trade_date", "ts_code", *families]], on=["trade_date", "ts_code"], how="left")
    pieces = []
    for date, group in merged.groupby("trade_date"):
        g = group.sort_values(["score", "ts_code"], ascending=[False, True]).head(TOP_N).copy()
        if g.empty:
            continue
        abs_values = g[families].abs()
        total_abs = abs_values.sum(axis=1).replace(0, np.nan)
        for family in families:
            g[f"{family}_abs_share"] = abs_values[family] / total_abs
        g["dominant_family"] = abs_values.idxmax(axis=1)
        g["dominant_family_name"] = g["dominant_family"].map(FAMILY_NAMES)
        g["rank"] = np.arange(1, len(g) + 1)
        g["fold"] = fold
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def build_payoff_attribution(top_picks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    families = [c for c in top_picks.columns if c.startswith("cluster_") and not c.endswith("_abs_share")]
    for fold, frame in top_picks.groupby("fold"):
        for family in [*families, "market_regime", "other"]:
            if family not in frame:
                continue
            share_col = f"{family}_abs_share"
            values = pd.to_numeric(frame[family], errors="coerce")
            labels = pd.to_numeric(frame["label"], errors="coerce")
            weights = pd.to_numeric(frame.get(share_col), errors="coerce").fillna(0.0)
            dominant = frame["dominant_family"].eq(family)
            rows.append(
                {
                    "fold": fold,
                    "family": family,
                    "family_name": FAMILY_NAMES.get(family, family),
                    "mean_abs_share_in_top5": float(weights.mean()),
                    "mean_signed_contribution_in_top5": float(values.mean()),
                    "dominant_pick_count": int(dominant.sum()),
                    "dominant_pick_ratio": float(dominant.mean()),
                    "dominant_mean_forward_return": float(labels[dominant].mean()) if dominant.any() else np.nan,
                    "dominant_hit_rate": float((labels[dominant] > 0).mean()) if dominant.any() else np.nan,
                    "weighted_forward_return": safe_weighted_mean(labels, weights),
                }
            )
    return pd.DataFrame(rows)


def build_yearly_family_summary(top_picks: pd.DataFrame) -> pd.DataFrame:
    p = top_picks.copy()
    p["year"] = pd.to_datetime(p["trade_date"]).dt.year
    rows = []
    for year, frame in p.groupby("year"):
        payoff = build_payoff_attribution(frame.assign(fold=str(year)))
        payoff["year"] = int(year)
        rows.append(payoff)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def rank_metrics(pred: pd.DataFrame, fold: str, family: str, model_type: str) -> list[dict[str, Any]]:
    rows = []
    daily = []
    for date, group in pred.groupby("trade_date"):
        if len(group) < 20:
            continue
        rank_ic = group["score"].corr(group["label"], method="spearman")
        if pd.notna(rank_ic):
            daily.append(rank_ic)
    values = pd.Series(daily, dtype=float)
    rows.append(
        {
            "fold": fold,
            "model_type": model_type,
            "dropped_family": family,
            "days": int(len(values)),
            "rank_ic_mean": float(values.mean()) if len(values) else np.nan,
            "icir": float(values.mean() / values.std(ddof=1) * np.sqrt(252)) if len(values) > 1 and values.std(ddof=1) > 0 else np.nan,
            "positive_ratio": float((values > 0).mean()) if len(values) else np.nan,
            **topk_stats(pred, 5),
        }
    )
    return rows


def topk_stats(pred: pd.DataFrame, n: int) -> dict[str, float]:
    means = []
    hits = []
    for _, group in pred.groupby("trade_date"):
        top = group.sort_values(["score", "ts_code"], ascending=[False, True]).head(n)
        if top.empty:
            continue
        means.append(float(top["label"].mean()))
        hits.append(float((top["label"] > 0).mean()))
    return {
        f"top{n}_mean_label": float(np.mean(means)) if means else np.nan,
        f"top{n}_hit_rate": float(np.mean(hits)) if hits else np.nan,
    }


def build_ablation_delta(ablation: pd.DataFrame) -> pd.DataFrame:
    full = ablation.loc[ablation["model_type"].eq("full")].set_index("fold")
    rows = []
    for row in ablation.loc[ablation["model_type"].eq("drop_family")].itertuples(index=False):
        base_row = full.loc[row.fold]
        rows.append(
            {
                "fold": row.fold,
                "dropped_family": row.dropped_family,
                "family_name": FAMILY_NAMES.get(row.dropped_family, row.dropped_family),
                "full_rank_ic": float(base_row.rank_ic_mean),
                "drop_rank_ic": float(row.rank_ic_mean),
                "rank_ic_loss": float(base_row.rank_ic_mean - row.rank_ic_mean),
                "full_top5_mean_label": float(base_row.top5_mean_label),
                "drop_top5_mean_label": float(row.top5_mean_label),
                "top5_mean_label_loss": float(base_row.top5_mean_label - row.top5_mean_label),
                "full_top5_hit_rate": float(base_row.top5_hit_rate),
                "drop_top5_hit_rate": float(row.top5_hit_rate),
                "top5_hit_rate_loss": float(base_row.top5_hit_rate - row.top5_hit_rate),
            }
        )
    return pd.DataFrame(rows).sort_values(["fold", "rank_ic_loss"], ascending=[True, False])


def safe_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any() or w[mask].sum() <= 0:
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def md_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(
    output: Path,
    feature_map: pd.DataFrame,
    family_shap: pd.DataFrame,
    payoff: pd.DataFrame,
    yearly: pd.DataFrame,
    ablation_delta: pd.DataFrame,
) -> None:
    shap_summary = (
        family_shap.groupby(["family", "family_name"], as_index=False)
        .agg(
            mean_abs_share=("mean_abs_share", "mean"),
            mean_abs_contribution=("mean_abs_contribution", "mean"),
            mean_signed_contribution=("mean_signed_contribution", "mean"),
            positive_contribution_ratio=("positive_contribution_ratio", "mean"),
        )
        .sort_values("mean_abs_share", ascending=False)
    )
    payoff_summary = (
        payoff.groupby(["family", "family_name"], as_index=False)
        .agg(
            mean_abs_share_in_top5=("mean_abs_share_in_top5", "mean"),
            dominant_pick_ratio=("dominant_pick_ratio", "mean"),
            dominant_mean_forward_return=("dominant_mean_forward_return", "mean"),
            dominant_hit_rate=("dominant_hit_rate", "mean"),
            weighted_forward_return=("weighted_forward_return", "mean"),
        )
        .sort_values("mean_abs_share_in_top5", ascending=False)
    )
    ablation_summary = (
        ablation_delta.groupby(["dropped_family", "family_name"], as_index=False)
        .agg(
            rank_ic_loss_mean=("rank_ic_loss", "mean"),
            top5_mean_label_loss_mean=("top5_mean_label_loss", "mean"),
            top5_hit_rate_loss_mean=("top5_hit_rate_loss", "mean"),
        )
        .sort_values("rank_ic_loss_mean", ascending=False)
    )
    lines = [
        "# Sell Impact Factor Attribution",
        "",
        "## 结论摘要",
        "- `mean_abs_share` 表示该因子族对模型分数波动的解释占比，不等于收益贡献。",
        "- `dominant_mean_forward_return` 表示 Top5 中由该族主导的股票后续 10 日开盘到开盘平均收益。",
        "- `rank_ic_loss` 表示剔除该族后 RankIC 下降幅度，越大说明该族对排序越不可替代。",
        "",
        "## 因子族映射",
        md_table(feature_map[["family", "family_name", "component_type", "feature"]].sort_values(["family", "component_type", "feature"]), 30),
        "",
        "## 1. 模型分数归因",
        md_table(shap_summary),
        "",
        "## 2. Top5 入选股票 payoff 归因",
        md_table(payoff_summary),
        "",
        "## 3. 年度 Top5 归因",
        md_table(yearly.sort_values(["year", "mean_abs_share_in_top5"], ascending=[True, False]), 40),
        "",
        "## 4. 因子族剔除实验",
        md_table(ablation_summary),
        "",
        "## 输出文件",
        "- `factor_family_map.csv`: 原始特征到经济因子族映射。",
        "- `family_shap_summary.csv`: 每年/fold 的因子族 SHAP 贡献。",
        "- `top5_pick_attribution.csv`: 每个 Top5 入选股票的逐族贡献。",
        "- `family_payoff_attribution.csv`: 因子族贡献与后续收益的对应关系。",
        "- `yearly_family_attribution.csv`: 年度归因。",
        "- `family_ablation_rank_ic.csv`: full/drop-family 原始结果。",
        "- `family_ablation_delta.csv`: 剔除某族后的排序损失。",
    ]
    (output / "factor_attribution_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
