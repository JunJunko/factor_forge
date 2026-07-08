from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sell_impact_sorting_repair as base


OUTPUT_ROOT = Path("artifacts/strategy_reviews")
BANDS = [0.85, 0.88, 0.90, 0.92, 0.95]
MODEL_VARIANTS = ["regime_aware_cluster_lgbm", "regime_aware_cluster_ranker"]
FOLDS = [
    {
        "fold": "test_2024",
        "train_start": "20220101",
        "train_end": "20221231",
        "valid_start": "20230101",
        "valid_end": "20231231",
        "test_start": "20240101",
        "test_end": "20241231",
    },
    {
        "fold": "test_2025",
        "train_start": "20220101",
        "train_end": "20231231",
        "valid_start": "20240101",
        "valid_end": "20241231",
        "test_start": "20250101",
        "test_end": "20251231",
    },
    {
        "fold": "test_2026",
        "train_start": "20220101",
        "train_end": "20241231",
        "valid_start": "20250101",
        "valid_end": "20251231",
        "test_start": "20260101",
        "test_end": "20260623",
    },
]


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_score_band_walkforward_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("loading panel and building 2022+ dataset")
    version, panel = base.load_panel()
    old_train_start = base.TRAIN_START
    base.TRAIN_START = "20220101"
    try:
        dataset = base.build_dataset(panel, log)
    finally:
        base.TRAIN_START = old_train_start
    dataset.to_parquet(output / "walkforward_dataset.parquet", index=False)
    log(f"dataset rows={len(dataset):,} dates={dataset['trade_date'].nunique():,} data_version={version}")

    rank_rows: list[dict] = []
    backtest_rows: list[dict] = []
    selection_rows: list[dict] = []
    audit_rows: list[dict] = []
    gain_frames: list[pd.DataFrame] = []
    shap_frames: list[pd.DataFrame] = []

    for fold in FOLDS:
        log(f"fold={fold['fold']} train={fold['train_start']}..{fold['train_end']} valid={fold['valid_start']}..{fold['valid_end']} test={fold['test_start']}..{fold['test_end']}")
        current = current_factor_predictions(dataset, fold)
        rank_rows.extend(rank_metrics(current, "current_factor", fold["fold"]))
        backtest_rows.extend(run_and_store_backtests(panel, dataset, current, "current_factor", fold, output, log))

        for variant in MODEL_VARIANTS:
            pred, gain, shap = fit_predict_fold(dataset, fold, variant, log)
            pred.to_parquet(output / f"predictions_{fold['fold']}_{variant}.parquet", index=False)
            if not gain.empty:
                gain["fold"] = fold["fold"]
                gain["variant"] = variant
                gain_frames.append(gain)
            if not shap.empty:
                shap["fold"] = fold["fold"]
                shap["variant"] = variant
                shap_frames.append(shap)

            direct_name = f"{variant}_direct_top"
            rank_rows.extend(rank_metrics(pred, direct_name, fold["fold"]))
            backtest_rows.extend(run_and_store_backtests(panel, dataset, pred, direct_name, fold, output, log))

            valid_candidates = []
            for band in BANDS:
                band_pred = score_band_predictions(pred, band)
                candidate_name = f"{variant}_band_{band:.2f}"
                valid_bt = run_backtests(panel, dataset, band_pred, candidate_name, fold, "valid")
                valid_top5 = pick_metric(valid_bt, top_n=5)
                valid_candidates.append(
                    {
                        "fold": fold["fold"],
                        "variant": variant,
                        "band": band,
                        "valid_top5_annualized_return": valid_top5["annualized_return"],
                        "valid_top5_excess": valid_top5["annualized_excess_return"],
                        "valid_top5_mdd": valid_top5["max_drawdown"],
                        "valid_top5_execution_rate": valid_top5["execution_rate"],
                    }
                )
            valid_df = pd.DataFrame(valid_candidates)
            selected = select_band(valid_df)
            selection_rows.extend(valid_candidates)
            selection_rows.append({**selected, "selected": True})
            selected_band = float(selected["band"])
            selected_pred = score_band_predictions(pred, selected_band)
            selected_name = f"{variant}_wf_selected_band_{selected_band:.2f}"
            rank_rows.extend(rank_metrics(selected_pred, selected_name, fold["fold"]))
            bt_rows = run_and_store_backtests(panel, dataset, selected_pred, selected_name, fold, output, log)
            backtest_rows.extend(bt_rows)
            audit_rows.extend(selection_audit(selected_pred, selected_name, fold, bt_rows))

    rank_df = pd.DataFrame(rank_rows)
    backtest_df = pd.DataFrame(backtest_rows)
    selection_df = pd.DataFrame(selection_rows)
    audit_df = pd.DataFrame(audit_rows)
    gains = pd.concat(gain_frames, ignore_index=True) if gain_frames else pd.DataFrame()
    shap_values = pd.concat(shap_frames, ignore_index=True) if shap_frames else pd.DataFrame()

    rank_df.to_csv(output / "rank_metrics.csv", index=False, encoding="utf-8-sig")
    backtest_df.to_csv(output / "backtest_metrics.csv", index=False, encoding="utf-8-sig")
    selection_df.to_csv(output / "band_selection.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(output / "selection_audit.csv", index=False, encoding="utf-8-sig")
    gains.to_csv(output / "feature_importance_gain.csv", index=False, encoding="utf-8-sig")
    shap_values.to_csv(output / "shap_like_contribution.csv", index=False, encoding="utf-8-sig")
    write_report(output, rank_df, backtest_df, selection_df, audit_df, gains, shap_values)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "data_version": version,
                "bands": BANDS,
                "model_variants": MODEL_VARIANTS,
                "folds": FOLDS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def fit_predict_fold(dataset: pd.DataFrame, fold: dict, variant: str, log) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import lightgbm as lgb

    features = base.features_for_variant(variant, dataset)
    train = base.sample_slice(dataset, fold["train_start"], fold["train_end"], features)
    valid = base.sample_slice(dataset, fold["valid_start"], fold["valid_end"], features)
    test = base.sample_slice(dataset, fold["test_start"], fold["test_end"], features)
    log(f"{fold['fold']} {variant} features={len(features)} train={len(train):,} valid={len(valid):,} test={len(test):,}")
    if len(train) < 20_000 or len(valid) < 5_000 or len(test) < 1_000:
        raise ValueError(f"not enough samples for {fold['fold']} {variant}")

    if "ranker" in variant:
        train = train.sort_values(["trade_date", "ts_code"])
        valid = valid.sort_values(["trade_date", "ts_code"])
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
    else:
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=350,
            learning_rate=0.03,
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
            train["label"],
            eval_set=[(valid[features], valid["label"])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )

    pred_frames = []
    for sample, frame in [("valid", valid), ("test", test)]:
        part = frame[["trade_date", "ts_code", "label"]].copy()
        part["score"] = model.predict(frame[features])
        part["sample"] = sample
        pred_frames.append(part)
    pred = pd.concat(pred_frames, ignore_index=True)
    gain = pd.DataFrame(
        {
            "feature": features,
            "gain_importance": model.booster_.feature_importance(importance_type="gain"),
            "split_importance": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("gain_importance", ascending=False)
    contrib = model.booster_.predict(test[features], pred_contrib=True)
    shap = pd.DataFrame(
        {
            "feature": features,
            "mean_abs_contribution_test": np.abs(contrib[:, :-1]).mean(axis=0),
            "mean_contribution_test": contrib[:, :-1].mean(axis=0),
        }
    ).sort_values("mean_abs_contribution_test", ascending=False)
    return pred, gain, shap


def current_factor_predictions(dataset: pd.DataFrame, fold: dict) -> pd.DataFrame:
    frame = dataset.loc[dataset["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["test_end"]))].copy()
    frame["score"] = frame["main_factor"]
    frame["sample"] = np.where(frame["trade_date"].le(pd.Timestamp(fold["valid_end"])), "valid", "test")
    return frame[["trade_date", "ts_code", "label", "score", "sample"]]


def score_band_predictions(pred: pd.DataFrame, band: float) -> pd.DataFrame:
    out = pred.copy()
    out["score_pct"] = out.groupby("trade_date")["score"].rank(pct=True, method="first")
    out["score"] = -(out["score_pct"] - band).abs()
    return out.drop(columns=["score_pct"])


def select_band(valid_df: pd.DataFrame) -> dict:
    candidates = valid_df.copy()
    candidates["objective"] = candidates["valid_top5_excess"]
    candidates.loc[candidates["valid_top5_mdd"].lt(-0.25), "objective"] -= 1.0
    candidates = candidates.sort_values(
        ["objective", "valid_top5_annualized_return", "valid_top5_execution_rate"],
        ascending=False,
    )
    selected = candidates.iloc[0].to_dict()
    selected["selected"] = True
    return selected


def rank_metrics(pred: pd.DataFrame, variant: str, fold: str) -> list[dict]:
    rows = []
    for sample, frame in pred.groupby("sample"):
        for item in base.evaluate_rank_metrics(frame, variant):
            item["fold"] = fold
            rows.append(item)
    return rows


def run_and_store_backtests(
    panel: pd.DataFrame,
    dataset: pd.DataFrame,
    pred: pd.DataFrame,
    variant: str,
    fold: dict,
    output: Path,
    log,
) -> list[dict]:
    rows = []
    for sample in ["valid", "test"]:
        bt_rows = run_backtests(panel, dataset, pred, variant, fold, sample)
        for row in bt_rows:
            row["fold"] = fold["fold"]
            rows.append(row)
        log(
            f"{fold['fold']} {variant} {sample}: "
            + ", ".join(
                f"top{r['top_n']} ann={r['annualized_return']:.2%} excess={r['annualized_excess_return']:.2%} mdd={r['max_drawdown']:.2%}"
                for r in bt_rows
            )
        )
    return rows


def run_backtests(panel: pd.DataFrame, dataset: pd.DataFrame, pred: pd.DataFrame, variant: str, fold: dict, sample: str) -> list[dict]:
    if sample == "valid":
        start, end = fold["valid_start"], fold["valid_end"]
    else:
        start, end = fold["test_start"], fold["test_end"]
    member = dataset.loc[dataset["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end)), ["trade_date", "ts_code", "condition_quantile"]].copy()
    member["selection_eligible"] = True
    factor_values = pred.loc[pred["sample"].eq(sample), ["trade_date", "ts_code", "score"]].rename(columns={"score": "factor_value"})
    panel_slice = panel.loc[panel["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    rows = []
    engine = base.BacktestEngine()
    for top_n in base.TOP_N:
        result = engine.run(
            panel_slice,
            factor_values,
            universe="liquid",
            top_n=top_n,
            holding_days=base.HOLDING_DAYS,
            initial_cash=1_000_000,
            lot_size=100,
            constraints=base.ExecutionConstraints(
                exclude_suspended=True,
                cannot_buy_limit_up=True,
                cannot_sell_limit_down=True,
                exclude_st=True,
                exclude_delisting_period=True,
                min_listing_days=60,
            ),
            cost_model=base.CostModel(
                commission_bps_per_side=3,
                slippage_bps_per_side=5,
                stamp_duty_bps_sell=5,
            ),
            cost_scenario_bps=base.COST_BPS,
            selection_membership=member,
        )
        daily_dir = Path("unused")
        row = {
            "variant": variant,
            "sample": sample,
            "top_n": top_n,
            "holding_days": base.HOLDING_DAYS,
            "cost_bps": base.COST_BPS,
            **result.metrics,
            "daily_return_sum": float(result.daily["return"].sum()),
            "largest_position_weight_max": float(result.daily["largest_position_weight"].max()) if "largest_position_weight" in result.daily else np.nan,
        }
        trades = result.trades.copy()
        if not trades.empty:
            row["top_stock_buy_share"] = stock_buy_share(trades)
            row["top_month_return_share"] = month_return_share(result.daily)
        else:
            row["top_stock_buy_share"] = np.nan
            row["top_month_return_share"] = np.nan
        rows.append(row)
    return rows


def pick_metric(rows: list[dict], top_n: int) -> dict:
    for row in rows:
        if int(row["top_n"]) == top_n:
            return row
    raise ValueError(f"top_n={top_n} not found")


def stock_buy_share(trades: pd.DataFrame) -> float:
    buys = trades.loc[trades["side"].eq("BUY")]
    if buys.empty or buys["gross_value"].sum() <= 0:
        return np.nan
    by_stock = buys.groupby("ts_code")["gross_value"].sum().sort_values(ascending=False)
    return float(by_stock.head(5).sum() / by_stock.sum())


def month_return_share(daily: pd.DataFrame) -> float:
    d = daily.copy()
    d["month"] = pd.to_datetime(d["trade_date"]).dt.to_period("M").astype(str)
    monthly = d.groupby("month")["return"].apply(lambda s: float((1.0 + s).prod() - 1.0))
    positive = monthly[monthly > 0]
    if positive.empty or positive.sum() <= 0:
        return np.nan
    return float(positive.max() / positive.sum())


def selection_audit(pred: pd.DataFrame, variant: str, fold: dict, bt_rows: list[dict]) -> list[dict]:
    rows = []
    p = pred.loc[pred["sample"].eq("test")].copy()
    p["selected_rank"] = p.groupby("trade_date")["score"].rank(ascending=False, method="first")
    for top_n in base.TOP_N:
        top = p.loc[p["selected_rank"].le(top_n)]
        metric = pick_metric([r for r in bt_rows if r["sample"] == "test"], top_n)
        rows.append(
            {
                "fold": fold["fold"],
                "variant": variant,
                "top_n": top_n,
                "avg_daily_candidates": float(p.groupby("trade_date").size().mean()),
                "avg_daily_selected": float(top.groupby("trade_date").size().mean()),
                "selected_stock_count": int(top["ts_code"].nunique()),
                "top5_stock_trade_share": metric.get("top_stock_buy_share"),
                "top_month_return_share": metric.get("top_month_return_share"),
                "execution_rate": metric.get("execution_rate"),
                "max_drawdown": metric.get("max_drawdown"),
            }
        )
    return rows


def write_report(
    output: Path,
    rank_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    gains: pd.DataFrame,
    shap_values: pd.DataFrame,
) -> None:
    selected = selection_df.loc[selection_df.get("selected", False).eq(True)] if "selected" in selection_df else pd.DataFrame()
    test_bt = backtest_df.loc[backtest_df["sample"].eq("test")].copy()
    selected_bt = test_bt[test_bt["variant"].str.contains("wf_selected_band", regex=False)]
    baseline_bt = test_bt[test_bt["variant"].isin(["current_factor", "regime_aware_cluster_lgbm_direct_top", "regime_aware_cluster_ranker_direct_top"])]
    test_rank = rank_df.loc[rank_df["sample"].eq("test")].copy()
    gain_summary = base.cluster_gain_summary(gains) if not gains.empty else pd.DataFrame()
    shap_summary = cluster_shap_summary(shap_values) if not shap_values.empty else pd.DataFrame()
    lines = [
        "# Score-Band Walk-Forward Validation",
        "",
        "## Protocol",
        "- Candidate bands are frozen: `0.85 / 0.88 / 0.90 / 0.92 / 0.95`.",
        "- Each fold selects the band only on the prior validation year, then applies it to the next test year.",
        "- Objective: valid Top5 annualized excess return, with a penalty for MDD below -25%.",
        "",
        "## Selected Bands",
        selected.round(6).to_markdown(index=False) if not selected.empty else "N/A",
        "",
        "## OOS Selected-Band Backtest",
        selected_bt.sort_values(["fold", "variant", "top_n"]).round(6).to_markdown(index=False),
        "",
        "## OOS Baselines",
        baseline_bt.sort_values(["fold", "variant", "top_n"]).round(6).to_markdown(index=False),
        "",
        "## OOS Rank Metrics",
        test_rank.sort_values(["fold", "rank_ic"], ascending=[True, False]).round(6).to_markdown(index=False),
        "",
        "## Selection Audit",
        audit_df.round(6).to_markdown(index=False) if not audit_df.empty else "N/A",
        "",
        "## Feature/Cluster Gain",
        gain_summary.round(6).to_markdown(index=False) if not gain_summary.empty else "N/A",
        "",
        "## SHAP-like Test Contribution",
        shap_summary.round(8).to_markdown(index=False) if not shap_summary.empty else "N/A",
        "",
        "## Decision Notes",
        "- Passing condition is not one hot year; selected-band OOS should beat current_factor in at least 2/3 folds and avoid concentration failures.",
        "- If 2026 improves but 2024/2025 fails, treat score-band as regime-specific and require an outer market gate.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cluster_shap_summary(shap_values: pd.DataFrame) -> pd.DataFrame:
    s = shap_values.copy()
    s["cluster"] = s["feature"].map(base.feature_cluster)
    return (
        s.groupby(["fold", "variant", "cluster"], dropna=False)[["mean_abs_contribution_test", "mean_contribution_test"]]
        .sum()
        .reset_index()
        .sort_values(["fold", "variant", "mean_abs_contribution_test"], ascending=[True, True, False])
    )


if __name__ == "__main__":
    main()
