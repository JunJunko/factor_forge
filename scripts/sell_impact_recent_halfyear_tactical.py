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

import sell_impact_condition_selected_regime_lgbm as selected_lgbm
import sell_impact_low_vol_regime_experiment as low_vol
import sell_impact_ranker_timing_compare as timing_compare
import sell_impact_score_band_walkforward as wf
import sell_impact_sorting_repair as base


OUTPUT_ROOT = Path("artifacts/strategy_reviews")
TEST_START = "20260101"
TEST_END = "20260623"
TOP_N = 5
HOLDING_DAYS = 10
COST_BPS = 20
INITIAL_CASH = 1_000_000
LOT_SIZE = 100
BANDS = [None, 0.70, 0.80, 0.85, 0.90, 0.95]

SPECS = [
    {
        "model": "B_long_2022_2024",
        "feature_set": "B",
        "train_start": "20220101",
        "train_end": "20241231",
        "valid_start": "20250101",
        "valid_end": "20251231",
        "weight_profile": "flat",
    },
    {
        "model": "K_long_2022_2024",
        "feature_set": "K",
        "train_start": "20220101",
        "train_end": "20241231",
        "valid_start": "20250101",
        "valid_end": "20251231",
        "weight_profile": "flat",
    },
    {
        "model": "B_recent_2024_2025q3",
        "feature_set": "B",
        "train_start": "20240101",
        "train_end": "20250930",
        "valid_start": "20251001",
        "valid_end": "20251231",
        "weight_profile": "flat",
    },
    {
        "model": "K_recent_2024_2025q3",
        "feature_set": "K",
        "train_start": "20240101",
        "train_end": "20250930",
        "valid_start": "20251001",
        "valid_end": "20251231",
        "weight_profile": "flat",
    },
    {
        "model": "K_recent_2025q1q3",
        "feature_set": "K",
        "train_start": "20250101",
        "train_end": "20250930",
        "valid_start": "20251001",
        "valid_end": "20251231",
        "weight_profile": "flat",
    },
    {
        "model": "K_recent_2024_2025q3_weighted",
        "feature_set": "K",
        "train_start": "20240101",
        "train_end": "20250930",
        "valid_start": "20251001",
        "valid_end": "20251231",
        "weight_profile": "recent_2025_3x",
    },
    {
        "model": "LV_recent_2024_2025q3",
        "feature_set": "LV",
        "train_start": "20240101",
        "train_end": "20250930",
        "valid_start": "20251001",
        "valid_end": "20251231",
        "weight_profile": "flat",
    },
    {
        "model": "LVPR_recent_2024_2025q3",
        "feature_set": "LVPR",
        "train_start": "20240101",
        "train_end": "20250930",
        "valid_start": "20251001",
        "valid_end": "20251231",
        "weight_profile": "flat",
    },
]


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_recent_halfyear_tactical_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    condition_run = selected_lgbm.latest_condition_run()
    selected = selected_lgbm.load_selected_interactions(condition_run)
    log(f"condition_run={condition_run}")
    log(f"selected_interactions alpha={selected['use_alpha'].sum()} all={selected['use_all'].sum()}")

    dataset = selected_lgbm.load_dataset(selected, log)
    dataset.to_parquet(output / "recent_halfyear_dataset.parquet", index=False)
    version, panel = base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    market_benchmark = timing_compare.load_market_benchmark(version)
    position_multiplier = timing_compare.load_position_multiplier(timing_compare.TIMING_DAILY)

    selected.to_csv(output / "selected_condition_interactions.csv", index=False, encoding="utf-8-sig")

    feature_maps: list[pd.DataFrame] = []
    training_rows: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []
    importance_rows: list[pd.DataFrame] = []
    contribution_rows: list[pd.DataFrame] = []
    exposure_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []

    for spec in SPECS:
        model_name = spec["model"]
        features = features_for_spec(dataset, selected, str(spec["feature_set"]))
        fmap = selected_lgbm.build_feature_map(features, model_name)
        feature_maps.append(fmap)
        train = base.sample_slice(dataset, spec["train_start"], spec["train_end"], features).sort_values(
            ["trade_date", "ts_code"]
        )
        valid = base.sample_slice(dataset, spec["valid_start"], spec["valid_end"], features).sort_values(
            ["trade_date", "ts_code"]
        )
        test = base.sample_slice(dataset, TEST_START, TEST_END, features).sort_values(["trade_date", "ts_code"])
        log(
            f"fit {model_name}: features={len(features)} train={len(train):,} "
            f"valid={len(valid):,} test={len(test):,} weights={spec['weight_profile']}"
        )
        model = fit_ranker(train, valid, features, str(spec["weight_profile"]))
        training_rows.append(
            {
                **spec,
                "feature_count": len(features),
                "train_rows": len(train),
                "valid_rows": len(valid),
                "test_rows": len(test),
                "best_iteration": int(model.best_iteration_ or model.n_estimators),
            }
        )
        for sample_name, frame in [("train", train), ("valid", valid), ("test_2026h1", test)]:
            pred = low_vol.predict_scores(model, frame, features)
            pred["sample"] = sample_name
            pred["model"] = model_name
            ic_rows.append(ic_summary(low_vol.daily_rank_ic(pred), model_name, sample_name))
            if sample_name == "test_2026h1":
                direct_pred = pred.copy()
                prediction_rows.append(direct_pred)
                importance_rows.append(feature_importance(model, features, fmap, model_name))
                contribution_rows.append(shap_contribution(model, test, features, fmap, model_name))
                exposure_rows.append(style_exposure(dataset, direct_pred, model_name, "direct"))
                for band in BANDS:
                    selection_name = "direct_top" if band is None else f"score_band_{band:.2f}"
                    selection_pred = direct_pred if band is None else score_band_predictions(direct_pred, band)
                    portfolio_rows.append(
                        run_backtest(
                            panel=panel,
                            dataset=dataset,
                            pred=selection_pred,
                            model_name=model_name,
                            selection=selection_name,
                            market_benchmark=market_benchmark,
                            position_multiplier=position_multiplier,
                            log=log,
                        )
                    )

    training = pd.DataFrame(training_rows)
    model_ic = pd.DataFrame(ic_rows)
    portfolio = pd.DataFrame(portfolio_rows)
    importance = pd.concat(importance_rows, ignore_index=True)
    contribution = pd.concat(contribution_rows, ignore_index=True)
    feature_map = pd.concat(feature_maps, ignore_index=True)
    exposure = pd.DataFrame(exposure_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    group_contribution = summarize_group_contribution(importance, contribution)
    comparison = build_comparison(portfolio, model_ic, exposure)

    training.to_csv(output / "lightgbm_training_result.csv", index=False, encoding="utf-8-sig")
    model_ic.to_csv(output / "model_train_valid_test_ic.csv", index=False, encoding="utf-8-sig")
    portfolio.to_csv(output / "recent_halfyear_portfolio_metrics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "recent_halfyear_comparison.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(output / "feature_importance_gain.csv", index=False, encoding="utf-8-sig")
    contribution.to_csv(output / "shap_contribution_summary.csv", index=False, encoding="utf-8-sig")
    group_contribution.to_csv(output / "factor_group_contribution.csv", index=False, encoding="utf-8-sig")
    feature_map.to_csv(output / "feature_map.csv", index=False, encoding="utf-8-sig")
    exposure.to_csv(output / "style_exposure.csv", index=False, encoding="utf-8-sig")
    predictions.to_parquet(output / "test_2026h1_predictions.parquet", index=False)
    write_report(output, condition_run, comparison, portfolio, model_ic, exposure, group_contribution)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "condition_run": str(condition_run),
                "data_version": version,
                "test_window": [TEST_START, TEST_END],
                "objective": "recent-halfyear tactical return, not long-horizon robustness",
                "top_n": TOP_N,
                "holding_days": HOLDING_DAYS,
                "cost_bps": COST_BPS,
                "timing_daily": str(timing_compare.TIMING_DAILY),
                "bands": BANDS,
                "specs": SPECS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def features_for_spec(dataset: pd.DataFrame, selected: pd.DataFrame, feature_set: str) -> list[str]:
    base_features = selected_lgbm.features_for_variant(dataset, selected, "B_original_plus_low_vol")
    if feature_set == "B":
        return base_features
    if feature_set == "K":
        return selected_lgbm.features_for_variant(dataset, selected, "K_condition_selected_alpha")
    if feature_set == "LV":
        chosen = selected.loc[selected["factor"].eq("stock_state_low_vol")].head(20)
    elif feature_set == "LVPR":
        low_vol_selected = selected.loc[selected["factor"].eq("stock_state_low_vol")].head(18)
        price_reversal_selected = selected.loc[selected["factor"].eq("cluster_price_reversal")].head(8)
        chosen = pd.concat([low_vol_selected, price_reversal_selected], ignore_index=True)
    else:
        raise ValueError(f"unknown feature_set={feature_set}")
    extras = [*chosen["state_z_feature"].drop_duplicates().tolist(), *chosen["interaction_feature"].tolist()]
    return [c for c in dict.fromkeys([*base_features, *extras]) if c in dataset.columns]


def fit_ranker(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], weight_profile: str):
    import lightgbm as lgb

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=220,
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
    weights = sample_weights(train, weight_profile)
    model.fit(
        train[features],
        base.relevance_labels(train),
        group=train.groupby("trade_date").size().to_list(),
        sample_weight=weights,
        eval_set=[(valid[features], base.relevance_labels(valid))],
        eval_group=[valid.groupby("trade_date").size().to_list()],
        callbacks=[lgb.early_stopping(25, verbose=False)],
    )
    return model


def sample_weights(train: pd.DataFrame, weight_profile: str) -> np.ndarray | None:
    if weight_profile == "flat":
        return None
    weights = pd.Series(1.0, index=train.index)
    if weight_profile == "recent_2025_3x":
        weights.loc[train["trade_date"].ge(pd.Timestamp("2025-01-01"))] = 3.0
        weights.loc[train["trade_date"].ge(pd.Timestamp("2025-07-01"))] = 4.0
        return weights.to_numpy()
    raise ValueError(f"unknown weight_profile={weight_profile}")


def score_band_predictions(pred: pd.DataFrame, band: float) -> pd.DataFrame:
    out = pred.copy()
    out["score_pct"] = out.groupby("trade_date")["score"].rank(pct=True, method="first")
    out["score"] = -(out["score_pct"] - band).abs()
    return out.drop(columns=["score_pct"])


def run_backtest(
    *,
    panel: pd.DataFrame,
    dataset: pd.DataFrame,
    pred: pd.DataFrame,
    model_name: str,
    selection: str,
    market_benchmark: pd.DataFrame,
    position_multiplier: pd.Series,
    log,
) -> dict[str, Any]:
    member = dataset.loc[
        dataset["trade_date"].between(pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)),
        ["trade_date", "ts_code", "condition_quantile"],
    ].copy()
    member["selection_eligible"] = True
    factor_values = pred[["trade_date", "ts_code", "score"]].rename(columns={"score": "factor_value"})
    panel_slice = panel.loc[panel["trade_date"].between(pd.Timestamp(TEST_START), pd.Timestamp(TEST_END))].copy()
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
        "model": model_name,
        "selection": selection,
        "top_n": TOP_N,
        "holding_days": HOLDING_DAYS,
        "cost_bps": COST_BPS,
        **result.metrics,
        "csi1000_annualized_return": csi1000,
        "annualized_excess_return_vs_csi1000": float(result.metrics["annualized_return"] - csi1000),
        "annualized_turnover": float(result.daily["portfolio_turnover"].mean() * 252),
        "top_stock_buy_share": timing_compare.stock_buy_share(result.trades),
        "top_month_return_share": timing_compare.month_return_share(result.daily),
    }
    log(
        f"bt {model_name} {selection}: ann={row['annualized_return']:.2%} "
        f"excess_csi1000={row['annualized_excess_return_vs_csi1000']:.2%} "
        f"mdd={row['max_drawdown']:.2%} exec={row.get('execution_rate', np.nan):.2%}"
    )
    return row


def feature_importance(model, features: list[str], fmap: pd.DataFrame, model_name: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "feature": features,
            "gain_importance": model.booster_.feature_importance(importance_type="gain"),
            "split_importance": model.booster_.feature_importance(importance_type="split"),
            "model": model_name,
        }
    )
    return out.merge(fmap[["feature", "feature_group"]], on="feature", how="left")


def shap_contribution(model, test: pd.DataFrame, features: list[str], fmap: pd.DataFrame, model_name: str) -> pd.DataFrame:
    contrib = model.booster_.predict(test[features], pred_contrib=True)
    data = pd.DataFrame(contrib[:, :-1], columns=features)
    mapping = fmap.set_index("feature")["feature_group"].to_dict()
    rows = []
    for feature in features:
        values = pd.to_numeric(data[feature], errors="coerce")
        rows.append(
            {
                "model": model_name,
                "feature": feature,
                "feature_group": mapping.get(feature, "other"),
                "mean_abs_shap": float(values.abs().mean()),
                "mean_shap": float(values.mean()),
                "positive_contribution_ratio": float((values > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def style_exposure(dataset: pd.DataFrame, pred: pd.DataFrame, model_name: str, selection: str) -> dict[str, Any]:
    row = low_vol.style_exposure(dataset, pred, model_name, selection)
    row["model"] = model_name
    row["selection"] = selection
    return row


def ic_summary(values: pd.Series, model_name: str, sample: str) -> dict[str, Any]:
    out = low_vol.ic_summary(values, model_name, "recent_halfyear", sample)
    out["model"] = out.pop("variant")
    return out


def summarize_group_contribution(importance: pd.DataFrame, contribution: pd.DataFrame) -> pd.DataFrame:
    gain = (
        importance.groupby(["model", "feature_group"], as_index=False)
        .agg(gain_importance=("gain_importance", "sum"), split_importance=("split_importance", "sum"))
    )
    shap = (
        contribution.groupby(["model", "feature_group"], as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "sum"), mean_shap=("mean_shap", "sum"))
    )
    merged = gain.merge(shap, on=["model", "feature_group"], how="outer").fillna(0.0)
    totals = merged.groupby("model")[["gain_importance", "mean_abs_shap"]].transform("sum").replace(0, np.nan)
    merged["gain_share"] = merged["gain_importance"] / totals["gain_importance"]
    merged["shap_abs_share"] = merged["mean_abs_shap"] / totals["mean_abs_shap"]
    return merged.sort_values(["model", "shap_abs_share"], ascending=[True, False])


def build_comparison(portfolio: pd.DataFrame, model_ic: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    test_ic = model_ic.loc[model_ic["sample"].eq("test_2026h1")]
    ic = test_ic[["model", "rank_ic_mean", "icir", "positive_ratio"]].rename(
        columns={
            "rank_ic_mean": "test_rank_ic",
            "icir": "test_icir",
            "positive_ratio": "test_positive_ratio",
        }
    )
    expo = exposure[
        [
            "model",
            "score_small_size_rank_corr",
            "score_low_vol_rank_corr",
            "top5_microcap_risk_share",
            "top5_mean_stock_state_small_size",
            "top5_mean_stock_state_low_vol",
        ]
    ]
    return (
        portfolio.merge(ic, on="model", how="left")
        .merge(expo, on="model", how="left")
        .sort_values(["annualized_return", "sharpe"], ascending=False)
    )


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(
    output: Path,
    condition_run: Path,
    comparison: pd.DataFrame,
    portfolio: pd.DataFrame,
    model_ic: pd.DataFrame,
    exposure: pd.DataFrame,
    group_contribution: pd.DataFrame,
) -> None:
    lines = [
        "# Recent Half-year Tactical Experiment",
        "",
        "## Scope",
        f"- Test window: `{TEST_START}` ~ `{TEST_END}`.",
        f"- Condition matrix source: `{condition_run}`.",
        "- Objective is recent-halfyear tactical return, not long-horizon robustness.",
        "- Main-board permission pool, Top5, 10-day holding, 20bps cost, timing overlay, CSI1000 benchmark.",
        "- Score-band rows are selected on 2026 H1 itself, so they are retrospective tactical candidates, not OOS proof.",
        "",
        "## Top Recent Candidates",
        md_table(
            comparison[
                [
                    "model",
                    "selection",
                    "annualized_return",
                    "annualized_excess_return_vs_csi1000",
                    "sharpe",
                    "max_drawdown",
                    "execution_rate",
                    "test_rank_ic",
                    "top_stock_buy_share",
                    "top_month_return_share",
                    "top5_microcap_risk_share",
                ]
            ],
            60,
        ),
        "",
        "## Direct Model IC",
        md_table(model_ic.sort_values(["model", "sample"]), 80),
        "",
        "## Direct Style Exposure",
        md_table(exposure.sort_values("model"), 80),
        "",
        "## Group Contribution",
        md_table(group_contribution, 120),
        "",
        "## Files",
        "- `recent_halfyear_comparison.csv`",
        "- `recent_halfyear_portfolio_metrics.csv`",
        "- `model_train_valid_test_ic.csv`",
        "- `style_exposure.csv`",
        "- `factor_group_contribution.csv`",
        "- `test_2026h1_predictions.parquet`",
    ]
    (output / "recent_halfyear_tactical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
