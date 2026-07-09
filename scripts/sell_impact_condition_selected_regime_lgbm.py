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

import sell_impact_factor_regime_condition_matrix as condition
import sell_impact_low_vol_regime_experiment as low_vol
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
    "A_original_regime_ranker": "original cluster + 6 market regime + original cluster*regime interactions",
    "B_original_plus_low_vol": "original feature set + stock_state_low_vol",
    "K_condition_selected_alpha": "B + condition-matrix selected interactions for low_vol/price_reversal/sell_impact",
    "L_condition_selected_all_reliable": "B + condition-matrix selected interactions including liquidity/industry inverse risk families",
}

ALPHA_INTERACTION_FACTORS = {"stock_state_low_vol", "cluster_price_reversal", "cluster_sell_impact"}
ALL_RELIABLE_FACTORS = {
    "stock_state_low_vol",
    "cluster_price_reversal",
    "cluster_sell_impact",
    "cluster_industry_context",
    "cluster_liquidity",
}
MAX_ALPHA_INTERACTIONS = 30
MAX_ALL_INTERACTIONS = 45


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_condition_selected_regime_lgbm_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    condition_run = latest_condition_run()
    log(f"condition_run={condition_run}")
    selected = load_selected_interactions(condition_run)
    selected.to_csv(output / "selected_condition_interactions.csv", index=False, encoding="utf-8-sig")
    log(
        "selected interactions: "
        f"alpha={selected['use_alpha'].sum()} all={selected['use_all'].sum()} "
        f"axes={selected['state_axis'].nunique()}"
    )

    log("loading source walk-forward dataset and panel")
    dataset = load_dataset(selected, log)
    dataset.to_parquet(output / "condition_selected_dataset.parquet", index=False)
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
        features = features_for_variant(dataset, selected, variant)
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
            model = low_vol.fit_ranker(train, valid, features)
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
                pred = low_vol.predict_scores(model, frame, features)
                pred["sample"] = sample_name
                pred["fold"] = fold_name
                pred["variant"] = variant
                fold_predictions.append(pred)
                rows["model_ic"].append(low_vol.ic_summary(low_vol.daily_rank_ic(pred), variant, fold_name, sample_name))

            pred_all = pd.concat(fold_predictions, ignore_index=True)
            rows["predictions"].append(pred_all)
            test_pred = pred_all.loc[pred_all["sample"].eq("test")].copy()
            rows["yearly_ic"].extend(low_vol.yearly_model_ic(test_pred, variant, fold_name))
            rows["importance"].append(low_vol.feature_importance(model, features, fmap, variant, fold_name))
            rows["contribution"].append(low_vol.shap_contribution(model, test, features, fmap, variant, fold_name))
            rows["style_exposure"].append(low_vol.style_exposure(dataset, test_pred, variant, fold_name))
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
    comparison = low_vol.build_variant_comparison(result["portfolio"], result["model_ic"], result["style_exposure"])
    group_importance = low_vol.summarize_group_importance(result["importance"], result["contribution"])
    interaction_shap = summarize_interaction_shap(group_importance)
    verdict = build_verdict(comparison, interaction_shap)

    result["feature_map"].to_csv(output / "feature_map.csv", index=False, encoding="utf-8-sig")
    result["training"].to_csv(output / "lightgbm_training_result.csv", index=False, encoding="utf-8-sig")
    result["model_ic"].to_csv(output / "model_train_valid_test_ic.csv", index=False, encoding="utf-8-sig")
    result["yearly_ic"].to_csv(output / "model_yearly_ic.csv", index=False, encoding="utf-8-sig")
    result["portfolio"].to_csv(output / "portfolio_metrics.csv", index=False, encoding="utf-8-sig")
    result["importance"].to_csv(output / "feature_importance_gain.csv", index=False, encoding="utf-8-sig")
    result["contribution"].to_csv(output / "shap_contribution_summary.csv", index=False, encoding="utf-8-sig")
    result["style_exposure"].to_csv(output / "style_exposure.csv", index=False, encoding="utf-8-sig")
    group_importance.to_csv(output / "factor_group_contribution.csv", index=False, encoding="utf-8-sig")
    interaction_shap.to_csv(output / "selected_interaction_shap_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "variant_comparison_summary.csv", index=False, encoding="utf-8-sig")
    verdict.to_csv(output / "final_verdict.csv", index=False, encoding="utf-8-sig")
    result["predictions"].to_parquet(output / "predictions.parquet", index=False)
    write_report(output, condition_run, selected, comparison, result, group_importance, interaction_shap, verdict)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "source_run": str(SOURCE_RUN),
                "condition_run": str(condition_run),
                "data_version": version,
                "stock_pool": "main-board permission pool",
                "top_n": TOP_N,
                "holding_days": HOLDING_DAYS,
                "cost_bps": COST_BPS,
                "timing_daily": str(timing_compare.TIMING_DAILY),
                "variants": VARIANTS,
                "selection_rules": {
                    "alpha_interaction_factors": sorted(ALPHA_INTERACTION_FACTORS),
                    "all_reliable_factors": sorted(ALL_RELIABLE_FACTORS),
                    "max_alpha_interactions": MAX_ALPHA_INTERACTIONS,
                    "max_all_interactions": MAX_ALL_INTERACTIONS,
                    "excluded_dynamic_factor": "cluster_stock_state",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def latest_condition_run() -> Path:
    candidates = sorted(
        OUTPUT_ROOT.glob("sell_impact_factor_regime_condition_matrix_*/factor_regime_interaction_candidates.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("no condition matrix run found")
    return candidates[0].parent


def load_selected_interactions(condition_run: Path) -> pd.DataFrame:
    candidates = pd.read_csv(condition_run / "factor_regime_interaction_candidates.csv")
    candidates = candidates.loc[candidates["reliable_state_dependency"].astype(bool)].copy()
    candidates = candidates.loc[candidates["factor"].isin(ALL_RELIABLE_FACTORS)].copy()
    candidates = candidates.sort_values(["candidate_score", "rank_ic_mean"], ascending=False)
    candidates = candidates.drop_duplicates(["factor", "state_axis"], keep="first")
    candidates["interaction_feature"] = candidates.apply(
        lambda row: interaction_feature_name(str(row["factor"]), str(row["state_axis"])),
        axis=1,
    )
    candidates["state_z_feature"] = candidates["state_axis"].map(state_z_feature_name)
    alpha = candidates.loc[candidates["factor"].isin(ALPHA_INTERACTION_FACTORS)].head(MAX_ALPHA_INTERACTIONS)
    all_reliable = candidates.head(MAX_ALL_INTERACTIONS)
    selected = candidates.loc[candidates.index.isin(set(alpha.index) | set(all_reliable.index))].copy()
    selected["use_alpha"] = selected.index.isin(alpha.index)
    selected["use_all"] = selected.index.isin(all_reliable.index)
    return selected.reset_index(drop=True)


def load_dataset(selected: pd.DataFrame, log) -> pd.DataFrame:
    dataset = pd.read_parquet(SOURCE_RUN / "walkforward_dataset.parquet")
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    before = len(dataset)
    dataset = dataset.loc[dataset["ts_code"].map(low_vol.permission_eligible)].copy()
    log(f"permission filter: {before:,} -> {len(dataset):,} rows")
    dataset["stock_state_low_vol"] = -pd.to_numeric(dataset["volatility_20_z"], errors="coerce")
    dataset["stock_state_small_size"] = -pd.to_numeric(dataset["log_circ_mv_z"], errors="coerce")

    axes = sorted(selected["state_axis"].unique())
    state = build_state_axes(dataset, axes, log)
    dataset = dataset.merge(state, on="trade_date", how="left")

    for row in selected.itertuples(index=False):
        factor = str(row.factor)
        state_z = str(row.state_z_feature)
        feature = str(row.interaction_feature)
        dataset[feature] = pd.to_numeric(dataset[factor], errors="coerce") * pd.to_numeric(dataset[state_z], errors="coerce")
    return dataset.replace([np.inf, -np.inf], np.nan)


def build_state_axes(dataset: pd.DataFrame, axes: list[str], log) -> pd.DataFrame:
    timing_path = condition.latest_timing_dataset()
    timing = condition.load_timing_states(timing_path)
    panel_sentiment = condition.load_panel_sentiment()
    state_frame, state_meta = condition.build_state_frame(dataset, timing, panel_sentiment)
    available = [axis for axis in axes if axis in state_frame.columns]
    missing = sorted(set(axes) - set(available))
    if missing:
        log(f"missing selected state axes skipped: {missing}")
    state = state_frame[["trade_date", *available]].copy().sort_values("trade_date")
    for axis in available:
        values = pd.to_numeric(state[axis], errors="coerce")
        mean = values.expanding(min_periods=20).mean()
        std = values.expanding(min_periods=20).std(ddof=0).replace(0.0, np.nan)
        state[state_z_feature_name(axis)] = ((values - mean) / std).clip(-5, 5).fillna(0.0)
    keep = ["trade_date", *[state_z_feature_name(axis) for axis in available]]
    log(f"state axes merged={len(available)} timing={timing_path} meta_axes={len(state_meta)}")
    return state[keep]


def features_for_variant(dataset: pd.DataFrame, selected: pd.DataFrame, variant: str) -> list[str]:
    original_interactions = [
        c
        for c in dataset.columns
        if "__x__regime_" in c and any(c.startswith(f"{cluster}__x__") for cluster in base.CLUSTER_COLS)
    ]
    original = [*base.CLUSTER_COLS, *base.REGIME_COLS, *original_interactions]
    if variant == "A_original_regime_ranker":
        return [c for c in original if c in dataset.columns]
    if variant == "B_original_plus_low_vol":
        return [c for c in [*original, "stock_state_low_vol"] if c in dataset.columns]
    if variant == "K_condition_selected_alpha":
        chosen = selected.loc[selected["use_alpha"].astype(bool)]
    elif variant == "L_condition_selected_all_reliable":
        chosen = selected.loc[selected["use_all"].astype(bool)]
    else:
        raise ValueError(f"unknown variant: {variant}")
    extras = [*chosen["state_z_feature"].drop_duplicates().tolist(), *chosen["interaction_feature"].tolist()]
    return [c for c in dict.fromkeys([*original, "stock_state_low_vol", *extras]) if c in dataset.columns]


def build_feature_map(features: list[str], variant: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant": variant,
                "feature": feature,
                "feature_group": feature_group(feature),
                "is_stock_state_related": feature_group(feature).startswith(("stock_state", "old_cluster_stock_state")),
            }
            for feature in features
        ]
    )


def feature_group(feature: str) -> str:
    if feature.endswith("__state_z"):
        return "market_state_axis"
    if "__x__state_" in feature:
        return f"selected_interaction::{feature.split('__x__state_', 1)[0]}"
    return low_vol.feature_group(feature)


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
        "top_stock_buy_share": timing_compare.stock_buy_share(result.trades),
        "top_month_return_share": timing_compare.month_return_share(result.daily),
    }
    log(
        f"bt {variant} {fold['fold']} ann={row['annualized_return']:.2%} "
        f"excess_csi1000={row['annualized_excess_return_vs_csi1000']:.2%} "
        f"mdd={row['max_drawdown']:.2%} exec={row.get('execution_rate', np.nan):.2%}"
    )
    return row


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


def summarize_interaction_shap(group_importance: pd.DataFrame) -> pd.DataFrame:
    frame = group_importance.loc[group_importance["feature_group"].str.startswith("selected_interaction::")].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=["variant", "fold", "selected_interaction_shap_share", "selected_interaction_gain_share"]
        )
    return (
        frame.groupby(["variant", "fold"], as_index=False)
        .agg(
            selected_interaction_shap_share=("shap_abs_share", "sum"),
            selected_interaction_gain_share=("gain_share", "sum"),
        )
        .sort_values(["variant", "fold"])
    )


def build_verdict(comparison: pd.DataFrame, interaction_shap: pd.DataFrame) -> pd.DataFrame:
    base_row = comparison.loc[comparison["variant"].eq("B_original_plus_low_vol")]
    base_excess = float(base_row["mean_excess_vs_csi1000"].iloc[0]) if len(base_row) else np.nan
    base_ic = float(base_row["mean_test_rank_ic"].iloc[0]) if len(base_row) else np.nan
    base_microcap = float(base_row["mean_top5_microcap_risk_share"].iloc[0]) if len(base_row) else np.nan
    shap_share = interaction_shap.groupby("variant")["selected_interaction_shap_share"].mean().to_dict()
    rows = []
    for row in comparison.itertuples(index=False):
        rows.append(
            {
                "variant": row.variant,
                "excess_vs_B_improved": bool(pd.notna(base_excess) and row.mean_excess_vs_csi1000 > base_excess),
                "rank_ic_vs_B_improved": bool(pd.notna(base_ic) and row.mean_test_rank_ic > base_ic),
                "microcap_not_worse": bool(pd.notna(base_microcap) and row.mean_top5_microcap_risk_share <= base_microcap),
                "selected_interaction_shap_share_mean": shap_share.get(row.variant, np.nan),
                "evidence": (
                    f"ann={row.mean_annualized_return:.2%}; excess={row.mean_excess_vs_csi1000:.2%}; "
                    f"rank_ic={row.mean_test_rank_ic:.4f}; microcap={row.mean_top5_microcap_risk_share:.2%}"
                ),
            }
        )
    return pd.DataFrame(rows)


def state_z_feature_name(axis: str) -> str:
    return f"state_{axis}__state_z"


def interaction_feature_name(factor: str, axis: str) -> str:
    return f"{factor}__x__state_{axis}"


def md_table(frame: pd.DataFrame, max_rows: int = 50) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(
    output: Path,
    condition_run: Path,
    selected: pd.DataFrame,
    comparison: pd.DataFrame,
    result: dict[str, pd.DataFrame],
    group_importance: pd.DataFrame,
    interaction_shap: pd.DataFrame,
    verdict: pd.DataFrame,
) -> None:
    interaction_groups = group_importance.loc[group_importance["feature_group"].str.startswith("selected_interaction::")].copy()
    lines = [
        "# Condition-Selected Regime-aware LightGBM",
        "",
        "## Scope",
        f"- Condition matrix source: `{condition_run}`",
        "- Same source walk-forward dataset, label, split, main-board permission pool, Top5, 10-day holding, 20bps cost.",
        "- Timing `target_position` overlay and CSI1000 benchmark are applied in portfolio tests.",
        "- `cluster_stock_state` is kept, but condition-selected dynamic interactions exclude it because its microcap exposure failed reliability checks.",
        "",
        "## Variants",
        *[f"- `{key}`: {value}" for key, value in VARIANTS.items()],
        "",
        "## Selected Interactions",
        md_table(
            selected[
                [
                    "factor",
                    "state_axis",
                    "state_bucket",
                    "rank_ic_mean",
                    "decile_spread_mean",
                    "top5_microcap_share",
                    "candidate_score",
                    "use_alpha",
                    "use_all",
                ]
            ],
            60,
        ),
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
                    "top_stock_buy_share",
                    "top_month_return_share",
                ]
            ].sort_values(["variant", "fold"]),
            80,
        ),
        "",
        "## Selected Interaction SHAP",
        md_table(interaction_shap, 80),
        "",
        "## Interaction Group Contribution",
        md_table(interaction_groups.sort_values(["variant", "fold", "shap_abs_share"], ascending=[True, True, False]), 120),
        "",
        "## Style Exposure",
        md_table(result["style_exposure"].sort_values(["variant", "fold"]), 80),
        "",
        "## Verdict",
        md_table(verdict, 20),
        "",
        "## Files",
        "- `selected_condition_interactions.csv`",
        "- `variant_comparison_summary.csv`",
        "- `model_train_valid_test_ic.csv`",
        "- `portfolio_metrics.csv`",
        "- `selected_interaction_shap_summary.csv`",
        "- `factor_group_contribution.csv`",
        "- `style_exposure.csv`",
        "- `final_verdict.csv`",
    ]
    (output / "condition_selected_regime_lgbm_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
