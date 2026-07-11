from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.backtest import BacktestEngine
from factor_forge.config import CostModel, ExecutionConstraints, load_project
from factor_forge.data import DataVersionRepository


DATA_VERSION = "data_v1_20260701T095408Z_c7b9995d"
PROJECT_CONFIG = "configs/project_sw_l2.yaml"
OUTPUT_ROOT = Path("artifacts/strategy_reviews")
START_DATE = "20220101"
TRAIN_START = "20230101"
TRAIN_END = "20241231"
VALID_START = "20250101"
VALID_END = "20251231"
TEST_START = "20260101"
TEST_END = "20260623"
HOLDING_DAYS = 10
COST_BPS = 20
TOP_N = [5, 10]

REGIME_COLS = [
    "market_ret_20",
    "market_ret_60",
    "market_vol_20",
    "market_breadth_20",
    "market_xsec_vol_20",
    "market_turnover_chg_5_20",
]
CLUSTER_COLS = [
    "cluster_sell_impact",
    "cluster_condition_deviation",
    "cluster_price_reversal",
    "cluster_liquidity",
    "cluster_stock_state",
    "cluster_industry_context",
    "cluster_market_context",
]
RAW_COLS = [
    "main_factor",
    "condition_deviation",
    "impact_efficiency",
    "sell_pressure_z",
    "relative_ret_1d_z",
    "ret_1d_z",
    "ret_5d_z",
    "ret_20d_z",
    "volatility_20_z",
    "turnover_rate_z",
    "amount_chg_5_20_z",
    "log_amount_20_z",
    "log_circ_mv_z",
    "stock_minus_industry_5d_z",
    "stock_minus_industry_20d_z",
    *REGIME_COLS,
]
VARIANTS = [
    "current_factor",
    "raw_lgbm_regressor",
    "cluster_lgbm_regressor",
    "regime_aware_cluster_lgbm",
    "recency_weighted_regime_cluster_lgbm",
    "cluster_lgbm_ranker",
    "regime_aware_cluster_ranker",
]


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_sorting_repair_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("loading panel")
    version, panel = load_panel()
    log(f"panel rows={len(panel):,} data_version={version}")
    dataset = build_dataset(panel, log)
    dataset.to_parquet(output / "sorting_repair_dataset.parquet", index=False)
    log(f"dataset rows={len(dataset):,} dates={dataset['trade_date'].nunique():,} stocks={dataset['ts_code'].nunique():,}")

    metrics_rows: list[dict] = []
    yearly_rows: list[dict] = []
    decile_rows: list[dict] = []
    gain_rows: list[pd.DataFrame] = []
    shap_rows: list[pd.DataFrame] = []
    backtest_rows: list[dict] = []

    predictions = []
    for variant in VARIANTS:
        log(f"running variant={variant}")
        pred, gain, shap = fit_predict_variant(dataset, variant, log)
        pred.to_parquet(output / f"predictions_{variant}.parquet", index=False)
        predictions.append(pred.assign(variant=variant))
        if not gain.empty:
            gain["variant"] = variant
            gain_rows.append(gain)
        if not shap.empty:
            shap["variant"] = variant
            shap_rows.append(shap)

        metrics_rows.extend(evaluate_rank_metrics(pred, variant))
        yearly_rows.extend(evaluate_yearly_metrics(pred, variant))
        decile_rows.extend(evaluate_deciles(pred, variant))
        backtest_rows.extend(run_backtests(panel, dataset, pred, variant, log))

    log("running score-band postprocess variants")
    for variant, pred in build_score_band_variants(predictions):
        pred.to_parquet(output / f"predictions_{variant}.parquet", index=False)
        predictions.append(pred.assign(variant=variant))
        metrics_rows.extend(evaluate_rank_metrics(pred, variant))
        yearly_rows.extend(evaluate_yearly_metrics(pred, variant))
        decile_rows.extend(evaluate_deciles(pred, variant))
        backtest_rows.extend(run_backtests(panel, dataset, pred, variant, log))

    metrics = pd.DataFrame(metrics_rows)
    yearly = pd.DataFrame(yearly_rows)
    deciles = pd.DataFrame(decile_rows)
    backtests = pd.DataFrame(backtest_rows)
    gains = pd.concat(gain_rows, ignore_index=True) if gain_rows else pd.DataFrame()
    shap_values = pd.concat(shap_rows, ignore_index=True) if shap_rows else pd.DataFrame()
    all_predictions = pd.concat(predictions, ignore_index=True)

    metrics.to_csv(output / "rank_metrics.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_rank_metrics.csv", index=False, encoding="utf-8-sig")
    deciles.to_csv(output / "decile_spreads.csv", index=False, encoding="utf-8-sig")
    backtests.to_csv(output / "backtest_metrics.csv", index=False, encoding="utf-8-sig")
    gains.to_csv(output / "feature_importance_gain.csv", index=False, encoding="utf-8-sig")
    shap_values.to_csv(output / "shap_like_contribution.csv", index=False, encoding="utf-8-sig")
    all_predictions.to_parquet(output / "all_predictions.parquet", index=False)
    write_report(output, metrics, yearly, deciles, backtests, gains, shap_values)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "data_version": version,
                "splits": {
                    "train": [TRAIN_START, TRAIN_END],
                    "valid": [VALID_START, VALID_END],
                    "test": [TEST_START, TEST_END],
                },
                "variants": VARIANTS,
                "cluster_cols": CLUSTER_COLS,
                "regime_cols": REGIME_COLS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def load_panel() -> tuple[str, pd.DataFrame]:
    project = load_project(PROJECT_CONFIG)
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, panel = repo.load_panel(DATA_VERSION)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel.loc[panel["trade_date"].ge(pd.Timestamp(START_DATE))].copy()
    return version, panel


def build_dataset(panel: pd.DataFrame, log, include_unlabeled_date: pd.Timestamp | None = None) -> pd.DataFrame:
    d = panel.sort_values(["ts_code", "trade_date"]).copy()
    industry_col = "industry_l2_code" if "industry_l2_code" in d.columns else "industry_l1_code"
    d["ret_1d"] = d["adj_close"].groupby(d["ts_code"]).pct_change(fill_method=None)
    d["ret_5d"] = d["adj_close"].groupby(d["ts_code"]).pct_change(5, fill_method=None)
    d["ret_20d"] = d["adj_close"].groupby(d["ts_code"]).pct_change(20, fill_method=None)
    d["volatility_20"] = d.groupby("ts_code")["ret_1d"].transform(lambda s: s.rolling(20, min_periods=10).std())
    d["log_amount_20"] = np.log1p(
        d.groupby("ts_code")["amount_cny"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    )
    d["amount_5"] = d.groupby("ts_code")["amount_cny"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    d["amount_20"] = d.groupby("ts_code")["amount_cny"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    d["amount_chg_5_20"] = d["amount_5"] / d["amount_20"] - 1.0
    d["log_circ_mv"] = np.log1p(d["circ_mv_cny"])

    log("building sell-impact factors")
    d["industry_ret_1d"] = d.groupby(["trade_date", industry_col])["ret_1d"].transform("mean")
    d["relative_ret_1d"] = d["ret_1d"] - d["industry_ret_1d"]
    d["turnover_rate_z"] = cs_zscore(d, "turnover_rate")
    d["sell_pressure"] = np.where(d["ret_1d"].lt(0), d["turnover_rate_z"] * d["ret_1d"].abs(), 0.0)
    d["sell_pressure_z"] = cs_zscore(d, "sell_pressure")
    d["relative_ret_1d_z"] = cs_zscore(d, "relative_ret_1d")
    d["impact_efficiency"] = d["sell_pressure_z"] - d["relative_ret_1d_z"]
    d["main_factor"] = cs_zscore(d, "impact_efficiency")
    hist = d.groupby("ts_code")["impact_efficiency"].shift(1)
    normal_mean = hist.groupby(d["ts_code"]).transform(lambda s: s.rolling(60, min_periods=30).mean())
    normal_std = hist.groupby(d["ts_code"]).transform(lambda s: s.rolling(60, min_periods=30).std())
    d["condition_deviation"] = ((d["impact_efficiency"] - normal_mean) / normal_std).abs()
    d["condition_deviation"] = d["condition_deviation"].replace([np.inf, -np.inf], np.nan)
    d["condition_deviation_z"] = cs_zscore(d, "condition_deviation")
    d["condition_quantile"] = daily_quantile(d, "condition_deviation", 5)

    log("building labels and context features")
    d["entry_open"] = d.groupby("ts_code")["adj_open"].shift(-1)
    d["exit_open"] = d.groupby("ts_code")["adj_open"].shift(-(HOLDING_DAYS + 1))
    d["exit_date"] = d.groupby("ts_code")["trade_date"].shift(-(HOLDING_DAYS + 1))
    d["label"] = d["exit_open"] / d["entry_open"] - 1.0
    d["industry_ret_5d"] = d.groupby(["trade_date", industry_col])["ret_5d"].transform("mean")
    d["industry_ret_20d"] = d.groupby(["trade_date", industry_col])["ret_20d"].transform("mean")
    d["stock_minus_industry_5d"] = d["ret_5d"] - d["industry_ret_5d"]
    d["stock_minus_industry_20d"] = d["ret_20d"] - d["industry_ret_20d"]
    for col in [
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "volatility_20",
        "amount_chg_5_20",
        "log_amount_20",
        "log_circ_mv",
        "stock_minus_industry_5d",
        "stock_minus_industry_20d",
    ]:
        d[f"{col}_z"] = cs_zscore(d, col)

    market = build_market_context(d)
    # Mapping the small daily context back avoids pandas materializing a second
    # full-width copy of the multi-million-row panel during merge.
    market = market.set_index("trade_date")
    for col in REGIME_COLS:
        d[col] = d["trade_date"].map(market[col])
    d = add_clusters(d)
    d = add_regime_interactions(d)
    d["eligible"] = (
        d["is_liquid"].fillna(False).astype(bool)
        & d["is_tradeable"].fillna(False).astype(bool)
        & ~d["is_suspended"].fillna(False).astype(bool)
        & ~d["is_st"].fillna(False).astype(bool)
        & ~d["is_delisting_period"].fillna(False).astype(bool)
        & d["listing_trade_days"].ge(60)
        & d["condition_quantile"].eq(5)
    )
    needed = ["trade_date", "ts_code", "label", "exit_date", "eligible", "main_factor", *RAW_COLS, *CLUSTER_COLS]
    interaction_cols = [c for c in d.columns if "__x__regime_" in c]
    keep = [c for c in dict.fromkeys([*needed, *interaction_cols, "condition_quantile"])]
    out = d.loc[d["trade_date"].between(pd.Timestamp(TRAIN_START), pd.Timestamp(TEST_END)), keep].copy()
    out = out.loc[out["eligible"]].replace([np.inf, -np.inf], np.nan).dropna(subset=["main_factor"])
    if include_unlabeled_date is None:
        out = out.dropna(subset=["label"])
    else:
        include_unlabeled_date = pd.Timestamp(include_unlabeled_date).normalize()
        out = out.loc[out["label"].notna() | out["trade_date"].eq(include_unlabeled_date)]
    out["year"] = out["trade_date"].dt.year
    return out.reset_index(drop=True)


def build_market_context(d: pd.DataFrame) -> pd.DataFrame:
    daily = (
        d.loc[d["is_liquid"].fillna(False).astype(bool)]
        .groupby("trade_date")
        .agg(
            market_ret_1d=("ret_1d", "mean"),
            market_breadth_1d=("ret_1d", lambda s: float((s > 0).mean())),
            market_xsec_vol_1d=("ret_1d", "std"),
            market_amount=("amount_cny", "sum"),
        )
        .sort_index()
    )
    daily["market_ret_20"] = daily["market_ret_1d"].rolling(20, min_periods=10).sum()
    daily["market_ret_60"] = daily["market_ret_1d"].rolling(60, min_periods=30).sum()
    daily["market_vol_20"] = daily["market_ret_1d"].rolling(20, min_periods=10).std()
    daily["market_breadth_20"] = daily["market_breadth_1d"].rolling(20, min_periods=10).mean()
    daily["market_xsec_vol_20"] = daily["market_xsec_vol_1d"].rolling(20, min_periods=10).mean()
    daily["market_amount_5"] = daily["market_amount"].rolling(5, min_periods=3).mean()
    daily["market_amount_20"] = daily["market_amount"].rolling(20, min_periods=10).mean()
    daily["market_turnover_chg_5_20"] = daily["market_amount_5"] / daily["market_amount_20"] - 1.0
    return daily[REGIME_COLS].reset_index()


def add_clusters(d: pd.DataFrame) -> pd.DataFrame:
    d["cluster_sell_impact"] = mean_cols(d, ["main_factor", "impact_efficiency", "sell_pressure_z"])
    d["cluster_condition_deviation"] = mean_cols(d, ["condition_deviation_z", "main_factor"])
    d["cluster_price_reversal"] = mean_cols(d, [-d["ret_1d_z"], -d["ret_5d_z"], -d["ret_20d_z"]])
    d["cluster_liquidity"] = mean_cols(d, ["log_amount_20_z", -d["amount_chg_5_20_z"], "turnover_rate_z"])
    d["cluster_stock_state"] = mean_cols(d, [-d["volatility_20_z"], -d["log_circ_mv_z"]])
    d["cluster_industry_context"] = mean_cols(d, ["stock_minus_industry_5d_z", "stock_minus_industry_20d_z"])
    d["cluster_market_context"] = mean_cols(
        d,
        ["market_ret_20", "market_ret_60", -d["market_vol_20"], "market_breadth_20"],
    )
    return d


def add_regime_interactions(d: pd.DataFrame) -> pd.DataFrame:
    for cluster in CLUSTER_COLS:
        for regime in REGIME_COLS:
            d[f"{cluster}__x__regime_{regime}"] = d[cluster] * d[regime]
    return d


def fit_predict_variant(dataset: pd.DataFrame, variant: str, log) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if variant == "current_factor":
        pred = dataset.loc[dataset["trade_date"].ge(pd.Timestamp(VALID_START)), ["trade_date", "ts_code", "label"]].copy()
        pred["score"] = dataset.loc[pred.index, "main_factor"].to_numpy()
        pred["sample"] = np.where(pred["trade_date"].le(pd.Timestamp(VALID_END)), "valid", "test")
        return pred, pd.DataFrame(), pd.DataFrame()

    import lightgbm as lgb

    features = features_for_variant(variant, dataset)
    train = sample_slice(dataset, TRAIN_START, TRAIN_END, features)
    valid = sample_slice(dataset, VALID_START, VALID_END, features)
    test = sample_slice(dataset, TEST_START, TEST_END, features)
    log(f"{variant} features={len(features)} train={len(train):,} valid={len(valid):,} test={len(test):,}")

    if "ranker" in variant:
        train_sorted = train.sort_values(["trade_date", "ts_code"])
        valid_sorted = valid.sort_values(["trade_date", "ts_code"])
        train_y = relevance_labels(train_sorted)
        valid_y = relevance_labels(valid_sorted)
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
            train_sorted[features],
            train_y,
            group=train_sorted.groupby("trade_date").size().to_list(),
            eval_set=[(valid_sorted[features], valid_y)],
            eval_group=[valid_sorted.groupby("trade_date").size().to_list()],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
    else:
        weights = pd.Series(1.0, index=train.index)
        if variant == "recency_weighted_regime_cluster_lgbm":
            weights.loc[train["trade_date"].ge(pd.Timestamp("2024-01-01"))] = 2.0
            weights.loc[train["trade_date"].ge(pd.Timestamp("2024-07-01"))] = 3.0
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
            sample_weight=weights,
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
            "mean_abs_contribution_2026": np.abs(contrib[:, :-1]).mean(axis=0),
            "mean_contribution_2026": contrib[:, :-1].mean(axis=0),
        }
    ).sort_values("mean_abs_contribution_2026", ascending=False)
    return pred, gain, shap


def run_backtests(panel: pd.DataFrame, dataset: pd.DataFrame, pred: pd.DataFrame, variant: str, log) -> list[dict]:
    rows = []
    membership = dataset[["trade_date", "ts_code", "condition_quantile"]].copy()
    membership["selection_eligible"] = True
    factor_values = pred.rename(columns={"score": "factor_value"})[["trade_date", "ts_code", "factor_value"]]
    bt_panel = panel.loc[panel["trade_date"].between(pd.Timestamp(VALID_START), pd.Timestamp(TEST_END))].copy()
    engine = BacktestEngine()
    for sample, start, end in [("valid", VALID_START, VALID_END), ("test", TEST_START, TEST_END)]:
        panel_slice = bt_panel.loc[bt_panel["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        factor_slice = factor_values.loc[factor_values["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        member_slice = membership.loc[membership["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        if factor_slice.empty:
            continue
        for top_n in TOP_N:
            result = engine.run(
                panel_slice,
                factor_slice,
                universe="liquid",
                top_n=top_n,
                holding_days=HOLDING_DAYS,
                initial_cash=1_000_000,
                lot_size=100,
                constraints=ExecutionConstraints(
                    exclude_suspended=True,
                    cannot_buy_limit_up=True,
                    cannot_sell_limit_down=True,
                    exclude_st=True,
                    exclude_delisting_period=True,
                    min_listing_days=60,
                ),
                cost_model=CostModel(
                    commission_bps_per_side=3,
                    slippage_bps_per_side=5,
                    stamp_duty_bps_sell=5,
                ),
                cost_scenario_bps=COST_BPS,
                selection_membership=member_slice,
            )
            row = {
                "variant": variant,
                "sample": sample,
                "top_n": top_n,
                "holding_days": HOLDING_DAYS,
                "cost_bps": COST_BPS,
                **result.metrics,
            }
            rows.append(row)
            log(
                f"{variant} {sample} top{top_n} ann={result.metrics['annualized_return']:.2%} "
                f"excess={result.metrics['annualized_excess_return']:.2%} mdd={result.metrics['max_drawdown']:.2%}"
            )
    return rows


def build_score_band_variants(predictions: list[pd.DataFrame]) -> list[tuple[str, pd.DataFrame]]:
    """Avoid the overfit extreme tail by selecting a high score percentile band."""
    variants = []
    source = pd.concat(predictions, ignore_index=True)
    for base in ["regime_aware_cluster_lgbm", "regime_aware_cluster_ranker", "cluster_lgbm_ranker"]:
        p = source.loc[source["variant"].eq(base)].copy()
        if p.empty:
            continue
        p["score_pct"] = p.groupby("trade_date")["score"].rank(pct=True, method="first")
        for target in [0.88, 0.92, 0.95]:
            q = p[["trade_date", "ts_code", "label", "sample", "score_pct"]].copy()
            q["score"] = -(q["score_pct"] - target).abs()
            q = q.drop(columns=["score_pct"])
            variants.append((f"{base}_score_band_{target:.2f}", q))
    return variants


def evaluate_rank_metrics(pred: pd.DataFrame, variant: str) -> list[dict]:
    rows = []
    for sample, frame in pred.groupby("sample"):
        daily_ic = frame.groupby("trade_date").apply(daily_spearman, include_groups=False).dropna()
        rows.append(
            {
                "variant": variant,
                "sample": sample,
                "days": int(len(daily_ic)),
                "rank_ic": float(daily_ic.mean()) if len(daily_ic) else np.nan,
                "icir": float(daily_ic.mean() / daily_ic.std() * np.sqrt(252)) if len(daily_ic) > 1 and daily_ic.std() > 0 else np.nan,
                "positive_ratio": float((daily_ic > 0).mean()) if len(daily_ic) else np.nan,
                **topk_stats(frame, 5),
                **{f"top10_{k}": v for k, v in topk_stats(frame, 10).items()},
            }
        )
    return rows


def evaluate_yearly_metrics(pred: pd.DataFrame, variant: str) -> list[dict]:
    rows = []
    p = pred.copy()
    p["year"] = p["trade_date"].dt.year
    for year, frame in p.groupby("year"):
        daily_ic = frame.groupby("trade_date").apply(daily_spearman, include_groups=False).dropna()
        rows.append(
            {
                "variant": variant,
                "year": int(year),
                "days": int(len(daily_ic)),
                "rank_ic": float(daily_ic.mean()) if len(daily_ic) else np.nan,
                "positive_ratio": float((daily_ic > 0).mean()) if len(daily_ic) else np.nan,
                **topk_stats(frame, 5),
                **{f"top10_{k}": v for k, v in topk_stats(frame, 10).items()},
            }
        )
    return rows


def evaluate_deciles(pred: pd.DataFrame, variant: str) -> list[dict]:
    rows = []
    for sample, frame in pred.groupby("sample"):
        decile_values = []
        for date, g in frame.groupby("trade_date"):
            if len(g) < 100 or g["score"].nunique() < 10:
                continue
            q = pd.qcut(g["score"].rank(method="first"), 10, labels=False) + 1
            h = g.assign(decile=q)
            dec = h.groupby("decile")["label"].mean()
            if 1 in dec.index and 10 in dec.index:
                decile_values.append({"spread_10_1": dec.loc[10] - dec.loc[1], **{f"q{int(k)}": v for k, v in dec.items()}})
        if decile_values:
            row = pd.DataFrame(decile_values).mean(numeric_only=True).to_dict()
            row.update({"variant": variant, "sample": sample, "days": len(decile_values)})
            rows.append(row)
    return rows


def topk_stats(frame: pd.DataFrame, k: int) -> dict:
    rows = []
    for _, g in frame.groupby("trade_date"):
        if len(g) < max(50, k * 5):
            continue
        top = g.nlargest(k, "score")
        q80 = g["label"].quantile(0.8)
        q20 = g["label"].quantile(0.2)
        rows.append(
            {
                f"top{k}_mean_label": top["label"].mean(),
                f"top{k}_hit_top20": float(top["label"].ge(q80).mean()),
                f"top{k}_bad_bottom20": float(top["label"].le(q20).mean()),
            }
        )
    if not rows:
        return {f"top{k}_mean_label": np.nan, f"top{k}_hit_top20": np.nan, f"top{k}_bad_bottom20": np.nan}
    return pd.DataFrame(rows).mean(numeric_only=True).to_dict()


def features_for_variant(variant: str, dataset: pd.DataFrame) -> list[str]:
    if variant == "raw_lgbm_regressor":
        return [c for c in RAW_COLS if c in dataset.columns]
    if variant in {"cluster_lgbm_regressor", "cluster_lgbm_ranker"}:
        return [c for c in CLUSTER_COLS if c in dataset.columns]
    if variant in {
        "regime_aware_cluster_lgbm",
        "recency_weighted_regime_cluster_lgbm",
        "regime_aware_cluster_ranker",
    }:
        interaction_cols = [c for c in dataset.columns if "__x__regime_" in c]
        return [*CLUSTER_COLS, *REGIME_COLS, *interaction_cols]
    raise ValueError(f"unknown variant: {variant}")


def sample_slice(dataset: pd.DataFrame, start: str, end: str, features: list[str]) -> pd.DataFrame:
    cols = ["trade_date", "ts_code", "label", *features]
    return dataset.loc[dataset["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end)), cols].dropna(
        subset=["label", *features]
    )


def relevance_labels(frame: pd.DataFrame) -> pd.Series:
    def local(g: pd.DataFrame) -> pd.Series:
        if g["label"].nunique() < 5:
            return pd.Series(0, index=g.index)
        return pd.qcut(g["label"].rank(method="first"), 5, labels=False).astype(int)

    return frame.groupby("trade_date", group_keys=False).apply(local)


def daily_spearman(g: pd.DataFrame) -> float:
    if len(g) < 30 or g["score"].nunique() < 2 or g["label"].nunique() < 2:
        return np.nan
    return g["score"].corr(g["label"], method="spearman")


def cs_zscore(d: pd.DataFrame, col: str) -> pd.Series:
    group = d.groupby("trade_date")[col]
    mean = group.transform("mean")
    std = group.transform("std")
    return ((d[col] - mean) / std).replace([np.inf, -np.inf], np.nan).clip(-5, 5)


def daily_quantile(d: pd.DataFrame, col: str, q: int) -> pd.Series:
    def local(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) < q * 20 or valid.nunique() < q:
            return pd.Series(np.nan, index=s.index)
        return pd.qcut(s.rank(method="first"), q, labels=False) + 1

    return d.groupby("trade_date")[col].transform(local)


def mean_cols(d: pd.DataFrame, cols: list[str | pd.Series]) -> pd.Series:
    values = []
    for col in cols:
        values.append(d[col] if isinstance(col, str) else col)
    return pd.concat(values, axis=1).mean(axis=1)


def write_report(
    output: Path,
    metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    deciles: pd.DataFrame,
    backtests: pd.DataFrame,
    gains: pd.DataFrame,
    shap_values: pd.DataFrame,
) -> None:
    test = metrics.loc[metrics["sample"].eq("test")].sort_values("rank_ic", ascending=False)
    valid = metrics.loc[metrics["sample"].eq("valid")].sort_values("rank_ic", ascending=False)
    bt_test = backtests.loc[backtests["sample"].eq("test")].sort_values("annualized_return", ascending=False)
    gain_summary = cluster_gain_summary(gains)
    shap_summary = cluster_shap_summary(shap_values)
    lines = [
        "# Sell Impact Sorting Repair Report",
        "",
        "## Setup",
        f"- Train: `{TRAIN_START}` ~ `{TRAIN_END}`",
        f"- Valid: `{VALID_START}` ~ `{VALID_END}`",
        f"- Test: `{TEST_START}` ~ `{TEST_END}`",
        "- Universe: liquid + tradeable + condition Q5 candidate pool",
        "- Label: T close signal, T+1 open entry, 10 trading-day open-to-open forward return",
        "",
        "## Test Rank Metrics",
        test.round(6).to_markdown(index=False),
        "",
        "## Valid Rank Metrics",
        valid.round(6).to_markdown(index=False),
        "",
        "## Backtest Metrics",
        bt_test.round(6).to_markdown(index=False),
        "",
        "## Yearly Rank Metrics",
        yearly.round(6).to_markdown(index=False),
        "",
        "## Decile Spread",
        deciles.round(6).to_markdown(index=False),
        "",
        "## Feature/Cluster Gain",
        gain_summary.round(6).to_markdown(index=False) if not gain_summary.empty else "N/A",
        "",
        "## 2026 SHAP-like Cluster Contribution",
        shap_summary.round(8).to_markdown(index=False) if not shap_summary.empty else "N/A",
        "",
        "## Reading",
        "- Prefer variants that improve 2026 Test RankIC and Top5/Top10 backtest together.",
        "- If LightGBM valid is strong but test turns negative, treat it as factor/regime overfit rather than a deployable improvement.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cluster_gain_summary(gains: pd.DataFrame) -> pd.DataFrame:
    if gains.empty:
        return pd.DataFrame()
    g = gains.copy()
    g["cluster"] = g["feature"].map(feature_cluster)
    return (
        g.groupby(["variant", "cluster"], dropna=False)[["gain_importance", "split_importance"]]
        .sum()
        .reset_index()
        .sort_values(["variant", "gain_importance"], ascending=[True, False])
    )


def cluster_shap_summary(shap_values: pd.DataFrame) -> pd.DataFrame:
    if shap_values.empty:
        return pd.DataFrame()
    s = shap_values.copy()
    s["cluster"] = s["feature"].map(feature_cluster)
    return (
        s.groupby(["variant", "cluster"], dropna=False)[["mean_abs_contribution_2026", "mean_contribution_2026"]]
        .sum()
        .reset_index()
        .sort_values(["variant", "mean_abs_contribution_2026"], ascending=[True, False])
    )


def feature_cluster(feature: str) -> str:
    for cluster in CLUSTER_COLS:
        if feature == cluster or feature.startswith(f"{cluster}__"):
            return cluster
    if feature in REGIME_COLS:
        return "regime"
    if "ret" in feature or "impact" in feature or "pressure" in feature:
        return "raw_sell_impact_price"
    if "amount" in feature or "turnover" in feature:
        return "raw_liquidity"
    if "vol" in feature or "mv" in feature:
        return "raw_stock_state"
    return "other"


if __name__ == "__main__":
    main()
