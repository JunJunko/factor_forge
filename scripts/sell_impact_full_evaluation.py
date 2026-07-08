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
TIMING_COMPARE_RUN = Path("artifacts/strategy_reviews/sell_impact_ranker_timing_compare_20260708T120449Z")
TRADE_AUDIT_RUN = Path("artifacts/strategy_reviews/sell_impact_ranker_trade_audit_20260708T121703Z")
OUTPUT_ROOT = Path("artifacts/strategy_reviews")

MODEL_VARIANT = "regime_aware_cluster_ranker"
PRODUCTION_SELECTION = "ranker_direct_top"
PRODUCTION_TIMING = "timing_target_position"
PRODUCTION_TOP_N = 5
HOLDING_DAYS = 10
INITIAL_CASH = 1_000_000
LOT_SIZE = 100
BENCHMARK_NAME = "CSI1000"
COST_GRID = [0, 10, 20, 30, 50]
TOP_N_GRID = [5, 10]


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_full_evaluation_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("loading walk-forward dataset and existing strategy artifacts")
    dataset = pd.read_parquet(SOURCE_RUN / "walkforward_dataset.parquet")
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    version, panel = base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    market_benchmark = timing_compare.load_market_benchmark(version)
    timing_multiplier = timing_compare.load_position_multiplier(timing_compare.TIMING_DAILY)

    log("factor layer")
    factor_outputs = build_factor_layer(dataset, output)

    log("model layer with refit for train/valid/test IC")
    model_outputs = build_model_layer(dataset, output, log)

    log("portfolio and stability layer")
    portfolio_outputs = build_portfolio_layer(output)

    log("cost sensitivity")
    cost_outputs = build_cost_sensitivity(panel, dataset, market_benchmark, timing_multiplier, output, log)

    log("live trading layer: audit, capacity, slippage")
    live_outputs = build_live_layer(panel, output)

    write_master_report(
        output,
        data_version=version,
        factor_outputs=factor_outputs,
        model_outputs=model_outputs,
        portfolio_outputs=portfolio_outputs,
        cost_outputs=cost_outputs,
        live_outputs=live_outputs,
    )
    summary = {
        "run_dir": str(output),
        "source_run": str(SOURCE_RUN),
        "timing_compare_run": str(TIMING_COMPARE_RUN),
        "trade_audit_run": str(TRADE_AUDIT_RUN),
        "data_version": version,
        "production_strategy": {
            "model_variant": MODEL_VARIANT,
            "selection": PRODUCTION_SELECTION,
            "timing": PRODUCTION_TIMING,
            "top_n": PRODUCTION_TOP_N,
            "holding_days": HOLDING_DAYS,
            "benchmark": BENCHMARK_NAME,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"done -> {output}")


def build_factor_layer(dataset: pd.DataFrame, output: Path) -> dict[str, Any]:
    factor_dir = output / "factor_layer"
    factor_dir.mkdir()
    factor_cols = [
        "main_factor",
        "impact_efficiency",
        "condition_deviation",
        *base.CLUSTER_COLS,
        *base.REGIME_COLS,
    ]
    factor_cols = [col for col in dict.fromkeys(factor_cols) if col in dataset.columns]
    start = pd.Timestamp("2024-01-01")
    end = dataset["trade_date"].max()
    sample = dataset.loc[dataset["trade_date"].between(start, end)].copy()

    daily_ic_rows = []
    summary_rows = []
    yearly_rows = []
    decile_rows = []
    for factor in factor_cols:
        daily = daily_ic(sample, factor, "label")
        daily["factor"] = factor
        daily_ic_rows.append(daily)
        summary_rows.append(ic_summary_row(daily["rank_ic"], factor, "2024_2026"))
        if not daily.empty:
            tmp = daily.copy()
            tmp["trade_date"] = pd.to_datetime(tmp["trade_date"])
            tmp["year"] = tmp["trade_date"].dt.year
            for year, y in tmp.groupby("year"):
                yearly_rows.append(ic_summary_row(y["rank_ic"], factor, int(year)))
        dec = decile_return(sample, factor, "label", bins=10)
        dec["factor"] = factor
        decile_rows.append(dec)

    daily_ic_df = pd.concat(daily_ic_rows, ignore_index=True) if daily_ic_rows else pd.DataFrame()
    ic_report = pd.DataFrame(summary_rows).sort_values("rank_ic_mean", ascending=False)
    yearly_ic = pd.DataFrame(yearly_rows)
    deciles = pd.concat(decile_rows, ignore_index=True) if decile_rows else pd.DataFrame()
    corr = factor_corr(sample, factor_cols)

    daily_ic_df.to_csv(factor_dir / "daily_ic.csv", index=False, encoding="utf-8-sig")
    ic_report.to_csv(factor_dir / "ic_report.csv", index=False, encoding="utf-8-sig")
    yearly_ic.to_csv(factor_dir / "yearly_ic.csv", index=False, encoding="utf-8-sig")
    deciles.to_csv(factor_dir / "decile_returns.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(factor_dir / "correlation_matrix.csv", encoding="utf-8-sig")
    plot_deciles(deciles, factor_dir / "decile_returns.png")
    plot_corr(corr, factor_dir / "correlation_matrix.png")
    write_factor_report(factor_dir, ic_report, yearly_ic, deciles, corr)
    return {
        "ic_report": ic_report,
        "yearly_ic": yearly_ic,
        "deciles": deciles,
        "corr": corr,
        "path": factor_dir,
    }


def build_model_layer(dataset: pd.DataFrame, output: Path, log) -> dict[str, Any]:
    model_dir = output / "model_layer"
    model_dir.mkdir()
    import lightgbm as lgb

    features = base.features_for_variant(MODEL_VARIANT, dataset)
    ic_rows = []
    gain_frames = []
    shap_frames = []
    training_rows = []
    for fold in wf.FOLDS:
        fold_name = fold["fold"]
        train = base.sample_slice(dataset, fold["train_start"], fold["train_end"], features).sort_values(["trade_date", "ts_code"])
        valid = base.sample_slice(dataset, fold["valid_start"], fold["valid_end"], features).sort_values(["trade_date", "ts_code"])
        test = base.sample_slice(dataset, fold["test_start"], fold["test_end"], features).sort_values(["trade_date", "ts_code"])
        log(f"fit {fold_name}: train={len(train):,} valid={len(valid):,} test={len(test):,} features={len(features)}")

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
        training_rows.append(
            {
                "fold": fold_name,
                "features": len(features),
                "train_rows": len(train),
                "valid_rows": len(valid),
                "test_rows": len(test),
                "best_iteration": int(model.best_iteration_ or model.n_estimators),
            }
        )
        for sample_name, frame in [("train", train), ("valid", valid), ("test", test)]:
            pred = frame[["trade_date", "ts_code", "label"]].copy()
            pred["score"] = model.predict(frame[features])
            ic = daily_ic(pred, "score", "label")
            ic_rows.append(ic_summary_row(ic["rank_ic"], fold_name, sample_name))

        gain = pd.DataFrame(
            {
                "feature": features,
                "gain_importance": model.booster_.feature_importance(importance_type="gain"),
                "split_importance": model.booster_.feature_importance(importance_type="split"),
                "fold": fold_name,
            }
        ).sort_values("gain_importance", ascending=False)
        gain_frames.append(gain)
        contrib = model.booster_.predict(test[features], pred_contrib=True)
        shap = pd.DataFrame(
            {
                "feature": features,
                "mean_abs_shap": np.abs(contrib[:, :-1]).mean(axis=0),
                "mean_shap": contrib[:, :-1].mean(axis=0),
                "fold": fold_name,
            }
        ).sort_values("mean_abs_shap", ascending=False)
        shap_frames.append(shap)

    model_ic = pd.DataFrame(ic_rows).rename(columns={"factor": "fold", "window": "sample"})
    gains = pd.concat(gain_frames, ignore_index=True)
    shap_values = pd.concat(shap_frames, ignore_index=True)
    training = pd.DataFrame(training_rows)
    gain_summary = aggregate_feature_table(gains, "gain_importance", "split_importance")
    shap_summary = aggregate_feature_table(shap_values, "mean_abs_shap", "mean_shap")

    training.to_csv(model_dir / "lightgbm_training_result.csv", index=False, encoding="utf-8-sig")
    model_ic.to_csv(model_dir / "train_valid_test_ic.csv", index=False, encoding="utf-8-sig")
    gains.to_csv(model_dir / "feature_importance_gain_by_fold.csv", index=False, encoding="utf-8-sig")
    gain_summary.to_csv(model_dir / "feature_importance_gain.csv", index=False, encoding="utf-8-sig")
    shap_values.to_csv(model_dir / "shap_by_fold.csv", index=False, encoding="utf-8-sig")
    shap_summary.to_csv(model_dir / "shap_summary.csv", index=False, encoding="utf-8-sig")
    write_model_report(model_dir, training, model_ic, gain_summary, shap_summary)
    return {"training": training, "ic": model_ic, "gain": gain_summary, "shap": shap_summary, "path": model_dir}


def build_portfolio_layer(output: Path) -> dict[str, Any]:
    portfolio_dir = output / "portfolio_layer"
    portfolio_dir.mkdir()
    metrics = pd.read_csv(TIMING_COMPARE_RUN / "ranker_timing_backtest_metrics.csv")
    daily = pd.read_parquet(TIMING_COMPARE_RUN / "ranker_timing_daily.parquet")
    selected = metrics.loc[
        metrics["selection"].eq(PRODUCTION_SELECTION)
        & metrics["timing"].eq(PRODUCTION_TIMING)
        & metrics["top_n"].eq(PRODUCTION_TOP_N)
    ].copy()
    turnover = annualized_turnover(daily)
    selected = selected.merge(turnover, on=["fold", "selection", "timing", "top_n"], how="left")
    selected.to_csv(portfolio_dir / "portfolio_metrics.csv", index=False, encoding="utf-8-sig")
    selected[
        [
            "fold",
            "annualized_return",
            "annualized_excess_return_vs_csi1000",
            "sharpe",
            "calmar",
            "max_drawdown",
            "annualized_turnover",
            "execution_rate",
            "top_stock_buy_share",
            "top_month_return_share",
        ]
    ].to_csv(portfolio_dir / "yearly_return_split.csv", index=False, encoding="utf-8-sig")
    sensitivity = metrics.loc[
        metrics["timing"].isin(["no_timing", PRODUCTION_TIMING])
        & metrics["top_n"].isin(TOP_N_GRID)
    ].copy()
    sensitivity.to_csv(portfolio_dir / "parameter_sensitivity.csv", index=False, encoding="utf-8-sig")
    write_portfolio_report(portfolio_dir, selected, sensitivity)
    return {"metrics": selected, "sensitivity": sensitivity, "path": portfolio_dir}


def build_cost_sensitivity(
    panel: pd.DataFrame,
    dataset: pd.DataFrame,
    market_benchmark: pd.DataFrame,
    timing_multiplier: pd.Series,
    output: Path,
    log,
) -> dict[str, Any]:
    stability_dir = output / "stability_layer"
    stability_dir.mkdir(exist_ok=True)
    rows = []
    daily_frames = []
    trade_frames = []
    for fold in wf.FOLDS:
        pred = pd.read_parquet(SOURCE_RUN / f"predictions_{fold['fold']}_{MODEL_VARIANT}.parquet")
        pred["trade_date"] = pd.to_datetime(pred["trade_date"])
        factor_values = pred.loc[pred["sample"].eq("test"), ["trade_date", "ts_code", "score"]].rename(columns={"score": "factor_value"})
        member = dataset.loc[
            dataset["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"])),
            ["trade_date", "ts_code", "condition_quantile"],
        ].copy()
        member["selection_eligible"] = True
        panel_slice = panel.loc[panel["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))].copy()
        for cost in COST_GRID:
            result = BacktestEngine().run(
                panel_slice,
                factor_values,
                universe="liquid",
                top_n=PRODUCTION_TOP_N,
                holding_days=HOLDING_DAYS,
                initial_cash=INITIAL_CASH,
                lot_size=LOT_SIZE,
                constraints=production_constraints(),
                cost_model=CostModel(commission_bps_per_side=0, slippage_bps_per_side=0, stamp_duty_bps_sell=0),
                cost_scenario_bps=cost,
                selection_membership=member,
                position_multiplier=timing_multiplier,
                market_benchmark=market_benchmark,
            )
            row = {
                "fold": fold["fold"],
                "cost_bps": cost,
                **result.metrics,
                "csi1000_annualized_return": result.metrics.get("market_index_annualized_return"),
                "annualized_excess_return_vs_csi1000": (
                    result.metrics["annualized_return"] - result.metrics.get("market_index_annualized_return", np.nan)
                ),
                "annualized_turnover": float(result.daily["portfolio_turnover"].mean() * 252),
            }
            rows.append(row)
            d = result.daily.copy()
            d["fold"] = fold["fold"]
            d["cost_bps"] = cost
            daily_frames.append(d)
            t = result.trades.copy()
            if not t.empty:
                t["fold"] = fold["fold"]
                t["cost_bps"] = cost
                trade_frames.append(t)
            log(f"cost {fold['fold']} {cost}bps ann={row['annualized_return']:.2%} mdd={row['max_drawdown']:.2%}")
    cost_df = pd.DataFrame(rows)
    cost_df.to_csv(stability_dir / "cost_sensitivity.csv", index=False, encoding="utf-8-sig")
    if daily_frames:
        pd.concat(daily_frames, ignore_index=True).to_parquet(stability_dir / "cost_sensitivity_daily.parquet", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_parquet(stability_dir / "cost_sensitivity_trades.parquet", index=False)
    write_stability_report(stability_dir, cost_df)
    return {"cost": cost_df, "path": stability_dir}


def build_live_layer(panel: pd.DataFrame, output: Path) -> dict[str, Any]:
    live_dir = output / "live_layer"
    live_dir.mkdir()
    audit = pd.read_csv(TRADE_AUDIT_RUN / "trade_execution_audit.csv")
    yearly_audit = pd.read_csv(TRADE_AUDIT_RUN / "yearly_trade_audit.csv")
    concentration = pd.read_csv(TRADE_AUDIT_RUN / "concentration_audit.csv")
    audit_summary = json.loads((TRADE_AUDIT_RUN / "audit_summary.json").read_text(encoding="utf-8"))
    capacity = capacity_analysis(panel, audit)
    slippage = slippage_analysis(audit, capacity)
    audit.to_csv(live_dir / "trade_execution_audit.csv", index=False, encoding="utf-8-sig")
    yearly_audit.to_csv(live_dir / "yearly_trade_audit.csv", index=False, encoding="utf-8-sig")
    concentration.to_csv(live_dir / "concentration_audit.csv", index=False, encoding="utf-8-sig")
    capacity.to_csv(live_dir / "capacity_analysis.csv", index=False, encoding="utf-8-sig")
    slippage.to_csv(live_dir / "slippage_analysis.csv", index=False, encoding="utf-8-sig")
    write_live_report(live_dir, audit_summary, yearly_audit, concentration, capacity, slippage)
    return {
        "audit_summary": audit_summary,
        "yearly_audit": yearly_audit,
        "concentration": concentration,
        "capacity": capacity,
        "slippage": slippage,
        "path": live_dir,
    }


def daily_ic(frame: pd.DataFrame, factor: str, label: str) -> pd.DataFrame:
    rows = []
    for date, g in frame.groupby("trade_date"):
        x = pd.to_numeric(g[factor], errors="coerce")
        y = pd.to_numeric(g[label], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 20 or x[mask].nunique() < 3 or y[mask].nunique() < 3:
            continue
        rows.append(
            {
                "trade_date": pd.Timestamp(date),
                "rank_ic": float(x[mask].corr(y[mask], method="spearman")),
                "pearson_ic": float(x[mask].corr(y[mask], method="pearson")),
                "n": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows, columns=["trade_date", "rank_ic", "pearson_ic", "n"])


def ic_summary_row(values: pd.Series, factor: Any, window: Any) -> dict[str, Any]:
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


def decile_return(frame: pd.DataFrame, factor: str, label: str, bins: int = 10) -> pd.DataFrame:
    rows = []
    for date, g in frame.groupby("trade_date"):
        h = g[[factor, label]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(h) < bins * 10 or h[factor].nunique() < bins:
            continue
        q = pd.qcut(h[factor].rank(method="first"), bins, labels=False) + 1
        for decile, local in h.assign(decile=q).groupby("decile"):
            rows.append(
                {
                    "trade_date": pd.Timestamp(date),
                    "decile": int(decile),
                    "mean_forward_return": float(local[label].mean()),
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


def factor_corr(frame: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    data = frame[factor_cols].replace([np.inf, -np.inf], np.nan)
    if len(data) > 250_000:
        data = data.sample(250_000, random_state=42)
    return data.corr(method="spearman")


def aggregate_feature_table(frame: pd.DataFrame, primary: str, secondary: str) -> pd.DataFrame:
    return (
        frame.groupby("feature", as_index=False)
        .agg(
            primary_mean=(primary, "mean"),
            primary_max=(primary, "max"),
            secondary_mean=(secondary, "mean"),
            folds=("fold", "nunique"),
        )
        .sort_values("primary_mean", ascending=False)
    )


def production_constraints() -> ExecutionConstraints:
    return ExecutionConstraints(
        exclude_suspended=True,
        cannot_buy_limit_up=True,
        cannot_sell_limit_down=True,
        exclude_st=True,
        exclude_delisting_period=True,
        min_listing_days=60,
    )


def annualized_turnover(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.loc[
        daily["selection"].eq(PRODUCTION_SELECTION)
        & daily["timing"].eq(PRODUCTION_TIMING)
        & daily["top_n"].eq(PRODUCTION_TOP_N)
    ].copy()
    return (
        d.groupby(["fold", "selection", "timing", "top_n"], as_index=False)["portfolio_turnover"]
        .mean()
        .rename(columns={"portfolio_turnover": "avg_daily_turnover"})
        .assign(annualized_turnover=lambda x: x["avg_daily_turnover"] * 252)
    )


def capacity_analysis(panel: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    trades = audit.copy()
    trades["trade_date"] = pd.to_datetime(trades["trade_date"])
    cols = ["trade_date", "ts_code", "amount_cny", "raw_open"]
    liquidity = panel[[c for c in cols if c in panel.columns]].copy()
    trades = trades.merge(liquidity, on=["trade_date", "ts_code"], how="left", suffixes=("", "_panel"))
    trades["participation"] = pd.to_numeric(trades["gross_value"], errors="coerce") / pd.to_numeric(trades["amount_cny"], errors="coerce")
    buys = trades.loc[trades["side"].eq("BUY")].copy()
    rows = []
    for fold, frame in buys.groupby("fold"):
        participation = frame["participation"].replace([np.inf, -np.inf], np.nan).dropna()
        gross = pd.to_numeric(frame["gross_value"], errors="coerce")
        amount = pd.to_numeric(frame["amount_cny"], errors="coerce")
        median_trade = float(gross.median()) if len(gross.dropna()) else np.nan
        p95_part = float(participation.quantile(0.95)) if len(participation) else np.nan
        max_part = float(participation.max()) if len(participation) else np.nan
        median_amount = float(amount.median()) if len(amount.dropna()) else np.nan
        rows.append(
            {
                "fold": fold,
                "buy_trades": int(len(frame)),
                "median_trade_value": median_trade,
                "median_amount_cny": median_amount,
                "median_participation": float(participation.median()) if len(participation) else np.nan,
                "p95_participation": p95_part,
                "max_participation": max_part,
                "capacity_at_5pct_participation_cash": safe_div(0.05, p95_part) * INITIAL_CASH if p95_part else np.nan,
                "capacity_at_10pct_participation_cash": safe_div(0.10, p95_part) * INITIAL_CASH if p95_part else np.nan,
            }
        )
    return pd.DataFrame(rows)


def slippage_analysis(audit: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bps in [5, 10, 20, 30, 50]:
        rows.append(
            {
                "round_trip_slippage_bps": bps,
                "estimated_annual_drag_at_turnover_1x": bps / 10_000.0,
                "note": "Linear turnover drag estimate; use cost_sensitivity.csv for full backtest effect.",
            }
        )
    if not capacity.empty:
        rows.append(
            {
                "round_trip_slippage_bps": np.nan,
                "estimated_annual_drag_at_turnover_1x": np.nan,
                "note": f"Observed p95 participation range: {capacity['p95_participation'].min():.4%}..{capacity['p95_participation'].max():.4%}",
            }
        )
    return pd.DataFrame(rows)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b and np.isfinite(b) else np.nan


def plot_deciles(deciles: pd.DataFrame, path: Path) -> None:
    if deciles.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [f for f in ["main_factor", *base.CLUSTER_COLS] if f in set(deciles["factor"])]
    fig, ax = plt.subplots(figsize=(10, 6))
    for factor in selected:
        local = deciles.loc[deciles["factor"].eq(factor)].sort_values("decile")
        ax.plot(local["decile"], local["mean_forward_return"], marker="o", label=factor)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Decile forward returns")
    ax.set_xlabel("Decile")
    ax.set_ylabel("Mean 10d forward return")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_corr(corr: pd.DataFrame, path: Path) -> None:
    if corr.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Spearman correlation matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def md_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_factor_report(path: Path, ic: pd.DataFrame, yearly: pd.DataFrame, deciles: pd.DataFrame, corr: pd.DataFrame) -> None:
    top_corr = corr.where(~np.eye(len(corr), dtype=bool)).abs().stack().sort_values(ascending=False).head(10)
    corr_rows = [{"factor_a": a, "factor_b": b, "abs_corr": v} for (a, b), v in top_corr.items() if a < b]
    lines = [
        "# Factor Report",
        "",
        "## IC Summary",
        md_table(ic),
        "",
        "## Yearly IC",
        md_table(yearly.sort_values(["factor", "window"])),
        "",
        "## Decile Returns",
        "See `decile_returns.csv` and `decile_returns.png`.",
        "",
        "## Correlation Matrix",
        "See `correlation_matrix.csv` and `correlation_matrix.png`.",
        "",
        "### Highest Pairwise Correlations",
        md_table(pd.DataFrame(corr_rows)),
    ]
    (path / "factor_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_report(path: Path, training: pd.DataFrame, ic: pd.DataFrame, gain: pd.DataFrame, shap: pd.DataFrame) -> None:
    lines = [
        "# LightGBM Model Report",
        "",
        "## Training Result",
        md_table(training),
        "",
        "## Train / Valid / Test IC",
        md_table(ic),
        "",
        "## Feature Importance",
        md_table(gain),
        "",
        "## SHAP Contribution",
        "LightGBM `pred_contrib=True` is used as SHAP-style contribution.",
        "",
        md_table(shap),
    ]
    (path / "model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_portfolio_report(path: Path, selected: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    lines = [
        "# Portfolio Report",
        "",
        "## Production Portfolio Metrics",
        md_table(selected),
        "",
        "## Parameter Sensitivity",
        md_table(sensitivity.sort_values(["fold", "selection", "timing", "top_n"])),
    ]
    (path / "portfolio_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stability_report(path: Path, cost: pd.DataFrame) -> None:
    lines = [
        "# Stability Report",
        "",
        "## Cost Sensitivity",
        md_table(cost.sort_values(["fold", "cost_bps"])),
    ]
    (path / "stability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_live_report(
    path: Path,
    audit_summary: dict[str, Any],
    yearly: pd.DataFrame,
    concentration: pd.DataFrame,
    capacity: pd.DataFrame,
    slippage: pd.DataFrame,
) -> None:
    lines = [
        "# Live Trading Report",
        "",
        "## Trade Audit",
        f"- Trade rows: `{audit_summary.get('trade_rows')}`",
        f"- Audit issue rows: `{audit_summary.get('audit_issue_rows')}`",
        f"- Buy issue rows: `{audit_summary.get('buy_issue_rows')}`",
        f"- Sell issue rows: `{audit_summary.get('sell_issue_rows')}`",
        "",
        "## Yearly Execution",
        md_table(yearly),
        "",
        "## Concentration",
        md_table(concentration),
        "",
        "## Capacity",
        md_table(capacity),
        "",
        "## Slippage",
        md_table(slippage),
    ]
    (path / "live_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_master_report(
    output: Path,
    *,
    data_version: str,
    factor_outputs: dict[str, Any],
    model_outputs: dict[str, Any],
    portfolio_outputs: dict[str, Any],
    cost_outputs: dict[str, Any],
    live_outputs: dict[str, Any],
) -> None:
    portfolio = portfolio_outputs["metrics"]
    model_ic = model_outputs["ic"]
    audit = live_outputs["audit_summary"]
    cost = cost_outputs["cost"]
    prod_mean = portfolio[["annualized_return", "annualized_excess_return_vs_csi1000", "sharpe", "calmar", "max_drawdown", "annualized_turnover"]].mean(numeric_only=True)
    lines = [
        "# Sell Impact Ranker Full Evaluation",
        "",
        "## Scope",
        f"- Data version: `{data_version}`",
        f"- Strategy: `{MODEL_VARIANT} + {PRODUCTION_SELECTION} + {PRODUCTION_TIMING} + Top{PRODUCTION_TOP_N} + hold{HOLDING_DAYS}`",
        f"- Benchmark: `{BENCHMARK_NAME}`",
        f"- Source walk-forward run: `{SOURCE_RUN}`",
        "",
        "## Executive Snapshot",
        f"- Mean annualized return: `{prod_mean.get('annualized_return', np.nan):.2%}`",
        f"- Mean annualized excess vs CSI1000: `{prod_mean.get('annualized_excess_return_vs_csi1000', np.nan):.2%}`",
        f"- Mean Sharpe: `{prod_mean.get('sharpe', np.nan):.2f}`",
        f"- Mean Calmar: `{prod_mean.get('calmar', np.nan):.2f}`",
        f"- Worst MDD: `{portfolio['max_drawdown'].min():.2%}`",
        f"- Mean annualized turnover: `{prod_mean.get('annualized_turnover', np.nan):.2f}`",
        f"- Trade audit issue rows: `{audit.get('audit_issue_rows')}`",
        "",
        "## 1. Factor Layer",
        f"- Report: `factor_layer/factor_report.md`",
        f"- IC: `factor_layer/ic_report.csv`, `factor_layer/yearly_ic.csv`, `factor_layer/daily_ic.csv`",
        f"- Decile plot: `factor_layer/decile_returns.png`",
        f"- Correlation matrix: `factor_layer/correlation_matrix.csv`, `factor_layer/correlation_matrix.png`",
        "",
        "Top factor IC:",
        md_table(factor_outputs["ic_report"], 10),
        "",
        "## 2. Model Layer",
        f"- Report: `model_layer/model_report.md`",
        f"- Train/Valid/Test IC: `model_layer/train_valid_test_ic.csv`",
        f"- Feature importance: `model_layer/feature_importance_gain.csv`",
        f"- SHAP-style contribution: `model_layer/shap_summary.csv`",
        "",
        md_table(model_ic.sort_values(["fold", "sample"]), 12),
        "",
        "## 3. Portfolio Layer",
        f"- Report: `portfolio_layer/portfolio_report.md`",
        f"- Metrics: `portfolio_layer/portfolio_metrics.csv`",
        "",
        md_table(portfolio[["fold", "annualized_return", "annualized_excess_return_vs_csi1000", "sharpe", "calmar", "max_drawdown", "annualized_turnover", "execution_rate"]]),
        "",
        "## 4. Stability",
        f"- Report: `stability_layer/stability_report.md`",
        f"- Yearly split: `portfolio_layer/yearly_return_split.csv`",
        f"- Parameter sensitivity: `portfolio_layer/parameter_sensitivity.csv`",
        f"- Cost sensitivity: `stability_layer/cost_sensitivity.csv`",
        "",
        md_table(cost[["fold", "cost_bps", "annualized_return", "annualized_excess_return_vs_csi1000", "sharpe", "max_drawdown", "annualized_turnover"]], 20),
        "",
        "## 5. Live Trading",
        f"- Report: `live_layer/live_report.md`",
        f"- Trade audit: `live_layer/trade_execution_audit.csv`",
        f"- Capacity: `live_layer/capacity_analysis.csv`",
        f"- Slippage: `live_layer/slippage_analysis.csv`",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
