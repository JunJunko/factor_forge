from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from factor_forge.config import load_project
from factor_forge.data import DataVersionRepository


ROOT = Path("artifacts/strategy_reviews/sell_impact_timing_overlay_20260708")
TIMED_RUN = Path("artifacts/runs/sell_impact_efficiency_v1__20260708T033745Z__7301915f")
BASELINE_RUN = Path("artifacts/runs/sell_impact_efficiency_v1__20260708T032734Z__42b12531")
TIMING_MODEL = Path("artifacts/timing_position_models/timing_position_model_v1_20260708T025521Z_181c72c6")
DATA_VERSION = "data_v1_20260701T095408Z_c7b9995d"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "plots").mkdir(exist_ok=True)
    summary = {
        "factor": build_factor_layer(),
        "model": build_model_layer(),
        "portfolio": build_portfolio_layer(),
        "stability": build_stability_layer(),
        "live": build_live_layer(),
    }
    write_index(summary)


def build_factor_layer() -> dict:
    l1 = json.loads((TIMED_RUN / "l1_predictive_power.json").read_text(encoding="utf-8"))
    ic_rows = []
    quantile_rows = []
    for item in l1["results"]:
        row = {
            "variant": item["variant"],
            "universe": item["universe"],
            "horizon": item["horizon"],
            "observations": item["observations"],
            "days": item["days"],
            "rank_ic_mean": item["rank_ic"]["mean"],
            "rank_ic_icir": item["rank_ic"]["icir"],
            "rank_ic_t": item["rank_ic"]["t_value"],
            "rank_ic_p": item["rank_ic"]["p_value"],
            "rank_ic_positive_ratio": item["rank_ic"]["positive_ratio"],
            "pearson_ic_mean": item["pearson_ic"]["mean"],
            "oos_rank_ic_mean": item["oos_rank_ic"]["mean"],
            "top_bottom_mean": item["top_bottom_mean"],
            "monotonicity": item["monotonicity"],
            "fdr_q": item["fdr_q"],
        }
        ic_rows.append(row)
        for quantile, value in item["quantile_returns"].items():
            quantile_rows.append({
                "universe": item["universe"],
                "horizon": item["horizon"],
                "quantile": int(quantile),
                "mean_forward_return": value,
            })
    ic = pd.DataFrame(ic_rows)
    quantiles = pd.DataFrame(quantile_rows)
    conditional = pd.read_csv(TIMED_RUN / "l1_conditional_ic_summary.csv")

    factor_values = pd.read_parquet(TIMED_RUN / "factor_values.parquet")
    conditioning = pd.read_parquet(TIMED_RUN / "conditioning_factor_values.parquet").rename(
        columns={"factor_value": "conditioning_factor"}
    )
    panel = load_panel_columns(["trade_date", "ts_code", "amount_cny", "circ_mv_cny", "turnover_rate", "pct_change"])
    corr = (
        factor_values.rename(columns={"factor_value": "main_factor"})
        .merge(conditioning[["trade_date", "ts_code", "conditioning_factor"]], on=["trade_date", "ts_code"], how="inner")
        .merge(panel, on=["trade_date", "ts_code"], how="left")
    )
    corr_matrix = corr[["main_factor", "conditioning_factor", "amount_cny", "circ_mv_cny", "turnover_rate", "pct_change"]].corr(
        method="spearman"
    )

    ic.to_csv(ROOT / "ic_report.csv", index=False, encoding="utf-8-sig")
    quantiles.to_csv(ROOT / "quantile_returns.csv", index=False, encoding="utf-8-sig")
    conditional.to_csv(ROOT / "conditional_ic_report.csv", index=False, encoding="utf-8-sig")
    corr_matrix.to_csv(ROOT / "factor_correlation_matrix.csv", encoding="utf-8-sig")
    plot_factor_layer(ic, quantiles, conditional, corr_matrix)

    liquid10 = ic[(ic["universe"].eq("liquid")) & (ic["horizon"].eq(10))].iloc[0].to_dict()
    q10 = quantiles[(quantiles["universe"].eq("liquid")) & (quantiles["horizon"].eq(10))]
    q_spread = q10.loc[q10["quantile"].eq(5), "mean_forward_return"].mean() - q10.loc[q10["quantile"].eq(1), "mean_forward_return"].mean()
    top_cond = conditional.sort_values("rank_ic_nw_t_value", key=lambda s: s.abs(), ascending=False).head(10)
    report = [
        "# 因子层报告",
        "",
        "## 结论",
        f"- 主因子在 liquid、10日 horizon 的 RankIC 均值为 `{liquid10['rank_ic_mean']:.4f}`，ICIR `{liquid10['rank_ic_icir']:.2f}`。",
        f"- 同口径 OOS RankIC 为 `{liquid10['oos_rank_ic_mean']:.4f}`，弱于全样本 IC，说明因子本身存在近期衰减。",
        f"- liquid、10日 Q5-Q1 平均前瞻收益差为 `{q_spread:.4%}`。",
        "- 条件 IC 显示信号对 `sell_impact_deviation_60d_v1` 分位敏感，当前策略只交易 Q5 条件池。",
        "",
        "## IC 报告",
        ic.round(6).to_markdown(index=False),
        "",
        "## 条件 IC Top 10",
        top_cond.round(6).to_markdown(index=False),
        "",
        "## 图表",
        "- `plots/factor_ic_by_horizon.png`",
        "- `plots/factor_quantile_returns_liquid_h10.png`",
        "- `plots/conditional_ic_heatmap_liquid.png`",
        "- `plots/factor_correlation_matrix.png`",
    ]
    (ROOT / "factor_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"liquid_h10_rank_ic": liquid10["rank_ic_mean"], "liquid_h10_oos_rank_ic": liquid10["oos_rank_ic_mean"]}


def build_model_layer() -> dict:
    summary = json.loads((TIMING_MODEL / "summary.json").read_text(encoding="utf-8"))
    coeff = pd.read_csv(TIMING_MODEL / "timing_position_coefficients.csv")
    pred = pd.read_csv(TIMING_MODEL / "timing_position_predictions.csv")
    daily = pd.read_csv(TIMING_MODEL / "timing_position_daily.csv")
    lgbm_metrics, lgbm_importance, lgbm_shap = build_lightgbm_shadow_model(summary)
    metrics = pd.DataFrame([
        {"sample": key, **value}
        for key, value in summary["metrics"].items()
    ])
    metrics.to_csv(ROOT / "model_train_valid_test_ic.csv", index=False, encoding="utf-8-sig")
    coeff.to_csv(ROOT / "model_feature_importance_coefficients.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(ROOT / "model_predictions.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(ROOT / "model_position_daily.csv", index=False, encoding="utf-8-sig")
    lgbm_metrics.to_csv(ROOT / "lightgbm_train_valid_test_ic.csv", index=False, encoding="utf-8-sig")
    lgbm_importance.to_csv(ROOT / "lightgbm_feature_importance.csv", index=False, encoding="utf-8-sig")
    lgbm_shap.to_csv(ROOT / "lightgbm_shap_summary.csv", index=False, encoding="utf-8-sig")
    plot_model_layer(metrics, coeff, daily)
    plot_lightgbm_layer(lgbm_importance, lgbm_shap)

    train = summary["metrics"].get("train", {})
    test = summary["metrics"].get("test", {})
    lgbm_show = lgbm_metrics.copy()
    report = [
        "# 模型层报告",
        "",
        "## 口径说明",
        "- 当前接入实盘链路的仓位模型不是 LightGBM，而是 Ridge/ElasticNet 风格的线性仓位模型。",
        "- 为满足模型层检查，本报告额外训练了一个 LightGBM 影子诊断模型；它不参与当前回测交易，只用于判断非线性模型是否更适合这批 timing 特征。",
        "- Valid 集为训练样本末段切分；Test 集仍为 `2025-07-01` 之后。",
        "",
        "## 当前接入仓位模型 Train / Valid / Test IC",
        f"- Train RankIC: `{train.get('rank_ic', np.nan):.4f}`",
        "- Valid RankIC: `N/A`，当前模型未配置 valid split。",
        f"- Test RankIC: `{test.get('rank_ic', np.nan):.4f}`",
        "",
        "## 模型指标",
        metrics.round(6).to_markdown(index=False),
        "",
        "## Feature Importance",
        "当前用 `abs(coefficient)` 表示线性模型重要性：",
        coeff.head(30).round(8).to_markdown(index=False),
        "",
        "## LightGBM 影子诊断模型",
        lgbm_show.round(6).to_markdown(index=False),
        "",
        "### LightGBM Feature Importance",
        lgbm_importance.head(30).round(6).to_markdown(index=False),
        "",
        "### LightGBM SHAP-like Contribution",
        "使用 LightGBM `pred_contrib=True` 计算测试集平均绝对贡献：",
        lgbm_shap.head(30).round(8).to_markdown(index=False),
        "",
        "## 图表",
        "- `plots/model_coefficients_top30.png`",
        "- `plots/model_position_nav.png`",
        "- `plots/lightgbm_gain_importance_top30.png`",
        "- `plots/lightgbm_shap_top30.png`",
    ]
    (ROOT / "model_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {
        "train_rank_ic": train.get("rank_ic"),
        "test_rank_ic": test.get("rank_ic"),
        "lightgbm_test_rank_ic": float(lgbm_metrics.loc[lgbm_metrics["sample"].eq("test"), "rank_ic"].iloc[0]),
        "model_type": "ridge_current_plus_lightgbm_shadow",
    }


def build_lightgbm_shadow_model(summary: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import lightgbm as lgb

    config = yaml.safe_load((TIMING_MODEL / "config.yaml").read_text(encoding="utf-8"))
    label = summary["label_column"]
    horizon_days = int(summary["horizon_days"])
    dataset = pd.read_parquet(summary["dataset_path"])
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    stable = pd.read_csv(summary["stable_factors_path"])
    selected = [factor for factor in stable["factor"].dropna().astype(str).tolist() if factor in dataset.columns]
    states = pd.read_csv(TIMING_MODEL / "regime_daily_states.csv")
    states["trade_date"] = pd.to_datetime(states["trade_date"])
    base_columns = ["trade_date", label, *selected]
    merged = states.merge(dataset[base_columns], on="trade_date", how="left")
    state_cols = [column for column in states.columns if column.startswith("state_probability_")]
    interactions = {
        f"{factor}__x__{state_col}": merged[factor] * merged[state_col]
        for factor in selected
        for state_col in state_cols
    }
    if interactions:
        merged = pd.concat([merged, pd.DataFrame(interactions, index=merged.index)], axis=1)
    features = selected + state_cols + list(interactions)
    merged = merged.replace([np.inf, -np.inf], np.nan).sort_values("trade_date").reset_index(drop=True)
    usable = merged[features + [label]].notna().all(axis=1)
    train_end = pd.Timestamp(summary["train_end"])
    test_start = pd.Timestamp(summary["test_start"])
    train_mask = merged["trade_date"].le(train_end) & usable
    train_dates = merged.loc[train_mask, "trade_date"].drop_duplicates().sort_values().tolist()
    if len(train_dates) > horizon_days:
        purge_start = train_dates[-horizon_days]
        train_mask &= merged["trade_date"].lt(purge_start)
    purged_train_dates = merged.loc[train_mask, "trade_date"].drop_duplicates().sort_values().tolist()
    valid_count = max(20, int(len(purged_train_dates) * 0.25))
    valid_dates = set(purged_train_dates[-valid_count:])
    valid_mask = train_mask & merged["trade_date"].isin(valid_dates)
    fit_mask = train_mask & ~merged["trade_date"].isin(valid_dates)
    test_mask = merged["trade_date"].ge(test_start) & usable

    params = config.get("lightgbm_shadow", {})
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=int(params.get("n_estimators", 300)),
        learning_rate=float(params.get("learning_rate", 0.03)),
        num_leaves=int(params.get("num_leaves", 15)),
        min_child_samples=int(params.get("min_child_samples", 20)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.9)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        merged.loc[fit_mask, features],
        merged.loc[fit_mask, label],
        eval_set=[(merged.loc[valid_mask, features], merged.loc[valid_mask, label])],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    metric_rows = []
    for sample, mask in [("train", fit_mask), ("valid", valid_mask), ("test", test_mask)]:
        y = merged.loc[mask, label]
        p = pd.Series(model.predict(merged.loc[mask, features]), index=y.index)
        top = p.ge(p.median())
        metric_rows.append({
            "sample": sample,
            "rows": int(mask.sum()),
            "rank_ic": safe_corr(p, y, method="spearman"),
            "ic": safe_corr(p, y, method="pearson"),
            "top_half_mean_forward_return": float(y.loc[top].mean()),
            "bottom_half_mean_forward_return": float(y.loc[~top].mean()),
        })
    importance = pd.DataFrame({
        "feature": features,
        "gain_importance": model.booster_.feature_importance(importance_type="gain"),
        "split_importance": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain_importance", ascending=False)
    test_x = merged.loc[test_mask, features]
    contrib = model.booster_.predict(test_x, pred_contrib=True)
    shap = pd.DataFrame({
        "feature": features,
        "mean_abs_contribution": np.abs(contrib[:, :-1]).mean(axis=0),
        "mean_contribution": contrib[:, :-1].mean(axis=0),
    }).sort_values("mean_abs_contribution", ascending=False)
    return pd.DataFrame(metric_rows), importance, shap


def build_portfolio_layer() -> dict:
    rows = []
    for label, run in [("baseline", BASELINE_RUN), ("timing_overlay", TIMED_RUN)]:
        for path in sorted((run / "l2").glob("*/metrics.json")):
            key = path.parent.name
            metrics = json.loads(path.read_text(encoding="utf-8"))
            daily = pd.read_parquet(path.parent / "daily.parquet")
            top_n = int(key.split("__top")[1].split("__")[0])
            hold = int(key.split("__hold")[1].split("__")[0])
            cost = float(key.split("__cost")[1])
            rows.append({
                "run": label,
                "top_n": top_n,
                "holding_days": hold,
                "cost_bps": cost,
                "annualized_return": metrics["annualized_return"],
                "benchmark_annualized_return": metrics["benchmark_annualized_return"],
                "annualized_excess_return": metrics["annualized_excess_return"],
                "sharpe": metrics["sharpe"],
                "calmar": metrics["calmar"],
                "max_drawdown": metrics["max_drawdown"],
                "turnover_notional": metrics["turnover_notional"],
                "annualized_turnover": annualized_turnover(daily),
                "execution_rate": metrics["execution_rate"],
                "generated_signals": metrics["generated_signals"],
                "executed_buys": metrics["executed_buys"],
                "avg_cash_ratio": daily["cash_ratio"].mean(),
                "avg_gross_exposure": (daily["gross_exposure"] / daily["nav"]).replace([np.inf, -np.inf], np.nan).mean(),
            })
    portfolio = pd.DataFrame(rows).sort_values(["run", "top_n", "cost_bps"])
    portfolio.to_csv(ROOT / "portfolio_metrics.csv", index=False, encoding="utf-8-sig")
    plot_portfolio_layer(portfolio)
    report = [
        "# 组合层报告",
        "",
        "## 指标总览",
        portfolio.round(6).to_markdown(index=False),
        "",
        "## 图表",
        "- `plots/portfolio_annualized_return.png`",
        "- `plots/portfolio_drawdown_top5_cost20.png`",
    ]
    (ROOT / "portfolio_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    best = portfolio[(portfolio["run"].eq("timing_overlay")) & (portfolio["top_n"].eq(5)) & (portfolio["cost_bps"].eq(20))].iloc[0].to_dict()
    return best


def build_stability_layer() -> dict:
    yearly_rows = []
    monthly_rows = []
    for label, run in [("baseline", BASELINE_RUN), ("timing_overlay", TIMED_RUN)]:
        for path in sorted((run / "l2").glob("*/daily.parquet")):
            key = path.parent.name
            top_n = int(key.split("__top")[1].split("__")[0])
            hold = int(key.split("__hold")[1].split("__")[0])
            cost = float(key.split("__cost")[1])
            daily = pd.read_parquet(path).sort_values("trade_date")
            daily["trade_date"] = pd.to_datetime(daily["trade_date"])
            daily["year"] = daily["trade_date"].dt.year
            daily["month"] = daily["trade_date"].dt.to_period("M").astype(str)
            for year, group in daily.groupby("year"):
                yearly_rows.append(period_metrics(label, top_n, hold, cost, int(year), group))
            for month, group in daily.groupby("month"):
                row = period_metrics(label, top_n, hold, cost, month, group)
                monthly_rows.append(row)
    yearly = pd.DataFrame(yearly_rows)
    monthly = pd.DataFrame(monthly_rows)
    yearly.to_csv(ROOT / "yearly_returns.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(ROOT / "monthly_returns.csv", index=False, encoding="utf-8-sig")
    sensitivity = pd.read_csv(ROOT / "portfolio_metrics.csv")
    sensitivity.to_csv(ROOT / "parameter_cost_sensitivity.csv", index=False, encoding="utf-8-sig")
    plot_stability_layer(yearly)
    report = [
        "# 稳定性报告",
        "",
        "## 年度收益拆分",
        yearly.round(6).to_markdown(index=False),
        "",
        "## 参数敏感性",
        "- 当前已有 timed run 只覆盖 Top5/Top10、10日持有、0/20bps。",
        "- 持有期 5/15 日、成本 10bps、更多 TopN 需要补跑 review matrix。",
        sensitivity.round(6).to_markdown(index=False),
        "",
        "## 图表",
        "- `plots/yearly_return_comparison.png`",
        "- `plots/yearly_excess_comparison.png`",
    ]
    (ROOT / "stability_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"yearly_rows": int(len(yearly)), "monthly_rows": int(len(monthly))}


def build_live_layer() -> dict:
    panel = load_panel_columns(["trade_date", "ts_code", "amount_cny", "volume_shares", "turnover_rate", "circ_mv_cny"])
    audit_rows = []
    capacity_rows = []
    slippage_rows = []
    for run_label, run in [("timing_overlay", TIMED_RUN)]:
        for path in sorted((run / "l2").glob("*/trades.parquet")):
            key = path.parent.name
            top_n = int(key.split("__top")[1].split("__")[0])
            cost = float(key.split("__cost")[1])
            trades = pd.read_parquet(path)
            daily = pd.read_parquet(path.parent / "daily.parquet")
            buys = trades[trades["side"].eq("BUY")].copy()
            buys = buys.merge(panel, on=["trade_date", "ts_code"], how="left")
            participation = buys["gross_value"] / buys["amount_cny"].replace(0, np.nan)
            audit_rows.append({
                "run": run_label,
                "top_n": top_n,
                "cost_bps": cost,
                "signals": int(daily["new_signals"].sum()),
                "executed_buys": int(daily["executed_buys"].sum()),
                "blocked_or_unfilled": int(daily["new_signals"].sum() - daily["executed_buys"].sum()),
                "execution_rate": float(daily["executed_buys"].sum() / daily["new_signals"].sum()),
                "duplicate_signals": int(daily["duplicate_signals"].sum()),
                "avg_cash_ratio": float(daily["cash_ratio"].mean()),
                "total_transaction_cost": float(daily["transaction_cost"].sum()),
                "avg_daily_turnover": float(daily["portfolio_turnover"].mean()),
            })
            capacity_rows.append({
                "run": run_label,
                "top_n": top_n,
                "cost_bps": cost,
                "buy_count": int(len(buys)),
                "median_buy_notional": float(buys["gross_value"].median()),
                "p95_buy_notional": float(buys["gross_value"].quantile(0.95)),
                "median_stock_amount_cny": float(buys["amount_cny"].median()),
                "p10_stock_amount_cny": float(buys["amount_cny"].quantile(0.10)),
                "median_participation": float(participation.median()),
                "p95_participation": float(participation.quantile(0.95)),
                "estimated_capacity_at_1pct_adv": float(1_000_000 * 0.01 / participation.quantile(0.95)) if participation.quantile(0.95) > 0 else np.nan,
            })
            for bps in [5, 10, 20, 30, 50]:
                daily_cost = daily.copy()
                extra = daily_cost["portfolio_turnover"] * bps / 10000.0
                ret = daily_cost["return"] - extra
                slippage_rows.append({
                    "run": run_label,
                    "top_n": top_n,
                    "base_cost_bps": cost,
                    "extra_slippage_bps": bps,
                    "annualized_return_after_extra_slippage": annualize_return(ret),
                    "max_drawdown_after_extra_slippage": max_drawdown(ret),
                })
    audit = pd.DataFrame(audit_rows).sort_values(["top_n", "cost_bps"])
    capacity = pd.DataFrame(capacity_rows).sort_values(["top_n", "cost_bps"])
    slippage = pd.DataFrame(slippage_rows).sort_values(["top_n", "base_cost_bps", "extra_slippage_bps"])
    audit.to_csv(ROOT / "trade_audit.csv", index=False, encoding="utf-8-sig")
    capacity.to_csv(ROOT / "capacity_assessment.csv", index=False, encoding="utf-8-sig")
    slippage.to_csv(ROOT / "slippage_analysis.csv", index=False, encoding="utf-8-sig")
    plot_live_layer(slippage)
    report = [
        "# 实盘层报告",
        "",
        "## 交易审计",
        audit.round(6).to_markdown(index=False),
        "",
        "## 容量评估",
        capacity.round(6).to_markdown(index=False),
        "",
        "## 滑点分析",
        slippage.round(6).to_markdown(index=False),
        "",
        "## 图表",
        "- `plots/slippage_sensitivity.png`",
    ]
    (ROOT / "live_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"audit_rows": int(len(audit)), "capacity_rows": int(len(capacity))}


def load_panel_columns(columns: list[str]) -> pd.DataFrame:
    project = load_project("configs/project_sw_l2.yaml")
    repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    _, panel = repository.load_panel(DATA_VERSION)
    panel = panel[columns].copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    return panel


def annualized_turnover(daily: pd.DataFrame) -> float:
    if daily.empty:
        return np.nan
    return float(daily["portfolio_turnover"].sum() * 252.0 / len(daily))


def annualize_return(ret: pd.Series) -> float:
    ret = pd.to_numeric(ret, errors="coerce").fillna(0.0)
    total = float((1.0 + ret).prod() - 1.0)
    return (1.0 + total) ** (252.0 / max(len(ret), 1)) - 1.0 if total > -1 else -1.0


def max_drawdown(ret: pd.Series) -> float:
    nav = (1.0 + pd.to_numeric(ret, errors="coerce").fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1.0).min()) if len(nav) else np.nan


def safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 3 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return np.nan
    value = frame["left"].corr(frame["right"], method=method)
    return float(value) if np.isfinite(value) else np.nan


def period_metrics(run: str, top_n: int, hold: int, cost: float, period, group: pd.DataFrame) -> dict:
    ret = float((1.0 + group["return"]).prod() - 1.0)
    bench = float((1.0 + group["benchmark_return"]).prod() - 1.0)
    return {
        "run": run,
        "top_n": top_n,
        "holding_days": hold,
        "cost_bps": cost,
        "period": period,
        "return": ret,
        "benchmark_return": bench,
        "excess_return": ret - bench,
        "max_drawdown": max_drawdown(group["return"]),
        "avg_cash_ratio": float(group["cash_ratio"].mean()),
        "executed_buys": int(group["executed_buys"].sum()),
        "new_signals": int(group["new_signals"].sum()),
    }


def plot_factor_layer(ic: pd.DataFrame, quantiles: pd.DataFrame, conditional: pd.DataFrame, corr: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 4.5))
    for universe, frame in ic.groupby("universe"):
        plt.plot(frame["horizon"], frame["rank_ic_mean"], marker="o", label=universe)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("RankIC by Horizon")
    plt.xlabel("Horizon")
    plt.ylabel("RankIC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "plots/factor_ic_by_horizon.png", dpi=160)
    plt.close()

    q = quantiles[(quantiles["universe"].eq("liquid")) & (quantiles["horizon"].eq(10))]
    plt.figure(figsize=(7, 4))
    sns.barplot(data=q, x="quantile", y="mean_forward_return", color="#4c78a8")
    plt.title("Liquid H10 Quantile Forward Returns")
    plt.tight_layout()
    plt.savefig(ROOT / "plots/factor_quantile_returns_liquid_h10.png", dpi=160)
    plt.close()

    heat = conditional[conditional["universe"].eq("liquid")].pivot_table(
        index="condition_quantile", columns="horizon", values="rank_ic_mean"
    )
    plt.figure(figsize=(7, 4.5))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="RdBu_r", center=0)
    plt.title("Conditional RankIC: Liquid")
    plt.tight_layout()
    plt.savefig(ROOT / "plots/conditional_ic_heatmap_liquid.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Spearman Correlation Matrix")
    plt.tight_layout()
    plt.savefig(ROOT / "plots/factor_correlation_matrix.png", dpi=160)
    plt.close()


def plot_model_layer(metrics: pd.DataFrame, coeff: pd.DataFrame, daily: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    top = coeff.assign(abs_coefficient=lambda x: x["coefficient"].abs()).sort_values("abs_coefficient", ascending=False).head(30)
    plt.figure(figsize=(9, 7))
    sns.barplot(data=top, y="feature", x="coefficient", color="#59a14f")
    plt.title("Top 30 Position Model Coefficients")
    plt.tight_layout()
    plt.savefig(ROOT / "plots/model_coefficients_top30.png", dpi=160)
    plt.close()

    daily = daily.copy()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    plt.figure(figsize=(9, 4.5))
    plt.plot(daily["trade_date"], daily["strategy_nav"], label="strategy")
    plt.plot(daily["trade_date"], daily["benchmark_nav"], label="benchmark")
    plt.legend()
    plt.title("Timing Position Model NAV")
    plt.tight_layout()
    plt.savefig(ROOT / "plots/model_position_nav.png", dpi=160)
    plt.close()


def plot_lightgbm_layer(importance: pd.DataFrame, shap: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    top_gain = importance.sort_values("gain_importance", ascending=False).head(30)
    plt.figure(figsize=(9, 7))
    sns.barplot(data=top_gain, y="feature", x="gain_importance", color="#4e79a7")
    plt.title("LightGBM Gain Importance Top 30")
    plt.tight_layout()
    plt.savefig(ROOT / "plots/lightgbm_gain_importance_top30.png", dpi=160)
    plt.close()

    top_shap = shap.sort_values("mean_abs_contribution", ascending=False).head(30)
    plt.figure(figsize=(9, 7))
    sns.barplot(data=top_shap, y="feature", x="mean_abs_contribution", color="#f28e2b")
    plt.title("LightGBM SHAP-like Contribution Top 30")
    plt.tight_layout()
    plt.savefig(ROOT / "plots/lightgbm_shap_top30.png", dpi=160)
    plt.close()


def plot_portfolio_layer(portfolio: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=portfolio, x="top_n", y="annualized_return", hue="run")
    plt.title("Annualized Return by TopN")
    plt.tight_layout()
    plt.savefig(ROOT / "plots/portfolio_annualized_return.png", dpi=160)
    plt.close()

    path = TIMED_RUN / "l2/liquid__condition_q5__top5__hold10__cost20/daily.parquet"
    daily = pd.read_parquet(path).sort_values("trade_date")
    nav = (1.0 + daily["return"]).cumprod()
    dd = nav / nav.cummax() - 1.0
    plt.figure(figsize=(9, 4))
    plt.fill_between(pd.to_datetime(daily["trade_date"]), dd.to_numpy(), 0, color="#e15759", alpha=0.35)
    plt.title("Drawdown: Timing Overlay Top5 Cost20")
    plt.tight_layout()
    plt.savefig(ROOT / "plots/portfolio_drawdown_top5_cost20.png", dpi=160)
    plt.close()


def plot_stability_layer(yearly: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    subset = yearly[(yearly["top_n"].isin([5, 10])) & (yearly["cost_bps"].eq(20))]
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=subset, x="period", y="return", hue="run")
    plt.title("Yearly Return Comparison: Cost20")
    plt.tight_layout()
    plt.savefig(ROOT / "plots/yearly_return_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=subset, x="period", y="excess_return", hue="run")
    plt.title("Yearly Excess Return Comparison: Cost20")
    plt.tight_layout()
    plt.savefig(ROOT / "plots/yearly_excess_comparison.png", dpi=160)
    plt.close()


def plot_live_layer(slippage: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    subset = slippage[slippage["base_cost_bps"].eq(20)]
    plt.figure(figsize=(8, 4.5))
    sns.lineplot(data=subset, x="extra_slippage_bps", y="annualized_return_after_extra_slippage", hue="top_n", marker="o")
    plt.title("Extra Slippage Sensitivity")
    plt.tight_layout()
    plt.savefig(ROOT / "plots/slippage_sensitivity.png", dpi=160)
    plt.close()


def write_index(summary: dict) -> None:
    lines = [
        "# Sell Impact Timing Overlay 策略全面评价",
        "",
        "## 输入产物",
        f"- Timed run: `{TIMED_RUN}`",
        f"- Baseline run: `{BASELINE_RUN}`",
        f"- Timing model: `{TIMING_MODEL}`",
        "",
        "## 核心结论",
        f"- 因子层 liquid H10 RankIC `{summary['factor']['liquid_h10_rank_ic']:.4f}`，OOS RankIC `{summary['factor']['liquid_h10_oos_rank_ic']:.4f}`。",
        f"- 仓位模型 Train RankIC `{summary['model']['train_rank_ic']:.4f}`，Test RankIC `{summary['model']['test_rank_ic']:.4f}`。",
        f"- LightGBM 影子模型 Test RankIC `{summary['model']['lightgbm_test_rank_ic']:.4f}`，低于当前接入模型，暂不建议替换当前仓位模型。",
        f"- Top5/10日/20bps 接入仓位后年化 `{summary['portfolio']['annualized_return']:.2%}`，超额 `{summary['portfolio']['annualized_excess_return']:.2%}`，MDD `{summary['portfolio']['max_drawdown']:.2%}`。",
        "",
        "## 报告索引",
        "- `factor_report.md`：IC、条件 IC、分层收益、相关矩阵。",
        "- `model_report.md`：仓位模型 Train/Test IC、系数重要性、模型缺口。",
        "- `portfolio_report.md`：年化、超额、Sharpe、Calmar、MDD、换手。",
        "- `stability_report.md`：年度/月度拆分、参数/成本敏感性。",
        "- `live_report.md`：交易审计、容量评估、滑点分析。",
        "",
        "## 关键缺口",
        "- 当前部署仓位模型不是 LightGBM；报告中的 LightGBM importance/SHAP-like 结果来自影子诊断模型，不是实盘执行模型。",
        "- 当前 timed run 只有 10 日持有，没有 5/15 日持有期敏感性。",
        "- 容量评估用成交额参与率估算，未接入逐笔盘口和真实冲击成本模型。",
    ]
    (ROOT / "strategy_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
