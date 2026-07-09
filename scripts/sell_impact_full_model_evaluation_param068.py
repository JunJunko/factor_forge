from __future__ import annotations

import json
import shutil
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

import sell_impact_ranker_timing_compare as timing_compare
import sell_impact_sorting_repair as base
import sell_impact_trade_param_ml_surface as surface
import sell_impact_trade_quality_optimization as tq
from factor_forge.config import CostModel, ExecutionConstraints, load_project
from factor_forge.data import DataVersionRepository


SOURCE_RUN = Path("artifacts/strategy_reviews/sell_impact_recent_halfyear_tactical_20260709T100107Z")
PARAM_SURFACE_RUN = Path("artifacts/strategy_reviews/sell_impact_trade_param_ml_surface_20260709T120518Z")
OUTPUT_ROOT = Path("artifacts/strategy_reviews")
PARAM_ID = "param_068"
INITIAL_CASH = 1_000_000.0
COST_BPS = 20.0
LOT_SIZE = 100
TEST_START = "20260101"
TEST_END = "20260623"

CORE_FACTORS = [
    "raw_score",
    "band_score",
    "cluster_sell_impact",
    "cluster_condition_deviation",
    "cluster_price_reversal",
    "cluster_liquidity",
    "cluster_stock_state",
    "cluster_industry_context",
    "cluster_market_context",
    "stock_state_low_vol",
    "stock_state_small_size",
]


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_full_model_evaluation_param068_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    cfg = load_param_config()
    log(f"param={PARAM_ID} cfg={cfg}")
    signals = tq.load_signals()
    exposures = build_factor_exposures(signals, cfg)
    exposures.to_csv(output / "factor_exposures_2026h1.csv", index=False, encoding="utf-8-sig")
    exposures.to_parquet(output / "factor_exposures_2026h1.parquet", index=False)
    log(f"factor exposures rows={len(exposures):,}")

    version, panel = base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel_slice = panel.loc[panel["trade_date"].between(pd.Timestamp(TEST_START), pd.Timestamp(TEST_END))].copy()
    timing = timing_compare.load_position_multiplier(timing_compare.TIMING_DAILY)
    market_benchmark = timing_compare.load_market_benchmark(version)
    constraints = ExecutionConstraints(
        exclude_suspended=True,
        cannot_buy_limit_up=True,
        cannot_sell_limit_down=True,
        exclude_st=True,
        exclude_delisting_period=True,
        min_listing_days=60,
    )
    cost_model = CostModel(commission_bps_per_side=3, slippage_bps_per_side=5, stamp_duty_bps_sell=5)

    log("running param_068 with timing")
    daily_timing, trades_timing, positions_timing, metrics_timing = tq.run_trade_quality_backtest(
        panel=panel_slice,
        signals=signals,
        timing=timing,
        market_benchmark=market_benchmark,
        constraints=constraints,
        cost_model=cost_model,
        cfg=cfg,
    )
    log("running param_068 without timing")
    no_timing = pd.Series(1.0, index=pd.DatetimeIndex(panel_slice["trade_date"].unique()).sort_values())
    daily_no_timing, trades_no_timing, _positions_no_timing, metrics_no_timing = tq.run_trade_quality_backtest(
        panel=panel_slice,
        signals=signals,
        timing=no_timing,
        market_benchmark=market_benchmark,
        constraints=constraints,
        cost_model=cost_model,
        cfg=cfg,
    )

    trades_timing = trades_timing.assign(strategy="factor_timing_param068")
    trades_no_timing = trades_no_timing.assign(strategy="factor_only_param068")
    daily_timing = daily_timing.assign(strategy="factor_timing_param068")
    daily_no_timing = daily_no_timing.assign(strategy="factor_only_param068")
    positions_timing = positions_timing.assign(strategy="factor_timing_param068")

    index_nav = benchmark_navs(version, daily_timing["trade_date"])
    nav_curve = build_nav_curve(daily_timing, index_nav)
    nav_curve.to_csv(output / "nav_curve_daily.csv", index=False, encoding="utf-8-sig")
    benchmark_coverage(nav_curve).to_csv(output / "benchmark_coverage.csv", index=False, encoding="utf-8-sig")
    pd.concat([daily_timing, daily_no_timing], ignore_index=True).to_parquet(output / "daily_returns_detail.parquet", index=False)
    pd.concat([trades_timing, trades_no_timing], ignore_index=True).to_csv(output / "trade_ledger_buy_sell.csv", index=False, encoding="utf-8-sig")
    positions_timing.to_parquet(output / "positions_daily.parquet", index=False)
    capacity_detail, capacity_summary = capacity_slippage_analysis(trades_timing, panel_slice)
    capacity_detail.to_csv(output / "trade_capacity_detail.csv", index=False, encoding="utf-8-sig")
    capacity_summary.to_csv(output / "capacity_slippage_analysis.csv", index=False, encoding="utf-8-sig")

    round_trip = build_round_trip_trades(trades_timing)
    round_trip.to_csv(output / "round_trip_trades.csv", index=False, encoding="utf-8-sig")

    returns_daily = daily_returns_table(daily_timing)
    returns_daily.to_csv(output / "daily_return_series.csv", index=False, encoding="utf-8-sig")
    monthly = monthly_return_table(daily_timing, trades_timing)
    monthly.to_csv(output / "monthly_return_risk_trade_count.csv", index=False, encoding="utf-8-sig")

    risk = risk_report(daily_timing, metrics_timing)
    pd.DataFrame([risk]).to_csv(output / "risk_metrics.csv", index=False, encoding="utf-8-sig")

    ic_daily, ic_monthly, ic_summary = factor_ic_tables(exposures)
    ic_daily.to_csv(output / "factor_ic_daily.csv", index=False, encoding="utf-8-sig")
    ic_monthly.to_csv(output / "factor_ic_monthly.csv", index=False, encoding="utf-8-sig")
    ic_summary.to_csv(output / "factor_ic_summary.csv", index=False, encoding="utf-8-sig")

    deciles = decile_test(exposures)
    deciles.to_csv(output / "factor_decile_returns.csv", index=False, encoding="utf-8-sig")
    corr = factor_correlation_matrix(exposures)
    corr.to_csv(output / "factor_correlation_matrix.csv", encoding="utf-8-sig")

    timing_signal = timing_signal_table(daily_timing, timing)
    timing_signal.to_csv(output / "timing_signal_daily.csv", index=False, encoding="utf-8-sig")
    timing_compare_df = timing_contribution(metrics_timing, metrics_no_timing)
    timing_compare_df.to_csv(output / "timing_contribution_comparison.csv", index=False, encoding="utf-8-sig")
    gate_stats = gate_distribution_and_payoff(daily_timing, timing_signal)
    gate_stats.to_csv(output / "gate_distribution_payoff.csv", index=False, encoding="utf-8-sig")

    param_sensitivity = parameter_sensitivity_summary()
    param_sensitivity.to_csv(output / "parameter_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    anti_overfit = anti_overfit_summary(param_sensitivity)
    anti_overfit.to_csv(output / "anti_overfit_evidence.csv", index=False, encoding="utf-8-sig")
    model_layer = copy_model_layer(output)

    write_report(
        output=output,
        cfg=cfg,
        metrics_timing=metrics_timing,
        metrics_no_timing=metrics_no_timing,
        risk=risk,
        monthly=monthly,
        ic_summary=ic_summary,
        deciles=deciles,
        timing_compare_df=timing_compare_df,
        gate_stats=gate_stats,
        param_sensitivity=param_sensitivity,
        anti_overfit=anti_overfit,
        model_layer=model_layer,
        capacity_summary=capacity_summary,
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "source_run": str(SOURCE_RUN),
                "param_surface_run": str(PARAM_SURFACE_RUN),
                "param_id": PARAM_ID,
                "model": tq.MODEL,
                "band_target": tq.BAND_TARGET,
                "test_window": [TEST_START, TEST_END],
                "initial_cash": INITIAL_CASH,
                "cost_bps": COST_BPS,
                "lot_size": LOT_SIZE,
                "data_version": version,
                "notes": "param_068 is a tactical parameter selected inside 2026H1 sensitivity search; treat anti-overfit evidence as diagnostic, not OOS proof.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def load_param_config() -> dict[str, Any]:
    frame = pd.read_csv(PARAM_SURFACE_RUN / "param_search_metrics.csv")
    row = frame.loc[frame["variant"].eq(PARAM_ID)].iloc[0]
    cfg = {
        "variant": PARAM_ID,
        "description": "robust candidate from ML parameter response-surface search",
        "entry_pool": "threshold",
        "sell_rule": "continue",
    }
    for col in surface.PARAM_COLUMNS:
        val = row[col]
        cfg[col] = int(val) if col in {"max_positions", "min_hold_days", "max_hold_days"} else float(val)
    return cfg


def build_factor_exposures(signals: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    frame = signals.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])

    dataset = pd.read_parquet(SOURCE_RUN / "recent_halfyear_dataset.parquet")
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    add_cols = [
        col
        for col in [*CORE_FACTORS, "condition_quantile"]
        if col in dataset.columns and col not in frame.columns and col not in {"raw_score", "band_score"}
    ]
    if add_cols:
        frame = frame.merge(
            dataset[["trade_date", "ts_code", *add_cols]],
            on=["trade_date", "ts_code"],
            how="left",
        )

    frame["target_position"] = timing_compare.load_position_multiplier(timing_compare.TIMING_DAILY).reindex(frame["trade_date"]).to_numpy()
    frame["entry_gate_param068"] = (
        frame["band_rank_pct"].ge(float(cfg["entry_band_rank_min"]))
        & frame["raw_rank_pct"].ge(float(cfg["entry_raw_rank_min"]))
        & frame["stock_state_small_size"].le(float(cfg["max_microcap_score"]))
        & frame["cluster_liquidity"].ge(float(cfg["min_liquidity"]))
        & frame["cluster_price_reversal"].ge(float(cfg["min_price_reversal"]))
    )
    columns = [
        "trade_date",
        "ts_code",
        "label",
        "raw_score",
        "raw_rank_pct",
        "band_score",
        "band_rank_pct",
        "target_position",
        "entry_gate_param068",
        *[col for col in CORE_FACTORS if col in frame.columns and col not in {"raw_score", "band_score"}],
        "condition_quantile",
    ]
    return frame[[col for col in dict.fromkeys(columns) if col in frame.columns]].sort_values(["trade_date", "ts_code"])


def benchmark_navs(data_version: str, dates: pd.Series) -> pd.DataFrame:
    project = load_project(base.PROJECT_CONFIG)
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    index_daily = repo.load_raw_dataset(data_version, "index_daily")
    sources = []
    if index_daily is not None and not index_daily.empty:
        sources.append(index_daily)
    timing_index = Path("data/timing/index_daily.parquet")
    if timing_index.exists():
        sources.append(pd.read_parquet(timing_index))
    if not sources:
        raise ValueError("index_daily missing")
    index_daily = pd.concat(sources, ignore_index=True)
    index_daily["trade_date"] = pd.to_datetime(index_daily["trade_date"])
    codes = {"000300.SH": "hs300", "000905.SH": "csi500", "000852.SH": "csi1000"}
    out = pd.DataFrame({"trade_date": pd.to_datetime(pd.Series(dates).drop_duplicates()).sort_values()})
    for code, name in codes.items():
        item = index_daily.loc[index_daily["ts_code"].eq(code), ["trade_date", "open"]].copy()
        if item.empty:
            out[f"{name}_nav"] = np.nan
            continue
        item = item.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
        merged = out[["trade_date"]].merge(item, on="trade_date", how="left")
        price = merged["open"].ffill()
        first_price = price.dropna().iloc[0] if price.notna().any() else np.nan
        out[f"{name}_nav"] = price / first_price if np.isfinite(first_price) and first_price != 0 else np.nan
    return out


def build_nav_curve(daily: pd.DataFrame, index_nav: pd.DataFrame) -> pd.DataFrame:
    frame = daily[["trade_date", "nav", "return", "cash_ratio", "gross_exposure_ratio", "target_position"]].copy()
    frame = frame.rename(columns={"nav": "strategy_nav_value", "return": "strategy_return"})
    frame["strategy_nav"] = frame["strategy_nav_value"] / frame["strategy_nav_value"].iloc[0]
    return frame.merge(index_nav, on="trade_date", how="left")[
        [
            "trade_date",
            "strategy_nav",
            "strategy_nav_value",
            "strategy_return",
            "hs300_nav",
            "csi500_nav",
            "csi1000_nav",
            "cash_ratio",
            "gross_exposure_ratio",
            "target_position",
        ]
    ]


def benchmark_coverage(nav_curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for benchmark, column in {
        "沪深300": "hs300_nav",
        "中证500": "csi500_nav",
        "中证1000": "csi1000_nav",
    }.items():
        series = nav_curve[column]
        valid = series.notna()
        rows.append(
            {
                "benchmark": benchmark,
                "nav_column": column,
                "available_days": int(valid.sum()),
                "total_days": int(len(series)),
                "start_nav": float(series[valid].iloc[0]) if valid.any() else np.nan,
                "end_nav": float(series[valid].iloc[-1]) if valid.any() else np.nan,
                "period_return": float(series[valid].iloc[-1] / series[valid].iloc[0] - 1.0) if valid.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_round_trip_trades(trades: pd.DataFrame) -> pd.DataFrame:
    buys = trades.loc[trades["side"].eq("BUY")].copy()
    sells = trades.loc[trades["side"].eq("SELL")].copy()
    pairs = buys.merge(sells, on=["ts_code", "entry_date"], suffixes=("_buy", "_sell"), how="left")
    pairs["buy_date"] = pd.to_datetime(pairs["trade_date_buy"]).dt.date.astype(str)
    pairs["sell_date"] = pd.to_datetime(pairs["trade_date_sell"]).dt.date.astype(str)
    pairs["signal_date"] = pd.to_datetime(pairs["signal_date_buy"]).dt.date.astype(str)
    pairs["holding_calendar_days"] = (
        pd.to_datetime(pairs["trade_date_sell"]) - pd.to_datetime(pairs["trade_date_buy"])
    ).dt.days
    pairs["buy_price"] = pairs["raw_open_buy"].astype(float)
    pairs["sell_price"] = pairs["raw_open_sell"].astype(float)
    pairs["shares"] = pairs["shares_buy"].astype("Int64")
    pairs["buy_gross_value"] = pairs["gross_value_buy"].astype(float)
    pairs["sell_gross_value"] = pairs["gross_value_sell"].astype(float)
    pairs["buy_cost"] = pairs["cost_buy"].astype(float)
    pairs["sell_cost"] = pairs["cost_sell"].astype(float)
    pairs["net_pnl"] = pairs["sell_gross_value"] - pairs["sell_cost"] - pairs["buy_gross_value"] - pairs["buy_cost"]
    pairs["net_return"] = pairs["net_pnl"] / (pairs["buy_gross_value"] + pairs["buy_cost"])
    cols = [
        "ts_code",
        "signal_date",
        "buy_date",
        "sell_date",
        "holding_calendar_days",
        "shares",
        "buy_price",
        "sell_price",
        "buy_gross_value",
        "sell_gross_value",
        "buy_cost",
        "sell_cost",
        "net_pnl",
        "net_return",
        "reason_sell",
        "entry_band_rank_pct_buy",
        "entry_raw_rank_pct_buy",
    ]
    return pairs[[c for c in cols if c in pairs.columns]].rename(
        columns={
            "reason_sell": "sell_reason",
            "entry_band_rank_pct_buy": "entry_band_rank_pct",
            "entry_raw_rank_pct_buy": "entry_raw_rank_pct",
        }
    )


def capacity_slippage_analysis(trades: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_cols = ["trade_date", "ts_code", "amount_cny", "volume_shares", "raw_open", "raw_close"]
    market = panel[[col for col in market_cols if col in panel.columns]].copy()
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    detail = trades.copy()
    detail["trade_date"] = pd.to_datetime(detail["trade_date"])
    detail = detail.merge(market, on=["trade_date", "ts_code"], how="left", suffixes=("", "_market"))
    detail["participation_amount"] = detail["gross_value"] / detail["amount_cny"].replace(0, np.nan)
    detail["cost_bps_realized"] = detail["cost"] / detail["gross_value"].replace(0, np.nan) * 10_000.0
    detail["capacity_multiple_at_1pct_adv"] = 0.01 / detail["participation_amount"]
    detail["capacity_multiple_at_3pct_adv"] = 0.03 / detail["participation_amount"]
    detail["capacity_multiple_at_5pct_adv"] = 0.05 / detail["participation_amount"]

    rows = []
    for side, group in detail.groupby("side"):
        participation = group["participation_amount"].dropna()
        p95 = participation.quantile(0.95) if len(participation) else np.nan
        rows.append(
            {
                "side": side,
                "trade_count": int(len(group)),
                "gross_value_mean": float(group["gross_value"].mean()),
                "gross_value_p95": float(group["gross_value"].quantile(0.95)),
                "gross_value_max": float(group["gross_value"].max()),
                "participation_mean": float(participation.mean()) if len(participation) else np.nan,
                "participation_p95": float(p95) if len(participation) else np.nan,
                "participation_max": float(participation.max()) if len(participation) else np.nan,
                "trades_over_1pct_adv": int((participation > 0.01).sum()),
                "trades_over_3pct_adv": int((participation > 0.03).sum()),
                "trades_over_5pct_adv": int((participation > 0.05).sum()),
                "cost_bps_mean": float(group["cost_bps_realized"].mean()),
                "capacity_cash_at_p95_1pct_adv": float(INITIAL_CASH * 0.01 / p95) if pd.notna(p95) and p95 > 0 else np.nan,
                "capacity_cash_at_p95_3pct_adv": float(INITIAL_CASH * 0.03 / p95) if pd.notna(p95) and p95 > 0 else np.nan,
                "capacity_cash_at_p95_5pct_adv": float(INITIAL_CASH * 0.05 / p95) if pd.notna(p95) and p95 > 0 else np.nan,
            }
        )
    return detail.sort_values(["trade_date", "side", "ts_code"]), pd.DataFrame(rows)


def daily_returns_table(daily: pd.DataFrame) -> pd.DataFrame:
    return daily[
        [
            "trade_date",
            "return",
            "nav",
            "gross_exposure_ratio",
            "cash_ratio",
            "portfolio_turnover",
            "target_position",
            "executed_buys",
            "executed_sells",
            "holding_count",
            "unique_holding_count",
        ]
    ].rename(columns={"return": "strategy_daily_return"})


def monthly_return_table(daily: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    return tq.period_breakdown(daily, trades, "factor_timing_param068", "M").rename(
        columns={"period": "month", "period_return": "monthly_return", "interval_max_drawdown": "monthly_mdd"}
    )


def risk_report(daily: pd.DataFrame, metrics: dict[str, Any]) -> dict[str, Any]:
    dd = drawdown_details(daily)
    returns = daily["return"]
    monthly = monthly_return_table(daily, pd.DataFrame())
    monthly_ret = monthly["monthly_return"]
    return {
        "start_date": pd.to_datetime(daily["trade_date"]).min().date().isoformat(),
        "end_date": pd.to_datetime(daily["trade_date"]).max().date().isoformat(),
        "initial_cash": INITIAL_CASH,
        "total_return": metrics["total_return"],
        "cagr": metrics["annualized_return"],
        "annualized_volatility": metrics["annualized_volatility"],
        "daily_return_std": float(returns.std(ddof=1)),
        "sharpe_annual": metrics["sharpe"],
        "sharpe_monthly": float(monthly_ret.mean() / monthly_ret.std(ddof=1) * np.sqrt(12)) if monthly_ret.std(ddof=1) > 0 else np.nan,
        "calmar": metrics["calmar"],
        "max_drawdown": metrics["max_drawdown"],
        **dd,
        "annualized_turnover": metrics["annualized_turnover"],
        "avg_gross_exposure_ratio": metrics["avg_gross_exposure_ratio"],
        "trade_count": metrics["trade_count"],
        "executed_buys": metrics["executed_buys"],
        "executed_sells": metrics["executed_sells"],
    }


def drawdown_details(daily: pd.DataFrame) -> dict[str, Any]:
    frame = daily[["trade_date", "nav"]].copy()
    frame["peak"] = frame["nav"].cummax()
    frame["drawdown"] = frame["nav"] / frame["peak"] - 1.0
    trough_idx = frame["drawdown"].idxmin()
    trough = frame.loc[trough_idx]
    peak_idx = frame.loc[:trough_idx, "nav"].idxmax()
    peak = frame.loc[peak_idx]
    after = frame.loc[trough_idx:].copy()
    recovered = after.loc[after["nav"].ge(float(peak["nav"]))]
    recovery_date = pd.NaT if recovered.empty else recovered.iloc[0]["trade_date"]
    return {
        "mdd_peak_date": pd.to_datetime(peak["trade_date"]).date().isoformat(),
        "mdd_trough_date": pd.to_datetime(trough["trade_date"]).date().isoformat(),
        "mdd_recovery_date": None if pd.isna(recovery_date) else pd.to_datetime(recovery_date).date().isoformat(),
        "mdd_duration_trading_days": int(trough_idx - peak_idx),
        "mdd_recovery_trading_days": None if recovered.empty else int(recovered.index[0] - peak_idx),
    }


def factor_ic_tables(exposures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    factors = [f for f in CORE_FACTORS if f in exposures.columns]
    for factor in factors:
        for date, group in exposures.groupby("trade_date"):
            data = group[[factor, "label"]].dropna()
            if len(data) < 30 or data[factor].nunique() < 3:
                continue
            rows.append(
                {
                    "trade_date": date,
                    "factor": factor,
                    "pearson_ic": float(data[factor].corr(data["label"], method="pearson")),
                    "spearman_rank_ic": float(data[factor].corr(data["label"], method="spearman")),
                    "n": int(len(data)),
                }
            )
    daily = pd.DataFrame(rows)
    daily["month"] = pd.to_datetime(daily["trade_date"]).dt.to_period("M").astype(str)
    monthly = (
        daily.groupby(["factor", "month"], as_index=False)
        .agg(pearson_ic=("pearson_ic", "mean"), spearman_rank_ic=("spearman_rank_ic", "mean"), days=("trade_date", "nunique"))
        .sort_values(["factor", "month"])
    )
    summary = (
        daily.groupby("factor", as_index=False)
        .agg(
            pearson_ic_mean=("pearson_ic", "mean"),
            pearson_ic_std=("pearson_ic", "std"),
            rank_ic_mean=("spearman_rank_ic", "mean"),
            rank_ic_std=("spearman_rank_ic", "std"),
            positive_rank_ic_ratio=("spearman_rank_ic", lambda s: float((s > 0).mean())),
            days=("trade_date", "nunique"),
        )
    )
    summary["rank_icir"] = summary["rank_ic_mean"] / summary["rank_ic_std"]
    summary["pearson_icir"] = summary["pearson_ic_mean"] / summary["pearson_ic_std"]
    return daily, monthly, summary.sort_values("rank_ic_mean", ascending=False)


def decile_test(exposures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in ["raw_score", "band_score", "cluster_price_reversal", "stock_state_low_vol"]:
        if factor not in exposures:
            continue
        for date, group in exposures.groupby("trade_date"):
            data = group[[factor, "label"]].dropna().copy()
            if len(data) < 100 or data[factor].nunique() < 10:
                continue
            data["decile"] = pd.qcut(data[factor].rank(method="first"), 10, labels=False) + 1
            for decile, dec in data.groupby("decile"):
                rows.append({"trade_date": date, "factor": factor, "decile": int(decile), "mean_label": float(dec["label"].mean())})
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["factor", "decile"], as_index=False)
        .agg(mean_forward_return=("mean_label", "mean"), positive_days=("mean_label", lambda s: float((s > 0).mean())), days=("trade_date", "nunique"))
        .sort_values(["factor", "decile"])
    )


def factor_correlation_matrix(exposures: pd.DataFrame) -> pd.DataFrame:
    factors = [f for f in CORE_FACTORS if f in exposures.columns]
    mats = []
    for _, group in exposures.groupby("trade_date"):
        data = group[factors].dropna(how="all")
        if len(data) < 30:
            continue
        mats.append(data.corr(method="spearman"))
    if not mats:
        return pd.DataFrame()
    return sum(mats) / len(mats)


def timing_signal_table(daily: pd.DataFrame, timing: pd.Series) -> pd.DataFrame:
    out = daily[["trade_date", "return", "gross_exposure_ratio", "cash_ratio"]].copy()
    out["target_position"] = pd.Series(timing.reindex(pd.to_datetime(out["trade_date"])).to_numpy()).fillna(1.0).clip(0, 1)
    out["timing_bucket"] = pd.qcut(out["target_position"].rank(method="first"), 3, labels=["low", "mid", "high"])
    return out


def timing_contribution(metrics_timing: dict[str, Any], metrics_no_timing: dict[str, Any]) -> pd.DataFrame:
    keys = ["annualized_return", "total_return", "sharpe", "max_drawdown", "trade_count", "annualized_turnover", "avg_gross_exposure_ratio"]
    return pd.DataFrame(
        [
            {"version": "factor_only_param068", **{k: metrics_no_timing.get(k) for k in keys}},
            {"version": "factor_timing_param068", **{k: metrics_timing.get(k) for k in keys}},
        ]
    )


def gate_distribution_and_payoff(daily: pd.DataFrame, timing_signal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket, group in timing_signal.groupby("timing_bucket", observed=False):
        rows.append(
            {
                "gate": "target_position",
                "bucket": str(bucket),
                "days": int(len(group)),
                "mean_gate": float(group["target_position"].mean()),
                "period_return": float((1.0 + group["return"]).prod() - 1.0),
                "mean_daily_return": float(group["return"].mean()),
                "vol_daily": float(group["return"].std(ddof=1)),
            }
        )
    exposure = build_factor_exposures(tq.load_signals(), load_param_config())
    for value, group in exposure.groupby("entry_gate_param068"):
        rows.append(
            {
                "gate": "entry_gate_param068",
                "bucket": str(bool(value)),
                "days": int(group["trade_date"].nunique()),
                "mean_gate": float(bool(value)),
                "period_return": np.nan,
                "mean_daily_return": float(group["label"].mean()),
                "vol_daily": float(group["label"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def parameter_sensitivity_summary() -> pd.DataFrame:
    metrics = pd.read_csv(PARAM_SURFACE_RUN / "param_search_metrics.csv")
    robust = pd.read_csv(PARAM_SURFACE_RUN / "robust_parameter_candidates.csv")
    importance = pd.read_csv(PARAM_SURFACE_RUN / "response_surface_feature_importance.csv")
    rows = []
    for col in surface.PARAM_COLUMNS:
        group = metrics.groupby(col).agg(
            samples=("variant", "count"),
            mean_objective=("objective", "mean"),
            median_ann=("annualized_return", "median"),
            median_mdd=("max_drawdown", "median"),
            median_trades=("trade_count", "median"),
        )
        best = group.sort_values("mean_objective", ascending=False).head(1).reset_index()
        rows.append(
            {
                "parameter": col,
                "best_value_by_mean_objective": best[col].iloc[0],
                "best_value_samples": int(best["samples"].iloc[0]),
                "best_value_mean_objective": float(best["mean_objective"].iloc[0]),
                "importance_rank": int(importance["parameter"].tolist().index(col) + 1) if col in importance["parameter"].tolist() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("importance_rank")


def anti_overfit_summary(param_sensitivity: pd.DataFrame) -> pd.DataFrame:
    summary = json.loads((PARAM_SURFACE_RUN / "summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(PARAM_SURFACE_RUN / "param_search_metrics.csv")
    target = metrics.loc[metrics["variant"].eq(PARAM_ID)].iloc[0]
    rows = [
        {
            "check": "sample_scope",
            "status": "WEAK",
            "evidence": "param_068 was selected from 2026H1 parameter search; not an untouched OOS result.",
        },
        {
            "check": "response_surface_cv_r2",
            "status": "WEAK" if summary["response_surface"]["cv_r2"] < 0.4 else "OK",
            "evidence": f"ExtraTrees CV R2={summary['response_surface']['cv_r2']:.3f}; parameter response is noisy.",
        },
        {
            "check": "parameter_neighborhood",
            "status": "WATCH",
            "evidence": f"{len(metrics)} sampled configs; param_068 objective={target['objective']:.3f}, rank should be validated in a local neighborhood and future shadow run.",
        },
        {
            "check": "trade_count",
            "status": "OK" if target["trade_count"] <= 320 else "WATCH",
            "evidence": f"param_068 trade_count={int(target['trade_count'])}; much lower than original 998.",
        },
        {
            "check": "month_concentration",
            "status": "OK" if target["top_month_return_share"] <= 0.55 else "WATCH",
            "evidence": f"top positive month return share={target['top_month_return_share']:.2%}.",
        },
    ]
    return pd.DataFrame(rows)


def copy_model_layer(output: Path) -> dict[str, pd.DataFrame]:
    model_dir = output / "model_layer"
    model_dir.mkdir(exist_ok=True)
    files = {
        "train_valid_test_ic": "model_train_valid_test_ic.csv",
        "lightgbm_training": "lightgbm_training_result.csv",
        "feature_importance": "feature_importance_gain.csv",
        "shap_summary": "shap_contribution_summary.csv",
        "factor_group_contribution": "factor_group_contribution.csv",
        "feature_map": "feature_map.csv",
        "style_exposure": "style_exposure.csv",
    }
    frames: dict[str, pd.DataFrame] = {}
    for key, filename in files.items():
        source = SOURCE_RUN / filename
        if not source.exists():
            frames[key] = pd.DataFrame()
            continue
        target = model_dir / filename
        shutil.copy2(source, target)
        frames[key] = pd.read_csv(target)
    return frames


def md_table(frame: pd.DataFrame, max_rows: int = 60) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(
    *,
    output: Path,
    cfg: dict[str, Any],
    metrics_timing: dict[str, Any],
    metrics_no_timing: dict[str, Any],
    risk: dict[str, Any],
    monthly: pd.DataFrame,
    ic_summary: pd.DataFrame,
    deciles: pd.DataFrame,
    timing_compare_df: pd.DataFrame,
    gate_stats: pd.DataFrame,
    param_sensitivity: pd.DataFrame,
    anti_overfit: pd.DataFrame,
    model_layer: dict[str, pd.DataFrame],
    capacity_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Full Model Evaluation: Factor + Timing + param_068",
        "",
        "## Model Definition",
        f"- Factor model: `{tq.MODEL}` LightGBM ranker prediction.",
        f"- Score transform: score-band target `{tq.BAND_TARGET}`.",
        f"- Timing: `target_position` from `{timing_compare.TIMING_DAILY}`.",
        f"- Trade parameters: `{PARAM_ID}` from `{PARAM_SURFACE_RUN}`.",
        f"- Param config: `{json.dumps(cfg, ensure_ascii=False)}`",
        "- Execution: T close signal, T+1 open trade; excludes suspended, ST, delisting, new listings <60 days; cannot buy limit-up open or sell limit-down open.",
        f"- Cost: total round-trip scenario `{COST_BPS}` bps, applied half on buy and half on sell.",
        "- Benchmarks: HS300 is loaded from `data/timing/index_daily.parquet`; CSI1000 is loaded from the backtest data version; CSI500 is left blank when no local index history is available.",
        "",
        "## Core Performance",
        md_table(pd.DataFrame([metrics_timing])),
        "",
        "## Risk Metrics",
        md_table(pd.DataFrame([risk])),
        "",
        "## Monthly Return / Drawdown / Trades",
        md_table(monthly, 20),
        "",
        "## Factor IC Summary",
        md_table(ic_summary, 20),
        "",
        "## LightGBM Model Layer",
        "### Train / Valid / Test IC",
        md_table(model_layer.get("train_valid_test_ic", pd.DataFrame()), 30),
        "",
        "### Feature Importance",
        md_table(model_layer.get("feature_importance", pd.DataFrame()), 30),
        "",
        "### SHAP Contribution",
        md_table(model_layer.get("shap_summary", pd.DataFrame()), 30),
        "",
        "### Factor Group Contribution",
        md_table(model_layer.get("factor_group_contribution", pd.DataFrame()), 30),
        "",
        "## Factor Decile Returns",
        md_table(deciles, 60),
        "",
        "## Timing Contribution",
        md_table(timing_compare_df, 10),
        "",
        "## Gate Distribution / Payoff",
        md_table(gate_stats, 20),
        "",
        "## Capacity / Slippage",
        md_table(capacity_summary, 10),
        "",
        "## Parameter Sensitivity",
        md_table(param_sensitivity, 20),
        "",
        "## Anti-overfit Evidence",
        md_table(anti_overfit, 20),
        "",
        "## Required Files",
        "- `nav_curve_daily.csv`",
        "- `benchmark_coverage.csv`",
        "- `daily_return_series.csv`",
        "- `monthly_return_risk_trade_count.csv`",
        "- `risk_metrics.csv`",
        "- `factor_exposures_2026h1.csv`",
        "- `factor_ic_daily.csv`, `factor_ic_monthly.csv`, `factor_ic_summary.csv`",
        "- `factor_decile_returns.csv`",
        "- `factor_correlation_matrix.csv`",
        "- `timing_signal_daily.csv`",
        "- `timing_contribution_comparison.csv`",
        "- `gate_distribution_payoff.csv`",
        "- `trade_ledger_buy_sell.csv`, `round_trip_trades.csv`",
        "- `trade_capacity_detail.csv`, `capacity_slippage_analysis.csv`",
        "- `parameter_sensitivity_summary.csv`",
        "- `anti_overfit_evidence.csv`",
        "- `model_layer/model_train_valid_test_ic.csv`",
        "- `model_layer/lightgbm_training_result.csv`",
        "- `model_layer/feature_importance_gain.csv`",
        "- `model_layer/shap_contribution_summary.csv`",
        "- `model_layer/factor_group_contribution.csv`",
    ]
    (output / "full_model_evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
