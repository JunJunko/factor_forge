from __future__ import annotations

import argparse
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

import sell_impact_ranker_timing_compare as timing_compare
import sell_impact_sorting_repair as base
import sell_impact_trade_param_ml_surface as surface
import sell_impact_trade_quality_optimization as tq
from factor_forge.config import CostModel, ExecutionConstraints


OUTPUT_ROOT = Path("artifacts/strategy_reviews/reliability_portfolio_results")
PARAM_SURFACE_RUN = Path("artifacts/strategy_reviews/sell_impact_trade_param_ml_surface_20260709T120518Z")
DEFAULT_RELIABILITY = Path(
    "artifacts/factor_reliability/factor_reliability_model_v1_20260709T143452Z/factor_reliability_daily.csv"
)
PARAM_ID = "param_068"
BASE_FACTOR_WEIGHTS = {
    "band": 0.50,
    "reversal": 0.30,
    "lowvol": 0.20,
}
FACTOR_COLUMN_CANDIDATES = {
    "band": ["band_rank_pct", "band_score"],
    "reversal": ["cluster_price_reversal", "price_reversal", "stock_price_reversal"],
    "lowvol": ["stock_state_low_vol", "low_vol", "stock_low_vol"],
}
FACTOR_RELIABILITY_ALIASES = {
    "band": ["band_score", "band_score_wf_selected", "band"],
    "reversal": ["price_reversal", "cluster_price_reversal", "reversal"],
    "lowvol": ["low_vol", "stock_state_low_vol", "lowvol"],
}
RANDOM_SEED = 20260709


def main() -> None:
    args = parse_args()
    output = OUTPUT_ROOT / f"reliability_portfolio_backtest_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("loading baseline signals, panel, timing and reliability")
    signals = tq.load_signals()
    signals["trade_date"] = pd.to_datetime(signals["trade_date"])
    version, panel = base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel_slice = panel.loc[panel["trade_date"].between(pd.Timestamp(tq.TEST_START), pd.Timestamp(tq.TEST_END))].copy()
    timing = timing_compare.load_position_multiplier(timing_compare.TIMING_DAILY)
    market_benchmark = timing_compare.load_market_benchmark(version)
    reliability = load_reliability(args.reliability_daily)
    regime = load_market_regime(args.reliability_dataset)
    cfg = load_param_config()
    constraints = ExecutionConstraints(
        exclude_suspended=True,
        cannot_buy_limit_up=True,
        cannot_sell_limit_down=True,
        exclude_st=True,
        exclude_delisting_period=True,
        min_listing_days=60,
    )
    cost_model = CostModel(commission_bps_per_side=3, slippage_bps_per_side=5, stamp_duty_bps_sell=5)
    factor_columns = detect_factor_columns(signals)
    log(f"factor_columns={factor_columns}")
    log(f"param_cfg={cfg}")

    variants = build_variant_specs()
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []
    weight_frames: list[pd.DataFrame] = []

    for spec in variants:
        name = spec["variant"]
        log(f"running {name}: {spec['description']}")
        variant_signals, weights = build_variant_signals(
            signals=signals,
            reliability=reliability,
            factor_columns=factor_columns,
            spec=spec,
        )
        daily, trades, positions, metrics = tq.run_trade_quality_backtest(
            panel=panel_slice,
            signals=variant_signals,
            timing=timing,
            market_benchmark=market_benchmark,
            constraints=constraints,
            cost_model=cost_model,
            cfg=cfg,
        )
        csi1000 = float(metrics.get("market_index_annualized_return", np.nan))
        metrics_row = {
            "variant": name,
            "horizon": spec.get("horizon"),
            "ablation": spec.get("ablation"),
            "description": spec["description"],
            **metrics,
            "annualized_excess_return_vs_csi1000": float(metrics["annualized_return"] - csi1000),
        }
        metric_rows.append(metrics_row)
        ic_rows.append(score_ic_summary(variant_signals, name))
        daily["variant"] = name
        trades["variant"] = name
        positions["variant"] = name
        weights["variant"] = name
        daily_frames.append(daily)
        trade_frames.append(trades)
        position_frames.append(positions)
        weight_frames.append(weights)
        log(
            f"{name}: ann={metrics['annualized_return']:.2%} sharpe={metrics['sharpe']:.2f} "
            f"mdd={metrics['max_drawdown']:.2%} buys={metrics['executed_buys']}"
        )

    daily_all = pd.concat(daily_frames, ignore_index=True)
    trades_all = pd.concat(trade_frames, ignore_index=True)
    positions_all = pd.concat(position_frames, ignore_index=True)
    weights_all = pd.concat(weight_frames, ignore_index=True)
    performance = pd.DataFrame(metric_rows)
    alpha_quality = pd.DataFrame(ic_rows)
    monthly = pd.concat(
        [tq.period_breakdown(frame, trades_all.loc[trades_all["variant"].eq(frame["variant"].iloc[0])], frame["variant"].iloc[0], "M") for frame in daily_frames],
        ignore_index=True,
    )
    drawdown = drawdown_analysis(daily_all)
    regime_analysis = build_regime_analysis(daily_all, regime)
    weight_logic = factor_weight_diagnostics(weights_all)

    daily_all.to_csv(output / "portfolio_nav.csv", index=False, encoding="utf-8-sig")
    trades_all.to_csv(output / "portfolio_trades.csv", index=False, encoding="utf-8-sig")
    positions_all.to_parquet(output / "portfolio_positions.parquet", index=False)
    weights_all.to_csv(output / "daily_factor_weights.csv", index=False, encoding="utf-8-sig")
    weights_all.to_csv(output / "factor_weight_history.csv", index=False, encoding="utf-8-sig")
    performance.to_csv(output / "performance_summary.csv", index=False, encoding="utf-8-sig")
    alpha_quality.to_csv(output / "alpha_quality_icir.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_breakdown.csv", index=False, encoding="utf-8-sig")
    drawdown.to_csv(output / "drawdown_analysis.csv", index=False, encoding="utf-8-sig")
    regime_analysis.to_csv(output / "regime_analysis.csv", index=False, encoding="utf-8-sig")
    weight_logic.to_csv(output / "factor_weight_diagnostics.csv", index=False, encoding="utf-8-sig")

    write_report(
        output=output,
        performance=performance,
        alpha_quality=alpha_quality,
        drawdown=drawdown,
        regime_analysis=regime_analysis,
        weights=weights_all,
        weight_logic=weight_logic,
        factor_columns=factor_columns,
        reliability_path=args.reliability_daily,
        data_version=version,
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "experiment": "Experiment 6: Reliability-aware Portfolio Backtest",
                "param_id": PARAM_ID,
                "source_signals": str(tq.SOURCE_RUN),
                "param_surface_run": str(PARAM_SURFACE_RUN),
                "reliability_daily": str(args.reliability_daily),
                "reliability_dataset": str(args.reliability_dataset),
                "data_version": version,
                "test_window": [tq.TEST_START, tq.TEST_END],
                "factor_columns": factor_columns,
                "base_factor_weights": BASE_FACTOR_WEIGHTS,
                "note": (
                    "Production backtest code is not modified. Variants differ only by research signal score "
                    "construction before the existing param_068 execution layer."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reliability-aware portfolio backtest for param_068.")
    parser.add_argument("--reliability-daily", type=Path, default=DEFAULT_RELIABILITY)
    parser.add_argument(
        "--reliability-dataset",
        type=Path,
        default=Path("artifacts/factor_reliability/factor_reliability_dataset_20260709T142147Z/factor_reliability_dataset.csv"),
    )
    return parser.parse_args()


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


def load_reliability(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["factor_name"] = frame["factor_name"].astype(str)
    return frame.replace([np.inf, -np.inf], np.nan)


def load_market_regime(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    date_col = "date" if "date" in frame.columns else "trade_date"
    frame[date_col] = pd.to_datetime(frame[date_col])
    out = frame.rename(columns={date_col: "trade_date"})[
        ["trade_date", "market_ret_20", "market_ret_60", "market_vol_20", "market_breadth_20", "market_xsec_vol_20"]
    ].copy()
    out = out.drop_duplicates("trade_date").sort_values("trade_date")
    vol_cut = out["market_vol_20"].median()
    out["market_state"] = np.where(out["market_ret_20"].ge(0.0), "bull", "bear")
    out["vol_state"] = np.where(out["market_vol_20"].ge(vol_cut), "high_volatility", "low_volatility")
    return out


def detect_factor_columns(signals: pd.DataFrame) -> dict[str, str]:
    found: dict[str, str] = {}
    for factor, candidates in FACTOR_COLUMN_CANDIDATES.items():
        for col in candidates:
            if col in signals.columns:
                found[factor] = col
                break
    missing = [factor for factor in BASE_FACTOR_WEIGHTS if factor not in found]
    if missing:
        raise ValueError(f"missing factor exposure columns for {missing}; available={signals.columns.tolist()}")
    return found


def build_variant_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "variant": "baseline_param068",
            "description": "Original param_068 signal and execution.",
            "mode": "baseline",
            "horizon": None,
            "ablation": "baseline",
        },
        {
            "variant": "fixed_factor_weight",
            "description": "Fixed explicit factor blend bridge: band/reversal/lowvol = 50/30/20.",
            "mode": "fixed",
            "horizon": None,
            "ablation": "fixed_bridge",
        },
    ]
    for horizon in [5, 10, 20]:
        specs.append(
            {
                "variant": f"reliability_{horizon}d_band_only",
                "description": f"Experiment A: only band weight uses reliability_{horizon}d; other factor weights fixed.",
                "mode": "reliability",
                "horizon": horizon,
                "ablation": "A_band_only",
                "dynamic_factors": ["band"],
            }
        )
        specs.append(
            {
                "variant": f"reliability_{horizon}d_all_available",
                "description": (
                    f"Experiment B: all detected factors use factor-specific reliability_{horizon}d when available; "
                    "missing factor reliability defaults to 1."
                ),
                "mode": "reliability",
                "horizon": horizon,
                "ablation": "B_all_available",
                "dynamic_factors": list(BASE_FACTOR_WEIGHTS),
            }
        )
    specs.append(
        {
            "variant": "placebo_random_reliability_5d",
            "description": "Experiment C: random reliability placebo on all factors, fixed seed.",
            "mode": "placebo",
            "horizon": 5,
            "ablation": "C_random_placebo",
            "dynamic_factors": list(BASE_FACTOR_WEIGHTS),
        }
    )
    return specs


def build_variant_signals(
    *,
    signals: pd.DataFrame,
    reliability: pd.DataFrame,
    factor_columns: dict[str, str],
    spec: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if spec["mode"] == "baseline":
        weights = fixed_weight_history(signals, spec["variant"])
        out = signals.copy()
        out["portfolio_score"] = out["band_score"]
        return out, weights

    rel = reliability_for_spec(signals, reliability, spec)
    weights = dynamic_weight_history(signals, rel, spec)
    out = signals.merge(weights, on="trade_date", how="left")
    out["portfolio_score"] = 0.0
    for factor, col in factor_columns.items():
        z = out.groupby("trade_date")[col].transform(cs_zscore)
        out["portfolio_score"] += z.fillna(0.0) * out[f"{factor}_weight"].fillna(BASE_FACTOR_WEIGHTS[factor])

    out["raw_score_original"] = out["raw_score"]
    out["raw_rank_pct_original"] = out["raw_rank_pct"]
    out["band_score_original"] = out["band_score"]
    out["band_rank_pct_original"] = out["band_rank_pct"]
    out["raw_score"] = out["portfolio_score"]
    out["raw_rank_pct"] = out.groupby("trade_date")["raw_score"].rank(pct=True, method="first")
    out["band_score"] = -(out["raw_rank_pct"] - tq.BAND_TARGET).abs()
    out["band_rank_pct"] = out.groupby("trade_date")["band_score"].rank(pct=True, method="first")
    return out.replace([np.inf, -np.inf], np.nan), weights


def fixed_weight_history(signals: pd.DataFrame, variant: str) -> pd.DataFrame:
    dates = pd.DataFrame({"trade_date": sorted(pd.to_datetime(signals["trade_date"]).unique())})
    dates["variant"] = variant
    for factor, weight in BASE_FACTOR_WEIGHTS.items():
        dates[f"{factor}_reliability"] = 1.0
        dates[f"{factor}_weight"] = float(weight)
    return dates


def reliability_for_spec(signals: pd.DataFrame, reliability: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    dates = pd.DataFrame({"trade_date": sorted(pd.to_datetime(signals["trade_date"]).unique())})
    rng = np.random.default_rng(RANDOM_SEED)
    horizon = int(spec.get("horizon") or 5)
    rel_col = f"reliability_{horizon}d"
    out = dates.copy()
    for factor in BASE_FACTOR_WEIGHTS:
        if spec["mode"] == "placebo":
            out[f"{factor}_reliability"] = rng.uniform(0.05, 1.0, size=len(out))
            continue
        if factor not in spec.get("dynamic_factors", []):
            out[f"{factor}_reliability"] = 1.0
            continue
        match = select_reliability_factor(reliability, factor, rel_col)
        if match is None:
            out[f"{factor}_reliability"] = 1.0
            continue
        values = reliability.loc[reliability["factor_name"].eq(match), ["date", rel_col]].rename(
            columns={"date": "trade_date", rel_col: f"{factor}_reliability"}
        )
        out = out.merge(values, on="trade_date", how="left")
        out[f"{factor}_reliability"] = out[f"{factor}_reliability"].ffill().fillna(1.0).clip(0.0, 1.0)
    return out


def select_reliability_factor(reliability: pd.DataFrame, factor: str, rel_col: str) -> str | None:
    if rel_col not in reliability.columns:
        return None
    names = sorted(reliability["factor_name"].dropna().astype(str).unique())
    aliases = FACTOR_RELIABILITY_ALIASES.get(factor, [factor])
    for alias in aliases:
        for name in names:
            if alias.lower() == name.lower():
                return name
    for alias in aliases:
        for name in names:
            if alias.lower() in name.lower():
                return name
    return None


def dynamic_weight_history(signals: pd.DataFrame, rel: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    out = rel.copy()
    raw_cols = []
    for factor, base_weight in BASE_FACTOR_WEIGHTS.items():
        raw_col = f"{factor}_raw_weight"
        raw_cols.append(raw_col)
        out[raw_col] = float(base_weight) * out[f"{factor}_reliability"].fillna(1.0).clip(lower=0.0)
    total = out[raw_cols].sum(axis=1).replace(0.0, np.nan)
    for factor in BASE_FACTOR_WEIGHTS:
        out[f"{factor}_weight"] = (out[f"{factor}_raw_weight"] / total).fillna(BASE_FACTOR_WEIGHTS[factor])
    out["variant"] = spec["variant"]
    out["horizon"] = spec.get("horizon")
    out["ablation"] = spec.get("ablation")
    return out


def cs_zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def score_ic_summary(signals: pd.DataFrame, variant: str) -> dict[str, Any]:
    score_col = "portfolio_score" if "portfolio_score" in signals.columns else "band_score"
    values = []
    for _, group in signals.groupby("trade_date"):
        data = group.dropna(subset=[score_col, "label"])
        if len(data) < 30 or data[score_col].nunique() < 2 or data["label"].nunique() < 2:
            continue
        value = data[score_col].corr(data["label"], method="spearman")
        if pd.notna(value):
            values.append(float(value))
    series = pd.Series(values, dtype=float)
    mean = float(series.mean()) if len(series) else np.nan
    std = float(series.std(ddof=1)) if len(series) > 1 else np.nan
    return {
        "variant": variant,
        "days": int(len(series)),
        "rank_ic_mean": mean,
        "rank_ic_std": std,
        "icir": float(mean / std * np.sqrt(252)) if std and np.isfinite(std) and std > 0 else np.nan,
        "positive_ratio": float((series > 0).mean()) if len(series) else np.nan,
        "alpha_decay": float(series.tail(20).mean() - series.head(20).mean()) if len(series) >= 40 else np.nan,
    }


def drawdown_analysis(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline = daily.loc[daily["variant"].eq("baseline_param068")].sort_values("trade_date").copy()
    base_start, base_trough, base_mdd = max_drawdown_window(baseline)
    for variant, group in daily.groupby("variant"):
        frame = group.sort_values("trade_date").copy()
        start, trough, mdd = max_drawdown_window(frame)
        same = frame.loc[frame["trade_date"].between(base_start, base_trough)].copy()
        same_loss = float(same["nav"].iloc[-1] / same["nav"].iloc[0] - 1.0) if len(same) > 1 else np.nan
        rows.append(
            {
                "variant": variant,
                "max_drawdown_start": start,
                "max_drawdown_trough": trough,
                "max_drawdown": mdd,
                "baseline_mdd_start": base_start,
                "baseline_mdd_trough": base_trough,
                "loss_during_baseline_mdd_window": same_loss,
                "baseline_mdd": base_mdd,
            }
        )
    return pd.DataFrame(rows).sort_values("max_drawdown")


def max_drawdown_window(frame: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    nav = pd.to_numeric(frame["nav"], errors="coerce")
    peaks = nav.cummax()
    drawdown = nav / peaks - 1.0
    trough_idx = drawdown.idxmin()
    peak_idx = nav.loc[:trough_idx].idxmax()
    return (
        pd.Timestamp(frame.loc[peak_idx, "trade_date"]),
        pd.Timestamp(frame.loc[trough_idx, "trade_date"]),
        float(drawdown.loc[trough_idx]),
    )


def build_regime_analysis(daily: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    frame = daily.merge(regime, on="trade_date", how="left")
    rows = []
    for variant, variant_frame in frame.groupby("variant"):
        for regime_col in ["market_state", "vol_state"]:
            for state, group in variant_frame.groupby(regime_col, dropna=False):
                returns = pd.to_numeric(group["return"], errors="coerce").fillna(0.0)
                total = float((1.0 + returns).prod() - 1.0)
                vol = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else np.nan
                annual = float((1.0 + total) ** (252 / max(len(group), 1)) - 1.0) if total > -1 else -1.0
                rows.append(
                    {
                        "variant": variant,
                        "regime_type": regime_col,
                        "regime": state,
                        "days": int(len(group)),
                        "total_return": total,
                        "annualized_return": annual,
                        "annualized_volatility": vol,
                        "sharpe": float(annual / vol) if vol and np.isfinite(vol) and vol > 0 else np.nan,
                        "avg_gross_exposure_ratio": float(group["gross_exposure_ratio"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def factor_weight_diagnostics(weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in weights.groupby("variant"):
        row = {"variant": variant, "days": int(len(group))}
        for factor in BASE_FACTOR_WEIGHTS:
            row[f"{factor}_weight_mean"] = float(group[f"{factor}_weight"].mean())
            row[f"{factor}_weight_min"] = float(group[f"{factor}_weight"].min())
            row[f"{factor}_weight_max"] = float(group[f"{factor}_weight"].max())
            row[f"{factor}_reliability_mean"] = float(group[f"{factor}_reliability"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(
    *,
    output: Path,
    performance: pd.DataFrame,
    alpha_quality: pd.DataFrame,
    drawdown: pd.DataFrame,
    regime_analysis: pd.DataFrame,
    weights: pd.DataFrame,
    weight_logic: pd.DataFrame,
    factor_columns: dict[str, str],
    reliability_path: Path,
    data_version: str,
) -> None:
    perf_cols = [
        "variant",
        "total_return",
        "annualized_return",
        "annualized_excess_return_vs_csi1000",
        "sharpe",
        "max_drawdown",
        "calmar",
        "annualized_turnover",
        "executed_buys",
        "trade_count",
    ]
    report_variants = [
        "baseline_param068",
        "fixed_factor_weight",
        "reliability_5d_band_only",
        "reliability_10d_band_only",
        "reliability_20d_band_only",
        "reliability_5d_all_available",
        "placebo_random_reliability_5d",
    ]
    perf_view = performance.loc[performance["variant"].isin(report_variants), perf_cols].sort_values(
        "annualized_return", ascending=False
    )
    alpha_view = alpha_quality.loc[alpha_quality["variant"].isin(report_variants)].sort_values("icir", ascending=False)
    dd_view = drawdown.loc[drawdown["variant"].isin(report_variants)].sort_values("loss_during_baseline_mdd_window")
    regime_view = regime_analysis.loc[regime_analysis["variant"].isin(report_variants)]
    weight_tail = weights.loc[weights["variant"].str.contains("reliability_5d|placebo", regex=True)].tail(20)
    lines = [
        "# Experiment 6: Reliability-aware Portfolio Backtest",
        "",
        "## Scope",
        f"- Baseline: original `param_068` account-level execution.",
        f"- Data version: `{data_version}`.",
        f"- Reliability input: `{reliability_path}`.",
        "- Production backtest and execution code were not modified.",
        "- Dynamic variants only rebuild research signal scores before the existing param_068 portfolio construction layer.",
        f"- Detected factor exposure columns: `{factor_columns}`.",
        "- Base explicit factor blend for dynamic-score variants: band/reversal/lowvol = 50%/30%/20%.",
        "- Only `band_score` has a trained reliability series in the current file; missing factor reliability defaults to 1.0.",
        "",
        "## 1. Performance Comparison",
        md_table(perf_view, 40),
        "",
        "## 2. Alpha Quality",
        md_table(alpha_view, 40),
        "",
        "## 3. Drawdown Analysis",
        "The `loss_during_baseline_mdd_window` column compares each variant over the original baseline max-drawdown window.",
        md_table(dd_view, 40),
        "",
        "## 4. Market Regime Analysis",
        md_table(regime_view, 80),
        "",
        "## 5. Factor Weight Evolution",
        md_table(weight_logic.sort_values("variant"), 40),
        "",
        "Recent reliability-driven weights:",
        md_table(weight_tail, 30),
        "",
        "## Ablations",
        "- A: `reliability_*d_band_only` adjusts only the band factor weight.",
        "- B: `reliability_*d_all_available` adjusts all factors with factor-specific reliability when available; with the current reliability file it is equivalent to A except for metadata.",
        "- C: `placebo_random_reliability_5d` tests whether random weight movement helps by accident.",
        "",
        "## Files",
        "- `portfolio_nav.csv`",
        "- `daily_factor_weights.csv`",
        "- `performance_summary.csv`",
        "- `drawdown_analysis.csv`",
        "- `regime_analysis.csv`",
        "- `factor_weight_history.csv`",
        "- `alpha_quality_icir.csv`",
        "- `monthly_breakdown.csv`",
        "- `portfolio_trades.csv`",
    ]
    (output / "reliability_portfolio_backtest_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
