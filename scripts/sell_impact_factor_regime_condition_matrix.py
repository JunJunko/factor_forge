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

import sell_impact_low_vol_regime_experiment as low_vol
import sell_impact_score_band_walkforward as wf
import sell_impact_sorting_repair as base


SOURCE_RUN = Path("artifacts/strategy_reviews/sell_impact_score_band_walkforward_20260708T091419Z")
OUTPUT_ROOT = Path("artifacts/strategy_reviews")
MIN_DAYS = 40
MIN_YEARS = 2
TOP_NS = [5, 10]

ALPHA_FACTORS = {
    "cluster_sell_impact": "sell_impact",
    "cluster_condition_deviation": "condition_deviation",
    "cluster_price_reversal": "price_reversal",
    "cluster_liquidity": "liquidity",
    "cluster_stock_state": "stock_state",
    "stock_state_low_vol": "stock_state_low_vol",
    "cluster_industry_context": "industry_context",
    "cluster_market_context": "market_context",
}

BASE_STATE_AXES = {
    "market_ret_20": "trend",
    "market_ret_60": "trend",
    "market_breadth_20": "breadth",
    "market_vol_20": "volatility_liquidity",
    "market_xsec_vol_20": "volatility_liquidity",
    "market_turnover_chg_5_20": "volatility_liquidity",
}

TIMING_STATE_AXES = {
    "index_ret_20d": "trend",
    "index_ret_60d": "trend",
    "index_drawdown_60d": "trend",
    "index_vol_20d": "volatility_liquidity",
    "up_ratio": "breadth",
    "up_ratio_ma20": "breadth",
    "breadth_thrust": "breadth",
    "rzmre_ratio": "leverage_funding",
    "put_call_log": "option_sentiment",
    "iv_atm": "option_sentiment",
    "iv_realized_spread": "option_sentiment",
    "fut_near_basis_ann": "futures_sentiment",
    "fut_ls_log": "futures_sentiment",
    "main_net_ratio": "moneyflow",
    "pmi": "macro",
    "epu_log": "macro",
}

PANEL_SENTIMENT_AXES = {
    "limit_up_open_rate": "sentiment",
    "limit_down_open_rate": "sentiment",
    "limit_pressure": "sentiment",
}


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_factor_regime_condition_matrix_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("loading sell-impact walk-forward dataset")
    dataset = load_dataset()
    log(f"dataset rows={len(dataset):,} dates={dataset['trade_date'].nunique():,}")
    timing_path = latest_timing_dataset()
    timing = load_timing_states(timing_path)
    log(f"timing states={timing_path} rows={len(timing):,}")
    panel_sentiment = load_panel_sentiment()
    log(f"panel sentiment rows={len(panel_sentiment):,}")

    state_frame, state_axis_meta = build_state_frame(dataset, timing, panel_sentiment)
    state_frame.to_csv(output / "market_state_axes_daily.csv", index=False, encoding="utf-8-sig")
    state_axis_meta.to_csv(output / "market_state_axis_metadata.csv", index=False, encoding="utf-8-sig")
    log(f"state axes available={state_axis_meta['state_axis'].nunique()}")

    log("precomputing daily factor metrics")
    daily_factor_metrics = build_daily_factor_metrics(dataset, log)
    daily_factor_metrics.to_parquet(output / "factor_daily_metrics.parquet", index=False)
    log(f"daily factor metrics rows={len(daily_factor_metrics):,}")

    matrix_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []

    state_axes = state_axis_meta["state_axis"].tolist()
    for state_axis in state_axes:
        buckets = state_frame[["trade_date", state_axis, f"{state_axis}__bucket"]].dropna().copy()
        if buckets.empty:
            continue
        metrics = daily_factor_metrics.merge(
            buckets[["trade_date", f"{state_axis}__bucket"]],
            on="trade_date",
            how="inner",
        ).rename(columns={f"{state_axis}__bucket": "state_bucket"})
        log(f"state_axis={state_axis} metric_rows={len(metrics):,} dates={metrics['trade_date'].nunique():,}")
        for factor, factor_group in ALPHA_FACTORS.items():
            factor_metrics = metrics.loc[metrics["factor"].eq(factor)].copy()
            if factor_metrics.empty:
                continue
            for bucket, group in factor_metrics.groupby("state_bucket", sort=True):
                if group["trade_date"].nunique() < 10:
                    continue
                matrix_rows.append(aggregate_condition_metrics(group, factor, factor_group, state_axis, str(bucket)))
                yearly_rows.extend(aggregate_yearly_metrics(group, factor, factor_group, state_axis, str(bucket)))
                exposure_rows.append(aggregate_exposure_metrics(group, factor, factor_group, state_axis, str(bucket)))
            interaction_rows.append(axis_factor_interaction_summary_from_metrics(factor_metrics, factor, factor_group, state_axis))

    matrix = pd.DataFrame(matrix_rows)
    yearly = pd.DataFrame(yearly_rows)
    exposure = pd.DataFrame(exposure_rows)
    interactions = pd.DataFrame(interaction_rows)
    reliability = build_reliability_flags(matrix, yearly, exposure)
    candidates = build_interaction_candidates(matrix, reliability, interactions)

    matrix.to_csv(output / "factor_regime_condition_matrix.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "factor_regime_condition_yearly.csv", index=False, encoding="utf-8-sig")
    exposure.to_csv(output / "factor_regime_style_exposure.csv", index=False, encoding="utf-8-sig")
    interactions.to_csv(output / "factor_regime_interaction_strength.csv", index=False, encoding="utf-8-sig")
    reliability.to_csv(output / "factor_regime_reliability_flags.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(output / "factor_regime_interaction_candidates.csv", index=False, encoding="utf-8-sig")
    write_report(output, state_axis_meta, matrix, yearly, exposure, interactions, reliability, candidates)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "source_run": str(SOURCE_RUN),
                "timing_dataset": str(timing_path),
                "alpha_factors": ALPHA_FACTORS,
                "min_days": MIN_DAYS,
                "min_years": MIN_YEARS,
                "purpose": "Factor cluster x continuous market-state conditional effectiveness matrix before regime-aware modeling.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def load_dataset() -> pd.DataFrame:
    dataset = pd.read_parquet(SOURCE_RUN / "walkforward_dataset.parquet")
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    dataset = dataset.loc[dataset["ts_code"].map(low_vol.permission_eligible)].copy()
    dataset["stock_state_low_vol"] = -pd.to_numeric(dataset["volatility_20_z"], errors="coerce")
    dataset["stock_state_small_size"] = -pd.to_numeric(dataset["log_circ_mv_z"], errors="coerce")
    keep = [
        "trade_date",
        "ts_code",
        "label",
        "stock_state_small_size",
        "log_circ_mv_z",
        *BASE_STATE_AXES.keys(),
        *ALPHA_FACTORS.keys(),
    ]
    keep = [column for column in dict.fromkeys(keep) if column in dataset.columns]
    return dataset[keep].replace([np.inf, -np.inf], np.nan).dropna(subset=["label"]).reset_index(drop=True)


def latest_timing_dataset() -> Path:
    root = Path("artifacts/timing_features")
    candidates = sorted(root.glob("timing_factor_library_v1_*/timing_dataset.parquet"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No timing_factor_library_v1 timing_dataset.parquet found")
    return candidates[0]


def load_timing_states(path: Path) -> pd.DataFrame:
    timing = pd.read_parquet(path)
    timing["trade_date"] = pd.to_datetime(timing["trade_date"])
    cols = ["trade_date", *[column for column in TIMING_STATE_AXES if column in timing.columns]]
    return timing[cols].replace([np.inf, -np.inf], np.nan)


def load_panel_sentiment() -> pd.DataFrame:
    _, panel = base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel.loc[panel["ts_code"].map(low_vol.permission_eligible)].copy()
    tradable = panel["is_liquid"].fillna(False).astype(bool) if "is_liquid" in panel else pd.Series(True, index=panel.index)
    panel = panel.loc[tradable].copy()
    out = panel.groupby("trade_date", as_index=False).agg(
        limit_up_open_rate=("is_limit_up_open", lambda s: float(s.fillna(False).astype(bool).mean())),
        limit_down_open_rate=("is_limit_down_open", lambda s: float(s.fillna(False).astype(bool).mean())),
    )
    out["limit_pressure"] = out["limit_up_open_rate"] - out["limit_down_open_rate"]
    return out


def build_state_frame(dataset: pd.DataFrame, timing: pd.DataFrame, panel_sentiment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_states = dataset.groupby("trade_date", as_index=False).agg(
        **{column: (column, "mean") for column in BASE_STATE_AXES if column in dataset.columns}
    )
    state = base_states.merge(timing, on="trade_date", how="left")
    state = state.merge(panel_sentiment, on="trade_date", how="left")
    metadata_rows = []
    all_axes = {
        **{k: v for k, v in BASE_STATE_AXES.items() if k in state.columns},
        **{k: v for k, v in TIMING_STATE_AXES.items() if k in state.columns},
        **{k: v for k, v in PANEL_SENTIMENT_AXES.items() if k in state.columns},
    }
    for axis, group in all_axes.items():
        coverage = float(state[axis].notna().mean())
        unique = int(state[axis].nunique(dropna=True))
        if coverage < 0.50 or unique < 20:
            continue
        bucket_col = f"{axis}__bucket"
        state[bucket_col] = bucketize(state[axis])
        metadata_rows.append(
            {
                "state_axis": axis,
                "state_group": group,
                "coverage": coverage,
                "unique_values": unique,
                "low_days": int(state[bucket_col].eq("low").sum()),
                "mid_days": int(state[bucket_col].eq("mid").sum()),
                "high_days": int(state[bucket_col].eq("high").sum()),
            }
        )
    metadata = pd.DataFrame(metadata_rows).sort_values(["state_group", "state_axis"]).reset_index(drop=True)
    return state, metadata


def bucketize(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    out = pd.Series(pd.NA, index=values.index, dtype="object")
    valid = values.dropna()
    if valid.nunique() < 3:
        return out
    q1, q2 = valid.quantile([1 / 3, 2 / 3])
    out.loc[values.le(q1)] = "low"
    out.loc[values.gt(q1) & values.le(q2)] = "mid"
    out.loc[values.gt(q2)] = "high"
    return out


def build_daily_factor_metrics(dataset: pd.DataFrame, log) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for factor, factor_group in ALPHA_FACTORS.items():
        if factor not in dataset.columns:
            continue
        log(f"daily metrics factor={factor}")
        prev_top_codes: dict[int, set[str] | None] = {top_n: None for top_n in TOP_NS}
        for date, day in dataset.groupby("trade_date", sort=True):
            data = day.dropna(subset=[factor, "label"]).copy()
            if len(data) < 30:
                continue
            row: dict[str, Any] = {
                "trade_date": date,
                "year": int(pd.Timestamp(date).year),
                "month": pd.Timestamp(date).to_period("M").strftime("%Y-%m"),
                "factor": factor,
                "factor_group": factor_group,
                "sample_rows": int(len(data)),
                "rank_ic": np.nan,
                "top_decile_return": np.nan,
                "bottom_decile_return": np.nan,
                "decile_spread": np.nan,
                "is_monotonic": np.nan,
                "score_small_size_rank_corr": np.nan,
            }
            if data[factor].nunique() >= 2 and data["label"].nunique() >= 2:
                row["rank_ic"] = float(data[factor].corr(data["label"], method="spearman"))
                if "stock_state_small_size" in data and data["stock_state_small_size"].nunique() >= 2:
                    row["score_small_size_rank_corr"] = float(
                        data[factor].corr(data["stock_state_small_size"], method="spearman")
                    )
            if len(data) >= 50 and data[factor].nunique() >= 10:
                data["decile"] = pd.qcut(data[factor].rank(method="first"), 10, labels=False) + 1
                avg = data.groupby("decile")["label"].mean()
                row["top_decile_return"] = float(avg.get(10, np.nan))
                row["bottom_decile_return"] = float(avg.get(1, np.nan))
                row["decile_spread"] = float(avg.get(10, np.nan) - avg.get(1, np.nan))
                row["is_monotonic"] = bool(avg.is_monotonic_increasing)
            q80 = data["label"].quantile(0.80)
            q20 = data["label"].quantile(0.20)
            for top_n in TOP_NS:
                if len(data) < max(50, top_n * 5):
                    continue
                top = data.nlargest(top_n, factor)
                codes = set(top["ts_code"].astype(str))
                prev_codes = prev_top_codes[top_n]
                turnover = np.nan if prev_codes is None else 1.0 - len(codes & prev_codes) / max(len(codes), 1)
                prev_top_codes[top_n] = codes
                row[f"top{top_n}_mean_label"] = float(top["label"].mean())
                row[f"top{top_n}_positive_ratio"] = float((top["label"] > 0).mean())
                row[f"top{top_n}_hit_top20_ratio"] = float(top["label"].ge(q80).mean())
                row[f"top{top_n}_bad_bottom20_ratio"] = float(top["label"].le(q20).mean())
                row[f"top{top_n}_small_size_mean"] = (
                    float(top["stock_state_small_size"].mean()) if "stock_state_small_size" in top else np.nan
                )
                row[f"top{top_n}_microcap_share"] = (
                    float(top["stock_state_small_size"].gt(1.0).mean()) if "stock_state_small_size" in top else np.nan
                )
                row[f"top{top_n}_turnover_proxy"] = turnover
                row[f"top{top_n}_codes"] = ",".join(sorted(codes))
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_condition_metrics(
    group: pd.DataFrame,
    factor: str,
    factor_group: str,
    state_axis: str,
    bucket: str,
) -> dict[str, Any]:
    row = {
        "factor": factor,
        "factor_group": factor_group,
        "state_axis": state_axis,
        "state_bucket": bucket,
        "sample_days": int(group["trade_date"].nunique()),
        "sample_rows": int(group["sample_rows"].sum()),
        "years": int(group["year"].nunique()),
        "rank_ic_mean": float(group["rank_ic"].mean()),
        "rank_ic_std": float(group["rank_ic"].std(ddof=1)),
        "icir": (
            float(group["rank_ic"].mean() / group["rank_ic"].std(ddof=1) * np.sqrt(252))
            if group["rank_ic"].notna().sum() > 1 and group["rank_ic"].std(ddof=1) > 0
            else np.nan
        ),
        "positive_ic_ratio": float(group["rank_ic"].gt(0).mean()),
        "top_decile_mean_label": float(group["top_decile_return"].mean()),
        "bottom_decile_mean_label": float(group["bottom_decile_return"].mean()),
        "decile_spread_mean": float(group["decile_spread"].mean()),
        "positive_decile_spread_ratio": float(group["decile_spread"].gt(0).mean()),
        "monotonic_ratio": float(group["is_monotonic"].astype("boolean").fillna(False).astype(bool).mean()),
    }
    for top_n in TOP_NS:
        for name in [
            "mean_label",
            "positive_ratio",
            "hit_top20_ratio",
            "bad_bottom20_ratio",
            "small_size_mean",
            "microcap_share",
            "turnover_proxy",
        ]:
            col = f"top{top_n}_{name}"
            row[col] = float(group[col].mean()) if col in group else np.nan
    return row


def aggregate_yearly_metrics(
    group: pd.DataFrame,
    factor: str,
    factor_group: str,
    state_axis: str,
    bucket: str,
) -> list[dict[str, Any]]:
    rows = []
    for year, year_group in group.groupby("year"):
        rows.append(
            {
                "factor": factor,
                "factor_group": factor_group,
                "state_axis": state_axis,
                "state_bucket": bucket,
                "year": int(year),
                "sample_days": int(year_group["trade_date"].nunique()),
                "rank_ic_mean": float(year_group["rank_ic"].mean()),
                "positive_ic_ratio": float(year_group["rank_ic"].gt(0).mean()),
                "top5_mean_label": float(year_group["top5_mean_label"].mean()),
                "top5_bad_bottom20_ratio": float(year_group["top5_bad_bottom20_ratio"].mean()),
                "top5_microcap_share": float(year_group["top5_microcap_share"].mean()),
            }
        )
    return rows


def aggregate_exposure_metrics(
    group: pd.DataFrame,
    factor: str,
    factor_group: str,
    state_axis: str,
    bucket: str,
) -> dict[str, Any]:
    stock_counts: dict[str, int] = {}
    for codes in group.get("top5_codes", pd.Series(dtype=str)).dropna():
        for code in str(codes).split(","):
            if code:
                stock_counts[code] = stock_counts.get(code, 0) + 1
    total_selects = sum(stock_counts.values())
    top5_stock_share = sum(sorted(stock_counts.values(), reverse=True)[:5]) / total_selects if total_selects else np.nan
    monthly = group.groupby("month")["top5_mean_label"].mean()
    positive = monthly[monthly > 0]
    month_share = float(positive.max() / positive.sum()) if positive.sum() > 0 else np.nan
    return {
        "factor": factor,
        "factor_group": factor_group,
        "state_axis": state_axis,
        "state_bucket": bucket,
        "score_small_size_rank_corr": float(group["score_small_size_rank_corr"].mean()),
        "top5_microcap_share": float(group["top5_microcap_share"].mean()),
        "top5_stock_selection_share": float(top5_stock_share),
        "top_positive_month_return_share": month_share,
        "unique_top5_stocks": int(len(stock_counts)),
    }


def axis_factor_interaction_summary_from_metrics(
    metrics: pd.DataFrame,
    factor: str,
    factor_group: str,
    state_axis: str,
) -> dict[str, Any]:
    grouped = metrics.groupby("state_bucket")["rank_ic"].mean().dropna()
    values = grouped.to_dict()
    return {
        "factor": factor,
        "factor_group": factor_group,
        "state_axis": state_axis,
        "available_buckets": int(len(grouped)),
        "ic_range": float(grouped.max() - grouped.min()) if len(grouped) else np.nan,
        "ic_std_across_buckets": float(grouped.std(ddof=0)) if len(grouped) else np.nan,
        "best_bucket": max(values, key=values.get) if values else None,
        "best_bucket_rank_ic": float(grouped.max()) if len(grouped) else np.nan,
        "worst_bucket": min(values, key=values.get) if values else None,
        "worst_bucket_rank_ic": float(grouped.min()) if len(grouped) else np.nan,
    }


def conditional_summary(
    group: pd.DataFrame,
    factor: str,
    factor_group: str,
    state_axis: str,
    bucket: str,
) -> dict[str, Any]:
    daily_ic = daily_rank_ic(group, factor)
    deciles = daily_decile_spread(group, factor)
    rows = {
        "factor": factor,
        "factor_group": factor_group,
        "state_axis": state_axis,
        "state_bucket": bucket,
        "sample_days": int(group["trade_date"].nunique()),
        "sample_rows": int(len(group)),
        "years": int(group["trade_date"].dt.year.nunique()),
        "rank_ic_mean": float(daily_ic.mean()) if len(daily_ic) else np.nan,
        "rank_ic_std": float(daily_ic.std(ddof=1)) if len(daily_ic) > 1 else np.nan,
        "icir": float(daily_ic.mean() / daily_ic.std(ddof=1) * np.sqrt(252)) if len(daily_ic) > 1 and daily_ic.std(ddof=1) > 0 else np.nan,
        "positive_ic_ratio": float((daily_ic > 0).mean()) if len(daily_ic) else np.nan,
        "top_decile_mean_label": float(deciles["top_decile_return"].mean()) if not deciles.empty else np.nan,
        "bottom_decile_mean_label": float(deciles["bottom_decile_return"].mean()) if not deciles.empty else np.nan,
        "decile_spread_mean": float(deciles["decile_spread"].mean()) if not deciles.empty else np.nan,
        "positive_decile_spread_ratio": float((deciles["decile_spread"] > 0).mean()) if not deciles.empty else np.nan,
        "monotonic_ratio": float(deciles["is_monotonic"].mean()) if not deciles.empty else np.nan,
    }
    for top_n in TOP_NS:
        payoff = topn_payoff(group, factor, top_n)
        rows.update({f"top{top_n}_{key}": value for key, value in payoff.items()})
    return rows


def daily_rank_ic(frame: pd.DataFrame, factor: str) -> pd.Series:
    values = []
    for _, group in frame.groupby("trade_date"):
        data = group.dropna(subset=[factor, "label"])
        if len(data) < 30 or data[factor].nunique() < 2 or data["label"].nunique() < 2:
            continue
        value = data[factor].corr(data["label"], method="spearman")
        if pd.notna(value):
            values.append(float(value))
    return pd.Series(values, dtype=float)


def daily_decile_spread(frame: pd.DataFrame, factor: str) -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("trade_date"):
        data = group.dropna(subset=[factor, "label"]).copy()
        if len(data) < 50 or data[factor].nunique() < 10:
            continue
        data["decile"] = pd.qcut(data[factor].rank(method="first"), 10, labels=False) + 1
        avg = data.groupby("decile")["label"].mean()
        rows.append(
            {
                "trade_date": date,
                "top_decile_return": float(avg.get(10, np.nan)),
                "bottom_decile_return": float(avg.get(1, np.nan)),
                "decile_spread": float(avg.get(10, np.nan) - avg.get(1, np.nan)),
                "is_monotonic": bool(avg.is_monotonic_increasing),
            }
        )
    return pd.DataFrame(rows)


def topn_payoff(frame: pd.DataFrame, factor: str, top_n: int) -> dict[str, float]:
    rows = []
    prev_codes: set[str] | None = None
    for _, group in frame.groupby("trade_date"):
        data = group.dropna(subset=[factor, "label"]).copy()
        if len(data) < max(50, top_n * 5):
            continue
        q80 = data["label"].quantile(0.80)
        q20 = data["label"].quantile(0.20)
        top = data.nlargest(top_n, factor)
        codes = set(top["ts_code"])
        turnover = np.nan if prev_codes is None else 1.0 - len(codes & prev_codes) / max(len(codes), 1)
        prev_codes = codes
        rows.append(
            {
                "mean_label": float(top["label"].mean()),
                "positive_ratio": float((top["label"] > 0).mean()),
                "hit_top20_ratio": float(top["label"].ge(q80).mean()),
                "bad_bottom20_ratio": float(top["label"].le(q20).mean()),
                "small_size_mean": float(top["stock_state_small_size"].mean()) if "stock_state_small_size" in top else np.nan,
                "microcap_share": float(top["stock_state_small_size"].gt(1.0).mean()) if "stock_state_small_size" in top else np.nan,
                "turnover_proxy": turnover,
            }
        )
    if not rows:
        return {
            "mean_label": np.nan,
            "positive_ratio": np.nan,
            "hit_top20_ratio": np.nan,
            "bad_bottom20_ratio": np.nan,
            "small_size_mean": np.nan,
            "microcap_share": np.nan,
            "turnover_proxy": np.nan,
        }
    return pd.DataFrame(rows).mean(numeric_only=True).to_dict()


def yearly_summary(
    group: pd.DataFrame,
    factor: str,
    factor_group: str,
    state_axis: str,
    bucket: str,
) -> list[dict[str, Any]]:
    rows = []
    frame = group.copy()
    frame["year"] = frame["trade_date"].dt.year
    for year, year_group in frame.groupby("year"):
        ic = daily_rank_ic(year_group, factor)
        top5 = topn_payoff(year_group, factor, 5)
        rows.append(
            {
                "factor": factor,
                "factor_group": factor_group,
                "state_axis": state_axis,
                "state_bucket": bucket,
                "year": int(year),
                "sample_days": int(year_group["trade_date"].nunique()),
                "rank_ic_mean": float(ic.mean()) if len(ic) else np.nan,
                "positive_ic_ratio": float((ic > 0).mean()) if len(ic) else np.nan,
                "top5_mean_label": top5["mean_label"],
                "top5_bad_bottom20_ratio": top5["bad_bottom20_ratio"],
                "top5_microcap_share": top5["microcap_share"],
            }
        )
    return rows


def topn_daily_summary(
    group: pd.DataFrame,
    factor: str,
    factor_group: str,
    state_axis: str,
    bucket: str,
) -> list[dict[str, Any]]:
    rows = []
    for date, day in group.groupby("trade_date"):
        data = day.dropna(subset=[factor, "label"]).copy()
        if len(data) < 50:
            continue
        q80 = data["label"].quantile(0.80)
        q20 = data["label"].quantile(0.20)
        for top_n in TOP_NS:
            top = data.nlargest(top_n, factor)
            rows.append(
                {
                    "trade_date": date,
                    "factor": factor,
                    "factor_group": factor_group,
                    "state_axis": state_axis,
                    "state_bucket": bucket,
                    "top_n": top_n,
                    "mean_label": float(top["label"].mean()),
                    "positive_ratio": float((top["label"] > 0).mean()),
                    "hit_top20_ratio": float(top["label"].ge(q80).mean()),
                    "bad_bottom20_ratio": float(top["label"].le(q20).mean()),
                    "microcap_share": float(top["stock_state_small_size"].gt(1.0).mean()) if "stock_state_small_size" in top else np.nan,
                    "top_codes": ",".join(top["ts_code"].astype(str).tolist()),
                }
            )
    return rows


def exposure_summary(
    group: pd.DataFrame,
    factor: str,
    factor_group: str,
    state_axis: str,
    bucket: str,
) -> dict[str, Any]:
    score_size_corr = []
    top_microcap = []
    stock_counts: dict[str, int] = {}
    month_returns = []
    for date, day in group.groupby("trade_date"):
        data = day.dropna(subset=[factor, "label", "stock_state_small_size"])
        if len(data) >= 30 and data[factor].nunique() > 2:
            score_size_corr.append(float(data[factor].corr(data["stock_state_small_size"], method="spearman")))
        top = data.nlargest(5, factor)
        if not top.empty:
            top_microcap.append(float(top["stock_state_small_size"].gt(1.0).mean()))
            for code in top["ts_code"].astype(str):
                stock_counts[code] = stock_counts.get(code, 0) + 1
            month_returns.append({"month": date.to_period("M").strftime("%Y-%m"), "return": float(top["label"].mean())})
    total_selects = sum(stock_counts.values())
    top5_stock_share = sum(sorted(stock_counts.values(), reverse=True)[:5]) / total_selects if total_selects else np.nan
    month_frame = pd.DataFrame(month_returns)
    if not month_frame.empty:
        monthly = month_frame.groupby("month")["return"].mean()
        positive = monthly[monthly > 0]
        top_month_share = float(positive.max() / positive.sum()) if positive.sum() > 0 else np.nan
    else:
        top_month_share = np.nan
    return {
        "factor": factor,
        "factor_group": factor_group,
        "state_axis": state_axis,
        "state_bucket": bucket,
        "score_small_size_rank_corr": float(pd.Series(score_size_corr).mean()) if score_size_corr else np.nan,
        "top5_microcap_share": float(pd.Series(top_microcap).mean()) if top_microcap else np.nan,
        "top5_stock_selection_share": float(top5_stock_share),
        "top_positive_month_return_share": top_month_share,
        "unique_top5_stocks": int(len(stock_counts)),
    }


def axis_factor_interaction_summary(
    data: pd.DataFrame,
    factor: str,
    factor_group: str,
    state_axis: str,
) -> dict[str, Any]:
    bucket_col = f"{state_axis}__bucket"
    bucket_ic = []
    for bucket, group in data.groupby(bucket_col):
        ic = daily_rank_ic(group, factor)
        bucket_ic.append((str(bucket), float(ic.mean()) if len(ic) else np.nan))
    valid = [(bucket, value) for bucket, value in bucket_ic if np.isfinite(value)]
    values = [value for _, value in valid]
    return {
        "factor": factor,
        "factor_group": factor_group,
        "state_axis": state_axis,
        "available_buckets": len(valid),
        "ic_range": float(max(values) - min(values)) if values else np.nan,
        "ic_std_across_buckets": float(np.std(values, ddof=0)) if values else np.nan,
        "best_bucket": max(valid, key=lambda item: item[1])[0] if valid else None,
        "best_bucket_rank_ic": max(values) if values else np.nan,
        "worst_bucket": min(valid, key=lambda item: item[1])[0] if valid else None,
        "worst_bucket_rank_ic": min(values) if values else np.nan,
    }


def build_reliability_flags(matrix: pd.DataFrame, yearly: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()
    rows = []
    yearly_non_null = yearly.dropna(subset=["rank_ic_mean"]).copy()
    for row in matrix.itertuples(index=False):
        y = yearly_non_null.loc[
            yearly_non_null["factor"].eq(row.factor)
            & yearly_non_null["state_axis"].eq(row.state_axis)
            & yearly_non_null["state_bucket"].eq(row.state_bucket)
            & yearly_non_null["sample_days"].ge(10)
        ]
        exp = exposure.loc[
            exposure["factor"].eq(row.factor)
            & exposure["state_axis"].eq(row.state_axis)
            & exposure["state_bucket"].eq(row.state_bucket)
        ]
        same_sign_years = 0
        if len(y) and pd.notna(row.rank_ic_mean) and row.rank_ic_mean != 0:
            same_sign_years = int((np.sign(y["rank_ic_mean"]) == np.sign(row.rank_ic_mean)).sum())
        microcap = float(exp["top5_microcap_share"].iloc[0]) if len(exp) else np.nan
        stock_conc = float(exp["top5_stock_selection_share"].iloc[0]) if len(exp) else np.nan
        month_conc = float(exp["top_positive_month_return_share"].iloc[0]) if len(exp) else np.nan
        reliable = (
            row.sample_days >= MIN_DAYS
            and row.years >= MIN_YEARS
            and same_sign_years >= MIN_YEARS
            and pd.notna(row.rank_ic_mean)
            and pd.notna(row.decile_spread_mean)
            and np.sign(row.rank_ic_mean) == np.sign(row.decile_spread_mean)
            and abs(row.rank_ic_mean) >= 0.02
            and (pd.isna(microcap) or microcap <= 0.15)
            and (pd.isna(stock_conc) or stock_conc <= 0.25)
            and (pd.isna(month_conc) or month_conc <= 0.30)
        )
        rows.append(
            {
                "factor": row.factor,
                "factor_group": row.factor_group,
                "state_axis": row.state_axis,
                "state_bucket": row.state_bucket,
                "sample_days": row.sample_days,
                "years": row.years,
                "same_sign_years": same_sign_years,
                "rank_ic_mean": row.rank_ic_mean,
                "decile_spread_mean": row.decile_spread_mean,
                "top5_mean_label": row.top5_mean_label,
                "top5_bad_bottom20_ratio": row.top5_bad_bottom20_ratio,
                "top5_microcap_share": microcap,
                "top5_stock_selection_share": stock_conc,
                "top_positive_month_return_share": month_conc,
                "low_sample_flag": row.sample_days < MIN_DAYS or row.years < MIN_YEARS,
                "direction_unstable_flag": same_sign_years < MIN_YEARS,
                "rankic_payoff_mismatch_flag": not (
                    pd.notna(row.rank_ic_mean)
                    and pd.notna(row.decile_spread_mean)
                    and np.sign(row.rank_ic_mean) == np.sign(row.decile_spread_mean)
                ),
                "microcap_flag": pd.notna(microcap) and microcap > 0.15,
                "stock_concentration_flag": pd.notna(stock_conc) and stock_conc > 0.25,
                "month_concentration_flag": pd.notna(month_conc) and month_conc > 0.30,
                "reliable_state_dependency": bool(reliable),
            }
        )
    return pd.DataFrame(rows).sort_values(["reliable_state_dependency", "rank_ic_mean"], ascending=[False, False])


def build_interaction_candidates(matrix: pd.DataFrame, reliability: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty or reliability.empty:
        return pd.DataFrame()
    reliable = reliability.loc[reliability["reliable_state_dependency"]].copy()
    if reliable.empty:
        return reliable
    joined = reliable.merge(interactions, on=["factor", "factor_group", "state_axis"], how="left")
    joined["candidate_score"] = (
        joined["rank_ic_mean"].abs().fillna(0)
        + joined["decile_spread_mean"].abs().fillna(0)
        + joined["top5_mean_label"].fillna(0).clip(lower=0)
        + joined["ic_range"].fillna(0) * 0.5
    )
    return joined.sort_values("candidate_score", ascending=False)


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(
    output: Path,
    state_axis_meta: pd.DataFrame,
    matrix: pd.DataFrame,
    yearly: pd.DataFrame,
    exposure: pd.DataFrame,
    interactions: pd.DataFrame,
    reliability: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    top_matrix = matrix.sort_values("rank_ic_mean", key=lambda s: s.abs(), ascending=False)
    lines = [
        "# Factor x Market-State Condition Matrix",
        "",
        "## Scope",
        "- This is a diagnostic step before Regime-aware LightGBM.",
        "- Alpha side: existing sell-impact factor clusters plus `stock_state_low_vol`.",
        "- State side: continuous market states from sell-impact regime columns and timing factor library.",
        "- Buckets are low/mid/high terciles, computed point-in-time from known daily state values.",
        "",
        "## State Axes",
        md_table(state_axis_meta, 60),
        "",
        "## Reliable Interaction Candidates",
        md_table(
            candidates[
                [
                    "factor",
                    "state_axis",
                    "state_bucket",
                    "sample_days",
                    "rank_ic_mean",
                    "decile_spread_mean",
                    "top5_mean_label",
                    "top5_microcap_share",
                    "ic_range",
                    "candidate_score",
                ]
            ] if not candidates.empty else candidates,
            80,
        ),
        "",
        "## Strongest Conditional Cells",
        md_table(
            top_matrix[
                [
                    "factor",
                    "state_axis",
                    "state_bucket",
                    "sample_days",
                    "years",
                    "rank_ic_mean",
                    "icir",
                    "decile_spread_mean",
                    "top5_mean_label",
                    "top5_bad_bottom20_ratio",
                    "top5_microcap_share",
                    "top5_turnover_proxy",
                ]
            ],
            80,
        ),
        "",
        "## Reliability Flags",
        md_table(reliability, 120),
        "",
        "## Interaction Strength",
        md_table(interactions.sort_values("ic_range", ascending=False), 80),
        "",
        "## Files",
        "- `market_state_axes_daily.csv`",
        "- `market_state_axis_metadata.csv`",
        "- `factor_daily_metrics.parquet`",
        "- `factor_regime_condition_matrix.csv`",
        "- `factor_regime_condition_yearly.csv`",
        "- `factor_regime_style_exposure.csv`",
        "- `factor_regime_interaction_strength.csv`",
        "- `factor_regime_reliability_flags.csv`",
        "- `factor_regime_interaction_candidates.csv`",
    ]
    (output / "factor_regime_condition_matrix_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
