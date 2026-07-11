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

import sell_impact_frozen_validation as frozen
import sell_impact_ranker_timing_compare as timing_compare
import sell_impact_sorting_repair as base
from factor_forge.config import CostModel, ExecutionConstraints


OUTPUT_DIR = Path("artifacts/strategy_reviews/experiment11_block_bootstrap")
BLOCK_LENGTHS = [10, 20, 60]
N_BOOTSTRAP = 1000
RANDOM_SEED = 20260710
START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2026-06-30")
VARIANTS = ["baseline_alpha", "frozen_reliability_lambda005", "random_reliability_lambda005"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("loading frozen predictions and panel")
    signals = frozen.build_variant_scores(frozen.load_predictions(frozen.PREDICTIONS))
    signals = signals.loc[signals["trade_date"].between(START, END)].copy()
    version, panel = base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    market_benchmark = timing_compare.load_market_benchmark(version)
    position_multiplier = timing_compare.load_position_multiplier(timing_compare.TIMING_DAILY)
    log(
        f"signals rows={len(signals):,} dates={signals['trade_date'].nunique():,} "
        f"range={signals['trade_date'].min().date()}..{signals['trade_date'].max().date()} data={version}"
    )

    log("running frozen daily portfolio paths")
    daily_paths = build_daily_paths(signals, panel, market_benchmark, position_multiplier, log)
    daily_paths.to_csv(OUTPUT_DIR / "daily_paths.csv", index=False, encoding="utf-8-sig")

    log("computing daily alpha IC")
    daily_ic = daily_ic_series(signals)
    daily_ic.to_csv(OUTPUT_DIR / "daily_ic.csv", index=False, encoding="utf-8-sig")

    log(f"block bootstrap n={N_BOOTSTRAP} lengths={BLOCK_LENGTHS}")
    bootstrap = run_bootstrap(daily_paths, daily_ic, random_seed=RANDOM_SEED)
    bootstrap.to_csv(OUTPUT_DIR / "bootstrap_results.csv", index=False, encoding="utf-8-sig")
    placebo = bootstrap[
        [
            "block_length",
            "iteration",
            "random_delta_return",
            "random_delta_sharpe",
            "random_delta_ICIR",
            "random_delta_max_drawdown",
        ]
    ].copy()
    placebo.to_csv(OUTPUT_DIR / "placebo_bootstrap.csv", index=False, encoding="utf-8-sig")

    summary = summarize_bootstrap(bootstrap)
    summary.to_csv(OUTPUT_DIR / "bootstrap_summary.csv", index=False, encoding="utf-8-sig")

    month_contrib = top_path_month_distribution(bootstrap)
    month_contrib.to_csv(OUTPUT_DIR / "top_path_month_distribution.csv", index=False, encoding="utf-8-sig")

    write_report(OUTPUT_DIR, summary, month_contrib, daily_paths, daily_ic, version)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "experiment": "Experiment 11B: Block Bootstrap Stability Test for Reliability Ranking",
                "output_dir": str(OUTPUT_DIR),
                "data_version": version,
                "date_range": [str(START.date()), str(END.date())],
                "block_lengths": BLOCK_LENGTHS,
                "bootstrap_iterations": N_BOOTSTRAP,
                "frozen_rule": "alpha_score + 0.05 * reliability_zscore",
                "no_retraining": True,
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {OUTPUT_DIR}")


def build_daily_paths(
    signals: pd.DataFrame,
    panel: pd.DataFrame,
    market_benchmark: pd.DataFrame,
    position_multiplier: pd.Series,
    log,
) -> pd.DataFrame:
    rows = []
    panel_slice = panel.loc[panel["trade_date"].between(START, END)].copy()
    member = signals[["trade_date", "ts_code"]].drop_duplicates().copy()
    member["selection_eligible"] = True
    member["condition_quantile"] = 0
    for variant in VARIANTS:
        log(f"portfolio path variant={variant}")
        factor_values = signals[["trade_date", "ts_code", variant]].rename(columns={variant: "factor_value"})
        result = base.BacktestEngine().run(
            panel_slice,
            factor_values,
            universe="liquid",
            top_n=frozen.TOP_N,
            holding_days=frozen.HOLDING_DAYS,
            initial_cash=frozen.INITIAL_CASH,
            lot_size=frozen.LOT_SIZE,
            constraints=ExecutionConstraints(
                exclude_suspended=True,
                cannot_buy_limit_up=True,
                cannot_sell_limit_down=True,
                exclude_st=True,
                exclude_delisting_period=True,
                min_listing_days=60,
            ),
            cost_model=CostModel(commission_bps_per_side=3, slippage_bps_per_side=5, stamp_duty_bps_sell=5),
            cost_scenario_bps=frozen.COST_BPS,
            selection_membership=member,
            position_multiplier=position_multiplier,
            market_benchmark=market_benchmark,
        )
        daily = result.daily.copy()
        daily["trade_date"] = pd.to_datetime(daily["trade_date"])
        daily["variant"] = variant
        daily["return"] = pd.to_numeric(daily["return"], errors="coerce").fillna(0.0)
        rows.append(daily[["trade_date", "variant", "nav", "return", "portfolio_turnover", "holding_count"]])
    return pd.concat(rows, ignore_index=True)


def daily_ic_series(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in signals.groupby("trade_date", sort=True):
        row: dict[str, Any] = {"trade_date": date}
        for variant in VARIANTS:
            data = group.dropna(subset=[variant, "future_trade_return"])
            if len(data) < 5 or data[variant].nunique() < 2 or data["future_trade_return"].nunique() < 2:
                row[f"{variant}_ic"] = np.nan
            else:
                row[f"{variant}_ic"] = data[variant].corr(data["future_trade_return"], method="spearman")
        rows.append(row)
    return pd.DataFrame(rows)


def run_bootstrap(daily_paths: pd.DataFrame, daily_ic: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    returns = daily_paths.pivot(index="trade_date", columns="variant", values="return").sort_index()
    ic = daily_ic.set_index("trade_date").sort_index()
    common_dates = returns.index.intersection(ic.index)
    returns = returns.loc[common_dates]
    ic = ic.loc[common_dates]
    dates = pd.Index(common_dates)
    rows = []
    for block_len in BLOCK_LENGTHS:
        for iteration in range(N_BOOTSTRAP):
            sampled_pos = sample_block_positions(len(dates), block_len, rng)
            sampled_dates = dates[sampled_pos]
            r = returns.iloc[sampled_pos].reset_index(drop=True)
            i = ic.iloc[sampled_pos].reset_index(drop=True)
            baseline = path_metrics(r["baseline_alpha"], i["baseline_alpha_ic"])
            real = path_metrics(r["frozen_reliability_lambda005"], i["frozen_reliability_lambda005_ic"])
            random = path_metrics(r["random_reliability_lambda005"], i["random_reliability_lambda005_ic"])
            sampled_month_counts = pd.Series(sampled_dates).dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
            rows.append(
                {
                    "block_length": block_len,
                    "iteration": iteration,
                    "sampled_days": int(len(sampled_pos)),
                    "delta_return": real["total_return"] - baseline["total_return"],
                    "delta_sharpe": real["sharpe"] - baseline["sharpe"],
                    "delta_ICIR": real["icir"] - baseline["icir"],
                    "delta_max_drawdown": real["max_drawdown"] - baseline["max_drawdown"],
                    "random_delta_return": random["total_return"] - baseline["total_return"],
                    "random_delta_sharpe": random["sharpe"] - baseline["sharpe"],
                    "random_delta_ICIR": random["icir"] - baseline["icir"],
                    "random_delta_max_drawdown": random["max_drawdown"] - baseline["max_drawdown"],
                    "baseline_return": baseline["total_return"],
                    "reliability_return": real["total_return"],
                    "random_return": random["total_return"],
                    "baseline_sharpe": baseline["sharpe"],
                    "reliability_sharpe": real["sharpe"],
                    "random_sharpe": random["sharpe"],
                    "baseline_icir": baseline["icir"],
                    "reliability_icir": real["icir"],
                    "random_icir": random["icir"],
                    "sampled_month_counts": json.dumps(
                        {str(month): int(count) for month, count in sampled_month_counts.items()},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )
    return pd.DataFrame(rows)


def sample_block_positions(n_dates: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    positions: list[int] = []
    max_start = max(n_dates - block_len, 0)
    while len(positions) < n_dates:
        start = int(rng.integers(0, max_start + 1))
        block = list(range(start, min(start + block_len, n_dates)))
        positions.extend(block)
    return np.asarray(positions[:n_dates], dtype=int)


def path_metrics(returns: pd.Series, ic_values: pd.Series) -> dict[str, float]:
    ret = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    nav = (1.0 + ret).cumprod()
    total_return = float(nav.iloc[-1] - 1.0) if len(nav) else np.nan
    vol = float(ret.std(ddof=1) * np.sqrt(252)) if len(ret) > 1 else np.nan
    ann_return = float((1.0 + total_return) ** (252 / max(len(ret), 1)) - 1.0) if np.isfinite(total_return) and total_return > -1 else np.nan
    sharpe = float(ann_return / vol) if np.isfinite(ann_return) and np.isfinite(vol) and vol > 0 else np.nan
    max_drawdown = float((nav / nav.cummax() - 1.0).min()) if len(nav) else np.nan
    ic = pd.to_numeric(ic_values, errors="coerce").dropna()
    ic_mean = float(ic.mean()) if len(ic) else np.nan
    ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    icir = float(ic_mean / ic_std * np.sqrt(252)) if np.isfinite(ic_mean) and np.isfinite(ic_std) and ic_std > 0 else np.nan
    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "icir": icir,
    }


def summarize_bootstrap(bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_pairs = [
        ("delta_return", "return"),
        ("delta_sharpe", "sharpe"),
        ("delta_ICIR", "ICIR"),
        ("delta_max_drawdown", "max_drawdown"),
        ("random_delta_return", "random_return"),
        ("random_delta_sharpe", "random_sharpe"),
        ("random_delta_ICIR", "random_ICIR"),
        ("random_delta_max_drawdown", "random_max_drawdown"),
    ]
    for block_len, group in bootstrap.groupby("block_length"):
        row: dict[str, Any] = {
            "block_length": int(block_len),
            "iterations": int(len(group)),
            "p_real_return_gt_baseline": float(group["delta_return"].gt(0).mean()),
            "p_real_sharpe_gt_baseline": float(group["delta_sharpe"].gt(0).mean()),
            "p_real_ICIR_gt_baseline": float(group["delta_ICIR"].gt(0).mean()),
            "p_real_mdd_improves": float(group["delta_max_drawdown"].gt(0).mean()),
            "p_random_return_gt_baseline": float(group["random_delta_return"].gt(0).mean()),
            "p_random_sharpe_gt_baseline": float(group["random_delta_sharpe"].gt(0).mean()),
            "p_random_ICIR_gt_baseline": float(group["random_delta_ICIR"].gt(0).mean()),
            "p_real_sharpe_gt_random": float((group["delta_sharpe"] - group["random_delta_sharpe"]).gt(0).mean()),
            "p_real_ICIR_gt_random": float((group["delta_ICIR"] - group["random_delta_ICIR"]).gt(0).mean()),
        }
        for col, label in metric_pairs:
            q = group[col].quantile([0.05, 0.50, 0.95])
            row[f"{label}_p05"] = float(q.loc[0.05])
            row[f"{label}_p50"] = float(q.loc[0.50])
            row[f"{label}_p95"] = float(q.loc[0.95])
            row[f"{label}_mean"] = float(group[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def top_path_month_distribution(bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for block_len, group in bootstrap.groupby("block_length"):
        cutoff = group["delta_sharpe"].quantile(0.90)
        top = group.loc[group["delta_sharpe"].ge(cutoff)]
        counts: dict[str, int] = {}
        for text in top["sampled_month_counts"].fillna("{}"):
            try:
                month_counts = json.loads(str(text))
            except json.JSONDecodeError:
                month_counts = {}
            for month, count in month_counts.items():
                counts[month] = counts.get(month, 0) + int(count)
        total = sum(counts.values())
        for month, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            rows.append(
                {
                    "block_length": int(block_len),
                    "top_bucket": "top_10pct_delta_sharpe",
                    "month": month,
                    "appearance_count": int(count),
                    "appearance_share": float(count / total) if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(
    output: Path,
    summary: pd.DataFrame,
    month_contrib: pd.DataFrame,
    daily_paths: pd.DataFrame,
    daily_ic: pd.DataFrame,
    data_version: str,
) -> None:
    compact_cols = [
        "block_length",
        "iterations",
        "p_real_return_gt_baseline",
        "p_real_sharpe_gt_baseline",
        "p_real_ICIR_gt_baseline",
        "p_real_mdd_improves",
        "p_random_sharpe_gt_baseline",
        "p_random_ICIR_gt_baseline",
        "p_real_sharpe_gt_random",
        "p_real_ICIR_gt_random",
        "sharpe_p05",
        "sharpe_p50",
        "sharpe_p95",
        "ICIR_p05",
        "ICIR_p50",
        "ICIR_p95",
        "return_p05",
        "return_p50",
        "return_p95",
        "max_drawdown_p05",
        "max_drawdown_p50",
        "max_drawdown_p95",
    ]
    lines = [
        "# Experiment 11B: Block Bootstrap Stability Test",
        "",
        "## Scope",
        "- Frozen baseline: `alpha_score`.",
        "- Frozen enhanced: `alpha_score + 0.05 * reliability_zscore`.",
        "- No retraining, no parameter search.",
        "- Bootstrap method: contiguous time block resampling with replacement.",
        f"- Date range: `{START.date()}` to `{END.date()}`.",
        f"- Data version: `{data_version}`.",
        f"- Bootstrap iterations per block length: `{N_BOOTSTRAP}`.",
        "",
        "## Probability Of Outperformance",
        md_table(summary[compact_cols], 20),
        "",
        "## Placebo",
        "Random reliability is the same-date shuffled reliability distribution from the frozen validation setup.",
        md_table(
            summary[
                [
                    "block_length",
                    "p_real_sharpe_gt_baseline",
                    "p_random_sharpe_gt_baseline",
                    "p_real_sharpe_gt_random",
                    "p_real_ICIR_gt_baseline",
                    "p_random_ICIR_gt_baseline",
                    "p_real_ICIR_gt_random",
                ]
            ],
            20,
        ),
        "",
        "## Top 10% Path Month Distribution",
        md_table(month_contrib, 60),
        "",
        "## Source Path Summary",
        md_table(source_path_summary(daily_paths, daily_ic), 20),
        "",
        "## Required Answers",
        "- Reliability提升概率：看 `p_real_*_gt_baseline`。",
        "- 稳定性：看 block 10/20/60 下概率和 5%/50%/95% 分位是否一致。",
        "- 是否依赖少数行情：看 Top 10% path 的月份分布是否高度集中。",
        "- 是否优于随机：看 `p_real_*_gt_random` 与 random placebo 概率。",
        "",
        "## Files",
        "- `bootstrap_results.csv`",
        "- `bootstrap_summary.csv`",
        "- `placebo_bootstrap.csv`",
        "- `top_path_month_distribution.csv`",
        "- `daily_paths.csv`",
        "- `daily_ic.csv`",
    ]
    (output / "bootstrap_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_path_summary(daily_paths: pd.DataFrame, daily_ic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ic = daily_ic.set_index("trade_date")
    for variant, group in daily_paths.groupby("variant"):
        metrics = path_metrics(group.sort_values("trade_date")["return"], ic[f"{variant}_ic"])
        rows.append({"variant": variant, **metrics})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
