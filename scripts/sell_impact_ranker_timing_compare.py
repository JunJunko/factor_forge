from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sell_impact_score_band_walkforward as wf
import sell_impact_sorting_repair as base
from factor_forge.config import load_project
from factor_forge.data import DataVersionRepository


SOURCE_RUN = Path("artifacts/strategy_reviews/sell_impact_score_band_walkforward_20260708T091419Z")
TIMING_DAILY = Path(
    "artifacts/timing_position_models/"
    "timing_position_model_v1_20260708T025521Z_181c72c6/"
    "timing_position_daily.csv"
)
OUTPUT_ROOT = Path("artifacts/strategy_reviews")
MODEL_VARIANT = "regime_aware_cluster_ranker"


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_ranker_timing_compare_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"source_run={SOURCE_RUN}")
    log(f"timing_daily={TIMING_DAILY}")
    version, panel = base.load_panel()
    market_benchmark = load_market_benchmark(version)
    dataset = pd.read_parquet(SOURCE_RUN / "walkforward_dataset.parquet")
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    multiplier = load_position_multiplier(TIMING_DAILY)
    log(
        "loaded "
        f"panel_rows={len(panel):,} dataset_rows={len(dataset):,} "
        f"timing_dates={len(multiplier):,} benchmark_rows={len(market_benchmark):,} "
        f"data_version={version}"
    )

    selected_bands = load_selected_bands(SOURCE_RUN / "band_selection.csv")
    metric_rows: list[dict] = []
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []

    for fold in wf.FOLDS:
        fold_name = fold["fold"]
        pred_path = SOURCE_RUN / f"predictions_{fold_name}_{MODEL_VARIANT}.parquet"
        pred = pd.read_parquet(pred_path)
        pred["trade_date"] = pd.to_datetime(pred["trade_date"])
        selected_band = float(selected_bands[(fold_name, MODEL_VARIANT)])
        variants = {
            "ranker_direct_top": pred,
            f"ranker_score_band_{selected_band:.2f}": wf.score_band_predictions(pred, selected_band),
        }
        log(f"{fold_name}: selected_band={selected_band:.2f} predictions={len(pred):,}")

        for selection_name, selection_pred in variants.items():
            for timing_name, timing_multiplier in [
                ("no_timing", None),
                ("timing_target_position", multiplier),
            ]:
                rows, daily, trades = run_test_backtests(
                    panel=panel,
                    dataset=dataset,
                    pred=selection_pred,
                    selection_name=selection_name,
                    timing_name=timing_name,
                    fold=fold,
                    position_multiplier=timing_multiplier,
                    market_benchmark=market_benchmark,
                )
                metric_rows.extend(rows)
                if daily:
                    daily_frames.extend(daily)
                if trades:
                    trade_frames.extend(trades)
                top5 = next(row for row in rows if int(row["top_n"]) == 5)
                log(
                    f"{fold_name} {selection_name} {timing_name} top5 "
                    f"ann={top5['annualized_return']:.2%} "
                    f"excess_csi1000={top5['annualized_excess_return_vs_csi1000']:.2%} "
                    f"mdd={top5['max_drawdown']:.2%} "
                    f"avg_entry_mult={top5['avg_entry_multiplier']:.2f}"
                )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "ranker_timing_backtest_metrics.csv", index=False, encoding="utf-8-sig")
    if daily_frames:
        pd.concat(daily_frames, ignore_index=True).to_parquet(output / "ranker_timing_daily.parquet", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_parquet(output / "ranker_timing_trades.parquet", index=False)

    summary = build_summary(metrics)
    summary.to_csv(output / "ranker_timing_summary.csv", index=False, encoding="utf-8-sig")
    write_report(output, metrics, summary)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "source_run": str(SOURCE_RUN),
                "timing_daily": str(TIMING_DAILY),
                "model_variant": MODEL_VARIANT,
                "data_version": version,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def load_position_multiplier(path: Path) -> pd.Series:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    required = {"trade_date", "target_position"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"timing file missing columns: {sorted(missing)}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    values = pd.to_numeric(frame["target_position"], errors="coerce").clip(0.0, 1.0)
    return pd.Series(values.to_numpy(), index=frame["trade_date"], name="target_position").sort_index()


def load_market_benchmark(data_version: str) -> pd.DataFrame:
    project = load_project(base.PROJECT_CONFIG)
    repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    frame = repository.load_raw_dataset(data_version, "index_daily")
    if frame is None or frame.empty:
        raise ValueError(f"index_daily not found for {data_version}")
    frame = frame.loc[frame["ts_code"].eq("000852.SH")].copy()
    if frame.empty:
        raise ValueError(f"CSI1000 000852.SH not found in index_daily for {data_version}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame.sort_values("trade_date")


def load_selected_bands(path: Path) -> dict[tuple[str, str], float]:
    frame = pd.read_csv(path)
    selected = frame.loc[frame.get("selected").eq(True)].copy()
    if selected.empty:
        raise ValueError("no selected bands found")
    return {
        (str(row.fold), str(row.variant)): float(row.band)
        for row in selected.itertuples(index=False)
    }


def run_test_backtests(
    *,
    panel: pd.DataFrame,
    dataset: pd.DataFrame,
    pred: pd.DataFrame,
    selection_name: str,
    timing_name: str,
    fold: dict,
    position_multiplier: pd.Series | None,
    market_benchmark: pd.DataFrame,
) -> tuple[list[dict], list[pd.DataFrame], list[pd.DataFrame]]:
    start, end = fold["test_start"], fold["test_end"]
    member = dataset.loc[
        dataset["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end)),
        ["trade_date", "ts_code", "condition_quantile"],
    ].copy()
    member["selection_eligible"] = True
    factor_values = pred.loc[pred["sample"].eq("test"), ["trade_date", "ts_code", "score"]].rename(
        columns={"score": "factor_value"}
    )
    panel_slice = panel.loc[panel["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    rows: list[dict] = []
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
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
            position_multiplier=position_multiplier,
            market_benchmark=market_benchmark,
        )
        csi1000_annual = float(result.metrics.get("market_index_annualized_return", np.nan))
        row = {
            "fold": fold["fold"],
            "selection": selection_name,
            "timing": timing_name,
            "top_n": top_n,
            "holding_days": base.HOLDING_DAYS,
            "cost_bps": base.COST_BPS,
            **result.metrics,
            "daily_return_sum": float(result.daily["return"].sum()),
            "largest_position_weight_max": float(result.daily["largest_position_weight"].max()),
            "avg_cash_ratio": float(result.daily["cash_ratio"].mean()),
            "csi1000_annualized_return": csi1000_annual,
            "annualized_excess_return_vs_csi1000": float(result.metrics["annualized_return"] - csi1000_annual),
            "top_stock_buy_share": stock_buy_share(result.trades),
            "top_month_return_share": month_return_share(result.daily),
            "avg_entry_multiplier": avg_entry_multiplier(result.trades, position_multiplier),
        }
        rows.append(row)

        daily = result.daily.copy()
        daily["fold"] = fold["fold"]
        daily["selection"] = selection_name
        daily["timing"] = timing_name
        daily["top_n"] = top_n
        daily_frames.append(daily)

        trades = result.trades.copy()
        if not trades.empty:
            trades["fold"] = fold["fold"]
            trades["selection"] = selection_name
            trades["timing"] = timing_name
            trades["top_n"] = top_n
            trade_frames.append(trades)
    return rows, daily_frames, trade_frames


def stock_buy_share(trades: pd.DataFrame) -> float:
    if trades.empty:
        return np.nan
    buys = trades.loc[trades["side"].eq("BUY")]
    if buys.empty or buys["gross_value"].sum() <= 0:
        return np.nan
    by_stock = buys.groupby("ts_code")["gross_value"].sum().sort_values(ascending=False)
    return float(by_stock.head(5).sum() / by_stock.sum())


def month_return_share(daily: pd.DataFrame) -> float:
    frame = daily.copy()
    frame["month"] = pd.to_datetime(frame["trade_date"]).dt.to_period("M").astype(str)
    monthly = frame.groupby("month")["return"].apply(lambda s: float((1.0 + s).prod() - 1.0))
    positive = monthly[monthly > 0]
    if positive.empty or positive.sum() <= 0:
        return np.nan
    return float(positive.max() / positive.sum())


def avg_entry_multiplier(trades: pd.DataFrame, multiplier: pd.Series | None) -> float:
    if multiplier is None:
        return 1.0
    if trades.empty:
        return np.nan
    dates = pd.to_datetime(trades.loc[trades["side"].eq("BUY"), "trade_date"])
    if dates.empty:
        return np.nan
    return float(pd.Series([multiplier.get(date, 1.0) for date in dates]).mean())


def build_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    top5 = metrics.loc[metrics["top_n"].eq(5)].copy()
    pivot_keys = ["fold", "selection"]
    base_rows = top5.loc[
        top5["timing"].eq("no_timing"),
        pivot_keys
        + [
            "annualized_return",
            "annualized_excess_return_vs_csi1000",
            "max_drawdown",
            "sharpe",
            "calmar",
        ],
    ]
    timed_rows = top5.loc[
        top5["timing"].eq("timing_target_position"),
        pivot_keys
        + [
            "annualized_return",
            "annualized_excess_return_vs_csi1000",
            "max_drawdown",
            "sharpe",
            "calmar",
            "avg_entry_multiplier",
        ],
    ]
    merged = base_rows.merge(timed_rows, on=pivot_keys, suffixes=("_no_timing", "_timing"))
    merged["ann_delta"] = merged["annualized_return_timing"] - merged["annualized_return_no_timing"]
    merged["excess_vs_csi1000_delta"] = (
        merged["annualized_excess_return_vs_csi1000_timing"]
        - merged["annualized_excess_return_vs_csi1000_no_timing"]
    )
    merged["mdd_delta"] = merged["max_drawdown_timing"] - merged["max_drawdown_no_timing"]
    return merged.sort_values(["fold", "selection"]).reset_index(drop=True)


def write_report(output: Path, metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    top5 = metrics.loc[metrics["top_n"].eq(5)].copy()
    top10 = metrics.loc[metrics["top_n"].eq(10)].copy()
    columns = [
        "fold",
        "selection",
        "timing",
        "top_n",
        "annualized_return",
        "csi1000_annualized_return",
        "annualized_excess_return_vs_csi1000",
        "sharpe",
        "calmar",
        "max_drawdown",
        "execution_rate",
        "avg_entry_multiplier",
        "avg_cash_ratio",
        "top_stock_buy_share",
        "top_month_return_share",
    ]
    lines = [
        "# Ranker Direct vs Score-Band With Timing",
        "",
        "## Setup",
        f"- Source predictions: `{SOURCE_RUN}`",
        f"- Timing file: `{TIMING_DAILY}`",
        "- OOS only: each fold uses the test year and the score-band selected by the previous validation year.",
        "- Timing overlay uses `target_position` only as a multiplier on new entry cash; existing holdings are not forced to rebalance.",
        "- Excess return columns in this report are relative to CSI1000 `000852.SH` open-to-open benchmark.",
        "",
        "## Top5 Timing Delta",
        summary.round(6).to_markdown(index=False),
        "",
        "## Top5 Detail",
        top5[columns].sort_values(["fold", "selection", "timing"]).round(6).to_markdown(index=False),
        "",
        "## Top10 Detail",
        top10[columns].sort_values(["fold", "selection", "timing"]).round(6).to_markdown(index=False),
        "",
        "## Notes",
        "- `mdd_delta` above is timing MDD minus no-timing MDD; positive means drawdown became shallower.",
        "- `avg_entry_multiplier` is averaged on executed buy dates. For 2024 it defaults to 1.0 before the timing model starts.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
