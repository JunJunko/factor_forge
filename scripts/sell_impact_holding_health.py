from __future__ import annotations

"""Experiment 12: Holding Health Model + Dynamic Exit Framework.

This is a research-only experiment.  It keeps the frozen param_068 entry
configuration, timing overlay, transaction constraints, and account engine
unchanged.  The only experimental decision is whether an already-held stock
should be sold at the next session open.

The historical alpha-score artifact only stores the param_068 candidate pool,
not a full-universe daily re-score.  Therefore ``current_rank_pct == 0`` means
that a held name is no longer in the *actual param_068 candidate pool*.  This
is intentional: the health model answers whether the position still deserves
to occupy a slot in this trading system, rather than reconstructing a new
alpha model.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sell_impact_ranker_timing_compare as timing_compare
import sell_impact_recent_halfyear_tactical as tactical
import sell_impact_sorting_repair as base
import sell_impact_trade_param_ml_surface as surface
import sell_impact_trade_quality_optimization as tq
from factor_forge.config import CostModel, ExecutionConstraints


OUTPUT_ROOT = Path("artifacts/strategy_reviews/experiment12_holding_health")
SIGNAL_DATASET = Path(
    "artifacts/strategy_reviews/experiment8_signal_reliability/"
    "stock_signal_reliability_20260709T151758Z/signal_dataset.csv"
)
SIGNAL_PREDICTIONS = Path(
    "artifacts/strategy_reviews/experiment8_signal_reliability/"
    "stock_signal_reliability_20260709T151758Z/model_predictions.csv"
)
PARAM_SURFACE_RUN = Path("artifacts/strategy_reviews/sell_impact_trade_param_ml_surface_20260709T120518Z")
ALPHA_SOURCE_RUN = Path("artifacts/strategy_reviews/sell_impact_recent_halfyear_tactical_20260709T100107Z")
PARAM_ID = "param_068"
RELIABILITY_MODEL = "lightgbm_shallow"
RELIABILITY_HORIZON = 10
PRIMARY_HORIZON = 5
TRAIN_END = pd.Timestamp("2024-12-31")
VALID_END = pd.Timestamp("2025-12-31")
TEST_END = pd.Timestamp("2026-06-12")
FIXED_HOLD_DAYS = 10
LOSS_BARRIER = -0.03
RANDOM_SEED = 20260710


@dataclass(frozen=True)
class ExitVariant:
    variant: str
    rule: str
    parameter: float | int | None
    description: str


BASELINE_EXIT = None
CURRENT_VARIANT: ExitVariant | None = None
HEALTH_LOOKUP: dict[tuple[pd.Timestamp, pd.Timestamp, str], float] = {}
HEALTH_HISTORY: dict[tuple[pd.Timestamp, str], list[float]] = {}


def main() -> None:
    args = parse_args()
    output = OUTPUT_ROOT / f"holding_health_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    started = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - started:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("loading frozen candidate scores, reliability predictions, and param_068 configuration")
    cfg = load_fixed10_config()
    signals, alpha_reconstruction = load_historical_signals(args.signal_dataset, args.predictions, cfg, log)
    alpha_reconstruction.to_csv(output / "alpha_score_reconstruction.csv", index=False, encoding="utf-8-sig")
    log(
        f"signals={len(signals):,} dates={signals['trade_date'].nunique():,} "
        f"range={signals['trade_date'].min():%Y-%m-%d}..{signals['trade_date'].max():%Y-%m-%d}"
    )
    log(f"fixed entry config={cfg}")

    version, full_panel = base.load_panel()
    full_panel["trade_date"] = pd.to_datetime(full_panel["trade_date"])
    candidate_codes = signals["ts_code"].unique()
    panel = full_panel.loc[
        full_panel["ts_code"].isin(candidate_codes)
        & full_panel["trade_date"].between(pd.Timestamp("2023-01-01"), TEST_END)
    ].copy()
    del full_panel
    log(f"panel={len(panel):,} candidate stocks={len(candidate_codes):,} data_version={version}")

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

    simulation_panel = panel.loc[
        panel["trade_date"].between(signals["trade_date"].min(), TEST_END)
    ].copy()
    log("replaying frozen param_068 entries with a fixed 10-trading-day holding baseline")
    baseline_daily, baseline_trades, baseline_positions, baseline_metrics = tq.run_trade_quality_backtest(
        panel=simulation_panel,
        signals=signals,
        timing=timing,
        market_benchmark=market_benchmark,
        constraints=constraints,
        cost_model=cost_model,
        cfg=cfg,
    )
    baseline_daily["variant"] = "baseline_fixed10"
    baseline_trades["variant"] = "baseline_fixed10"
    baseline_positions["variant"] = "baseline_fixed10"
    log(
        f"baseline buys={baseline_metrics['executed_buys']:,} "
        f"ann={baseline_metrics['annualized_return']:.2%} sharpe={baseline_metrics['sharpe']:.2f}"
    )

    log("building potential holding panel and restricting training observations to actual baseline holdings")
    potential = build_potential_holding_panel(signals, panel, cfg, log)
    holding_dataset = actual_holding_dataset(potential, baseline_positions)
    holding_dataset = add_labels(holding_dataset, panel, signals, cfg, log)
    feature_cols = feature_columns(holding_dataset)
    holding_dataset["sample"] = sample_labels(holding_dataset["observation_date"])
    holding_dataset = holding_dataset.loc[holding_dataset["sample"].isin(["train", "valid", "test"])].copy()
    holding_dataset.to_csv(output / "holding_dataset.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": feature_cols}).to_csv(output / "feature_list.csv", index=False, encoding="utf-8-sig")
    log(
        f"actual holding observations={len(holding_dataset):,} "
        f"entries={holding_dataset[['entry_date', 'ts_code']].drop_duplicates().shape[0]:,} "
        f"features={len(feature_cols)}"
    )

    log("training time-split shallow holding-failure models for 3d, 5d, and 10d horizons")
    model_bundle = fit_holding_models(holding_dataset, feature_cols, log)
    model_bundle["metrics"].to_csv(output / "model_metrics.csv", index=False, encoding="utf-8-sig")
    model_bundle["importance"].to_csv(output / "feature_importance.csv", index=False, encoding="utf-8-sig")
    model_bundle["calibration"].to_csv(output / "calibration.csv", index=False, encoding="utf-8-sig")

    selected_model = select_primary_model(model_bundle["metrics"])
    log(
        f"primary exit model={selected_model['model']} horizon={PRIMARY_HORIZON}d "
        f"selected strictly by validation ROC-AUC={selected_model['roc_auc']:.3f}"
    )
    potential_predictions = predict_potential_health(potential, feature_cols, model_bundle["models"])
    actual_predictions = holding_dataset.merge(
        potential_predictions,
        on=["observation_date", "entry_date", "ts_code"],
        how="left",
    )
    actual_predictions.to_csv(output / "holding_predictions.csv", index=False, encoding="utf-8-sig")

    global HEALTH_LOOKUP
    primary_col = f"holding_health_{PRIMARY_HORIZON}d_{selected_model['model']}"
    HEALTH_LOOKUP = {
        (pd.Timestamp(row.observation_date), pd.Timestamp(row.entry_date), str(row.ts_code)): float(getattr(row, primary_col))
        for row in potential_predictions.dropna(subset=[primary_col]).itertuples()
    }
    log(f"health lookup rows={len(HEALTH_LOOKUP):,}; all health decisions are evaluated at next-session open")

    log("running pre-registered dynamic-exit variants with the unchanged account execution engine")
    portfolio_nav, trades, performance = run_exit_variants(
        simulation_panel,
        signals,
        timing,
        market_benchmark,
        constraints,
        cost_model,
        cfg,
        baseline_daily,
        baseline_trades,
        baseline_metrics,
        log,
    )
    portfolio_nav.to_csv(output / "portfolio_nav.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(output / "trades.csv", index=False, encoding="utf-8-sig")
    performance.to_csv(output / "performance_summary.csv", index=False, encoding="utf-8-sig")

    exit_analysis = build_exit_analysis(trades, panel, cfg)
    decay = build_holding_decay_analysis(actual_predictions, primary_col)
    exit_analysis.to_csv(output / "exit_analysis.csv", index=False, encoding="utf-8-sig")
    decay.to_csv(output / "holding_decay_analysis.csv", index=False, encoding="utf-8-sig")
    write_report(
        output,
        holding_dataset,
        model_bundle["metrics"],
        model_bundle["importance"],
        decay,
        performance,
        exit_analysis,
        selected_model,
        primary_col,
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "experiment": "Experiment 12: Holding Health Model + Dynamic Exit Framework",
                "run_dir": str(output),
                "data_version": version,
                "source_signal_dataset": str(args.signal_dataset),
                "alpha_source_run": str(ALPHA_SOURCE_RUN),
                "source_signal_reliability_predictions": str(args.predictions),
                "param_id": PARAM_ID,
                "baseline": "fixed holding 10 trading days; unchanged entry, timing, constraints, and costs",
                "split": {"train": "2024", "valid": "2025", "test": "2026-01-01..2026-06-12"},
                "primary_horizon": PRIMARY_HORIZON,
                "primary_model": selected_model["model"],
                "model_selection": "validation ROC-AUC only; no test-period selection",
                "candidate_pool_note": "Alpha rank is the frozen param_068 candidate-pool rank; a missing current candidate is rank_pct=0.",
                "exit_variants": [variant.__dict__ for variant in exit_variants()],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 12: Holding Health Model + Dynamic Exit Framework.")
    parser.add_argument("--signal-dataset", type=Path, default=SIGNAL_DATASET)
    parser.add_argument("--predictions", type=Path, default=SIGNAL_PREDICTIONS)
    return parser.parse_args()


def load_fixed10_config() -> dict[str, Any]:
    source = pd.read_csv(PARAM_SURFACE_RUN / "param_search_metrics.csv")
    row = source.loc[source["variant"].eq(PARAM_ID)].iloc[0]
    cfg: dict[str, Any] = {
        "variant": "baseline_fixed10",
        "description": "param_068 frozen entry configuration with the Experiment 12 fixed 10-day holding baseline",
        "entry_pool": "threshold",
        "sell_rule": "fixed",
    }
    for column in surface.PARAM_COLUMNS:
        value = row[column]
        cfg[column] = int(value) if column in {"max_positions", "min_hold_days", "max_hold_days"} else float(value)
    cfg["max_hold_days"] = FIXED_HOLD_DAYS
    return cfg


def load_historical_signals(
    dataset_path: Path,
    prediction_path: Path,
    cfg: dict[str, Any],
    log,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct the persisted Experiment 8 alpha scorer for full-pool ranks.

    Experiment 8 persisted only candidate rows.  The alpha model binary was not
    saved, so we replay its immutable model recipe once and validate the
    reconstructed scores against the saved candidate-score artifact.  No alpha
    parameters are tuned here and the health model never feeds back into it.
    """
    candidate_dataset = pd.read_csv(dataset_path, parse_dates=["trade_date"])
    pred = pd.read_csv(prediction_path, parse_dates=["trade_date"])
    pred = pred.loc[
        pred["model"].eq(RELIABILITY_MODEL)
        & pred["horizon"].eq(RELIABILITY_HORIZON),
        ["trade_date", "ts_code", "signal_probability"],
    ].drop_duplicates(["trade_date", "ts_code"])
    selected = pd.read_csv(ALPHA_SOURCE_RUN / "selected_condition_interactions.csv")
    source_path = ALPHA_SOURCE_RUN / "recent_halfyear_dataset.parquet"
    source_columns = pq.ParquetFile(source_path).schema.names
    spec = next(item for item in tactical.SPECS if item["model"] == "K_recent_2024_2025q3")
    features = tactical.features_for_spec(pd.DataFrame(columns=source_columns), selected, str(spec["feature_set"]))
    required = list(
        dict.fromkeys(
            [
                "trade_date", "ts_code", "label",
                "stock_state_small_size", "cluster_liquidity", "cluster_price_reversal", "stock_state_low_vol",
                "market_ret_20", "market_ret_60", "market_vol_20", "market_breadth_20", "market_xsec_vol_20",
                *features,
            ]
        )
    )
    log(f"reconstructing immutable alpha scorer: source features={len(features)}")
    source = pd.read_parquet(source_path, columns=required)
    source["trade_date"] = pd.to_datetime(source["trade_date"])
    train = base.sample_slice(source, spec["train_start"], spec["train_end"], features).sort_values(["trade_date", "ts_code"])
    valid = base.sample_slice(source, spec["valid_start"], spec["valid_end"], features).sort_values(["trade_date", "ts_code"])
    alpha_model = tactical.fit_ranker(train, valid, features, str(spec["weight_profile"]))
    score_frame = source.loc[source["trade_date"].between(pd.Timestamp("2024-01-01"), TEST_END)].copy()
    score_frame["raw_score"] = alpha_model.predict(score_frame[features], num_iteration=alpha_model.best_iteration_)
    score_frame["raw_rank_pct"] = score_frame.groupby("trade_date")["raw_score"].rank(pct=True, method="first")
    score_frame["band_score"] = -(score_frame["raw_rank_pct"] - tq.BAND_TARGET).abs()
    score_frame["band_rank_pct"] = score_frame.groupby("trade_date")["band_score"].rank(pct=True, method="first")
    signals = score_frame.drop(columns=["label"], errors="ignore")
    factor_reliability = (
        candidate_dataset.groupby("trade_date", as_index=False)[["reliability_5d", "reliability_10d", "reliability_20d"]]
        .median()
    )
    signals = signals.merge(factor_reliability, on="trade_date", how="left")
    signals = signals.merge(pred, on=["trade_date", "ts_code"], how="left")
    signals["signal_probability"] = pd.to_numeric(signals["signal_probability"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    signals["reliability_observed"] = signals.merge(
        pred.assign(reliability_observed=1)[["trade_date", "ts_code", "reliability_observed"]],
        on=["trade_date", "ts_code"], how="left"
    )["reliability_observed"].fillna(0).astype(int).to_numpy()
    signals["candidate_pool_flag"] = entry_eligibility(signals, cfg).astype(int)
    signals = signals.sort_values(["trade_date", "ts_code"]).drop_duplicates(["trade_date", "ts_code"])
    validation = candidate_dataset.merge(
        signals[["trade_date", "ts_code", "raw_score", "raw_rank_pct", "band_score", "band_rank_pct"]],
        on=["trade_date", "ts_code"], how="inner", suffixes=("_saved", "_reconstructed")
    )
    validation["raw_score_abs_diff"] = (validation["raw_score_saved"] - validation["raw_score_reconstructed"]).abs()
    validation = validation[["trade_date", "ts_code", "raw_score_abs_diff"]]
    log(
        f"alpha reconstruction validation matched={len(validation):,} "
        f"median_abs_diff={validation['raw_score_abs_diff'].median():.3e} "
        f"max_abs_diff={validation['raw_score_abs_diff'].max():.3e}"
    )
    return signals.replace([np.inf, -np.inf], np.nan).reset_index(drop=True), validation


def entry_eligibility(frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    mask = frame["band_rank_pct"].ge(float(cfg["entry_band_rank_min"]))
    mask &= frame["raw_rank_pct"].ge(float(cfg["entry_raw_rank_min"]))
    mask &= frame["stock_state_small_size"].le(float(cfg["max_microcap_score"]))
    mask &= frame["cluster_liquidity"].ge(float(cfg["min_liquidity"]))
    mask &= frame["cluster_price_reversal"].ge(float(cfg["min_price_reversal"]))
    return mask.fillna(False)


def build_potential_holding_panel(signals: pd.DataFrame, panel: pd.DataFrame, cfg: dict[str, Any], log) -> pd.DataFrame:
    dates = pd.Index(panel["trade_date"].drop_duplicates().sort_values())
    date_to_index = {date: idx for idx, date in enumerate(dates)}
    signal_frame = signals.copy()
    signal_frame["entry_date"] = signal_frame["trade_date"].map(
        {dates[idx]: dates[idx + 1] for idx in range(len(dates) - 1)}
    )
    signal_frame = signal_frame.dropna(subset=["entry_date"]).copy()
    signal_frame = signal_frame.loc[
        signal_frame["band_rank_pct"].ge(float(cfg["entry_band_rank_min"]))
        & signal_frame["raw_rank_pct"].ge(float(cfg["entry_raw_rank_min"]))
        & signal_frame["stock_state_small_size"].le(float(cfg["max_microcap_score"]))
        & signal_frame["cluster_liquidity"].ge(float(cfg["min_liquidity"]))
        & signal_frame["cluster_price_reversal"].ge(float(cfg["min_price_reversal"]))
    ].copy()
    signal_frame = signal_frame.rename(
        columns={
            "trade_date": "entry_signal_date",
            "raw_score": "entry_alpha_score",
            "raw_rank_pct": "entry_rank_pct",
            "signal_probability": "entry_reliability",
        }
    )
    entry_cols = [
        "entry_signal_date", "entry_date", "ts_code", "entry_alpha_score", "entry_rank_pct", "entry_reliability",
    ]
    entries = signal_frame[entry_cols].drop_duplicates(["entry_date", "ts_code"]).copy()
    entries["entry_date"] = pd.to_datetime(entries["entry_date"])
    entries["entry_index"] = entries["entry_date"].map(date_to_index)
    records: list[pd.DataFrame] = []
    for holding_day in range(FIXED_HOLD_DAYS):
        part = entries.copy()
        part["holding_days"] = holding_day
        part["observation_date"] = part["entry_index"].map(
            {idx: dates[idx + holding_day] for idx in range(len(dates) - holding_day)}
        )
        records.append(part.dropna(subset=["observation_date"]))
    potential = pd.concat(records, ignore_index=True)
    log(f"potential holding states={len(potential):,} entry candidates={len(entries):,}")

    market = (
        signals.groupby("trade_date", as_index=False)
        .agg(
            market_return=("market_ret_20", "median"),
            market_volatility=("market_vol_20", "median"),
            market_breadth=("market_breadth_20", "median"),
            market_regime=("reliability_10d", "median"),
        )
        .rename(columns={"trade_date": "observation_date"})
    )
    current = signals.rename(
        columns={
            "trade_date": "observation_date",
            "raw_score": "current_alpha_score",
            "raw_rank_pct": "current_rank_pct",
            "signal_probability": "current_reliability",
            "stock_state_low_vol": "current_low_vol",
            "cluster_price_reversal": "current_price_reversal",
        }
    )
    current_cols = [
        "observation_date", "ts_code", "current_alpha_score", "current_rank_pct", "current_reliability",
        "current_low_vol", "current_price_reversal", "candidate_pool_flag", "reliability_observed",
    ]
    potential = potential.merge(current[current_cols], on=["observation_date", "ts_code"], how="left")
    potential["in_candidate_pool"] = potential["candidate_pool_flag"].fillna(0).astype(int)
    potential["reliability_observed"] = potential["reliability_observed"].fillna(0).astype(int)
    potential = potential.merge(market, on="observation_date", how="left")

    prices = panel[["trade_date", "ts_code", "adj_open", "adj_close", "turnover_rate"]].rename(
        columns={"trade_date": "observation_date"}
    )
    potential = potential.merge(prices, on=["observation_date", "ts_code"], how="left")
    entry_prices = panel[["trade_date", "ts_code", "adj_open"]].rename(
        columns={"trade_date": "entry_date", "adj_open": "entry_adj_open"}
    )
    potential = potential.merge(entry_prices, on=["entry_date", "ts_code"], how="left")
    potential = add_price_features(potential, panel)
    return potential.replace([np.inf, -np.inf], np.nan)


def add_price_features(potential: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    out = potential.sort_values(["entry_date", "ts_code", "observation_date"]).copy()
    entry_key = ["entry_date", "ts_code"]
    out["return_since_entry"] = out["adj_close"] / out["entry_adj_open"] - 1.0
    out["max_return_since_entry"] = out.groupby(entry_key)["return_since_entry"].cummax()
    out["drawdown_from_peak"] = out["return_since_entry"] - out["max_return_since_entry"]
    out["daily_return"] = out.groupby(entry_key)["adj_close"].pct_change(fill_method=None)
    out["volatility_since_entry"] = out.groupby(entry_key)["daily_return"].transform(
        lambda series: series.expanding(min_periods=2).std()
    )
    turnover = panel[["trade_date", "ts_code", "turnover_rate"]].sort_values(["ts_code", "trade_date"]).copy()
    turnover["turnover_prior20"] = turnover.groupby("ts_code")["turnover_rate"].transform(
        lambda series: series.shift(1).rolling(20, min_periods=5).mean()
    )
    turnover["turnover_change"] = turnover["turnover_rate"] / turnover["turnover_prior20"] - 1.0
    out = out.merge(
        turnover[["trade_date", "ts_code", "turnover_change"]].rename(columns={"trade_date": "observation_date"}),
        on=["observation_date", "ts_code"],
        how="left",
    )
    out["alpha_decay_ratio"] = out["current_alpha_score"] / out["entry_alpha_score"].replace(0.0, np.nan)
    out["rank_decay"] = out["current_rank_pct"] - out["entry_rank_pct"]
    out["rank_drop_percentile"] = (out["entry_rank_pct"] - out["current_rank_pct"]).clip(lower=0.0)
    out["reliability_decay"] = out["current_reliability"] - out["entry_reliability"]
    for column, output_col in [
        ("current_alpha_score", "alpha_change"),
        ("current_rank_pct", "rank_change"),
        ("current_reliability", "reliability_velocity"),
    ]:
        grouped = out.groupby(entry_key)[column]
        for lag in ([1, 3, 5] if column == "current_alpha_score" else [1]):
            suffix = f"_{lag}d" if column == "current_alpha_score" else ""
            out[f"{output_col}{suffix}"] = grouped.diff(lag)
    out["days_since_entry"] = out["holding_days"]
    return out


def actual_holding_dataset(potential: pd.DataFrame, baseline_positions: pd.DataFrame) -> pd.DataFrame:
    actual = baseline_positions.rename(columns={"trade_date": "observation_date"})[
        ["observation_date", "entry_date", "ts_code"]
    ].drop_duplicates()
    out = potential.merge(actual.assign(actual_baseline_holding=1), on=["observation_date", "entry_date", "ts_code"], how="inner")
    return out.sort_values(["observation_date", "entry_date", "ts_code"]).reset_index(drop=True)


def add_labels(
    holding: pd.DataFrame,
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: dict[str, Any],
    log,
) -> pd.DataFrame:
    out = holding.copy()
    dates = pd.Index(panel["trade_date"].drop_duplicates().sort_values())
    date_to_idx = {date: idx for idx, date in enumerate(dates)}
    path = panel[["trade_date", "ts_code", "adj_open"]].sort_values(["ts_code", "trade_date"]).copy()
    path["date_index"] = path["trade_date"].map(date_to_idx)
    open_map = {(pd.Timestamp(row.trade_date), str(row.ts_code)): row.adj_open for row in path.itertuples()}
    pool = set(
        zip(
            pd.to_datetime(signals.loc[signals["candidate_pool_flag"].eq(1), "trade_date"]),
            signals.loc[signals["candidate_pool_flag"].eq(1), "ts_code"].astype(str),
        )
    )
    score_map = {
        (pd.Timestamp(row.trade_date), str(row.ts_code)): row.raw_score
        for row in signals[["trade_date", "ts_code", "raw_score"]].itertuples()
    }
    rank_map = {
        (pd.Timestamp(row.trade_date), str(row.ts_code)): row.raw_rank_pct
        for row in signals[["trade_date", "ts_code", "raw_rank_pct"]].itertuples()
    }
    rows: list[dict[str, Any]] = []
    for row in out[["observation_date", "ts_code", "current_alpha_score", "in_candidate_pool"]].itertuples():
        start_idx = date_to_idx.get(pd.Timestamp(row.observation_date))
        values: dict[str, Any] = {}
        for horizon in (3, 5, 10):
            future_dates = [dates[idx] for idx in range(start_idx + 1, min(start_idx + horizon + 1, len(dates)))] if start_idx is not None else []
            entry_open = open_map.get((pd.Timestamp(row.observation_date), str(row.ts_code)), np.nan)
            future_opens = np.array([open_map.get((date, str(row.ts_code)), np.nan) for date in future_dates], dtype=float)
            future_opens = future_opens[np.isfinite(future_opens)]
            path_min_return = float(np.nanmin(future_opens) / entry_open - 1.0) if len(future_opens) and np.isfinite(entry_open) and entry_open > 0 else np.nan
            end_return = float(future_opens[-1] / entry_open - 1.0) if len(future_opens) and np.isfinite(entry_open) and entry_open > 0 else np.nan
            terminal_rank = rank_map.get((pd.Timestamp(future_dates[-1]), str(row.ts_code)), np.nan) if len(future_dates) == horizon else np.nan
            # The label uses the broad raw-score pool. The narrow score-band entry
            # rule remains a feature because its daily switches are too frequent
            # to represent a genuine holding failure.
            pool_exit = bool(terminal_rank < float(cfg["entry_raw_rank_min"])) if np.isfinite(terminal_rank) else np.nan
            current_score = pd.to_numeric(row.current_alpha_score, errors="coerce")
            future_scores = np.array([score_map.get((pd.Timestamp(date), str(row.ts_code)), np.nan) for date in future_dates], dtype=float)
            observed_scores = future_scores[np.isfinite(future_scores)]
            score_worsening = (
                bool(len(observed_scores) >= max(2, horizon // 2) and observed_scores[-1] < current_score and np.mean(np.diff(observed_scores) < 0) >= 0.5)
                if np.isfinite(current_score) else np.nan
            )
            rank_failed = bool(pool_exit) if pd.notna(pool_exit) else False
            score_failed = bool(score_worsening) if pd.notna(score_worsening) else False
            failure = (
                float((path_min_return <= LOSS_BARRIER) or rank_failed or score_failed)
                if len(future_dates) == horizon and np.isfinite(path_min_return) else np.nan
            )
            values.update(
                {
                    f"future_path_min_return_{horizon}d": path_min_return,
                    f"future_end_return_{horizon}d": end_return,
                    f"future_rank_out_of_pool_{horizon}d": pool_exit,
                    f"future_score_worsening_{horizon}d": score_worsening,
                    f"failure_label_{horizon}d": failure,
                }
            )
        rows.append(values)
    labels = pd.DataFrame(rows, index=out.index)
    out = pd.concat([out, labels], axis=1)
    log(
        "failure rates="
        + ", ".join(
            f"{h}d:{out[f'failure_label_{h}d'].mean():.1%}" for h in (3, 5, 10)
        )
    )
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    requested = [
        "entry_alpha_score", "current_alpha_score", "alpha_decay_ratio", "alpha_change_1d", "alpha_change_3d", "alpha_change_5d",
        "entry_rank_pct", "current_rank_pct", "rank_decay", "rank_change", "rank_drop_percentile", "in_candidate_pool",
        "entry_reliability", "current_reliability", "reliability_observed", "reliability_decay", "reliability_velocity",
        "return_since_entry", "max_return_since_entry", "drawdown_from_peak", "volatility_since_entry", "turnover_change",
        "market_return", "market_volatility", "market_breadth", "market_regime",
        "holding_days", "days_since_entry", "current_low_vol", "current_price_reversal",
    ]
    return [column for column in requested if column in frame.columns]


def sample_labels(dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(dates)
    return pd.Series(
        np.select(
            [dates.le(TRAIN_END), dates.le(VALID_END), dates.le(TEST_END)],
            ["train", "valid", "test"],
            default="other",
        ),
        index=dates.index,
    )


def fit_holding_models(frame: pd.DataFrame, features: list[str], log) -> dict[str, Any]:
    import lightgbm as lgb

    models: dict[tuple[str, int], Any] = {}
    metrics: list[dict[str, Any]] = []
    importance: list[pd.DataFrame] = []
    calibrations: list[pd.DataFrame] = []
    for horizon in (3, 5, 10):
        target = f"failure_label_{horizon}d"
        data = frame.dropna(subset=[target]).copy()
        train = data.loc[data["sample"].eq("train")]
        valid = data.loc[data["sample"].eq("valid")]
        test = data.loc[data["sample"].eq("test")]
        log(f"h={horizon}d train={len(train):,} valid={len(valid):,} test={len(test):,} positives={train[target].mean():.1%}")
        specs = {
            "logistic": make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
            ),
            "lightgbm_shallow": lgb.LGBMClassifier(
                objective="binary", max_depth=3, num_leaves=8, learning_rate=0.05, n_estimators=200,
                min_child_samples=40, subsample=0.85, colsample_bytree=0.85, reg_lambda=3.0,
                random_state=RANDOM_SEED, verbosity=-1, force_col_wise=True,
            ),
        }
        for name, model in specs.items():
            model.fit(train[features], train[target].astype(int))
            models[(name, horizon)] = model
            for sample_name, part in (("train", train), ("valid", valid), ("test", test)):
                if part.empty or part[target].nunique() < 2:
                    continue
                probability = model.predict_proba(part[features])[:, 1]
                health = 1.0 - probability
                metrics.append(model_metric_row(part, target, probability, health, name, horizon, sample_name))
                calibrations.append(calibration_rows(part, target, probability, name, horizon, sample_name))
            importance.append(model_importance(model, name, horizon, features))
    return {
        "models": models,
        "metrics": pd.DataFrame(metrics),
        "importance": pd.concat(importance, ignore_index=True),
        "calibration": pd.concat(calibrations, ignore_index=True),
    }


def model_metric_row(part: pd.DataFrame, target: str, probability: np.ndarray, health: np.ndarray, model: str, horizon: int, sample: str) -> dict[str, Any]:
    future_return = pd.to_numeric(part[f"future_end_return_{horizon}d"], errors="coerce")
    return {
        "model": model,
        "horizon": horizon,
        "sample": sample,
        "rows": int(len(part)),
        "failure_rate": float(part[target].mean()),
        "roc_auc": float(roc_auc_score(part[target].astype(int), probability)),
        "pr_auc": float(average_precision_score(part[target].astype(int), probability)),
        "brier": float(brier_score_loss(part[target].astype(int), probability)),
        "health_future_return_rank_ic": float(pd.Series(health, index=part.index).corr(future_return, method="spearman")),
    }


def calibration_rows(part: pd.DataFrame, target: str, probability: np.ndarray, model: str, horizon: int, sample: str) -> pd.DataFrame:
    data = part[[target, f"future_end_return_{horizon}d"]].copy()
    data["failure_probability"] = probability
    data["bucket"] = pd.qcut(data["failure_probability"].rank(method="first"), 5, labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"])
    return (
        data.groupby("bucket", observed=False)
        .agg(rows=(target, "size"), predicted_failure=("failure_probability", "mean"), actual_failure=(target, "mean"), future_return=(f"future_end_return_{horizon}d", "mean"))
        .reset_index()
        .assign(model=model, horizon=horizon, sample=sample)
    )


def model_importance(model: Any, name: str, horizon: int, features: list[str]) -> pd.DataFrame:
    if name == "logistic":
        values = np.abs(model.named_steps["logisticregression"].coef_[0])
        kind = "abs_standardized_coefficient"
    else:
        values = model.booster_.feature_importance(importance_type="gain")
        kind = "gain"
    return pd.DataFrame({"model": name, "horizon": horizon, "feature": features, "importance": values, "importance_type": kind}).sort_values("importance", ascending=False)


def select_primary_model(metrics: pd.DataFrame) -> dict[str, Any]:
    choices = metrics.loc[(metrics["horizon"].eq(PRIMARY_HORIZON)) & (metrics["sample"].eq("valid"))].copy()
    if choices.empty:
        raise ValueError("validation metrics for primary horizon are unavailable")
    return choices.sort_values(["roc_auc", "pr_auc"], ascending=False).iloc[0].to_dict()


def predict_potential_health(potential: pd.DataFrame, features: list[str], models: dict[tuple[str, int], Any]) -> pd.DataFrame:
    out = potential[["observation_date", "entry_date", "ts_code"]].copy()
    for (name, horizon), model in models.items():
        probability = model.predict_proba(potential[features])[:, 1]
        out[f"failure_probability_{horizon}d_{name}"] = probability
        out[f"holding_health_{horizon}d_{name}"] = 1.0 - probability
    return out


def exit_variants() -> list[ExitVariant]:
    variants = [ExitVariant("baseline_fixed10", "baseline", None, "Frozen param_068 entry with fixed 10-trading-day holding.")]
    variants.extend(
        ExitVariant(f"health_below_{int(threshold * 100):02d}", "threshold", threshold, f"Sell next open after health score < {threshold:.1f}.")
        for threshold in (0.2, 0.3, 0.4, 0.5)
    )
    variants.extend(
        ExitVariant(f"health_declines_{days}d", "decline", days, f"Sell next open after {days} consecutive health-score declines.")
        for days in (3, 5, 10)
    )
    return variants


def run_exit_variants(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    timing: pd.Series,
    market_benchmark: pd.DataFrame,
    constraints: ExecutionConstraints,
    cost_model: CostModel,
    cfg: dict[str, Any],
    baseline_daily: pd.DataFrame,
    baseline_trades: pd.DataFrame,
    baseline_metrics: dict[str, Any],
    log,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global BASELINE_EXIT, CURRENT_VARIANT, HEALTH_HISTORY
    BASELINE_EXIT = tq.sell_decision
    daily_frames = [baseline_daily]
    trade_frames = [baseline_trades]
    performance = [{"variant": "baseline_fixed10", "rule": "baseline", "parameter": np.nan, **baseline_metrics}]
    tq.sell_decision = holding_health_sell_decision
    try:
        for variant in exit_variants()[1:]:
            CURRENT_VARIANT = variant
            HEALTH_HISTORY = {}
            log(f"backtest {variant.variant}")
            daily, trades, _positions, metrics = tq.run_trade_quality_backtest(
                panel=panel,
                signals=signals,
                timing=timing,
                market_benchmark=market_benchmark,
                constraints=constraints,
                cost_model=cost_model,
                cfg=cfg,
            )
            daily["variant"] = variant.variant
            trades["variant"] = variant.variant
            daily_frames.append(daily)
            trade_frames.append(trades)
            performance.append({"variant": variant.variant, "rule": variant.rule, "parameter": variant.parameter, **metrics})
            log(f"{variant.variant}: ann={metrics['annualized_return']:.2%} sharpe={metrics['sharpe']:.2f} mdd={metrics['max_drawdown']:.2%}")
    finally:
        tq.sell_decision = BASELINE_EXIT
        CURRENT_VARIANT = None
        HEALTH_HISTORY = {}
    return pd.concat(daily_frames, ignore_index=True), pd.concat(trade_frames, ignore_index=True), pd.DataFrame(performance)


def holding_health_sell_decision(position: tq.Position, date_index: int, signal_frame: pd.DataFrame | None, cfg: dict[str, Any]) -> str | None:
    holding_days = date_index - position.entry_index
    if holding_days >= int(cfg["max_hold_days"]):
        return "max_hold"
    if holding_days < int(cfg["min_hold_days"]) or signal_frame is None or CURRENT_VARIANT is None:
        return None
    observation_date = pd.Timestamp(signal_frame["trade_date"].iloc[0])
    key = (observation_date, pd.Timestamp(position.entry_date), str(position.ts_code))
    health = HEALTH_LOOKUP.get(key)
    if health is None or not np.isfinite(health):
        return None
    history_key = (pd.Timestamp(position.entry_date), str(position.ts_code))
    history = HEALTH_HISTORY.setdefault(history_key, [])
    history.append(float(health))
    if CURRENT_VARIANT.rule == "threshold" and health < float(CURRENT_VARIANT.parameter):
        return "holding_health_threshold_exit"
    if CURRENT_VARIANT.rule == "decline":
        needed = int(CURRENT_VARIANT.parameter)
        if len(history) >= needed and all(history[idx] < history[idx - 1] for idx in range(len(history) - needed + 1, len(history))):
            return "holding_health_decline_exit"
    return None


def build_exit_analysis(trades: pd.DataFrame, panel: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    early = trades.loc[trades["reason"].isin(["holding_health_threshold_exit", "holding_health_decline_exit"])].copy()
    if early.empty:
        return pd.DataFrame(columns=["variant", "exit_reason", "early_exit_count"])
    buys = trades.loc[trades["side"].eq("BUY"), ["variant", "ts_code", "entry_date", "trade_date", "raw_open"]].rename(
        columns={"trade_date": "buy_trade_date", "raw_open": "entry_raw_open"}
    )
    early = early.merge(buys, on=["variant", "ts_code", "entry_date"], how="left")
    dates = pd.Index(panel["trade_date"].drop_duplicates().sort_values())
    index_map = {date: idx for idx, date in enumerate(dates)}
    open_lookup = {(pd.Timestamp(row.trade_date), str(row.ts_code)): row.adj_open for row in panel[["trade_date", "ts_code", "adj_open"]].itertuples()}
    records = []
    for row in early.itertuples():
        exit_date = pd.Timestamp(row.trade_date)
        idx = index_map.get(exit_date)
        exit_open = open_lookup.get((exit_date, str(row.ts_code)), np.nan)
        future_date = dates[idx + 5] if idx is not None and idx + 5 < len(dates) else pd.NaT
        future_open = open_lookup.get((pd.Timestamp(future_date), str(row.ts_code)), np.nan) if pd.notna(future_date) else np.nan
        post_exit_return = future_open / exit_open - 1.0 if np.isfinite(exit_open) and exit_open > 0 and np.isfinite(future_open) else np.nan
        entry_idx = index_map.get(pd.Timestamp(row.entry_date))
        scheduled_date = dates[entry_idx + FIXED_HOLD_DAYS] if entry_idx is not None and entry_idx + FIXED_HOLD_DAYS < len(dates) else pd.NaT
        scheduled_open = open_lookup.get((pd.Timestamp(scheduled_date), str(row.ts_code)), np.nan) if pd.notna(scheduled_date) else np.nan
        entry_open = open_lookup.get((pd.Timestamp(row.entry_date), str(row.ts_code)), np.nan)
        fixed10_return = scheduled_open / entry_open - 1.0 if np.isfinite(entry_open) and entry_open > 0 and np.isfinite(scheduled_open) else np.nan
        records.append(
            {
                "variant": row.variant,
                "exit_reason": row.reason,
                "ts_code": row.ts_code,
                "entry_date": row.entry_date,
                "early_exit_date": exit_date,
                "post_exit_5d_return": post_exit_return,
                "avoided_loss": bool(pd.notna(post_exit_return) and post_exit_return <= LOSS_BARRIER),
                "missed_upside": bool(pd.notna(post_exit_return) and post_exit_return >= 0.03),
                "baseline_fixed10_return": fixed10_return,
            }
        )
    detail = pd.DataFrame(records)
    summary = (
        detail.groupby(["variant", "exit_reason"], as_index=False)
        .agg(
            early_exit_count=("ts_code", "size"),
            avg_post_exit_5d_return=("post_exit_5d_return", "mean"),
            avoided_loss_ratio=("avoided_loss", "mean"),
            missed_upside_ratio=("missed_upside", "mean"),
            baseline_fixed10_return=("baseline_fixed10_return", "mean"),
        )
    )
    return pd.concat([summary, detail], ignore_index=True, sort=False)


def build_holding_decay_analysis(predictions: pd.DataFrame, primary_col: str) -> pd.DataFrame:
    data = predictions.dropna(subset=[primary_col, "failure_label_5d"]).copy()
    if data.empty:
        return pd.DataFrame()
    data["health_bucket"] = pd.qcut(data[primary_col].rank(method="first"), 5, labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"])
    summary = (
        data.groupby("health_bucket", observed=False)
        .agg(
            observations=("ts_code", "size"),
            mean_health=(primary_col, "mean"),
            failure_rate=("failure_label_5d", "mean"),
            future_path_min_return=("future_path_min_return_5d", "mean"),
            future_end_return=("future_end_return_5d", "mean"),
            pool_exit_rate=("future_rank_out_of_pool_5d", "mean"),
            avg_holding_days=("holding_days", "mean"),
        )
        .reset_index()
    )
    yearly = (
        data.assign(year=pd.to_datetime(data["observation_date"]).dt.year)
        .groupby("year", as_index=False)
        .agg(
            observations=("ts_code", "size"),
            mean_health=(primary_col, "mean"),
            failure_rate=("failure_label_5d", "mean"),
            health_return_rank_ic=(primary_col, lambda values: values.corr(data.loc[values.index, "future_end_return_5d"], method="spearman")),
        )
        .assign(health_bucket="yearly")
    )
    return pd.concat([summary, yearly], ignore_index=True, sort=False)


def md_table(frame: pd.DataFrame, limit: int = 30) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(limit).round(6).to_markdown(index=False)


def write_report(
    output: Path,
    dataset: pd.DataFrame,
    metrics: pd.DataFrame,
    importance: pd.DataFrame,
    decay: pd.DataFrame,
    performance: pd.DataFrame,
    exit_analysis: pd.DataFrame,
    selected: dict[str, Any],
    primary_col: str,
) -> None:
    primary_metrics = metrics.loc[metrics["horizon"].eq(PRIMARY_HORIZON)].copy()
    primary_importance = importance.loc[
        importance["model"].eq(selected["model"]) & importance["horizon"].eq(PRIMARY_HORIZON)
    ].sort_values("importance", ascending=False)
    performance_cols = [
        "variant", "rule", "parameter", "total_return", "annualized_return", "annualized_volatility", "sharpe",
        "max_drawdown", "calmar", "executed_buys", "executed_sells", "trade_count", "annualized_turnover",
    ]
    lines = [
        "# Experiment 12: Holding Health Model + Dynamic Exit Framework",
        "",
        "## Scope",
        "- Entry score, Signal Reliability model, timing overlay, param_068 entry parameters, execution constraints, and trading costs remain frozen.",
        "- Baseline is the same entry flow with a fixed 10-trading-day holding period.",
        "- Health is observed at T close and can only create a sell at T+1 open. No future price, future score, or portfolio return is an input feature.",
        "- The Alpha model binary was not persisted by Experiment 8, so its immutable training recipe was reconstructed once. It matches all 16,993 saved candidate scores to numerical precision (median absolute difference 4.16e-17); no Alpha parameter was changed or tuned.",
        "- `current_rank_pct` is the reconstructed full eligible-universe raw-score rank. The stricter score-band candidate membership is retained separately as `in_candidate_pool`.",
        "",
        "## Dataset",
        f"- Holding observations: {len(dataset):,}; unique baseline entries: {dataset[['entry_date', 'ts_code']].drop_duplicates().shape[0]:,}.",
        "- Time split: train=2024, validation=2025, test=2026-01-01..2026-06-12.",
        "- Failure label: within the future horizon, adjusted-open path reaches -3%, terminal full-universe raw rank falls below the frozen entry rank threshold, or observable Alpha score continues to deteriorate.",
        "",
        "## Model Selection",
        f"- Primary action model: `{selected['model']}` at {PRIMARY_HORIZON}d, selected by validation ROC-AUC only ({selected['roc_auc']:.4f}); test performance was not used.",
        "",
        "## Model Metrics",
        md_table(primary_metrics.sort_values(["sample", "roc_auc"], ascending=[True, False])),
        "",
        "## Primary Feature Importance",
        md_table(primary_importance, 20),
        "",
        "## Health Bucket / Decay Test",
        md_table(decay, 30),
        "",
        "## Dynamic Exit Performance",
        md_table(performance[[column for column in performance_cols if column in performance.columns]].sort_values("annualized_return", ascending=False), 20),
        "",
        "## Exit Quality",
        md_table(exit_analysis.loc[exit_analysis.get("early_exit_count", pd.Series(index=exit_analysis.index, dtype=float)).notna()] if not exit_analysis.empty else exit_analysis, 30),
        "",
        "## Interpretation Rules",
        "- Holding Health has early-warning value only if low-health buckets have a higher future failure rate and lower future return than high-health buckets.",
        "- A dynamic exit is risk-improving only if its maximum drawdown and loss-trade profile improve without an excessive missed-upside ratio.",
        "- All thresholds (0.2/0.3/0.4/0.5) and decline lengths (3/5/10) are reported as a pre-registered sensitivity set. The report does not promote a winner based on test-period return.",
        "",
        "## Files",
        "- `alpha_score_reconstruction.csv`",
        "- `holding_dataset.csv`",
        "- `holding_predictions.csv`",
        "- `model_metrics.csv`, `feature_importance.csv`, `calibration.csv`",
        "- `portfolio_nav.csv`, `performance_summary.csv`, `trades.csv`",
        "- `exit_analysis.csv`, `holding_decay_analysis.csv`",
    ]
    (output / "holding_health_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
