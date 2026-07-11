from __future__ import annotations

"""Experiment 13: Alpha Residual Survival Model.

Research-only holding-layer experiment.  It deliberately excludes market
direction, realised drawdown, volatility, and price-return features used by
Experiment 12.  The model answers whether the original *alpha thesis* is
still intact, not whether the next market move will be favourable.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sell_impact_holding_health as hh
import sell_impact_ranker_timing_compare as timing_compare
import sell_impact_sorting_repair as base
import sell_impact_trade_quality_optimization as tq
from factor_forge.config import CostModel, ExecutionConstraints


OUTPUT_ROOT = Path("artifacts/strategy_reviews/experiment13_alpha_residual_survival")
PRIMARY_HORIZON = 5
RANDOM_SEED = 20260710
COST_BUFFER = 0.002
RANK_DECAY_MIN = 0.05
REPLACEMENT_SURVIVAL_PCTL = 0.20
REPLACEMENT_RANK_ADVANTAGE = 0.05
FACTOR_COLS = [
    "cluster_sell_impact",
    "cluster_condition_deviation",
    "cluster_price_reversal",
    "cluster_liquidity",
    "cluster_stock_state",
    "stock_state_low_vol",
]


def main() -> None:
    args = parse_args()
    output = OUTPUT_ROOT / f"alpha_residual_survival_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    log_path = output / "run.log"

    def log(message: str) -> None:
        line = f"[{time.time() - started:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("loading the immutable Alpha reconstruction, historical reliability, and param_068 entry configuration")
    cfg = hh.load_fixed10_config()
    signals, alpha_validation = hh.load_historical_signals(args.signal_dataset, args.predictions, cfg, log)
    alpha_validation.to_csv(output / "alpha_score_reconstruction.csv", index=False, encoding="utf-8-sig")
    version, full_panel = base.load_panel()
    full_panel["trade_date"] = pd.to_datetime(full_panel["trade_date"])
    panel = full_panel.loc[
        full_panel["ts_code"].isin(signals["ts_code"].unique())
        & full_panel["trade_date"].between(pd.Timestamp("2023-01-01"), hh.TEST_END)
    ].copy()
    del full_panel
    simulation_panel = panel.loc[panel["trade_date"].between(signals["trade_date"].min(), hh.TEST_END)].copy()
    log(f"signals={len(signals):,}; panel={len(panel):,}; data_version={version}")

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

    log("replaying the frozen fixed-10-day baseline ledger")
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
    log(f"baseline entries={baseline_metrics['executed_buys']:,} ann={baseline_metrics['annualized_return']:.2%}")

    log("building alpha-residual holding states from entry versus current score, rank, reliability, and factor exposures")
    potential = hh.build_potential_holding_panel(signals, panel, cfg, log)
    potential = add_residual_features(potential, signals)
    actual = hh.actual_holding_dataset(potential, baseline_positions)
    actual = add_alpha_failure_labels(actual, signals, panel, cfg, log)
    actual["sample"] = hh.sample_labels(actual["observation_date"])
    actual = actual.loc[actual["sample"].isin(["train", "valid", "test"])].copy()
    features = feature_columns(actual)
    actual.to_csv(output / "alpha_residual_dataset.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": features}).to_csv(output / "feature_list.csv", index=False, encoding="utf-8-sig")
    log(f"actual observations={len(actual):,}; entries={actual[['entry_date','ts_code']].drop_duplicates().shape[0]:,}; features={len(features)}")

    log("fitting Logistic and shallow LightGBM residual-survival models with chronological splits")
    fitted = fit_models(actual, features, log)
    fitted["metrics"].to_csv(output / "model_metrics.csv", index=False, encoding="utf-8-sig")
    fitted["importance"].to_csv(output / "feature_importance.csv", index=False, encoding="utf-8-sig")
    fitted["calibration"].to_csv(output / "calibration.csv", index=False, encoding="utf-8-sig")
    selected = select_primary_model(fitted["metrics"])
    primary_model = str(selected["model"])
    log(f"primary={primary_model} {PRIMARY_HORIZON}d; validation ROC-AUC={selected['roc_auc']:.3f}")

    potential_predictions = predict_survival(potential, features, fitted["models"])
    primary_survival = f"alpha_survival_probability_{PRIMARY_HORIZON}d_{primary_model}"
    potential_predictions["survival_percentile_by_age"] = potential_predictions.groupby(
        ["observation_date", "holding_days"]
    )[primary_survival].rank(pct=True, method="first")
    actual_predictions = actual.merge(
        potential_predictions,
        on=["observation_date", "entry_date", "ts_code", "holding_days"],
        how="left",
    )
    actual_predictions.to_csv(output / "alpha_residual_predictions.csv", index=False, encoding="utf-8-sig")

    survival_lookup = {
        (pd.Timestamp(row.observation_date), pd.Timestamp(row.entry_date), str(row.ts_code)): (
            float(getattr(row, primary_survival)), float(row.survival_percentile_by_age)
        )
        for row in potential_predictions.dropna(subset=[primary_survival]).itertuples()
    }
    log(f"survival lookup={len(survival_lookup):,}; decisions use T-close data for T+1-open execution")

    log("running two pre-registered replacement policies; neither treats low survival as an unconditional sell")
    daily_frames = [baseline_daily]
    trade_frames = [baseline_trades]
    performance = [{"variant": "baseline_fixed10", "rule": "baseline", **baseline_metrics}]
    for variant in ("survival_priority_replace", "survival_stronger_candidate_replace"):
        daily, trades, metrics = run_replacement_backtest(
            panel=simulation_panel,
            signals=signals,
            timing=timing,
            market_benchmark=market_benchmark,
            constraints=constraints,
            cost_model=cost_model,
            cfg=cfg,
            survival_lookup=survival_lookup,
            variant=variant,
        )
        daily["variant"] = variant
        trades["variant"] = variant
        daily_frames.append(daily)
        trade_frames.append(trades)
        performance.append({"variant": variant, "rule": "replacement", **metrics})
        log(f"{variant}: ann={metrics['annualized_return']:.2%} sharpe={metrics['sharpe']:.2f} mdd={metrics['max_drawdown']:.2%}")
    portfolio_nav = pd.concat(daily_frames, ignore_index=True)
    trades = pd.concat(trade_frames, ignore_index=True)
    portfolio_results = pd.DataFrame(performance)
    portfolio_nav.to_csv(output / "portfolio_nav.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(output / "trades.csv", index=False, encoding="utf-8-sig")
    portfolio_results.to_csv(output / "portfolio_results.csv", index=False, encoding="utf-8-sig")

    survival_buckets = survival_bucket_analysis(actual_predictions, primary_survival)
    residual_decay = residual_decay_analysis(actual_predictions)
    replacements = replacement_analysis(trades, panel)
    holding_quality = holding_quality_analysis(trades)
    health_comparison = compare_to_holding_health(fitted["metrics"])
    survival_buckets.to_csv(output / "survival_bucket_analysis.csv", index=False, encoding="utf-8-sig")
    residual_decay.to_csv(output / "residual_decay_analysis.csv", index=False, encoding="utf-8-sig")
    replacements.to_csv(output / "replacement_analysis.csv", index=False, encoding="utf-8-sig")
    holding_quality.to_csv(output / "holding_quality.csv", index=False, encoding="utf-8-sig")
    health_comparison.to_csv(output / "holding_health_comparison.csv", index=False, encoding="utf-8-sig")
    write_report(output, actual, fitted["metrics"], fitted["importance"], survival_buckets, residual_decay, portfolio_results, replacements, holding_quality, health_comparison, selected)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "experiment": "Experiment 13: Alpha Residual Survival Model",
                "run_dir": str(output),
                "data_version": version,
                "split": {"train": "2024", "valid": "2025", "test": "2026H1 through 2026-06-12"},
                "primary_horizon": PRIMARY_HORIZON,
                "primary_model": primary_model,
                "selection_rule": "validation ROC-AUC only",
                "replacement_rules": {
                    "survival_priority_replace": "new candidate exists; holder is lowest age-normalized survival quintile; holding age >= frozen minimum",
                    "survival_stronger_candidate_replace": "priority rule plus new candidate raw-rank advantage >= 5 percentage points",
                },
                "no_production_code_modified": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 13: Alpha Residual Survival Model.")
    parser.add_argument("--signal-dataset", type=Path, default=hh.SIGNAL_DATASET)
    parser.add_argument("--predictions", type=Path, default=hh.SIGNAL_PREDICTIONS)
    return parser.parse_args()


def add_residual_features(potential: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    out = potential.copy()
    factors = [column for column in FACTOR_COLS if column in signals.columns]
    entry = signals.rename(columns={"trade_date": "entry_signal_date"})[["entry_signal_date", "ts_code", "band_score", *factors]].copy()
    entry = entry.rename(columns={"band_score": "entry_band_score", **{column: f"entry_{column}" for column in factors}})
    current = signals.rename(columns={"trade_date": "observation_date"})[["observation_date", "ts_code", "band_score", *factors]].copy()
    current = current.rename(columns={"band_score": "current_band_score", **{column: f"current_{column}" for column in factors}})
    out = out.merge(entry, on=["entry_signal_date", "ts_code"], how="left")
    out = out.merge(current, on=["observation_date", "ts_code"], how="left")
    key = ["entry_date", "ts_code"]
    out = out.sort_values([*key, "observation_date"]).copy()
    out["alpha_decay"] = out["current_alpha_score"] - out["entry_alpha_score"]
    out["alpha_decay_ratio"] = out["current_alpha_score"] / out["entry_alpha_score"].replace(0.0, np.nan)
    out["rank_velocity"] = out.groupby(key)["current_rank_pct"].diff()
    out["rank_acceleration"] = out.groupby(key)["rank_velocity"].diff()
    out["alpha_decay_velocity"] = out.groupby(key)["alpha_decay"].diff()
    out["reliability_decay"] = out["current_reliability"] - out["entry_reliability"]
    out["reliability_velocity"] = out.groupby(key)["current_reliability"].diff()
    drift_components = []
    for factor in factors:
        drift = (out[f"current_{factor}"] - out[f"entry_{factor}"]).abs()
        drift_components.append(drift)
    out["factor_drift_score"] = pd.concat(drift_components, axis=1).mean(axis=1) if drift_components else 0.0
    out["factor_drift_velocity"] = out.groupby(key)["factor_drift_score"].diff()
    out["band_score_drift"] = out["current_band_score"] - out["entry_band_score"]
    out["days_since_entry"] = out["holding_days"]
    return out.replace([np.inf, -np.inf], np.nan)


def add_alpha_failure_labels(frame: pd.DataFrame, signals: pd.DataFrame, panel: pd.DataFrame, cfg: dict[str, Any], log) -> pd.DataFrame:
    out = frame.copy()
    dates = pd.Index(panel["trade_date"].drop_duplicates().sort_values())
    date_index = {date: idx for idx, date in enumerate(dates)}
    maps = {
        column: {(pd.Timestamp(row.trade_date), str(row.ts_code)): getattr(row, column) for row in signals[["trade_date", "ts_code", column]].itertuples()}
        for column in ["raw_rank_pct", "raw_score", "signal_probability", "reliability_observed"]
    }
    score_dispersion = signals.groupby("trade_date")["raw_score"].std().to_dict()
    opens = {(pd.Timestamp(row.trade_date), str(row.ts_code)): row.adj_open for row in panel[["trade_date", "ts_code", "adj_open"]].itertuples()}
    labels: list[dict[str, Any]] = []
    for row in out[["observation_date", "ts_code", "current_rank_pct", "current_alpha_score", "current_reliability"]].itertuples():
        start = date_index.get(pd.Timestamp(row.observation_date))
        values: dict[str, Any] = {}
        for horizon in (3, 5, 10):
            future_dates = [dates[i] for i in range(start + 1, min(start + horizon + 1, len(dates)))] if start is not None else []
            complete = len(future_dates) == horizon
            ranks = np.array([maps["raw_rank_pct"].get((date, str(row.ts_code)), np.nan) for date in future_dates], dtype=float)
            scores = np.array([maps["raw_score"].get((date, str(row.ts_code)), np.nan) for date in future_dates], dtype=float)
            rels = np.array([maps["signal_probability"].get((date, str(row.ts_code)), np.nan) for date in future_dates], dtype=float)
            rel_seen = np.array([maps["reliability_observed"].get((date, str(row.ts_code)), 0) for date in future_dates], dtype=float)
            rank_values = ranks[np.isfinite(ranks)]
            score_values = scores[np.isfinite(scores)]
            rank_change = float(rank_values[-1] - row.current_rank_pct) if len(rank_values) and np.isfinite(row.current_rank_pct) else np.nan
            score_change = float(score_values[-1] - row.current_alpha_score) if len(score_values) and np.isfinite(row.current_alpha_score) else np.nan
            rank_failed = bool(
                len(rank_values) == horizon
                and rank_change <= -RANK_DECAY_MIN
                and np.mean(np.diff(rank_values) < 0.0) >= 0.5
            )
            dispersion = float(score_dispersion.get(pd.Timestamp(row.observation_date), np.nan))
            score_failed = bool(
                len(score_values) == horizon
                and np.isfinite(dispersion)
                and score_change <= -0.25 * dispersion
                and np.mean(np.diff(score_values) < 0.0) >= 0.5
            )
            observed_rel = rels[(rel_seen > 0) & np.isfinite(rels)]
            reliability_change = float(observed_rel[-1] - row.current_reliability) if len(observed_rel) and np.isfinite(row.current_reliability) else np.nan
            reliability_failed = bool(len(observed_rel) >= 2 and reliability_change <= -0.10 and np.mean(np.diff(observed_rel) < 0.0) >= 0.5)
            current_open = opens.get((pd.Timestamp(row.observation_date), str(row.ts_code)), np.nan)
            future_open = opens.get((pd.Timestamp(future_dates[-1]), str(row.ts_code)), np.nan) if complete else np.nan
            future_return = future_open / current_open - 1.0 if np.isfinite(current_open) and current_open > 0 and np.isfinite(future_open) else np.nan
            failure = float(rank_failed or score_failed or reliability_failed or (np.isfinite(future_return) and future_return < COST_BUFFER)) if complete else np.nan
            values.update(
                {
                    f"future_rank_change_{horizon}d": rank_change,
                    f"future_alpha_change_{horizon}d": score_change,
                    f"future_reliability_change_{horizon}d": reliability_change,
                    f"future_residual_return_{horizon}d": future_return,
                    f"future_rank_deterioration_{horizon}d": rank_failed,
                    f"future_alpha_deterioration_{horizon}d": score_failed,
                    f"future_reliability_deterioration_{horizon}d": reliability_failed,
                    f"alpha_failure_{horizon}d": failure,
                }
            )
        labels.append(values)
    out = pd.concat([out, pd.DataFrame(labels, index=out.index)], axis=1)
    log("alpha failure rates=" + ", ".join(f"{h}d:{out[f'alpha_failure_{h}d'].mean():.1%}" for h in (3, 5, 10)))
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    requested = [
        "entry_rank_pct", "current_rank_pct", "rank_decay", "rank_velocity", "rank_acceleration",
        "entry_alpha_score", "current_alpha_score", "alpha_decay", "alpha_decay_ratio", "alpha_decay_velocity",
        "entry_reliability", "current_reliability", "reliability_observed", "reliability_decay", "reliability_velocity",
        "entry_band_score", "current_band_score", "band_score_drift", "factor_drift_score", "factor_drift_velocity",
        "holding_days", "days_since_entry",
    ]
    return [column for column in requested if column in frame.columns]


def fit_models(frame: pd.DataFrame, features: list[str], log) -> dict[str, Any]:
    import lightgbm as lgb

    models: dict[tuple[str, int], Any] = {}
    rows: list[dict[str, Any]] = []
    importance: list[pd.DataFrame] = []
    calibration: list[pd.DataFrame] = []
    for horizon in (3, 5, 10):
        target = f"alpha_failure_{horizon}d"
        data = frame.dropna(subset=[target]).copy()
        train, valid, test = (data.loc[data["sample"].eq(sample)] for sample in ("train", "valid", "test"))
        log(f"h={horizon}d train={len(train):,} valid={len(valid):,} test={len(test):,} positives={train[target].mean():.1%}")
        specs = {
            "logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)),
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
                failure = model.predict_proba(part[features])[:, 1]
                survival = 1.0 - failure
                future_quality = pd.to_numeric(part[f"future_alpha_change_{horizon}d"], errors="coerce") + pd.to_numeric(part[f"future_rank_change_{horizon}d"], errors="coerce")
                rows.append(
                    {
                        "model": name, "horizon": horizon, "sample": sample_name, "rows": int(len(part)),
                        "failure_rate": float(part[target].mean()), "roc_auc": float(roc_auc_score(part[target].astype(int), failure)),
                        "pr_auc": float(average_precision_score(part[target].astype(int), failure)),
                        "brier": float(brier_score_loss(part[target].astype(int), failure)),
                        "survival_future_alpha_quality_rank_ic": float(pd.Series(survival, index=part.index).corr(future_quality, method="spearman")),
                    }
                )
                calibration.append(calibration_rows(part, target, failure, name, horizon, sample_name))
            importance.append(model_importance(model, name, horizon, features))
    return {"models": models, "metrics": pd.DataFrame(rows), "importance": pd.concat(importance, ignore_index=True), "calibration": pd.concat(calibration, ignore_index=True)}


def calibration_rows(part: pd.DataFrame, target: str, failure: np.ndarray, model: str, horizon: int, sample: str) -> pd.DataFrame:
    data = part[[target, f"future_residual_return_{horizon}d"]].copy()
    data["failure_probability"] = failure
    data["bucket"] = pd.qcut(data["failure_probability"].rank(method="first"), 5, labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"])
    return data.groupby("bucket", observed=False).agg(rows=(target, "size"), predicted_failure=("failure_probability", "mean"), actual_failure=(target, "mean"), future_return=(f"future_residual_return_{horizon}d", "mean")).reset_index().assign(model=model, horizon=horizon, sample=sample)


def model_importance(model: Any, name: str, horizon: int, features: list[str]) -> pd.DataFrame:
    if name == "logistic":
        values, kind = np.abs(model.named_steps["logisticregression"].coef_[0]), "abs_standardized_coefficient"
    else:
        values, kind = model.booster_.feature_importance(importance_type="gain"), "gain"
    return pd.DataFrame({"model": name, "horizon": horizon, "feature": features, "importance": values, "importance_type": kind}).sort_values("importance", ascending=False)


def select_primary_model(metrics: pd.DataFrame) -> dict[str, Any]:
    valid = metrics.loc[(metrics["horizon"].eq(PRIMARY_HORIZON)) & (metrics["sample"].eq("valid"))]
    return valid.sort_values(["roc_auc", "pr_auc"], ascending=False).iloc[0].to_dict()


def predict_survival(potential: pd.DataFrame, features: list[str], models: dict[tuple[str, int], Any]) -> pd.DataFrame:
    out = potential[["observation_date", "entry_date", "ts_code", "holding_days"]].copy()
    for (name, horizon), model in models.items():
        failure = model.predict_proba(potential[features])[:, 1]
        out[f"alpha_failure_probability_{horizon}d_{name}"] = failure
        out[f"alpha_survival_probability_{horizon}d_{name}"] = 1.0 - failure
    return out


def run_replacement_backtest(*, panel: pd.DataFrame, signals: pd.DataFrame, timing: pd.Series, market_benchmark: pd.DataFrame, constraints: ExecutionConstraints, cost_model: CostModel, cfg: dict[str, Any], survival_lookup: dict, variant: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data = panel.merge(signals, on=["trade_date", "ts_code"], how="left").sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    dates = list(pd.Index(data["trade_date"].unique()).sort_values())
    by_date = {date: group.set_index("ts_code") for date, group in data.groupby("trade_date")}
    marks = tq.acct.mark_price_lookup(data)
    cash, positions, trades, daily_rows, generated = tq.INITIAL_CASH, [], [], [], 0
    for date_index, date in enumerate(dates):
        today = by_date[date]
        signal_date = dates[date_index - 1] if date_index else None
        signal_frame = by_date[signal_date] if signal_date is not None else None
        sell_count = buy_count = generated_today = blocked_trade = blocked_cash = 0
        turnover = costs = 0.0
        replacement_candidate: str | None = None
        remaining = []
        for position in positions:
            if date_index - position.entry_index >= int(cfg["max_hold_days"]) and tq.acct.can_sell(today, position.ts_code, constraints):
                value = tq.acct.position_value(tq.to_account_position(position), date, marks)
                cost = tq.acct.cost(value, "SELL", cost_model, tq.COST_BPS)
                cash += value - cost; sell_count += 1; turnover += value; costs += cost
                trade = tq.trade_row(position, date, "SELL", value, cost, today, "max_hold")
                trade["holding_trade_days"] = date_index - position.entry_index
                trades.append(trade)
            else:
                remaining.append(position)
        positions = remaining
        nav_before, gross_before, _ = tq.account_value(cash, positions, date, marks)
        target_position = float(np.clip(timing.get(date, 1.0), 0.0, 1.0))
        candidate_pool = tq.entry_candidates(signal_frame, positions, cfg) if signal_frame is not None else pd.DataFrame()
        if len(positions) >= int(cfg["max_positions"]) and not candidate_pool.empty and signal_date is not None:
            eligible = []
            for position in positions:
                if date_index - position.entry_index < int(cfg["min_hold_days"]):
                    continue
                item = survival_lookup.get((pd.Timestamp(signal_date), pd.Timestamp(position.entry_date), str(position.ts_code)))
                if item is not None:
                    eligible.append((item[0], item[1], position))
            if eligible:
                survival, percentile, victim = min(eligible, key=lambda item: (item[0], item[2].ts_code))
                best = candidate_pool.iloc[0]
                holder_rank = np.nan
                if victim.ts_code in signal_frame.index:
                    holder = signal_frame.loc[victim.ts_code]
                    holder = holder.iloc[0] if isinstance(holder, pd.DataFrame) else holder
                    holder_rank = float(holder.get("raw_rank_pct", np.nan))
                replace = percentile <= REPLACEMENT_SURVIVAL_PCTL
                if variant == "survival_stronger_candidate_replace":
                    replace &= np.isfinite(holder_rank) and float(best.get("raw_rank_pct", np.nan)) >= holder_rank + REPLACEMENT_RANK_ADVANTAGE
                if replace and tq.acct.can_sell(today, victim.ts_code, constraints):
                    replacement_candidate = str(best.name)
                    value = tq.acct.position_value(tq.to_account_position(victim), date, marks)
                    cost = tq.acct.cost(value, "SELL", cost_model, tq.COST_BPS)
                    cash += value - cost; sell_count += 1; turnover += value; costs += cost
                    positions.remove(victim)
                    trade = tq.trade_row(position=victim, date=date, side="SELL", gross=value, trade_cost=cost, today=today, reason=f"{variant}_exit")
                    trade["holding_trade_days"] = date_index - victim.entry_index
                    trades.append(trade)
        nav_before, gross_before, _ = tq.account_value(cash, positions, date, marks)
        vacancies = max(0, int(cfg["max_positions"]) - len(positions))
        if signal_frame is not None and vacancies:
            candidates = tq.entry_candidates(signal_frame, positions, cfg)
            generated_today = len(candidates); generated += generated_today
            if not candidates.empty:
                budget = max(0.0, min(cash, nav_before * target_position - gross_before))
                slot_value = nav_before * target_position / int(cfg["max_positions"])
                per_buy = min(slot_value, budget / vacancies)
                for ts_code, signal_row in candidates.head(vacancies).iterrows():
                    if not tq.acct.can_buy(today, ts_code, constraints): blocked_trade += 1; continue
                    row = today.loc[ts_code]; row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
                    shares = int(per_buy // (float(row.raw_open) * tq.LOT_SIZE)) * tq.LOT_SIZE
                    if shares <= 0: blocked_cash += 1; continue
                    gross = shares * float(row.raw_open); cost = tq.acct.cost(gross, "BUY", cost_model, tq.COST_BPS)
                    if gross + cost > cash: blocked_cash += 1; continue
                    position = tq.Position(str(ts_code), shares, date, date_index, float(row.raw_open), float(row.adj_open), signal_date, float(signal_row.get("band_rank_pct", np.nan)), float(signal_row.get("raw_rank_pct", np.nan)))
                    cash -= gross + cost; buy_count += 1; turnover += gross; costs += cost; positions.append(position)
                    entry_reason = "replacement_entry" if str(ts_code) == replacement_candidate else "entry"
                    trades.append(tq.trade_row(position, date, "BUY", gross, cost, today, entry_reason))
        nav, gross, values = tq.account_value(cash, positions, date, marks)
        prior_nav = daily_rows[-1]["nav"] if daily_rows else tq.INITIAL_CASH
        daily_rows.append({"trade_date": date, "nav": nav, "gross_exposure": gross, "cash": cash, "target_position": target_position, "new_signals": generated_today, "executed_buys": buy_count, "executed_sells": sell_count, "cash_blocked_candidates": blocked_cash, "trade_blocked_candidates": blocked_trade, "holding_count": len(positions), "unique_holding_count": len({p.ts_code for p in positions}), "portfolio_turnover": turnover / prior_nav if prior_nav else 0.0, "cash_ratio": cash / nav if nav else 0.0, "gross_exposure_ratio": gross / nav if nav else 0.0, "largest_position_weight": max((v for _, v in values), default=0.0) / nav if nav else 0.0, "transaction_cost": costs})
    daily = pd.DataFrame(daily_rows)
    daily["return"] = daily.nav.pct_change().fillna(0.0)
    daily["market_index_return"] = tq.acct.market_index_returns(market_benchmark, dates)
    trade_frame = pd.DataFrame(trades)
    return daily, trade_frame, tq.metrics_from_daily(daily, trade_frame, generated)


def survival_bucket_analysis(predictions: pd.DataFrame, survival_col: str) -> pd.DataFrame:
    data = predictions.dropna(subset=[survival_col, "alpha_failure_5d"]).copy()
    data["bucket"] = pd.qcut(data[survival_col].rank(method="first"), 5, labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"])
    return data.groupby("bucket", observed=False).agg(observations=("ts_code", "size"), survival=(survival_col, "mean"), failure_rate=("alpha_failure_5d", "mean"), future_rank_change=("future_rank_change_5d", "mean"), future_alpha_change=("future_alpha_change_5d", "mean"), future_reliability_change=("future_reliability_change_5d", "mean"), future_residual_return=("future_residual_return_5d", "mean")).reset_index()


def residual_decay_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions.dropna(subset=["alpha_decay", "alpha_failure_5d"]).copy()
    data["alpha_decay_bucket"] = pd.qcut(data["alpha_decay"].rank(method="first"), 5, labels=["Q1_most_negative", "Q2", "Q3", "Q4", "Q5_positive"])
    return data.groupby("alpha_decay_bucket", observed=False).agg(observations=("ts_code", "size"), failure_rate=("alpha_failure_5d", "mean"), future_rank_change=("future_rank_change_5d", "mean"), future_alpha_change=("future_alpha_change_5d", "mean"), future_return=("future_residual_return_5d", "mean"), factor_drift=("factor_drift_score", "mean")).reset_index()


def replacement_analysis(trades: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    exits = trades.loc[trades.reason.astype(str).str.contains("replace_exit", na=False)].copy()
    if exits.empty: return pd.DataFrame()
    prices = panel[["trade_date", "ts_code", "adj_open"]].sort_values(["ts_code", "trade_date"]).copy()
    prices["future_adj_open_5d"] = prices.groupby("ts_code")["adj_open"].shift(-5)
    exits = exits.merge(prices[["trade_date", "ts_code", "adj_open", "future_adj_open_5d"]], on=["trade_date", "ts_code"], how="left")
    exits["replaced_future_5d_return"] = exits.future_adj_open_5d / exits.adj_open - 1.0
    buys = trades.loc[(trades.side.eq("BUY")) & (trades.reason.eq("replacement_entry")), ["variant", "trade_date", "ts_code"]].copy()
    buys = buys.merge(prices[["trade_date", "ts_code", "adj_open", "future_adj_open_5d"]], on=["trade_date", "ts_code"], how="left")
    buys["new_entry_future_5d_return"] = buys.future_adj_open_5d / buys.adj_open - 1.0
    new_by_date = buys.groupby(["variant", "trade_date"], as_index=False).agg(
        new_entry_count=("ts_code", "size"), new_entry_future_5d_return=("new_entry_future_5d_return", "mean")
    )
    exits = exits.merge(new_by_date, on=["variant", "trade_date"], how="left")
    return exits.groupby("variant", as_index=False).agg(
        replacement_count=("ts_code", "size"),
        replaced_future_5d_return=("replaced_future_5d_return", "mean"),
        new_entry_count=("new_entry_count", "sum"),
        new_entry_future_5d_return=("new_entry_future_5d_return", "mean"),
    )


def holding_quality_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    buys = trades.loc[trades.side.eq("BUY")].copy(); sells = trades.loc[trades.side.eq("SELL")].copy()
    pairs = buys.merge(sells, on=["variant", "ts_code", "entry_date"], suffixes=("_buy", "_sell"))
    if pairs.empty: return pd.DataFrame()
    pairs["trade_return"] = (pairs.gross_value_sell - pairs.cost_sell) / (pairs.gross_value_buy + pairs.cost_buy) - 1.0
    pairs["holding_days"] = (pd.to_datetime(pairs.trade_date_sell) - pd.to_datetime(pairs.trade_date_buy)).dt.days
    return pairs.groupby("variant", as_index=False).agg(round_trips=("ts_code", "size"), mean_holding_return=("trade_return", "mean"), loss_trade_ratio=("trade_return", lambda s: float((s <= 0).mean())), avg_holding_calendar_days=("holding_days", "mean"))


def compare_to_holding_health(metrics: pd.DataFrame) -> pd.DataFrame:
    current = metrics.loc[(metrics.horizon.eq(5)) & (metrics["sample"].eq("test")), ["model", "roc_auc", "pr_auc", "brier"]].copy(); current["source"] = "alpha_residual_survival"
    prior = Path("artifacts/strategy_reviews/experiment12_holding_health/holding_health_20260710T071304Z/model_metrics.csv")
    if prior.exists():
        health = pd.read_csv(prior); health = health.loc[(health.horizon.eq(5)) & (health["sample"].eq("test")), ["model", "roc_auc", "pr_auc", "brier"]]; health["source"] = "holding_health"
        return pd.concat([current, health], ignore_index=True)
    return current


def md_table(frame: pd.DataFrame, limit: int = 30) -> str:
    return "_empty_" if frame is None or frame.empty else frame.head(limit).round(6).to_markdown(index=False)


def write_report(output: Path, dataset: pd.DataFrame, metrics: pd.DataFrame, importance: pd.DataFrame, buckets: pd.DataFrame, decay: pd.DataFrame, portfolio: pd.DataFrame, replacements: pd.DataFrame, holding: pd.DataFrame, comparison: pd.DataFrame, selected: dict[str, Any]) -> None:
    primary = metrics.loc[metrics.horizon.eq(PRIMARY_HORIZON)]
    imp = importance.loc[(importance.horizon.eq(PRIMARY_HORIZON)) & importance.model.eq(selected["model"])].sort_values("importance", ascending=False)
    lines = [
        "# Experiment 13: Alpha Residual Survival Model", "",
        "## Scope", "- Predicts whether the original Alpha thesis survives, not stock return or market direction.", "- Features exclude market direction, price drawdown, realised return, volatility, and turnover. Inputs are entry-versus-current Alpha rank, score, Signal Reliability, factor-exposure drift, and holding age.", "- Alpha scoring is reconstructed from the immutable Experiment 8 recipe and matches 16,993 persisted candidate scores to numerical precision; no Alpha parameter is tuned.", "- All decisions use T-close observations and execute at T+1 open.", "",
        "## Dataset", f"- Holding observations: {len(dataset):,}; baseline entries: {dataset[['entry_date','ts_code']].drop_duplicates().shape[0]:,}.", "- Time split: 2024 train / 2025 validation / 2026H1 test.", "- Failure: future rank continues to deteriorate, Alpha score continues to decline, observed Signal Reliability continues to decline, or residual return does not cover 20bps.", "",
        "## Primary Model", f"- `{selected['model']}` at 5d, selected using validation ROC-AUC only: {selected['roc_auc']:.4f}.", "", "## Metrics", md_table(primary.sort_values(["sample","roc_auc"], ascending=[True, False])), "", "## Feature Importance", md_table(imp, 20), "", "## Survival Bucket Test", md_table(buckets), "", "## Residual Decay", md_table(decay), "", "## Holding Health Comparison", md_table(comparison), "", "## Portfolio Replacement Test", md_table(portfolio.sort_values("annualized_return", ascending=False)), "", "## Replacement Quality", md_table(replacements), "", "## Holding Quality", md_table(holding), "",
        "## Interpretation", "- Survival value is supported only when high-survival buckets have lower Alpha-failure rates and better future Alpha quality.", "- Replacement policies are diagnostics, not parameter searches. A policy is not promoted if it merely increases turnover or harms Sharpe/drawdown.", "- `survival_priority_replace` requires a fresh candidate and a holder in the lowest age-normalized survival quintile. `survival_stronger_candidate_replace` also requires a 5-point raw-rank advantage for the fresh candidate.", "",
        "## Files", "- `alpha_residual_dataset.csv`, `alpha_residual_predictions.csv`", "- `survival_bucket_analysis.csv`, `residual_decay_analysis.csv`", "- `portfolio_results.csv`, `replacement_analysis.csv`, `holding_quality.csv`", "- `trades.csv`, `portfolio_nav.csv`, `model_metrics.csv`, `feature_importance.csv`",
    ]
    (output / "survival_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
