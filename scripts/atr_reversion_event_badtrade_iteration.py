"""Event-triggered ATR reversion iteration with bad-trade filtering and dynamic exits.

This experiment keeps the live-trading assumptions strict:
- signal features are known at T close;
- buys and sells execute at the next available open;
- limit-up buys, limit-down sells, ST, delisting, suspended and young listings are blocked;
- universe is the permission-eligible main-board subset inside PIT liquidity Top1000.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from atr_reversion_benchmark import csi1000_open_to_open_returns
from atr_reversion_fit_quality_gate import HMM_VARIANT, PIT_RUN, SOURCE_RUN, _apply_score_direction, _yearly
from atr_reversion_fit_quality_sensitivity import _rolling_controls
from atr_reversion_permission_filtered_backtest import (
    COST,
    LOOKBACK,
    MIN_OBS,
    TOP_N,
    _permission_eligible,
    _restricted_fit_metrics,
)
from atr_reversion_pit_hmm_calibrated_backtest import _rank_states_from_validation, _tiered_weight
from atr_reversion_regime_aware_cluster_matrix import CLUSTER_COLS, _add_clusters
from atr_reversion_regime_small_backtest import INITIAL_CASH, LOT_SIZE
from atr_reversion_small_portfolio_backtest import _can_sell_at_open, _json_default, _metrics, _position_value
from atr_reversion_walk_forward import FOLDS, REBALANCE_DAYS

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository


POLICY = "event_badtrade_iteration"
DATASET_CACHE = PIT_RUN / "atr_reversion_permission_upper_shadow_pit_dataset_w20_top1000.parquet"
LABEL_CACHE = PIT_RUN / "permission_forward_outcomes_10d.parquet"
REGIME_CACHE = PIT_RUN / "permission_daily_regime_features.parquet"
OUTPUT_PREFIX = "event_badtrade_iteration"
BAD_PENALTY = 0.65
CANDIDATE_N = 20
MAX_PER_INDUSTRY = 2
BAD_PROB_BUY_CAP = 0.60
OPEN_GAP_CAP = 0.07
EARLY_EXIT_BAD_PROB = 0.62
EARLY_EXIT_MIN_HOLD_DAYS = 3
TAKE_PROFIT = 0.08
STOP_LOSS = -0.06

VARIANTS = [
    "cluster_alpha_top5",
    "cluster_alpha_payoff_gate_top5",
    "cluster_alpha_payoff_gate_dynamic_exit_top5",
    "cluster_alpha_bad_penalty_top5",
    "cluster_alpha_health_bad_penalty_top5",
    "cluster_alpha_bad_penalty_top20_secondary",
    "cluster_alpha_bad_penalty_top20_secondary_dynamic_exit",
]


def main(source_run: str = str(SOURCE_RUN)) -> None:
    source = Path(source_run)
    output = PIT_RUN / f"{OUTPUT_PREFIX}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    version, panel = _load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["permission_eligible"] = panel["ts_code"].map(_permission_eligible).astype(bool)
    pit = pd.read_parquet(PIT_RUN / "pit_liquidity_flags.parquet")
    pit["trade_date"] = pd.to_datetime(pit["trade_date"])
    pit = pit.copy()
    pit.loc[~pit["ts_code"].map(_permission_eligible), "pit_top1000"] = False

    dataset = _load_model_dataset(log)
    outcomes = _load_or_build_outcomes(panel, pit, log)
    regime = _load_regime(log)
    model_dataset = _prepare_event_dataset(dataset, outcomes, log)
    model_dataset.to_parquet(output / "event_model_dataset.parquet", index=False)
    log(f"prepared model dataset rows={len(model_dataset):,} event_rows={int(model_dataset['event_trigger'].sum()):,}")

    rows: list[dict] = []
    yearly_frames: list[pd.DataFrame] = []
    train_rows: list[dict] = []
    gate_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []

    for fold in FOLDS:
        fold_name = fold["name"]
        fold_dir = output / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        log(f"{fold_name}: train alpha + bad-trade models")
        pred, train_summary = _train_predict(model_dataset, fold, log)
        pred.to_parquet(fold_dir / "predictions_event_alpha_bad.parquet", index=False)
        train_summary["fold"] = fold_name
        train_rows.append(train_summary)

        states = pd.read_csv(source / fold_name / HMM_VARIANT / "hmm_daily_states.csv")
        states["trade_date"] = pd.to_datetime(states["trade_date"])
        fit_daily = _restricted_fit_metrics(panel, pit, _pred_for_fit(pred), fold, log)
        controls = _rolling_controls(states["trade_date"], fit_daily, LOOKBACK, MIN_OBS)
        controls["fold"] = fold_name
        controls["lookback"] = LOOKBACK
        controls["min_obs"] = MIN_OBS
        controls["cost_bps"] = COST
        controls["policy"] = POLICY
        controls = _attach_market_payoff_gate(controls, regime)
        controls.to_csv(fold_dir / "fit_quality_controls.csv", index=False, encoding="utf-8-sig")
        gate_frames.append(controls)

        adjusted = _apply_score_direction(_pred_for_fit(pred), controls)
        pred = pred.drop(columns=["factor_value"], errors="ignore").merge(
            adjusted[["trade_date", "ts_code", "factor_value"]],
            on=["trade_date", "ts_code"],
            how="inner",
        )
        pred["score_alpha_only"] = pred["factor_value"]
        pred["score_bad_penalty"] = _daily_rank_score(pred, "factor_value", "bad_prob")
        pred = _attach_health_aware_score(pred, controls)
        pred.to_parquet(fold_dir / "predictions_adjusted_scored.parquet", index=False)

        panel_bt = panel[
            panel["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["test_end"]))
        ].merge(pit, on=["trade_date", "ts_code"], how="left")
        panel_bt["pit_top1000"] = panel_bt["pit_top1000"].fillna(False).astype(bool)
        panel_bt.loc[~panel_bt["permission_eligible"], "pit_top1000"] = False
        valid_panel = panel_bt[
            panel_bt["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["valid_end"]))
        ]
        test_panel = panel_bt[
            panel_bt["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))
        ]
        valid_pred = pred[pred["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["valid_end"]))]
        states_ext = states.merge(
            controls[["trade_date", "strategy_gate", "payoff_gate"]],
            on="trade_date",
            how="left",
        )
        states_ext["strategy_gate"] = states_ext["strategy_gate"].fillna(1.0)
        states_ext["payoff_gate"] = states_ext["payoff_gate"].fillna(1.0)

        valid_daily, _, _ = _run_event_backtest(
            valid_panel,
            valid_pred,
            states_ext,
            score_col="score_bad_penalty",
            secondary_filter=True,
            dynamic_exit=False,
            policy=lambda row: float(row.get("strategy_gate", 1.0)),
        )
        ranks, state_perf = _rank_states_from_validation(valid_daily, states_ext)
        state_perf.to_csv(fold_dir / "state_validation_perf_event_cost20.csv", index=False, encoding="utf-8-sig")

        test_pred = pred[pred["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))]
        for variant in VARIANTS:
            if variant == "cluster_alpha_top5":
                score_col = "score_alpha_only"
            elif variant in {"cluster_alpha_payoff_gate_top5", "cluster_alpha_payoff_gate_dynamic_exit_top5"}:
                score_col = "score_alpha_only"
            elif variant == "cluster_alpha_health_bad_penalty_top5":
                score_col = "score_health_bad_penalty"
            else:
                score_col = "score_bad_penalty"
            secondary = "top20_secondary" in variant
            dynamic_exit = "dynamic_exit" in variant
            use_payoff_gate = "payoff_gate" in variant
            daily, trades, candidate_audit = _run_event_backtest(
                test_panel,
                test_pred,
                states_ext,
                score_col=score_col,
                secondary_filter=secondary,
                dynamic_exit=dynamic_exit,
                policy=lambda row, ranks=ranks, use_payoff_gate=use_payoff_gate: (
                    _tiered_weight(row, ranks)
                    * float(row.get("strategy_gate", 1.0))
                    * (float(row.get("payoff_gate", 1.0)) if use_payoff_gate else 1.0)
                ),
            )
            tag = f"{variant}_{fold_name}_top{TOP_N}_cost{COST}"
            daily.to_parquet(output / f"daily_{tag}.parquet", index=False)
            trades.to_parquet(output / f"trades_{tag}.parquet", index=False)
            candidate_audit["variant"] = variant
            candidate_audit["fold"] = fold_name
            candidate_frames.append(candidate_audit)
            metrics = _metrics(daily, trades)
            test_controls = controls[
                controls["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))
            ]
            row = {
                "variant": variant,
                "fold": fold_name,
                "policy": POLICY,
                "top_n": TOP_N,
                "candidate_n": CANDIDATE_N if secondary else TOP_N,
                "cost_bps": COST,
                "hmm_variant": HMM_VARIANT,
                "lookback": LOOKBACK,
                "min_obs": MIN_OBS,
                "best_state": ranks["best"],
                "neutral_state": ranks["neutral"],
                "worst_state": ranks["worst"],
                "avg_exposure": float(daily["exposure"].mean()),
                "avg_holding_count": float(daily["holding_count"].mean()),
                "avg_daily_turnover": float(daily["turnover"].mean()),
                "flip_ratio": float(test_controls["score_direction"].lt(0.0).mean()),
                "payoff_gate_mean": float(test_controls["payoff_gate"].mean()),
                "payoff_gate_flat_ratio": float(test_controls["payoff_gate"].eq(0.0).mean()),
                "health_bad_penalty_ratio": float(
                    test_controls.get("health_bad_penalty_on", pd.Series(False, index=test_controls.index)).mean()
                ),
                "early_sell_count": int((trades["side"].eq("EARLY_SELL")).sum()) if not trades.empty and "side" in trades else 0,
                "risk_sell_count": int((trades["side"].eq("RISK_SELL")).sum()) if not trades.empty and "side" in trades else 0,
                **metrics,
            }
            rows.append(row)
            y = _yearly(row, daily)
            y["variant"] = variant
            yearly_frames.append(y)
            log(
                f"{tag} ann={metrics['annualized_return']:.2%} "
                f"excess={metrics['annualized_excess_return']:.2%} "
                f"maxdd={metrics['max_drawdown']:.2%} trades={len(trades)}"
            )

    metrics_df = pd.DataFrame(rows)
    yearly_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    train_df = pd.DataFrame(train_rows)
    gates_df = pd.concat(gate_frames, ignore_index=True) if gate_frames else pd.DataFrame()
    candidates_df = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    summary_df = _summary(metrics_df)

    metrics_df.to_csv(output / "event_iteration_metrics.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(output / "event_iteration_yearly.csv", index=False, encoding="utf-8-sig")
    train_df.to_csv(output / "event_iteration_train_summary.csv", index=False, encoding="utf-8-sig")
    gates_df.to_csv(output / "event_iteration_gate_scores.csv", index=False, encoding="utf-8-sig")
    candidates_df.to_csv(output / "event_iteration_candidate_audit.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output / "event_iteration_summary.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "data_version": version,
                "run_dir": str(output),
                "policy": POLICY,
                "variants": VARIANTS,
                "event_rule": "lower_shadow_pct>=0.60 or lower_shadow_atr>=0.80; plus downside/core confirmation",
                "metrics": metrics_df.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(summary_df, metrics_df, yearly_df, train_df), encoding="utf-8")
    log("done")
    print(f"run_dir={output}")


def _load_panel() -> tuple[str, pd.DataFrame]:
    project = load_project("configs/project.yaml")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest("latest")
    _, panel = repo.load_panel(version)
    return version, panel


def _load_model_dataset(log) -> pd.DataFrame:
    if not DATASET_CACHE.exists():
        raise FileNotFoundError(f"missing cached permission dataset: {DATASET_CACHE}")
    log(f"loading dataset {DATASET_CACHE}")
    dataset = pd.read_parquet(DATASET_CACHE)
    dataset["datetime"] = pd.to_datetime(dataset["datetime"])
    dataset = _add_clusters(dataset)
    return dataset


def _load_regime(log) -> pd.DataFrame:
    if not REGIME_CACHE.exists():
        raise FileNotFoundError(f"missing cached permission regime features: {REGIME_CACHE}")
    log(f"loading regime features {REGIME_CACHE}")
    regime = pd.read_parquet(REGIME_CACHE)
    regime["trade_date"] = pd.to_datetime(regime["trade_date"])
    return regime.sort_values("trade_date").reset_index(drop=True)


def _attach_market_payoff_gate(controls: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    r = regime[
        [
            "trade_date",
            "market_breadth_20",
            "reversal_strength_20",
            "market_vol_20",
            "turnover_chg_5_20",
        ]
    ].copy()
    for col in ["market_breadth_20", "reversal_strength_20", "market_vol_20", "turnover_chg_5_20"]:
        r[f"{col}_bucket"] = _historical_tercile_bucket(r[col])

    out = controls.merge(r, on="trade_date", how="left")
    score = pd.Series(0.0, index=out.index)
    score += np.where(out["market_breadth_20_bucket"].eq("low"), 1.0, 0.0)
    score += np.where(out["market_breadth_20_bucket"].eq("mid"), -1.0, 0.0)
    score += np.where(out["reversal_strength_20_bucket"].isin(["low", "mid"]), 1.0, 0.0)
    score += np.where(out["reversal_strength_20_bucket"].eq("high"), -1.0, 0.0)
    score += np.where(out["market_vol_20_bucket"].eq("high"), 1.0, 0.0)
    score += np.where(out["market_vol_20_bucket"].eq("mid"), -1.0, 0.0)
    score += np.where(out["turnover_chg_5_20_bucket"].eq("high"), 0.5, 0.0)
    out["payoff_gate_score"] = score
    out["payoff_gate"] = np.select(
        [score >= 1.0, score >= 0.0],
        [1.0, 0.5],
        default=0.0,
    )
    missing = out[["market_breadth_20_bucket", "reversal_strength_20_bucket", "market_vol_20_bucket"]].isna().any(axis=1)
    out.loc[missing, "payoff_gate"] = 1.0
    out.loc[missing, "payoff_gate_score"] = np.nan
    return out


def _historical_tercile_bucket(s: pd.Series, min_obs: int = 120) -> pd.Series:
    values = s.replace([np.inf, -np.inf], np.nan).astype(float)
    low = values.expanding(min_periods=min_obs).quantile(1 / 3).shift(1)
    high = values.expanding(min_periods=min_obs).quantile(2 / 3).shift(1)
    bucket = pd.Series(pd.NA, index=s.index, dtype="object")
    bucket = bucket.mask(values.le(low), "low")
    bucket = bucket.mask(values.gt(low) & values.lt(high), "mid")
    bucket = bucket.mask(values.ge(high), "high")
    return bucket


def _load_or_build_outcomes(panel: pd.DataFrame, pit: pd.DataFrame, log) -> pd.DataFrame:
    if LABEL_CACHE.exists():
        log(f"loading cached forward outcomes {LABEL_CACHE}")
        out = pd.read_parquet(LABEL_CACHE)
        out["trade_date"] = pd.to_datetime(out["trade_date"])
        return out
    log("building forward 10d outcomes from panel open/low data")
    cols = [
        "trade_date",
        "ts_code",
        "adj_open",
        "adj_low",
        "industry_l1_code",
        "is_tradeable",
        "is_suspended",
        "is_st",
        "is_delisting_period",
        "listing_trade_days",
    ]
    d = panel[cols].merge(pit[["trade_date", "ts_code", "pit_top1000"]], on=["trade_date", "ts_code"], how="left")
    d["pit_top1000"] = d["pit_top1000"].fillna(False).astype(bool)
    d = d[d["ts_code"].map(_permission_eligible)].sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    stocks = d["ts_code"]
    entry_open = d["adj_open"].groupby(stocks, sort=False).shift(-1)
    exit_open = d["adj_open"].groupby(stocks, sort=False).shift(-(REBALANCE_DAYS + 1))
    fwd_ret = exit_open / entry_open - 1.0
    future_low = d["adj_low"].groupby(stocks, sort=False).transform(_future_low_10)
    fwd_dd = future_low / entry_open - 1.0
    valid = (
        d["pit_top1000"]
        & d["is_tradeable"].fillna(False).astype(bool)
        & ~d["is_suspended"].fillna(True).astype(bool)
        & ~d["is_st"].fillna(True).astype(bool)
        & ~d["is_delisting_period"].fillna(True).astype(bool)
        & d["listing_trade_days"].ge(60)
    )
    industry_mean = fwd_ret.where(valid).groupby([d["trade_date"], d["industry_l1_code"]], sort=False).transform("mean")
    daily_rank = fwd_ret.where(valid).groupby(d["trade_date"], sort=False).rank(pct=True)
    out = d[["trade_date", "ts_code", "industry_l1_code"]].copy()
    out["fwd_ret_10"] = fwd_ret
    out["fwd_industry_excess_10"] = fwd_ret - industry_mean
    out["fwd_drawdown_10"] = fwd_dd
    out["bad_trade_10"] = ((fwd_ret <= -0.05) | (fwd_dd <= -0.08)).where(fwd_ret.notna() & fwd_dd.notna())
    out["top_decile_hit_10"] = (daily_rank >= 0.90).where(daily_rank.notna())
    out.to_parquet(LABEL_CACHE, index=False)
    log(f"cached forward outcomes rows={len(out):,} -> {LABEL_CACHE}")
    return out


def _future_low_10(s: pd.Series) -> pd.Series:
    return s.shift(-1).iloc[::-1].rolling(REBALANCE_DAYS, min_periods=1).min().iloc[::-1]


def _prepare_event_dataset(dataset: pd.DataFrame, outcomes: pd.DataFrame, log) -> pd.DataFrame:
    d = dataset.rename(columns={"datetime": "trade_date", "instrument": "ts_code"}).copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    d = d.merge(outcomes, on=["trade_date", "ts_code"], how="left")
    d["event_trigger"] = _event_trigger(d)
    for col in CLUSTER_COLS:
        d[col] = d[col].replace([np.inf, -np.inf], np.nan)
    usable = d["pit_top1000"].fillna(False).astype(bool) & d["event_trigger"]
    log(
        "event trigger by year: "
        + ", ".join(
            f"{int(y)}={float(g['event_trigger'].mean()):.1%}"
            for y, g in d[d["pit_top1000"].fillna(False)].groupby(d["trade_date"].dt.year)
            if y >= 2022
        )
    )
    d.loc[~usable, ["fwd_industry_excess_10", "bad_trade_10", "top_decile_hit_10"]] = np.nan
    return d


def _event_trigger(d: pd.DataFrame) -> pd.Series:
    lower = d["lower_shadow_pct"].ge(0.60) | d["lower_shadow_atr"].ge(0.80)
    downside = d["down_deviation_pct"].ge(0.50) | d["down_deviation_atr"].ge(0.0)
    confirm = d["core_signal"].notna() & d["intraday_repair"].notna()
    no_extreme_upper = d["upper_shadow_pct"].fillna(1.0).le(0.95)
    return lower & downside & confirm & no_extreme_upper


def _train_predict(dataset: pd.DataFrame, fold: dict, log) -> tuple[pd.DataFrame, dict]:
    from lightgbm import LGBMClassifier, LGBMRegressor

    train_end = pd.Timestamp(fold["train_end"])
    train_start = train_end - pd.DateOffset(years=2) + pd.Timedelta(days=1)
    feature_cols = CLUSTER_COLS
    cols = [
        "trade_date",
        "ts_code",
        *feature_cols,
        "sample_weight",
        "pit_top1000",
        "event_trigger",
        "fwd_industry_excess_10",
        "bad_trade_10",
        "top_decile_hit_10",
    ]
    train = dataset.loc[
        dataset["trade_date"].between(train_start, train_end)
        & dataset["pit_top1000"].fillna(False).astype(bool)
        & dataset["event_trigger"].fillna(False).astype(bool),
        cols,
    ].dropna(subset=[*feature_cols, "fwd_industry_excess_10", "bad_trade_10"]).copy()
    valid_test = dataset.loc[
        dataset["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["test_end"]))
        & dataset["pit_top1000"].fillna(False).astype(bool)
        & dataset["event_trigger"].fillna(False).astype(bool),
        cols,
    ].dropna(subset=feature_cols).copy()
    if train.empty:
        raise RuntimeError(f"empty event training set fold={fold['name']}")
    weights = train["sample_weight"].fillna(1.0).astype(float)
    alpha = LGBMRegressor(
        objective="regression",
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        n_estimators=450,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )
    bad = LGBMClassifier(
        objective="binary",
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        n_estimators=350,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=43,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )
    log(
        f"fit fold={fold['name']} train={len(train):,} "
        f"date={train['trade_date'].min().date()}..{train['trade_date'].max().date()} "
        f"bad_rate={train['bad_trade_10'].mean():.1%} predict={len(valid_test):,}"
    )
    alpha.fit(train[feature_cols], train["fwd_industry_excess_10"], sample_weight=weights)
    bad.fit(train[feature_cols], train["bad_trade_10"].astype(int), sample_weight=weights)
    out = valid_test[["trade_date", "ts_code", *feature_cols, "top_decile_hit_10"]].copy()
    out["alpha_pred"] = alpha.predict(valid_test[feature_cols])
    out["bad_prob"] = bad.predict_proba(valid_test[feature_cols])[:, 1]
    out["factor_value"] = out["alpha_pred"]
    summary = {
        "actual_train_start": train["trade_date"].min(),
        "actual_train_end": train["trade_date"].max(),
        "train_rows": int(len(train)),
        "predict_rows": int(len(valid_test)),
        "bad_rate": float(train["bad_trade_10"].mean()),
        "top_decile_rate": float(train["top_decile_hit_10"].mean()),
        "sample_weight_mean": float(weights.mean()),
    }
    return out, summary


def _pred_for_fit(pred: pd.DataFrame) -> pd.DataFrame:
    return pred[["trade_date", "ts_code", "factor_value"]].copy()


def _daily_rank_score(pred: pd.DataFrame, alpha_col: str, bad_col: str) -> pd.Series:
    alpha_rank = pred.groupby("trade_date")[alpha_col].rank(pct=True)
    bad_rank = pred.groupby("trade_date")[bad_col].rank(pct=True)
    return alpha_rank - BAD_PENALTY * bad_rank


def _attach_health_aware_score(pred: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    c = controls[["trade_date", "fit_obs", "rank_ic_rolling", "decile_spread_rolling", "top5_excess_rolling"]].copy()
    c["health_bad_penalty_on"] = (
        c["fit_obs"].ge(MIN_OBS)
        & c["top5_excess_rolling"].lt(0.0)
        & c["rank_ic_rolling"].lt(0.0)
        & c["decile_spread_rolling"].lt(0.0)
    )
    out = out.merge(c[["trade_date", "health_bad_penalty_on"]], on="trade_date", how="left")
    out["health_bad_penalty_on"] = out["health_bad_penalty_on"].fillna(False).astype(bool)
    alpha_rank = out.groupby("trade_date")["factor_value"].rank(pct=True)
    bad_rank = out.groupby("trade_date")["bad_prob"].rank(pct=True)
    out["score_health_bad_penalty"] = alpha_rank - BAD_PENALTY * bad_rank * out["health_bad_penalty_on"].astype(float)
    controls["health_bad_penalty_on"] = c["health_bad_penalty_on"].to_numpy()
    return out


def _run_event_backtest(
    panel: pd.DataFrame,
    pred: pd.DataFrame,
    states: pd.DataFrame,
    *,
    score_col: str,
    secondary_filter: bool,
    dynamic_exit: bool,
    policy,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = panel.merge(pred, on=["trade_date", "ts_code"], how="left")
    data = data.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    dates = list(pd.Index(data["trade_date"].unique()).sort_values())
    by_date = {d: g.set_index("ts_code") for d, g in data.groupby("trade_date")}
    pred_by_date = {d: g.set_index("ts_code") for d, g in pred.groupby("trade_date")}
    states_by_date = states.set_index("trade_date")
    cash = INITIAL_CASH
    positions: dict[str, dict] = {}
    daily_rows: list[dict] = []
    trade_rows: list[dict] = []
    candidate_rows: list[dict] = []
    half_cost = COST / 2.0 / 10_000.0
    exposure = 0.0

    for i, date in enumerate(dates):
        today = by_date[date]
        turnover = cost = 0.0
        buys = sells = early_sells = risk_sells = skipped = 0

        if dynamic_exit and i > 0:
            signal_date = dates[i - 1]
            latest = pred_by_date.get(signal_date)
            for code, pos in list(positions.items()):
                if code not in today.index or not _can_sell_at_open(today.loc[code]):
                    continue
                hold_days = i - int(pos.get("entry_i", i))
                value = _position_value(pos, today.loc[code])
                pnl = value / float(pos["entry_value"]) - 1.0
                bad_prob = float(latest.loc[code, "bad_prob"]) if latest is not None and code in latest.index else np.nan
                reason = None
                if pnl >= TAKE_PROFIT or pnl <= STOP_LOSS:
                    reason = "RISK_SELL"
                    risk_sells += 1
                elif hold_days >= EARLY_EXIT_MIN_HOLD_DAYS and np.isfinite(bad_prob) and bad_prob >= EARLY_EXIT_BAD_PROB and pnl <= 0.0:
                    reason = "EARLY_SELL"
                    early_sells += 1
                if reason is None:
                    continue
                sell_cost = value * half_cost
                cash += value - sell_cost
                turnover += value
                cost += sell_cost
                sells += 1
                trade_rows.append(
                    {
                        "trade_date": date,
                        "signal_date": signal_date,
                        "ts_code": code,
                        "side": reason,
                        "value": value,
                        "cost": sell_cost,
                        "exposure": exposure,
                        "hold_days": hold_days,
                        "pnl_at_open": pnl,
                        "bad_prob": bad_prob,
                    }
                )
                del positions[code]

        scheduled_rebalance = i > 0 and (i - 1) % REBALANCE_DAYS == 0
        if scheduled_rebalance:
            signal_date = dates[i - 1]
            state_row = states_by_date.loc[signal_date] if signal_date in states_by_date.index else pd.Series(dtype=float)
            exposure = float(np.clip(policy(state_row), 0.0, 1.0))
            for code, pos in list(positions.items()):
                if code not in today.index or not _can_sell_at_open(today.loc[code]):
                    continue
                value = _position_value(pos, today.loc[code])
                sell_cost = value * half_cost
                cash += value - sell_cost
                turnover += value
                cost += sell_cost
                sells += 1
                trade_rows.append(
                    {
                        "trade_date": date,
                        "signal_date": signal_date,
                        "ts_code": code,
                        "side": "SELL",
                        "value": value,
                        "cost": sell_cost,
                        "exposure": exposure,
                    }
                )
                del positions[code]

            signals = pred_by_date.get(signal_date)
            deploy_cash = cash * exposure
            if signals is not None and deploy_cash > 0:
                eligible = signals.join(
                    today[
                        [
                            "raw_open",
                            "adj_open",
                            "pre_close",
                            "industry_l1_code",
                            "is_tradeable",
                            "is_suspended",
                            "is_st",
                            "is_delisting_period",
                            "listing_trade_days",
                            "is_limit_up_open",
                        ]
                    ],
                    how="inner",
                )
                mask = (
                    eligible[score_col].notna()
                    & eligible["is_tradeable"].fillna(False).astype(bool)
                    & ~eligible["is_suspended"].fillna(True).astype(bool)
                    & ~eligible["is_st"].fillna(True).astype(bool)
                    & ~eligible["is_delisting_period"].fillna(True).astype(bool)
                    & eligible["listing_trade_days"].ge(60)
                    & ~eligible["is_limit_up_open"].fillna(False).astype(bool)
                )
                pool = eligible[mask].sort_values(score_col, ascending=False).head(CANDIDATE_N if secondary_filter else TOP_N)
                picks, skipped = _secondary_select(pool, score_col) if secondary_filter else (pool.head(TOP_N), 0)
                candidate_rows.append(
                    {
                        "trade_date": date,
                        "signal_date": signal_date,
                        "candidate_count": int(len(pool)),
                        "selected_count": int(len(picks)),
                        "skipped_secondary": int(skipped),
                        "avg_bad_prob_selected": float(picks["bad_prob"].mean()) if len(picks) else np.nan,
                    }
                )
                target_cash = deploy_cash / TOP_N
                for code, row in picks.iterrows():
                    price = float(row["raw_open"])
                    shares = int(target_cash // (price * LOT_SIZE)) * LOT_SIZE
                    if shares <= 0:
                        continue
                    gross = shares * price
                    buy_cost = gross * half_cost
                    if gross + buy_cost > cash:
                        continue
                    cash -= gross + buy_cost
                    turnover += gross
                    cost += buy_cost
                    buys += 1
                    positions[code] = {
                        "shares": shares,
                        "entry_raw_open": price,
                        "entry_adj_open": float(row["adj_open"]),
                        "entry_i": i,
                        "entry_value": gross,
                    }
                    trade_rows.append(
                        {
                            "trade_date": date,
                            "signal_date": signal_date,
                            "ts_code": code,
                            "side": "BUY",
                            "value": gross,
                            "cost": buy_cost,
                            "exposure": exposure,
                            "score": float(row[score_col]),
                            "alpha_pred": float(row.get("alpha_pred", np.nan)),
                            "bad_prob": float(row.get("bad_prob", np.nan)),
                            "industry_l1_code": row.get("industry_l1_code"),
                        }
                    )

        pos_value = 0.0
        for code, pos in positions.items():
            pos_value += _position_value(pos, today.loc[code]) if code in today.index else pos["shares"] * pos["entry_raw_open"]
        nav = cash + pos_value
        daily_rows.append(
            {
                "trade_date": date,
                "nav": nav,
                "cash": cash,
                "cash_ratio": cash / nav if nav > 0 else np.nan,
                "holding_count": len(positions),
                "turnover": turnover / (daily_rows[-1]["nav"] if daily_rows else INITIAL_CASH),
                "transaction_cost": cost,
                "executed_buys": buys,
                "executed_sells": sells,
                "early_sells": early_sells,
                "risk_sells": risk_sells,
                "skipped_secondary": skipped,
                "exposure": exposure,
            }
        )
    daily = pd.DataFrame(daily_rows)
    daily["benchmark_return"] = csi1000_open_to_open_returns(dates)
    daily["return"] = daily["nav"].pct_change().fillna(0.0)
    daily["excess_return"] = daily["return"] - daily["benchmark_return"]
    return daily, pd.DataFrame(trade_rows), pd.DataFrame(candidate_rows)


def _secondary_select(pool: pd.DataFrame, score_col: str) -> tuple[pd.DataFrame, int]:
    selected = []
    industry_counts: dict[str, int] = {}
    skipped = 0
    for code, row in pool.sort_values(score_col, ascending=False).iterrows():
        if len(selected) >= TOP_N:
            break
        gap = float(row["raw_open"] / row["pre_close"] - 1.0) if row.get("pre_close", np.nan) else np.nan
        industry = str(row.get("industry_l1_code", ""))
        bad_prob = float(row.get("bad_prob", np.nan))
        if np.isfinite(bad_prob) and bad_prob > BAD_PROB_BUY_CAP:
            skipped += 1
            continue
        if np.isfinite(gap) and gap > OPEN_GAP_CAP:
            skipped += 1
            continue
        if industry_counts.get(industry, 0) >= MAX_PER_INDUSTRY:
            skipped += 1
            continue
        selected.append(code)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
    return pool.loc[selected].copy(), skipped


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby("variant")
        .agg(
            mean_ann=("annualized_return", "mean"),
            median_ann=("annualized_return", "median"),
            mean_excess=("annualized_excess_return", "mean"),
            median_excess=("annualized_excess_return", "median"),
            positive_excess_folds=("annualized_excess_return", lambda s: int((s > 0.0).sum())),
            mean_sharpe=("sharpe", "mean"),
            worst_drawdown=("max_drawdown", "min"),
            mean_exposure=("avg_exposure", "mean"),
            mean_payoff_gate=("payoff_gate_mean", "mean"),
            mean_payoff_gate_flat_ratio=("payoff_gate_flat_ratio", "mean"),
            mean_health_bad_penalty_ratio=("health_bad_penalty_ratio", "mean"),
            mean_trades=("trade_count", "mean"),
            mean_early_sells=("early_sell_count", "mean"),
            mean_risk_sells=("risk_sell_count", "mean"),
        )
        .reset_index()
        .sort_values(["mean_ann", "mean_excess"], ascending=False)
    )


def _fmt_pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    return out


def _report(summary: pd.DataFrame, metrics: pd.DataFrame, yearly: pd.DataFrame, train: pd.DataFrame) -> str:
    s = _fmt_pct(
        summary,
        [
            "mean_ann",
            "median_ann",
            "mean_excess",
            "median_excess",
            "worst_drawdown",
            "mean_exposure",
            "mean_payoff_gate",
            "mean_payoff_gate_flat_ratio",
            "mean_health_bad_penalty_ratio",
        ],
    )
    s["mean_sharpe"] = s["mean_sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    show = metrics[
        [
            "variant",
            "fold",
            "annualized_return",
            "benchmark_annualized_return",
            "annualized_excess_return",
            "sharpe",
            "max_drawdown",
            "avg_exposure",
            "payoff_gate_mean",
            "payoff_gate_flat_ratio",
            "health_bad_penalty_ratio",
            "trade_count",
            "early_sell_count",
            "risk_sell_count",
        ]
    ].copy()
    show = _fmt_pct(
        show,
        [
            "annualized_return",
            "benchmark_annualized_return",
            "annualized_excess_return",
            "max_drawdown",
            "avg_exposure",
            "payoff_gate_mean",
            "payoff_gate_flat_ratio",
            "health_bad_penalty_ratio",
        ],
    )
    show["sharpe"] = show["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    y = yearly.copy()
    y = _fmt_pct(y, ["return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"])
    t = train.copy()
    for col in ["bad_rate", "top_decile_rate", "sample_weight_mean"]:
        if col in t:
            t[col] = t[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    return "\n".join(
        [
            "# Event + Bad-Trade ATR Iteration",
            "",
            "- universe: permission-eligible main-board subset inside PIT rolling liquidity Top1000",
            "- alpha label: future 10d open-to-open industry excess return",
            "- bad label: future 10d return <= -5% or forward drawdown <= -8%",
            "- execution: signal at T close, trade at T+1 open, Top5, 20bps, HMM tiered + frozen fit-quality flip",
            "",
            "## Variant Summary",
            "",
            s.to_markdown(index=False),
            "",
            "## Fold Metrics",
            "",
            show.to_markdown(index=False),
            "",
            "## Yearly",
            "",
            y.to_markdown(index=False),
            "",
            "## Training",
            "",
            t.to_markdown(index=False),
            "",
        ]
    )


if __name__ == "__main__":
    main()
