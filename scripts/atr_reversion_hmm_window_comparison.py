"""Compare PIT HMM training windows for ATR lower-shadow reversion.

Alpha model protocol is fixed to LightGBM rolling_2y.  Only the market-regime
HMM training window changes:
- hmm_expanding_pit: all prior standardized market observations
- hmm_rolling_3y_pit: prior 756 sessions
- hmm_rolling_2y_pit: prior 504 sessions
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository
from factor_forge.ml.atr_reversion_config import load_atr_reversion_config
from factor_forge.ml.atr_reversion_dataset import FEATURE_GROUPS

from atr_reversion_pit_hmm_calibrated_backtest import (
    FEATURES,
    SEED,
    ZSCORE_MIN_PERIODS,
    ZSCORE_WINDOW,
    _filtered,
    _market_features,
    _rank_states_from_validation,
    _soft_prob_weight,
    _tiered_weight,
)
from atr_reversion_pit_regime_backtest import _run_regime_backtest_pit
from atr_reversion_small_portfolio_backtest import _json_default, _metrics
from atr_reversion_training_window_experiment import _train_predict_variant
from atr_reversion_walk_forward import FOLDS, REBALANCE_DAYS


TOP_N = 5
COSTS = [10, 20]
ALPHA_VARIANT = "rolling_2y"
HMM_VARIANTS = {
    "hmm_expanding_pit": {"mode": "expanding", "history_days": None, "min_history_days": 504},
    "hmm_rolling_3y_pit": {"mode": "rolling", "history_days": 756, "min_history_days": 756},
    "hmm_rolling_2y_pit": {"mode": "rolling", "history_days": 504, "min_history_days": 504},
}
POLICIES = ["atr_hmm_tiered", "atr_hmm_soft_prob"]


def main(
    config_path: str = "configs/ml/atr_reversion_lightgbm_v1.yaml",
    pit_run: str = "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z",
) -> None:
    cfg = load_atr_reversion_config(config_path)
    pit_run_path = Path(pit_run)
    output = pit_run_path / f"hmm_window_comparison_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
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
    pit = pd.read_parquet(pit_run_path / "pit_liquidity_flags.parquet")
    pit["trade_date"] = pd.to_datetime(pit["trade_date"])
    dataset = pd.read_parquet(pit_run_path / "pit_model_dataset.parquet")
    dataset["datetime"] = pd.to_datetime(dataset["datetime"])
    features = FEATURE_GROUPS["all"]
    market = _market_features(panel)
    standardized_market = _standardize_market(market).dropna().reset_index(drop=True)
    log(
        f"loaded panel={len(panel):,} pit={len(pit):,} dataset={len(dataset):,} "
        f"market_days={len(standardized_market):,} data_version={version}"
    )

    rows: list[dict] = []
    yearly_frames: list[pd.DataFrame] = []
    train_summaries: list[dict] = []
    state_summaries: list[dict] = []

    for fold in FOLDS:
        fold_name = fold["name"]
        fold_dir = output / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        log(f"{fold_name}: train LightGBM alpha={ALPHA_VARIANT}")
        pred, train_summary = _train_predict_variant(
            dataset,
            features,
            cfg,
            fold,
            ALPHA_VARIANT,
            set(),
            log,
        )
        pred.to_parquet(fold_dir / "predictions_valid_test_rolling_2y.parquet", index=False)
        train_summary.update({"fold": fold_name, "alpha_variant": ALPHA_VARIANT})
        train_summaries.append(train_summary)

        panel_bt = panel[
            panel["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["test_end"]))
        ].merge(pit, on=["trade_date", "ts_code"], how="left")
        panel_bt["pit_top1000"] = panel_bt["pit_top1000"].fillna(False).astype(bool)
        valid_panel = panel_bt[
            panel_bt["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["valid_end"]))
        ]
        test_panel = panel_bt[
            panel_bt["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))
        ]
        valid_pred = pred[
            pred["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["valid_end"]))
        ]
        test_pred = pred[
            pred["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))
        ]

        for hmm_name, hmm_cfg in HMM_VARIANTS.items():
            hmm_dir = fold_dir / hmm_name
            hmm_dir.mkdir(parents=True, exist_ok=True)
            log(f"{fold_name} {hmm_name}: fit/predict PIT filtered states")
            states = _walk_forward_hmm_window(
                standardized_market,
                fold["valid_start"],
                fold["test_end"],
                mode=str(hmm_cfg["mode"]),
                history_days=hmm_cfg["history_days"],
                min_history_days=int(hmm_cfg["min_history_days"]),
                log=log,
            )
            states.to_csv(hmm_dir / "hmm_daily_states.csv", index=False, encoding="utf-8-sig")
            state_summaries.extend(_state_summary(fold_name, hmm_name, states))

            for cost in COSTS:
                valid_daily, _ = _run_regime_backtest_pit(
                    valid_panel,
                    valid_pred,
                    states,
                    top_n=TOP_N,
                    rebalance_days=REBALANCE_DAYS,
                    cost_bps=cost,
                    policy=lambda _row: 1.0,
                )
                ranks, state_perf = _rank_states_from_validation(valid_daily, states)
                state_perf.to_csv(
                    hmm_dir / f"state_validation_perf_top{TOP_N}_cost{cost}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                policies = {
                    "atr_hmm_tiered": lambda row, ranks=ranks: _tiered_weight(row, ranks),
                    "atr_hmm_soft_prob": lambda row, ranks=ranks: _soft_prob_weight(row, ranks),
                }
                for policy_name in POLICIES:
                    daily, trades = _run_regime_backtest_pit(
                        test_panel,
                        test_pred,
                        states,
                        top_n=TOP_N,
                        rebalance_days=REBALANCE_DAYS,
                        cost_bps=cost,
                        policy=policies[policy_name],
                    )
                    tag = f"{fold_name}_{hmm_name}_{policy_name}_top{TOP_N}_cost{cost}"
                    daily.to_parquet(hmm_dir / f"daily_{tag}.parquet", index=False)
                    trades.to_parquet(hmm_dir / f"trades_{tag}.parquet", index=False)
                    metrics = _metrics(daily, trades)
                    row = {
                        "fold": fold_name,
                        "test_start": fold["test_start"],
                        "test_end": fold["test_end"],
                        "alpha_variant": ALPHA_VARIANT,
                        "hmm_variant": hmm_name,
                        "policy": policy_name,
                        "top_n": TOP_N,
                        "rebalance_days": REBALANCE_DAYS,
                        "cost_bps": cost,
                        "best_state": ranks["best"],
                        "neutral_state": ranks["neutral"],
                        "worst_state": ranks["worst"],
                        "avg_exposure": float(daily["exposure"].mean()),
                        "avg_daily_turnover": float(daily["turnover"].mean()),
                        **metrics,
                    }
                    rows.append(row)
                    yearly_frames.append(_yearly_row(row, daily))
                    log(
                        f"{tag} ann={metrics['annualized_return']:.2%} "
                        f"excess={metrics['annualized_excess_return']:.2%} "
                        f"sharpe={metrics['sharpe']:.2f} maxdd={metrics['max_drawdown']:.2%} "
                        f"exposure={row['avg_exposure']:.2%}"
                    )

    metrics = pd.DataFrame(rows)
    yearly = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    train_df = pd.DataFrame(train_summaries)
    state_df = pd.DataFrame(state_summaries)
    summary = _summary(metrics)
    metrics.to_csv(output / "hmm_window_metrics.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "hmm_window_yearly.csv", index=False, encoding="utf-8-sig")
    train_df.to_csv(output / "alpha_train_summary.csv", index=False, encoding="utf-8-sig")
    state_df.to_csv(output / "hmm_state_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "hmm_window_summary.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "data_version": version,
                "pit_run": str(pit_run_path),
                "run_dir": str(output),
                "alpha_variant": ALPHA_VARIANT,
                "hmm_variants": HMM_VARIANTS,
                "metrics": metrics.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(metrics, summary, yearly, train_df, state_df), encoding="utf-8")
    log("wrote hmm window comparison artifacts")
    log("done")
    print(f"run_dir={output}")


def _load_panel() -> tuple[str, pd.DataFrame]:
    project = load_project("configs/project.yaml")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest("latest")
    _, panel = repo.load_panel(version)
    return version, panel


def _standardize_market(raw: pd.DataFrame) -> pd.DataFrame:
    f = raw.sort_values("trade_date").copy().reset_index(drop=True)
    for col in FEATURES:
        past_mean = f[col].rolling(ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).mean()
        past_std = f[col].rolling(ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).std(ddof=0)
        f[col + "_z"] = (f[col] - past_mean) / past_std.replace(0, np.nan)
    return f


def _walk_forward_hmm_window(
    standardized_market: pd.DataFrame,
    start: str,
    end: str,
    *,
    mode: str,
    history_days: int | None,
    min_history_days: int,
    log,
) -> pd.DataFrame:
    from hmmlearn.hmm import GaussianHMM

    zcols = [c + "_z" for c in FEATURES]
    f = standardized_market.copy()
    periods = (
        f.loc[f["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end)), "trade_date"]
        .dt.to_period("M")
        .drop_duplicates()
    )
    outputs = []
    for period in periods:
        month = f[f["trade_date"].dt.to_period("M").eq(period)].copy()
        month = month[month["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        if month.empty:
            continue
        cutoff = month["trade_date"].min()
        prior = f[f["trade_date"].lt(cutoff)]
        train = prior if mode == "expanding" else prior.tail(int(history_days or 0))
        if len(train) < min_history_days:
            continue
        model = GaussianHMM(
            n_components=3,
            covariance_type="diag",
            n_iter=300,
            tol=1e-4,
            min_covar=1e-4,
            random_state=SEED,
        )
        model.fit(train[zcols].to_numpy(float))
        means = model.means_
        friendliness = (
            means[:, zcols.index("market_return_20d_z")]
            + means[:, zcols.index("market_breadth_20d_z")]
            - means[:, zcols.index("market_volatility_20d_z")]
        )
        order = np.argsort(friendliness)
        prob = _filtered(model, pd.concat([train, month])[zcols].to_numpy(float))[-len(month):, order]
        out = month[["trade_date"]].copy()
        for state in range(3):
            out[f"state_probability_{state}"] = prob[:, state]
        out["predicted_state"] = prob.argmax(axis=1)
        out["hmm_train_start"] = train["trade_date"].min()
        out["hmm_train_end"] = train["trade_date"].max()
        out["hmm_train_rows"] = len(train)
        outputs.append(out)
    if not outputs:
        raise RuntimeError(f"no HMM predictions for {start}..{end}, mode={mode}, min_history_days={min_history_days}")
    states = pd.concat(outputs, ignore_index=True).sort_values("trade_date")
    log(
        f"HMM {mode} states={len(states):,} "
        f"range={states.trade_date.min().date()}..{states.trade_date.max().date()} "
        f"train_rows={states['hmm_train_rows'].min()}..{states['hmm_train_rows'].max()}"
    )
    return states


def _state_summary(fold: str, hmm_variant: str, states: pd.DataFrame) -> list[dict]:
    rows = []
    for state, g in states.groupby("predicted_state"):
        rows.append(
            {
                "fold": fold,
                "hmm_variant": hmm_variant,
                "predicted_state": int(state),
                "days": int(len(g)),
                "day_ratio": float(len(g) / len(states)),
                "avg_probability": float(g[f"state_probability_{int(state)}"].mean()),
                "min_train_rows": int(g["hmm_train_rows"].min()),
                "max_train_rows": int(g["hmm_train_rows"].max()),
            }
        )
    return rows


def _yearly_row(meta: dict, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = daily.copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    for year, g in d.groupby(d["trade_date"].dt.year):
        if len(g) < 2:
            continue
        total = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1.0
        bench = (1.0 + g["benchmark_return"]).prod() - 1.0
        dd = g["nav"] / g["nav"].cummax() - 1.0
        rows.append(
            {
                "fold": meta["fold"],
                "alpha_variant": meta["alpha_variant"],
                "hmm_variant": meta["hmm_variant"],
                "policy": meta["policy"],
                "top_n": meta["top_n"],
                "rebalance_days": meta["rebalance_days"],
                "cost_bps": meta["cost_bps"],
                "year": int(year),
                "return": float(total),
                "benchmark_return": float(bench),
                "excess_return": float(total - bench),
                "max_drawdown": float(dd.min()),
                "avg_exposure": float(g["exposure"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    return (
        metrics.groupby(["hmm_variant", "policy", "cost_bps"])
        .agg(
            mean_ann=("annualized_return", "mean"),
            median_ann=("annualized_return", "median"),
            mean_excess=("annualized_excess_return", "mean"),
            median_excess=("annualized_excess_return", "median"),
            positive_excess_folds=("annualized_excess_return", lambda s: int((s > 0).sum())),
            mean_sharpe=("sharpe", "mean"),
            worst_drawdown=("max_drawdown", "min"),
            mean_exposure=("avg_exposure", "mean"),
            mean_turnover=("avg_daily_turnover", "mean"),
        )
        .reset_index()
        .sort_values(["cost_bps", "policy", "mean_excess"], ascending=[True, True, False])
    )


def _fmt_pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    return out


def _report(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    train: pd.DataFrame,
    state_summary: pd.DataFrame,
) -> str:
    summ = _fmt_pct(
        summary,
        [
            "mean_ann",
            "median_ann",
            "mean_excess",
            "median_excess",
            "worst_drawdown",
            "mean_exposure",
            "mean_turnover",
        ],
    )
    if "mean_sharpe" in summ:
        summ["mean_sharpe"] = summ["mean_sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    fold = metrics[
        [
            "fold",
            "hmm_variant",
            "policy",
            "cost_bps",
            "annualized_return",
            "annualized_excess_return",
            "sharpe",
            "max_drawdown",
            "avg_exposure",
        ]
    ].copy()
    fold = _fmt_pct(fold, ["annualized_return", "annualized_excess_return", "max_drawdown", "avg_exposure"])
    fold["sharpe"] = fold["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    y = yearly.copy()
    y = _fmt_pct(y, ["return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"])

    tr = train[["fold", "alpha_variant", "actual_train_start", "actual_train_end", "train_rows"]].copy()
    ss = state_summary.copy()
    if not ss.empty:
        ss["day_ratio"] = ss["day_ratio"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        ss["avg_probability"] = ss["avg_probability"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

    return "\n".join(
        [
            "# ATR HMM Window Comparison",
            "",
            "Alpha is fixed to LightGBM rolling_2y.  Only the PIT HMM training window changes.",
            "",
            "## Summary",
            "",
            summ.to_markdown(index=False),
            "",
            "## Fold Metrics",
            "",
            fold.to_markdown(index=False),
            "",
            "## Yearly",
            "",
            y.to_markdown(index=False),
            "",
            "## Alpha Training Samples",
            "",
            tr.to_markdown(index=False),
            "",
            "## HMM State Usage",
            "",
            ss.to_markdown(index=False) if not ss.empty else "No state summary.",
            "",
        ]
    )


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/ml/atr_reversion_lightgbm_v1.yaml"
    pit_run = sys.argv[2] if len(sys.argv) > 2 else "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z"
    main(config, pit_run)
