"""Calibrate an HMM regime gate specifically for ATR PIT-liquidity signals.

HMM states are inferred from market-wide features using only past data.  State
ranking / gate weights are calibrated on the ATR strategy's validation-period
performance, then frozen for the test-period backtest.
"""

from __future__ import annotations

import json
import math
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

from atr_reversion_pit_regime_backtest import _run_regime_backtest_pit
from atr_reversion_small_portfolio_backtest import _json_default, _metrics


FEATURES = [
    "market_return_20d",
    "market_return_slope_20d",
    "market_volatility_20d",
    "market_breadth_5d",
    "market_breadth_20d",
    "industry_breadth",
    "industry_dispersion",
    "market_turnover_change_5_20",
]

TOP_NS = [5, 10]
COST_BPS = [10, 20]
REBALANCE_DAYS = 10
HISTORY_DAYS = 756
ZSCORE_WINDOW = 756
ZSCORE_MIN_PERIODS = 60
SEED = 42


def main(
    config_path: str = "configs/ml/atr_reversion_lightgbm_v1.yaml",
    pit_run: str = "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z",
) -> None:
    cfg = load_atr_reversion_config(config_path)
    pit_run_path = Path(pit_run)
    output = pit_run_path / f"atr_hmm_calibrated_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
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
    log(f"loaded panel rows={len(panel):,} version={version}")

    pit = pd.read_parquet(pit_run_path / "pit_liquidity_flags.parquet")
    pit["trade_date"] = pd.to_datetime(pit["trade_date"])
    dataset = pd.read_parquet(pit_run_path / "pit_model_dataset.parquet")
    dataset["datetime"] = pd.to_datetime(dataset["datetime"])
    features = FEATURE_GROUPS["all"]
    log(f"loaded PIT model dataset rows={len(dataset):,}")

    predictions = _train_predict_valid_test(dataset, features, cfg, log)
    predictions.to_parquet(output / "predictions_valid_test.parquet", index=False)
    log(f"wrote predictions valid+test rows={len(predictions):,}")

    market = _market_features(panel)
    states = _walk_forward_hmm(market, cfg.segments.valid.start, cfg.segments.test.end, log)
    states.to_csv(output / "atr_hmm_daily_states.csv", index=False, encoding="utf-8-sig")
    log(f"wrote HMM states rows={len(states):,}")

    panel_bt = panel[
        panel["trade_date"].between(pd.Timestamp(cfg.segments.valid.start), pd.Timestamp(cfg.segments.test.end))
    ].merge(pit, on=["trade_date", "ts_code"], how="left")
    panel_bt["pit_top1000"] = panel_bt["pit_top1000"].fillna(False).astype(bool)
    rows = []
    yearly_frames = []

    for top_n in TOP_NS:
        for cost in COST_BPS:
            log(f"calibrating gates top_n={top_n} cost={cost} on validation")
            valid_pred = predictions[predictions["trade_date"].between(
                pd.Timestamp(cfg.segments.valid.start), pd.Timestamp(cfg.segments.valid.end)
            )]
            valid_panel = panel_bt[panel_bt["trade_date"].between(
                pd.Timestamp(cfg.segments.valid.start), pd.Timestamp(cfg.segments.valid.end)
            )]
            valid_daily, _ = _run_regime_backtest_pit(
                valid_panel,
                valid_pred,
                states,
                top_n=top_n,
                rebalance_days=REBALANCE_DAYS,
                cost_bps=cost,
                policy=lambda _row: 1.0,
            )
            ranks, state_perf = _rank_states_from_validation(valid_daily, states)
            state_perf.to_csv(output / f"state_validation_perf_top{top_n}_cost{cost}.csv", index=False, encoding="utf-8-sig")
            log(f"state ranks top_n={top_n} cost={cost}: {ranks}")

            test_pred = predictions[predictions["trade_date"].between(
                pd.Timestamp(cfg.segments.test.start), pd.Timestamp(cfg.segments.test.end)
            )]
            test_panel = panel_bt[panel_bt["trade_date"].between(
                pd.Timestamp(cfg.segments.test.start), pd.Timestamp(cfg.segments.test.end)
            )]
            policies = {
                "ungated": lambda _row, ranks=ranks: 1.0,
                "atr_hmm_hard_best": lambda row, ranks=ranks: 1.0 if int(row.get("predicted_state", -1)) == ranks["best"] else 0.0,
                "atr_hmm_tiered": lambda row, ranks=ranks: _tiered_weight(row, ranks),
                "atr_hmm_soft_prob": lambda row, ranks=ranks: _soft_prob_weight(row, ranks),
            }
            for policy_name, policy in policies.items():
                log(f"test backtest policy={policy_name} top_n={top_n} cost={cost}")
                daily, trades = _run_regime_backtest_pit(
                    test_panel,
                    test_pred,
                    states,
                    top_n=top_n,
                    rebalance_days=REBALANCE_DAYS,
                    cost_bps=cost,
                    policy=policy,
                )
                tag = f"{policy_name}_top{top_n}_rebalance{REBALANCE_DAYS}_cost{cost}"
                daily.to_parquet(output / f"daily_{tag}.parquet", index=False)
                trades.to_parquet(output / f"trades_{tag}.parquet", index=False)
                metrics = _metrics(daily, trades)
                metrics.update({
                    "policy": policy_name,
                    "top_n": top_n,
                    "rebalance_days": REBALANCE_DAYS,
                    "cost_bps": cost,
                    "avg_holding_count": float(daily["holding_count"].mean()),
                    "avg_cash_ratio": float(daily["cash_ratio"].mean()),
                    "avg_daily_turnover": float(daily["turnover"].mean()),
                    "avg_exposure": float(daily["exposure"].mean()),
                    "active_rebalance_ratio": float((daily["exposure"] > 0).mean()),
                    "best_state": ranks["best"],
                    "neutral_state": ranks["neutral"],
                    "worst_state": ranks["worst"],
                })
                rows.append(metrics)
                yearly_frames.append(_yearly(tag, daily, ranks))
                log(
                    f"done ann={metrics['annualized_return']:.2%} "
                    f"excess={metrics['annualized_excess_return']:.2%} "
                    f"sharpe={metrics['sharpe']:.2f} maxdd={metrics['max_drawdown']:.2%}"
                )

    metrics_df = pd.DataFrame(rows).sort_values(
        ["annualized_excess_return", "sharpe"], ascending=False
    )
    yearly = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    metrics_df.to_csv(output / "atr_hmm_calibrated_metrics.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "atr_hmm_calibrated_yearly.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "data_version": version,
                "pit_run": str(pit_run_path),
                "run_dir": str(output),
                "best": metrics_df.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(metrics_df, yearly), encoding="utf-8")
    log("wrote metrics/yearly/report")
    log("done")
    print(f"run_dir={output}")


def _load_panel():
    project = load_project("configs/project.yaml")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest("latest")
    _, panel = repo.load_panel(version)
    return version, panel


def _train_predict_valid_test(dataset, features, cfg, log) -> pd.DataFrame:
    from lightgbm import LGBMRegressor

    cols = ["datetime", "instrument", *features, "label", "sample_weight", "pit_top1000"]
    train = dataset.loc[
        dataset["datetime"].between(pd.Timestamp(cfg.segments.train.start), pd.Timestamp(cfg.segments.train.end))
        & dataset["pit_top1000"],
        cols,
    ].dropna(subset=[*features, "label"])
    valid_test = dataset.loc[
        dataset["datetime"].between(pd.Timestamp(cfg.segments.valid.start), pd.Timestamp(cfg.segments.test.end))
        & dataset["pit_top1000"],
        cols,
    ].dropna(subset=features)
    log(f"model slices train={len(train):,} valid_test_predictable={len(valid_test):,}")
    params = cfg.model.model_dump()
    params.pop("objective", None)
    params.setdefault("verbosity", -1)
    params.setdefault("force_col_wise", True)
    model = LGBMRegressor(objective="regression", **params)
    fit_kwargs = {"sample_weight": train["sample_weight"].fillna(1.0)} if "sample_weight" in train else {}
    log("fitting LightGBM for valid+test predictions")
    model.fit(train[features], train["label"], **fit_kwargs)
    out = valid_test[["datetime", "instrument"]].copy()
    out["factor_value"] = model.predict(valid_test[features])
    return out.rename(columns={"datetime": "trade_date", "instrument": "ts_code"})


def _market_features(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["ts_code", "trade_date"]).copy()
    ret = p["pct_change"] / 100.0 if "pct_change" in p else p.groupby("ts_code")["adj_close"].pct_change(fill_method=None)
    eligible = p.get("is_factor_eligible", p.get("is_tradeable", pd.Series(True, index=p.index))).fillna(False).astype(bool)
    src = pd.DataFrame({
        "trade_date": p["trade_date"],
        "ret": ret,
        "amount": p["amount_cny"],
        "industry": p["industry_l1_code"],
    }).loc[eligible]
    g = src.groupby("trade_date", sort=True)
    daily = pd.DataFrame({
        "market_return": g["ret"].mean(),
        "breadth": g["ret"].apply(lambda s: float((s.dropna() > 0).mean())),
        "turnover": g["amount"].sum(min_count=1),
    })
    ind = src.groupby(["trade_date", "industry"], observed=True)["ret"].mean().reset_index()
    ig = ind.groupby("trade_date")["ret"]
    daily["industry_breadth"] = ig.apply(lambda s: float((s.dropna() > 0).mean()))
    daily["industry_dispersion"] = ig.std(ddof=0)
    daily["market_return_20d"] = (1 + daily["market_return"]).rolling(20).apply(np.prod, raw=True) - 1
    x = np.arange(20, dtype=float)
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    logret = np.log1p(daily["market_return"].clip(lower=-0.999))
    daily["market_return_slope_20d"] = logret.rolling(20).apply(lambda a: np.dot(xc, np.cumsum(a)) / denom, raw=True)
    daily["market_volatility_20d"] = daily["market_return"].rolling(20).std(ddof=0)
    daily["market_breadth_5d"] = daily["breadth"].rolling(5).mean()
    daily["market_breadth_20d"] = daily["breadth"].rolling(20).mean()
    daily["market_turnover_change_5_20"] = daily["turnover"].rolling(5).mean() / daily["turnover"].rolling(20).mean() - 1
    return daily.reset_index()[["trade_date", *FEATURES]].replace([np.inf, -np.inf], np.nan).dropna()


def _walk_forward_hmm(market: pd.DataFrame, start: str, end: str, log) -> pd.DataFrame:
    from hmmlearn.hmm import GaussianHMM

    f = _standardize_market(market).dropna().reset_index(drop=True)
    zcols = [c + "_z" for c in FEATURES]
    periods = f.loc[f["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end)), "trade_date"].dt.to_period("M").drop_duplicates()
    outputs = []
    for period in periods:
        month = f[f["trade_date"].dt.to_period("M").eq(period)].copy()
        month = month[month["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        if month.empty:
            continue
        cutoff = month["trade_date"].min()
        train = f[f["trade_date"].lt(cutoff)].tail(HISTORY_DAYS)
        if len(train) < HISTORY_DAYS:
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
        # Stable state labels: rank by market-return/breadth friendliness.
        means = model.means_
        friendliness = means[:, zcols.index("market_return_20d_z")] + means[:, zcols.index("market_breadth_20d_z")] - means[:, zcols.index("market_volatility_20d_z")]
        order = np.argsort(friendliness)
        prob = _filtered(model, pd.concat([train, month])[zcols].to_numpy(float))[-len(month):, order]
        out = month[["trade_date"]].copy()
        for state in range(3):
            out[f"state_probability_{state}"] = prob[:, state]
        out["predicted_state"] = prob.argmax(axis=1)
        out["hmm_train_start"] = train["trade_date"].min()
        out["hmm_train_end"] = train["trade_date"].max()
        outputs.append(out)
    states = pd.concat(outputs, ignore_index=True).sort_values("trade_date")
    log(f"HMM states date range={states.trade_date.min().date()}..{states.trade_date.max().date()}")
    return states


def _standardize_market(raw: pd.DataFrame) -> pd.DataFrame:
    f = raw.sort_values("trade_date").copy().reset_index(drop=True)
    for col in FEATURES:
        past_mean = f[col].rolling(ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).mean()
        past_std = f[col].rolling(ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).std(ddof=0)
        f[col + "_z"] = (f[col] - past_mean) / past_std.replace(0, np.nan)
    return f


def _filtered(model, observations: np.ndarray) -> np.ndarray:
    var = np.maximum(getattr(model, "_covars_", model.covars_), 1e-8)
    means = model.means_
    ll = -0.5 * (
        observations.shape[1] * math.log(2 * math.pi)
        + np.log(var).sum(axis=1)[None, :]
        + (((observations[:, None, :] - means[None, :, :]) ** 2) / var[None, :, :]).sum(axis=2)
    )
    likelihood = np.exp(ll - ll.max(axis=1, keepdims=True))
    out = np.empty((len(observations), model.n_components), dtype=float)
    prior = model.startprob_.copy()
    for idx in range(len(observations)):
        if idx:
            prior = out[idx - 1] @ model.transmat_
        post = prior * likelihood[idx]
        out[idx] = post / post.sum() if post.sum() else np.full(model.n_components, 1 / model.n_components)
    return out


def _rank_states_from_validation(valid_daily: pd.DataFrame, states: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    d = valid_daily.merge(states[["trade_date", "predicted_state"]], on="trade_date", how="inner")
    perf = d.groupby("predicted_state").agg(
        mean_excess=("excess_return", "mean"),
        mean_return=("return", "mean"),
        days=("trade_date", "size"),
    ).reindex([0, 1, 2]).reset_index()
    ordered = perf.sort_values("mean_excess", ascending=False)["predicted_state"].astype(int).tolist()
    return {"best": ordered[0], "neutral": ordered[1], "worst": ordered[2]}, perf


def _tiered_weight(row: pd.Series, ranks: dict) -> float:
    state = int(row.get("predicted_state", -1))
    if state == ranks["best"]:
        return 1.0
    if state == ranks["neutral"]:
        return 0.5
    return 0.0


def _soft_prob_weight(row: pd.Series, ranks: dict) -> float:
    weights = {ranks["best"]: 1.0, ranks["neutral"]: 0.5, ranks["worst"]: 0.0}
    return sum(float(row.get(f"state_probability_{state}", 0.0)) * weight for state, weight in weights.items())


def _yearly(tag: str, daily: pd.DataFrame, ranks: dict) -> pd.DataFrame:
    rows = []
    for year, g in daily.groupby(pd.to_datetime(daily["trade_date"]).dt.year):
        if len(g) < 2:
            continue
        total = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1
        bench = (1 + g["benchmark_return"]).prod() - 1
        dd = g["nav"] / g["nav"].cummax() - 1
        rows.append({
            "run_key": tag,
            "year": int(year),
            "return": float(total),
            "benchmark_return": float(bench),
            "excess_return": float(total - bench),
            "max_drawdown": float(dd.min()),
            "avg_exposure": float(g["exposure"].mean()),
            **{f"{k}_state": v for k, v in ranks.items()},
        })
    return pd.DataFrame(rows)


def _report(metrics: pd.DataFrame, yearly: pd.DataFrame) -> str:
    show = metrics[[
        "policy", "top_n", "cost_bps", "annualized_return", "benchmark_annualized_return",
        "annualized_excess_return", "sharpe", "max_drawdown", "avg_exposure",
        "avg_daily_turnover", "best_state", "neutral_state", "worst_state",
    ]].copy()
    for col in ["annualized_return", "benchmark_annualized_return", "annualized_excess_return", "max_drawdown", "avg_exposure", "avg_daily_turnover"]:
        show[col] = show[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    show["sharpe"] = show["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    y = yearly.copy()
    for col in ["return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"]:
        y[col] = y[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    return "\n".join([
        "# ATR-Calibrated HMM Regime Gate On PIT Liquidity Backtest",
        "",
        "State ranks are calibrated only on the validation period's ATR strategy daily excess return.",
        "",
        "## Overall",
        "",
        show.to_markdown(index=False),
        "",
        "## Yearly",
        "",
        y.to_markdown(index=False),
        "",
    ])


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/ml/atr_reversion_lightgbm_v1.yaml"
    pit_run = sys.argv[2] if len(sys.argv) > 2 else "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z"
    main(config, pit_run)

