"""Mine strategy-aware regime gates for the ATR reversion strategy.

This script works on an existing walk-forward run.  For every fold it builds
rebalance-cycle samples from the validation period, selects a simple one-rule
kill switch, then applies the frozen rule to the test period on top of the
ATR-calibrated HMM tiered gate.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository

from atr_reversion_pit_hmm_calibrated_backtest import (
    FEATURES as HMM_FEATURES,
    _rank_states_from_validation,
    _tiered_weight,
)
from atr_reversion_pit_regime_backtest import _run_regime_backtest_pit
from atr_reversion_small_portfolio_backtest import _json_default, _metrics
from atr_reversion_walk_forward import FOLDS, REBALANCE_DAYS


TOP_N = 5
COSTS = [10, 20]
IC_REALIZATION_LAG_DAYS = 10
MIN_ACTIVE_RATIO = 0.35
MAX_ACTIVE_RATIO = 0.90
MIN_OFF_CYCLES = 2
RULE_QUANTILES = [0.2, 0.3, 0.4, 0.6, 0.7, 0.8]


def main(
    walk_forward_dir: str = "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/walk_forward_20260706T102017Z",
) -> None:
    wf_path = Path(walk_forward_dir)
    pit_run = wf_path.parent
    output = pit_run / f"strategy_regime_mining_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
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
    pit = pd.read_parquet(pit_run / "pit_liquidity_flags.parquet")
    pit["trade_date"] = pd.to_datetime(pit["trade_date"])
    dataset = pd.read_parquet(pit_run / "pit_model_dataset.parquet")
    dataset["datetime"] = pd.to_datetime(dataset["datetime"])
    log(f"loaded panel={len(panel):,} pit={len(pit):,} dataset={len(dataset):,} version={version}")

    regime_features = _build_regime_features(panel, pit, dataset)
    regime_features.to_parquet(output / "daily_regime_features.parquet", index=False)
    log(f"built daily regime features rows={len(regime_features):,}")

    rows: list[dict] = []
    yearly_frames: list[pd.DataFrame] = []
    rule_rows: list[dict] = []
    cycle_frames: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []

    for fold in FOLDS:
        fold_name = fold["name"]
        fold_dir = wf_path / fold_name
        pred = pd.read_parquet(fold_dir / "predictions_valid_test.parquet")
        pred["trade_date"] = pd.to_datetime(pred["trade_date"])
        states = pd.read_csv(fold_dir / "hmm_daily_states.csv")
        states["trade_date"] = pd.to_datetime(states["trade_date"])
        fold_features = _add_signal_features(regime_features, pred, dataset, panel)
        fold_features.to_parquet(output / f"regime_features_{fold_name}.parquet", index=False)

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
        valid_pred = pred[pred["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["valid_end"]))]
        test_pred = pred[pred["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))]

        for cost in COSTS:
            log(f"{fold_name} cost={cost}: calibrating HMM tiered base")
            valid_ungated, _ = _run_regime_backtest_pit(
                valid_panel,
                valid_pred,
                states,
                top_n=TOP_N,
                rebalance_days=REBALANCE_DAYS,
                cost_bps=cost,
                policy=lambda _row: 1.0,
            )
            ranks, _state_perf = _rank_states_from_validation(valid_ungated, states)
            valid_base, _ = _run_regime_backtest_pit(
                valid_panel,
                valid_pred,
                states,
                top_n=TOP_N,
                rebalance_days=REBALANCE_DAYS,
                cost_bps=cost,
                policy=lambda row, ranks=ranks: _tiered_weight(row, ranks),
            )
            valid_cycles = _cycle_samples(valid_base, fold_features, fold_name, "valid", cost)
            valid_cycles.to_csv(output / f"cycles_{fold_name}_valid_cost{cost}.csv", index=False, encoding="utf-8-sig")
            cycle_frames.append(valid_cycles)
            if len(valid_cycles) < 8:
                rule = _default_rule(fold_name, cost, "too_few_validation_cycles")
                rule_diag = pd.DataFrame()
            else:
                rule, rule_diag = _select_rule(valid_cycles, fold_name, cost)
                if not rule_diag.empty:
                    diagnostics.append(rule_diag)
                    rule_diag.to_csv(output / f"rule_candidates_{fold_name}_cost{cost}.csv", index=False, encoding="utf-8-sig")
            rule_rows.append(rule)
            log(
                f"{fold_name} cost={cost}: selected {rule['rule_text']} "
                f"valid_delta={rule['valid_mean_excess_delta']:.2%} active={rule['valid_active_ratio']:.1%}"
            )

            test_features = fold_features[fold_features["trade_date"].between(
                pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"])
            )]
            gate_map = _gate_map_from_rule(test_features, rule)
            states_ext = states.merge(
                pd.DataFrame({"trade_date": list(gate_map.keys()), "strategy_gate": list(gate_map.values())}),
                on="trade_date",
                how="left",
            )
            states_ext["strategy_gate"] = states_ext["strategy_gate"].fillna(1.0)
            policy: Callable[[pd.Series], float] = (
                lambda row, ranks=ranks: _tiered_weight(row, ranks) * float(row.get("strategy_gate", 1.0))
            )
            test_daily, test_trades = _run_regime_backtest_pit(
                test_panel,
                test_pred,
                states_ext,
                top_n=TOP_N,
                rebalance_days=REBALANCE_DAYS,
                cost_bps=cost,
                policy=policy,
            )
            tag = f"{fold_name}_strategy_gate_tiered_top{TOP_N}_cost{cost}"
            test_daily.to_parquet(output / f"daily_{tag}.parquet", index=False)
            test_trades.to_parquet(output / f"trades_{tag}.parquet", index=False)
            test_cycles = _cycle_samples(test_daily, fold_features, fold_name, "test", cost)
            test_cycles.to_csv(output / f"cycles_{fold_name}_test_cost{cost}.csv", index=False, encoding="utf-8-sig")
            cycle_frames.append(test_cycles)

            metrics = _metrics(test_daily, test_trades)
            metrics.update({
                "fold": fold_name,
                "policy": "strategy_gate_tiered",
                "top_n": TOP_N,
                "cost_bps": cost,
                "best_state": ranks["best"],
                "neutral_state": ranks["neutral"],
                "worst_state": ranks["worst"],
                "avg_exposure": float(test_daily["exposure"].mean()),
                "avg_daily_turnover": float(test_daily["turnover"].mean()),
                "rule_text": rule["rule_text"],
            })
            rows.append(metrics)
            yearly_frames.append(_yearly_row(metrics, test_daily))
            log(
                f"{tag} ann={metrics['annualized_return']:.2%} "
                f"excess={metrics['annualized_excess_return']:.2%} "
                f"sharpe={metrics['sharpe']:.2f} maxdd={metrics['max_drawdown']:.2%}"
            )

    metrics_df = pd.DataFrame(rows)
    rules_df = pd.DataFrame(rule_rows)
    cycles_df = pd.concat(cycle_frames, ignore_index=True) if cycle_frames else pd.DataFrame()
    yearly_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    diagnostics_df = pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()
    baseline = _load_baseline_metrics(wf_path)
    comparison = _compare(metrics_df, baseline)
    feature_summary = _feature_bad_good_summary(cycles_df)

    metrics_df.to_csv(output / "strategy_gate_metrics.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(output / "strategy_gate_yearly.csv", index=False, encoding="utf-8-sig")
    rules_df.to_csv(output / "selected_rules.csv", index=False, encoding="utf-8-sig")
    cycles_df.to_csv(output / "cycle_regime_samples.csv", index=False, encoding="utf-8-sig")
    diagnostics_df.to_csv(output / "rule_candidates_all.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "walk_forward_comparison.csv", index=False, encoding="utf-8-sig")
    feature_summary.to_csv(output / "feature_bad_good_summary.csv", index=False, encoding="utf-8-sig")
    summary = {
        "data_version": version,
        "walk_forward_dir": str(wf_path),
        "pit_run": str(pit_run),
        "run_dir": str(output),
        "metrics": metrics_df.to_dict("records"),
        "selected_rules": rules_df.to_dict("records"),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    (output / "report.md").write_text(_report(metrics_df, comparison, rules_df, feature_summary), encoding="utf-8")
    log("wrote strategy regime mining report")
    log("done")
    print(f"run_dir={output}")


def _load_panel() -> tuple[str, pd.DataFrame]:
    project = load_project("configs/project.yaml")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest("latest")
    _, panel = repo.load_panel(version)
    return version, panel


def _build_regime_features(panel: pd.DataFrame, pit: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["ts_code", "trade_date"]).copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    ret = p["pct_change"] / 100.0 if "pct_change" in p else p.groupby("ts_code")["adj_close"].pct_change(fill_method=None)
    p["ret"] = ret.replace([np.inf, -np.inf], np.nan)
    p["ret_5"] = p.groupby("ts_code")["adj_close"].pct_change(5, fill_method=None)
    p["ret_20"] = p.groupby("ts_code")["adj_close"].pct_change(20, fill_method=None)
    p["mom_signal"] = p.groupby("ts_code")["ret_20"].shift(1)
    p["rev_signal"] = -p.groupby("ts_code")["ret_5"].shift(1)
    p = p.merge(pit, on=["trade_date", "ts_code"], how="left")
    p["pit_top1000"] = p["pit_top1000"].fillna(False).astype(bool)
    eligible = p["pit_top1000"] & p.get("is_tradeable", pd.Series(True, index=p.index)).fillna(False).astype(bool)
    src = p.loc[eligible].copy()

    daily = src.groupby("trade_date", sort=True).agg(
        pit_return=("ret", "mean"),
        pit_breadth=("ret", lambda s: float((s.dropna() > 0).mean()) if s.notna().any() else np.nan),
        pit_xsec_vol=("ret", lambda s: float(s.dropna().std(ddof=0)) if s.notna().sum() > 1 else np.nan),
        pit_turnover=("amount_cny", "sum"),
        pit_limit_up_ratio=("is_limit_up_open", _true_ratio),
        pit_limit_down_ratio=("is_limit_down_open", _true_ratio),
    )
    daily["market_ret_20"] = (1.0 + daily["pit_return"]).rolling(20).apply(np.prod, raw=True) - 1.0
    daily["market_ret_60"] = (1.0 + daily["pit_return"]).rolling(60).apply(np.prod, raw=True) - 1.0
    wealth = (1.0 + daily["pit_return"].fillna(0.0)).cumprod()
    daily["market_drawdown_60"] = wealth / wealth.rolling(60).max() - 1.0
    daily["market_vol_20"] = daily["pit_return"].rolling(20).std(ddof=0)
    daily["market_breadth_20"] = daily["pit_breadth"].rolling(20).mean()
    daily["xsec_vol_20"] = daily["pit_xsec_vol"].rolling(20).mean()
    daily["turnover_chg_5_20"] = daily["pit_turnover"].rolling(5).mean() / daily["pit_turnover"].rolling(20).mean() - 1.0

    style = _style_environment(src)
    daily = daily.join(style, how="left")
    hmm_like = _hmm_feature_frame(panel)
    out = daily.reset_index().merge(hmm_like, on="trade_date", how="left")
    out = out.replace([np.inf, -np.inf], np.nan).sort_values("trade_date")
    return out


def _true_ratio(s: pd.Series) -> float:
    if s.empty:
        return np.nan
    return float(s.fillna(False).astype(bool).mean())


def _style_environment(src: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, g in src.groupby("trade_date", sort=True):
        row = {"trade_date": date}
        for signal_col, out_col in [("mom_signal", "momentum_ls"), ("rev_signal", "reversal_ls")]:
            h = g[["ret", signal_col]].dropna()
            if len(h) < 100:
                row[out_col] = np.nan
                continue
            high = h[signal_col] >= h[signal_col].quantile(0.8)
            low = h[signal_col] <= h[signal_col].quantile(0.2)
            row[out_col] = float(h.loc[high, "ret"].mean() - h.loc[low, "ret"].mean())
        rows.append(row)
    style = pd.DataFrame(rows).set_index("trade_date")
    style["momentum_strength_20"] = style["momentum_ls"].rolling(20).mean().shift(1)
    style["reversal_strength_20"] = style["reversal_ls"].rolling(20).mean().shift(1)
    style["momentum_minus_reversal_20"] = style["momentum_strength_20"] - style["reversal_strength_20"]
    return style[["momentum_strength_20", "reversal_strength_20", "momentum_minus_reversal_20"]]


def _hmm_feature_frame(panel: pd.DataFrame) -> pd.DataFrame:
    # Reuse the market features already validated for HMM, but keep them as
    # ordinary continuous regime variables for rule mining.
    from atr_reversion_pit_hmm_calibrated_backtest import _market_features

    out = _market_features(panel)
    keep = ["trade_date", *HMM_FEATURES]
    return out[keep].copy()


def _add_signal_features(
    regime_features: pd.DataFrame,
    pred: pd.DataFrame,
    dataset: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    pred = pred.copy()
    pred["trade_date"] = pd.to_datetime(pred["trade_date"])
    panel_keys = panel[["trade_date", "ts_code", "industry_l1_code"]].copy()
    panel_keys["trade_date"] = pd.to_datetime(panel_keys["trade_date"])
    px = pred.merge(panel_keys, on=["trade_date", "ts_code"], how="left")
    daily = px.groupby("trade_date", sort=True).agg(
        pred_count=("factor_value", "size"),
        pred_score_mean=("factor_value", "mean"),
        pred_score_std=("factor_value", "std"),
        pred_score_p90=("factor_value", lambda s: float(s.quantile(0.9))),
        pred_score_p50=("factor_value", lambda s: float(s.quantile(0.5))),
    )
    top = px.sort_values(["trade_date", "factor_value"], ascending=[True, False]).groupby("trade_date").head(TOP_N)
    daily["top_score_mean"] = top.groupby("trade_date")["factor_value"].mean()
    daily["top_score_spread"] = daily["top_score_mean"] - daily["pred_score_p50"]
    daily["top_industry_hhi"] = top.groupby("trade_date")["industry_l1_code"].apply(_hhi)

    labels = dataset[["datetime", "instrument", "label"]].rename(
        columns={"datetime": "trade_date", "instrument": "ts_code"}
    )
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])
    ic_src = pred.merge(labels, on=["trade_date", "ts_code"], how="inner")
    ic = (
        ic_src.groupby("trade_date")[["factor_value", "label"]]
        .apply(_spearman_ic)
        .rename("daily_signal_ic")
        .to_frame()
    )
    ic["signal_ic_20"] = ic["daily_signal_ic"].rolling(20, min_periods=8).mean().shift(IC_REALIZATION_LAG_DAYS)
    ic["signal_ic_60"] = ic["daily_signal_ic"].rolling(60, min_periods=20).mean().shift(IC_REALIZATION_LAG_DAYS)
    daily = daily.join(ic[["signal_ic_20", "signal_ic_60"]], how="left")
    out = regime_features.merge(daily.reset_index(), on="trade_date", how="left")
    return out.replace([np.inf, -np.inf], np.nan).sort_values("trade_date")


def _hhi(s: pd.Series) -> float:
    counts = s.fillna("UNKNOWN").value_counts(normalize=True)
    return float((counts * counts).sum())


def _spearman_ic(g: pd.DataFrame) -> float:
    h = g[["factor_value", "label"]].dropna()
    if len(h) < 30:
        return np.nan
    return float(h["factor_value"].rank().corr(h["label"].rank()))


def _cycle_samples(
    daily: pd.DataFrame,
    features: pd.DataFrame,
    fold: str,
    segment: str,
    cost: int,
) -> pd.DataFrame:
    d = daily.copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    d = d.sort_values("trade_date").reset_index(drop=True)
    f = features.set_index("trade_date")
    rows = []
    for signal_idx in range(0, len(d) - 1, REBALANCE_DAYS):
        end_idx = min(signal_idx + REBALANCE_DAYS, len(d) - 1)
        if end_idx <= signal_idx:
            continue
        signal_date = d.loc[signal_idx, "trade_date"]
        if signal_date not in f.index:
            continue
        nav0 = float(d.loc[signal_idx, "nav"])
        nav1 = float(d.loc[end_idx, "nav"])
        strat_ret = nav1 / nav0 - 1.0 if nav0 > 0 else np.nan
        bench = float((1.0 + d.loc[signal_idx + 1 : end_idx, "benchmark_return"]).prod() - 1.0)
        row = f.loc[signal_date].to_dict()
        row.update({
            "fold": fold,
            "segment": segment,
            "cost_bps": cost,
            "signal_date": signal_date,
            "cycle_start": d.loc[signal_idx + 1, "trade_date"],
            "cycle_end": d.loc[end_idx, "trade_date"],
            "cycle_return": strat_ret,
            "cycle_benchmark_return": bench,
            "cycle_excess": strat_ret - bench,
            "bad_regime": bool((strat_ret < 0.0) and ((strat_ret - bench) < -0.02)),
            "avg_exposure": float(d.loc[signal_idx + 1 : end_idx, "exposure"].mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _candidate_features(df: pd.DataFrame) -> list[str]:
    exclude = {
        "fold",
        "segment",
        "cost_bps",
        "trade_date",
        "signal_date",
        "cycle_start",
        "cycle_end",
        "cycle_return",
        "cycle_benchmark_return",
        "cycle_excess",
        "bad_regime",
        "avg_exposure",
    }
    out = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def _select_rule(train: pd.DataFrame, fold: str, cost: int) -> tuple[dict, pd.DataFrame]:
    base_mean = float(train["cycle_excess"].mean())
    base_return = float(train["cycle_return"].mean())
    rows = []
    for feature in _candidate_features(train):
        s = train[feature].replace([np.inf, -np.inf], np.nan)
        if s.notna().sum() < 8 or s.nunique(dropna=True) < 4:
            continue
        thresholds = sorted(set(float(s.quantile(q)) for q in RULE_QUANTILES if pd.notna(s.quantile(q))))
        for threshold in thresholds:
            for direction in ["le", "ge"]:
                off = s.le(threshold) if direction == "le" else s.ge(threshold)
                off = off.fillna(False)
                off_count = int(off.sum())
                active_ratio = float((~off).mean())
                if off_count < MIN_OFF_CYCLES or active_ratio < MIN_ACTIVE_RATIO or active_ratio > MAX_ACTIVE_RATIO:
                    continue
                gated_return = train["cycle_return"].where(~off, 0.0)
                gated_excess = gated_return - train["cycle_benchmark_return"]
                bad_capture = float((off & train["bad_regime"].astype(bool)).sum() / max(int(train["bad_regime"].sum()), 1))
                false_off = float((off & ~train["bad_regime"].astype(bool)).sum() / max(off_count, 1))
                rows.append({
                    "fold": fold,
                    "cost_bps": cost,
                    "feature": feature,
                    "direction": direction,
                    "threshold": threshold,
                    "rule_text": _rule_text(feature, direction, threshold),
                    "off_cycles": off_count,
                    "active_ratio": active_ratio,
                    "bad_capture": bad_capture,
                    "false_off_ratio": false_off,
                    "base_mean_return": base_return,
                    "gated_mean_return": float(gated_return.mean()),
                    "mean_return_delta": float(gated_return.mean() - base_return),
                    "base_mean_excess": base_mean,
                    "gated_mean_excess": float(gated_excess.mean()),
                    "mean_excess_delta": float(gated_excess.mean() - base_mean),
                })
    if not rows:
        return _default_rule(fold, cost, "no_candidate_rule"), pd.DataFrame()
    cand = pd.DataFrame(rows).sort_values(
        ["mean_return_delta", "mean_excess_delta", "bad_capture", "false_off_ratio"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    top = cand.iloc[0].to_dict()
    rule = {
        "fold": fold,
        "cost_bps": cost,
        "feature": top["feature"],
        "direction": top["direction"],
        "threshold": float(top["threshold"]),
        "rule_text": top["rule_text"],
        "valid_active_ratio": float(top["active_ratio"]),
        "valid_off_cycles": int(top["off_cycles"]),
        "valid_bad_capture": float(top["bad_capture"]),
        "valid_false_off_ratio": float(top["false_off_ratio"]),
        "valid_mean_return_delta": float(top["mean_return_delta"]),
        "valid_mean_excess_delta": float(top["mean_excess_delta"]),
        "reason": "selected_by_validation_cycle_return_delta",
    }
    return rule, cand


def _default_rule(fold: str, cost: int, reason: str) -> dict:
    return {
        "fold": fold,
        "cost_bps": cost,
        "feature": "",
        "direction": "",
        "threshold": np.nan,
        "rule_text": "no extra strategy gate",
        "valid_active_ratio": 1.0,
        "valid_off_cycles": 0,
        "valid_bad_capture": 0.0,
        "valid_false_off_ratio": 0.0,
        "valid_mean_return_delta": 0.0,
        "valid_mean_excess_delta": 0.0,
        "reason": reason,
    }


def _rule_text(feature: str, direction: str, threshold: float) -> str:
    op = "<=" if direction == "le" else ">="
    return f"flat if {feature} {op} {threshold:.6g}"


def _gate_map_from_rule(features: pd.DataFrame, rule: dict) -> dict[pd.Timestamp, float]:
    if not rule.get("feature"):
        return {pd.Timestamp(d): 1.0 for d in features["trade_date"]}
    s = features[rule["feature"]]
    if rule["direction"] == "le":
        off = s.le(float(rule["threshold"]))
    else:
        off = s.ge(float(rule["threshold"]))
    return {
        pd.Timestamp(date): (0.0 if bool(is_off) else 1.0)
        for date, is_off in zip(features["trade_date"], off.fillna(False))
    }


def _load_baseline_metrics(wf_path: Path) -> pd.DataFrame:
    path = wf_path / "walk_forward_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[
        (df["policy"].eq("atr_hmm_tiered"))
        & (df["top_n"].eq(TOP_N))
        & (df["cost_bps"].isin(COSTS))
    ].copy()


def _compare(metrics: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    if baseline.empty or metrics.empty:
        return pd.DataFrame()
    left = baseline[[
        "fold",
        "cost_bps",
        "annualized_return",
        "annualized_excess_return",
        "sharpe",
        "max_drawdown",
        "avg_exposure",
    ]].rename(columns={
        "annualized_return": "base_ann",
        "annualized_excess_return": "base_excess",
        "sharpe": "base_sharpe",
        "max_drawdown": "base_maxdd",
        "avg_exposure": "base_exposure",
    })
    right = metrics[[
        "fold",
        "cost_bps",
        "annualized_return",
        "annualized_excess_return",
        "sharpe",
        "max_drawdown",
        "avg_exposure",
        "rule_text",
    ]].rename(columns={
        "annualized_return": "gate_ann",
        "annualized_excess_return": "gate_excess",
        "sharpe": "gate_sharpe",
        "max_drawdown": "gate_maxdd",
        "avg_exposure": "gate_exposure",
    })
    out = left.merge(right, on=["fold", "cost_bps"], how="inner")
    out["ann_delta"] = out["gate_ann"] - out["base_ann"]
    out["excess_delta"] = out["gate_excess"] - out["base_excess"]
    out["sharpe_delta"] = out["gate_sharpe"] - out["base_sharpe"]
    out["maxdd_delta"] = out["gate_maxdd"] - out["base_maxdd"]
    return out


def _feature_bad_good_summary(cycles: pd.DataFrame) -> pd.DataFrame:
    valid = cycles[cycles["segment"].eq("valid")].copy()
    if valid.empty or "bad_regime" not in valid:
        return pd.DataFrame()
    rows = []
    for feature in _candidate_features(valid):
        h = valid[[feature, "bad_regime"]].dropna()
        if len(h) < 10 or h["bad_regime"].nunique() < 2:
            continue
        bad = h.loc[h["bad_regime"].astype(bool), feature]
        good = h.loc[~h["bad_regime"].astype(bool), feature]
        pooled = h[feature].std(ddof=0)
        rows.append({
            "feature": feature,
            "bad_mean": float(bad.mean()),
            "good_mean": float(good.mean()),
            "mean_diff": float(bad.mean() - good.mean()),
            "standardized_diff": float((bad.mean() - good.mean()) / pooled) if pooled and np.isfinite(pooled) else np.nan,
            "bad_count": int(len(bad)),
            "good_count": int(len(good)),
        })
    return pd.DataFrame(rows).sort_values("standardized_diff", key=lambda s: s.abs(), ascending=False)


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
        rows.append({
            "fold": meta["fold"],
            "policy": meta["policy"],
            "top_n": meta["top_n"],
            "cost_bps": meta["cost_bps"],
            "year": int(year),
            "return": float(total),
            "benchmark_return": float(bench),
            "excess_return": float(total - bench),
            "max_drawdown": float(dd.min()),
            "avg_exposure": float(g["exposure"].mean()),
            "rule_text": meta["rule_text"],
        })
    return pd.DataFrame(rows)


def _fmt_pct(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    return out


def _report(metrics: pd.DataFrame, comparison: pd.DataFrame, rules: pd.DataFrame, feature_summary: pd.DataFrame) -> str:
    m = metrics[[
        "fold",
        "cost_bps",
        "annualized_return",
        "annualized_excess_return",
        "sharpe",
        "max_drawdown",
        "avg_exposure",
        "rule_text",
    ]].copy()
    m = _fmt_pct(m, ["annualized_return", "annualized_excess_return", "max_drawdown", "avg_exposure"])
    m["sharpe"] = m["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    c = comparison.copy()
    if not c.empty:
        c = c[[
            "fold",
            "cost_bps",
            "base_ann",
            "gate_ann",
            "ann_delta",
            "base_excess",
            "gate_excess",
            "excess_delta",
            "base_maxdd",
            "gate_maxdd",
            "gate_exposure",
        ]]
        c = _fmt_pct(c, [
            "base_ann",
            "gate_ann",
            "ann_delta",
            "base_excess",
            "gate_excess",
            "excess_delta",
            "base_maxdd",
            "gate_maxdd",
            "gate_exposure",
        ])
    r = rules[[
        "fold",
        "cost_bps",
        "rule_text",
        "valid_active_ratio",
        "valid_bad_capture",
        "valid_false_off_ratio",
        "valid_mean_return_delta",
        "valid_mean_excess_delta",
    ]].copy()
    r = _fmt_pct(r, [
        "valid_active_ratio",
        "valid_bad_capture",
        "valid_false_off_ratio",
        "valid_mean_return_delta",
        "valid_mean_excess_delta",
    ])
    fs = feature_summary.head(15).copy()
    return "\n".join([
        "# ATR Strategy-Aware Regime Mining",
        "",
        "The extra gate is selected inside each fold's validation period and then frozen on the next test period.",
        "The objective is cycle-level absolute return improvement; excess return is reported because going flat still trails a rising benchmark.",
        "",
        "## Gated Walk-Forward",
        "",
        m.to_markdown(index=False),
        "",
        "## Versus ATR-HMM Tiered Baseline",
        "",
        c.to_markdown(index=False) if not c.empty else "No baseline metrics found.",
        "",
        "## Selected Validation Rules",
        "",
        r.to_markdown(index=False),
        "",
        "## Bad Versus Good Validation Cycles",
        "",
        fs.to_markdown(index=False) if not fs.empty else "No feature summary available.",
        "",
    ])


if __name__ == "__main__":
    wf = sys.argv[1] if len(sys.argv) > 1 else "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/walk_forward_20260706T102017Z"
    main(wf)
