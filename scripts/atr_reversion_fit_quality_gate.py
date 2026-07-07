"""Online fit-quality gate for ATR lower-shadow reversion.

The gate uses only completed historical signal outcomes.  For a signal date T,
the rolling fit window includes a past signal S only when its 10-day exit open
is already observable by T.
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

from atr_reversion_pit_hmm_calibrated_backtest import _rank_states_from_validation, _tiered_weight
from atr_reversion_pit_regime_backtest import _run_regime_backtest_pit
from atr_reversion_small_portfolio_backtest import _json_default, _metrics
from atr_reversion_walk_forward import FOLDS, REBALANCE_DAYS


PIT_RUN = Path("artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z")
SOURCE_RUN = PIT_RUN / "hmm_window_comparison_20260707T002239Z"
BASELINE_RUN = PIT_RUN / "hmm_window_comparison_20260707T002239Z_csi1000_benchmark_20260707T020312Z"
HMM_VARIANT = "hmm_rolling_3y_pit"
BASE_POLICY = "atr_hmm_tiered"
TOP_N = 5
COSTS = [10, 20]
LOOKBACK = 40
MIN_OBS = 15


def main(
    source_run: str = str(SOURCE_RUN),
    baseline_run: str = str(BASELINE_RUN),
) -> None:
    source = Path(source_run)
    baseline = Path(baseline_run)
    output = PIT_RUN / f"fit_quality_gate_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
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
    pit = pd.read_parquet(PIT_RUN / "pit_liquidity_flags.parquet")
    pit["trade_date"] = pd.to_datetime(pit["trade_date"])
    log(f"loaded panel={len(panel):,} pit={len(pit):,} version={version}")

    rows: list[dict] = []
    yearly_frames: list[pd.DataFrame] = []
    gate_frames: list[pd.DataFrame] = []
    fit_frames: list[pd.DataFrame] = []

    policies: dict[str, Callable[[pd.Series], float]] = {
        "fit_quality_gate": _hard_gate,
        "fit_quality_gate_soft": _soft_gate,
    }
    directional_policies = {
        "fit_quality_flip_only": _flip_only_controls,
        "fit_quality_flip_guarded": _flip_guarded_controls,
    }

    for fold in FOLDS:
        fold_name = fold["name"]
        log(f"{fold_name}: prepare online fit quality")
        pred = pd.read_parquet(source / fold_name / "predictions_valid_test_rolling_2y.parquet")
        pred["trade_date"] = pd.to_datetime(pred["trade_date"])
        states = pd.read_csv(source / fold_name / HMM_VARIANT / "hmm_daily_states.csv")
        states["trade_date"] = pd.to_datetime(states["trade_date"])

        fit_daily = _daily_fit_metrics(panel, pit, pred, fold, log)
        fit_roll = _rolling_fit_quality(states["trade_date"], fit_daily)
        fit_daily["fold"] = fold_name
        fit_roll["fold"] = fold_name
        fit_frames.append(fit_daily)

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
                output / f"state_validation_perf_{fold_name}_cost{cost}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            for policy_name, gate_fn in policies.items():
                gates = fit_roll.copy()
                gates["cost_bps"] = cost
                gates["policy"] = policy_name
                gates["strategy_gate"] = gates.apply(gate_fn, axis=1).astype(float)
                gates["score_direction"] = 1.0
                gates.to_csv(
                    output / f"fit_quality_scores_{fold_name}_{policy_name}_cost{cost}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                gate_frames.append(gates)
                states_ext = states.merge(
                    gates[["trade_date", "strategy_gate"]],
                    on="trade_date",
                    how="left",
                )
                states_ext["strategy_gate"] = states_ext["strategy_gate"].fillna(1.0)
                policy = lambda row, ranks=ranks: _tiered_weight(row, ranks) * float(row.get("strategy_gate", 1.0))
                daily, trades = _run_regime_backtest_pit(
                    test_panel,
                    test_pred,
                    states_ext,
                    top_n=TOP_N,
                    rebalance_days=REBALANCE_DAYS,
                    cost_bps=cost,
                    policy=policy,
                )
                tag = f"{fold_name}_{policy_name}_top{TOP_N}_cost{cost}"
                daily.to_parquet(output / f"daily_{tag}.parquet", index=False)
                trades.to_parquet(output / f"trades_{tag}.parquet", index=False)
                metrics = _metrics(daily, trades)
                metrics.update(
                    {
                        "fold": fold_name,
                        "policy": policy_name,
                        "top_n": TOP_N,
                        "cost_bps": cost,
                        "hmm_variant": HMM_VARIANT,
                        "avg_exposure": float(daily["exposure"].mean()),
                        "avg_daily_turnover": float(daily["turnover"].mean()),
                        "avg_strategy_gate": float(gates.loc[_test_mask(gates, fold), "strategy_gate"].mean()),
                        "flat_gate_ratio": float(gates.loc[_test_mask(gates, fold), "strategy_gate"].eq(0.0).mean()),
                        "half_gate_ratio": float(gates.loc[_test_mask(gates, fold), "strategy_gate"].eq(0.5).mean()),
                    }
                )
                rows.append(metrics)
                yearly_frames.append(_yearly(metrics, daily))
                log(
                    f"{tag} ann={metrics['annualized_return']:.2%} "
                    f"bench={metrics['benchmark_annualized_return']:.2%} "
                    f"excess={metrics['annualized_excess_return']:.2%} "
                    f"gate={metrics['avg_strategy_gate']:.1%}"
                )

            for policy_name, control_fn in directional_policies.items():
                controls = fit_roll.copy()
                control_values = controls.apply(control_fn, axis=1, result_type="expand")
                controls["strategy_gate"] = control_values["strategy_gate"].astype(float)
                controls["score_direction"] = control_values["score_direction"].astype(float)
                controls["cost_bps"] = cost
                controls["policy"] = policy_name
                controls.to_csv(
                    output / f"fit_quality_scores_{fold_name}_{policy_name}_cost{cost}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                gate_frames.append(controls)
                adjusted_pred = _apply_score_direction(pred, controls)
                adjusted_valid_pred = adjusted_pred[
                    adjusted_pred["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["valid_end"]))
                ]
                adjusted_test_pred = adjusted_pred[
                    adjusted_pred["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))
                ]
                states_ext = states.merge(
                    controls[["trade_date", "strategy_gate"]],
                    on="trade_date",
                    how="left",
                )
                states_ext["strategy_gate"] = states_ext["strategy_gate"].fillna(1.0)
                valid_daily, _ = _run_regime_backtest_pit(
                    valid_panel,
                    adjusted_valid_pred,
                    states_ext,
                    top_n=TOP_N,
                    rebalance_days=REBALANCE_DAYS,
                    cost_bps=cost,
                    policy=lambda row: float(row.get("strategy_gate", 1.0)),
                )
                ranks, state_perf = _rank_states_from_validation(valid_daily, states_ext)
                state_perf.to_csv(
                    output / f"state_validation_perf_{fold_name}_{policy_name}_cost{cost}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                policy = lambda row, ranks=ranks: _tiered_weight(row, ranks) * float(row.get("strategy_gate", 1.0))
                daily, trades = _run_regime_backtest_pit(
                    test_panel,
                    adjusted_test_pred,
                    states_ext,
                    top_n=TOP_N,
                    rebalance_days=REBALANCE_DAYS,
                    cost_bps=cost,
                    policy=policy,
                )
                tag = f"{fold_name}_{policy_name}_top{TOP_N}_cost{cost}"
                daily.to_parquet(output / f"daily_{tag}.parquet", index=False)
                trades.to_parquet(output / f"trades_{tag}.parquet", index=False)
                metrics = _metrics(daily, trades)
                test_controls = controls.loc[_test_mask(controls, fold)]
                metrics.update(
                    {
                        "fold": fold_name,
                        "policy": policy_name,
                        "top_n": TOP_N,
                        "cost_bps": cost,
                        "hmm_variant": HMM_VARIANT,
                        "avg_exposure": float(daily["exposure"].mean()),
                        "avg_daily_turnover": float(daily["turnover"].mean()),
                        "avg_strategy_gate": float(test_controls["strategy_gate"].mean()),
                        "flat_gate_ratio": float(test_controls["strategy_gate"].eq(0.0).mean()),
                        "half_gate_ratio": float(test_controls["strategy_gate"].eq(0.5).mean()),
                        "flip_ratio": float(test_controls["score_direction"].lt(0.0).mean()),
                    }
                )
                rows.append(metrics)
                yearly_frames.append(_yearly(metrics, daily))
                log(
                    f"{tag} ann={metrics['annualized_return']:.2%} "
                    f"bench={metrics['benchmark_annualized_return']:.2%} "
                    f"excess={metrics['annualized_excess_return']:.2%} "
                    f"gate={metrics['avg_strategy_gate']:.1%} "
                    f"flip={metrics['flip_ratio']:.1%}"
                )

    metrics_df = pd.DataFrame(rows)
    yearly_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    gates_df = pd.concat(gate_frames, ignore_index=True) if gate_frames else pd.DataFrame()
    fit_df = pd.concat(fit_frames, ignore_index=True) if fit_frames else pd.DataFrame()
    baseline_df = _load_baseline_metrics(baseline)
    comparison = _comparison(metrics_df, baseline_df)
    metrics_df.to_csv(output / "fit_quality_gate_metrics.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(output / "fit_quality_gate_yearly.csv", index=False, encoding="utf-8-sig")
    gates_df.to_csv(output / "fit_quality_gate_scores.csv", index=False, encoding="utf-8-sig")
    fit_df.to_csv(output / "fit_quality_daily_metrics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "fit_quality_gate_comparison.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "source_run": str(source),
                "baseline_run": str(baseline),
                "run_dir": str(output),
                "lookback": LOOKBACK,
                "min_obs": MIN_OBS,
                "metrics": metrics_df.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(metrics_df, yearly_df, comparison, gates_df), encoding="utf-8")
    log("done")
    print(f"run_dir={output}")


def _load_panel() -> tuple[str, pd.DataFrame]:
    project = load_project("configs/project.yaml")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest("latest")
    _, panel = repo.load_panel(version)
    return version, panel


def _daily_fit_metrics(panel: pd.DataFrame, pit: pd.DataFrame, pred: pd.DataFrame, fold: dict, log) -> pd.DataFrame:
    start = pd.Timestamp(fold["valid_start"])
    end = pd.Timestamp(fold["test_end"])
    p = panel[panel["trade_date"].between(start, end)].merge(
        pit[["trade_date", "ts_code", "pit_top1000"]],
        on=["trade_date", "ts_code"],
        how="left",
    )
    p["pit_top1000"] = p["pit_top1000"].fillna(False).astype(bool)
    data = p.merge(pred, on=["trade_date", "ts_code"], how="left").sort_values(["ts_code", "trade_date"])
    data["next_open"] = data.groupby("ts_code")["adj_open"].shift(-1)
    data["exit_open"] = data.groupby("ts_code")["adj_open"].shift(-(REBALANCE_DAYS + 1))
    data["exit_date"] = data.groupby("ts_code")["trade_date"].shift(-(REBALANCE_DAYS + 1))
    data["fwd_ret"] = data["exit_open"] / data["next_open"] - 1.0
    eligible = (
        data["pit_top1000"]
        & data["factor_value"].notna()
        & data["fwd_ret"].replace([np.inf, -np.inf], np.nan).notna()
        & data["exit_date"].notna()
    )
    data = data.loc[eligible].copy()
    rows = []
    for date, g in data.groupby("trade_date", sort=True):
        if len(g) < 100:
            continue
        ranked = g["factor_value"].rank(method="first")
        decile = pd.qcut(ranked, 10, labels=False, duplicates="drop")
        g = g.assign(decile=decile)
        dec = g.groupby("decile", observed=True)["fwd_ret"].mean()
        decile_spread = float(dec.loc[dec.index.max()] - dec.loc[dec.index.min()]) if len(dec) >= 2 else np.nan
        top = g.nlargest(TOP_N, "factor_value")
        rows.append(
            {
                "trade_date": date,
                "known_date": pd.to_datetime(g["exit_date"].max()),
                "rank_ic": float(g["factor_value"].corr(g["fwd_ret"], method="spearman")),
                "top5_forward_return": float(top["fwd_ret"].mean()),
                "top5_hit_rate": float((top["fwd_ret"] > 0.0).mean()),
                "universe_forward_return": float(g["fwd_ret"].mean()),
                "top5_excess_forward_return": float(top["fwd_ret"].mean() - g["fwd_ret"].mean()),
                "decile_spread": decile_spread,
            }
        )
    out = pd.DataFrame(rows).sort_values("trade_date")
    log(f"{fold['name']}: daily fit rows={len(out)}")
    return out


def _rolling_fit_quality(state_dates: pd.Series, fit_daily: pd.DataFrame) -> pd.DataFrame:
    fit = fit_daily.sort_values("known_date").reset_index(drop=True)
    rows = []
    for date in pd.to_datetime(state_dates).sort_values():
        hist = fit[fit["known_date"].le(date)].tail(LOOKBACK)
        row = {
            "trade_date": date,
            "fit_obs": int(len(hist)),
            "rank_ic_rolling": np.nan,
            "rank_ic_positive_ratio": np.nan,
            "top5_excess_rolling": np.nan,
            "top5_forward_rolling": np.nan,
            "top5_hit_rolling": np.nan,
            "decile_spread_rolling": np.nan,
        }
        if len(hist):
            row.update(
                {
                    "rank_ic_rolling": float(hist["rank_ic"].mean()),
                    "rank_ic_positive_ratio": float((hist["rank_ic"] > 0.0).mean()),
                    "top5_excess_rolling": float(hist["top5_excess_forward_return"].mean()),
                    "top5_forward_rolling": float(hist["top5_forward_return"].mean()),
                    "top5_hit_rolling": float(hist["top5_hit_rate"].mean()),
                    "decile_spread_rolling": float(hist["decile_spread"].mean()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _hard_gate(row: pd.Series) -> float:
    if int(row.get("fit_obs", 0)) < MIN_OBS:
        return 1.0
    ic_bad = float(row.get("rank_ic_rolling", 0.0)) < 0.0
    top_bad = float(row.get("top5_excess_rolling", 0.0)) < 0.0
    slope_bad = float(row.get("decile_spread_rolling", 0.0)) < 0.0
    hit_bad = float(row.get("top5_hit_rolling", 0.5)) < 0.45
    if (ic_bad and top_bad) or (slope_bad and top_bad):
        return 0.0
    if top_bad or ic_bad or slope_bad or hit_bad:
        return 0.5
    return 1.0


def _soft_gate(row: pd.Series) -> float:
    if int(row.get("fit_obs", 0)) < MIN_OBS:
        return 1.0
    warnings = sum(
        [
            float(row.get("rank_ic_rolling", 0.0)) < 0.0,
            float(row.get("top5_excess_rolling", 0.0)) < 0.0,
            float(row.get("decile_spread_rolling", 0.0)) < 0.0,
            float(row.get("top5_hit_rolling", 0.5)) < 0.45,
        ]
    )
    if warnings >= 3:
        return 0.25
    if warnings >= 1:
        return 0.5
    return 1.0


def _flip_only_controls(row: pd.Series) -> pd.Series:
    if int(row.get("fit_obs", 0)) < MIN_OBS:
        return pd.Series({"strategy_gate": 1.0, "score_direction": 1.0})
    flip = float(row.get("rank_ic_rolling", 0.0)) < 0.0 and float(row.get("decile_spread_rolling", 0.0)) < 0.0
    return pd.Series({"strategy_gate": 1.0, "score_direction": -1.0 if flip else 1.0})


def _flip_guarded_controls(row: pd.Series) -> pd.Series:
    if int(row.get("fit_obs", 0)) < MIN_OBS:
        return pd.Series({"strategy_gate": 1.0, "score_direction": 1.0})
    ic_bad = float(row.get("rank_ic_rolling", 0.0)) < 0.0
    top_bad = float(row.get("top5_excess_rolling", 0.0)) < 0.0
    slope_bad = float(row.get("decile_spread_rolling", 0.0)) < 0.0
    hit_bad = float(row.get("top5_hit_rolling", 0.5)) < 0.45
    if ic_bad and slope_bad:
        return pd.Series({"strategy_gate": 1.0, "score_direction": -1.0})
    if top_bad or hit_bad:
        return pd.Series({"strategy_gate": 0.5, "score_direction": 1.0})
    return pd.Series({"strategy_gate": 1.0, "score_direction": 1.0})


def _apply_score_direction(pred: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    out = pred.merge(
        controls[["trade_date", "score_direction"]],
        on="trade_date",
        how="left",
    )
    out["score_direction"] = out["score_direction"].fillna(1.0)
    out["factor_value"] = out["factor_value"] * out["score_direction"]
    return out.drop(columns=["score_direction"])


def _test_mask(df: pd.DataFrame, fold: dict) -> pd.Series:
    return df["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))


def _yearly(meta: dict, daily: pd.DataFrame) -> pd.DataFrame:
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
                "policy": meta["policy"],
                "top_n": meta["top_n"],
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


def _load_baseline_metrics(baseline: Path) -> pd.DataFrame:
    df = pd.read_csv(baseline / "hmm_window_metrics.csv")
    return df[
        df["hmm_variant"].eq(HMM_VARIANT)
        & df["policy"].eq(BASE_POLICY)
        & df["top_n"].eq(TOP_N)
        & df["cost_bps"].isin(COSTS)
    ].copy()


def _comparison(metrics: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    base = baseline[[
        "fold",
        "cost_bps",
        "annualized_return",
        "benchmark_annualized_return",
        "annualized_excess_return",
        "sharpe",
        "max_drawdown",
        "avg_exposure",
    ]].rename(
        columns={
            "annualized_return": "base_ann",
            "benchmark_annualized_return": "benchmark_ann",
            "annualized_excess_return": "base_excess",
            "sharpe": "base_sharpe",
            "max_drawdown": "base_max_drawdown",
            "avg_exposure": "base_exposure",
        }
    )
    cur = metrics[[
        "fold",
        "policy",
        "cost_bps",
        "annualized_return",
        "annualized_excess_return",
        "sharpe",
        "max_drawdown",
        "avg_exposure",
        "avg_strategy_gate",
        "flat_gate_ratio",
        "half_gate_ratio",
        "flip_ratio",
    ]]
    out = cur.merge(base, on=["fold", "cost_bps"], how="left")
    out["delta_ann"] = out["annualized_return"] - out["base_ann"]
    out["delta_excess"] = out["annualized_excess_return"] - out["base_excess"]
    out["delta_drawdown"] = out["max_drawdown"] - out["base_max_drawdown"]
    return out


def _fmt_pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    return out


def _report(metrics: pd.DataFrame, yearly: pd.DataFrame, comparison: pd.DataFrame, gates: pd.DataFrame) -> str:
    summary = metrics.groupby(["policy", "cost_bps"]).agg(
        mean_ann=("annualized_return", "mean"),
        mean_excess=("annualized_excess_return", "mean"),
        mean_sharpe=("sharpe", "mean"),
        worst_drawdown=("max_drawdown", "min"),
        mean_exposure=("avg_exposure", "mean"),
        mean_gate=("avg_strategy_gate", "mean"),
    ).reset_index()
    summary = _fmt_pct(summary, ["mean_ann", "mean_excess", "worst_drawdown", "mean_exposure", "mean_gate"])
    summary["mean_sharpe"] = summary["mean_sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    comp = _fmt_pct(
        comparison,
        [
            "annualized_return",
            "annualized_excess_return",
            "max_drawdown",
            "avg_exposure",
            "avg_strategy_gate",
            "flat_gate_ratio",
            "half_gate_ratio",
            "base_ann",
            "benchmark_ann",
            "base_excess",
            "base_max_drawdown",
            "base_exposure",
            "delta_ann",
            "delta_excess",
            "delta_drawdown",
        ],
    )
    y = _fmt_pct(yearly, ["return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"])
    test_gates = gates.copy()
    if not test_gates.empty:
        test_gates["year"] = pd.to_datetime(test_gates["trade_date"]).dt.year
        gate_summary = test_gates.groupby(["year", "policy", "cost_bps"]).agg(
            avg_gate=("strategy_gate", "mean"),
            flat_ratio=("strategy_gate", lambda s: float(s.eq(0.0).mean())),
            half_ratio=("strategy_gate", lambda s: float(s.eq(0.5).mean())),
            rank_ic_rolling=("rank_ic_rolling", "mean"),
            top5_excess_rolling=("top5_excess_rolling", "mean"),
            decile_spread_rolling=("decile_spread_rolling", "mean"),
        ).reset_index()
        gate_summary = _fmt_pct(gate_summary, ["avg_gate", "flat_ratio", "half_ratio", "top5_excess_rolling", "decile_spread_rolling"])
        gate_summary["rank_ic_rolling"] = gate_summary["rank_ic_rolling"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    else:
        gate_summary = pd.DataFrame()
    return "\n".join(
        [
            "# ATR Fit-Quality Gate",
            "",
            "Gate uses completed historical 10-day signal outcomes only.",
            "",
            "## Summary",
            "",
            summary.to_markdown(index=False),
            "",
            "## Versus Baseline",
            "",
            comp.to_markdown(index=False),
            "",
            "## Yearly",
            "",
            y.to_markdown(index=False),
            "",
            "## Gate Diagnostics",
            "",
            gate_summary.to_markdown(index=False) if not gate_summary.empty else "No gate diagnostics.",
            "",
        ]
    )


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else str(SOURCE_RUN)
    baseline = sys.argv[2] if len(sys.argv) > 2 else str(BASELINE_RUN)
    main(source, baseline)
