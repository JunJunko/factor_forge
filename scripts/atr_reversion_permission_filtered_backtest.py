"""Backtest the frozen ATR fit-quality rule on an investor-permission universe.

The restricted universe removes STAR Market, ChiNext, and Beijing Stock Exchange
names before fit-quality direction estimation and before candidate selection.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository

from atr_reversion_fit_quality_gate import (
    BASELINE_RUN,
    HMM_VARIANT,
    PIT_RUN,
    SOURCE_RUN,
    _apply_score_direction,
    _load_baseline_metrics,
    _yearly,
)
from atr_reversion_fit_quality_sensitivity import _comparison, _rolling_controls
from atr_reversion_pit_hmm_calibrated_backtest import _rank_states_from_validation, _tiered_weight
from atr_reversion_pit_regime_backtest import _run_regime_backtest_pit
from atr_reversion_small_portfolio_backtest import _json_default, _metrics
from atr_reversion_walk_forward import FOLDS, REBALANCE_DAYS


LOOKBACK = 40
MIN_OBS = 15
TOP_N = 5
COST = 20
POLICY = "fit_quality_flip_only_permission_filtered"


def main(source_run: str = str(SOURCE_RUN), baseline_run: str = str(BASELINE_RUN)) -> None:
    source = Path(source_run)
    baseline = Path(baseline_run)
    output = PIT_RUN / f"permission_filtered_frozen_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
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
    panel["permission_eligible"] = panel["ts_code"].map(_permission_eligible).astype(bool)
    log(f"loaded panel={len(panel):,} pit={len(pit):,} version={version}")
    log("excluded boards: STAR 688/689.SH, ChiNext 300/301/302.SZ, Beijing *.BJ")

    rows: list[dict] = []
    yearly_frames: list[pd.DataFrame] = []
    gate_frames: list[pd.DataFrame] = []
    fit_frames: list[pd.DataFrame] = []
    universe_frames: list[pd.DataFrame] = []

    for fold in FOLDS:
        fold_name = fold["name"]
        pred = pd.read_parquet(source / fold_name / "predictions_valid_test_rolling_2y.parquet")
        pred["trade_date"] = pd.to_datetime(pred["trade_date"])
        pred = pred[pred["ts_code"].map(_permission_eligible)].copy()
        states = pd.read_csv(source / fold_name / HMM_VARIANT / "hmm_daily_states.csv")
        states["trade_date"] = pd.to_datetime(states["trade_date"])

        fit_daily = _restricted_fit_metrics(panel, pit, pred, fold, log)
        controls = _rolling_controls(states["trade_date"], fit_daily, LOOKBACK, MIN_OBS)
        controls["fold"] = fold_name
        controls["lookback"] = LOOKBACK
        controls["min_obs"] = MIN_OBS
        controls["cost_bps"] = COST
        controls["policy"] = POLICY
        gate_frames.append(controls)
        fit_daily["fold"] = fold_name
        fit_frames.append(fit_daily)

        panel_bt = panel[
            panel["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["test_end"]))
        ].merge(pit, on=["trade_date", "ts_code"], how="left")
        panel_bt["pit_top1000"] = panel_bt["pit_top1000"].fillna(False).astype(bool)
        panel_bt["permission_eligible"] = panel_bt["permission_eligible"].fillna(False).astype(bool)
        panel_bt.loc[~panel_bt["permission_eligible"], "pit_top1000"] = False

        valid_panel = panel_bt[
            panel_bt["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["valid_end"]))
        ]
        test_panel = panel_bt[
            panel_bt["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))
        ]
        adjusted_pred = _apply_score_direction(pred, controls)
        valid_pred = adjusted_pred[
            adjusted_pred["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["valid_end"]))
        ]
        test_pred = adjusted_pred[
            adjusted_pred["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))
        ]
        states_ext = states.merge(controls[["trade_date", "strategy_gate"]], on="trade_date", how="left")
        states_ext["strategy_gate"] = states_ext["strategy_gate"].fillna(1.0)

        valid_daily, _ = _run_regime_backtest_pit(
            valid_panel,
            valid_pred,
            states_ext,
            top_n=TOP_N,
            rebalance_days=REBALANCE_DAYS,
            cost_bps=COST,
            policy=lambda row: float(row.get("strategy_gate", 1.0)),
        )
        ranks, _state_perf = _rank_states_from_validation(valid_daily, states_ext)
        daily, trades = _run_regime_backtest_pit(
            test_panel,
            test_pred,
            states_ext,
            top_n=TOP_N,
            rebalance_days=REBALANCE_DAYS,
            cost_bps=COST,
            policy=lambda row, ranks=ranks: _tiered_weight(row, ranks) * float(row.get("strategy_gate", 1.0)),
        )
        tag = f"{fold_name}_{POLICY}_top{TOP_N}_cost{COST}"
        daily.to_parquet(output / f"daily_{tag}.parquet", index=False)
        trades.to_parquet(output / f"trades_{tag}.parquet", index=False)
        metrics = _metrics(daily, trades)
        test_controls = controls[
            controls["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))
        ]
        row = {
            "fold": fold_name,
            "policy": POLICY,
            "top_n": TOP_N,
            "cost_bps": COST,
            "hmm_variant": HMM_VARIANT,
            "lookback": LOOKBACK,
            "min_obs": MIN_OBS,
            "avg_exposure": float(daily["exposure"].mean()),
            "avg_daily_turnover": float(daily["turnover"].mean()),
            "avg_strategy_gate": float(test_controls["strategy_gate"].mean()),
            "flip_ratio": float(test_controls["score_direction"].lt(0.0).mean()),
            **metrics,
        }
        rows.append(row)
        y = _yearly(row, daily)
        y["lookback"] = LOOKBACK
        y["min_obs"] = MIN_OBS
        yearly_frames.append(y)
        universe_frames.append(_universe_summary(panel_bt, pred, fold, fold_name))
        log(
            f"{tag} ann={metrics['annualized_return']:.2%} "
            f"excess={metrics['annualized_excess_return']:.2%} "
            f"maxdd={metrics['max_drawdown']:.2%} flip={row['flip_ratio']:.1%}"
        )

    metrics_df = pd.DataFrame(rows)
    yearly_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    gates_df = pd.concat(gate_frames, ignore_index=True) if gate_frames else pd.DataFrame()
    fit_df = pd.concat(fit_frames, ignore_index=True) if fit_frames else pd.DataFrame()
    universe_df = pd.concat(universe_frames, ignore_index=True) if universe_frames else pd.DataFrame()
    baseline_df = _load_baseline_metrics(baseline)
    baseline_df = baseline_df[baseline_df["cost_bps"].eq(COST)]
    comparison = _comparison(metrics_df, baseline_df)

    metrics_df.to_csv(output / "permission_filtered_metrics.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(output / "permission_filtered_yearly.csv", index=False, encoding="utf-8-sig")
    gates_df.to_csv(output / "permission_filtered_gate_scores.csv", index=False, encoding="utf-8-sig")
    fit_df.to_csv(output / "permission_filtered_fit_daily.csv", index=False, encoding="utf-8-sig")
    universe_df.to_csv(output / "permission_filtered_universe_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "permission_filtered_comparison.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "data_version": version,
                "source_run": str(source),
                "baseline_run": str(baseline),
                "run_dir": str(output),
                "policy": POLICY,
                "lookback": LOOKBACK,
                "min_obs": MIN_OBS,
                "top_n": TOP_N,
                "cost_bps": COST,
                "excluded_boards": ["STAR 688/689.SH", "ChiNext 300/301/302.SZ", "Beijing *.BJ"],
                "metrics": metrics_df.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(metrics_df, yearly_df, comparison, universe_df), encoding="utf-8")
    log("done")
    print(f"run_dir={output}")


def _load_panel() -> tuple[str, pd.DataFrame]:
    project = load_project("configs/project.yaml")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest("latest")
    _, panel = repo.load_panel(version)
    return version, panel


def _permission_eligible(ts_code: str) -> bool:
    code = str(ts_code)
    if code.endswith(".BJ"):
        return False
    if code.endswith(".SH") and code[:3] in {"688", "689"}:
        return False
    if code.endswith(".SZ") and code[:3] in {"300", "301", "302"}:
        return False
    return True


def _restricted_fit_metrics(panel: pd.DataFrame, pit: pd.DataFrame, pred: pd.DataFrame, fold: dict, log) -> pd.DataFrame:
    start = pd.Timestamp(fold["valid_start"])
    end = pd.Timestamp(fold["test_end"])
    p = panel[
        panel["trade_date"].between(start, end) & panel["permission_eligible"]
    ].merge(
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
                "eligible_count": int(len(g)),
            }
        )
    out = pd.DataFrame(rows).sort_values("trade_date")
    log(f"{fold['name']}: restricted fit rows={len(out)}")
    return out


def _universe_summary(panel_bt: pd.DataFrame, pred: pd.DataFrame, fold: dict, fold_name: str) -> pd.DataFrame:
    test = panel_bt[panel_bt["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))]
    by_date = test.groupby("trade_date").agg(
        pit_top1000_count=("pit_top1000", "sum"),
        permission_eligible_count=("permission_eligible", "sum"),
    )
    pred_count = pred[pred["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))].groupby("trade_date").size()
    out = by_date.join(pred_count.rename("predictable_permission_candidates"), how="left").reset_index()
    out["fold"] = fold_name
    return out


def _report(metrics: pd.DataFrame, yearly: pd.DataFrame, comparison: pd.DataFrame, universe: pd.DataFrame) -> str:
    show = metrics[[
        "fold", "annualized_return", "benchmark_annualized_return", "annualized_excess_return",
        "sharpe", "max_drawdown", "avg_exposure", "avg_daily_turnover", "flip_ratio",
        "trade_count",
    ]].copy()
    for col in [
        "annualized_return", "benchmark_annualized_return", "annualized_excess_return",
        "max_drawdown", "avg_exposure", "avg_daily_turnover", "flip_ratio",
    ]:
        show[col] = show[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    show["sharpe"] = show["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    y = yearly.copy()
    for col in ["return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"]:
        y[col] = y[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    c = comparison[[
        "fold", "annualized_return", "base_ann", "delta_ann",
        "annualized_excess_return", "base_excess", "delta_excess", "max_drawdown",
    ]].copy()
    for col in [
        "annualized_return", "base_ann", "delta_ann",
        "annualized_excess_return", "base_excess", "delta_excess", "max_drawdown",
    ]:
        c[col] = c[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    u = universe.groupby("fold").agg(
        avg_pit_top1000_count=("pit_top1000_count", "mean"),
        avg_predictable_permission_candidates=("predictable_permission_candidates", "mean"),
    ).reset_index()
    return "\n".join(
        [
            "# Permission-Filtered Frozen ATR Reversion Backtest",
            "",
            "- universe: PIT rolling liquidity Top1000 after removing STAR, ChiNext, and Beijing Exchange names",
            "- rule: lookback=40, min_obs=15, flip when completed rolling RankIC < 0 and decile_spread < 0",
            "- portfolio: Top5, 10-day rebalance/holding, 20bps cost",
            "- benchmark: CSI1000 open-to-open; note it still includes names outside the investor-permission universe",
            "",
            "## Overall",
            "",
            show.to_markdown(index=False),
            "",
            "## Versus Unfiltered Baseline/Frozen Reference",
            "",
            c.to_markdown(index=False),
            "",
            "## Yearly",
            "",
            y.to_markdown(index=False),
            "",
            "## Universe Size",
            "",
            u.to_markdown(index=False),
            "",
        ]
    )


if __name__ == "__main__":
    main()
