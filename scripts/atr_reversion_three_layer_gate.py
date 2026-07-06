"""Three-layer strategy-aware gate for ATR lower-shadow reversion.

Layers:
1. Whether the market is rewarding reversal / lower-shadow repair.
2. Whether the strategy signal has been healthy recently.
3. Whether selected names fit the current market mainline.
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
from atr_reversion_strategy_regime_mining import (
    _add_signal_features,
    _build_regime_features,
    _compare,
    _hhi,
)
from atr_reversion_walk_forward import FOLDS, REBALANCE_DAYS


TOP_N = 5
COSTS = [10, 20]
REALIZATION_LAG_DAYS = 10
HEALTH_ROUNDS = 5


def main(
    walk_forward_dir: str = "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/walk_forward_20260706T102017Z",
) -> None:
    wf_path = Path(walk_forward_dir)
    pit_run = wf_path.parent
    output = pit_run / f"three_layer_gate_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
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

    base_regime = _build_regime_features(panel, pit, dataset)
    base_regime = _add_lower_shadow_style(base_regime, dataset)
    base_regime.to_parquet(output / "base_three_layer_features.parquet", index=False)
    log(f"built base regime/style features rows={len(base_regime):,}")

    rows: list[dict] = []
    yearly_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []

    for fold in FOLDS:
        fold_name = fold["name"]
        fold_dir = wf_path / fold_name
        pred = pd.read_parquet(fold_dir / "predictions_valid_test.parquet")
        pred["trade_date"] = pd.to_datetime(pred["trade_date"])
        states = pd.read_csv(fold_dir / "hmm_daily_states.csv")
        states["trade_date"] = pd.to_datetime(states["trade_date"])
        fold_features = _add_signal_features(base_regime, pred, dataset, panel)
        fold_features = _add_top_selection_features(fold_features, pred, panel, pit)
        fold_features.to_parquet(output / f"three_layer_features_{fold_name}.parquet", index=False)

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
            log(f"{fold_name} cost={cost}: calibrating HMM and baseline health")
            valid_ungated, _ = _run_regime_backtest_pit(
                valid_panel,
                valid_pred,
                states,
                top_n=TOP_N,
                rebalance_days=REBALANCE_DAYS,
                cost_bps=cost,
                policy=lambda _row: 1.0,
            )
            ranks, _ = _rank_states_from_validation(valid_ungated, states)
            base_policy = lambda row, ranks=ranks: _tiered_weight(row, ranks)
            valid_base, valid_base_trades = _run_regime_backtest_pit(
                valid_panel,
                valid_pred,
                states,
                top_n=TOP_N,
                rebalance_days=REBALANCE_DAYS,
                cost_bps=cost,
                policy=base_policy,
            )
            test_base, _test_base_trades = _run_regime_backtest_pit(
                test_panel,
                test_pred,
                states,
                top_n=TOP_N,
                rebalance_days=REBALANCE_DAYS,
                cost_bps=cost,
                policy=base_policy,
            )
            health = _strategy_health_features(valid_base, test_base, fold_name, cost)
            score_input = fold_features.merge(health, on="trade_date", how="left")
            scored = _score_three_layers(score_input, fold_name, cost)
            scored.to_csv(output / f"gate_scores_{fold_name}_cost{cost}.csv", index=False, encoding="utf-8-sig")
            score_frames.append(scored)

            test_scores = scored[scored["trade_date"].between(
                pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"])
            )]
            states_ext = states.merge(
                test_scores[["trade_date", "strategy_gate", "gate_score"]],
                on="trade_date",
                how="left",
            )
            states_ext["strategy_gate"] = states_ext["strategy_gate"].fillna(1.0)
            policy: Callable[[pd.Series], float] = (
                lambda row, ranks=ranks: _tiered_weight(row, ranks) * float(row.get("strategy_gate", 1.0))
            )
            daily, trades = _run_regime_backtest_pit(
                test_panel,
                test_pred,
                states_ext,
                top_n=TOP_N,
                rebalance_days=REBALANCE_DAYS,
                cost_bps=cost,
                policy=policy,
            )
            tag = f"{fold_name}_three_layer_gate_top{TOP_N}_cost{cost}"
            daily.to_parquet(output / f"daily_{tag}.parquet", index=False)
            trades.to_parquet(output / f"trades_{tag}.parquet", index=False)

            metrics = _metrics(daily, trades)
            metrics.update({
                "fold": fold_name,
                "policy": "three_layer_gate",
                "top_n": TOP_N,
                "cost_bps": cost,
                "best_state": ranks["best"],
                "neutral_state": ranks["neutral"],
                "worst_state": ranks["worst"],
                "avg_exposure": float(daily["exposure"].mean()),
                "avg_daily_turnover": float(daily["turnover"].mean()),
                "avg_strategy_gate": float(test_scores["strategy_gate"].mean()),
                "full_gate_ratio": float(test_scores["strategy_gate"].eq(1.0).mean()),
                "half_gate_ratio": float(test_scores["strategy_gate"].eq(0.5).mean()),
                "flat_gate_ratio": float(test_scores["strategy_gate"].eq(0.0).mean()),
                "rule_text": "three-layer gate_score: >=2 full, =1 half, <=0 flat",
            })
            rows.append(metrics)
            yearly_frames.append(_yearly(metrics, daily))
            log(
                f"{tag} ann={metrics['annualized_return']:.2%} "
                f"excess={metrics['annualized_excess_return']:.2%} "
                f"sharpe={metrics['sharpe']:.2f} maxdd={metrics['max_drawdown']:.2%} "
                f"avg_gate={metrics['avg_strategy_gate']:.1%}"
            )

    metrics_df = pd.DataFrame(rows)
    yearly = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    baseline = _load_baseline(wf_path)
    comparison = _compare(metrics_df, baseline)
    score_summary = _score_summary(scores)

    metrics_df.to_csv(output / "three_layer_gate_metrics.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "three_layer_gate_yearly.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(output / "three_layer_gate_scores.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "walk_forward_comparison.csv", index=False, encoding="utf-8-sig")
    score_summary.to_csv(output / "score_summary.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "data_version": version,
                "walk_forward_dir": str(wf_path),
                "pit_run": str(pit_run),
                "run_dir": str(output),
                "metrics": metrics_df.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(metrics_df, comparison, yearly, score_summary), encoding="utf-8")
    log("wrote three-layer gate metrics/report")
    log("done")
    print(f"run_dir={output}")


def _load_panel() -> tuple[str, pd.DataFrame]:
    project = load_project("configs/project.yaml")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest("latest")
    _, panel = repo.load_panel(version)
    return version, panel


def _add_lower_shadow_style(regime: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "datetime",
        "instrument",
        "pit_top1000",
        "label",
        "lower_shadow_atr",
        "intraday_repair",
        "core_signal",
    ]
    d = dataset[cols].rename(columns={"datetime": "trade_date", "instrument": "ts_code"}).copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    d = d[d["pit_top1000"].fillna(False).astype(bool)]
    rows = []
    for date, g in d.groupby("trade_date", sort=True):
        row = {"trade_date": date}
        for signal, name in [
            ("lower_shadow_atr", "lower_shadow_style_raw"),
            ("intraday_repair", "repair_style_raw"),
            ("core_signal", "core_signal_style_raw"),
        ]:
            h = g[[signal, "label"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(h) < 100 or h[signal].nunique() < 5:
                row[name] = np.nan
                continue
            high = h[signal] >= h[signal].quantile(0.8)
            low = h[signal] <= h[signal].quantile(0.2)
            row[name] = float(h.loc[high, "label"].mean() - h.loc[low, "label"].mean())
        rows.append(row)
    style = pd.DataFrame(rows).sort_values("trade_date")
    for raw, out in [
        ("lower_shadow_style_raw", "lower_shadow_style_20"),
        ("repair_style_raw", "repair_style_20"),
        ("core_signal_style_raw", "core_signal_style_20"),
    ]:
        style[out] = style[raw].rolling(20, min_periods=8).mean().shift(REALIZATION_LAG_DAYS)
    return regime.merge(
        style[["trade_date", "lower_shadow_style_20", "repair_style_20", "core_signal_style_20"]],
        on="trade_date",
        how="left",
    )


def _add_top_selection_features(
    features: pd.DataFrame,
    pred: pd.DataFrame,
    panel: pd.DataFrame,
    pit: pd.DataFrame,
) -> pd.DataFrame:
    p = panel[["trade_date", "ts_code", "adj_close", "industry_l1_code", "is_tradeable"]].copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    p = p.sort_values(["ts_code", "trade_date"])
    p["ret_20"] = p.groupby("ts_code")["adj_close"].pct_change(20, fill_method=None)
    p = p.merge(pit[["trade_date", "ts_code", "pit_top1000"]], on=["trade_date", "ts_code"], how="left")
    eligible = p["pit_top1000"].fillna(False).astype(bool) & p["is_tradeable"].fillna(False).astype(bool)
    p = p.loc[eligible].copy()
    p["selected_momentum_rank"] = p.groupby("trade_date")["ret_20"].rank(pct=True)
    industry = (
        p.groupby(["trade_date", "industry_l1_code"], observed=True)["ret_20"]
        .mean()
        .rename("industry_ret_20")
        .reset_index()
    )
    industry["selected_industry_strength"] = industry.groupby("trade_date")["industry_ret_20"].rank(pct=True)
    industry["is_hot_industry"] = industry["selected_industry_strength"].ge(0.8)
    p = p.merge(
        industry[["trade_date", "industry_l1_code", "selected_industry_strength", "is_hot_industry"]],
        on=["trade_date", "industry_l1_code"],
        how="left",
    )

    top = (
        pred.sort_values(["trade_date", "factor_value"], ascending=[True, False])
        .groupby("trade_date", sort=True)
        .head(TOP_N)
        .merge(
            p[[
                "trade_date",
                "ts_code",
                "industry_l1_code",
                "selected_momentum_rank",
                "selected_industry_strength",
                "is_hot_industry",
            ]],
            on=["trade_date", "ts_code"],
            how="left",
        )
    )
    top_features = top.groupby("trade_date", sort=True).agg(
        selected_momentum_rank=("selected_momentum_rank", "mean"),
        selected_industry_strength=("selected_industry_strength", "mean"),
        selected_hot_industry_ratio=("is_hot_industry", lambda s: float(s.astype("boolean").fillna(False).mean())),
        selected_industry_hhi=("industry_l1_code", _hhi),
    ).reset_index()
    return features.merge(top_features, on="trade_date", how="left")


def _strategy_health_features(
    valid_daily: pd.DataFrame,
    test_daily: pd.DataFrame,
    fold: str,
    cost: int,
) -> pd.DataFrame:
    cycles = pd.concat(
        [
            _base_cycles(valid_daily, fold, "valid", cost),
            _base_cycles(test_daily, fold, "test", cost),
        ],
        ignore_index=True,
    ).sort_values("signal_date")
    cycles["prev_excess"] = cycles["cycle_excess"].shift(1)
    cycles["top5_excess_5round"] = cycles["prev_excess"].rolling(HEALTH_ROUNDS, min_periods=2).mean()
    cycles["top5_winrate_5round"] = (
        cycles["prev_excess"].gt(0.0).astype(float).rolling(HEALTH_ROUNDS, min_periods=2).mean()
    )
    return cycles[[
        "signal_date",
        "top5_excess_5round",
        "top5_winrate_5round",
    ]].rename(columns={"signal_date": "trade_date"})


def _base_cycles(daily: pd.DataFrame, fold: str, segment: str, cost: int) -> pd.DataFrame:
    d = daily.copy().sort_values("trade_date").reset_index(drop=True)
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    rows = []
    for signal_idx in range(0, len(d) - 1, REBALANCE_DAYS):
        end_idx = min(signal_idx + REBALANCE_DAYS, len(d) - 1)
        if end_idx <= signal_idx:
            continue
        nav0 = float(d.loc[signal_idx, "nav"])
        nav1 = float(d.loc[end_idx, "nav"])
        strategy_return = nav1 / nav0 - 1.0 if nav0 > 0 else np.nan
        bench = float((1.0 + d.loc[signal_idx + 1 : end_idx, "benchmark_return"]).prod() - 1.0)
        rows.append({
            "fold": fold,
            "segment": segment,
            "cost_bps": cost,
            "signal_date": d.loc[signal_idx, "trade_date"],
            "cycle_return": strategy_return,
            "cycle_benchmark_return": bench,
            "cycle_excess": strategy_return - bench,
        })
    return pd.DataFrame(rows)


def _score_three_layers(features: pd.DataFrame, fold: str, cost: int) -> pd.DataFrame:
    f = features.copy().sort_values("trade_date")
    f["fold"] = fold
    f["cost_bps"] = cost

    reversal_good = (
        f["reversal_strength_20"].fillna(0.0).gt(0.0).astype(int)
        + f["lower_shadow_style_20"].fillna(0.0).gt(0.0).astype(int)
        + f["momentum_minus_reversal_20"].fillna(0.0).le(0.0).astype(int)
    )
    f["reversal_env_score"] = np.select(
        [reversal_good.ge(2), reversal_good.le(0)],
        [1, -1],
        default=0,
    )

    health_good = (
        f["signal_ic_60"].fillna(0.0).gt(0.0).astype(int)
        + f["top5_excess_5round"].fillna(0.0).gt(0.0).astype(int)
        + f["top5_winrate_5round"].fillna(0.5).ge(0.5).astype(int)
    )
    f["signal_health_score"] = np.select(
        [health_good.ge(2), health_good.le(1)],
        [1, -1],
        default=0,
    )

    strong_market = f["market_ret_20"].fillna(0.0).gt(0.02) | f["market_ret_60"].fillna(0.0).gt(0.06)
    selected_mainline = (
        f["selected_momentum_rank"].fillna(0.5).ge(0.50).astype(int)
        + f["selected_industry_strength"].fillna(0.5).ge(0.50).astype(int)
        + f["selected_hot_industry_ratio"].fillna(0.0).ge(0.40).astype(int)
    )
    mainline_rejected = strong_market & selected_mainline.le(1)
    f["mainline_fit_score"] = np.select(
        [mainline_rejected, selected_mainline.ge(2)],
        [-1, 1],
        default=0,
    )

    f["gate_score"] = f["reversal_env_score"] + f["signal_health_score"] + f["mainline_fit_score"]
    f["strategy_gate"] = np.select(
        [f["gate_score"].ge(2), f["gate_score"].eq(1)],
        [1.0, 0.5],
        default=0.0,
    )
    return f


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
        })
    return pd.DataFrame(rows)


def _load_baseline(wf_path: Path) -> pd.DataFrame:
    path = wf_path / "walk_forward_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[
        (df["policy"].eq("atr_hmm_tiered"))
        & (df["top_n"].eq(TOP_N))
        & (df["cost_bps"].isin(COSTS))
    ].copy()


def _score_summary(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    s = scores.copy()
    s["segment"] = "unknown"
    for fold in FOLDS:
        mask_valid = s["trade_date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["valid_end"])) & s["fold"].eq(fold["name"])
        mask_test = s["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"])) & s["fold"].eq(fold["name"])
        s.loc[mask_valid, "segment"] = "valid"
        s.loc[mask_test, "segment"] = "test"
    return s.groupby(["fold", "cost_bps", "segment"]).agg(
        avg_gate=("strategy_gate", "mean"),
        full_ratio=("strategy_gate", lambda x: float(x.eq(1.0).mean())),
        half_ratio=("strategy_gate", lambda x: float(x.eq(0.5).mean())),
        flat_ratio=("strategy_gate", lambda x: float(x.eq(0.0).mean())),
        reversal_score=("reversal_env_score", "mean"),
        health_score=("signal_health_score", "mean"),
        mainline_score=("mainline_fit_score", "mean"),
        signal_ic_60=("signal_ic_60", "mean"),
        top5_excess_5round=("top5_excess_5round", "mean"),
        selected_momentum_rank=("selected_momentum_rank", "mean"),
        selected_industry_strength=("selected_industry_strength", "mean"),
        selected_hot_industry_ratio=("selected_hot_industry_ratio", "mean"),
    ).reset_index()


def _fmt_pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    return out


def _report(
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    yearly: pd.DataFrame,
    score_summary: pd.DataFrame,
) -> str:
    m = metrics[[
        "fold",
        "cost_bps",
        "annualized_return",
        "annualized_excess_return",
        "sharpe",
        "max_drawdown",
        "avg_exposure",
        "avg_strategy_gate",
        "flat_gate_ratio",
    ]].copy()
    m = _fmt_pct(m, [
        "annualized_return",
        "annualized_excess_return",
        "max_drawdown",
        "avg_exposure",
        "avg_strategy_gate",
        "flat_gate_ratio",
    ])
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

    y = yearly.copy()
    if not y.empty:
        y = y[["fold", "cost_bps", "year", "return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"]]
        y = _fmt_pct(y, ["return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"])

    s = score_summary[score_summary["segment"].eq("test")].copy()
    if not s.empty:
        show_cols = [
            "fold",
            "cost_bps",
            "avg_gate",
            "flat_ratio",
            "reversal_score",
            "health_score",
            "mainline_score",
            "signal_ic_60",
            "top5_excess_5round",
            "selected_momentum_rank",
            "selected_industry_strength",
            "selected_hot_industry_ratio",
        ]
        s = s[show_cols]
        s = _fmt_pct(s, [
            "avg_gate",
            "flat_ratio",
            "signal_ic_60",
            "top5_excess_5round",
            "selected_momentum_rank",
            "selected_industry_strength",
            "selected_hot_industry_ratio",
        ])

    return "\n".join([
        "# ATR Three-Layer Strategy Gate",
        "",
        "The final exposure is ATR-HMM tiered exposure multiplied by a strategy gate.",
        "The gate combines reversal reward, recent strategy health, and mainline fit.",
        "",
        "## Gated Walk-Forward",
        "",
        m.to_markdown(index=False),
        "",
        "## Versus ATR-HMM Tiered Baseline",
        "",
        c.to_markdown(index=False) if not c.empty else "No baseline metrics found.",
        "",
        "## Yearly",
        "",
        y.to_markdown(index=False) if not y.empty else "No yearly metrics.",
        "",
        "## Test Score Summary",
        "",
        s.to_markdown(index=False) if not s.empty else "No score summary.",
        "",
    ])


if __name__ == "__main__":
    wf = sys.argv[1] if len(sys.argv) > 1 else "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/walk_forward_20260706T102017Z"
    main(wf)
