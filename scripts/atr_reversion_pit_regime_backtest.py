"""Apply existing HMM regime gates to PIT-liquidity ATR predictions."""

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

from atr_reversion_benchmark import csi1000_open_to_open_returns
from atr_reversion_pit_liquidity_backtest import _yearly_tables
from atr_reversion_regime_small_backtest import _run_regime_backtest
from atr_reversion_small_portfolio_backtest import _json_default, _metrics


TOP_NS = [5, 10]
COST_BPS = [10, 20]
REBALANCE_DAYS = 10


def main(
    pit_run: str = "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z",
    states_path: str = "artifacts/value_hmm_regime_validations/value_hmm_regime_v1_20260704T164011Z_4ef4866a/hmm_daily_states.csv",
) -> None:
    pit_run_path = Path(pit_run)
    out_dir = pit_run_path / f"pit_regime_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out_dir.mkdir(parents=True, exist_ok=False)
    log_path = out_dir / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    pred = pd.read_parquet(pit_run_path / "predictions_pit_all_features.parquet")
    pred["trade_date"] = pd.to_datetime(pred["trade_date"])
    pit = pd.read_parquet(pit_run_path / "pit_liquidity_flags.parquet")
    pit["trade_date"] = pd.to_datetime(pit["trade_date"])
    states = pd.read_csv(states_path)
    states["trade_date"] = pd.to_datetime(states["trade_date"])
    version, panel = _load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    test_panel = panel[
        panel["trade_date"].between(pred["trade_date"].min(), pred["trade_date"].max())
    ].merge(pit, on=["trade_date", "ts_code"], how="left")
    test_panel["pit_top1000"] = test_panel["pit_top1000"].fillna(False).astype(bool)
    log(f"loaded PIT pred rows={len(pred):,}; test panel rows={len(test_panel):,}; version={version}")

    policies = {
        "ungated": lambda row: 1.0,
        "hmm_position": lambda row: float(row.get("position_multiplier", 1.0)),
        "hmm_hard_full_only": lambda row: 1.0 if float(row.get("position_multiplier", 1.0)) >= 0.95 else 0.0,
    }
    rows = []
    for top_n in TOP_NS:
        for cost in COST_BPS:
            for policy_name, policy in policies.items():
                log(f"backtest policy={policy_name} top_n={top_n} cost={cost}")
                daily, trades = _run_regime_backtest_pit(
                    test_panel,
                    pred,
                    states,
                    top_n=top_n,
                    rebalance_days=REBALANCE_DAYS,
                    cost_bps=cost,
                    policy=policy,
                )
                tag = f"{policy_name}_top{top_n}_rebalance{REBALANCE_DAYS}_cost{cost}"
                daily.to_parquet(out_dir / f"daily_{tag}.parquet", index=False)
                trades.to_parquet(out_dir / f"trades_{tag}.parquet", index=False)
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
                })
                rows.append(metrics)
                log(
                    f"done ann={metrics['annualized_return']:.2%} "
                    f"excess={metrics['annualized_excess_return']:.2%} "
                    f"sharpe={metrics['sharpe']:.2f} maxdd={metrics['max_drawdown']:.2%}"
                )
    metrics_df = pd.DataFrame(rows).sort_values(
        ["annualized_excess_return", "sharpe"], ascending=False
    )
    metrics_df.to_csv(out_dir / "pit_regime_metrics.csv", index=False, encoding="utf-8-sig")
    yearly = _regime_yearly_tables(out_dir)
    yearly.to_csv(out_dir / "pit_regime_yearly_metrics.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "data_version": version,
                "pit_run": str(pit_run_path),
                "states_path": states_path,
                "run_dir": str(out_dir),
                "best": metrics_df.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(_report(metrics_df, yearly), encoding="utf-8")
    log("wrote metrics/yearly/report")
    log("done")
    print(f"run_dir={out_dir}")


def _run_regime_backtest_pit(panel, pred, states, *, top_n, rebalance_days, cost_bps, policy):
    daily, trades = _run_regime_backtest(
        panel,
        pred,
        states,
        top_n=top_n,
        rebalance_days=rebalance_days,
        cost_bps=cost_bps,
        policy=policy,
    )
    dates = list(pd.Index(daily["trade_date"].unique()).sort_values())
    daily["benchmark_return"] = csi1000_open_to_open_returns(dates)
    daily["excess_return"] = daily["return"] - daily["benchmark_return"]
    return daily, trades


def _regime_yearly_tables(out_dir: Path) -> pd.DataFrame:
    rows = []
    for path in out_dir.glob("daily_*.parquet"):
        tag = path.stem.removeprefix("daily_")
        parts = tag.split("_")
        policy = "_".join(parts[:-3])
        top_n = int(parts[-3].removeprefix("top"))
        rebalance = int(parts[-2].removeprefix("rebalance"))
        cost = int(parts[-1].removeprefix("cost"))
        df = pd.read_parquet(path)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        for year, g in df.groupby(df["trade_date"].dt.year):
            if len(g) < 2:
                continue
            total = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1
            bench = (1 + g["benchmark_return"]).prod() - 1
            dd = g["nav"] / g["nav"].cummax() - 1
            rows.append({
                "policy": policy,
                "top_n": top_n,
                "rebalance_days": rebalance,
                "cost_bps": cost,
                "year": int(year),
                "return": float(total),
                "benchmark_return": float(bench),
                "excess_return": float(total - bench),
                "max_drawdown": float(dd.min()),
                "avg_exposure": float(g["exposure"].mean()),
            })
    return pd.DataFrame(rows)


def _load_panel():
    project = load_project("configs/project.yaml")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest("latest")
    _, panel = repo.load_panel(version)
    return version, panel


def _report(metrics, yearly):
    show = metrics[[
        "policy", "top_n", "cost_bps", "annualized_return", "benchmark_annualized_return",
        "annualized_excess_return", "sharpe", "max_drawdown", "avg_exposure", "avg_daily_turnover",
    ]].copy()
    for col in ["annualized_return", "benchmark_annualized_return", "annualized_excess_return", "max_drawdown", "avg_exposure", "avg_daily_turnover"]:
        show[col] = show[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    show["sharpe"] = show["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    y = yearly.copy()
    for col in ["return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"]:
        y[col] = y[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    return "\n".join([
        "# ATR PIT Liquidity Backtest With HMM Regime Gates",
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
    pit_run = sys.argv[1] if len(sys.argv) > 1 else "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z"
    states_path = sys.argv[2] if len(sys.argv) > 2 else "artifacts/value_hmm_regime_validations/value_hmm_regime_v1_20260704T164011Z_4ef4866a/hmm_daily_states.csv"
    main(pit_run, states_path)
