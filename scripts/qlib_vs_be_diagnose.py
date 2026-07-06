"""Diagnose the Qlib-vs-BacktestEngine 2x divergence on the SAME Model B signal.

Decomposes the gap by reporting, on identical predictions:
- Qlib gross (from `return`) vs Qlib net (return - cost) vs the account/NAV growth;
- BacktestEngine at cost=0 (gross) and cost=20bps (net), at holding_days=5 (config) and
  holding_days=1 (daily rotation, closer to TopkDropout mechanics).
Benchmarks are removed by comparing GROSS/NET portfolio returns, not excess-vs-benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.backtest.engine import BacktestEngine
from factor_forge.config import CostModel, ExecutionConstraints, load_project
from factor_forge.data.repository import DataVersionRepository

RUN = Path("artifacts/supply_contraction_runs/supply_run_top1000_20260705T072422Z_813efa50")
TEST_START, TEST_END = "2023-01-01", "2026-06-30"


def _ann_sharpe(daily: pd.Series) -> dict:
    daily = daily.dropna()
    if daily.empty:
        return {"ann_return": float("nan"), "sharpe": float("nan")}
    ann = float((1 + daily).prod() ** (252.0 / len(daily)) - 1.0)
    vol = float(daily.std(ddof=1) * np.sqrt(252))
    return {"ann_return": ann, "sharpe": float(ann / vol) if vol > 0 else float("nan")}


def main():
    port = pd.read_parquet(RUN / "portfolio_daily_model_b_supply.parquet")
    port["datetime"] = pd.to_datetime(port["datetime"])
    port = port[(port["datetime"] >= TEST_START) & (port["datetime"] <= TEST_END)].copy()
    # Qlib: return is GROSS; net daily = return - cost; NAV growth is the ground truth.
    qlib_gross = _ann_sharpe(port["return"].astype(float))
    qlib_net = _ann_sharpe((port["return"] - port["cost"]).astype(float))
    nav_growth = port["account"].iloc[-1] / port["account"].iloc[0] - 1
    avg_turnover = float(port["turnover"].mean())
    avg_cost_bps = float(port["cost"].mean() * 10_000)

    # ---- BacktestEngine on the same signal ----
    pred = pd.read_parquet(RUN / "predictions_model_b_supply.parquet")
    pred["trade_date"] = pd.to_datetime(pred["trade_date"])
    imap = json.loads((RUN / "instrument_map.json").read_text())
    reverse = {v: k for k, v in imap.items()}
    pred["ts_code"] = pred["qlib_code"].map(reverse)
    predictions = pred[["trade_date", "ts_code", "factor_value"]].dropna()

    proj = load_project("configs/project.yaml")
    repo = DataVersionRepository(proj.paths.data_root, proj.paths.metadata_db)
    _v, panel = repo.load_panel("latest")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    med = panel.groupby("ts_code")["amount_cny"].median()
    panel = panel[panel["ts_code"].isin(med.nlargest(1000).index)]
    test_panel = panel[(panel["trade_date"] >= TEST_START) & (panel["trade_date"] <= TEST_END)].copy()

    be_rows = []
    for cost_bps in (0, 20):
        for hold in (5, 1):
            res = BacktestEngine().run(
                test_panel, predictions,
                universe="tradeable", top_n=50, holding_days=hold,
                initial_cash=10_000_000, lot_size=100,
                constraints=ExecutionConstraints(), cost_model=CostModel(),
                cost_scenario_bps=cost_bps,
            )
            be_rows.append({
                "config": f"BE cost={cost_bps}bps hold={hold}d",
                "ann_return": res.metrics.get("annualized_return"),
                "sharpe": res.metrics.get("sharpe"),
                "turnover": res.daily.get("portfolio_turnover", pd.Series()).mean() if hasattr(res.daily, "get") else None,
            })

    print("================ Qlib vs BacktestEngine (Model B, same signal) ================")
    print(f"{'config':35s} {'ann_return':>12s} {'sharpe':>8s} {'turnover':>10s}")
    print(f"{'Qlib GROSS (return col)':35s} {qlib_gross['ann_return']:>12.4f} {qlib_gross['sharpe']:>8.3f} {avg_turnover:>10.3f}")
    print(f"{'Qlib NET (return - cost)':35s} {qlib_net['ann_return']:>12.4f} {qlib_net['sharpe']:>8.3f} {avg_turnover:>10.3f}")
    print(f"{'Qlib NAV growth (truth)':35s} {nav_growth:>12.4f} {'':>8s} {avg_turnover:>10.3f}")
    print(f"{'  (avg daily cost, bps)':35s} {avg_cost_bps:>12.2f}")
    for r in be_rows:
        tv = r["turnover"]
        print(f"{r['config']:35s} {r['ann_return']:>12.4f} {r['sharpe']:>8.3f} {tv if tv is not None else float('nan'):>10.3f}")
    print("\n读法：")
    print("- Qlib NET 与 BE cost=20 同口径（都扣 20bps）应可比；若仍差很多 → 机制/接线差异；")
    print("- BE cost=0 vs cost=20 → 成本对 BE 的影响幅度；")
    print("- BE hold=1 vs hold=5 → 换手/持仓机制对收益的影响（TopkDropout 接近日换手）。")


if __name__ == "__main__":
    main()
