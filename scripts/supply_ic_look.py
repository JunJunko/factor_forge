"""One-off IC look for the supply-contraction factor family on the real data version.

Liquid universe (top 2000 by median amount) for speed; full 2016-2026 history so rolling
features warm up before the 2017-2026 evaluation window.  Writes a markdown + json report.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository
from factor_forge.ml.supply_ic import compute_factor_ic, load_dataset_for_ic, quantile_2d_sort


def main(out_path: str):
    out_path = Path(out_path)
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    log("loading panel")
    proj = load_project("configs/project.yaml")
    repo = DataVersionRepository(proj.paths.data_root, proj.paths.metadata_db)
    version, panel = repo.load_panel("latest")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    log(f"panel={panel.shape} version={version}")

    log("selecting liquid universe (top 1000 by median amount)")
    med = panel.groupby("ts_code")["amount_cny"].median()
    liquid = med.nlargest(1000).index
    sub = panel[panel["ts_code"].isin(liquid)].copy()
    log(f"liquid subset={sub.shape}")

    cache = Path("supply_ic_dataset.parquet")
    if cache.exists():
        log(f"loading cached dataset from {cache}")
        ds = pd.read_parquet(cache)
    else:
        log("building IC dataset (raw, no zscore)")
        ds = load_dataset_for_ic(sub)
        ds.to_parquet(cache)
        log(f"cached dataset -> {cache}")
    log(f"dataset rows={len(ds)}  cols={len(ds.columns)}")

    factors = [
        # supply core
        "volume_residual", "scarcity", "volume_residual_5d_mean",
        "scarcity_days_ratio_5", "scarcity_slope_5",
        "risk_adjusted_ret_5", "excess_ret_5", "price_impact_5", "up_days_ratio_5",
        # composites
        "simple_low_volume_rise", "conditional_scarcity_factor",
        "close_quality_scarcity_factor", "persistent_scarcity_factor",
        "price_adjusted_scarcity_factor",
        # controls (for comparison)
        "volatility_20", "log_float_market_cap", "log_avg_amount_20",
        "amihud_illiquidity_20", "market_breadth", "turnover_zscore_60",
    ]
    eval_start, eval_end = "2017-01-01", "2026-06-30"
    log(f"computing IC over {eval_start}..{eval_end}")
    ic = compute_factor_ic(ds, factors, eval_start, eval_end)
    log("computing 2D sort: excess_ret_5 x volume_residual")
    grid_vres = quantile_2d_sort(ds, "excess_ret_5", "volume_residual", eval_start, eval_end)
    grid_scar = quantile_2d_sort(ds, "excess_ret_5", "scarcity", eval_start, eval_end)

    # ---- write report ----
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    lines = []
    lines.append(f"# 供给收缩因子 IC 报告\n")
    lines.append(f"- 数据版本: `{version}`")
    lines.append(f"- universe: 流动性 top 2000（按中位成交额）")
    lines.append(f"- 评估窗口: {eval_start} ~ {eval_end}")
    lines.append(f"- 标签: 5 日行业中性前瞻收益（开到开，留一法行业）\n")
    lines.append("## 单因子 IC（日频）\n")
    lines.append("> scarcity/volume_residual 方向：scarcity 越大=供给收缩越强，预期 IC 为正；")
    lines.append("volume_residual 越负=越缩量，预期 IC 为负。\n")
    lines.append(ic.to_markdown(floatfmt=".4f"))
    lines.append("\n\n## 二维分组：excess_ret_5（行，价格上涨强度）× volume_residual（列）\n")
    lines.append("> 同一行（相同上涨强度）内，列从左到右 volume_residual 升高（缩量→放量）。")
    lines.append("若「越缩量未来收益越高」，则每行应从左到右递减。\n")
    lines.append(grid_vres.round(4).to_markdown())
    lines.append("\n\n## 二维分组：excess_ret_5 × scarcity（=-volume_residual）\n")
    lines.append("> 若 thesis 成立，每行应从左到右递增（scarcity 越大收益越高）。\n")
    lines.append(grid_scar.round(4).to_markdown())
    lines.append("\n")
    report = "\n".join(lines)
    out_path.write_text(report, encoding="utf-8")
    (out_path.with_suffix(".json")).write_text(
        json.dumps({
            "version": version,
            "universe": "liquid_top_2000",
            "eval_window": [eval_start, eval_end],
            "ic": ic.reset_index().to_dict(orient="records"),
        }, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    log(f"report -> {out_path}")
    print("\n" + ic.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "supply_ic_report.md")
