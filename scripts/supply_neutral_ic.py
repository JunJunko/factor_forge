"""Independence check: does the scarcity family IC survive neutralizing for the controls?

Uses the cached IC dataset (supply_ic_dataset.parquet).  For each factor, compares the raw
daily IC against the IC after daily cross-sectional residualization on successively larger
control sets (volatility -> +size -> +liquidity).  If the IC collapses, the factor is just
a proxy for low-vol / size / liquidity; if it survives, it carries independent alpha.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.ml.supply_ic import compute_factor_ic, ic_of_series, neutralized_residual


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.1f}s] {m}", flush=True)
    cache = Path("supply_ic_dataset.parquet")
    if not cache.exists():
        raise SystemExit("missing supply_ic_dataset.parquet -- run supply_ic_look.py first")
    log("loading cached dataset")
    ds = pd.read_parquet(cache)
    ds["datetime"] = pd.to_datetime(ds["datetime"])
    eval_start, eval_end = "2017-01-01", "2026-06-30"

    factors = ["scarcity", "scarcity_days_ratio_5", "scarcity_slope_5", "volume_residual"]
    specs = [
        ("raw", []),
        ("neut: vol20", ["volatility_20"]),
        ("neut: vol20 + size", ["volatility_20", "log_float_market_cap"]),
        ("neut: vol20 + size + liq", ["volatility_20", "log_float_market_cap", "log_avg_amount_20"]),
        ("neut: vol20 + size + liq + turn_z", ["volatility_20", "log_float_market_cap", "log_avg_amount_20", "turnover_zscore_60"]),
    ]

    rows = []
    for f in factors:
        log(f"factor {f}")
        for spec_name, controls in specs:
            if not controls:
                ic = compute_factor_ic(ds, [f], eval_start, eval_end).iloc[0].to_dict()
            else:
                resid = neutralized_residual(ds, f, controls, eval_start, eval_end)
                ic = ic_of_series(ds, resid, eval_start, eval_end, name=f"{f}_neut")
            rows.append({
                "factor": f,
                "spec": spec_name,
                "rank_ic_mean": ic.get("rank_ic_mean"),
                "rank_ic_ir": ic.get("rank_ic_ir"),
                "rank_ic_newey_t": ic.get("rank_ic_newey_t"),
                "rank_ic_positive_ratio": ic.get("rank_ic_positive_ratio"),
            })

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    out = Path("supply_neutral_ic.md")
    lines = [
        "# scarcity 家族独立性检验（横截面中性化后 IC）\n",
        f"- universe: 流动性 top 1000；窗口 {eval_start} ~ {eval_end}",
        "- 每日横截面 OLS 残差化因子对控制变量，再算残差对 5 日行业中性标签的日频 RankIC。\n",
        table.to_markdown(floatfmt=".4f"),
        "\n## 读法\n",
        "- 沿每一行从上到下：若 IC 随控制变量加入而**坍缩到 ~0/不显著**，说明该因子只是控制变量的代理；",
        "- 若 IC **保持显著且方向不变**，则该因子对控制变量有**独立增量**。",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"report -> {out}")
    print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
