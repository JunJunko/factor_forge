"""Window-length scan: does the low-volume-rise structure strengthen as the event window
lengthens from 2 days toward 120 days?

Reuses the cached task-5 dataset (``supply_v2_ic_dataset.parquet``), which already carries
``excess_ret_{1,2,3,5,10}``, ``scarcity`` (120d OLS), ``recent_volume_z_max_2_clip``,
``price_strength_2``, ``vol20`` and the h=5 label -- and only adds ``excess_ret_{20,60,120}``
computed on the same top-1000 panel. Reports:

  * RankIC of ``excess_ret_P`` for P in {1,2,3,5,10,20,60,120} -- does the price-strength
    signal flip from short-term reversal to mid-horizon structure as P grows?
  * the mid-horizon no-volume benchmark (``scarcity``, 120d OLS) and the v2 2-day signals
    for reference;
  * 2-D sort ``scarcity`` x ``excess_ret_P`` for P in {2, 20, 120} -- where is the
    "mid-horizon rise + mid-horizon no-volume" interaction strongest?

Usage:  python scripts/supply_v2_window_scan.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository
from factor_forge.ml import supply_features as sf
from factor_forge.ml.supply_ic import compute_factor_ic, quantile_2d_sort

WINDOWS = [1, 2, 3, 5, 10, 20, 60, 120]
EXTRA_WINDOWS = [20, 60, 120]
EVAL_START, EVAL_END = "2017-01-01", "2026-06-30"


def main() -> None:
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

    log("loading cached task-5 dataset")
    ds = pd.read_parquet("supply_v2_ic_dataset.parquet")
    log(f"dataset rows={len(ds)} cols={len(ds.columns)}")

    log("loading panel (top 1000) to compute long-horizon excess returns")
    proj = load_project("configs/project.yaml")
    repo = DataVersionRepository(proj.paths.data_root, proj.paths.metadata_db)
    _, panel = repo.load_panel("latest")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    liquid = panel.groupby("ts_code")["amount_cny"].median().nlargest(1000).index
    sub = (
        panel[panel["ts_code"].isin(liquid)]
        .sort_values(["ts_code", "trade_date"])
        .reset_index(drop=True)
    )
    stocks = sub["ts_code"]
    dates = pd.to_datetime(sub["trade_date"])
    industries = sub["industry_l1_code"]
    excess = sf.excess_returns(sub["adj_close"], stocks, dates, industries, windows=EXTRA_WINDOWS)
    extra = pd.DataFrame({"datetime": dates.to_numpy(), "instrument": stocks.to_numpy()})
    for P in EXTRA_WINDOWS:
        extra[f"excess_ret_{P}"] = excess[P].to_numpy()
    ds = ds.merge(extra, on=["datetime", "instrument"], how="left")
    log(f"merged long-horizon columns; rows={len(ds)}")

    factors = [f"excess_ret_{P}" for P in WINDOWS] + [
        "scarcity", "recent_volume_z_max_2_clip", "price_strength_2",
    ]
    log(f"IC scan @ h=5 over {EVAL_START}..{EVAL_END}")
    ic = compute_factor_ic(ds, factors, EVAL_START, EVAL_END, label_col="label", n_lag=4)
    show = ic[["rank_ic_mean", "rank_ic_newey_t", "rank_ic_positive_ratio", "n_days"]].round(4)
    print("\n=== window-scan IC (h=5) ===")
    print(show.to_string())

    log("2-D sorts: scarcity (mid-horizon no-volume) x excess_ret_P")
    grids = {
        P: quantile_2d_sort(ds, f"excess_ret_{P}", "scarcity", EVAL_START, EVAL_END, n_bins=5)
        for P in [2, 20, 120]
    }

    pd.set_option("display.width", 200)
    L = [
        "# 窗口扫描：无量上涨结构的事件窗口长度敏感性\n",
        f"- universe: 流动性 top 1000；窗口 {EVAL_START} ~ {EVAL_END}；标签 h=5 行业超额（开到开 LOO）",
        "- 动机：任务 5 显示 `price_strength_2` 是 2 日反转（IC −0.014）。本扫描把价格上涨窗口 P 从 1 扫到 120，",
        "  看信号是否随窗口由「反转」转为「结构」，以及中期无量（`scarcity` 120d OLS）× 上涨的交互在哪里最强。\n",
        "## 1. 价格强度 RankIC 随事件窗口 P\n",
        "> `excess_ret_P` = P 日行业超额（LOO，未做波动率标准化）。看 IC 符号与强度随 P 的变化。",
        "> 参照：`scarcity`（v1，120d 条件成交收缩 OLS 残差）、`recent_volume_z_max_2_clip`（v2，2 日）、`price_strength_2`（v2）。\n",
        show.to_markdown(),
        "\n\n## 2. 中期无量 × 价格强度二维分组（行 = 上涨强度五分位，列 = scarcity 五分位）\n",
        "> 若 thesis 成立，每一行（固定上涨强度）应从左到右递增（scarcity 越高 = 收缩越强 = 未来越好）。\n",
    ]
    for P in [2, 20, 120]:
        L.append(f"\n### 行 = excess_ret_{P}（{P} 日上涨强度）× 列 = scarcity（120d 无量）")
        L.append(grids[P].round(5).to_markdown())
    L.append("\n## 3. 解读要点\n")
    L.append("- 第 1 节：若 `excess_ret_P` 的 IC 随 P 增大由负转正（或由弱转强），说明无量上涨结构需要更长窗口；P=2 的反转是周期太短的伪影。")
    L.append("- 第 2 节：关注每行（固定上涨强度）从左到右是否单调递增——这是「相同上涨下，越无量未来越好」的直接检验（交接 §6.1）。")
    L.append("- 第 2 节还要跨 P 比较：P=20/120 的交互是否比 P=2 更清晰。")
    L.append("- 注意：`scarcity` 已含中期（120d）信息，它本身 IC +0.012 是基准；这里看的是它与上涨窗口的搭配。\n")
    Path("window_scan_report.md").write_text("\n".join(L), encoding="utf-8")
    Path("window_scan_ic.csv").write_text(ic.to_csv(encoding="utf-8-sig"), encoding="utf-8")
    log("-> window_scan_report.md, window_scan_ic.csv")


if __name__ == "__main__":
    main()
