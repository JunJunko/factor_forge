"""Full factor diagnostics for the supply-contraction family (8 checks).

Uses the cached IC dataset (supply_ic_dataset.parquet, top 1000 liquid, raw features) and
recomputes multi-horizon industry-neutral labels from the panel.  Produces one markdown
report answering: best IC horizon vs holding, yearly stability, quantile monotonicity,
top-N structure, long/short decomposition, post-neutralization IC, net-of-cost, and
overlap with control factors.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository
from factor_forge.ml.supply_ic import (
    compute_factor_ic,
    factor_rank_correlation,
    multi_horizon_labels,
    neutralized_residual,
    quantile_decomp,
    topn_returns,
    yearly_ic,
)

EVAL_START, EVAL_END = "2017-01-01", "2026-06-30"
HEADLINE = "scarcity"


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    cache = Path("supply_ic_dataset.parquet")
    if not cache.exists():
        raise SystemExit("missing supply_ic_dataset.parquet -- run supply_ic_look.py first")
    log("loading cached dataset")
    ds = pd.read_parquet(cache)
    ds["datetime"] = pd.to_datetime(ds["datetime"])

    log("loading panel + multi-horizon labels")
    proj = load_project("configs/project.yaml")
    repo = DataVersionRepository(proj.paths.data_root, proj.paths.metadata_db)
    _v, panel = repo.load_panel("latest")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    med = panel.groupby("ts_code")["amount_cny"].median()
    panel = panel[panel["ts_code"].isin(med.nlargest(1000).index)].copy()
    labels = multi_horizon_labels(panel, [1, 3, 5, 10], method="open_to_open")
    for h, s in labels.items():
        ds[f"label_{h}d"] = s.reindex(pd.MultiIndex.from_frame(ds[["datetime", "instrument"]])).to_numpy()
    ds["label"] = ds["label_5d"]  # default for functions that read "label"

    lines = [
        "# 供给收缩因子完整诊断报告\n",
        f"- universe: 流动性 top 1000；评估窗口 {EVAL_START} ~ {EVAL_END}",
        f"- 主因子: `{HEADLINE}` (= -volume_residual)；标签: 行业中性 open-to-open 前瞻收益\n",
    ]

    # 1) multi-horizon IC
    log("1) multi-horizon IC")
    rows = []
    for f in [HEADLINE, "scarcity_slope_5", "scarcity_days_ratio_5", "volume_residual"]:
        for h in [1, 3, 5, 10]:
            ic = compute_factor_ic(ds, [f], EVAL_START, EVAL_END, label_col=f"label_{h}d").iloc[0]
            rows.append({"factor": f, "horizon": f"{h}d", "rank_ic": ic["rank_ic_mean"],
                         "icir": ic["rank_ic_ir"], "newey_t": ic["rank_ic_newey_t"]})
    mh = pd.DataFrame(rows)
    lines += ["## 1. IC 计算周期（1/3/5/10 日）\n",
              "> 找出最强周期，并看是否与持仓周期（5 日）一致。\n",
              mh.pivot(index="factor", columns="horizon", values="rank_ic").to_markdown(floatfmt=".4f"),
              "\n\n", mh.pivot(index="factor", columns="horizon", values="newey_t").to_markdown(floatfmt=".2f"),
              "\n"]

    # 2) yearly stability
    log("2) yearly IC stability")
    yr = yearly_ic(ds, HEADLINE, "label_5d", EVAL_START, EVAL_END)
    lines += ["## 2. 年度稳定性\n",
              "> 若某两年 IC 特别高、其余近 0，则因子不稳定、靠个别行情。\n",
              yr.to_markdown(floatfmt=".4f"), "\n"]

    # 3) quantile monotonicity
    log("3) quantile monotonicity")
    qd = quantile_decomp(ds, HEADLINE, "label_5d", EVAL_START, EVAL_END, n_bins=10)
    mono = qd["mean_fwd_return"].corr(pd.Series(range(len(qd)), index=range(len(qd)), dtype=float), method="spearman")
    lines += ["## 3. 分组单调性（Q1=最低 → Q10=最高 scarcity）\n",
              f"> 单调性 Spearman = {mono:.3f}（接近 +1 表示完美单调递增）。\n",
              (qd["mean_fwd_return"] * 10_000).round(2).to_frame("bps_per_5d").to_markdown(), "\n"]

    # 4 + 5) top-N structure + long/short
    log("4+5) top-N returns + long/short")
    tn = topn_returns(ds, HEADLINE, "label_5d", EVAL_START, EVAL_END, [50, 100, 200, 500])
    qd5 = quantile_decomp(ds, HEADLINE, "label_5d", EVAL_START, EVAL_END, n_bins=5)
    q_top = qd5["mean_fwd_return"].iloc[-1]
    q_bot = qd5["mean_fwd_return"].iloc[0]
    lines += ["## 4. 顶部结构（Top-N 5 日前瞻超额，单位 bps/5日）\n",
              "> 20bps 回合成本对应约 20 bps/持仓期；超额需 > 20 bps/5日 才扣成本后为正。\n",
              (tn[["avg_daily_fwd", "avg_universe_fwd", "avg_daily_excess"]] * 10_000).round(2).to_markdown(), "\n"]
    lines += ["## 5. 多空拆分（5 分组，bps/5日）\n",
              f"> 多头腿 Q5 = {q_top*1e4:.2f}；空头腿 Q1 = {q_bot*1e4:.2f}；",
              f"多空价差 = {(q_top-q_bot)*1e4:.2f}。\n",
              "> 价差由多头上涨还是空头下跌主导，看两腿相对 0 的位置。\n",
              (qd5["mean_fwd_return"] * 10_000).round(2).to_frame("bps_per_5d").to_markdown(), "\n"]

    # 6) neutralized
    log("6) neutralized IC")
    controls = ["volatility_20", "log_float_market_cap", "log_avg_amount_20", "turnover_zscore_60"]
    raw_ic = compute_factor_ic(ds, [HEADLINE], EVAL_START, EVAL_END, label_col="label_5d").iloc[0]["rank_ic_mean"]
    resid = neutralized_residual(ds, HEADLINE, controls, EVAL_START, EVAL_END)
    ds["_neut"] = resid.reindex(ds.index).to_numpy()
    neut_ic = compute_factor_ic(ds, ["_neut"], EVAL_START, EVAL_END, label_col="label_5d").iloc[0]
    lines += ["## 6. 中性化后表现\n",
              f"> 控制 {controls} 后 `{HEADLINE}` 的 IC。\n",
              f"- 原始 RankIC: **{raw_ic:.4f}**",
              f"- 中性化后 RankIC: **{neut_ic['rank_ic_mean']:.4f}** (ICIR {neut_ic['rank_ic_ir']:.2f}, Newey t {neut_ic['rank_ic_newey_t']:.2f})",
              f"- 存留比例: {neut_ic['rank_ic_mean']/raw_ic*100:.0f}%\n"]

    # 7) net of cost (use Top-100 as a tractable long leg)
    log("7) net of cost")
    tn100 = topn_returns(ds, HEADLINE, "label_5d", EVAL_START, EVAL_END, [100]).iloc[0]
    gross_per_period_bps = tn100["avg_daily_excess"] * 10_000
    net_per_period_bps = gross_per_period_bps - 20.0
    lines += ["## 7. 扣成本收益（Top-100 多头，5 日持仓，20bps 回合成本）\n",
              f"- 每 5 日毛超额: {gross_per_period_bps:.2f} bps",
              f"- 每 5 日净超额: {net_per_period_bps:.2f} bps  (>0 则扣成本后仍有效)",
              f"- 年化毛超额(近似, 50 个持仓期/年): {((1+tn100['avg_daily_excess'])**50-1)*100:.2f}%\n"]

    # 8) correlation with control factors
    log("8) correlation with controls")
    corr = factor_rank_correlation(ds, HEADLINE,
        ["volatility_20", "log_float_market_cap", "log_avg_amount_20", "turnover_zscore_20",
         "turnover_zscore_60", "amihud_illiquidity_20", "amount_zscore_20", "excess_ret_5",
         "risk_adjusted_ret_5", "price_impact_5", "close_location"],
        EVAL_START, EVAL_END)
    lines += ["## 8. 与控制/已有因子的相关性（时序平均横截面 Spearman）\n",
              "> |corr| 高（>0.5）说明与现有因子高度重合，独立增量有限。\n",
              corr.round(3).to_frame("corr").to_markdown(), "\n"]

    out = Path("supply_diagnostics.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"report -> {out}")
    print("\n=== multi-horizon IC (rank_ic) ===")
    print(mh.pivot(index="factor", columns="horizon", values="rank_ic").round(4).to_string())
    print("\n=== yearly IC ===")
    print(yr.round(4).to_string())
    print("\n=== quantile (bps/5d) ===")
    print((qd["mean_fwd_return"]*1e4).round(2).to_string())
    print("\n=== correlation ===")
    print(corr.round(3).to_string())


if __name__ == "__main__":
    main()
