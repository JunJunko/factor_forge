"""Task 5: univariate analysis of the V2 stable-baseline + no-volume-rise factors.

For each of the 7 handoff-doc factors (turnover_vol_rank_28, turnover_stability_28,
price_strength_2, recent_volume_z_mean_2_clip, recent_volume_z_max_2_clip,
baseline_turnover_mean_28, baseline_amount_mean_28) compute, on the liquid (top-1000)
universe over 2017-01..2026-06:

  * Pearson IC, RankIC, ICIR, Newey-West t (lag scaled to the label horizon)
  * 1/3/5/10/15-day IC decay                       -> factor_decay.csv
  * yearly RankIC                                   -> factor_yearly_ic.csv
  * quintile / decile mean forward return
  * sample count + coverage
  * cross-sectional exposure to price / size / liquidity / volatility / turnover

Deliverables (handoff doc task 5):
    univariate_factor_metrics.csv, factor_decay.csv, factor_yearly_ic.csv,
    univariate_factor_report.md.

Usage:  python -m scripts.supply_v2_univariate [out_dir]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository
from factor_forge.ml.supply_ic import (
    compute_factor_ic,
    factor_rank_correlation,
    load_dataset_for_ic,
    multi_horizon_labels,
    quantile_decomp,
    yearly_ic,
)

V2_FACTORS = [
    "turnover_vol_rank_28",
    "turnover_stability_28",
    "price_strength_2",
    "recent_volume_z_mean_2_clip",
    "recent_volume_z_max_2_clip",
    "baseline_turnover_mean_28",
    "baseline_amount_mean_28",
]
HORIZONS = [1, 3, 5, 10, 15]
# Newey-West lag >= horizon - 1 (handoff doc sec. 5: 5d->4, 10d->9, 15d->14).
NW_LAG = {1: 1, 3: 2, 5: 4, 10: 9, 15: 14}
EXPOSURES = ["log_raw_price", "log_float_market_cap", "log_avg_amount_20",
             "volatility_20", "turnover_zscore_60"]
EVAL_START, EVAL_END = "2017-01-01", "2026-06-30"


def _attach_multi_horizon_labels(ds: pd.DataFrame, panel: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Attach ``label_{h}`` for each horizon, masked to the dataset's valid rows.

    ``build_supply_dataset`` already NaN'd invalid rows' ``label`` (h=5), so its notna()
    is the validity mask.  h=5 is taken from the build's own ``label`` to stay identical
    to the trainer; only the other horizons are attached here.
    """
    mi = pd.MultiIndex.from_frame(ds[["datetime", "instrument"]])
    valid = ds["label"].notna().to_numpy()
    attach = [h for h in horizons if f"label_{h}" not in ds.columns and h != 5]
    if not attach:
        return ds
    labels = multi_horizon_labels(panel, attach, method="open_to_open")
    for h, s in labels.items():
        vals = s.reindex(mi).to_numpy()
        ds[f"label_{h}"] = np.where(valid, vals, np.nan)
    return ds


def main(out_dir: str = ".") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

    log("loading panel (latest version)")
    proj = load_project("configs/project.yaml")
    repo = DataVersionRepository(proj.paths.data_root, proj.paths.metadata_db)
    version, panel = repo.load_panel("latest")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    log(f"panel={panel.shape} version={version}")

    log("selecting liquid universe (top 1000 by median amount)")
    liquid = panel.groupby("ts_code")["amount_cny"].median().nlargest(1000).index
    sub = panel[panel["ts_code"].isin(liquid)].copy()
    log(f"liquid subset={sub.shape}")

    cache = Path("supply_v2_ic_dataset.parquet")
    if cache.exists():
        log(f"loading cached dataset from {cache}")
        ds = pd.read_parquet(cache)
    else:
        log("building IC dataset (raw, no zscore; includes V2 baseline_structure fields)")
        ds = load_dataset_for_ic(sub)
        ds.to_parquet(cache)
        log(f"cached -> {cache}")
    log(f"dataset rows={len(ds)} cols={len(ds.columns)}")

    missing = [f for f in V2_FACTORS if f not in ds.columns]
    if missing:
        raise SystemExit(f"V2 factors missing from dataset (stale cache?): {missing}")

    log(f"attaching multi-horizon labels for decay {[h for h in HORIZONS if h != 5]}")
    ds = _attach_multi_horizon_labels(ds, sub, HORIZONS)
    ds.to_parquet(cache)  # persist labels so re-runs skip this step

    # ---- main metrics at h=5 (the trainer horizon; label_col='label') ----
    log(f"main IC @ h=5 over {EVAL_START}..{EVAL_END}")
    main_ic = compute_factor_ic(ds, V2_FACTORS, EVAL_START, EVAL_END,
                                label_col="label", n_lag=NW_LAG[5])

    # coverage + sample counts
    mask = ds["datetime"].between(pd.Timestamp(EVAL_START), pd.Timestamp(EVAL_END))
    sub_eval = ds.loc[mask]
    cov_rows = []
    for f in V2_FACTORS:
        pair = sub_eval[[f, "label"]].dropna()
        n_samples = int(len(pair))
        coverage = float(pair[f].notna().sum() / len(sub_eval)) if len(sub_eval) else np.nan
        cov_rows.append({"factor": f, "n_samples": n_samples, "coverage": coverage})
    coverage = pd.DataFrame(cov_rows).set_index("factor")

    # exposures (time-series-averaged daily cross-section spearman)
    log("exposures vs price/size/liquidity/vol/turnover")
    exp_rows = {f: factor_rank_correlation(ds, f, EXPOSURES, EVAL_START, EVAL_END) for f in V2_FACTORS}
    exposures = pd.DataFrame(exp_rows).T
    exposures.index.name = "factor"

    metrics = main_ic.join(coverage).join(exposures)
    metrics.to_csv(out_dir / "univariate_factor_metrics.csv", encoding="utf-8-sig")
    log("-> univariate_factor_metrics.csv")

    # ---- IC decay across horizons ----
    log("IC decay across horizons 1/3/5/10/15")
    decay = {}
    for h in HORIZONS:
        col = "label" if h == 5 else f"label_{h}"
        ic_h = compute_factor_ic(ds, V2_FACTORS, EVAL_START, EVAL_END, label_col=col, n_lag=NW_LAG[h])
        decay[h] = ic_h["rank_ic_mean"]
    decay_df = pd.DataFrame(decay)
    decay_df.columns = [f"rank_ic_h{h}" for h in HORIZONS]
    decay_df.index.name = "factor"
    decay_df.to_csv(out_dir / "factor_decay.csv", encoding="utf-8-sig")
    log("-> factor_decay.csv")

    # ---- yearly IC (h=5) ----
    log("yearly RankIC @ h=5")
    yearly_rows = {f: yearly_ic(ds, f, "label", EVAL_START, EVAL_END)["rank_ic_mean"] for f in V2_FACTORS}
    yearly_df = pd.DataFrame(yearly_rows).T
    yearly_df.index.name = "factor"
    yearly_df.to_csv(out_dir / "factor_yearly_ic.csv", encoding="utf-8-sig")
    log("-> factor_yearly_ic.csv")

    # ---- quintile / decile mean forward return (h=5) ----
    log("quintile / decile decompositions @ h=5")
    quintile_tables = {f: quantile_decomp(ds, f, "label", EVAL_START, EVAL_END, n_bins=5) for f in V2_FACTORS}
    decile_tables = {f: quantile_decomp(ds, f, "label", EVAL_START, EVAL_END, n_bins=10) for f in V2_FACTORS}

    # ---- report ----
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    L = []
    L.append("# V2 稳定基线 + 无量上涨 — 单变量分析报告（任务 5）\n")
    L.append(f"- 数据版本: `{version}`")
    L.append("- universe: 流动性 top 1000（按中位成交额）")
    L.append(f"- 评估窗口: {EVAL_START} ~ {EVAL_END}")
    L.append("- 标签: 行业中性前瞻收益（开到开，留一法）；主口径 h=5（`label`），衰减另用 h∈{1,3,5,10,15}")
    L.append("- Newey-West 滞后按标签重叠: h=1→1, h=3→2, h=5→4, h=10→9, h=15→14（交接 sec. 5）")
    L.append("- 覆盖率 = (因子非空且 label 非空) / 评估窗口总行数\n")
    L.append("## 1. 主指标（h=5 行业超额）\n")
    L.append("> 方向预期：`price_strength_2` 正向；`recent_volume_z_*`/`turnover_vol_rank_28` 负向（越未放量/越波动差越好）；`turnover_stability_28` 正向；`baseline_*_mean_28` 为控制变量，方向待定。\n")
    L.append(metrics.round(4).to_markdown())
    L.append("\n\n## 2. IC 衰减（rank_ic，各持有期）\n")
    L.append(decay_df.round(4).to_markdown())
    L.append("\n\n## 3. 分年度 RankIC（h=5）\n")
    L.append(yearly_df.round(4).to_markdown())
    L.append("\n\n## 4. 五分位前瞻收益（h=5）\n")
    for f in V2_FACTORS:
        L.append(f"\n### {f}")
        L.append(quintile_tables[f].round(5).to_markdown())
    L.append("\n\n## 5. 十分位前瞻收益（h=5）\n")
    for f in V2_FACTORS:
        L.append(f"\n### {f}")
        L.append(decile_tables[f].round(5).to_markdown())
    L.append("\n\n## 6. 暴露（与控制变量截面 spearman 均值）\n")
    L.append(exposures.round(3).to_markdown())
    L.append("\n\n## 7. 关键判断（对应交接 §13 核心问题）\n")
    L.append(_verdict(metrics, decay_df, yearly_df))
    L.append("\n")
    (out_dir / "univariate_factor_report.md").write_text("\n".join(L), encoding="utf-8")
    log("-> univariate_factor_report.md")
    print("\n=== main metrics (h=5) ===")
    print(main_ic.round(4).to_string())


def _val(metrics: pd.DataFrame, f: str, col: str) -> float:
    try:
        return float(metrics.loc[f, col])
    except Exception:
        return float("nan")


def _verdict(metrics: pd.DataFrame, decay: pd.DataFrame, yearly: pd.DataFrame) -> str:
    lines = []
    ps_ic = _val(metrics, "price_strength_2", "rank_ic_mean")
    ps_t = _val(metrics, "price_strength_2", "rank_ic_newey_t")
    maxz_ic = _val(metrics, "recent_volume_z_max_2_clip", "rank_ic_mean")
    maxz_t = _val(metrics, "recent_volume_z_max_2_clip", "rank_ic_newey_t")
    meanz_ic = _val(metrics, "recent_volume_z_mean_2_clip", "rank_ic_mean")
    vol_ic = _val(metrics, "turnover_vol_rank_28", "rank_ic_mean")
    stab_ic = _val(metrics, "turnover_stability_28", "rank_ic_mean")
    btm_ic = _val(metrics, "baseline_turnover_mean_28", "rank_ic_mean")
    bam_ic = _val(metrics, "baseline_amount_mean_28", "rank_ic_mean")
    ps_sign = "正向" if ps_ic > 0 else "负向（2 日反转）"
    lines.append(f"- **`price_strength_2`（2 日上涨强度）**：RankIC={ps_ic:+.4f}, NW-t={ps_t:+.2f}（{ps_sign}）。"
                 f"交接假设为正向；若实际为负，「上涨」腿单变量不成立，需任务 6 交互检验（§13.1）。")
    lines.append(f"- **`recent_volume_z_max_2_clip`（未放量主口径）**：RankIC={maxz_ic:+.4f}, NW-t={maxz_t:+.2f}。"
                 f"预期**负向**（越未放量未来越好）；负且显著支持「未激活成交」腿（§13.1）。")
    lines.append(f"- **max vs mean**（§9.4）：max RankIC={maxz_ic:+.4f}, mean RankIC={meanz_ic:+.4f}。"
                 f"比较 |IC| 与 t 判断 `recent_volume_z_max_2` 是否优于 `mean_2`。")
    lines.append(f"- **成交稳定度方向**（§9.1/9.2）：`turnover_vol_rank_28`（越高=越波动）RankIC={vol_ic:+.4f}；"
                 f"`turnover_stability_28`（越高=越稳定）RankIC={stab_ic:+.4f}。"
                 f"若近似单调（vol 负 / stability 正），稳定度可能是线性信号；若弱，需任务 6 二维矩阵查倒 U。")
    lines.append(f"- **基线水平**：`baseline_turnover_mean_28` RankIC={btm_ic:+.4f}；"
                 f"`baseline_amount_mean_28` RankIC={bam_ic:+.4f}（控制变量，本身不要求显著）。")
    lines.append("- **衰减**（§3 表）：观察各因子 IC 在 h=1→15 的形状；结构信号通常在中周期（h=5/10）较强、长端衰减。")
    lines.append("- **分年度一致性**（§3 表）：主信号若在某年翻负（如 2026），需任务 6/8 条件矩阵与中性化进一步排查 regime，本节只记录不下结论。")
    lines.append("- **暴露**（§6 表）：|spearman| 与控制变量高（>0.5）的因子，独立增量需任务 8 中性化确认。")
    return "\n".join(lines)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
