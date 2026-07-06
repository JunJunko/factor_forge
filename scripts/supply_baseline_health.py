"""Baseline-health check: did the volume_residual 120d OLS baseline degrade in 2025-2026?

Replicates the factor's per-stock time-series regression (log_turnover ~ 5 drivers on a
trailing 120-day window, min 80) at each month-end, for every top-1000 stock, and tracks:
  - mean in-window R² (goodness of fit -> baseline explanatory power)
  - mean in-window residual std (baseline noise)
  - cross-sectional std of the actual factor residual (volume_residual) per month
  - cross-sectional std of raw log_turnover (control: genuine volume heterogeneity)
A sharp drop in R² / rise in residual std in 2025-2026 => the AI-rotation thesis corroded
the baseline, on top of the regime flip.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository
from factor_forge.ml import supply_features as sf


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    log("loading panel (top 1000)")
    proj = load_project("configs/project.yaml")
    repo = DataVersionRepository(proj.paths.data_root, proj.paths.metadata_db)
    _v, panel = repo.load_panel("latest")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    med = panel.groupby("ts_code")["amount_cny"].median()
    panel = panel[panel["ts_code"].isin(med.nlargest(1000).index)].sort_values(
        ["ts_code", "trade_date"]).reset_index(drop=True)
    stocks, dates, industries = panel["ts_code"], panel["trade_date"], panel["industry_l1_code"]

    log("computing regressors")
    lr = sf.log_returns(panel["adj_close"], stocks)
    excess1 = sf.excess_returns(panel["adj_close"], stocks, dates, industries, [1])[1]
    intraday = sf.intraday_range(panel["raw_high"], panel["raw_low"], panel["pre_close"])
    vol20 = sf.volatility(lr, stocks, 20, ddof=1)
    mkt_tz = sf.market_turnover_z(panel["turnover_rate"], dates, stocks, 60)
    ind_tz = sf.industry_turnover_z(panel["turnover_rate"], dates, industries, stocks, 60)
    y_raw = np.log1p(panel["turnover_rate"].to_numpy(dtype=float) / 100.0)

    df = pd.DataFrame({
        "date": dates.to_numpy(), "ts_code": stocks.to_numpy(), "y": y_raw,
        "x1": excess1.abs().to_numpy(), "x2": intraday.to_numpy(), "x3": mkt_tz.to_numpy(),
        "x4": ind_tz.to_numpy(), "x5": vol20.to_numpy(),
    })
    log("pivoting to date x stock matrices")
    Y = df.pivot(index="date", columns="ts_code", values="y")
    X = {i: df.pivot(index="date", columns="ts_code", values=f"x{i}") for i in range(1, 6)}
    Ynp = Y.to_numpy(dtype=float)
    Xnp = [np.ones_like(Ynp)] + [X[i].reindex(index=Y.index, columns=Y.columns).to_numpy(dtype=float) for i in range(1, 6)]
    dates_idx = Y.index
    n_dates, n_stocks = Ynp.shape

    # cached factor residual for dispersion
    cache = Path("supply_ic_dataset.parquet")
    ds = pd.read_parquet(cache) if cache.exists() else None
    if ds is not None:
        ds["datetime"] = pd.to_datetime(ds["datetime"])

    # month-end trading dates
    s = pd.Series(dates_idx, index=dates_idx)
    month_ends = s.groupby(s.index.to_period("M")).max().sort_values().to_numpy()

    log(f"looping {len(month_ends)} month-ends x {n_stocks} stocks")
    rows = []
    for me in month_ends:
        e = dates_idx.get_loc(pd.Timestamp(me))
        start = e - 119
        if start < 0:
            continue
        idx = np.arange(start, e + 1)
        yw = Ynp[idx, :]              # (120, N)
        xw = np.stack([Xnp[k][idx, :] for k in range(6)], axis=2)  # (120, N, 6)
        r2s, rstds = [], []
        for j in range(n_stocks):
            yj = yw[:, j]
            xj = xw[:, j, :]
            finite = np.isfinite(yj) & np.isfinite(xj).all(axis=1)
            if finite.sum() < 80:
                continue
            yv = yj[finite]
            xv = xj[finite]
            xv_design = np.column_stack([np.ones(len(yv)), xv[:, 1:]])
            beta, *_ = np.linalg.lstsq(xv_design, yv, rcond=None)
            resid = yv - xv_design @ beta
            ss_res = float((resid ** 2).sum())
            ss_tot = float(((yv - yv.mean()) ** 2).sum())
            if ss_tot > 1e-12:
                r2s.append(1 - ss_res / ss_tot)
                rstds.append(resid.std(ddof=1))
        if not r2s:
            continue
        me_ts = pd.Timestamp(me)
        rec = {"month": me_ts.strftime("%Y-%m"), "r2_mean": float(np.mean(r2s)),
               "r2_p10": float(np.percentile(r2s, 10)), "resstd_mean": float(np.mean(rstds)),
               "n_stocks": len(r2s)}
        if ds is not None and "volume_residual" in ds.columns:
            m = ds["datetime"].dt.to_period("M") == me_ts.to_period("M")
            rec["vr_std"] = float(ds.loc[m, "volume_residual"].std())
            rec["raw_turnover_std"] = float(np.nanstd(yw))
        rows.append(rec)

    out = pd.DataFrame(rows).set_index("month")
    out_2024 = out[out.index >= "2024-01"]
    pd.set_option("display.width", 200)
    print("\n================ BASELINE HEALTH (per-stock 120d OLS, top 1000) ================")
    print(out_2024.round({"r2_mean": 3, "r2_p10": 3, "resstd_mean": 3, "vr_std": 3, "raw_turnover_std": 3, "n_stocks": 0}).to_string())
    # contrast windows
    def agg(lo, hi):
        w = out[(out.index >= lo) & (out.index <= hi)]
        return w
    pre = agg("2022-01", "2024-12")
    post = agg("2025-01", "2026-06")
    print("\n=== window contrast ===")
    print(f"{'window':18s} {'R2 mean':>8} {'R2 p10':>8} {'resid std':>10} {'VR std':>8} {'raw_lt std':>11}")
    for name, w in [("2022-2024 (pre)", pre), ("2025-2026 (post)", post)]:
        if len(w):
            print(f"{name:18s} {w['r2_mean'].mean():>8.3f} {w['r2_p10'].mean():>8.3f} "
                  f"{w['resstd_mean'].mean():>10.3f} {w.get('vr_std', pd.Series()).mean() if 'vr_std' in w else float('nan'):>8.3f} "
                  f"{w.get('raw_turnover_std', pd.Series()).mean() if 'raw_turnover_std' in w else float('nan'):>11.3f}")
    Path("supply_baseline_health.md").write_text(
        "# volume_residual 基准健康度（逐月每股 120 日 OLS）\n\n" + out.round(3).to_markdown() + "\n", encoding="utf-8")
    log("report -> supply_baseline_health.md")
    print("\n读法：R² 明显下降 + 残差 std 上升 → 基准在 2025-2026 变脏（与 AI 轮动 + regime 翻转吻合）。")


if __name__ == "__main__":
    main()
