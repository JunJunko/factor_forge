"""Regime-guard validation v2 -- diagnostic-first, leakage-free.

v1 failed: a 60d rolling vol20-IC detector never fired in 2026 (too slow).  Here we first
PRINT what the lagged rolling detector actually sees in 2025-2026 (does it rise in 2026?),
then evaluate gated scarcity IC only for detectors that demonstrably fire in the bad window.
Primary discriminator: the SIZE factor (was +0.05~+0.10 in 2026 Q2).  All signals are
lagged by the 5-day label horizon so the detector uses only realized factor IC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_forge.ml.supply_ic import compute_factor_ic, topn_returns, yearly_ic

EVAL_START, EVAL_END = "2017-01-01", "2026-06-30"
HORIZON = 5


def daily_ic(ds, factor, label_col="label"):
    pair = ds[["datetime", factor, label_col]].dropna()
    d = pair.groupby("datetime").apply(
        lambda g: g[factor].corr(g[label_col], method="spearman") if len(g) >= 20 else np.nan,
        include_groups=False,
    ).dropna()
    d.index = pd.to_datetime(d.index)
    return d


def main():
    ds = pd.read_parquet("supply_ic_dataset.parquet")
    ds["datetime"] = pd.to_datetime(ds["datetime"])
    if "label" not in ds.columns and "label_5d" in ds.columns:
        ds["label"] = ds["label_5d"]
    full = ds[ds["datetime"].between(pd.Timestamp(EVAL_START), pd.Timestamp(EVAL_END))].copy()

    # raw daily factor IC, then rolling + lag (real-time)
    vol_raw = daily_ic(full, "volatility_20")
    size_raw = daily_ic(full, "log_float_market_cap")
    scar_raw = daily_ic(full, "scarcity")

    # ---- STEP 1: can a real-time detector see 2026? Try windows 20/40/60. ----
    print("=" * 95)
    print("STEP 1: real-time detector visibility in 2026 (rolling mean of daily factor IC, lagged 5d)")
    print("=" * 95)
    for win in (20, 40, 60):
        v = vol_raw.rolling(win, min_periods=win // 2).mean().shift(HORIZON)
        s = size_raw.rolling(win, min_periods=win // 2).mean().shift(HORIZON)
        sc = scar_raw.rolling(win, min_periods=win // 2).mean().shift(HORIZON)
        m = pd.DataFrame({"vol20": v, "size": s, "scarcity": sc}).dropna()
        m2026 = m[m.index >= "2026-01-01"]
        monthly = m2026.resample("MS").mean().round(4)
        print(f"\n-- rolling window = {win}d, lagged 5d --  (2026 monthly avg of detector)")
        print(monthly.to_string())

    # pick the window/detector that actually rises in 2026 Q2; size@20d is the candidate
    win = 20
    size_roll = size_raw.rolling(win, min_periods=win // 2).mean().shift(HORIZON)
    vol_roll = vol_raw.rolling(win, min_periods=win // 2).mean().shift(HORIZON)
    det = pd.DataFrame({"size": size_roll, "vol20": vol_roll}).dropna()

    # ---- STEP 2: evaluate gated scarcity under several detector rules ----
    print("\n" + "=" * 95)
    print("STEP 2: gated scarcity IC (full 2017-2026) under leakage-free 20d detectors")
    print("=" * 95)
    base = compute_factor_ic(full, ["scarcity"], EVAL_START, EVAL_END).iloc[0]
    base_top = topn_returns(full, "scarcity", "label", EVAL_START, EVAL_END, [100]).iloc[0]
    print(f"\n{'baseline (all days)':45s} IC {base['rank_ic_mean']:+.4f}  ICIR {base['rank_ic_ir']:.2f}  "
          f"Top100 {base_top['avg_daily_excess']*1e4:+.2f} bps/5d")

    rules = {
        "size_ic <= 0.02": det["size"] <= 0.02,
        "size_ic <= 0": det["size"] <= 0,
        "size_ic<=0.02 AND vol20_ic<0": (det["size"] <= 0.02) & (det["vol20"] < 0),
    }
    best_ir = -1e9
    best_name = None
    best_gated_yearly = None
    best_good_days = set()
    cutoff2026 = pd.Timestamp("2026-01-01")
    for name, good_mask in rules.items():
        good_days = set(det.index[good_mask])
        sub = full[full["datetime"].isin(good_days)]
        bad = full[~full["datetime"].isin(good_days)]
        if len(sub) < 100:
            continue
        ic = compute_factor_ic(sub, ["scarcity"], EVAL_START, EVAL_END).iloc[0]
        ic_bad = compute_factor_ic(bad, ["scarcity"], EVAL_START, EVAL_END).iloc[0] if len(bad) else None
        top = topn_returns(sub, "scarcity", "label", EVAL_START, EVAL_END, [100]).iloc[0]
        bad_s = f"bad IC {ic_bad['rank_ic_mean']:+.4f}" if ic_bad is not None else ""
        n_bad_2026 = sum(1 for d in det.index[~good_mask] if d >= cutoff2026)
        print(f"{name:45s} IC {ic['rank_ic_mean']:+.4f}  ICIR {ic['rank_ic_ir']:.2f}  "
              f"Top100 {top['avg_daily_excess']*1e4:+.2f} bps  good {len(good_days)}d  {bad_s}  2026bad={n_bad_2026}d")
        if ic["rank_ic_ir"] > best_ir:
            best_ir = ic["rank_ic_ir"]
            best_name = name
            best_gated_yearly = yearly_ic(sub, "scarcity", "label", EVAL_START, EVAL_END)
            best_good_days = good_days

    # ---- STEP 3: yearly raw vs best gated ----
    if best_name:
        print(f"\n--- yearly IC: raw vs gated [{best_name}] ---")
        yr_raw = yearly_ic(full, "scarcity", "label", EVAL_START, EVAL_END)
        cmp = yr_raw[["rank_ic_mean"]].rename(columns={"rank_ic_mean": "raw"}).join(
            best_gated_yearly[["rank_ic_mean"]].rename(columns={"rank_ic_mean": "gated"})
        )
        print(cmp.round(4).to_string())
        d2026 = sum(1 for d in pd.to_datetime(full["datetime"].unique()) if d >= cutoff2026)
        g2026 = sum(1 for d in best_good_days if d >= cutoff2026)
        print(f"\n2026: {g2026}/{d2026} days kept (gated), {d2026-g2026} skipped as bad regime")

    print("\n读法：")
    print("- STEP 1 看 size 列在 2026 Q2 是否 > 0（检测器能否实时看见）；")
    print("- STEP 2 看 gated IC/ICIR/Top100 是否高于 baseline 且 bad-regime IC 为负；")
    print("- 2026 skip 天数 > 0 才说明守卫真的拦到了目标期；")
    print("- 滞后 5 日无前视；但阈值在样本内标定，OOS 稳定性待验证。")


if __name__ == "__main__":
    main()
