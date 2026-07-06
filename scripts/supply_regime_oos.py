"""OOS validation of the size-regime guard for scarcity.

Calibrate the leakage-free detector threshold (size-factor 20d rolling IC, lagged 5d) on
the TRAIN window only -- pick the threshold maximizing gated scarcity ICIR -- freeze it,
then evaluate on the TEST window.  If the guard generalizes, test-window gated IC/ICIR
should beat raw and the 2026 reversal should still be neutralized without having seen 2026.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_forge.ml.supply_ic import compute_factor_ic, topn_returns, yearly_ic

TRAIN_START, TRAIN_END = "2017-01-01", "2021-12-31"
TEST_START, TEST_END = "2022-01-01", "2026-06-30"
HORIZON, ROLL = 5, 20
THRESHOLDS = [-0.010, -0.005, 0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030]


def daily_ic(ds, factor, label_col="label"):
    pair = ds[["datetime", factor, label_col]].dropna()
    d = pair.groupby("datetime").apply(
        lambda g: g[factor].corr(g[label_col], method="spearman") if len(g) >= 20 else np.nan,
        include_groups=False,
    ).dropna()
    d.index = pd.to_datetime(d.index)
    return d


def gated_stats(ds_subset, good_days):
    sub = ds_subset[ds_subset["datetime"].isin(good_days)]
    if len(sub) < 200:
        return None
    ic = compute_factor_ic(sub, ["scarcity"], sub["datetime"].min().strftime("%Y-%m-%d"),
                           sub["datetime"].max().strftime("%Y-%m-%d")).iloc[0]
    top = topn_returns(sub, "scarcity", "label", sub["datetime"].min().strftime("%Y-%m-%d"),
                       sub["datetime"].max().strftime("%Y-%m-%d"), [100])
    return {"ic": ic["rank_ic_mean"], "icir": ic["rank_ic_ir"], "top100_bps": top.iloc[0]["avg_daily_excess"] * 1e4,
            "n_days": len(sub)}


def main():
    ds = pd.read_parquet("supply_ic_dataset.parquet")
    ds["datetime"] = pd.to_datetime(ds["datetime"])
    if "label" not in ds.columns and "label_5d" in ds.columns:
        ds["label"] = ds["label_5d"]

    full = ds[ds["datetime"].between(pd.Timestamp(TRAIN_START), pd.Timestamp(TEST_END))].copy()
    size_roll = daily_ic(full, "log_float_market_cap").rolling(ROLL, min_periods=ROLL // 2).mean().shift(HORIZON)
    det = pd.DataFrame({"size_ic": size_roll}).dropna()
    day_to_size = det["size_ic"].to_dict()

    train = full[full["datetime"].between(pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END))].copy()
    test = full[full["datetime"].between(pd.Timestamp(TEST_START), pd.Timestamp(TEST_END))].copy()

    def good_days_for(thr, period_df):
        return {d for d in period_df["datetime"].unique() if day_to_size.get(d, np.nan) <= thr}

    # ---- TRAIN: calibrate threshold ----
    print("=" * 95)
    print(f"TRAIN {TRAIN_START}~{TRAIN_END}: calibrate size_ic threshold (keep if size_ic <= thr, maximize gated ICIR)")
    print("=" * 95)
    print(f"{'threshold':>10} {'gated_IC':>10} {'gated_ICIR':>11} {'Top100_bps':>11} {'n_days':>8}")
    train_rows = []
    for thr in THRESHOLDS:
        gd = good_days_for(thr, train)
        st = gated_stats(train, gd)
        if st is None:
            continue
        train_rows.append((thr, st))
        print(f"{thr:>10.3f} {st['ic']:>10.4f} {st['icir']:>11.2f} {st['top100_bps']:>11.2f} {st['n_days']:>8}")
    train_raw = compute_factor_ic(train, ["scarcity"], TRAIN_START, TRAIN_END).iloc[0]
    train_top = topn_returns(train, "scarcity", "label", TRAIN_START, TRAIN_END, [100]).iloc[0]
    print(f"{'raw':>10} {train_raw['rank_ic_mean']:>10.4f} {train_raw['rank_ic_ir']:>11.2f} "
          f"{train_top['avg_daily_excess']*1e4:>11.2f} {len(train['datetime'].unique()):>8}")

    best = max(train_rows, key=lambda x: x[1]["icir"])
    tau = best[0]
    print(f"\n>>> chosen threshold (max train ICIR): size_ic <= {tau:.3f}   "
          f"(train ICIR {best[1]['icir']:.2f} vs raw {train_raw['rank_ic_ir']:.2f})")

    # ---- TEST: apply frozen threshold ----
    print("\n" + "=" * 95)
    print(f"TEST {TEST_START}~{TEST_END}: apply FROZEN threshold size_ic <= {tau:.3f}")
    print("=" * 95)
    test_raw = compute_factor_ic(test, ["scarcity"], TEST_START, TEST_END).iloc[0]
    test_top = topn_returns(test, "scarcity", "label", TEST_START, TEST_END, [100]).iloc[0]
    print(f"{'raw':>12} IC {test_raw['rank_ic_mean']:+.4f}  ICIR {test_raw['rank_ic_ir']:.2f}  "
          f"Top100 {test_top['avg_daily_excess']*1e4:+.2f} bps/5d")
    gd_test = good_days_for(tau, test)
    st = gated_stats(test, gd_test)
    print(f"{'gated':>12} IC {st['ic']:+.4f}  ICIR {st['icir']:.2f}  "
          f"Top100 {st['top100_bps']:+.2f} bps/5d  kept {st['n_days']}d")

    # robustness sweep on TEST (all thresholds, to show stability around tau)
    print(f"\n--- TEST robustness: gated stats for every threshold (frozen choice = {tau:.3f}) ---")
    print(f"{'threshold':>10} {'gated_IC':>10} {'gated_ICIR':>11} {'Top100_bps':>11} {'n_days':>8}")
    for thr in THRESHOLDS:
        gd = good_days_for(thr, test)
        s = gated_stats(test, gd)
        if s is None:
            continue
        mark = "  <- chosen" if abs(thr - tau) < 1e-9 else ""
        print(f"{thr:>10.3f} {s['ic']:>10.4f} {s['icir']:>11.2f} {s['top100_bps']:>11.2f} {s['n_days']:>8}{mark}")

    # yearly TEST raw vs gated(tau)
    print(f"\n--- TEST yearly IC: raw vs gated (size_ic <= {tau:.3f}) ---")
    yr_raw = yearly_ic(test, "scarcity", "label", TEST_START, TEST_END)
    sub = test[test["datetime"].isin(gd_test)]
    yr_gated = yearly_ic(sub, "scarcity", "label", TEST_START, TEST_END)
    cmp = yr_raw[["rank_ic_mean"]].rename(columns={"rank_ic_mean": "raw"}).join(
        yr_gated[["rank_ic_mean"]].rename(columns={"rank_ic_mean": "gated"})
    )
    print(cmp.round(4).to_string())

    print("\n判定：")
    print("- 若 TEST gated ICIR > raw ICIR 且方向稳定 → 守卫 OOS 有效，可上线；")
    print("- 若 TEST 反而恶化或阈值极度敏感 → 样本内现象，不可用。")


if __name__ == "__main__":
    main()
