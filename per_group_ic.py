"""Per-L2-industry diagnostic for the sw_l2 rank-acceleration run.

Reads stored factor_values.parquet (carries group_code = industry_l2_code) and the
curated panel for close-to-close forward returns. Computes, per industry group and
horizon: coverage + within-group daily rank IC (mean / ICIR / t / positive_ratio),
plus a year-by-year IC stability view for the standout groups.

Read-only wrt the run. Writes per_group_ic.csv next to the artifacts.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:  # make Chinese industry names render on Windows consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RUN = "artifacts/runs/sw_l2_industry_rank_acceleration__20260701T002826Z__49922b72"
PANEL = "data/versions/data_v1_20260630T162655Z_3e5bc559/curated/stock_daily_panel.parquet"
HORIZONS = (1, 3, 5, 10, 15)
TOP_N = 8  # standout groups to show yearly stability for


def bucket_rank_ic(d: pd.DataFrame, hcol: str) -> pd.DataFrame:
    """Daily within-(date,group) Spearman IC; one row per (trade_date, group_code)."""
    d = d.dropna(subset=["factor_value", hcol]).copy()
    if d.empty:
        return pd.DataFrame(columns=["trade_date", "group_code", "ic"])
    g = d.groupby(["trade_date", "group_code"])
    d["fr"] = g["factor_value"].rank(pct=True)
    d["rr"] = g[hcol].rank(pct=True)
    fc = d["fr"] - g["fr"].transform("mean")
    rc = d["rr"] - g["rr"].transform("mean")
    d["p"] = fc * rc
    s = d.groupby(["trade_date", "group_code"])["p"].sum()
    cnt = d.groupby(["trade_date", "group_code"])[hcol].size()
    varf = d.groupby(["trade_date", "group_code"])["fr"].var()
    varr = d.groupby(["trade_date", "group_code"])["rr"].var()
    ic = s / ((cnt - 1) * np.sqrt(varf * varr))
    ic = ic.replace([np.inf, -np.inf], np.nan).dropna()
    return ic.rename("ic").reset_index()


def main() -> None:
    fv = pd.read_parquet(f"{RUN}/factor_values.parquet")
    fv["trade_date"] = pd.to_datetime(fv["trade_date"])
    fv = fv[fv["factor_valid"].astype(bool)].copy()

    px = pd.read_parquet(
        PANEL,
        columns=["trade_date", "ts_code", "adj_close", "industry_l2_code", "industry_l2_name"],
    )
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.sort_values(["ts_code", "trade_date"])
    for h in HORIZONS:
        px[f"fwd{h}"] = px.groupby("ts_code")["adj_close"].shift(-h) / px["adj_close"] - 1
    fwd = px[["trade_date", "ts_code"] + [f"fwd{h}" for h in HORIZONS]]

    name_map = (
        px[["industry_l2_code", "industry_l2_name"]]
        .dropna()
        .drop_duplicates()
        .set_index("industry_l2_code")["industry_l2_name"]
        .to_dict()
    )

    m = fv.merge(fwd, on=["trade_date", "ts_code"], how="left")

    # coverage
    cov = pd.DataFrame({
        "n_rows": fv.groupby("group_code").size(),
        "n_days": fv.groupby("group_code")["trade_date"].nunique(),
    })
    cov["mean_daily_size"] = cov["n_rows"] / cov["n_days"]
    cov["name"] = cov.index.map(lambda c: name_map.get(c, ""))

    # per-horizon per-group IC
    ic_frames = {}  # h -> daily bucket frame (for yearly breakdown later)
    wide = cov.copy()
    summary_rows = []
    for h in HORIZONS:
        ic = bucket_rank_ic(m, f"fwd{h}")
        ic_frames[h] = ic
        stat = (
            ic.groupby("group_code")["ic"]
            .agg(mean="mean", std="std", n="count", pos_ratio=lambda x: (x > 0).mean())
        )
        stat["icir"] = stat["mean"] / stat["std"]
        stat["t"] = stat["mean"] / stat["std"] * np.sqrt(stat["n"])
        wide[f"ic_h{h}"] = stat["mean"]
        wide[f"icir_h{h}"] = stat["icir"]
        wide[f"t_h{h}"] = stat["t"]
        wide[f"pos_h{h}"] = stat["pos_ratio"]
        summary_rows.append({
            "h": h,
            "mean_ic": stat["mean"].mean(),
            "n_positive": int((stat["mean"] > 0).sum()),
            "n_t_gt2": int((stat["t"] > 2).sum()),
            "n_t_lt_neg2": int((stat["t"] < -2).sum()),
        })

    # persist full table
    out_csv = f"{RUN}/per_group_ic.csv"
    wide.reset_index().to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"wrote {out_csv}  ({len(wide)} groups x {len(HORIZONS)} horizons)")

    print("\n# groups:", len(wide))
    print("\n=== horizon summary (equal-weighted across groups) ===")
    print(pd.DataFrame(summary_rows).set_index("h").to_string(
        float_format=lambda v: f"{v:.4f}"))

    # standout positives at h=1 with yearly stability
    ic1 = ic_frames[1].copy()
    ic1["year"] = ic1["trade_date"].dt.year
    yearly = ic1.groupby(["group_code", "year"])["ic"].mean().unstack("year")
    top = wide.sort_values("ic_h1", ascending=False).head(TOP_N).index.tolist()
    print(f"\n=== top {TOP_N} by h=1 IC: yearly stability (mean daily IC by year) ===")
    rows = []
    for gc in top:
        row = {"code": gc, "name": name_map.get(gc, "")}
        row.update({y: (yearly.loc[gc, y] if y in yearly.columns and gc in yearly.index else np.nan)
                    for y in sorted(yearly.columns)})
        rows.append(row)
    ydf = pd.DataFrame(rows).set_index("code")
    ydf.insert(0, "name", ydf.pop("name"))
    print(ydf.to_string(float_format=lambda v: f"{v:+.4f}"))

    print(f"\n(read {out_csv} for the full 122-group ranking across all horizons)")


if __name__ == "__main__":
    main()
