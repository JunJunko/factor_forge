"""Investigate the 2026 scarcity-IC reversal.

Monthly IC of scarcity alongside reference factors (is the flip scarcity-specific or
market-wide?), the 60-day rolling IC to see whether the turn was sudden or gradual, and a
monthly market-regime backdrop (breadth + cross-sectional volatility).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.ml.supply_ic import monthly_ic, rolling_ic

EVAL_START, EVAL_END = "2024-01-01", "2026-06-30"


def main():
    ds = pd.read_parquet("supply_ic_dataset.parquet")
    ds["datetime"] = pd.to_datetime(ds["datetime"])
    # relabel 5d
    if "label" not in ds.columns and "label_5d" in ds.columns:
        ds["label"] = ds["label_5d"]

    factors = ["scarcity", "volatility_20", "log_float_market_cap", "excess_ret_5", "risk_adjusted_ret_5", "turnover_zscore_60"]
    mic = monthly_ic(ds, factors, "label", EVAL_START, EVAL_END)

    # regime backdrop: monthly mean of daily cross-sectional breadth and avg volatility_20
    sub = ds[["datetime", "market_breadth", "volatility_20", "label"]].copy()
    sub["datetime"] = pd.to_datetime(sub["datetime"])
    daily_breadth = sub.groupby("datetime")["market_breadth"].mean()
    daily_vol = sub.groupby("datetime")["volatility_20"].mean()
    daily_mkt = sub.groupby("datetime")["label"].mean()  # universe mean 5d fwd (industry-neutral-ish)
    regime = pd.DataFrame({
        "breadth": daily_breadth.resample("MS").mean(),
        "avg_vol20": daily_vol.resample("MS").mean(),
        "univ_fwd_5d_bps": (daily_mkt.resample("MS").mean()) * 10_000,
    })
    table = mic.join(regime)

    roll = rolling_ic(ds, "scarcity", "label", EVAL_START, EVAL_END, window=60)

    # also: scarcity monthly IC restricted to 2026 only, vs same months prior years
    ds_all = ds
    scarcity_yearmonth = []
    for f in ["scarcity"]:
        for year in [2024, 2025, 2026]:
            m = ds_all[ds_all["datetime"].dt.year == year]
            mm = monthly_ic(m, [f], "label", str(year) + "-01-01", str(year) + "-12-31")
            mm.columns = [f"{f}_{year}"]
            scarcity_yearmonth.append(mm)
    seasonal = pd.concat(scarcity_yearmonth, axis=1)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    lines = [
        "# 2026 scarcity IC 反转排查\n",
        f"- universe: 流动性 top 1000；主因子 scarcity；标签 5 日行业中性\n",
        "## 月度 IC（2024-2026）+ 市场背景\n",
        "> scarcity 与参照因子同窗对比；breadth=上涨股占比；avg_vol20=横截面平均波动；univ_fwd=全市场 5 日前瞻均值(bps)。\n",
        table.round({"scarcity": 4, "volatility_20": 4, "log_float_market_cap": 4, "excess_ret_5": 4,
                     "risk_adjusted_ret_5": 4, "turnover_zscore_60": 4, "breadth": 3, "avg_vol20": 4,
                     "univ_fwd_5d_bps": 2}).to_markdown(),
        "\n\n## 同月跨年对比（scarcity 月度 IC：2024 vs 2025 vs 2026）\n",
        "> 看某个月份在 2026 是否系统性比往年差（季节性），还是全盘逆转。\n",
        seasonal.round(4).to_markdown(),
        "\n\n## 结论线索\n",
        "- 若 scarcity 在 2026 转负而参照因子（vol/size）未同步 → **scarcity 专属崩塌**，命题在该 regime 失效；",
        "- 若全部因子同向剧烈波动 → **市场 regime 切换**，非 scarcity 独有；",
        "- 看 breadth/avg_vol20 在 2026 是否突变定位 regime 性质。",
    ]
    out = Path("supply_2026_probe.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("=== monthly IC + regime ===")
    print(table.round(4).to_string())
    print("\n=== seasonal (scarcity by month, 2024/2025/2026) ===")
    print(seasonal.round(4).to_string())
    roll_clean = roll.dropna()
    last_roll = f"{roll_clean.iloc[-1]:.4f}" if len(roll_clean) else "nan"
    print(f"\n60d rolling IC of scarcity: last value {last_roll}")
    print(f"-> report: {out}")


if __name__ == "__main__":
    main()
