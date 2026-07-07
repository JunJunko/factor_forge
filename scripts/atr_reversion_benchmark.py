"""Benchmark helpers for ATR reversion backtests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CSI1000_CODE = "000852.SH"
DEFAULT_CSI1000_PATH = Path("data/benchmarks/csi1000_000852_index_daily.parquet")


def csi1000_open_to_open_returns(
    dates: list[pd.Timestamp] | pd.Series | pd.Index,
    *,
    path: str | Path = DEFAULT_CSI1000_PATH,
) -> list[float]:
    """Return CSI1000 open-to-open returns aligned to strategy daily rows.

    The strategy marks NAV at open, so the benchmark also uses index open prices.
    The first requested date is set to 0.0 to match strategy daily return
    convention.
    """

    ordered_dates = pd.to_datetime(pd.Index(dates)).sort_values()
    if len(ordered_dates) == 0:
        return []
    index_daily = load_csi1000_index_daily(path)
    prices = index_daily.set_index("trade_date")["open"].sort_index()
    missing = ordered_dates.difference(prices.index)
    if len(missing):
        sample = ", ".join(d.strftime("%Y-%m-%d") for d in missing[:5])
        raise RuntimeError(f"CSI1000 benchmark missing {len(missing)} dates, e.g. {sample}")
    aligned = prices.reindex(ordered_dates).astype(float)
    returns = aligned.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return returns.to_list()


def load_csi1000_index_daily(path: str | Path = DEFAULT_CSI1000_PATH) -> pd.DataFrame:
    """Load CSI1000 index daily data from the explicit benchmark cache or raw versions."""

    explicit = Path(path)
    candidates = [explicit]
    candidates.extend(
        sorted(
            Path("data/versions").glob("*/raw/tushare/index_daily.parquet"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    )
    frames = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        df = pd.read_parquet(candidate)
        if df.empty or "ts_code" not in df or "trade_date" not in df or "open" not in df:
            continue
        df = df[df["ts_code"].eq(CSI1000_CODE)].copy()
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"CSI1000 index data not found. Expected {explicit}; fetch Tushare index_daily {CSI1000_CODE} first."
        )
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["ts_code", "trade_date"], keep="first")
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out = out.sort_values("trade_date").reset_index(drop=True)
    if len(out) < 2:
        raise RuntimeError(f"CSI1000 index data has too few rows: {len(out)}")
    return out
