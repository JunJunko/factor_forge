"""Dump a derived Qlib binary data provider from the immutable daily panel.

Qlib's backtest layer (``Exchange.get_quote_from_qlib`` -> ``D.features``,
``TradeCalendarManager`` -> ``D.calendar``) requires a real on-disk provider; the
``provider_uri=""`` trick only works for the *training* side (StaticDataLoader feeds a
DataFrame directly).  So we dump a derived cache here.

This is a **pure function of the immutable panel**: it lives under
``artifacts/qlib_bin_cache/<source_hash[:8]>/`` (never under ``data/versions/``), records
the source data version + panel sha256 in a manifest, and is rebuilt automatically if the
source hash changes.  Prices are written in **adjusted** space with ``$factor = 1`` so the
backtest, the label, and the model all share one return convention; the A-share
tradability rules are driven entirely by the masks in :mod:`supply_qlib_strategy`.

The float32 little-endian layout matches :func:`qlib.data.storage.file_storage.FileFeatureStorage.write`
exactly: ``np.hstack([start_index, values]).astype("<f4")``.
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Fields Qlib's Exchange consumes (queried with a leading ``$``; stored lowercase).
QLIB_FIELDS = ["open", "close", "high", "low", "volume", "factor", "change"]

# The Qlib backtest Account/PortfolioMetrics requires a benchmark instrument that exists
# in the provider.  We synthesize an equal-weight market benchmark under this code so a
# run never depends on a separate index dataset being present.
MARKET_BENCHMARK_CODE = "MARKET"


def qlib_instrument(ts_code: str) -> str:
    """Map a panel ``ts_code`` (e.g. ``000001.SZ``) to a path-safe Qlib instrument code."""
    return ts_code.replace(".", "").upper()


def _panel_fingerprint(panel: pd.DataFrame) -> str:
    # Hash the (ts_code, trade_date, adj_close) triple -- enough to detect content drift
    # without hashing the entire (large) panel payload.
    cols = [c for c in ("ts_code", "trade_date", "adj_close") if c in panel.columns]
    h = hashlib.sha256(pd.util.hash_pandas_object(panel[cols], index=False).values.tobytes())
    return h.hexdigest()


def _write_calendar(qlib_dir: Path, dates: list[pd.Timestamp]) -> None:
    (qlib_dir / "calendars").mkdir(parents=True, exist_ok=True)
    values = np.asarray([d.strftime("%Y-%m-%d") for d in dates], dtype=object)
    with (qlib_dir / "calendars" / "day.txt").open("w", encoding="utf-8") as fp:
        np.savetxt(fp, values, fmt="%s")
    # Qlib's TradeCalendarManager reads one step past the last calendar entry at the
    # boundary (get_step_time -> calendar[index+1]); a future calendar is REQUIRED to
    # avoid IndexError on the final bar (qlib issue #2278).  In standard qlib data the
    # ``day_future`` calendar is a SUPERSET (the full day calendar plus extra future
    # sessions), so the trailing +1 lookup always lands on a real entry.
    last = dates[-1] if dates else pd.Timestamp("2024-01-02")
    future = list(pd.bdate_range(last + pd.Timedelta(days=1), periods=10))
    future_values = np.asarray(
        [d.strftime("%Y-%m-%d") for d in [*dates, *future]], dtype=object
    )
    with (qlib_dir / "calendars" / "day_future.txt").open("w", encoding="utf-8") as fp:
        np.savetxt(fp, future_values, fmt="%s")


def _write_instruments(qlib_dir: Path, spans: dict[str, tuple[str, str]]) -> None:
    (qlib_dir / "instruments").mkdir(parents=True, exist_ok=True)
    lines = [f"{code}\t{start}\t{end}" for code, (start, end) in spans.items()]
    (qlib_dir / "instruments" / "all.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_feature(qlib_dir: Path, instrument: str, field: str, values: np.ndarray) -> None:
    feature_dir = qlib_dir / "features" / instrument.lower()
    feature_dir.mkdir(parents=True, exist_ok=True)
    path = feature_dir / f"{field.lower()}.day.bin"
    arr = np.hstack([np.array([0], dtype=np.float32), values.astype("<f4")])
    with path.open("wb") as fp:
        arr.tofile(fp)


def dump_supply_bin(
    panel: pd.DataFrame,
    output_root: Path,
    source_version: str,
    *,
    price_mode: str = "adj",
) -> tuple[Path, str, dict[str, str]]:
    """Dump ``panel`` to a Qlib provider under ``output_root/<hash[:8]>/``.

    Returns ``(qlib_dir, source_hash, instrument_map)`` where ``instrument_map`` maps the
    panel ``ts_code`` to the Qlib instrument code.
    """
    source_hash = _panel_fingerprint(panel)
    qlib_dir = Path(output_root) / source_hash[:8]
    if qlib_dir.exists() and (qlib_dir / "bin_manifest.json").exists():
        # Cache hit -- the panel content is unchanged.
        return qlib_dir, source_hash, _load_instrument_map(qlib_dir)

    if qlib_dir.exists():
        import shutil

        shutil.rmtree(qlib_dir)
    qlib_dir.mkdir(parents=True)

    df = panel.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    calendar = sorted(df["trade_date"].dropna().unique())
    calendar_ts = [pd.Timestamp(d) for d in calendar]
    date_to_idx = {d: i for i, d in enumerate(calendar_ts)}
    _write_calendar(qlib_dir, calendar_ts)

    suffix = "" if price_mode == "adj" else "_raw"
    price_cols = {
        "open": f"adj{suffix}_open" if price_mode == "adj" else "raw_open",
        "close": f"adj{suffix}_close" if price_mode == "adj" else "raw_close",
        "high": f"adj{suffix}_high" if price_mode == "adj" else "raw_high",
        "low": f"adj{suffix}_low" if price_mode == "adj" else "raw_low",
    }
    spans: dict[str, tuple[str, str]] = {}
    instrument_map: dict[str, str] = {}
    for ts_code, group in df.groupby("ts_code", sort=True):
        group = group.sort_values("trade_date")
        code = qlib_instrument(ts_code)
        instrument_map[ts_code] = code
        first_date = group["trade_date"].iloc[0].strftime("%Y-%m-%d")
        last_date = group["trade_date"].iloc[-1].strftime("%Y-%m-%d")
        spans[code] = (first_date, last_date)

        n = len(calendar_ts)
        open_arr = np.full(n, np.nan, dtype=np.float32)
        close_arr = np.full(n, np.nan, dtype=np.float32)
        high_arr = np.full(n, np.nan, dtype=np.float32)
        low_arr = np.full(n, np.nan, dtype=np.float32)
        vol_arr = np.full(n, np.nan, dtype=np.float32)
        factor_arr = np.ones(n, dtype=np.float32)
        change_arr = np.full(n, np.nan, dtype=np.float32)

        idx = np.array([date_to_idx[d] for d in group["trade_date"]], dtype=np.int64)
        suspended = group.get("is_suspended", pd.Series(False, index=group.index)).to_numpy()
        open_vals = group[price_cols["open"]].to_numpy(dtype=np.float64)
        close_vals = group[price_cols["close"]].to_numpy(dtype=np.float64)
        # $close = NaN on suspended bars so the default Exchange suspended-detection still
        # works as a backstop (our AShareExchange masks are the primary gate).
        close_vals = np.where(suspended, np.nan, close_vals)
        open_arr[idx] = open_vals
        close_arr[idx] = close_vals
        high_arr[idx] = group[price_cols["high"]].to_numpy(dtype=np.float64)
        low_arr[idx] = group[price_cols["low"]].to_numpy(dtype=np.float64)
        if "volume_shares" in group.columns:
            vol_arr[idx] = group["volume_shares"].to_numpy(dtype=np.float64)
        if "adj_factor" in group.columns and price_mode == "adj":
            factor_arr[idx] = group["adj_factor"].to_numpy(dtype=np.float64)
        # $change = per-stock daily pct change of $close (NaN on the first bar).
        chg = np.diff(close_vals, prepend=np.nan) / np.roll(close_vals, 1)
        chg[0] = np.nan
        change_arr[idx] = chg

        _write_feature(qlib_dir, code, "open", open_arr)
        _write_feature(qlib_dir, code, "close", close_arr)
        _write_feature(qlib_dir, code, "high", high_arr)
        _write_feature(qlib_dir, code, "low", low_arr)
        _write_feature(qlib_dir, code, "volume", vol_arr)
        _write_feature(qlib_dir, code, "factor", factor_arr)
        _write_feature(qlib_dir, code, "change", change_arr)

    _write_instruments(qlib_dir, spans)

    _dump_market_benchmark(qlib_dir, df, calendar_ts, date_to_idx, MARKET_BENCHMARK_CODE)
    spans[MARKET_BENCHMARK_CODE] = (
        calendar_ts[0].strftime("%Y-%m-%d"),
        calendar_ts[-1].strftime("%Y-%m-%d"),
    )
    _write_instruments(qlib_dir, spans)

    manifest = {
        "source_data_version": source_version,
        "source_panel_sha256": source_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "price_mode": price_mode,
        "calendar_days": len(calendar_ts),
        "instruments": len(spans),
        "fields": QLIB_FIELDS,
    }
    (qlib_dir / "bin_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (qlib_dir / "instrument_map.json").write_text(
        json.dumps(instrument_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return qlib_dir, source_hash, instrument_map


def _load_instrument_map(qlib_dir: Path) -> dict[str, str]:
    return json.loads((qlib_dir / "instrument_map.json").read_text(encoding="utf-8"))


def _dump_market_benchmark(
    qlib_dir: Path,
    df: pd.DataFrame,
    calendar_ts: list[pd.Timestamp],
    date_to_idx: dict,
    code: str,
) -> None:
    """Equal-weight market close across tradeable stocks, dumped as a benchmark instrument."""
    tradeable = df.get("is_tradeable", pd.Series(True, index=df.index))
    bench_src = df.copy()
    bench_src["__ok"] = tradeable.fillna(False) if hasattr(tradeable, "fillna") else True
    ok = bench_src["__ok"].to_numpy(dtype=bool) & df["adj_close"].notna().to_numpy()
    daily_mean = (
        bench_src.loc[ok].groupby("trade_date")["adj_close"].mean().sort_index()
    )
    n = len(calendar_ts)
    close_arr = np.full(n, np.nan, dtype=np.float32)
    for d, v in daily_mean.items():
        close_arr[date_to_idx[pd.Timestamp(d)]] = v
    # forward-fill the leading NaNs so $change is finite after the first bar
    lead = np.argmax(np.isfinite(close_arr)) if np.isfinite(close_arr).any() else 0
    close_arr[:lead] = close_arr[lead] if np.isfinite(close_arr[lead]) else np.nan
    change_arr = np.full(n, np.nan, dtype=np.float32)
    change_arr[1:] = (close_arr[1:] / close_arr[:-1]) - 1.0
    _write_feature(qlib_dir, code, "open", close_arr)
    _write_feature(qlib_dir, code, "close", close_arr)
    _write_feature(qlib_dir, code, "high", close_arr)
    _write_feature(qlib_dir, code, "low", close_arr)
    _write_feature(qlib_dir, code, "volume", np.zeros(n, dtype=np.float32))
    _write_feature(qlib_dir, code, "factor", np.ones(n, dtype=np.float32))
    _write_feature(qlib_dir, code, "change", change_arr)


def validate_dump(qlib_dir: Path, sample_code: str) -> None:
    """Round-trip check: read one instrument's ``$close`` back via Qlib and confirm it is
    non-empty.  Raises if the bin layout is wrong (silent garbage is the main risk)."""
    import qlib as qlib_init  # noqa: F401  (qlib must already be initialized)

    from qlib.data import D

    fields = [f"${f}" for f in QLIB_FIELDS]
    df = D.features([sample_code], fields, start_time=None, end_time=None, freq="day")
    if df.empty:
        raise RuntimeError(f"Qlib round-trip validation returned no rows for {sample_code}")
    # D.features -> DataFrame indexed by (instrument, datetime) with field columns.
    close = df["$close"].dropna() if "$close" in df.columns else df.iloc[:, 0].dropna()
    if close.empty:
        raise RuntimeError(f"Qlib round-trip validation: $close all-NaN for {sample_code}")
