"""Regime conditional payoff diagnostics for ATR lower-shadow event pool.

This script does not train a model. It asks whether the event itself has
different future payoff under different market/regime states.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from atr_reversion_fit_quality_gate import HMM_VARIANT, PIT_RUN, SOURCE_RUN
from atr_reversion_small_portfolio_backtest import _json_default
from atr_reversion_walk_forward import FOLDS


EVENT_RUN_GLOB = "event_badtrade_iteration_*"
REGIME_PATH = PIT_RUN / "permission_daily_regime_features.parquet"
OUTPUT_PREFIX = "regime_conditional_payoff"
REGIME_COLS = [
    "market_ret_20",
    "market_ret_60",
    "market_breadth_20",
    "market_vol_20",
    "xsec_vol_20",
    "turnover_chg_5_20",
    "reversal_strength_20",
    "momentum_minus_reversal_20",
]


def main(source_run: str = str(SOURCE_RUN)) -> None:
    output = PIT_RUN / f"{OUTPUT_PREFIX}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    event_run = _latest_event_run()
    event = pd.read_parquet(event_run / "event_model_dataset.parquet")
    event["trade_date"] = pd.to_datetime(event["trade_date"])
    regime = pd.read_parquet(REGIME_PATH)
    regime["trade_date"] = pd.to_datetime(regime["trade_date"])
    states = _load_hmm_states(Path(source_run))
    log(f"loaded event_run={event_run.name} event_rows={len(event):,} regime_days={len(regime):,}")

    sample = _prepare_event_sample(event, regime, states)
    sample.to_parquet(output / "event_payoff_sample.parquet", index=False)
    log(f"sample events={len(sample):,} days={sample['trade_date'].nunique():,}")

    daily = _daily_event_payoff(sample)
    daily.to_csv(output / "daily_event_payoff.csv", index=False, encoding="utf-8-sig")
    daily.to_parquet(output / "daily_event_payoff.parquet", index=False)

    date_weighted = _conditional_payoff(daily, weight_mode="date")
    event_weighted = _conditional_payoff(sample, weight_mode="event")
    by_year = _conditional_payoff_by_year(daily)
    hmm = _hmm_state_payoff(daily)
    year_profile = _year_profile(daily)
    top_alpha = _top_alpha_payoff(event_run, sample, regime, states, log)

    date_weighted.to_csv(output / "conditional_payoff_date_weighted.csv", index=False, encoding="utf-8-sig")
    event_weighted.to_csv(output / "conditional_payoff_event_weighted.csv", index=False, encoding="utf-8-sig")
    by_year.to_csv(output / "conditional_payoff_by_year_regime.csv", index=False, encoding="utf-8-sig")
    hmm.to_csv(output / "hmm_state_payoff.csv", index=False, encoding="utf-8-sig")
    year_profile.to_csv(output / "year_profile.csv", index=False, encoding="utf-8-sig")
    top_alpha.to_csv(output / "top_alpha_payoff_by_regime.csv", index=False, encoding="utf-8-sig")

    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "event_run": str(event_run),
                "regime_path": str(REGIME_PATH),
                "events": int(len(sample)),
                "days": int(sample["trade_date"].nunique()),
                "date_weighted": date_weighted.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(
        _report(date_weighted, by_year, hmm, year_profile, top_alpha),
        encoding="utf-8",
    )
    log("done")
    print(f"run_dir={output}")


def _latest_event_run() -> Path:
    runs = [p for p in PIT_RUN.glob(EVENT_RUN_GLOB) if (p / "event_model_dataset.parquet").exists()]
    if not runs:
        raise FileNotFoundError("no event_badtrade_iteration run with event_model_dataset.parquet found")
    return max(runs, key=lambda p: p.stat().st_mtime)


def _load_hmm_states(source: Path) -> pd.DataFrame:
    frames = []
    for fold in FOLDS:
        path = source / fold["name"] / HMM_VARIANT / "hmm_daily_states.csv"
        if not path.exists():
            continue
        s = pd.read_csv(path)
        s["trade_date"] = pd.to_datetime(s["trade_date"])
        s["fold"] = fold["name"]
        test = s[s["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))]
        frames.append(test[["trade_date", "fold", "predicted_state"]])
    return pd.concat(frames, ignore_index=True).drop_duplicates("trade_date") if frames else pd.DataFrame()


def _prepare_event_sample(event: pd.DataFrame, regime: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "trade_date",
        "ts_code",
        "pit_top1000",
        "event_trigger",
        "fwd_ret_10",
        "fwd_industry_excess_10",
        "fwd_drawdown_10",
        "bad_trade_10",
        "top_decile_hit_10",
    ]
    sample = event.loc[
        event["trade_date"].ge(pd.Timestamp("2022-01-01"))
        & event["pit_top1000"].fillna(False).astype(bool)
        & event["event_trigger"].fillna(False).astype(bool),
        cols,
    ].dropna(subset=["fwd_ret_10", "fwd_industry_excess_10"]).copy()
    sample = sample.merge(regime[["trade_date", *REGIME_COLS]], on="trade_date", how="left")
    if not states.empty:
        sample = sample.merge(states[["trade_date", "predicted_state"]], on="trade_date", how="left")
    for col in REGIME_COLS:
        sample[f"{col}_bucket"] = _tercile_bucket(sample[col])
    sample["year"] = sample["trade_date"].dt.year
    return sample


def _tercile_bucket(s: pd.Series) -> pd.Series:
    valid = s.replace([np.inf, -np.inf], np.nan)
    try:
        return pd.qcut(valid.rank(method="first"), 3, labels=["low", "mid", "high"])
    except ValueError:
        return pd.Series(pd.NA, index=s.index, dtype="object")


def _daily_event_payoff(sample: pd.DataFrame) -> pd.DataFrame:
    regime_cols = [*REGIME_COLS, *[f"{c}_bucket" for c in REGIME_COLS], "predicted_state"]
    rows = []
    for date, g in sample.groupby("trade_date", sort=True):
        row = {
            "trade_date": date,
            "year": int(g["year"].iloc[0]),
            "event_count": int(len(g)),
            "mean_fwd_ret_10": float(g["fwd_ret_10"].mean()),
            "median_fwd_ret_10": float(g["fwd_ret_10"].median()),
            "mean_industry_excess_10": float(g["fwd_industry_excess_10"].mean()),
            "hit_rate_10": float((g["fwd_ret_10"] > 0.0).mean()),
            "bad_rate_10": float(g["bad_trade_10"].astype(float).mean()),
            "top_decile_rate_10": float(g["top_decile_hit_10"].astype(float).mean()),
            "mean_drawdown_10": float(g["fwd_drawdown_10"].mean()),
        }
        for col in regime_cols:
            if col in g:
                row[col] = g[col].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def _conditional_payoff(data: pd.DataFrame, *, weight_mode: str) -> pd.DataFrame:
    rows = []
    for col in REGIME_COLS:
        bucket_col = f"{col}_bucket"
        if bucket_col not in data:
            continue
        for bucket, g in data.groupby(bucket_col, observed=True, dropna=True):
            rows.append(_payoff_row(g, regime_feature=col, bucket=str(bucket), weight_mode=weight_mode))
    out = pd.DataFrame(rows)
    return out.sort_values(["regime_feature", "bucket"]) if not out.empty else out


def _conditional_payoff_by_year(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in REGIME_COLS:
        bucket_col = f"{col}_bucket"
        if bucket_col not in daily:
            continue
        for (year, bucket), g in daily.groupby(["year", bucket_col], observed=True, dropna=True):
            row = _payoff_row(g, regime_feature=col, bucket=str(bucket), weight_mode="date")
            row["year"] = int(year)
            rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["regime_feature", "year", "bucket"]) if not out.empty else out


def _hmm_state_payoff(daily: pd.DataFrame) -> pd.DataFrame:
    if "predicted_state" not in daily:
        return pd.DataFrame()
    rows = []
    for state, g in daily.groupby("predicted_state", dropna=True):
        rows.append(_payoff_row(g, regime_feature="hmm_predicted_state", bucket=str(int(state)), weight_mode="date"))
    return pd.DataFrame(rows)


def _year_profile(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in daily.groupby("year"):
        row = _payoff_row(g, regime_feature="year", bucket=str(year), weight_mode="date")
        row["year"] = int(year)
        rows.append(row)
    return pd.DataFrame(rows)


def _payoff_row(g: pd.DataFrame, *, regime_feature: str, bucket: str, weight_mode: str) -> dict:
    ret_col = "mean_fwd_ret_10" if weight_mode == "date" else "fwd_ret_10"
    excess_col = "mean_industry_excess_10" if weight_mode == "date" else "fwd_industry_excess_10"
    hit_col = "hit_rate_10" if weight_mode == "date" else None
    bad_col = "bad_rate_10" if weight_mode == "date" else None
    top_col = "top_decile_rate_10" if weight_mode == "date" else None
    dd_col = "mean_drawdown_10" if weight_mode == "date" else "fwd_drawdown_10"
    return {
        "regime_feature": regime_feature,
        "bucket": bucket,
        "weight_mode": weight_mode,
        "days": int(g["trade_date"].nunique()),
        "events": int(g["event_count"].sum()) if "event_count" in g else int(len(g)),
        "mean_fwd_ret_10": float(g[ret_col].mean()),
        "median_fwd_ret_10": float(g[ret_col].median()),
        "mean_industry_excess_10": float(g[excess_col].mean()),
        "hit_rate_10": float(g[hit_col].mean()) if hit_col else float((g["fwd_ret_10"] > 0.0).mean()),
        "bad_rate_10": float(g[bad_col].mean()) if bad_col else float(g["bad_trade_10"].astype(float).mean()),
        "top_decile_rate_10": float(g[top_col].mean()) if top_col else float(g["top_decile_hit_10"].astype(float).mean()),
        "mean_drawdown_10": float(g[dd_col].mean()),
        "avg_event_count_per_day": float(g["event_count"].mean()) if "event_count" in g else np.nan,
    }


def _top_alpha_payoff(
    event_run: Path,
    sample: pd.DataFrame,
    regime: pd.DataFrame,
    states: pd.DataFrame,
    log,
) -> pd.DataFrame:
    frames = []
    sample_key = sample[
        [
            "trade_date",
            "ts_code",
            "fwd_ret_10",
            "fwd_industry_excess_10",
            "fwd_drawdown_10",
            "bad_trade_10",
            "top_decile_hit_10",
        ]
    ]
    for fold in FOLDS:
        path = event_run / fold["name"] / "predictions_adjusted_scored.parquet"
        if not path.exists():
            continue
        pred = pd.read_parquet(path)
        pred["trade_date"] = pd.to_datetime(pred["trade_date"])
        test = pred[pred["trade_date"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))]
        picks = test.sort_values(["trade_date", "score_alpha_only"], ascending=[True, False]).groupby("trade_date").head(5)
        picks = picks[["trade_date", "ts_code", "score_alpha_only", "bad_prob"]].merge(
            sample_key,
            on=["trade_date", "ts_code"],
            how="inner",
        )
        frames.append(picks)
    if not frames:
        return pd.DataFrame()
    top = pd.concat(frames, ignore_index=True)
    top = top.merge(regime[["trade_date", *REGIME_COLS]], on="trade_date", how="left")
    if not states.empty:
        top = top.merge(states[["trade_date", "predicted_state"]], on="trade_date", how="left")
    for col in REGIME_COLS:
        top[f"{col}_bucket"] = _tercile_bucket(top[col])
    top["year"] = top["trade_date"].dt.year
    daily = _daily_event_payoff(top.assign(event_count=1))
    rows = []
    rows.extend(_year_profile(daily).to_dict("records"))
    core = _conditional_payoff(daily, weight_mode="date")
    core["scope"] = "top_alpha"
    year = _year_profile(daily)
    year["scope"] = "top_alpha_year"
    out = pd.concat([year, core], ignore_index=True, sort=False)
    log(f"top alpha payoff picks={len(top):,} days={top['trade_date'].nunique():,}")
    return out


def _fmt_pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    return out


def _report(
    date_weighted: pd.DataFrame,
    by_year: pd.DataFrame,
    hmm: pd.DataFrame,
    year_profile: pd.DataFrame,
    top_alpha: pd.DataFrame,
) -> str:
    pct_cols = [
        "mean_fwd_ret_10",
        "median_fwd_ret_10",
        "mean_industry_excess_10",
        "hit_rate_10",
        "bad_rate_10",
        "top_decile_rate_10",
        "mean_drawdown_10",
    ]
    key_features = ["reversal_strength_20", "momentum_minus_reversal_20", "market_breadth_20", "market_ret_20"]
    core = date_weighted[date_weighted["regime_feature"].isin(key_features)].copy()
    core = _fmt_pct(core, pct_cols)
    yp = _fmt_pct(year_profile.copy(), pct_cols)
    h = _fmt_pct(hmm.copy(), pct_cols) if not hmm.empty else hmm
    by = by_year[by_year["regime_feature"].isin(["reversal_strength_20", "market_breadth_20"])].copy()
    by = _fmt_pct(by, pct_cols)
    top_y = top_alpha[top_alpha.get("scope", "").eq("top_alpha_year")].copy() if not top_alpha.empty else pd.DataFrame()
    top_y = _fmt_pct(top_y, pct_cols) if not top_y.empty else top_y
    return "\n".join(
        [
            "# Regime Conditional Payoff",
            "",
            "- sample: event-triggered permission main-board PIT Top1000 rows, 2022+",
            "- payoff: future 10 trading days, signal at T close and T+1 open entry convention",
            "- date-weighted tables average each signal date first, then group by regime",
            "",
            "## Year Profile",
            "",
            yp.to_markdown(index=False),
            "",
            "## Key Regime Buckets Date-Weighted",
            "",
            core.to_markdown(index=False),
            "",
            "## Year x Selected Regimes",
            "",
            by.to_markdown(index=False),
            "",
            "## HMM State Payoff",
            "",
            h.to_markdown(index=False) if not h.empty else "_No HMM states found._",
            "",
            "## Top Alpha Pick Payoff By Year",
            "",
            top_y.to_markdown(index=False) if not top_y.empty else "_No top alpha predictions found._",
            "",
        ]
    )


if __name__ == "__main__":
    main()
