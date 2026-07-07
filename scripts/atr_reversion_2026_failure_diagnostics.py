"""Diagnose 2026 degradation for the ATR reversion strategy."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository

from atr_reversion_small_portfolio_backtest import _json_default
from atr_reversion_walk_forward import FOLDS, REBALANCE_DAYS


PIT_RUN = Path("artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z")
SOURCE_RUN = PIT_RUN / "hmm_window_comparison_20260707T002239Z"
CSI_RUN = PIT_RUN / "hmm_window_comparison_20260707T002239Z_csi1000_benchmark_20260707T020312Z"
HMM_VARIANT = "hmm_rolling_3y_pit"
POLICY = "atr_hmm_tiered"
TOP_N = 5
COST = 20


def main(
    source_run: str = str(SOURCE_RUN),
    csi_run: str = str(CSI_RUN),
) -> None:
    source = Path(source_run)
    csi = Path(csi_run)
    output = csi.parent / f"failure_diagnostics_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    version, panel = _load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    pit = pd.read_parquet(PIT_RUN / "pit_liquidity_flags.parquet")
    pit["trade_date"] = pd.to_datetime(pit["trade_date"])
    log(f"loaded panel={len(panel):,} pit={len(pit):,} version={version}")

    cycle_rows: list[dict] = []
    trade_rows: list[dict] = []
    fit_rows: list[dict] = []
    decile_rows: list[dict] = []
    monthly_rows: list[dict] = []

    for fold in FOLDS:
        fold_name = fold["name"]
        log(f"diagnosing {fold_name}")
        tag = f"{fold_name}_{HMM_VARIANT}_{POLICY}_top{TOP_N}_cost{COST}"
        daily = pd.read_parquet(csi / fold_name / HMM_VARIANT / f"daily_{tag}.parquet")
        trades = pd.read_parquet(csi / fold_name / HMM_VARIANT / f"trades_{tag}.parquet")
        pred = pd.read_parquet(source / fold_name / "predictions_valid_test_rolling_2y.parquet")
        states = pd.read_csv(source / fold_name / HMM_VARIANT / "hmm_daily_states.csv")
        daily["trade_date"] = pd.to_datetime(daily["trade_date"])
        trades["trade_date"] = pd.to_datetime(trades["trade_date"])
        pred["trade_date"] = pd.to_datetime(pred["trade_date"])
        states["trade_date"] = pd.to_datetime(states["trade_date"])

        cycle = _cycle_stats(daily, fold_name)
        cycle_rows.extend(cycle.to_dict("records"))
        trade_rows.extend(_trade_roundtrips(trades, fold_name).to_dict("records"))
        fit, decile, monthly = _factor_fit(panel, pit, pred, states, fold, fold_name, log)
        fit_rows.extend(fit)
        decile_rows.extend(decile)
        monthly_rows.extend(monthly)

    cycle_df = pd.DataFrame(cycle_rows)
    trade_df = pd.DataFrame(trade_rows)
    fit_df = pd.DataFrame(fit_rows)
    decile_df = pd.DataFrame(decile_rows)
    monthly_df = pd.DataFrame(monthly_rows)
    year_summary = _year_summary(cycle_df, trade_df, fit_df)
    month_summary = _month_summary(monthly_df)

    cycle_df.to_csv(output / "cycle_returns.csv", index=False, encoding="utf-8-sig")
    trade_df.to_csv(output / "roundtrip_trades.csv", index=False, encoding="utf-8-sig")
    fit_df.to_csv(output / "factor_fit_daily.csv", index=False, encoding="utf-8-sig")
    decile_df.to_csv(output / "factor_decile_returns.csv", index=False, encoding="utf-8-sig")
    monthly_df.to_csv(output / "factor_fit_monthly.csv", index=False, encoding="utf-8-sig")
    year_summary.to_csv(output / "year_failure_summary.csv", index=False, encoding="utf-8-sig")
    month_summary.to_csv(output / "month_failure_summary.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "source_run": str(source),
                "csi_run": str(csi),
                "run_dir": str(output),
                "hmm_variant": HMM_VARIANT,
                "policy": POLICY,
                "top_n": TOP_N,
                "cost_bps": COST,
                "year_summary": year_summary.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(year_summary, month_summary, decile_df), encoding="utf-8")
    log("done")
    print(f"run_dir={output}")


def _load_panel() -> tuple[str, pd.DataFrame]:
    project = load_project("configs/project.yaml")
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest("latest")
    _, panel = repo.load_panel(version)
    return version, panel


def _cycle_stats(daily: pd.DataFrame, fold: str) -> pd.DataFrame:
    rows = []
    d = daily.sort_values("trade_date").reset_index(drop=True).copy()
    dates = d["trade_date"].tolist()
    for signal_idx in range(0, len(dates) - 1, REBALANCE_DAYS):
        start = signal_idx + 1
        end = min(signal_idx + REBALANCE_DAYS, len(dates) - 1)
        segment = d.iloc[start : end + 1].copy()
        if segment.empty:
            continue
        ret = float((1.0 + segment["return"].fillna(0.0)).prod() - 1.0)
        bench = float((1.0 + segment["benchmark_return"].fillna(0.0)).prod() - 1.0)
        wealth = (1.0 + segment["return"].fillna(0.0)).cumprod()
        dd = float((wealth / wealth.cummax() - 1.0).min())
        rows.append(
            {
                "fold": fold,
                "year": int(segment["trade_date"].iloc[0].year),
                "signal_date": dates[signal_idx],
                "start_date": segment["trade_date"].iloc[0],
                "end_date": segment["trade_date"].iloc[-1],
                "cycle_return": ret,
                "cycle_benchmark_return": bench,
                "cycle_excess_return": ret - bench,
                "cycle_drawdown": dd,
                "avg_exposure": float(segment["exposure"].mean()),
                "avg_holding_count": float(segment["holding_count"].mean()),
                "turnover": float(segment["turnover"].sum()),
                "active": bool(segment["exposure"].mean() > 0.0),
            }
        )
    return pd.DataFrame(rows)


def _trade_roundtrips(trades: pd.DataFrame, fold: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    buys: dict[str, list[dict]] = {}
    rows = []
    for row in trades.sort_values(["trade_date", "side"]).to_dict("records"):
        code = row["ts_code"]
        if row["side"] == "BUY":
            buys.setdefault(code, []).append(row)
        elif row["side"] == "SELL" and buys.get(code):
            buy = buys[code].pop(0)
            gross = float(row["value"]) / float(buy["value"]) - 1.0
            net = (float(row["value"]) - float(row["cost"])) / (float(buy["value"]) + float(buy["cost"])) - 1.0
            rows.append(
                {
                    "fold": fold,
                    "year": int(pd.Timestamp(row["trade_date"]).year),
                    "ts_code": code,
                    "buy_date": buy["trade_date"],
                    "sell_date": row["trade_date"],
                    "buy_value": float(buy["value"]),
                    "sell_value": float(row["value"]),
                    "gross_return": gross,
                    "net_return": net,
                    "buy_exposure": float(buy.get("exposure", np.nan)),
                    "sell_exposure": float(row.get("exposure", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def _factor_fit(
    panel: pd.DataFrame,
    pit: pd.DataFrame,
    pred: pd.DataFrame,
    states: pd.DataFrame,
    fold: dict,
    fold_name: str,
    log,
) -> tuple[list[dict], list[dict], list[dict]]:
    test_start = pd.Timestamp(fold["test_start"])
    test_end = pd.Timestamp(fold["test_end"])
    p = panel[panel["trade_date"].between(test_start, test_end)].merge(
        pit[["trade_date", "ts_code", "pit_top1000"]],
        on=["trade_date", "ts_code"],
        how="left",
    )
    p["pit_top1000"] = p["pit_top1000"].fillna(False).astype(bool)
    pred = pred[pred["trade_date"].between(test_start, test_end)].copy()
    data = p.merge(pred, on=["trade_date", "ts_code"], how="left")
    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    data["next_open"] = data.groupby("ts_code")["adj_open"].shift(-1)
    data["exit_open"] = data.groupby("ts_code")["adj_open"].shift(-(REBALANCE_DAYS + 1))
    data["fwd_ret_10d"] = data["exit_open"] / data["next_open"] - 1.0
    eligible = (
        data["pit_top1000"]
        & data["factor_value"].notna()
        & data["fwd_ret_10d"].replace([np.inf, -np.inf], np.nan).notna()
    )
    data = data.loc[eligible].copy()
    state_map = states.set_index("trade_date")["predicted_state"].to_dict()
    data["hmm_state"] = data["trade_date"].map(state_map)
    daily_rows = []
    decile_rows = []
    monthly_rows = []
    for date, g in data.groupby("trade_date", sort=True):
        if len(g) < 100:
            continue
        ic = g["factor_value"].corr(g["fwd_ret_10d"], method="spearman")
        top = g.nlargest(TOP_N, "factor_value")
        q = pd.qcut(g["factor_value"].rank(method="first"), 10, labels=False, duplicates="drop")
        g = g.assign(decile=q)
        dec = g.groupby("decile", observed=True)["fwd_ret_10d"].mean()
        daily_rows.append(
            {
                "fold": fold_name,
                "trade_date": date,
                "year": int(pd.Timestamp(date).year),
                "month": pd.Timestamp(date).strftime("%Y-%m"),
                "rank_ic": float(ic) if pd.notna(ic) else np.nan,
                "top5_forward_return": float(top["fwd_ret_10d"].mean()),
                "top5_hit_rate": float((top["fwd_ret_10d"] > 0.0).mean()),
                "universe_forward_return": float(g["fwd_ret_10d"].mean()),
                "top5_excess_forward_return": float(top["fwd_ret_10d"].mean() - g["fwd_ret_10d"].mean()),
                "top5_score_mean": float(top["factor_value"].mean()),
                "top5_score_spread": float(top["factor_value"].max() - top["factor_value"].min()),
                "hmm_state": int(g["hmm_state"].dropna().iloc[0]) if g["hmm_state"].notna().any() else -1,
            }
        )
        for decile, value in dec.items():
            decile_rows.append(
                {
                    "fold": fold_name,
                    "trade_date": date,
                    "year": int(pd.Timestamp(date).year),
                    "decile": int(decile),
                    "mean_forward_return": float(value),
                }
            )
    daily = pd.DataFrame(daily_rows)
    if not daily.empty:
        for (year, month), g in daily.groupby(["year", "month"], sort=True):
            monthly_rows.append(
                {
                    "fold": fold_name,
                    "year": int(year),
                    "month": month,
                    "days": int(len(g)),
                    "rank_ic_mean": float(g["rank_ic"].mean()),
                    "top5_forward_return_mean": float(g["top5_forward_return"].mean()),
                    "top5_hit_rate_mean": float(g["top5_hit_rate"].mean()),
                    "top5_excess_forward_return_mean": float(g["top5_excess_forward_return"].mean()),
                }
            )
    log(f"{fold_name}: factor fit rows={len(daily_rows)} eligible_rows={len(data):,}")
    return daily_rows, decile_rows, monthly_rows


def _year_summary(cycle: pd.DataFrame, trades: pd.DataFrame, fit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    years = sorted(set(cycle["year"].dropna().astype(int)) | set(fit["year"].dropna().astype(int)))
    for year in years:
        c = cycle[cycle["year"].eq(year)]
        active = c[c["active"]]
        t = trades[trades["year"].eq(year)] if not trades.empty else pd.DataFrame()
        f = fit[fit["year"].eq(year)]
        row = {
            "year": year,
            "cycles": int(len(c)),
            "active_cycles": int(len(active)),
            "cycle_win_rate": float((active["cycle_return"] > 0.0).mean()) if len(active) else np.nan,
            "cycle_excess_win_rate": float((active["cycle_excess_return"] > 0.0).mean()) if len(active) else np.nan,
            "avg_cycle_return": float(active["cycle_return"].mean()) if len(active) else np.nan,
            "median_cycle_return": float(active["cycle_return"].median()) if len(active) else np.nan,
            "avg_cycle_excess": float(active["cycle_excess_return"].mean()) if len(active) else np.nan,
            "avg_win_cycle": float(active.loc[active["cycle_return"] > 0, "cycle_return"].mean()) if len(active) else np.nan,
            "avg_loss_cycle": float(active.loc[active["cycle_return"] <= 0, "cycle_return"].mean()) if len(active) else np.nan,
            "worst_cycle": float(active["cycle_return"].min()) if len(active) else np.nan,
            "avg_exposure": float(c["avg_exposure"].mean()) if len(c) else np.nan,
            "roundtrip_trades": int(len(t)),
            "trade_win_rate": float((t["net_return"] > 0.0).mean()) if len(t) else np.nan,
            "avg_trade_return": float(t["net_return"].mean()) if len(t) else np.nan,
            "median_trade_return": float(t["net_return"].median()) if len(t) else np.nan,
            "avg_trade_win": float(t.loc[t["net_return"] > 0, "net_return"].mean()) if len(t) else np.nan,
            "avg_trade_loss": float(t.loc[t["net_return"] <= 0, "net_return"].mean()) if len(t) else np.nan,
            "rank_ic_mean": float(f["rank_ic"].mean()) if len(f) else np.nan,
            "rank_ic_positive_ratio": float((f["rank_ic"] > 0.0).mean()) if len(f) else np.nan,
            "top5_forward_return_mean": float(f["top5_forward_return"].mean()) if len(f) else np.nan,
            "top5_hit_rate_mean": float(f["top5_hit_rate"].mean()) if len(f) else np.nan,
            "top5_excess_forward_return_mean": float(f["top5_excess_forward_return"].mean()) if len(f) else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _month_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    return monthly.sort_values("month").reset_index(drop=True)


def _fmt_pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    return out


def _report(year_summary: pd.DataFrame, month_summary: pd.DataFrame, decile: pd.DataFrame) -> str:
    pct_cols = [
        "cycle_win_rate",
        "cycle_excess_win_rate",
        "avg_cycle_return",
        "median_cycle_return",
        "avg_cycle_excess",
        "avg_win_cycle",
        "avg_loss_cycle",
        "worst_cycle",
        "avg_exposure",
        "trade_win_rate",
        "avg_trade_return",
        "median_trade_return",
        "avg_trade_win",
        "avg_trade_loss",
        "rank_ic_positive_ratio",
        "top5_forward_return_mean",
        "top5_hit_rate_mean",
        "top5_excess_forward_return_mean",
    ]
    y = _fmt_pct(year_summary, pct_cols)
    if "rank_ic_mean" in y:
        y["rank_ic_mean"] = y["rank_ic_mean"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    m = _fmt_pct(month_summary, [
        "top5_forward_return_mean",
        "top5_hit_rate_mean",
        "top5_excess_forward_return_mean",
    ])
    if "rank_ic_mean" in m:
        m["rank_ic_mean"] = m["rank_ic_mean"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    dec = decile.groupby(["year", "decile"], observed=True)["mean_forward_return"].mean().reset_index()
    dec = _fmt_pct(dec, ["mean_forward_return"])
    return "\n".join(
        [
            "# ATR 2026 Failure Diagnostics",
            "",
            "Main strategy: rolling_2y alpha + hmm_rolling_3y_pit + tiered + Top5 + 10-day rebalance + 20bps.",
            "",
            "## Year Summary",
            "",
            y.to_markdown(index=False),
            "",
            "## Monthly Factor Fit",
            "",
            m.to_markdown(index=False) if not m.empty else "No monthly rows.",
            "",
            "## Decile Forward Returns",
            "",
            dec.to_markdown(index=False) if not dec.empty else "No decile rows.",
            "",
        ]
    )


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else str(SOURCE_RUN)
    csi = sys.argv[2] if len(sys.argv) > 2 else str(CSI_RUN)
    main(source, csi)
