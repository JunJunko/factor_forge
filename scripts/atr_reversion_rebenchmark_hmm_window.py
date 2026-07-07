"""Recompute an ATR HMM-window experiment with CSI1000 benchmark returns."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from atr_reversion_benchmark import CSI1000_CODE, csi1000_open_to_open_returns
from atr_reversion_hmm_window_comparison import _report, _summary, _yearly_row
from atr_reversion_small_portfolio_backtest import _json_default, _metrics


def main(
    source_run: str = (
        "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/"
        "hmm_window_comparison_20260707T002239Z"
    ),
) -> None:
    source = Path(source_run)
    output = source.parent / f"{source.name}_csi1000_benchmark_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    original = pd.read_csv(source / "hmm_window_metrics.csv")
    train = pd.read_csv(source / "alpha_train_summary.csv")
    state = pd.read_csv(source / "hmm_state_summary.csv")
    rows: list[dict] = []
    yearly_frames: list[pd.DataFrame] = []

    for row in original.to_dict("records"):
        fold = row["fold"]
        hmm_variant = row["hmm_variant"]
        policy = row["policy"]
        top_n = int(row["top_n"])
        cost = int(row["cost_bps"])
        tag = f"{fold}_{hmm_variant}_{policy}_top{top_n}_cost{cost}"
        src_dir = source / fold / hmm_variant
        out_dir = output / fold / hmm_variant
        out_dir.mkdir(parents=True, exist_ok=True)
        daily = pd.read_parquet(src_dir / f"daily_{tag}.parquet")
        trades = pd.read_parquet(src_dir / f"trades_{tag}.parquet")
        daily["trade_date"] = pd.to_datetime(daily["trade_date"])
        daily["benchmark_return"] = csi1000_open_to_open_returns(daily["trade_date"])
        daily["excess_return"] = daily["return"] - daily["benchmark_return"]
        daily.to_parquet(out_dir / f"daily_{tag}.parquet", index=False)
        trades.to_parquet(out_dir / f"trades_{tag}.parquet", index=False)
        metrics = _metrics(daily, trades)
        meta = {
            key: row[key]
            for key in [
                "fold",
                "test_start",
                "test_end",
                "alpha_variant",
                "hmm_variant",
                "policy",
                "top_n",
                "rebalance_days",
                "cost_bps",
                "best_state",
                "neutral_state",
                "worst_state",
                "avg_exposure",
                "avg_daily_turnover",
            ]
            if key in row
        }
        meta.update(metrics)
        rows.append(meta)
        yearly_frames.append(_yearly_row(meta, daily))
        log(
            f"{tag} ann={metrics['annualized_return']:.2%} "
            f"bench={metrics['benchmark_annualized_return']:.2%} "
            f"excess={metrics['annualized_excess_return']:.2%}"
        )

    metrics = pd.DataFrame(rows)
    yearly = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    summary = _summary(metrics)
    metrics.to_csv(output / "hmm_window_metrics.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "hmm_window_yearly.csv", index=False, encoding="utf-8-sig")
    train.to_csv(output / "alpha_train_summary.csv", index=False, encoding="utf-8-sig")
    state.to_csv(output / "hmm_state_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "hmm_window_summary.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "source_run": str(source),
                "run_dir": str(output),
                "benchmark": CSI1000_CODE,
                "benchmark_return": "open_to_open",
                "metrics": metrics.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(metrics, summary, yearly, train, state), encoding="utf-8")
    log("done")
    print(f"run_dir={output}")


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else (
        "artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/"
        "hmm_window_comparison_20260707T002239Z"
    )
    main(source)
