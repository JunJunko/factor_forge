"""Generate a shadow-trading signal sheet from a PIT ATR prediction run."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main(
    prediction_path: str,
    states_path: str,
    out_dir: str | None = None,
    signal_date: str | None = None,
    top_n: int = 5,
    policy: str = "atr_hmm_tiered",
) -> None:
    pred_path = Path(prediction_path)
    pred = pd.read_parquet(pred_path)
    pred["trade_date"] = pd.to_datetime(pred["trade_date"])
    if signal_date is None:
        date = pred["trade_date"].max()
    else:
        date = pd.Timestamp(signal_date)
    states = pd.read_csv(states_path)
    states["trade_date"] = pd.to_datetime(states["trade_date"])
    state = states.loc[states["trade_date"].eq(date)]
    if state.empty:
        exposure = 1.0
        state_info = {}
    else:
        row = state.iloc[0]
        # Default state ranks from the calibrated ATR HMM runs observed so far:
        # best=0, neutral=1, worst=2.  The generator is intentionally simple and
        # writes the exposure so a human can review before placing orders.
        if policy == "ungated":
            exposure = 1.0
        elif policy == "atr_hmm_hard_best":
            exposure = 1.0 if int(row["predicted_state"]) == 0 else 0.0
        else:
            exposure = {0: 1.0, 1: 0.5, 2: 0.0}.get(int(row["predicted_state"]), 1.0)
        state_info = row.to_dict()
    daily = pred.loc[pred["trade_date"].eq(date)].dropna(subset=["factor_value"]).copy()
    picks = daily.sort_values(["factor_value", "ts_code"], ascending=[False, True]).head(top_n)
    picks["target_weight"] = exposure / max(len(picks), 1)
    picks["policy"] = policy
    picks["exposure"] = exposure
    for key, value in state_info.items():
        if key != "trade_date":
            picks[f"hmm_{key}"] = value
    out = Path(out_dir) if out_dir else pred_path.parent
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"shadow_signal_{date:%Y%m%d}_{policy}_top{top_n}.csv"
    picks.rename(columns={"trade_date": "signal_date"})[
        ["signal_date", "ts_code", "factor_value", "target_weight", "policy", "exposure"]
        + [c for c in picks.columns if c.startswith("hmm_")]
    ].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"wrote {out_path}")
    print(picks[["trade_date", "ts_code", "factor_value", "target_weight"]].to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: python scripts/atr_reversion_shadow_signal.py PREDICTIONS_PARQUET HMM_STATES_CSV [OUT_DIR] [SIGNAL_DATE] [TOP_N] [POLICY]"
        )
    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3] if len(sys.argv) > 3 else None,
        sys.argv[4] if len(sys.argv) > 4 else None,
        int(sys.argv[5]) if len(sys.argv) > 5 else 5,
        sys.argv[6] if len(sys.argv) > 6 else "atr_hmm_tiered",
    )

