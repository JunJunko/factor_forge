from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import web_app.server as server
from web_app.server import _factor_health_state


def test_factor_health_state_identifies_short_term_recovery() -> None:
    row = pd.Series(
        {
            "rolling_rank_ic_20": 0.02,
            "rolling_rank_ic_60": 0.01,
            "spread_20": 0.003,
            "spread_60": -0.004,
            "ic_velocity_20_60": 0.01,
            "spread_velocity_20_60": 0.007,
        }
    )

    assert _factor_health_state(row) == "RECOVERY"


def test_read_signal_explains_reliability_top5_changes(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {
        "signal_date": "2026-07-09",
        "signal_algorithm": "sell_impact_frozen_alpha_signal_reliability_lambda005_v1",
    }
    (tmp_path / "signal_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    candidates = pd.DataFrame(
        {
            "ts_code": ["A", "B", "C", "D", "E", "F"],
            "alpha_score": [0.60, 0.50, 0.40, 0.30, 0.20, 0.10],
            "final_score": [0.60, 0.50, 0.40, 0.30, 0.05, 0.35],
        }
    ).sort_values("final_score", ascending=False)
    candidates.to_csv(tmp_path / "top100_candidates.csv", index=False)
    candidates.head(5).assign(rank=range(1, 6)).to_csv(tmp_path / "top_recommendations.csv", index=False)

    monkeypatch.setattr(server, "ROOT", tmp_path.parent)
    signal = server._read_signal(tmp_path)

    assert signal["summary"]["reliability_impact"]["top5_replaced_count"] == 1
    promoted = next(row for row in signal["top"] if row["ts_code"] == "F")
    assert promoted["alpha_rank"] == 6
    assert promoted["final_rank"] == 4
    assert promoted["rank_change"] == 2


def test_signal_history_keeps_only_latest_active_run_per_date(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    signal_root = tmp_path / "signals"
    signal_root.mkdir()
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "SIGNAL_ROOT", signal_root)

    def create_run(name: str, signal_date: str, exposure: float, mtime: float, algorithm: str | None = None) -> None:
        run = signal_root / name
        run.mkdir()
        (run / "signal_summary.json").write_text(
            json.dumps(
                {
                    "signal_date": signal_date,
                    "entry_date_for_timing": "2026-07-10",
                    "signal_algorithm": algorithm or server.ACTIVE_SIGNAL_ALGORITHM,
                    "final_exposure": exposure,
                    "top_n": 5,
                    "predictable_candidates": 100,
                    "reliability_probability_mean": 0.7,
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame([{"rank": 1, "ts_code": "000001.SZ", "name": "Test", "target_weight": exposure}]).to_csv(
            run / "top_recommendations.csv", index=False
        )
        os.utime(run, (mtime, mtime))

    create_run("old", "2026-07-08", 0.4, 100.0)
    create_run("new", "2026-07-08", 0.8, 200.0)
    create_run("next", "2026-07-09", 1.0, 150.0)
    create_run("other_strategy", "2026-07-10", 1.0, 300.0, algorithm="other")

    history = server._signal_history_runs()

    assert [item["signal_date"] for item in history] == ["2026-07-09", "2026-07-08"]
    assert history[-1]["final_exposure"] == 0.8
    assert server._signal_history_detail("2026-07-08")["summary"]["final_exposure"] == 0.8
