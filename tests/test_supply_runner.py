"""End-to-end smoke test for the supply-contraction Qlib pipeline.

Publishes a synthetic drift panel as a real data version, writes a minimal config, and
runs ``SupplyContractionRunner.run`` -- exercising bin dump, qlib.init, dataset build,
two-model A/B LightGBM training, the A-share Qlib backtest, and the BacktestEngine
cross-check.  Skipped unless both qlib and lightgbm are importable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("qlib")
pytest.importorskip("lightgbm")

from factor_forge.data import DataVersionRepository  # noqa: E402
from factor_forge.ml.supply_runner import SupplyContractionRunner  # noqa: E402


def _write(path: Path, value: dict):
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _make_panel(days: int = 300, n_stocks: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2022-01-03", periods=days)
    rows = []
    for s in range(n_stocks):
        code = f"{s:06d}.SZ"
        price = 10.0 + s
        base_turnover = 0.5 + s * 0.3
        for d, date in enumerate(dates):
            ret = rng.normal(0, 0.02)
            price = max(1.0, price * (1 + ret))
            raw_open = price * (1 + rng.normal(0, 0.005))
            raw_high = max(price, raw_open) * (1 + abs(rng.normal(0, 0.008)))
            raw_low = min(price, raw_open) * (1 - abs(rng.normal(0, 0.008)))
            rows.append({
                "trade_date": date, "ts_code": code,
                "raw_open": raw_open, "raw_high": raw_high, "raw_low": raw_low,
                "raw_close": price, "pre_close": price / (1 + ret),
                "adj_factor": 1.0, "adj_open": raw_open, "adj_high": raw_high,
                "adj_low": raw_low, "adj_close": price,
                "volume_shares": 1_000_000.0, "amount_cny": 50_000_000.0 * (1 + s * 0.1),
                "pct_change": ret * 100,
                "turnover_rate": max(0.05, base_turnover + rng.normal(0, 0.4)),
                "total_mv_cny": 1e9 * (s + 1), "circ_mv_cny": 8e8 * (s + 1),
                "log_total_mv": np.log(1e9 * (s + 1)), "log_circ_mv": np.log(8e8 * (s + 1)),
                "industry_l1_code": f"I{s % 3}", "industry_l1_name": f"行业{s % 3}",
                "limit_up_price": price * 1.1, "limit_down_price": price * 0.9,
                "is_suspended": False, "is_limit_up_open": False, "is_limit_down_open": False,
                "is_st": False, "is_delisting_period": False,
                "listing_trade_days": 300 + d,
                "is_factor_eligible": True, "is_tradeable": True, "is_liquid": True,
                "st_status_known": True,
            })
    return pd.DataFrame(rows)


@pytest.mark.slow
def test_supply_runner_end_to_end(tmp_path):
    panel = _make_panel()
    dates = sorted(panel["trade_date"].unique())
    data_root, db = tmp_path / "data", tmp_path / "metadata.sqlite3"
    version = DataVersionRepository(data_root, db).publish(panel, source="test")

    project_path = tmp_path / "project.yaml"
    _write(project_path, {
        "project_name": "test_supply", "timezone": "Asia/Shanghai",
        "paths": {"data_root": str(data_root), "metadata_db": str(db), "artifacts_root": str(tmp_path / "runs")},
    })

    train_end = pd.Timestamp(dates[179]).strftime("%Y-%m-%d")
    valid_start = pd.Timestamp(dates[180]).strftime("%Y-%m-%d")
    valid_end = pd.Timestamp(dates[219]).strftime("%Y-%m-%d")
    test_start = pd.Timestamp(dates[220]).strftime("%Y-%m-%d")
    test_end = pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")

    config_path = tmp_path / "supply.yaml"
    _write(config_path, {
        "version": 1,
        "name": "supply_smoke",
        "project_config": str(project_path),
        "data_version": version,
        "require_full_segment_coverage": True,
        "segments": {
            "train": {"start": pd.Timestamp(dates[0]).strftime("%Y-%m-%d"), "end": train_end},
            "valid": {"start": valid_start, "end": valid_end},
            "test": {"start": test_start, "end": test_end},
        },
        "features": {
            "volume_residual_window": 80, "volume_residual_min_periods": 40,
            "volatility_window": 20, "min_listing_days": 30,
        },
        "label": {"horizon": 5, "label_method": "open_to_open", "industry_neutralize": True, "industry_aggregation": "leave_one_out"},
        "model": {"loss": "mse", "num_boost_round": 40, "early_stopping_rounds": 10, "learning_rate": 0.05, "num_leaves": 8, "min_data_in_leaf": 5},
        "backtest": {"topk": 5, "n_drop": 1, "hold_thresh": 1, "initial_cash": 1_000_000, "round_trip_cost_bps": 20, "trade_unit": 100},
        "crosscheck": {"enabled": True, "universe": "tradeable", "top_n": 5, "holding_days": 5, "cost_bps": 20},
        "ablation": {"models": [
            {"name": "model_a_controls", "feature_groups": ["controls"]},
            {"name": "model_b_supply", "feature_groups": ["controls", "supply_core", "composite"]},
        ]},
        "qlib_provider_root": str(tmp_path / "qlib_bin_cache"),
        "output_root": str(tmp_path / "supply_runs"),
    })

    summary = SupplyContractionRunner().run(config_path)

    # Core contract: the headline ablation verdict exists.
    assert "incremental_alpha" in summary
    inc = summary["incremental_alpha"]
    assert inc["baseline"] == "model_a_controls"
    assert inc["contender"] == "model_b_supply"
    assert inc["verdict"] in {"SUPPLY_ALPHA_PRESENT", "NO_INCREMENTAL_ALPHA"}
    assert "rank_ic_mean" in inc

    # Both models produced metrics.
    assert set(summary["models"]) == {"model_a_controls", "model_b_supply"}
    for m in summary["models"].values():
        assert "rank_ic_mean" in m and "annualized_return" in m

    # Artifacts landed.
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "predictions_model_a_controls.parquet").exists()
    assert (run_dir / "predictions_model_b_supply.parquet").exists()

    # Cross-check ran (advisory; only checks the field is present).
    assert summary.get("crosscheck") is not None
