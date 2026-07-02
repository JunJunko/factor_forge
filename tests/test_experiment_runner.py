from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import pytest
from pydantic import ValidationError

from factor_forge.data import DataVersionRepository
from factor_forge.experiments import ExperimentRunner
from factor_forge.config import ExperimentSpec


def _write(path: Path, value: dict):
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def test_unknown_yaml_key_is_rejected_instead_of_using_a_default():
    with pytest.raises(ValidationError, match="top_nn"):
        ExperimentSpec.model_validate({"name": "typo", "stage_l2": {"top_nn": [2]}})


def test_declarative_experiment_runs_without_factor_code_changes(tmp_path):
    dates = pd.bdate_range("2023-01-02", periods=90)
    rows = []
    for stock in range(30):
        drift = 0.0002 + stock * 0.00008
        for day, date in enumerate(dates):
            price = 10 * (1 + drift) ** day
            rows.append({
                "trade_date": date, "ts_code": f"{stock:06d}.SZ",
                "raw_open": price, "raw_high": price * 1.01, "raw_low": price * 0.99,
                "raw_close": price, "pre_close": price / (1 + drift), "adj_factor": 1.0,
                "adj_open": price, "adj_high": price * 1.01, "adj_low": price * 0.99,
                "adj_close": price, "volume_shares": 5e6, "amount_cny": 8e7,
                "pct_change": drift * 100, "total_mv_cny": 1e9 + stock * 1e7,
                "circ_mv_cny": 8e8 + stock * 1e7,
                "log_total_mv": np.log(1e9 + stock * 1e7),
                "log_circ_mv": np.log(8e8 + stock * 1e7), "turnover_rate": 1.0,
                "industry_l1_code": f"I{stock % 3}", "industry_l1_name": f"行业{stock % 3}",
                "limit_up_price": price * 1.1, "limit_down_price": price * 0.9,
                "is_suspended": False, "is_limit_up_open": False, "is_limit_down_open": False,
                "is_st": False, "is_delisting_period": False, "listing_trade_days": 100 + day,
                "is_factor_eligible": True, "is_tradeable": True, "is_liquid": True,
                "st_status_known": True,
            })
    panel = pd.DataFrame(rows)
    data_root, db = tmp_path / "data", tmp_path / "metadata.sqlite3"
    version = DataVersionRepository(data_root, db).publish(panel, source="test")
    project_path, factor_path, experiment_path = tmp_path / "project.yaml", tmp_path / "factor.yaml", tmp_path / "experiment.yaml"
    _write(project_path, {
        "project_name": "test", "timezone": "Asia/Shanghai",
        "paths": {"data_root": str(data_root), "metadata_db": str(db), "artifacts_root": str(tmp_path / "runs")},
    })
    _write(factor_path, {
        "version": 1,
        "factor": {"name": "momentum_3d", "label": "三日动量", "description": "测试",
                   "hypothesis": "趋势延续", "direction": "positive", "expected_shape": "monotonic"},
        "data": {"frequency": "daily", "required_fields": ["close"], "lookback_days": 3},
        "scope": {"universe": "default", "cross_section": "market", "min_group_size": 2},
        "calculation": {"formula": "ret(close, 3)", "missing_policy": "skip",
                        "winsorize": "none", "standardize": "none"},
        "output": {"value_field": "factor_value"},
    })
    _write(experiment_path, {
        "version": 1, "name": "test_v1", "project_config": str(project_path),
        "factor_config": str(factor_path), "data_version": version,
        "sample_start_date": dates[10].strftime("%Y%m%d"),
        "stage_l0": {"min_coverage": 0.7, "max_missing_rate": 0.3,
                     "min_daily_cross_section": 20, "min_unique_ratio": 0.01},
        "stage_l1": {"forward_horizons": [1, 3], "quantile_groups": 5,
                     "min_cross_section": 20, "universes": ["liquid"]},
        "stage_l2": {"universes": ["liquid"], "holding_periods": [1, 3],
                     "top_n": [2, 5], "cost_scenarios_bps": [0, 20],
                     "execution_constraints": {"min_listing_days": 60}},
    })
    result = ExperimentRunner().run(experiment_path)
    assert result["status"] == "SUCCESS"
    run_dir = Path(result["run_dir"])
    assert (run_dir / "factor_values.parquet").exists()
    assert (run_dir / "alpha_assessment.json").exists()
    assert len(list((run_dir / "l2").glob("*/metrics.json"))) == 8
    saved_factors = pd.read_parquet(run_dir / "factor_values.parquet")
    first_date = saved_factors["trade_date"].min()
    assert first_date == dates[10]
    assert saved_factors.loc[saved_factors["trade_date"] == first_date, "factor_value"].notna().all()
