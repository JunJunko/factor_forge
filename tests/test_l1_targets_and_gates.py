from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from factor_forge.config import ExperimentSpec, FactorSpec, L1Config, PrimaryGateConfig
from factor_forge.evaluation import compare_daily_rank_ic
from factor_forge.evaluation.l1 import (
    _evaluate_primary_gate,
    build_forward_targets,
    evaluate_conditional_ic,
    evaluate_predictive_power,
)


pytestmark = [
    pytest.mark.filterwarnings("ignore:An input array is constant"),
    pytest.mark.filterwarnings("ignore:invalid value encountered in divide"),
]


def _factor_spec() -> FactorSpec:
    return FactorSpec.model_validate({
        "version": 1,
        "factor": {
            "name": "test_factor",
            "label": "test",
            "description": "test",
            "hypothesis": "test",
            "direction": "positive",
            "expected_shape": "monotonic",
        },
        "data": {"frequency": "daily", "required_fields": ["close"], "lookback_days": 1},
        "scope": {"universe": "default", "cross_section": "market", "min_group_size": 2},
        "calculation": {
            "formula": "ret(close, 1)",
            "missing_policy": "skip",
            "winsorize": "none",
            "standardize": "none",
        },
        "output": {"value_field": "factor_value"},
    })


def _target_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=3)
    rows = []
    for code, opens in (("A", [10, 10, 12]), ("B", [20, 20, 22]), ("C", [30, 30, 33])):
        for date, value in zip(dates, opens, strict=True):
            rows.append({
                "trade_date": date,
                "ts_code": code,
                "adj_open": value,
                "industry_l1_code": "I1",
            })
    return pd.DataFrame(rows).sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def test_forward_targets_use_signal_date_leave_one_out_industry_return():
    panel = _target_panel()
    targets = build_forward_targets(panel, 1)
    first = panel["trade_date"].eq(panel["trade_date"].min())
    assert targets["stock_return"][first].round(6).tolist() == [0.2, 0.1, 0.1]
    assert targets["stock_minus_sw_l1_return"][first].round(6).tolist() == [0.1, -0.05, -0.05]


def test_forward_targets_keep_point_in_time_industry_membership():
    panel = _target_panel()
    second_date = panel["trade_date"].sort_values().unique()[1]
    panel.loc[(panel["trade_date"] == second_date) & panel["ts_code"].eq("A"), "industry_l1_code"] = "NEW"
    targets = build_forward_targets(panel, 1)
    first = panel["trade_date"].eq(panel["trade_date"].min())
    assert targets["stock_minus_sw_l1_return"][first].round(6).tolist() == [0.1, -0.05, -0.05]


def _evaluation_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=12)
    rows = []
    values = []
    conditions = []
    for stock in range(20):
        industry = "HIGH" if stock >= 10 else "LOW"
        growth = 0.01 if industry == "HIGH" else 0.0
        for day, date in enumerate(dates):
            price = 10 * (1 + growth) ** day
            rows.append({
                "trade_date": date,
                "ts_code": f"S{stock:02}",
                "adj_open": price,
                "log_total_mv": np.nan,
                "industry_l1_code": industry,
                "is_liquid": True,
            })
            values.append({
                "trade_date": date,
                "ts_code": f"S{stock:02}",
                "factor_value": float(stock),
            })
            conditions.append({
                "trade_date": date,
                "ts_code": f"S{stock:02}",
                "factor_value": float(stock),
            })
    return pd.DataFrame(rows), pd.DataFrame(values), pd.DataFrame(conditions)


def test_predictive_power_executes_each_configured_target():
    panel, values, _ = _evaluation_panel()
    config = L1Config(
        forward_horizons=[1],
        quantile_groups=5,
        min_cross_section=5,
        universes=["liquid"],
        targets=["stock_return", "stock_minus_sw_l1_return"],
    )
    result = evaluate_predictive_power(panel, values, _factor_spec(), config)
    by_target = {row["target"]: row for row in result["results"]}
    assert set(by_target) == {"stock_return", "stock_minus_sw_l1_return"}
    assert by_target["stock_return"]["rank_ic"]["mean"] is not None
    assert by_target["stock_minus_sw_l1_return"]["rank_ic"]["mean"] is None


def test_standard_l1_uses_hac_p_values_for_fdr():
    panel, values, _ = _evaluation_panel()
    config = L1Config(
        forward_horizons=[5],
        quantile_groups=5,
        min_cross_section=5,
        universes=["liquid"],
        targets=["stock_return"],
    )
    result = evaluate_predictive_power(panel, values, _factor_spec(), config)
    row = result["results"][0]
    assert row["rank_ic"]["nw_lags"] == 4
    assert "nw_p_value" in row["rank_ic"]
    assert row["fdr_q"] == row["rank_ic"]["nw_p_value"]
    assert result["inference"].startswith("Newey-West HAC")


def test_primary_gate_selects_exact_target_variant_universe_and_horizon():
    rows = [
        {
            "target": "stock_return", "variant": "raw", "universe": "liquid", "horizon": 5,
            "rank_ic": {"mean": 0.03, "positive_ratio": 0.60}, "fdr_q": 0.01,
            "top_bottom_mean": 0.02,
        },
        {
            "target": "stock_minus_sw_l1_return", "variant": "raw",
            "universe": "liquid", "horizon": 5,
            "rank_ic": {"mean": 0.005, "positive_ratio": 0.70}, "fdr_q": 0.01,
            "top_bottom_mean": 0.02,
        },
    ]
    gate = PrimaryGateConfig(
        target="stock_minus_sw_l1_return", variant="raw", universe="liquid", horizon=5,
        min_mean=0.01, min_positive_ratio=0.50, max_fdr_q=0.10,
        allow_top_tail_fallback=False,
    )
    result = _evaluate_primary_gate(rows, gate)
    assert result["passed"] is False
    assert result["checks"]["mean"] is False
    assert result["tail_fallback_used"] is False


def test_conditional_ic_executes_industry_relative_target_and_q5_gate():
    panel, values, conditions = _evaluation_panel()
    config = L1Config.model_validate({
        "forward_horizons": [1],
        "quantile_groups": 5,
        "min_cross_section": 5,
        "universes": ["liquid"],
        "targets": ["stock_minus_sw_l1_return"],
        "conditional_ic": {
            "enabled": True,
            "quantile_groups": 5,
            "min_group_size": 3,
            "primary_gate": {
                "target": "stock_minus_sw_l1_return",
                "variant": "raw",
                "universe": "liquid",
                "horizon": 1,
                "condition_quantile": 5,
                "max_fdr_q": None,
            },
        },
    })
    result, daily = evaluate_conditional_ic(
        panel, values, conditions, _factor_spec(), config, "condition"
    )
    assert {row["target"] for row in result["results"]} == {"stock_minus_sw_l1_return"}
    assert result["primary_gate"]["selector"]["condition_quantile"] == 5
    assert "target" in daily.columns


def test_primary_gate_must_reference_declared_experiment_axes():
    with pytest.raises(ValidationError, match="target must be listed"):
        ExperimentSpec.model_validate({
            "name": "bad_gate",
            "stage_l1": {
                "forward_horizons": [5],
                "universes": ["liquid"],
                "targets": ["stock_return"],
                "primary_gate": {
                    "target": "stock_minus_sw_l1_return",
                    "variant": "raw",
                    "universe": "liquid",
                    "horizon": 5,
                },
            },
        })


def test_paired_ic_comparator_classifies_retained_and_composition_cases():
    dates = pd.bdate_range("2024-01-01", periods=100)
    wave = np.sin(np.arange(100)) * 0.001
    market = pd.DataFrame({"trade_date": dates, "rank_ic": 0.02 + wave})
    retained = pd.DataFrame({"trade_date": dates, "rank_ic": 0.015 + wave * 0.5})
    composition = pd.DataFrame({"trade_date": dates, "rank_ic": 0.004 + wave * 0.2})

    retained_result = compare_daily_rank_ic(market, retained, horizon=5)
    composition_result = compare_daily_rank_ic(market, composition, horizon=5)

    assert retained_result["classification"] == "weakens_cross_section_composition"
    assert retained_result["retention_ratio"] == pytest.approx(0.75, rel=0.02)
    assert composition_result["classification"] == "supports_cross_section_composition"
    assert composition_result["nw_p_value"] <= 0.10
