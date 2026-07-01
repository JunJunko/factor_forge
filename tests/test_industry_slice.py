from __future__ import annotations

import numpy as np
import pandas as pd

from factor_forge.config import ExperimentSpec
from factor_forge.research.industry.context import IndustryContextBuilder
from factor_forge.research.industry.neutralize import IndustryNeutralizer
from factor_forge.research.industry.residual_return import IndustryResidualReturnBuilder
from factor_forge.research.industry.selector import IndustrySelector
from factor_forge.research.industry.slice_mapper import IndustrySliceMapper
from factor_forge.experiments.runner import industry_slice_enabled


def test_old_and_concise_experiment_shapes_are_compatible():
    old = ExperimentSpec.model_validate({"version": 1, "name": "old", "factor_config": "factor.yaml"})
    assert old.industry_slice is None
    assert old.stage_l1.targets == ["stock_return"]
    concise = ExperimentSpec.model_validate({
        "version": 1, "experiment": {"id": "slice", "level": "L1"},
        "evaluation": {"horizons": [1, 3], "targets": ["stock_return", "stock_minus_sw_l1_return"]},
        "industry_slice": {"enabled": True},
    })
    assert concise.name == "slice"
    assert concise.stage_l1.forward_horizons == [1, 3]
    assert concise.industry_slice.scopes == ["all", "top5", "bottom5"]


def test_execution_branch_for_missing_disabled_and_enabled_config():
    missing = ExperimentSpec.model_validate({"name": "missing", "factor_config": "f.yaml"})
    disabled = ExperimentSpec.model_validate({"name": "disabled", "factor_config": "f.yaml",
                                              "industry_slice": {"enabled": False}})
    enabled = ExperimentSpec.model_validate({"name": "enabled", "factor_config": "f.yaml",
                                             "industry_slice": {"enabled": True}})
    assert not industry_slice_enabled(missing)
    assert not industry_slice_enabled(disabled)
    assert industry_slice_enabled(enabled)


def test_context_keeps_point_in_time_industry_change():
    panel = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "ts_code": ["A", "A"], "industry_l1_code": ["OLD", "NEW"],
        "industry_l1_name": ["Old", "New"],
    })
    result = IndustryContextBuilder().build(panel)
    assert result["industry_code"].tolist() == ["OLD", "NEW"]


def test_daily_ridge_neutralization_reduces_control_correlation():
    rng = np.random.default_rng(7)
    n = 60
    controls = rng.normal(size=(n, 3))
    raw = controls @ np.array([2.0, -1.5, .8]) + rng.normal(scale=.15, size=n)
    frame = pd.DataFrame({"trade_date": pd.Timestamp("2024-01-02"),
                          "industry_code": [f"I{x:02}" for x in range(n)], "raw_score": raw,
                          "relative_strength_level": controls[:, 0],
                          "industry_excess_return_5d": controls[:, 1],
                          "log_industry_amount_share": controls[:, 2]})
    before = max(abs(frame["raw_score"].corr(frame[c])) for c in IndustryNeutralizer.controls)
    result = IndustryNeutralizer().transform(frame, alpha=.01)
    after = max(abs(result["neutral_score"].corr(result[c])) for c in IndustryNeutralizer.controls)
    assert after < before * .2


def test_top_bottom_are_bounded_stable_and_disjoint():
    n = 12
    frame = pd.DataFrame({"trade_date": pd.Timestamp("2024-01-02"),
                          "industry_code": [f"I{x:02}" for x in range(n)],
                          "relative_strength_velocity": np.arange(n), "breadth_velocity": np.arange(n),
                          "relative_strength_level": np.linspace(0, 1, n),
                          "industry_excess_return_5d": np.linspace(1, 0, n),
                          "log_industry_amount_share": np.sin(np.arange(n))})
    result = IndustrySelector().select(frame)
    assert result["top2_flag"].sum() <= 2
    assert result["top5_flag"].sum() <= 5
    assert result["top10_flag"].sum() <= 10
    assert result["bottom5_flag"].sum() <= 5
    assert not (result["top5_flag"] & result["bottom5_flag"]).any()


def test_slice_mapping_and_leave_one_out_residual_return():
    dates = pd.bdate_range("2024-01-01", periods=3)
    stocks = pd.DataFrame([
        {"trade_date": date, "ts_code": code, "industry_code": "I1", "industry_name": "One",
         "adj_open": opens[i]}
        for code, opens in [("A", [10, 10, 12]), ("B", [20, 20, 22]), ("C", [30, 30, 33])]
        for i, date in enumerate(dates)
    ]).sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    industry = pd.DataFrame({"trade_date": dates, "industry_code": "I1", "industry_name": "One",
                             "raw_score": 1., "neutral_score": 1., "rotation_rank": 1.,
                             "top2_flag": True, "top5_flag": True, "top10_flag": True, "bottom5_flag": False})
    mapped = IndustrySliceMapper().map(stocks, industry)
    targets = IndustryResidualReturnBuilder().build(mapped, [1])
    first = mapped["trade_date"].eq(dates[0])
    residual = targets["stock_minus_sw_l1_return"][1][first].round(6).tolist()
    # A earns 20%; its leave-one-out peers earn 10%, while B/C earn 10% against 15% peers.
    assert residual == [.1, -.05, -.05]
    assert mapped.loc[mapped["sw_l1_rotation_top5_flag"], "sw_l1_industry_code"].eq("I1").all()
