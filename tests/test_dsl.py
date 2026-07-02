from __future__ import annotations

import numpy as np
import pytest

from factor_forge.config import FactorSpec, load_factor
from factor_forge.exceptions import DSLValidationError
from factor_forge.factors import FactorEngine
from factor_forge.factors.dsl import DSLContext, FormulaEvaluator


def factor_spec(formula: str, features: dict | None = None) -> FactorSpec:
    return FactorSpec.model_validate({
        "version": 1,
        "factor": {"name": "test_factor", "label": "测试", "description": "test",
                   "hypothesis": "test", "direction": "positive", "expected_shape": "monotonic"},
        "data": {"frequency": "daily", "required_fields": ["close"], "lookback_days": 2},
        "scope": {"universe": "default", "cross_section": "market", "min_group_size": 2},
        "calculation": {"features": features or {}, "formula": formula,
                        "missing_policy": "skip", "winsorize": "none", "standardize": "none"},
        "output": {"value_field": "factor_value"},
    })


def test_ret_has_exact_lag_semantics(panel):
    result = FactorEngine().compute(panel, factor_spec("ret(close, 2)"))
    stock = result[result.ts_code == "000000.SZ"].reset_index(drop=True)
    assert stock.factor_value.iloc[:2].isna().all()
    expected = panel.query("ts_code == '000000.SZ'").adj_close.iloc[2] / panel.query("ts_code == '000000.SZ'").adj_close.iloc[0] - 1
    assert stock.factor_value.iloc[2] == pytest.approx(expected)


def test_features_are_declarative_and_reusable(panel):
    spec = factor_spec("cs_rank(momentum)", {"momentum": "ret(close, 2)"})
    result = FactorEngine().compute(panel, spec)
    valid = result[result.trade_date == result.trade_date.max()].sort_values("factor_value")
    assert valid.factor_value.between(0, 1).all()
    assert valid.factor_value.is_monotonic_increasing


def test_arbitrary_python_is_rejected(panel):
    indexed = FactorEngine._normalize_panel(panel)
    evaluator = FormulaEvaluator(DSLContext(indexed, {}))
    with pytest.raises(DSLValidationError):
        evaluator.evaluate("close.__class__")
    with pytest.raises(DSLValidationError):
        evaluator.evaluate("__import__(1)")
    with pytest.raises(DSLValidationError):
        evaluator.evaluate("lag(close, -1)")


def test_declared_lookback_cannot_hide_formula_history(panel):
    spec = factor_spec("ret(close, 3)")
    spec.data.lookback_days = 2
    with pytest.raises(Exception, match="formula requires 3"):
        FactorEngine().compute(panel, spec)


def test_formula_cannot_read_an_undeclared_panel_field(panel):
    spec = factor_spec("ret(amount, 1)")
    with pytest.raises(DSLValidationError, match="not declared"):
        FactorEngine().compute(panel, spec)


def test_temporal_audit_recomputes_historical_prefixes(panel):
    spec = factor_spec("ret(close, 2)")
    baseline = FactorEngine().compute(panel, spec)
    audit = FactorEngine.audit_temporal_consistency(
        panel, baseline, lambda prefix: FactorEngine().compute(prefix, spec)
    )
    assert audit["checked_cutoffs"] > 0
    assert audit["future_data_violations"] == 0


def test_industry_rank_respects_group_boundaries(panel):
    spec = factor_spec("cs_rank(close, by=industry)")
    spec.scope.cross_section = "industry"
    spec.scope.min_group_size = 2
    result = FactorEngine().compute(panel, spec)
    latest = result[result.trade_date == result.trade_date.max()]
    assert latest.groupby("group_code").factor_value.min().eq(0.5).all()
    assert latest.groupby("group_code").factor_value.max().eq(1.0).all()


def test_l2_industry_rank_uses_configured_group_field(panel):
    panel = panel.copy()
    panel["industry_l2_code"] = panel["industry_l1_code"]
    spec = factor_spec("cs_rank(close, by=industry_l2_code)")
    spec.data.required_fields.append("industry_l2_code")
    spec.scope.cross_section = "industry"
    spec.scope.group_field = "industry_l2_code"
    spec.scope.industry_level = "L2"
    spec.scope.min_group_size = 2
    result = FactorEngine().compute(panel, spec)
    latest = result[result.trade_date == result.trade_date.max()]
    assert set(latest.group_code) == set(panel.industry_l2_code)
    assert latest.groupby("group_code").factor_value.min().eq(0.5).all()
    assert latest.groupby("group_code").factor_value.max().eq(1.0).all()


def test_sw_l2_rank_acceleration_config_computes(panel):
    panel = panel.copy()
    panel["industry_l2_code"] = panel["industry_l1_code"]
    spec = load_factor("configs/factors/sw_l2_industry_rank_acceleration.yaml")
    spec.scope.min_group_size = 2
    result = FactorEngine().compute(panel, spec)
    assert list(result.columns[:3]) == ["trade_date", "ts_code", "factor_value"]
    assert "group_code" in result
