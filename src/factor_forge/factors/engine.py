from __future__ import annotations

import numpy as np
import pandas as pd

from factor_forge.config import FactorSpec
from factor_forge.exceptions import ContractError
from .dsl import DSLContext, FIELD_ALIASES, FormulaEvaluator, infer_lookback
from .operators import standardize_zscore, winsorize_mad


class FactorEngine:
    def compute(self, panel: pd.DataFrame, spec: FactorSpec) -> pd.DataFrame:
        indexed = self._normalize_panel(panel)
        self._validate_fields(indexed, spec)
        calculation_mask = None
        if spec.scope.universe == "factor_eligible":
            calculation_mask = indexed["is_factor_eligible"].fillna(False).astype(bool)
        values = {
            name: parameter.value
            for name, parameter in spec.calculation.parameters.items()
        }
        allowed_fields = {FIELD_ALIASES.get(field, field) for field in spec.data.required_fields}
        if spec.scope.cross_section == "industry":
            allowed_fields.add(spec.scope.group_field)
            if spec.scope.group_field == "industry_l1_code":
                allowed_fields.add(FIELD_ALIASES["industry"])
        context = DSLContext(
            indexed, values, spec.scope.min_group_size, allowed_fields,
            cross_section_mask=calculation_mask,
        )
        evaluator = FormulaEvaluator(context)
        feature_lookbacks: dict[str, int] = {}
        for name, formula in spec.calculation.features.items():
            if name in indexed.columns or name in FIELD_ALIASES:
                raise ContractError(f"Feature shadows a standard field: {name}")
            feature_lookbacks[name] = infer_lookback(formula, values, feature_lookbacks)
            values[name] = evaluator.evaluate(formula)
        actual_lookback = infer_lookback(spec.calculation.formula, values, feature_lookbacks)
        if actual_lookback > spec.data.lookback_days:
            raise ContractError(
                f"Declared lookback_days={spec.data.lookback_days}, but formula requires {actual_lookback}"
            )
        factor = evaluator.evaluate(spec.calculation.formula)
        if spec.factor.direction == "negative":
            factor = -factor
        if calculation_mask is not None:
            factor = factor.where(calculation_mask)
        if spec.calculation.winsorize == "mad":
            factor = winsorize_mad(factor, spec.calculation.mad_scale)
        if spec.calculation.standardize == "zscore":
            factor = standardize_zscore(factor)
        if spec.scope.universe != "default":
            universe_field = f"is_{spec.scope.universe}"
            if universe_field not in indexed:
                raise ContractError(f"Panel is missing scope field: {universe_field}")
            factor = factor.where(indexed[universe_field].fillna(False).astype(bool))
        result = factor.rename("factor_value").reset_index()
        result["factor_valid"] = np.isfinite(result["factor_value"])
        result["invalid_reason"] = np.where(result["factor_valid"], None, "MISSING_INPUT_OR_LOOKBACK")
        if spec.scope.cross_section == "industry":
            result["group_code"] = indexed[spec.scope.group_field].to_numpy()
        return result

    @staticmethod
    def audit_temporal_consistency(
        panel: pd.DataFrame,
        baseline: pd.DataFrame,
        compute,
        max_cutoffs: int = 3,
    ) -> dict:
        """Recompute prefixes and prove that later rows cannot change earlier values."""
        dates = pd.Index(pd.to_datetime(panel["trade_date"]).unique()).sort_values()
        if len(dates) < 2:
            return {"checked_cutoffs": 0, "future_data_violations": 0}
        candidates = np.arange(max(1, len(dates) // 4), len(dates) - 1)
        indices = np.unique(np.linspace(0, len(candidates) - 1, min(max_cutoffs, len(candidates)), dtype=int))
        cutoffs = [dates[candidates[index]] for index in indices]
        violations = 0
        for cutoff in cutoffs:
            prefix = panel.loc[pd.to_datetime(panel["trade_date"]) <= cutoff].copy()
            recomputed = compute(prefix)
            expected = baseline.loc[pd.to_datetime(baseline["trade_date"]) == cutoff,
                                    ["trade_date", "ts_code", "factor_value"]]
            actual = recomputed.loc[pd.to_datetime(recomputed["trade_date"]) == cutoff,
                                    ["trade_date", "ts_code", "factor_value"]]
            compared = expected.merge(actual, on=["trade_date", "ts_code"], how="outer",
                                      suffixes=("_expected", "_actual"))
            equal = np.isclose(compared["factor_value_expected"], compared["factor_value_actual"],
                               rtol=1e-10, atol=1e-12, equal_nan=True)
            violations += int((~equal).sum())
        return {"checked_cutoffs": len(cutoffs), "future_data_violations": violations}

    @staticmethod
    def _normalize_panel(panel: pd.DataFrame) -> pd.DataFrame:
        required = {"trade_date", "ts_code"}
        if not required <= set(panel.columns):
            raise ContractError("Panel requires trade_date and ts_code")
        if panel.duplicated(["trade_date", "ts_code"]).any():
            raise ContractError("Panel primary key trade_date + ts_code is not unique")
        result = panel.copy()
        result["trade_date"] = pd.to_datetime(result["trade_date"])
        return result.sort_values(["trade_date", "ts_code"]).set_index(["trade_date", "ts_code"])

    @staticmethod
    def _validate_fields(panel: pd.DataFrame, spec: FactorSpec) -> None:
        missing = []
        for field in spec.data.required_fields:
            canonical = FIELD_ALIASES.get(field, field)
            if canonical not in panel.columns:
                missing.append(f"{field} ({canonical})")
        if missing:
            raise ContractError("Panel is missing factor fields: " + ", ".join(missing))
        if spec.scope.cross_section == "industry" and spec.scope.group_field not in panel.columns:
            raise ContractError(
                f"Panel is missing industry group field: {spec.scope.group_field}"
            )
        if spec.scope.universe == "factor_eligible" and "is_factor_eligible" not in panel.columns:
            raise ContractError("Panel is missing scope field: is_factor_eligible")
