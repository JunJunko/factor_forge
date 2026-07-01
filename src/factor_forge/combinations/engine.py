from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from factor_forge.config import (
    CombinationFilter, FactorCombinationSpec, FactorSpec, factor_source_kind,
    load_factor, load_factor_combination,
)
from factor_forge.exceptions import FactorCombinationError
from factor_forge.factors import FactorEngine
from .cache import AtomicFactorCache


@dataclass
class CombinationResult:
    factor_values: pd.DataFrame
    variants: dict[str, pd.DataFrame] = field(default_factory=dict)
    leave_one_out: dict[str, pd.DataFrame] = field(default_factory=dict)
    components: dict[str, pd.DataFrame] = field(default_factory=dict)
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)
    component_coverage: pd.DataFrame = field(default_factory=pd.DataFrame)
    normalization_issues: list[dict] = field(default_factory=list)
    cache_status: dict[str, str] = field(default_factory=dict)


class FactorCombinationEngine:
    """Generate factor values only; evaluation and backtesting remain external."""

    def __init__(self, cache_root: str | Path = ".factor_forge/cache/atomic"):
        self.cache = AtomicFactorCache(cache_root)

    def run(
        self, panel: pd.DataFrame, combination_path: str | Path, *, scope_mask: pd.Series | None = None,
        cache_context: dict | None = None, atomic_compute: Callable[[pd.DataFrame, FactorSpec], pd.DataFrame] | None = None,
    ) -> CombinationResult:
        path = Path(combination_path).resolve()
        spec = load_factor_combination(path)
        combo = spec.factor_combination
        compute = atomic_compute or FactorEngine().compute
        scope = self._scope(panel, scope_mask)
        context = self._cache_context(panel, cache_context)
        component_values, cache_status = {}, {}
        for component in combo.components:
            source_path = self._resolve_atomic(path, component.source.path, combo.id, component.id)
            frame, status = self._load_atomic(panel, source_path, context, compute, combo.id, component.id)
            component_values[component.id], cache_status[component.id] = frame, status
        filter_values = {}
        for item in combo.filters:
            source_path = self._resolve_atomic(path, item.source.path, combo.id, item.id)
            frame, status = self._load_atomic(panel, source_path, context, compute, combo.id, item.id)
            filter_values[item.id], cache_status[f"filter:{item.id}"] = frame, status
        issues: list[dict] = []
        main = self._combine(panel, scope, combo, component_values, filter_values, None, issues)
        variants = {
            variant.id: self._combine(panel, scope, combo, component_values, filter_values, variant, issues)
            for variant in combo.variants
        }
        leave_one_out = {}
        if len(combo.components) <= 8:
            from factor_forge.config import CombinationVariant
            all_ids = [item.id for item in combo.components]
            for removed in all_ids:
                selected = [item for item in all_ids if item != removed]
                leave_one_out[removed] = self._combine(
                    panel, scope, combo, component_values, filter_values,
                    CombinationVariant(id=f"without_{removed}", components=selected,
                                       filters=[item.id for item in combo.filters]), issues)
        coverage, component_coverage = self._coverage(main, component_values, scope)
        return CombinationResult(main, variants, leave_one_out, component_values, coverage, component_coverage, issues, cache_status)

    @staticmethod
    def _scope(panel: pd.DataFrame, scope_mask: pd.Series | None) -> pd.Series:
        if scope_mask is None:
            scope_mask = panel.get("is_factor_eligible", pd.Series(True, index=panel.index))
        scope = pd.Series(scope_mask, index=panel.index).fillna(False).astype(bool)
        if len(scope) != len(panel):
            raise FactorCombinationError("Experiment scope mask is not aligned with the panel")
        return scope

    @staticmethod
    def _cache_context(panel: pd.DataFrame, supplied: dict | None) -> dict:
        dates = pd.to_datetime(panel["trade_date"])
        return {"start_date": dates.min().date(), "end_date": dates.max().date(), **(supplied or {})}

    @staticmethod
    def _resolve_atomic(combo_path: Path, relative: Path, combo_id: str, component_id: str) -> Path:
        path = relative if relative.is_absolute() else combo_path.parent / relative
        path = path.resolve()
        if not path.exists():
            raise FactorCombinationError(
                f"combination={combo_id}; component={component_id}; yaml={path}; stage=resolve; "
                "referenced factor YAML does not exist; fix source.path"
            )
        kind = factor_source_kind(path)
        if kind == "factor_combination":
            raise FactorCombinationError(
                f"combination={combo_id}; component={component_id}; yaml={path}; stage=resolve; "
                "Nested factor combinations are not supported in V1; reference an atomic factor YAML"
            )
        return path

    def _load_atomic(self, panel, path, context, compute, combo_id, component_id):
        try:
            key, metadata = self.cache.key(path, context)
            cached = self.cache.load(key, metadata)
            if cached is not None:
                return cached, "hit"
            frame = compute(panel, load_factor(path)).copy()
            if "factor_valid" in frame and "valid_flag" not in frame:
                frame["valid_flag"] = frame["factor_valid"]
            AtomicFactorCache.validate(frame)
            expected_keys = panel[["trade_date", "ts_code"]].copy()
            actual_keys = frame[["trade_date", "ts_code"]].copy()
            expected_keys["trade_date"] = pd.to_datetime(expected_keys["trade_date"])
            actual_keys["trade_date"] = pd.to_datetime(actual_keys["trade_date"])
            if len(expected_keys.merge(actual_keys, on=["trade_date", "ts_code"], how="outer", indicator=True).query("_merge != 'both'")):
                raise FactorCombinationError("Atomic factor date/key range does not match the experiment panel")
            self.cache.save(key, metadata, frame)
            return frame, "miss"
        except Exception as exc:
            if isinstance(exc, FactorCombinationError) and f"combination={combo_id}" in str(exc):
                raise
            raise FactorCombinationError(
                f"combination={combo_id}; component={component_id}; yaml={path}; stage=atomic_factor; "
                f"{exc}; validate the atomic YAML, data fields, and cache"
            ) from exc

    def _combine(self, panel, scope, combo, component_values, filter_values, variant, issues):
        ids = variant.components if variant else [item.id for item in combo.components]
        filters = variant.filters if variant else [item.id for item in combo.filters]
        definitions = {item.id: item for item in combo.components}
        keys = panel[["trade_date", "ts_code"]].copy()
        keys["trade_date"] = pd.to_datetime(keys["trade_date"])
        values = pd.DataFrame(index=keys.index)
        for component_id in ids:
            frame = component_values[component_id][["trade_date", "ts_code", "factor_value"]].copy()
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
            aligned = keys.merge(frame, on=["trade_date", "ts_code"], how="left")["factor_value"]
            aligned = aligned.where(scope)
            normalized = self._preprocess(keys, aligned, combo.preprocessing, component_id, issues)
            values[component_id] = normalized * (-1 if definitions[component_id].direction == "negative" else 1)
        result = self._weighted(values, ids, definitions, combo.preprocessing.missing_value, combo.combination)
        output = keys.copy()
        output["factor_value"] = result.where(scope)
        output["valid_flag"] = np.isfinite(output["factor_value"])
        output["factor_valid"] = output["valid_flag"]
        output["invalid_reason"] = np.where(output["valid_flag"], None, "INSUFFICIENT_VALID_COMPONENTS")
        filter_defs = {item.id: item for item in combo.filters}
        for filter_id in filters:
            output = self._apply_filter(keys, scope, output, filter_values[filter_id], filter_defs[filter_id], issues)
        return output

    @staticmethod
    def _preprocess(keys, values, preprocessing, component_id, issues):
        work = pd.DataFrame({"trade_date": keys["trade_date"], "value": values})
        winsor = preprocessing.winsorize
        if winsor.enabled:
            work["value"] = work.groupby("trade_date")["value"].transform(
                lambda x: x.clip(x.quantile(winsor.lower), x.quantile(winsor.upper)) if x.notna().any() else x
            )
        method, minimum = preprocessing.normalization.method, preprocessing.normalization.min_samples
        def normalize(group):
            valid = group.dropna()
            if len(valid) < minimum:
                issues.append({"trade_date": str(group.name), "component_id": component_id, "reason": "INSUFFICIENT_NORMALIZATION_SAMPLE"})
                return pd.Series(np.nan, index=group.index)
            if method == "cs_zscore":
                std = valid.std(ddof=0)
                if not np.isfinite(std) or std == 0:
                    issues.append({"trade_date": str(group.name), "component_id": component_id, "reason": "ZERO_STANDARD_DEVIATION"})
                    return pd.Series(np.nan, index=group.index)
                return (group - valid.mean()) / std
            ranks = group.rank(method="average")
            if method == "cs_rank":
                return ranks
            return ranks / len(valid)
        return work.groupby("trade_date", group_keys=False)["value"].apply(normalize)

    @staticmethod
    def _weighted(values, ids, definitions, missing, method):
        valid = values[ids].notna()
        count = valid.sum(axis=1)
        if missing.method == "intersection":
            eligible = count == len(ids)
        elif missing.method == "require_minimum_components":
            eligible = count >= min(missing.minimum_valid_components, len(ids))
        else:
            eligible = pd.Series(True, index=values.index)
        filled = values[ids].fillna(missing.missing_score_after_normalization)
        if method.method == "equal_weight":
            denom = valid.sum(axis=1).replace(0, np.nan)
            score = (filled * valid).sum(axis=1) / denom
        else:
            weights = pd.Series({item: definitions[item].weight for item in ids})
            active = valid.mul(weights, axis=1)
            if method.normalize_weights:
                denom = active.abs().sum(axis=1).replace(0, np.nan)
                score = (filled * active).sum(axis=1) / denom
            else:
                score = (filled * active).sum(axis=1)
        return score.where(eligible)

    def _apply_filter(self, keys, scope, output, raw, definition: CombinationFilter, issues):
        frame = raw[["trade_date", "ts_code", "factor_value"]].copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        aligned = keys.merge(frame, on=["trade_date", "ts_code"], how="left")["factor_value"].where(scope)
        processed = self._preprocess(keys, aligned, definition.preprocessing, definition.id, issues)
        op = definition.condition.operator
        condition = {"gt": processed.gt, "gte": processed.ge, "lt": processed.lt, "lte": processed.le}[op](definition.condition.value)
        if definition.action.type == "exclude":
            output.loc[condition, "factor_value"] = np.nan
            output.loc[condition, "valid_flag"] = False
            output.loc[condition, "factor_valid"] = False
            output.loc[condition, "invalid_reason"] = f"FILTER_EXCLUDED:{definition.id}"
        else:
            output.loc[condition, "factor_value"] -= float(definition.action.value)
        return output

    @staticmethod
    def _coverage(main, components, scope):
        daily = main.loc[scope].groupby("trade_date")["valid_flag"].agg([("valid_stocks", "sum"), ("scope_stocks", "size")]).reset_index()
        daily["coverage"] = daily["valid_stocks"] / daily["scope_stocks"].replace(0, np.nan)
        rows = []
        scoped_keys = main.loc[scope, ["trade_date", "ts_code"]]
        for component_id, frame in components.items():
            aligned = scoped_keys.merge(frame[["trade_date", "ts_code", "factor_value"]], on=["trade_date", "ts_code"], how="left")
            valid = aligned["factor_value"].notna()
            rows.append({"component_id": component_id, "coverage": float(valid.mean()), "valid_rows": int(valid.sum()), "total_rows": int(len(aligned))})
        return daily, pd.DataFrame(rows)
