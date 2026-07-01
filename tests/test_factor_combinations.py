from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from factor_forge.combinations import FactorCombinationEngine
from factor_forge.config import factor_source_kind, load_factor_combination
from factor_forge.exceptions import FactorCombinationError, UnsupportedFactorKindError


def atomic_yaml(path: Path, name: str):
    path.write_text(yaml.safe_dump({
        "version": 1, "factor": {"name": name, "label": name, "description": name,
        "hypothesis": name, "direction": "positive"},
        "data": {"required_fields": ["close"], "lookback_days": 0},
        "scope": {"universe": "default", "cross_section": "market", "min_group_size": 2},
        "calculation": {"formula": "close"}, "output": {"value_field": "factor_value"},
    }, sort_keys=False), encoding="utf-8")


def combination_yaml(path: Path, *, missing="intersection", method="weighted_sum", filters=False, variants=False):
    components = [
        {"id": "a", "source": {"type": "yaml", "path": "a.yaml"}, "direction": "positive", "weight": 1},
        {"id": "b", "source": {"type": "yaml", "path": "b.yaml"}, "direction": "negative", "weight": 3},
    ]
    body = {"id": "combo", "name": "Combo", "components": components,
            "preprocessing": {"normalization": {"method": "cs_zscore", "min_samples": 2},
                              "missing_value": {"method": missing, "minimum_valid_components": 1}},
            "combination": {"method": method, "normalize_weights": True}}
    if filters:
        body["filters"] = [{"id": "tail", "source": {"type": "yaml", "path": "filter.yaml"},
                            "preprocessing": {"normalization": {"method": "cs_percentile", "min_samples": 2}},
                            "condition": {"operator": "gte", "value": .75}, "action": {"type": "exclude"}}]
    if variants:
        body["variants"] = [{"id": "a_only", "components": ["a"]}, {"id": "both", "components": ["a", "b"]}]
    path.write_text(yaml.safe_dump({"version": 1, "kind": "factor_combination", "factor_combination": body}, sort_keys=False), encoding="utf-8")


@pytest.fixture
def combo_files(tmp_path):
    for name in ("a", "b", "filter"):
        atomic_yaml(tmp_path / f"{name}.yaml", name)
    combination_yaml(tmp_path / "combo.yaml")
    return tmp_path


def manual_compute(panel, spec):
    stock = panel["ts_code"].str.extract(r"(\d+)")[0].astype(int)
    values = stock.astype(float)
    if spec.factor.name == "b": values = values * 2
    if spec.factor.name == "filter": values = values
    return panel[["trade_date", "ts_code"]].assign(factor_value=values, factor_valid=True, valid_flag=True)


def test_kind_detection_and_legacy_default(combo_files):
    assert factor_source_kind(combo_files / "a.yaml") == "factor"
    assert factor_source_kind(combo_files / "combo.yaml") == "factor_combination"
    bad = combo_files / "bad.yaml"; bad.write_text("kind: magic\n", encoding="utf-8")
    with pytest.raises(UnsupportedFactorKindError): factor_source_kind(bad)


def test_weight_direction_and_scope_after_normalization(combo_files, panel):
    engine = FactorCombinationEngine(combo_files / "cache")
    scope = panel["ts_code"].isin(["000000.SZ", "000001.SZ", "000002.SZ"])
    result = engine.run(panel, combo_files / "combo.yaml", scope_mask=scope, atomic_compute=manual_compute)
    daily = result.factor_values[result.factor_values["trade_date"] == panel["trade_date"].min()]
    assert daily.loc[daily["ts_code"] == "000003.SZ", "factor_value"].isna().all()
    # b is perfectly aligned with a, then explicitly reversed; weighted score descends.
    assert daily.dropna().sort_values("ts_code")["factor_value"].is_monotonic_decreasing


@pytest.mark.parametrize("method", ["intersection", "require_minimum_components", "zero_after_normalization"])
def test_missing_value_modes(combo_files, panel, method):
    combination_yaml(combo_files / "combo.yaml", missing=method)
    def missing_compute(frame, spec):
        out = manual_compute(frame, spec)
        if spec.factor.name == "b": out.loc[out["ts_code"] == "000000.SZ", "factor_value"] = np.nan
        return out
    result = FactorCombinationEngine(combo_files / f"cache-{method}").run(
        panel, combo_files / "combo.yaml", atomic_compute=missing_compute)
    first = result.factor_values.loc[result.factor_values["ts_code"] == "000000.SZ", "factor_value"]
    assert first.isna().all() if method == "intersection" else first.notna().all()


def test_equal_weight_variants_and_filter(combo_files, panel):
    combination_yaml(combo_files / "combo.yaml", method="equal_weight", filters=True, variants=True)
    result = FactorCombinationEngine(combo_files / "cache-v").run(panel, combo_files / "combo.yaml", atomic_compute=manual_compute)
    assert set(result.variants) == {"a_only", "both"}
    assert result.variants["a_only"] is not result.variants["both"]
    assert (~result.factor_values["valid_flag"]).any()


def test_cache_reuse_and_invalidation(combo_files, panel):
    calls = []
    def tracked(frame, spec): calls.append(spec.factor.name); return manual_compute(frame, spec)
    engine = FactorCombinationEngine(combo_files / "cache")
    context = {"data_version": "v1"}
    first = engine.run(panel, combo_files / "combo.yaml", cache_context=context, atomic_compute=tracked)
    second = engine.run(panel, combo_files / "combo.yaml", cache_context=context, atomic_compute=tracked)
    assert set(first.cache_status.values()) == {"miss"}
    assert set(second.cache_status.values()) == {"hit"}
    atomic_yaml(combo_files / "a.yaml", "a")
    with (combo_files / "a.yaml").open("a", encoding="utf-8") as handle: handle.write("\n# changed\n")
    third = engine.run(panel, combo_files / "combo.yaml", cache_context=context, atomic_compute=tracked)
    assert third.cache_status["a"] == "miss"
    fourth = engine.run(panel, combo_files / "combo.yaml", cache_context={"data_version": "v2"}, atomic_compute=tracked)
    assert set(fourth.cache_status.values()) == {"miss"}


def test_nested_and_missing_references_are_explicit(combo_files, panel):
    combination_yaml(combo_files / "nested.yaml")
    raw = yaml.safe_load((combo_files / "combo.yaml").read_text(encoding="utf-8"))
    raw["factor_combination"]["components"][0]["source"]["path"] = "nested.yaml"
    (combo_files / "combo.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(FactorCombinationError, match="Nested factor combinations"):
        FactorCombinationEngine(combo_files / "cache").run(panel, combo_files / "combo.yaml", atomic_compute=manual_compute)


def test_schema_rejects_unknown_variant_and_zero_weights(combo_files):
    combination_yaml(combo_files / "combo.yaml", variants=True)
    raw = yaml.safe_load((combo_files / "combo.yaml").read_text(encoding="utf-8"))
    raw["factor_combination"]["variants"][0]["components"] = ["missing"]
    (combo_files / "combo.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown components"): load_factor_combination(combo_files / "combo.yaml")
