from factor_forge.feature_registry import load_feature_registry


def test_downgraded_feature_registry_preserves_prohibited_uses():
    entries = load_feature_registry("configs/feature_registry/registry_v1.yaml")
    assert len(entries) == 1
    entry = entries[0]
    assert entry.id == "turnover_abs_return_industry_size_neutralized_v1"
    assert entry.lifecycle == "research_only"
    assert {"standalone_alpha", "trade_signal", "validation_substitute"} <= set(entry.prohibited_uses)
