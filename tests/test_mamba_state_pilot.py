from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from conftest import make_panel
from factor_forge.ml.mamba_state_config import (
    MambaStatePilotConfig,
    SequenceConfig,
    load_mamba_state_config,
)
from factor_forge.ml.mamba_state_dataset import (
    build_sequence_store,
    sequence_prefix_audit,
)
from factor_forge.ml.mamba_state_features import build_state_feature_frame
from factor_forge.ml.mamba_state_runner import MambaStateLightGBMRunner
from factor_forge.data import DataVersionRepository
from factor_forge.radar.scanner import RelationAnomalyScanner
from factor_forge.radar.templates import RADAR_TEMPLATE_ADAPTER


def _radar_template():
    return RADAR_TEMPLATE_ADAPTER.validate_python({
        "version": 1,
        "id": "price_drop_history_test",
        "kind": "price_drop_without_volume_confirmation",
        "description": "test",
        "observation_type": "relation_anomaly",
        "data": {
            "required_fields": ["adj_close", "volume_shares"],
            "universe_field": "is_liquid", "entity_field": "ts_code",
            "date_field": "trade_date", "industry_field": "industry_l1_code",
        },
        "scan": {"discovery_window_days": 40, "recent_window_days": 10},
        "parameters": {
            "return_horizon": 3,
            "return_history": {"window": 20, "min_periods": 8},
            "volume_history": {"window": 10, "min_periods": 5},
            "return_percentile_lte": 0.20, "volume_percentile_lte": 0.80,
        },
    })


def test_repository_pilot_config_freezes_exactly_eight_templates():
    config = load_mamba_state_config("configs/ml/mamba_state_lightgbm_pilot_v1.yaml")
    assert len(config.sequence.event_templates) == 8
    assert all(path.exists() for path in config.sequence.event_templates)
    assert config.encoder.backend == "torch_reference"


def test_sequence_config_rejects_invalid_contracts():
    with pytest.raises(ValidationError, match="min_valid_days"):
        SequenceConfig(length=10, min_valid_days=11, include_event_channels=False)
    with pytest.raises(ValidationError, match="exactly eight"):
        SequenceConfig(length=10, min_valid_days=5, include_event_channels=True)


def test_historical_event_channels_are_label_free_event_masked_and_prefix_invariant():
    panel = make_panel(days=90, stocks=5)
    panel["future_return_5"] = 999.0  # Must never escape through the whitelist.
    template = _radar_template()
    scanner = RelationAnomalyScanner()
    prefix_panel = panel.loc[panel["trade_date"].le(panel["trade_date"].sort_values().unique()[69])]
    prefix = scanner.measure_event_channels(prefix_panel, template)
    full = scanner.measure_event_channels(panel, template)
    cutoff = pd.Timestamp(prefix["trade_date"].max())
    full_prefix = full.loc[full["trade_date"].le(cutoff)].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix.reset_index(drop=True), full_prefix)
    assert not any("future" in column or "label" in column for column in full.columns)
    event_col = f"{template.id}__event"
    severity_col = f"{template.id}__severity"
    valid_col = f"{template.id}__valid"
    assert full.loc[full[event_col].eq(0) & full[valid_col].eq(1), severity_col].eq(0).all()

    scan = scanner.scan(prefix_panel, template, data_version="test")
    event_keys = set(map(tuple, prefix.loc[prefix[event_col].eq(1), ["trade_date", "ts_code"]].to_numpy()))
    scan_keys = set(map(tuple, scan.events[["trade_date", "ts_code"]].to_numpy()))
    assert scan_keys <= event_keys  # scan() intentionally keeps only its discovery window


def test_state_frame_and_sequence_store_are_causal_and_do_not_cross_stocks():
    panel = make_panel(days=30, stocks=3)
    config = load_mamba_state_config("configs/ml/mamba_state_lightgbm_pilot_v1.yaml")
    state = build_state_feature_frame(panel, config.features, config.label)
    assert state.event_channel_names == []
    store = build_sequence_store(
        state.frame, state.state_feature_names, length=10, min_valid_days=5,
        validity_feature_names=state.raw_feature_names,
    )
    assert len(store.samples) > 0
    values, mask = store.take([0, len(store.samples) - 1])
    assert values.shape == mask.shape == (2, 10, len(state.state_feature_names))
    for sample in store.samples.itertuples(index=False):
        window = store.frame.iloc[sample.start_row:sample.end_row + 1]
        assert len(window) == 10
        assert window["instrument"].nunique() == 1
        assert window["instrument"].iloc[0] == sample.instrument
        assert pd.Timestamp(window["datetime"].iloc[-1]) == pd.Timestamp(sample.datetime)

    extended_panel = make_panel(days=35, stocks=3)
    extended_state = build_state_feature_frame(extended_panel, config.features, config.label)
    extended_store = build_sequence_store(
        extended_state.frame, extended_state.state_feature_names, length=10, min_valid_days=5,
        validity_feature_names=extended_state.raw_feature_names,
    )
    # Future labels change maturity at the tail, but sequence inputs must remain invariant.
    assert sequence_prefix_audit(store, extended_store)


def test_state_frame_schema_is_stable_and_excludes_label_from_encoder_inputs():
    panel = make_panel(days=40, stocks=4)
    config = load_mamba_state_config("configs/ml/mamba_state_lightgbm_pilot_v1.yaml")
    first = build_state_feature_frame(panel, config.features, config.label)
    second = build_state_feature_frame(panel.sample(frac=1, random_state=7), config.features, config.label)
    assert first.feature_schema_hash == second.feature_schema_hash
    assert first.state_feature_names == second.state_feature_names
    assert "label" not in first.state_feature_names
    assert json.dumps(first.template_hashes) == "{}"


def test_reference_encoder_is_causal_and_checkpoint_round_trips(tmp_path):
    torch = pytest.importorskip("torch")
    from factor_forge.ml.mamba_state_model import ReferenceSelectiveStateEncoder

    torch.manual_seed(3)
    model = ReferenceSelectiveStateEncoder(
        4, d_model=8, d_state=5, layers=2, embedding_dim=3, dropout=0.0
    ).eval()
    values = torch.randn(2, 12, 4)
    mask = torch.ones_like(values)
    with torch.no_grad():
        prefix_embedding = model.encode(values[:, :8], mask[:, :8])
        hidden = model.hidden_sequence(values, mask)
        full_prefix_embedding = model.embedding_head(hidden[:, 7])
    torch.testing.assert_close(prefix_embedding, full_prefix_embedding)
    path = tmp_path / "encoder.pt"
    torch.save(model.state_dict(), path)
    restored = ReferenceSelectiveStateEncoder(
        4, d_model=8, d_state=5, layers=2, embedding_dim=3, dropout=0.0
    )
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(model.encode(values, mask), restored.encode(values, mask))


def test_tiny_mamba_state_lightgbm_pilot_runs_all_three_frozen_arms(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("lightgbm")
    panel = make_panel(days=120, stocks=12)
    stock_number = panel["ts_code"].str[:6].astype(int)
    day_number = panel.groupby("ts_code").cumcount()
    panel["amount_cny"] *= 1.0 + stock_number * 0.03 + np.sin(day_number / 7) * 0.05
    panel["turnover_rate"] = 0.5 + stock_number * 0.07 + (day_number % 11) * 0.01
    data_root, metadata = tmp_path / "data", tmp_path / "metadata.sqlite3"
    version = DataVersionRepository(data_root, metadata).publish(panel, source="mamba-test")
    project = tmp_path / "project.yaml"
    project.write_text(yaml.safe_dump({
        "project_name": "mamba_test", "timezone": "Asia/Shanghai",
        "paths": {
            "data_root": str(data_root), "metadata_db": str(metadata),
            "artifacts_root": str(tmp_path / "factor_runs"),
        },
    }), encoding="utf-8")
    dates = list(pd.bdate_range("2024-01-02", periods=120))
    config = tmp_path / "pilot.yaml"
    config.write_text(yaml.safe_dump({
        "version": 1, "name": "tiny_mamba_pilot", "project_config": str(project),
        "data_version": version, "require_full_segment_coverage": False,
        "segments": {
            "train": {"start": str(dates[15].date()), "end": str(dates[64].date())},
            "valid": {"start": str(dates[65].date()), "end": str(dates[84].date())},
            "test": {"start": str(dates[85].date()), "end": str(dates[110].date())},
        },
        "features": {"windows": [5, 10], "winsor_quantile": 0.01, "cross_sectional_zscore": True},
        "label": {"horizon": 3, "price": "adj_open", "excess_over_universe": True},
        "sequence": {"length": 10, "min_valid_days": 5, "include_event_channels": False},
        "encoder": {
            "backend": "torch_reference", "d_model": 8, "d_state": 5,
            "layers": 1, "embedding_dim": 3, "dropout": 0.0, "mask_probability": 0.2,
        },
        "training": {
            "epochs": 2, "batch_size": 64, "learning_rate": 0.003,
            "weight_decay": 0.0, "patience": 2, "validation_fraction": 0.2,
            "max_train_samples": None, "max_valid_samples": None,
            "random_seeds": [7], "device": "cpu", "num_workers": 0,
        },
        "lightgbm": {
            "objective": "regression", "learning_rate": 0.05, "num_leaves": 7,
            "max_depth": 3, "n_estimators": 20, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1,
            "random_state": 42, "n_jobs": 1,
        },
        "portfolio": {
            "universe": "liquid", "top_n": 2, "holding_days": 3,
            "initial_cash": 1_000_000, "lot_size": 100, "cost_bps": 15,
        },
        "output_root": str(tmp_path / "mamba_runs"),
    }, sort_keys=False), encoding="utf-8")

    result = MambaStateLightGBMRunner().run(config)
    assert set(result["variants"]) == {"raw", "state", "raw_state"}
    assert result["variants"]["raw"]["train_rows"] == result["variants"]["state"]["train_rows"]
    assert result["variants"]["state"]["test_rows"] == result["variants"]["raw_state"]["test_rows"]
    run_dir = Path(result["run_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((run_dir / "temporal_audit.json").read_text(encoding="utf-8"))
    assert manifest["contains_future_labels_in_event_channels"] is False
    assert audit["encoder_uses_future_return_labels"] is False
    assert (run_dir / "state_embeddings.parquet").exists()
    assert (run_dir / "model_comparison.csv").exists()
