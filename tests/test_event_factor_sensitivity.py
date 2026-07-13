from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from factor_forge.ml.event_factor_basis import build_event_factor_basis
from factor_forge.ml.event_factor_sensitivity_config import (
    FACTOR_BASIS,
    load_event_factor_sensitivity_config,
)
from factor_forge.ml.event_factor_sensitivity_runner import _audit_oof, _build_blocks
from factor_forge.ml.event_mamba_model import event_mamba_from_config
from factor_forge.ml.event_factor_sensitivity_runner import EventFactorSensitivityRunner


def test_repository_event_factor_sensitivity_contract_is_frozen():
    cfg = load_event_factor_sensitivity_config(
        "configs/ml/event_rankers/event_factor_sensitivity_oof_v1.yaml"
    )
    assert cfg.event.factor_basis == FACTOR_BASIS
    assert cfg.primary_metric == "paired_daily_rank_ic_delta_e2_vs_e1"
    assert cfg.oof.evaluation_blocks == 3
    assert cfg.training.random_seeds == [17]


def _panel(periods=35):
    dates = pd.bdate_range("2025-01-02", periods=periods)
    rows = []
    for j, code in enumerate(["A", "B", "C", "D"]):
        for i, date in enumerate(dates):
            close = 10 + j + i * (0.02 + j * 0.005)
            rows.append({
                "trade_date": date, "ts_code": code, "adj_open": close - 0.03,
                "adj_high": close + 0.10, "adj_low": close - 0.12, "adj_close": close,
                "amount_cny": 1e7 * (1 + j + i / 100), "turnover_rate": 0.01 * (j + 1),
                "industry_l1_code": "I1" if j < 2 else "I2",
            })
    return pd.DataFrame(rows)


def test_named_factor_basis_is_point_in_time_prefix_invariant():
    panel = _panel(35)
    cutoff = pd.Timestamp("2025-02-14")
    prefix = panel.loc[panel["trade_date"].le(cutoff)]
    left = build_event_factor_basis(prefix)
    right = build_event_factor_basis(panel)
    right = right.loc[right["trade_date"].le(cutoff)].reset_index(drop=True)
    pd.testing.assert_frame_equal(left.reset_index(drop=True), right)


def test_event_mamba_outputs_named_beta_and_exact_gated_product():
    import torch

    cfg = load_event_factor_sensitivity_config(
        "configs/ml/event_rankers/event_factor_sensitivity_oof_v1.yaml"
    )
    model = event_mamba_from_config(10, len(FACTOR_BASIS), 7, cfg)
    values = torch.randn(4, 12, 10)
    mask = torch.ones_like(values)
    factors = torch.randn(4, len(FACTOR_BASIS))
    output = model(values, mask, torch.tensor([0, 1, 2, 3]), torch.ones(4), factors)
    assert output["beta"].shape == (4, len(FACTOR_BASIS))
    assert output["residual_embedding"].shape == (4, cfg.event_mamba.residual_embedding_dim)
    assert torch.allclose(output["gated"], output["beta"] * factors)
    assert torch.all(output["beta"].abs() <= 1)


def test_oof_blocks_have_purge_and_audit_rejects_encoder_overlap():
    cfg = load_event_factor_sensitivity_config(
        "configs/ml/event_rankers/event_factor_sensitivity_oof_v1.yaml"
    )
    dates = pd.bdate_range("2025-01-02", periods=252)
    blocks = _build_blocks(dates, cfg)
    assert len(blocks) == 6
    for block in blocks:
        assert dates.get_loc(block.valid_start) - dates.get_loc(block.train_end) == 7
        assert dates.get_loc(block.oof_start) - dates.get_loc(block.valid_end) == 7
    sample = pd.DataFrame({
        "episode_id": ["A", "B"], "oof_block": [0, 1],
        "encoder_valid_end": [blocks[0].valid_end, blocks[1].oof_start],
        "trade_date": [blocks[0].oof_start, blocks[1].oof_start],
    })
    audit = _audit_oof(sample, blocks)
    assert audit["encoder_date_violations"] == 1
    assert not audit["passed"]


def test_stacking_source_uses_only_prior_oof_blocks():
    source = inspect.getsource(EventFactorSensitivityRunner.run)
    assert 'oof["oof_block"].lt(block.block)' in source
    assert 'oof["label_available_date"].le(block.oof_start)' in source
