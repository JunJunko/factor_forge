from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from factor_forge.backtest import build_condition_membership
from factor_forge.config import ExperimentSpec


def test_condition_membership_uses_daily_universe_quantiles():
    dates = pd.bdate_range("2024-01-02", periods=2)
    rows = [
        {"trade_date": date, "ts_code": f"{stock:06d}.SZ", "is_liquid": stock < 10}
        for date in dates for stock in range(12)
    ]
    panel = pd.DataFrame(rows)
    values = panel[["trade_date", "ts_code"]].copy()
    values["factor_value"] = values["ts_code"].str[:6].astype(float)
    result = build_condition_membership(
        panel, values, universe="liquid", quantile_groups=5,
        include_quantiles=[5], min_cross_section=10,
    )
    selected = result[result["selection_eligible"]]
    assert selected.groupby("trade_date").size().tolist() == [2, 2]
    assert set(selected["ts_code"]) == {"000008.SZ", "000009.SZ"}
    assert set(selected["condition_quantile"]) == {5}


def test_l2_condition_filter_requires_l1_conditional_ic():
    with pytest.raises(ValidationError, match="requires stage_l1.conditional_ic"):
        ExperimentSpec.model_validate({
            "name": "invalid_condition_filter",
            "stage_l2": {"condition_filter": {"enabled": True}},
        })
