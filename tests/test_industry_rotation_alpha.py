import numpy as np
import pandas as pd

from factor_forge.research.industry_rotation_alpha import (
    attach_pit_membership,
    build_rotation_dataset,
    normalize_sw_l2_membership,
)


def test_attach_pit_membership_respects_out_date_and_new_interval():
    panel = pd.DataFrame({
        "trade_date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
        "ts_code": ["A", "A", "A"],
    })
    membership = pd.DataFrame({
        "ts_code": ["A", "A"], "l2_code": ["OLD", "NEW"],
        "l2_name": ["Old", "New"], "in_date": ["20190101", "20200106"],
        "out_date": ["20200103", None], "is_new": ["N", "Y"],
    })
    result = attach_pit_membership(panel, membership)
    assert result["industry_code"].tolist() == ["OLD", None, "NEW"]


def test_normalize_rejects_multiple_current_memberships():
    membership = pd.DataFrame({
        "ts_code": ["A", "A"], "l2_code": ["X", "Y"], "l2_name": ["X", "Y"],
        "in_date": ["20200101", "20210101"], "out_date": [None, None], "is_new": ["Y", "Y"],
    })
    try:
        normalize_sw_l2_membership(membership)
    except ValueError as exc:
        assert "multiple current" in str(exc)
    else:
        raise AssertionError("expected multiple-current-membership error")


def test_rotation_features_and_labels_use_open_t1():
    dates = pd.bdate_range("2020-01-01", periods=75)
    rows = []
    for industry, drift in [("I1", 0.003), ("I2", -0.001)]:
        for stock in range(8):
            code = f"{industry}_{stock}"
            close = 10 * np.cumprod(np.full(len(dates), 1 + drift + stock * 0.00001))
            for index, date in enumerate(dates):
                rows.append({
                    "trade_date": date, "ts_code": code, "industry_code": industry,
                    "industry_name": industry, "adj_close": close[index],
                    "adj_open": close[index] * 0.999, "circ_mv_cny": 1e9 + stock,
                    "amount_cny": 1e8, "is_tradeable": True, "is_liquid": True,
                })
    stocks, groups = build_rotation_dataset(pd.DataFrame(rows), horizons=[5])
    mature = groups.dropna(subset=["breadth_float", "forward_excess_5d"])
    assert len(stocks) == len(rows)
    assert mature["industry_code"].nunique() == 2
    assert mature.loc[mature["industry_code"].eq("I1"), "forward_excess_5d"].mean() > 0
