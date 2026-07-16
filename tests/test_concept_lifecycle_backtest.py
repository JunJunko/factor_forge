import numpy as np
import pandas as pd

from factor_forge.research.concept_lifecycle_backtest import (
    attach_enhanced_stock_features,
    attach_lifecycle_fields,
    build_document_market_regimes,
    build_lifecycle_targets,
)


def test_enhanced_features_use_free_float_and_past_windows():
    dates = pd.bdate_range("2025-01-01", periods=30)
    panel = pd.DataFrame({
        "trade_date": dates, "ts_code": "000001.SZ", "raw_close": np.arange(10, 40.0),
        "adj_close": np.arange(10, 40.0), "amount_cny": 100_000_000.0,
        "circ_mv_cny": 2_000_000_000.0, "is_tradeable": True,
    })
    basic = pd.DataFrame({
        "trade_date": dates, "ts_code": "000001.SZ", "free_share": 10_000.0,
        "turnover_rate_f": np.arange(30.0), "volume_ratio": 1.0,
    })
    result = attach_enhanced_stock_features(panel, basic)
    assert result.loc[0, "free_float_mv_cny"] == 10 * 10_000 * 10_000
    assert np.isclose(result.loc[20, "amount_ratio_5_20"], 1.0)
    assert result.loc[20, "stock_return_20d"] > 0


def test_market_regime_is_prefix_stable():
    dates = pd.bdate_range("2025-01-01", periods=150)
    rows = []
    for day, date in enumerate(dates):
        for stock in range(40):
            close = 10 + day * (0.01 + stock / 100_000)
            rows.append({
                "trade_date": date, "ts_code": f"{stock:06d}.SZ", "adj_close": close,
                "stock_return_1d": 0.001, "stock_return_20d": 0.02,
                "circ_mv_cny": 1e9 + stock * 1e8, "is_tradeable": True,
                "amount_ratio_5_20": 1.0,
            })
    panel = pd.DataFrame(rows)
    short = build_document_market_regimes(panel.loc[panel["trade_date"].le(dates[119])])
    full = build_document_market_regimes(panel)
    columns = ["trade_date", "market_breadth", "breadth_delta_5d", "market_return_20d", "regime"]
    pd.testing.assert_frame_equal(
        short[columns].reset_index(drop=True),
        full.loc[full["trade_date"].le(dates[119]), columns].reset_index(drop=True),
    )


def test_lifecycle_transitions_and_placebo_preserve_daily_distribution():
    dates = pd.bdate_range("2025-01-01", periods=4)
    rows = []
    quadrants = ["lagging", "improving", "leading", "weakening"]
    for concept in ["A", "B"]:
        for i, date in enumerate(dates):
            rows.append({
                "trade_date": date, "concept_code": concept,
                "rrg_quadrant": quadrants[i], "common_breadth_delta_smooth5": 0.1,
                "common_delta_rank": 0.2 + 0.6 * (concept == "B"),
            })
    result = attach_lifecycle_fields(pd.DataFrame(rows))
    a = result.loc[result["concept_code"].eq("A")].set_index("trade_date")
    assert a.loc[dates[1], "lifecycle"] == "new_improving"
    assert a.loc[dates[2], "lifecycle"] == "confirmed_leading"
    assert a.loc[dates[3], "lifecycle"] == "exit"
    for _, day in result.groupby("trade_date"):
        assert sorted(day["common_delta_rank"]) == sorted(day["breadth_placebo_rank"])


def test_target_builder_applies_repair_catchup_and_weight_cap():
    dates = pd.bdate_range("2025-01-01", periods=8)
    concepts = ["C1", "C2"]
    feature_rows = []
    member_rows = []
    stock_rows = []
    for date in dates:
        for concept in concepts:
            feature_rows.append({
                "trade_date": date, "concept_code": concept, "eligible_concept": True,
                "rrg_quadrant": "improving", "signal_rrg_only": 1.0,
                "breadth_float": 0.8, "common_delta_rank": 0.9,
                "breadth_placebo_rank": 0.9, "breadth_negative_2d": False,
                "lifecycle_weight": 0.5, "lifecycle": "persistent_improving",
                "concept_return_20d": 0.05,
            })
            for stock in range(4):
                code = f"{concept}{stock}"
                member_rows.append({"trade_date": date, "concept_code": concept, "ts_code": code})
                stock_rows.append({
                    "trade_date": date, "ts_code": code, "is_tradeable": True,
                    "amount_ma20": 100_000_000.0, "stock_return_20d": 0.01 + stock * 0.01,
                    "stock_return_5d": 0.01, "amount_ratio_5_20": 1.2,
                    "turnover_f_delta5": 0.5, "free_float_mv_cny": 1e9 + stock * 1e8,
                })
    regimes = pd.DataFrame({
        "trade_date": dates, "regime": "repair", "target_exposure": 1.0, "stock_mode": "catchup",
    })
    result = build_lifecycle_targets(
        pd.DataFrame(feature_rows), pd.DataFrame(member_rows), pd.DataFrame(stock_rows), regimes,
        variant="B_breadth_filter", start=dates[0], end=dates[-1], offset=0,
    )
    assert not result.targets.empty
    assert result.targets["stock_mode"].eq("catchup").all()
    assert result.targets["target_weight"].max() <= 0.10 + 1e-12
