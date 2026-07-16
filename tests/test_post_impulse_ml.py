from __future__ import annotations

import numpy as np
import pandas as pd

from factor_forge.ml.config import Segment, Segments
from factor_forge.ml.post_impulse_config import (
    PostImpulseFeatureConfig,
    PostImpulseLabelConfig,
    load_post_impulse_ml_config,
)
from factor_forge.ml.post_impulse_dataset import (
    aggregate_event_path,
    assign_purged_splits,
    build_post_impulse_dataset,
)
from factor_forge.ml.post_impulse_m3 import (
    ABSORPTION_GROUPS,
    VARIANT_GROUPS,
    load_post_impulse_m3_config,
)
from factor_forge.ml.post_impulse_m3_walkforward import (
    EXPECTED_SIGNS,
    MINIMAL_M3_FEATURES,
    _purged_train_mask,
    load_post_impulse_m3_walkforward_config,
)
from factor_forge.ml.post_impulse_m2_walkforward import (
    _feature_sets,
    _pair_closed_trades,
    load_post_impulse_m2_walkforward_config,
)
from factor_forge.ml.post_impulse_m21 import (
    COMPRESSED_FEATURES,
    build_compressed_pressure_features,
    load_post_impulse_m21_config,
)
from factor_forge.ml.post_impulse_path import (
    _break_even_cost_bps,
    _net_return,
    _newey_west_mean_test,
    _top_selections,
    load_post_impulse_m2_path_config,
)


def test_repository_post_impulse_config_uses_pressure_scope_and_fixed_ablation():
    cfg = load_post_impulse_ml_config("configs/ml/post_impulse_path_ml_v1.yaml")
    assert cfg.training.sample_scope == "pressure_events"
    assert cfg.training.arms == ["m0", "m1", "m2", "m3", "m4", "m5"]
    assert cfg.labels.horizon == 10
    assert cfg.features.pressure_min_components == 2


def test_m3_diagnostic_is_fixed_to_mechanism_subblocks_without_interactions():
    cfg = load_post_impulse_m3_config("configs/ml/post_impulse_m3_diagnostic_v1.yaml")
    assert cfg.ridge_alpha == 1000.0
    assert list(VARIANT_GROUPS) == [
        "m2", "m3_impact", "m3_low", "m3_close", "m3_range",
        "m3_path_core", "m3_summary", "m3_full",
    ]
    flattened = {
        feature for group in ABSORPTION_GROUPS.values() for feature in group
    }
    assert all(feature.startswith("absorb__") for feature in flattened)
    assert not any(feature.startswith("interaction__") for feature in flattened)


def test_minimal_m3_walkforward_freezes_features_folds_and_expected_signs():
    cfg = load_post_impulse_m3_walkforward_config(
        "configs/ml/post_impulse_m3_minimal_walkforward_v1.yaml"
    )
    assert [fold.id for fold in cfg.folds] == [
        "fold_2023", "fold_2024", "fold_2025", "fold_2026h1"
    ]
    assert cfg.purge_trading_days == 11
    assert cfg.logistic_c == 1.0
    assert cfg.gate.minimum_positive_folds == 3
    assert MINIMAL_M3_FEATURES == [
        "absorb__impact_observed_days", "absorb__impact_slope",
        "absorb__low_slope_atr", "absorb__close_vwap_slope",
    ]
    assert [EXPECTED_SIGNS[name] for name in MINIMAL_M3_FEATURES] == [-1, -1, 1, 1]


def test_walkforward_purge_uses_trading_calendar_not_event_count():
    cfg = load_post_impulse_m3_walkforward_config(
        "configs/ml/post_impulse_m3_minimal_walkforward_v1.yaml"
    )
    calendar = pd.bdate_range("2022-11-01", "2023-01-31")
    # Sparse event dates must still purge eleven market trading days.
    event_dates = calendar[::3]
    events = pd.DataFrame({"signal_date": event_dates})
    train, audit = _purged_train_mask(
        events, cfg.folds[0], calendar, cfg.purge_trading_days
    )
    cutoff_ordinal = int(np.searchsorted(
        calendar.values, np.datetime64(cfg.folds[0].test_start), side="left"
    )) - cfg.purge_trading_days - 1
    expected_last = calendar[cutoff_ordinal]
    assert events.loc[train, "signal_date"].max() <= expected_last
    assert audit["purged_events"] > 0


def test_m1_m2_backtest_config_is_frozen_and_m2_only_adds_pressure():
    cfg = load_post_impulse_m2_walkforward_config(
        "configs/ml/post_impulse_m1_m2_oof_backtest_v1.yaml"
    )
    assert cfg.top_n == [5, 10, 20]
    assert cfg.cost_bps == [20.0, 40.0, 60.0]
    assert cfg.holding_days == 10
    frame = pd.DataFrame(columns=[
        "coord__size", "event__impulse", "pressure__level", "pressure__present",
        "absorb__low_slope_atr",
    ])
    features = _feature_sets(frame)
    assert features["m1"] == ["coord__size", "event__impulse"]
    assert features["m2"] == ["coord__size", "event__impulse", "pressure__level"]


def test_closed_trade_pairing_includes_both_costs_and_metadata():
    trades = pd.DataFrame([
        {
            "trade_date": "2025-01-02", "signal_date": "2025-01-01",
            "entry_date": "2025-01-02", "sleeve_id": 0, "ts_code": "A",
            "side": "BUY", "gross_value": 1000.0, "cost": 1.0,
        },
        {
            "trade_date": "2025-01-16", "signal_date": "2025-01-01",
            "entry_date": "2025-01-02", "sleeve_id": 0, "ts_code": "A",
            "side": "SELL", "gross_value": 1100.0, "cost": 1.1,
        },
    ])
    metadata = pd.DataFrame({
        "signal_date": [pd.Timestamp("2025-01-01")], "ts_code": ["A"],
        "industry_l1_code": ["I1"], "size_bucket": ["small"],
    })
    closed = _pair_closed_trades(trades, metadata)
    assert len(closed) == 1
    assert np.isclose(closed.iloc[0]["net_pnl"], 97.9)
    assert closed.iloc[0]["industry_l1_code"] == "I1"


def test_m21_config_freezes_top5_primary_and_corrected_cost_stress():
    cfg = load_post_impulse_m21_config(
        "configs/ml/post_impulse_m21_compressed_reranker_v1.yaml"
    )
    assert cfg.top_n == [5, 10]
    assert cfg.cost_bps == [20.0, 40.0, 60.0]
    assert cfg.gate.primary_top_n == 5
    assert cfg.gate.primary_cost_bps == 40.0
    assert cfg.gate.minimum_positive_delta_years == 3
    assert cfg.initial_cash == 100_000_000


def test_m21_compression_separates_shock_persistence_and_resolution():
    events = pd.DataFrame({
        "signal_date": pd.to_datetime(["2025-01-02"] * 3),
        "pressure__close_below_vwap_level": [0.9, 0.6, 0.1],
        "pressure__high_to_close_level": [0.9, 0.6, 0.1],
        "pressure__down_turnover_level": [0.9, 0.6, 0.1],
        "pressure__component_count": [2.0, 4.0, 1.0],
        "pressure__active_days": [1.0, 3.0, 1.0],
        "pressure__slope": [-0.5, 0.5, 0.0],
    })
    compressed = build_compressed_pressure_features(events)
    assert compressed.loc[0, "m21__shock_intensity"] > compressed.loc[2, "m21__shock_intensity"]
    assert compressed.loc[1, "m21__pressure_persistence"] > compressed.loc[0, "m21__pressure_persistence"]
    assert compressed.loc[0, "m21__pressure_resolution"] > compressed.loc[1, "m21__pressure_resolution"]
    assert compressed[COMPRESSED_FEATURES].notna().all().all()


def test_m2_path_config_freezes_horizons_top5_and_primary_cost():
    cfg = load_post_impulse_m2_path_config(
        "configs/ml/post_impulse_m2_path_v1.yaml"
    )
    assert cfg.horizons == [1, 2, 3, 5, 7, 10]
    assert cfg.top_n == 5
    assert cfg.gate.primary_cost_bps == 40.0
    assert cfg.gate.require_adjacent_support is True


def test_m2_path_cost_and_top_selection_are_deterministic():
    gross = 0.01
    assert _net_return(gross, 40.0) < gross
    assert np.isclose(_net_return(gross, _break_even_cost_bps(gross)), 0.0)
    frame = pd.DataFrame({
        "variant": ["c0_event"] * 3,
        "trade_date": pd.to_datetime(["2025-01-02"] * 3),
        "factor_value": [0.1, 0.3, 0.2],
        "event_id": ["a", "b", "c"],
    })
    selected = _top_selections(frame, 2, pd.Timestamp("2025-01-02"))
    assert list(selected.sort_values("selection_rank")["event_id"]) == ["b", "c"]
    statistic, p_value = _newey_west_mean_test(pd.Series([0.01] * 20 + [0.02] * 20))
    assert statistic > 0
    assert 0 <= p_value <= 1


def _path(pressure: float, impact: list[float | None]) -> pd.DataFrame:
    rows = []
    for offset in [1, 2, 3]:
        rows.append({
            "event_id": "E1", "offset": offset,
            "__p_down_turnover_pct": pressure,
            "__p_upper_shadow_pct": pressure,
            "__p_close_below_vwap_pct": pressure,
            "__p_high_to_close_pct": pressure,
            "__pressure_daily": pressure,
            "turnover_rate": 3.0 - 0.2 * offset,
            "__downside_impact_pct": impact[offset - 1],
            "__low_from_event_atr": -1.0 + 0.4 * offset,
            "__close_location": 0.2 + 0.2 * offset,
            "__close_vwap_gap": -0.03 + 0.015 * offset,
            "__range_atr": 2.0 - 0.3 * offset,
        })
    return pd.DataFrame(rows)


def test_absorption_requires_pressure_and_never_treats_no_sell_as_perfect_absorption():
    cfg = PostImpulseFeatureConfig(
        history_window=10, min_history=5, pressure_threshold=0.6,
        pressure_min_components=2,
    )
    no_pressure = aggregate_event_path(_path(0.2, [None, None, None]), cfg).iloc[0]
    assert no_pressure["pressure__present"] == 0.0
    assert np.isnan(no_pressure["absorb__impact_resilience"])
    assert no_pressure["absorb__impact_missing"] == 1.0

    absorbed = aggregate_event_path(_path(0.8, [0.8, 0.5, 0.2]), cfg).iloc[0]
    assert absorbed["pressure__present"] == 1.0
    assert absorbed["pressure__component_count"] == 4
    assert absorbed["absorb__impact_slope"] < 0
    assert absorbed["absorb__low_slope_atr"] > 0
    assert absorbed["absorb__range_slope_atr"] < 0


def test_split_purges_horizon_plus_one_trading_days_before_boundaries():
    calendar = pd.bdate_range("2025-01-01", periods=40)
    events = pd.DataFrame({
        "event_id": [f"E{i}" for i in range(40)],
        "signal_date": calendar,
    })
    segments = Segments(
        train=Segment(start=str(calendar[0].date()), end=str(calendar[19].date())),
        valid=Segment(start=str(calendar[20].date()), end=str(calendar[29].date())),
        test=Segment(start=str(calendar[30].date()), end=str(calendar[39].date())),
    )
    split = assign_purged_splits(events, segments, calendar, horizon=3)
    assert list(split.loc[split["split"].eq("purged"), "signal_date"]) == list(
        calendar[16:20]
    ) + list(calendar[26:30])
    assert split.loc[split["split"].eq("train"), "signal_date"].max() == calendar[15]
    assert split.loc[split["split"].eq("valid"), "signal_date"].max() == calendar[25]


def _synthetic_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=40)
    rows = []
    for code_index, code in enumerate(["A", "B", "C", "D"]):
        price = 10.0 + code_index
        for index, date in enumerate(dates):
            drift = 1.0 + 0.001 * (1 + ((index + code_index) % 3))
            price *= drift
            if code == "A" and index == 18:
                price *= 1.18
            open_price = price * (1.0 - 0.002 * ((index + code_index) % 2))
            high = max(open_price, price) * 1.01
            low = min(open_price, price) * 0.99
            volume = 1_000_000.0 + 10_000 * code_index
            rows.append({
                "trade_date": date, "ts_code": code,
                "raw_open": open_price, "raw_high": high, "raw_low": low, "raw_close": price,
                "adj_open": open_price, "adj_high": high, "adj_low": low, "adj_close": price,
                "volume_shares": volume, "amount_cny": price * volume,
                "turnover_rate": 1.0 + 0.1 * ((index + code_index) % 4),
                "circ_mv_cny": 1e10 * (code_index + 1), "industry_l1_code": "I1",
                "is_liquid": True, "is_tradeable": True, "is_suspended": False,
                "is_st": False, "is_delisting_period": False, "listing_trade_days": 300 + index,
            })
    return pd.DataFrame(rows)


def test_future_prices_change_labels_but_not_signal_close_features():
    feature_cfg = PostImpulseFeatureConfig(
        history_window=10, min_history=5, atr_window=3, beta_window=10,
        impulse_percentile=0.9, industry_percentile=0.8,
        observation_days=3, event_cooldown_days=5, min_listing_days=1,
        market_window=5, industry_breadth_window=2,
    )
    label_cfg = PostImpulseLabelConfig()
    original = _synthetic_panel()
    changed = original.copy()
    future = changed["trade_date"].ge(pd.bdate_range("2024-01-02", periods=40)[23]) & changed["ts_code"].eq("A")
    for column in [
        "raw_open", "raw_high", "raw_low", "raw_close",
        "adj_open", "adj_high", "adj_low", "adj_close",
    ]:
        changed.loc[future, column] *= 1.4
    changed.loc[future, "amount_cny"] *= 1.4

    before = build_post_impulse_dataset(original, feature_cfg, label_cfg)
    after = build_post_impulse_dataset(changed, feature_cfg, label_cfg)
    event_date = pd.bdate_range("2024-01-02", periods=40)[18]
    left = before.events.loc[
        before.events["ts_code"].eq("A") & before.events["event_date"].eq(event_date)
    ].iloc[0]
    right = after.events.loc[
        after.events["ts_code"].eq("A") & after.events["event_date"].eq(event_date)
    ].iloc[0]
    feature_columns = [
        column for columns in before.feature_blocks.values() for column in columns
    ]
    np.testing.assert_allclose(
        left[feature_columns].to_numpy(float), right[feature_columns].to_numpy(float),
        equal_nan=True,
    )
    assert left["label__return_10d"] != right["label__return_10d"]
