import numpy as np
import pandas as pd
import pytest
import yaml

from factor_forge.ml.value_dataset import (
    CrossSectionGroups,
    FUNDAMENTAL_FIELDS,
    VALUE_FEATURES,
    ValueFeatureParameters,
    attach_point_in_time_fundamentals,
    build_value_dataset,
    _group_residualize,
)
from factor_forge.ml.value_regression import (
    ValueRegressionConfig,
    ValueRegressionRunner,
    _purge_tail,
)


def test_point_in_time_fundamentals_never_join_future_snapshot():
    panel = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "ts_code": "000001.SZ",
    })
    fundamentals = pd.DataFrame({
        "ts_code": ["000001.SZ", "000001.SZ"],
        "available_date": pd.to_datetime(["2024-01-03", "2024-01-04"]),
        **{field: [1.0, 2.0] for field in FUNDAMENTAL_FIELDS},
    })
    joined = attach_point_in_time_fundamentals(panel, fundamentals)
    assert pd.isna(joined.loc[0, "revenue_ttm"])
    assert joined.loc[1, "revenue_ttm"] == 1.0
    assert joined.loc[2, "revenue_ttm"] == 2.0
    assert (joined["available_date"].dropna() <= joined.loc[joined["available_date"].notna(), "trade_date"]).all()


def _synthetic_panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", periods=175)
    rows = []
    for stock in range(24):
        price = 10 + stock / 5
        industry = "801010" if stock < 12 else "801020"
        quality = (stock % 12 + 1) / 12
        for position, date in enumerate(dates):
            relative_cycle = 0.0015 * np.sin(position / 11 + stock / 4)
            price *= np.exp(0.0003 + relative_cycle + rng.normal(0, 0.006))
            revision = (position >= 80 + stock % 7) * (0.03 + stock * 0.0005)
            revenue = 1e9 * (1 + quality + revision)
            assets = 8e8 * (1 + quality / 2 + revision / 2)
            mv = 1e9 * (1 + quality) * price / 10
            rows.append({
                "trade_date": date, "ts_code": f"{stock:06d}.SZ",
                "industry_l1_code": industry, "adj_open": price * 0.999,
                "adj_close": price, "amount_cny": 5e7 * (1 + quality) * (1 + abs(relative_cycle) * 50),
                "turnover_rate": 0.8 + quality + abs(relative_cycle) * 20,
                "log_total_mv": np.log(mv), "log_circ_mv": np.log(mv * 0.8),
                "revenue_ttm": revenue, "net_assets": assets,
                "roe_ttm": 0.05 + quality * 0.12 + revision,
                "revenue_growth_yoy": 0.03 + quality * 0.1 + revision,
                "roe_change_yoy": revision, "debt_to_assets": 0.65 - quality * 0.2,
                "net_profit_ttm": revenue * (0.02 + quality * 0.05),
                "is_tradeable": True, "is_liquid": True,
            })
    return pd.DataFrame(rows)


def test_value_features_use_non_overlapping_price_windows_and_multihorizon_labels():
    data, features, labels = build_value_dataset(
        _synthetic_panel(),
        horizons=[5, 10, 20],
        parameters=ValueFeatureParameters(
            delay_window=60, min_industry_size=10, ridge_alpha=1.0
        ),
    )
    assert features == VALUE_FEATURES
    assert labels == ["label_5d", "label_10d", "label_20d"]
    assert data[features].notna().any().all()
    assert data[labels].notna().any().all()
    assert np.isfinite(data[features].to_numpy(dtype=float)[data[features].notna().to_numpy()]).all()


def test_value_config_normalizes_blend_weights_and_purges_segment_tail():
    cfg = ValueRegressionConfig.model_validate({
        "segments": {
            "train": {"start": "2020-01-01", "end": "2021-12-31"},
            "valid": {"start": "2022-01-01", "end": "2022-12-31"},
            "test": {"start": "2023-01-01", "end": "2023-12-31"},
        },
        "labels": {"horizons": [5, 20], "blend_weights": {5: 1, 20: 3}},
    })
    assert cfg.labels.blend_weights == {5: 0.25, 20: 0.75}
    dates = pd.Series(pd.bdate_range("2024-01-01", periods=30))
    purged = _purge_tail(pd.Series(True, index=dates.index), dates, 20)
    assert purged.sum() == 10


def test_failed_run_persists_progress_log_and_traceback(tmp_path, monkeypatch):
    config = {
        "name": "logging_test",
        "segments": {
            "train": {"start": "2020-01-01", "end": "2021-12-31"},
            "valid": {"start": "2022-01-01", "end": "2022-12-31"},
            "test": {"start": "2023-01-01", "end": "2023-12-31"},
        },
        "output_root": str(tmp_path / "runs"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def fail(*args, **kwargs):
        raise RuntimeError("intentional training failure")

    monkeypatch.setattr(ValueRegressionRunner, "_execute", fail)
    with pytest.raises(RuntimeError, match="intentional training failure"):
        ValueRegressionRunner().run(path)
    run_dir = next((tmp_path / "runs").iterdir())
    log = (run_dir / "run.log").read_text(encoding="utf-8")
    error = (run_dir / "error.json").read_text(encoding="utf-8")
    assert "run_started" in log and "run_failed" in log
    assert "intentional training failure" in error
    assert "traceback" in error


def test_numpy_cross_section_residual_is_orthogonal_to_controls():
    dates = pd.Series(pd.to_datetime(["2024-01-02"] * 20 + ["2024-01-03"] * 20))
    industries = pd.Series(["801010"] * 40)
    x = pd.Series(np.tile(np.linspace(-2, 2, 20), 2))
    noise = pd.Series(np.tile(np.sin(np.arange(20)), 2))
    y = 3.0 + 2.5 * x + noise
    groups = CrossSectionGroups.build(dates, industries)
    residual, _ = _group_residualize(
        y, [x], dates, industries, min_size=10, groups=groups
    )
    for date in dates.unique():
        mask = dates.eq(date)
        assert abs(residual[mask].mean()) < 1e-12
        assert abs(residual[mask].corr(x[mask])) < 1e-12
