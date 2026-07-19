import numpy as np
import pandas as pd
import pytest

from factor_forge.research.concept_etf_coordinated_r4 import (
    CoordinatedR4Rules,
    deserialize_weights,
    simulate_coordinated_r4,
)
from factor_forge.research.concept_etf_shadow import simulate_staggered_sleeves


def coordinated_panel(periods: int = 110) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=periods)
    codes = ["A", "B", "C", "D", "E", "F"]
    rows = []
    for code_index, code in enumerate(codes):
        price = 100.0
        for date_index, date in enumerate(dates):
            common = 0.001 * np.sin(date_index / 7)
            specific = 0.0001 * (code_index + 1) * np.cos(date_index / 5)
            price *= 1 + common + specific
            rows.append({
                "trade_date": date,
                "ts_code": code,
                "adj_open": price,
                "etf_return_1d": common + specific,
                "etf_momentum_60d": 0.10,
                "volatility_20d": 0.01 + code_index / 1000,
                "score_etf_momentum": 10 - code_index,
                "cluster": f"cluster_{code_index}",
                "mapping_pass": True,
                "eligible_concept": True,
                "match_type": "exact",
                "etf_name": code,
                "concept_name": f"concept_{code}",
            })
    return pd.DataFrame(rows)


def test_coordinated_base_reproduces_independent_r4():
    panel = coordinated_panel()
    kwargs = {"start": "2025-03-03", "end": "2025-06-03", "roundtrip_cost_bps": 20}
    expected, expected_sleeves, _ = simulate_staggered_sleeves(
        panel, "R4_rank_buffer", **kwargs,
    )
    actual, actual_sleeves, _, _ = simulate_coordinated_r4(
        panel, "R4_A_base", **kwargs,
    )
    pd.testing.assert_series_equal(
        actual["net_nav"], expected["net_nav"], check_names=False,
    )
    pd.testing.assert_series_equal(
        actual_sleeves.sort_values(["sleeve", "return_date"])["net_nav"].reset_index(drop=True),
        expected_sleeves.sort_values(["sleeve", "return_date"])["net_nav"].reset_index(drop=True),
        check_names=False,
    )


def test_r4b_enforces_frequency_and_aggregate_etf_cap_at_rebalance():
    panel = coordinated_panel()
    _, sleeves, _, audit = simulate_coordinated_r4(
        panel,
        "R4_B_concentration",
        start="2025-03-03",
        end="2025-06-03",
        rules=CoordinatedR4Rules(
            maximum_sleeves_per_etf=3,
            maximum_aggregate_etf_weight=0.20,
        ),
    )
    assert audit["maximum_sleeve_frequency"].le(3).all()
    assert audit["maximum_rebalanced_etf_weight"].le(0.2000001).all()
    latest = sleeves.sort_values("return_date").groupby("sleeve", observed=True).tail(1)
    frequency = {}
    for value in latest["target_weights"]:
        for code, weight in deserialize_weights(value).items():
            if code != "__CASH__" and weight > 0:
                frequency[code] = frequency.get(code, 0) + 1
    assert max(frequency.values()) <= 3


def test_r4c_enforces_cluster_cap_and_blocks_highly_correlated_new_pairs():
    panel = coordinated_panel()
    panel.loc[panel["ts_code"].isin(["A", "B"]), "cluster"] = "technology"
    _, _, _, audit = simulate_coordinated_r4(
        panel,
        "R4_C_correlation",
        start="2025-03-03",
        end="2025-06-03",
        rules=CoordinatedR4Rules(
            maximum_sleeves_per_etf=3,
            maximum_aggregate_etf_weight=0.20,
            maximum_aggregate_cluster_weight=0.30,
            maximum_pairwise_correlation=0.75,
        ),
    )
    assert audit["maximum_aggregate_cluster_weight"].le(0.3000001).all()
    observed = audit["maximum_entry_pairwise_correlation"].dropna()
    assert observed.le(0.7500001).all()


def test_invalid_coordinated_variant_is_rejected():
    with pytest.raises(ValueError, match="unknown coordinated R4 variant"):
        simulate_coordinated_r4(
            coordinated_panel(10), "R4_X", start="2025-01-01", end="2025-01-10",
        )
