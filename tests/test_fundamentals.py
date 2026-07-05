import numpy as np
import pandas as pd

from factor_forge.data.fundamentals import (
    TushareFundamentalIngestor,
    build_pit_fundamentals,
    quarter_periods,
)


def _statement_rows(code: str, announcements: dict[str, str], values: dict[str, tuple]):
    income, balance = [], []
    for period, announcement in announcements.items():
        revenue, profit, assets, liabilities, equity = values[period]
        common = {
            "ts_code": code, "ann_date": announcement, "f_ann_date": announcement,
            "end_date": period, "report_type": "1", "update_flag": "1",
        }
        income.append({**common, "total_revenue": revenue, "revenue": revenue,
                       "n_income_attr_p": profit})
        balance.append({**common, "total_assets": assets, "total_liab": liabilities,
                        "total_hldr_eqy_exc_min_int": equity,
                        "total_hldr_eqy_inc_min_int": equity})
    return income, balance


def test_quarter_periods_stop_at_requested_end_date():
    periods = quarter_periods(2023, "2024-06-30")
    assert periods[0] == "20230331"
    assert periods[-1] == "20240630"
    assert "20240930" not in periods


def test_pit_builder_computes_ttm_and_uses_session_after_announcement():
    announcements = {
        "20230331": "20230420", "20231231": "20240320",
        "20240331": "20240419", "20241231": "20250320",
        "20250331": "20250418",
    }
    values = {
        "20230331": (30, 3, 100, 40, 60),
        "20231231": (140, 14, 120, 45, 75),
        "20240331": (40, 4, 125, 45, 80),
        "20241231": (180, 18, 150, 50, 100),
        "20250331": (55, 5.5, 160, 55, 105),
    }
    income, balance = _statement_rows("000001.SZ", announcements, values)
    sessions = pd.bdate_range("2023-01-01", "2025-05-01")
    result = build_pit_fundamentals(pd.DataFrame(income), pd.DataFrame(balance), sessions)
    q1_2025 = result.loc[result["report_period"].eq(pd.Timestamp("2025-03-31"))].iloc[-1]
    assert q1_2025["revenue_ttm"] == 55 + 180 - 40
    assert q1_2025["net_profit_ttm"] == 5.5 + 18 - 4
    assert np.isclose(q1_2025["debt_to_assets"], 55 / 160)
    assert q1_2025["available_date"] > q1_2025["source_event_date"]


def test_later_revision_creates_new_snapshot_without_rewriting_history():
    announcements = {
        "20231231": "20240320", "20241231": "20250320",
    }
    values = {
        "20231231": (100, 10, 100, 40, 60),
        "20241231": (120, 12, 120, 45, 75),
    }
    income, balance = _statement_rows("000001.SZ", announcements, values)
    revised = {**income[-1], "ann_date": "20250410", "f_ann_date": "20250410",
               "total_revenue": 125, "revenue": 125}
    income.append(revised)
    sessions = pd.bdate_range("2023-01-01", "2025-05-01")
    result = build_pit_fundamentals(pd.DataFrame(income), pd.DataFrame(balance), sessions)
    original = result.loc[result["source_event_date"].eq(pd.Timestamp("2025-03-20"))].iloc[-1]
    revision = result.loc[result["source_event_date"].eq(pd.Timestamp("2025-04-10"))].iloc[-1]
    assert original["revenue_ttm"] == 120
    assert revision["revenue_ttm"] == 125
    assert original["available_date"] < revision["available_date"]


def test_vip_fetch_paginates_below_endpoint_caps():
    class Provider:
        def __init__(self):
            self.offsets = []

        def query(self, endpoint, **kwargs):
            self.offsets.append(kwargs["offset"])
            size = 5000 if kwargs["offset"] == 0 else 123
            return pd.DataFrame({"ts_code": [f"{i:06d}.SZ" for i in range(size)]})

    provider = Provider()
    result = TushareFundamentalIngestor(provider)._fetch(
        "income_vip", "20241231", ["ts_code"]
    )
    assert len(result) == 5123
    assert provider.offsets == [0, 5000]
