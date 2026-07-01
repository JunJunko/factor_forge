from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv


ENDPOINTS = [
    "stock_basic", "trade_cal", "daily", "adj_factor", "daily_basic", "stk_limit",
    "suspend_d", "stock_st", "index_classify", "index_member_all", "index_daily",
]


class TushareProvider:
    def __init__(self, token: str | None = None):
        load_dotenv()
        token = token or os.getenv("TUSHARE_TOKEN")
        if not token:
            raise RuntimeError("TUSHARE_TOKEN is not set")
        try:
            import tushare as ts
        except ImportError as exc:
            raise RuntimeError("Install the 'tushare' optional dependency") from exc
        self.pro = ts.pro_api(token)

    def query(self, endpoint: str, **kwargs) -> pd.DataFrame:
        for attempt in range(6):
            try:
                method = getattr(self.pro, endpoint, None)
                if method is not None:
                    return method(**kwargs)
                return self.pro.query(endpoint, **kwargs)
            except requests.exceptions.RequestException:
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 16))
        raise RuntimeError("unreachable")

    def permission_report(self, sample_date: str = "20250102") -> list[dict]:
        probes = {
            "stock_basic": {"exchange": "", "list_status": "L"},
            "trade_cal": {"exchange": "SSE", "start_date": sample_date, "end_date": sample_date},
            "daily": {"trade_date": sample_date}, "adj_factor": {"trade_date": sample_date},
            "daily_basic": {"trade_date": sample_date}, "stk_limit": {"trade_date": sample_date},
            "suspend_d": {"trade_date": sample_date}, "stock_st": {"trade_date": sample_date},
            "index_classify": {"level": "L1", "src": "SW2021"},
            "index_member_all": {"l1_code": "801010.SI"},
            "index_daily": {"ts_code": "000852.SH", "start_date": sample_date, "end_date": sample_date},
        }
        checked = datetime.now(timezone.utc).isoformat()
        report = []
        for endpoint, arguments in probes.items():
            try:
                frame = self.query(endpoint, **arguments)
                event_endpoints = {"suspend_d", "stock_st"}
                status = "AVAILABLE" if (not frame.empty or endpoint in event_endpoints) else "INSUFFICIENT_HISTORY"
                detail = f"rows={len(frame)}"
            except Exception as exc:  # provider error messages carry the permission reason
                status, detail = "PERMISSION_DENIED", str(exc)
            report.append({"endpoint": endpoint, "status": status, "checked_at": checked, "detail": detail})
        return report
