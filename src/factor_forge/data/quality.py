from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityIssue:
    rule_name: str
    severity: str
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


class DataQualityValidator:
    def validate(self, panel: pd.DataFrame) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        if panel.empty:
            issues.append(QualityIssue("nonempty_dataset", "BLOCKING", "Panel is empty"))
            return issues
        if panel.duplicated(["trade_date", "ts_code"]).any():
            issues.append(QualityIssue("unique_primary_key", "BLOCKING", "Duplicate primary keys"))
        priced = panel.dropna(subset=["raw_open", "raw_high", "raw_low", "raw_close"])
        invalid_ohlc = (
            (priced[["raw_open", "raw_high", "raw_low", "raw_close"]] <= 0).any(axis=1)
            | (priced["raw_high"] < priced[["raw_open", "raw_close", "raw_low"]].max(axis=1))
            | (priced["raw_low"] > priced[["raw_open", "raw_close", "raw_high"]].min(axis=1))
        )
        if invalid_ohlc.any():
            issues.append(QualityIssue("valid_ohlc", "BLOCKING", f"{int(invalid_ohlc.sum())} invalid rows"))
        invalid_adj = priced["adj_factor"].isna() | (priced["adj_factor"] <= 0)
        if invalid_adj.any():
            issues.append(QualityIssue("positive_adj_factor", "BLOCKING", f"{int(invalid_adj.sum())} invalid rows"))
        for field in ["volume_shares", "amount_cny"]:
            if field in panel and (panel[field].dropna() < 0).any():
                issues.append(QualityIssue("nonnegative_volume_and_amount", "BLOCKING", f"Negative {field}"))
        if "st_status_known" not in panel or not panel["st_status_known"].fillna(False).all():
            issues.append(QualityIssue("point_in_time_st_status", "BLOCKING", "ST status coverage is incomplete"))
        industry_fields = [
            field for field in ["industry_l1_code", "industry_l2_code"] if field in panel
        ]
        if industry_fields:
            industry_field = industry_fields[0]
            eligible = panel.get("is_factor_eligible", pd.Series(True, index=panel.index)).astype(bool)
            coverage = panel.loc[eligible, industry_field].notna().mean() if eligible.any() else 0.0
            if not np.isfinite(coverage) or coverage < 0.95:
                issues.append(QualityIssue(
                    "industry_history_coverage", "FEATURE_BLOCKING",
                    f"{industry_field} coverage={coverage:.2%}",
                ))
        for field in ["total_mv_cny", "circ_mv_cny"]:
            if field not in panel or panel[field].notna().mean() < 0.95:
                issues.append(QualityIssue("daily_basic_coverage", "FEATURE_BLOCKING", f"Insufficient {field}"))
                break
        return issues


def has_blocking_issues(issues: list[QualityIssue]) -> bool:
    return any(issue.severity == "BLOCKING" for issue in issues)
