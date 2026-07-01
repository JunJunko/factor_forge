from __future__ import annotations

import numpy as np
import pandas as pd


def cs_zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    return (values - values.mean()) / std if pd.notna(std) and std > 0 else pd.Series(np.nan, index=values.index)


class IndustryNeutralizer:
    controls = ["relative_strength_level", "industry_excess_return_5d", "log_industry_amount_share"]

    def transform(self, panel: pd.DataFrame, alpha: float = 1.0) -> pd.DataFrame:
        rows = []
        for _, daily in panel.groupby("trade_date", sort=True):
            daily = daily.copy()
            usable = daily[["raw_score", *self.controls]].replace([np.inf, -np.inf], np.nan).dropna()
            daily["neutral_score_raw"] = np.nan
            if len(usable) >= len(self.controls) + 2:
                x = usable[self.controls].to_numpy(float)
                x = np.column_stack([np.ones(len(x)), x])
                y = usable["raw_score"].to_numpy(float)
                penalty = np.eye(x.shape[1]) * alpha
                penalty[0, 0] = 0
                beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
                daily.loc[usable.index, "neutral_score_raw"] = y - x @ beta
            daily["neutral_score"] = cs_zscore(daily["neutral_score_raw"])
            rows.append(daily)
        return pd.concat(rows).sort_values(["trade_date", "industry_code"]).reset_index(drop=True)
