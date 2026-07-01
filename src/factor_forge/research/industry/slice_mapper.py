from __future__ import annotations

import pandas as pd


class IndustrySliceMapper:
    def map(self, stocks: pd.DataFrame, industries: pd.DataFrame) -> pd.DataFrame:
        fields = ["trade_date", "industry_code", "industry_name", "raw_score", "neutral_score", "rotation_rank",
                  "top2_flag", "top5_flag", "top10_flag", "bottom5_flag"]
        mapped = stocks.merge(industries[fields], on=["trade_date", "industry_code"], how="left", suffixes=("", "_selected"), validate="many_to_one")
        rename = {"industry_code": "sw_l1_industry_code", "industry_name": "sw_l1_industry_name",
                  "raw_score": "sw_l1_rotation_raw_score", "neutral_score": "sw_l1_rotation_neutral_score",
                  "rotation_rank": "sw_l1_rotation_rank"}
        rename.update({f"{x}_flag": f"sw_l1_rotation_{x}_flag" for x in ("top2", "top5", "top10", "bottom5")})
        return mapped.rename(columns=rename)
