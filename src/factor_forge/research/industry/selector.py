from __future__ import annotations

import pandas as pd

from .neutralize import IndustryNeutralizer, cs_zscore


class IndustrySelector:
    def select(self, panel: pd.DataFrame, ridge_alpha: float = 1.0) -> pd.DataFrame:
        result = panel.copy()
        velocity_z = result.groupby("trade_date")["relative_strength_velocity"].transform(cs_zscore)
        breadth_z = result.groupby("trade_date")["breadth_velocity"].transform(cs_zscore)
        result["raw_score"] = .5 * velocity_z + .5 * breadth_z
        result = IndustryNeutralizer().transform(result, ridge_alpha)
        result["rotation_rank"] = result.groupby("trade_date")["neutral_score"].rank(method="first", ascending=False)
        count = result.groupby("trade_date")["neutral_score"].transform("count")
        for n in (2, 5, 10):
            result[f"top{n}_flag"] = result["neutral_score"].notna() & result["rotation_rank"].le(n)
        result["bottom5_flag"] = result["neutral_score"].notna() & result["rotation_rank"].gt((count - 5).clip(lower=0))
        # Never let undersized universes put the same industry in both tails.
        result["bottom5_flag"] &= ~result["top5_flag"]
        return result
