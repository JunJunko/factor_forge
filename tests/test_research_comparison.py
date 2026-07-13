from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from factor_forge.research.comparison import create_composition_comparison


def test_plan_driven_comparison_selects_exact_daily_ic_and_is_immutable(tmp_path: Path):
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        """plan_id: plan_test
composition_comparison:
  inputs:
    selector: {target: stock_minus_sw_l1_return, variant: raw, universe: liquid, horizon: 5}
  thresholds: {min_overlap_days: 2, min_retention_ratio: 0.5, max_p_value: 0.1}
""",
        encoding="utf-8",
    )
    dates = pd.bdate_range("2024-01-01", periods=3)
    market = pd.DataFrame({
        "trade_date": dates, "target": "stock_minus_sw_l1_return", "variant": "raw",
        "universe": "liquid", "horizon": 5, "rank_ic": [0.02, 0.021, 0.019],
    })
    industry = market.assign(rank_ic=[0.01, 0.011, 0.009])
    market_path = tmp_path / "market" / "l1_daily_rank_ic.parquet"
    industry_path = tmp_path / "industry" / "l1_daily_rank_ic.parquet"
    market_path.parent.mkdir()
    industry_path.parent.mkdir()
    market.to_parquet(market_path, index=False)
    industry.to_parquet(industry_path, index=False)

    result = create_composition_comparison(plan, market_path.parent, industry_path.parent, output_root=tmp_path / "out")

    artifact = Path(result["artifact_path"])
    assert (artifact / "comparison.json").is_file()
    assert (artifact / "aligned_daily_rank_ic.parquet").is_file()
    assert result["inputs"]["selector"]["horizon"] == 5
    with pytest.raises(FileExistsError, match="Immutable"):
        create_composition_comparison(plan, market_path.parent, industry_path.parent, output_root=tmp_path / "out")
