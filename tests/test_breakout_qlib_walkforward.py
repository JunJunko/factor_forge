from factor_forge.ml.breakout_qlib_walkforward import (
    BreakoutQlibWalkForwardRunner,
    WalkForwardConfig,
)


def test_walkforward_segments_use_only_pre_test_history():
    config = WalkForwardConfig.model_validate(
        {
            "research_run": "artifacts/example",
            "first_history_year": 2021,
            "history_years": 3,
        }
    )
    runner = BreakoutQlibWalkForwardRunner()
    fold_2023 = runner._segments_for_year(config, 2023)
    assert fold_2023.train.start == "2021-01-01"
    assert fold_2023.train.end == "2021-12-31"
    assert fold_2023.valid.start == "2022-01-01"
    assert fold_2023.test.start == "2023-01-01"

    fold_2026 = runner._segments_for_year(config, 2026)
    assert fold_2026.train.start == "2023-01-01"
    assert fold_2026.train.end == "2024-12-31"
    assert fold_2026.valid.start == "2025-01-01"
    assert fold_2026.test.start == "2026-01-01"
