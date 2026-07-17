from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from factor_forge.ml.bullish_divergence_conditional_ml import (
    EvaluationConfig,
    ModelConfig,
    STRUCTURE_ARM_BLOCKS,
    TestPeriod,
    attach_structure_features,
    build_daily_evaluation,
    compare_arms,
    persist_experiment,
    run_oof_predictions,
    summarize_evaluation,
)


LONG_PERIODS = (
    TestPeriod(1, "2022-01-01", "2022-12-31"),
    TestPeriod(2, "2023-01-01", "2023-12-31"),
    TestPeriod(3, "2024-01-01", "2024-12-31"),
    TestPeriod(4, "2025-01-01", "2025-12-31"),
    TestPeriod(5, "2026-01-01", "2026-07-10"),
)

PIT_PERIODS = (
    TestPeriod(1, "2025-07-01", "2025-09-30"),
    TestPeriod(2, "2025-10-01", "2025-12-31"),
    TestPeriod(3, "2026-01-01", "2026-03-31"),
    TestPeriod(4, "2026-04-01", "2026-07-10"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="H0-H5 walk-forward OOF ablation for consecutive bullish-divergence trend."
    )
    parser.add_argument("--base-conditional-dataset", type=Path, required=True)
    parser.add_argument("--structure-episodes", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/bullish_divergence_structure_ablation"),
    )
    parser.add_argument(
        "--scope",
        choices=("long_industry_context", "pit_concept_context", "both"),
        default="both",
    )
    parser.add_argument("--long-placebo-repeats", type=int, default=5)
    parser.add_argument("--pit-placebo-repeats", type=int, default=20)
    parser.add_argument("--skip-placebo", action="store_true")
    return parser.parse_args()


def run_scope(
    dataset: pd.DataFrame,
    *,
    scope: str,
    periods: tuple[TestPeriod, ...],
    output_dir: Path,
    model_config: ModelConfig,
    run_placebo: bool,
    metadata: dict[str, object],
) -> None:
    predictions, importance, folds = run_oof_predictions(
        dataset,
        scope=scope,
        periods=periods,
        arms=tuple(STRUCTURE_ARM_BLOCKS),
        model_config=model_config,
        run_shuffle_placebo=run_placebo,
        placebo_arms=("H3", "H5"),
    )
    daily = build_daily_evaluation(predictions)
    summary, fold_summary = summarize_evaluation(daily)
    comparisons = compare_arms(daily)
    persist_experiment(
        output_dir,
        dataset=dataset,
        predictions=predictions,
        importance=importance,
        folds=folds,
        daily_evaluation=daily,
        summary=summary,
        fold_summary=fold_summary,
        comparisons=comparisons,
        model_config=model_config,
        evaluation_config=EvaluationConfig(),
        metadata=metadata,
    )
    primary = summary.loc[
        summary["top_n"].eq(10)
        & summary["cost_bps"].eq(40)
        & ~summary["arm"].str.contains("_shuffle_", regex=False)
    ].sort_values(["algorithm", "top_minus_all_mean"], ascending=[True, False])
    print(f"\n[{scope}] rows={len(dataset):,} oof_predictions={len(predictions):,}")
    print(
        primary.loc[:, [
            "algorithm",
            "arm",
            "rank_ic_mean",
            "top_net_mean",
            "top_minus_all_mean",
            "top_minus_bottom_mean",
        ]].to_string(index=False)
    )
    print(f"saved={output_dir}")


def main() -> None:
    args = parse_args()
    created = datetime.now(timezone.utc)
    run_root = args.output_root / created.strftime("structure_ablation_%Y%m%dT%H%M%SZ")
    run_root.mkdir(parents=True, exist_ok=False)

    base = pd.read_parquet(args.base_conditional_dataset)
    episodes = pd.read_parquet(args.structure_episodes)
    dataset = attach_structure_features(base, episodes)
    dataset.to_parquet(run_root / "full_structure_event_dataset.parquet", index=False)
    structure_columns = sorted(
        column for column in dataset.columns if column.startswith("structure__")
    )
    metadata = {
        "base_conditional_dataset": str(args.base_conditional_dataset),
        "structure_episodes": str(args.structure_episodes),
        "generated_at_utc": created.isoformat(),
        "hypothesis": "P-to-A and A-to-B consecutive divergence trend adds OOF ranking alpha",
        "structure_columns": structure_columns,
        "arm_blocks": {key: list(value) for key, value in STRUCTURE_ARM_BLOCKS.items()},
        "label": "T+1 open to T+11 open industry excess return",
        "purge": "train label_available_date strictly before test_start",
        "training_weight": "equal trading-date weight",
        "primary_evaluation": "Top10, 40 bps total round-trip event cost deducted once",
    }
    (run_root / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.scope in {"long_industry_context", "both"}:
        long_dataset = dataset.loc[
            dataset["trade_date"].between("2021-01-01", "2026-07-10")
        ].copy()
        long_dataset = long_dataset.drop(
            columns=[
                column
                for column in long_dataset.columns
                if column.startswith("concept__")
                or (column.startswith("interaction__") and "concept" in column)
            ]
        )
        run_scope(
            long_dataset,
            scope="long_industry_context",
            periods=LONG_PERIODS,
            output_dir=run_root / "long_industry_context",
            model_config=ModelConfig(placebo_repeats=args.long_placebo_repeats),
            run_placebo=not args.skip_placebo,
            metadata=metadata,
        )

    if args.scope in {"pit_concept_context", "both"}:
        if "concept__snapshot_available" not in dataset.columns:
            raise ValueError("PIT concept scope requires concept__snapshot_available")
        pit_dataset = dataset.loc[
            dataset["trade_date"].between("2024-12-30", "2026-07-10")
            & dataset["concept__snapshot_available"].eq(1.0)
        ].copy()
        run_scope(
            pit_dataset,
            scope="pit_concept_context",
            periods=PIT_PERIODS,
            output_dir=run_root / "pit_concept_context",
            model_config=ModelConfig(placebo_repeats=args.pit_placebo_repeats),
            run_placebo=not args.skip_placebo,
            metadata=metadata,
        )


if __name__ == "__main__":
    main()
