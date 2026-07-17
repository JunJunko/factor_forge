from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from factor_forge.ml.bullish_divergence_conditional_ml import (
    EvaluationConfig,
    ModelConfig,
    TestPeriod,
    aggregate_importance,
    build_daily_evaluation,
    compare_paired_placebo_arms,
    feature_columns_by_block,
    run_oof_predictions,
    summarize_evaluation,
    summarize_paired_placebos,
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

VARIANTS = ("FLAG", "LEG", "DELTA", "SCORE", "FULL")
BASE_CONTEXTS: dict[str, tuple[str, ...]] = {
    "X": ("X",),
    "DT": ("X", "D", "T"),
    "CTX": ("X", "D", "T", "R", "C", "M"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Walk-forward OOF sensitivity for flag, leg, delta, score, and full "
            "consecutive-divergence structure blocks with paired placebo deltas."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/bullish_divergence_structure_variants"),
    )
    parser.add_argument(
        "--scope",
        choices=("long_industry_context", "pit_concept_context", "both"),
        default="both",
    )
    parser.add_argument("--long-placebo-repeats", type=int, default=5)
    parser.add_argument("--pit-placebo-repeats", type=int, default=20)
    parser.add_argument(
        "--placebo-algorithms",
        nargs="+",
        choices=("ridge", "lightgbm"),
        default=("ridge",),
    )
    parser.add_argument(
        "--placebo-arms",
        nargs="+",
        default=None,
        help="Optional subset of custom arm names to fit under shuffled labels.",
    )
    parser.add_argument("--skip-placebo", action="store_true")
    return parser.parse_args()


def structure_variant_blocks(columns: Sequence[str]) -> dict[str, list[str]]:
    structure = sorted(column for column in columns if column.startswith("structure__"))
    flags = {
        "structure__triple_history_available",
        "structure__double_divergence_present",
    }
    leg = flags | {
        column
        for column in structure
        if (
            column.startswith("structure__first_")
            or column.startswith("structure__second_")
        )
        and not column.endswith("_divergence_present")
    }
    delta = flags | {
        column
        for column in structure
        if column.endswith("_trend") or column == "structure__trend_positive_count"
    }
    score = flags | {"structure__double_divergence_trend_score"}
    blocks = {
        "S_FLAG": sorted(flags & set(structure)),
        "S_LEG": sorted(leg & set(structure)),
        "S_DELTA": sorted(delta & set(structure)),
        "S_SCORE": sorted(score & set(structure)),
        "S_FULL": structure,
    }
    empty = [name for name, fields in blocks.items() if not fields]
    if empty:
        raise ValueError(f"Empty structure variant blocks: {empty}")
    return blocks


def variant_arm_blocks() -> tuple[
    dict[str, tuple[str, ...]], list[tuple[str, str]]
]:
    arms: dict[str, tuple[str, ...]] = {}
    pairs: list[tuple[str, str]] = []
    for base_name, base_blocks in BASE_CONTEXTS.items():
        baseline = f"{base_name}_BASE"
        arms[baseline] = base_blocks
        for variant in VARIANTS:
            challenger = f"{base_name}_{variant}"
            arms[challenger] = (*base_blocks, f"S_{variant}")
            pairs.append((challenger, baseline))
    return arms, pairs


def _persist_scope(
    output: Path,
    *,
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    folds: pd.DataFrame,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    fold_summary: pd.DataFrame,
    paired_comparisons: pd.DataFrame,
    paired_summary: pd.DataFrame,
    feature_blocks: Mapping[str, Sequence[str]],
    arm_blocks: Mapping[str, Sequence[str]],
    model_config: ModelConfig,
    evaluation_config: EvaluationConfig,
    metadata: Mapping[str, object],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output / "conditional_event_dataset.parquet", index=False)
    predictions.to_parquet(output / "oof_predictions.parquet", index=False)
    importance.to_parquet(output / "feature_importance_by_fold.parquet", index=False)
    aggregate_importance(importance).to_parquet(
        output / "feature_importance_aggregate.parquet", index=False
    )
    folds.to_csv(output / "walk_forward_folds.csv", index=False)
    daily.to_parquet(output / "daily_oof_evaluation.parquet", index=False)
    summary.to_csv(output / "oof_summary.csv", index=False)
    fold_summary.to_csv(output / "oof_fold_summary.csv", index=False)
    paired_comparisons.to_csv(output / "paired_incremental_comparisons.csv", index=False)
    paired_summary.to_csv(output / "paired_incremental_placebo_summary.csv", index=False)
    manifest = {
        "metadata": dict(metadata),
        "model_config": asdict(model_config),
        "evaluation_config": asdict(evaluation_config),
        "arm_blocks": {name: list(blocks) for name, blocks in arm_blocks.items()},
        "feature_blocks": {name: list(fields) for name, fields in feature_blocks.items()},
        "row_count": int(len(dataset)),
        "prediction_rows": int(len(predictions)),
        "daily_evaluation_rows": int(len(daily)),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_scope(
    dataset: pd.DataFrame,
    *,
    scope: str,
    periods: tuple[TestPeriod, ...],
    output: Path,
    placebo_repeats: int,
    run_placebo: bool,
    placebo_algorithms: Sequence[str],
    placebo_arms: Sequence[str] | None,
    metadata: Mapping[str, object],
) -> None:
    evaluation_config = EvaluationConfig()
    # Attribution-safe tree ablation: all arms see every feature, and the OOF
    # engine uses the same fold seed for baseline and challenger.
    model_config = ModelConfig(
        placebo_repeats=placebo_repeats,
        lgb_feature_fraction=1.0,
    )
    feature_blocks = feature_columns_by_block(dataset)
    feature_blocks.update(structure_variant_blocks(dataset.columns))
    arm_blocks, pairs = variant_arm_blocks()
    arms = tuple(arm_blocks)
    active_placebo_arms = tuple(placebo_arms or arms)
    unknown_placebo_arms = sorted(set(active_placebo_arms) - set(arms))
    if unknown_placebo_arms:
        raise ValueError(f"Unknown placebo arms: {unknown_placebo_arms}")

    predictions, importance, folds = run_oof_predictions(
        dataset,
        scope=scope,
        periods=periods,
        arms=arms,
        algorithms=("ridge", "lightgbm"),
        model_config=model_config,
        run_shuffle_placebo=False,
        feature_blocks=feature_blocks,
        arm_blocks=arm_blocks,
    )
    main_daily = build_daily_evaluation(predictions, config=evaluation_config)
    daily_parts = [main_daily]

    if run_placebo and placebo_repeats > 0:
        placebo_predictions, _, _ = run_oof_predictions(
            dataset,
            scope=scope,
            periods=periods,
            arms=arms,
            algorithms=tuple(placebo_algorithms),
            model_config=model_config,
            run_shuffle_placebo=True,
            placebo_arms=active_placebo_arms,
            feature_blocks=feature_blocks,
            arm_blocks=arm_blocks,
        )
        placebo_predictions = placebo_predictions.loc[
            placebo_predictions["arm"].str.contains("_shuffle_", regex=False)
        ]
        daily_parts.append(
            build_daily_evaluation(placebo_predictions, config=evaluation_config)
        )

    daily = pd.concat(daily_parts, ignore_index=True)
    summary, fold_summary = summarize_evaluation(daily, config=evaluation_config)
    paired = compare_paired_placebo_arms(daily, pairs, config=evaluation_config)
    paired_summary = summarize_paired_placebos(paired)
    _persist_scope(
        output,
        dataset=dataset,
        predictions=predictions,
        importance=importance,
        folds=folds,
        daily=daily,
        summary=summary,
        fold_summary=fold_summary,
        paired_comparisons=paired,
        paired_summary=paired_summary,
        feature_blocks=feature_blocks,
        arm_blocks=arm_blocks,
        model_config=model_config,
        evaluation_config=evaluation_config,
        metadata=metadata,
    )

    actual = paired.loc[~paired["is_placebo"]].copy()
    primary = actual.loc[
        actual["metric"].isin(["rank_ic", "top_minus_all"])
    ].sort_values(["algorithm", "baseline", "metric", "mean_delta"], ascending=False)
    print(f"\n[{scope}] rows={len(dataset):,} main_predictions={len(predictions):,}")
    print(primary.to_string(index=False))
    print(f"saved={output}")


def main() -> None:
    args = parse_args()
    created = datetime.now(timezone.utc)
    run_root = args.output_root / created.strftime("structure_variants_%Y%m%dT%H%M%SZ")
    run_root.mkdir(parents=True, exist_ok=False)
    full = pd.read_parquet(args.dataset)
    full["trade_date"] = pd.to_datetime(full["trade_date"]).dt.normalize()
    full["label_available_date"] = pd.to_datetime(
        full["label_available_date"]
    ).dt.normalize()
    variant_blocks = structure_variant_blocks(full.columns)
    arm_blocks, pairs = variant_arm_blocks()
    metadata = {
        "generated_at_utc": created.isoformat(),
        "source_dataset": str(args.dataset),
        "hypothesis": (
            "Separate risk flag, leg levels, leg deltas, composite score, and full "
            "structure to locate incremental consecutive-divergence information."
        ),
        "paired_placebo": (
            "challenger_shuffle_k minus baseline_shuffle_k under the same within-date "
            "training-label permutation"
        ),
        "placebo_algorithms": list(args.placebo_algorithms),
        "placebo_arms": args.placebo_arms or "all custom arms",
        "variant_blocks": variant_blocks,
        "arm_blocks": {name: list(blocks) for name, blocks in arm_blocks.items()},
        "paired_arms": pairs,
        "label": "T+1 open to T+11 open industry excess return",
        "purge": "train label_available_date strictly before test_start",
        "primary_evaluation": "Top10, 40 bps total round-trip event cost",
        "tree_attribution": (
            "LightGBM feature_fraction=1.0 and identical fold seed across arms"
        ),
    }
    (run_root / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.scope in {"long_industry_context", "both"}:
        long = full.loc[
            full["trade_date"].between("2021-01-01", "2026-07-10")
        ].copy()
        long = long.drop(columns=[
            column
            for column in long.columns
            if column.startswith("concept__")
            or (column.startswith("interaction__") and "concept" in column)
        ])
        run_scope(
            long,
            scope="long_industry_context",
            periods=LONG_PERIODS,
            output=run_root / "long_industry_context",
            placebo_repeats=args.long_placebo_repeats,
            run_placebo=not args.skip_placebo,
            placebo_algorithms=args.placebo_algorithms,
            placebo_arms=args.placebo_arms,
            metadata=metadata,
        )

    if args.scope in {"pit_concept_context", "both"}:
        if "concept__snapshot_available" not in full.columns:
            raise ValueError("PIT scope requires concept__snapshot_available")
        pit = full.loc[
            full["trade_date"].between("2024-12-30", "2026-07-10")
            & full["concept__snapshot_available"].eq(1.0)
        ].copy()
        run_scope(
            pit,
            scope="pit_concept_context",
            periods=PIT_PERIODS,
            output=run_root / "pit_concept_context",
            placebo_repeats=args.pit_placebo_repeats,
            run_placebo=not args.skip_placebo,
            placebo_algorithms=args.placebo_algorithms,
            placebo_arms=args.placebo_arms,
            metadata=metadata,
        )


if __name__ == "__main__":
    main()
