from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from factor_forge.ml.bullish_divergence_conditional_ml import ModelConfig, TestPeriod
from factor_forge.ml.bullish_divergence_target_alignment import (
    TARGET_OBJECTIVES,
    TargetAlignmentConfig,
    aggregate_target_importance,
    build_target_aligned_daily_evaluation,
    dt_score_feature_sets,
    run_target_alignment_oof,
    summarize_target_alignment,
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
        description=(
            "Target-aligned DT_SCORE walk-forward OOF using regression, LambdaRank, "
            "and within-date top-decile classification."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/bullish_divergence_target_alignment"),
    )
    parser.add_argument(
        "--scope",
        choices=("long_industry_context", "pit_concept_context", "both"),
        default="both",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=TARGET_OBJECTIVES,
        default=TARGET_OBJECTIVES,
    )
    parser.add_argument("--long-placebo-repeats", type=int, default=20)
    parser.add_argument("--pit-placebo-repeats", type=int, default=20)
    parser.add_argument("--skip-placebo", action="store_true")
    return parser.parse_args()


def persist_scope(
    output: Path,
    *,
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    folds: pd.DataFrame,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    paired_placebo: pd.DataFrame,
    model_config: ModelConfig,
    alignment_config: TargetAlignmentConfig,
    metadata: dict[str, object],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output / "conditional_event_dataset.parquet", index=False)
    predictions.loc[~predictions["is_placebo"]].to_parquet(
        output / "oof_predictions.parquet", index=False
    )
    importance.loc[~importance["is_placebo"]].to_parquet(
        output / "feature_importance_by_fold.parquet", index=False
    )
    aggregate_target_importance(importance).to_parquet(
        output / "feature_importance_aggregate.parquet", index=False
    )
    folds.to_csv(output / "walk_forward_folds.csv", index=False)
    daily.to_parquet(output / "daily_portfolio_evaluation.parquet", index=False)
    summary.to_csv(output / "portfolio_summary.csv", index=False)
    paired_placebo.to_csv(output / "paired_incremental_placebo_summary.csv", index=False)
    manifest = {
        "metadata": metadata,
        "model_config": asdict(model_config),
        "alignment_config": asdict(alignment_config),
        "feature_sets": dt_score_feature_sets(dataset),
        "row_count": int(len(dataset)),
        "main_prediction_rows": int((~predictions["is_placebo"]).sum()),
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
    placebo_repeats: int,
    objectives: tuple[str, ...],
    run_placebo: bool,
    output: Path,
    metadata: dict[str, object],
) -> None:
    model_config = ModelConfig(
        lgb_feature_fraction=1.0,
        lgb_min_child_samples=50,
    )
    alignment_config = TargetAlignmentConfig(placebo_repeats=placebo_repeats)
    predictions, importance, folds = run_target_alignment_oof(
        dataset,
        scope=scope,
        periods=periods,
        model_config=model_config,
        alignment_config=alignment_config,
        objectives=objectives,
        run_placebo=run_placebo,
    )
    daily = build_target_aligned_daily_evaluation(
        predictions, config=alignment_config
    )
    summary, paired_placebo = summarize_target_alignment(
        daily, config=alignment_config
    )
    persist_scope(
        output,
        dataset=dataset,
        predictions=predictions,
        importance=importance,
        folds=folds,
        daily=daily,
        summary=summary,
        paired_placebo=paired_placebo,
        model_config=model_config,
        alignment_config=alignment_config,
        metadata=metadata,
    )
    print(f"\n[{scope}] rows={len(dataset):,} prediction_rows={len(predictions):,}")
    if paired_placebo.empty:
        print("paired_placebo=skipped")
    else:
        actual = paired_placebo.loc[
            paired_placebo["portfolio"].isin([
                "top_5",
                "top_10",
                "top_20",
                "top_10pct",
                "top_20pct",
                "rank_weighted_long",
                "rank_weighted_ls",
            ])
            & paired_placebo["metric"].isin(["rank_ic", "minus_all"])
        ]
        print(actual.to_string(index=False))
    print(f"saved={output}")


def main() -> None:
    args = parse_args()
    created = datetime.now(timezone.utc)
    run_root = args.output_root / created.strftime("target_alignment_%Y%m%dT%H%M%SZ")
    run_root.mkdir(parents=True, exist_ok=False)
    full = pd.read_parquet(args.dataset)
    full["trade_date"] = pd.to_datetime(full["trade_date"]).dt.normalize()
    full["label_available_date"] = pd.to_datetime(
        full["label_available_date"]
    ).dt.normalize()
    metadata = {
        "created_at_utc": created.isoformat(),
        "source_dataset": str(args.dataset),
        "objectives": list(args.objectives),
        "arms": ["DT_BASE", "DT_SCORE"],
        "attribution_safe": (
            "sorted features, same fold seed across arms, LightGBM feature_fraction=1.0"
        ),
        "training_targets": {
            "lgb_regression": "continuous 10-day industry excess return",
            "lgb_lambdarank": "within-date return decile relevance 0-9",
            "lgb_top_decile": "within-date top ceil(10% of n) binary target",
            "logit_top_decile": "same binary target with regularized logistic model",
        },
        "portfolios": [
            "Top5/10/20",
            "Top10%/20%",
            "rank-weighted long-only",
            "rank-weighted dollar-neutral long-short",
        ],
        "label_purge": "train label_available_date strictly before test_start",
    }
    (run_root / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    objectives = tuple(args.objectives)
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
            placebo_repeats=args.long_placebo_repeats,
            objectives=objectives,
            run_placebo=not args.skip_placebo,
            output=run_root / "long_industry_context",
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
            placebo_repeats=args.pit_placebo_repeats,
            objectives=objectives,
            run_placebo=not args.skip_placebo,
            output=run_root / "pit_concept_context",
            metadata=metadata,
        )


if __name__ == "__main__":
    main()
