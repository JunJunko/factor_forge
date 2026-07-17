from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from factor_forge.ml.bullish_divergence_conditional_ml import (
    EvaluationConfig,
    ModelConfig,
    TestPeriod,
    assemble_conditional_dataset,
    build_daily_evaluation,
    compare_arms,
    persist_experiment,
    run_oof_predictions,
    summarize_evaluation,
)


DEFAULT_LONG_PERIODS = (
    TestPeriod(1, "2022-01-01", "2022-12-31"),
    TestPeriod(2, "2023-01-01", "2023-12-31"),
    TestPeriod(3, "2024-01-01", "2024-12-31"),
    TestPeriod(4, "2025-01-01", "2025-12-31"),
    TestPeriod(5, "2026-01-01", "2026-07-10"),
)

DEFAULT_PIT_PERIODS = (
    TestPeriod(1, "2025-07-01", "2025-09-30"),
    TestPeriod(2, "2025-10-01", "2025-12-31"),
    TestPeriod(3, "2026-01-01", "2026-03-31"),
    TestPeriod(4, "2026-04-01", "2026-07-10"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward OOF M0-M7 ablation for bullish-divergence conditional alpha."
    )
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--labeled-events", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--industry", type=Path, required=True)
    parser.add_argument("--concept-ml-dataset", type=Path)
    parser.add_argument("--concept-snapshot-root", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/bullish_divergence_conditional_ml"))
    parser.add_argument(
        "--scope",
        choices=("long_industry_context", "pit_concept_context", "both"),
        default="both",
    )
    parser.add_argument("--skip-placebo", action="store_true")
    parser.add_argument("--placebo-repeats", type=int, default=5)
    parser.add_argument("--reuse-dataset", type=Path)
    return parser.parse_args()


def run_scope(
    dataset: pd.DataFrame,
    *,
    scope: str,
    periods: tuple[TestPeriod, ...],
    output_dir: Path,
    run_placebo: bool,
    model_config: ModelConfig,
    metadata: dict[str, object],
) -> None:
    predictions, importance, folds = run_oof_predictions(
        dataset,
        scope=scope,
        periods=periods,
        model_config=model_config,
        run_shuffle_placebo=run_placebo,
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
        summary["top_n"].eq(10) & summary["cost_bps"].eq(40)
    ].sort_values(["algorithm", "top_minus_all_mean"], ascending=[True, False])
    print(f"\n[{scope}] rows={len(dataset):,} oof_predictions={len(predictions):,}")
    print(primary.loc[:, [
        "algorithm",
        "arm",
        "rank_ic_mean",
        "top_net_mean",
        "top_minus_all_mean",
        "top_minus_bottom_mean",
    ]].to_string(index=False))
    print(f"saved={output_dir}")


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.output_root / f"conditional_ml_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    if args.reuse_dataset:
        full_dataset = pd.read_parquet(args.reuse_dataset)
        full_dataset["trade_date"] = pd.to_datetime(full_dataset["trade_date"]).dt.normalize()
        full_dataset["label_available_date"] = pd.to_datetime(
            full_dataset["label_available_date"]
        ).dt.normalize()
    else:
        full_dataset = assemble_conditional_dataset(
            episodes_path=args.episodes,
            labeled_events_path=args.labeled_events,
            panel_path=args.panel,
            timing_path=args.timing,
            industry_path=args.industry,
            concept_ml_dataset_path=args.concept_ml_dataset,
            concept_snapshot_roots=args.concept_snapshot_root,
        )
    full_dataset.to_parquet(run_root / "full_conditional_event_dataset.parquet", index=False)

    metadata = {
        "episodes": str(args.episodes),
        "labeled_events": str(args.labeled_events),
        "panel": str(args.panel),
        "timing": str(args.timing),
        "industry": str(args.industry),
        "concept_ml_dataset": str(args.concept_ml_dataset) if args.concept_ml_dataset else None,
        "concept_snapshot_roots": [str(path) for path in args.concept_snapshot_root],
        "generated_at_utc": timestamp,
        "label": "T+1 open to T+11 open industry excess return",
        "training_weight": "equal trading-date weight",
        "purge": "train label_available_date strictly before test_start",
        "primary_evaluation": "Top10, 40 bps total round-trip event cost deducted once",
        "placebo_repeats": args.placebo_repeats,
    }
    model_config = ModelConfig(placebo_repeats=args.placebo_repeats)
    (run_root / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.scope in {"long_industry_context", "both"}:
        long_dataset = full_dataset.loc[
            full_dataset["trade_date"].between("2021-01-01", "2026-07-10")
        ].copy()
        long_dataset = long_dataset.drop(
            columns=[
                column
                for column in long_dataset.columns
                if column.startswith("concept__")
                or (
                    column.startswith("interaction__")
                    and "concept" in column
                )
            ]
        )
        run_scope(
            long_dataset,
            scope="long_industry_context",
            periods=DEFAULT_LONG_PERIODS,
            output_dir=run_root / "long_industry_context",
            run_placebo=not args.skip_placebo,
            model_config=model_config,
            metadata=metadata,
        )

    if args.scope in {"pit_concept_context", "both"}:
        if "concept__snapshot_available" not in full_dataset.columns:
            raise ValueError(
                "pit_concept_context requires --concept-ml-dataset and --concept-snapshot-root"
            )
        pit_dataset = full_dataset.loc[
            full_dataset["trade_date"].between("2024-12-30", "2026-07-10")
            & full_dataset["concept__snapshot_available"].eq(1.0)
        ].copy()
        run_scope(
            pit_dataset,
            scope="pit_concept_context",
            periods=DEFAULT_PIT_PERIODS,
            output_dir=run_root / "pit_concept_context",
            run_placebo=not args.skip_placebo,
            model_config=model_config,
            metadata=metadata,
        )


if __name__ == "__main__":
    main()
