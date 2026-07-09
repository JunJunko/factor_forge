from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from factor_forge.factor_reliability import (  # noqa: E402
    ReliabilityModelConfig,
    ReliabilitySplitConfig,
    build_reliability_scores,
    dynamic_weighting_simulation,
    load_feature_list,
    load_reliability_dataset,
    run_reliability_regression,
    write_reliability_model_report,
)


DEFAULT_DATASET = Path("artifacts/factor_reliability/factor_reliability_dataset_20260709T142147Z/factor_reliability_dataset.parquet")
DEFAULT_FEATURES = Path("artifacts/factor_reliability/factor_reliability_dataset_20260709T142147Z/factor_reliability_feature_list.csv")
OUTPUT_ROOT = Path("artifacts/factor_reliability")


def main() -> None:
    args = parse_args()
    output = args.output or OUTPUT_ROOT / f"factor_reliability_model_v1_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"loading dataset={args.dataset}")
    dataset = load_reliability_dataset(args.dataset)
    features = load_feature_list(args.features, dataset)
    log(f"rows={len(dataset):,} features={len(features)} dates={dataset['date'].nunique():,}")
    split_cfg = ReliabilitySplitConfig(
        train_start=args.train_start,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    model_cfg = ReliabilityModelConfig(horizons=tuple(args.horizons))
    results = run_reliability_regression(dataset, features, split_config=split_cfg, model_config=model_cfg)
    reliability_daily = build_reliability_scores(results["predictions"], model_name=args.score_model)
    simulation = dynamic_weighting_simulation(dataset, reliability_daily, horizon=args.simulation_horizon)

    dataset.to_parquet(output / "input_dataset_snapshot.parquet", index=False)
    results["predictions"].to_csv(output / "factor_reliability_predictions.csv", index=False, encoding="utf-8-sig")
    results["metrics"].to_csv(output / "model_comparison_metrics.csv", index=False, encoding="utf-8-sig")
    results["bucket_test"].to_csv(output / "reliability_bucket_test.csv", index=False, encoding="utf-8-sig")
    results["calibration"].to_csv(output / "calibration_table.csv", index=False, encoding="utf-8-sig")
    results["stability"].to_csv(output / "stability_by_year.csv", index=False, encoding="utf-8-sig")
    results["feature_importance"].to_csv(output / "feature_importance.csv", index=False, encoding="utf-8-sig")
    reliability_daily.to_csv(output / "factor_reliability_daily.csv", index=False, encoding="utf-8-sig")
    simulation.to_csv(output / "dynamic_factor_weighting_simulation.csv", index=False, encoding="utf-8-sig")
    write_reliability_model_report(
        output=output,
        dataset=dataset,
        results=results,
        reliability_daily=reliability_daily,
        simulation=simulation,
        source_path=str(args.dataset),
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "dataset": str(args.dataset),
                "feature_count": len(features),
                "score_model": args.score_model,
                "horizons": args.horizons,
                "split": {
                    "train": [args.train_start, args.train_end],
                    "valid": [args.valid_start, args.valid_end],
                    "test": [args.test_start, args.test_end],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Factor Reliability Model v1 regression baselines.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--score-model", default="ridge", choices=["historical_mean", "ridge", "lightgbm_shallow"])
    parser.add_argument("--simulation-horizon", type=int, default=10)
    parser.add_argument("--train-start", default="2024-01-02")
    parser.add_argument("--train-end", default="2025-06-30")
    parser.add_argument("--valid-start", default="2025-07-01")
    parser.add_argument("--valid-end", default="2025-12-31")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="2026-06-30")
    return parser.parse_args()


if __name__ == "__main__":
    main()
