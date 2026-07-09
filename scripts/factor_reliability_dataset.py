from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from factor_forge.factor_reliability import (  # noqa: E402
    ReliabilityFeatureConfig,
    ReliabilityLabelConfig,
    build_reliability_features,
    build_reliability_labels,
    reliability_feature_columns,
    write_reliability_dataset_report,
)


DEFAULT_HEALTH = Path("artifacts/factor_state/factor_state_transition_20260709T140009Z/factor_health_daily.parquet")
OUTPUT_ROOT = Path("artifacts/factor_reliability")


def main() -> None:
    args = parse_args()
    output = args.output or OUTPUT_ROOT / f"factor_reliability_dataset_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"loading health={args.health}")
    health = read_frame(args.health)
    health["date"] = pd.to_datetime(health["date"])
    log(f"health rows={len(health):,} dates={health['date'].nunique():,}")

    feature_cfg = ReliabilityFeatureConfig(cost_buffer=args.cost_buffer)
    label_cfg = ReliabilityLabelConfig(cost_buffer=args.cost_buffer, horizons=tuple(args.horizons))
    features = build_reliability_features(health, feature_cfg)
    labels = build_reliability_labels(health, label_cfg)
    dataset = features.merge(labels, on=["date", "factor_name"], how="left")
    feature_list = pd.DataFrame({"feature": reliability_feature_columns(dataset)})

    features.to_parquet(output / "factor_reliability_features.parquet", index=False)
    features.to_csv(output / "factor_reliability_features.csv", index=False, encoding="utf-8-sig")
    labels.to_csv(output / "factor_reliability_labels.csv", index=False, encoding="utf-8-sig")
    dataset.to_parquet(output / "factor_reliability_dataset.parquet", index=False)
    dataset.to_csv(output / "factor_reliability_dataset.csv", index=False, encoding="utf-8-sig")
    feature_list.to_csv(output / "factor_reliability_feature_list.csv", index=False, encoding="utf-8-sig")
    write_reliability_dataset_report(
        output=output,
        dataset=dataset,
        source_path=str(args.health),
        cost_buffer=args.cost_buffer,
        horizons=tuple(args.horizons),
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "health": str(args.health),
                "rows": int(len(dataset)),
                "start_date": str(dataset["date"].min().date()),
                "end_date": str(dataset["date"].max().date()),
                "horizons": args.horizons,
                "cost_buffer": args.cost_buffer,
                "feature_count": int(len(feature_list)),
                "note": "Feature and label generation only. No reliability model was trained.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Factor Reliability features and short-horizon labels.")
    parser.add_argument("--health", type=Path, default=DEFAULT_HEALTH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cost-buffer", type=float, default=0.002)
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20])
    return parser.parse_args()


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


if __name__ == "__main__":
    main()
