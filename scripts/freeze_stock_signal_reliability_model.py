"""Freeze the selected stock-level signal reliability model.

This materializes the Experiment 8 selected model so live signal generation can
load a fixed artifact instead of retraining during web requests.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_RUN = Path(
    "artifacts/strategy_reviews/experiment8_signal_reliability/"
    "stock_signal_reliability_20260709T151758Z"
)
OUTPUT_DIR = Path("artifacts/frozen_models/stock_signal_reliability_lambda005_v1")
MODEL_NAME = "lightgbm_shallow"
HORIZON = 10
TARGET = f"success_{HORIZON}d"
ACTUAL = f"future_trade_return_{HORIZON}d"
RANDOM_SEED = 20260709


def main() -> None:
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signal_dataset = pd.read_csv(SOURCE_RUN / "signal_dataset.csv", parse_dates=["trade_date"])
    feature_cols = pd.read_csv(SOURCE_RUN / "feature_list.csv")["feature"].astype(str).to_list()
    train = signal_dataset.loc[signal_dataset["sample"].eq("train")].dropna(subset=[TARGET, ACTUAL]).copy()
    if train.empty:
        raise ValueError("No training rows found for frozen reliability model.")

    model = lgb.LGBMClassifier(
        objective="binary",
        max_depth=3,
        num_leaves=8,
        learning_rate=0.05,
        n_estimators=200,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        random_state=RANDOM_SEED,
        verbosity=-1,
        force_col_wise=True,
    )
    model.fit(train[feature_cols].fillna(0.0), train[TARGET].astype(int))
    model_path = OUTPUT_DIR / "signal_reliability_lgbm_10d.txt"
    model.booster_.save_model(model_path)
    (OUTPUT_DIR / "feature_list.json").write_text(
        json.dumps(feature_cols, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics = []
    for sample in ["train", "valid", "test"]:
        part = signal_dataset.loc[signal_dataset["sample"].eq(sample)].dropna(subset=[TARGET, ACTUAL]).copy()
        if part.empty:
            continue
        prob = model.predict_proba(part[feature_cols].fillna(0.0))[:, 1]
        y = part[TARGET].astype(int)
        rank_ic = pd.Series(prob, index=part.index).corr(part[ACTUAL], method="spearman")
        metrics.append(
            {
                "sample": sample,
                "rows": int(len(part)),
                "positive_ratio": float(y.mean()),
                "roc_auc": float(roc_auc_score(y, prob)) if y.nunique() > 1 else np.nan,
                "pr_auc": float(average_precision_score(y, prob)) if y.nunique() > 1 else np.nan,
                "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
                "mean_future_return": float(part[ACTUAL].mean()),
            }
        )
    pd.DataFrame(metrics).to_csv(OUTPUT_DIR / "frozen_model_metrics.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "model_version": "stock_signal_reliability_lambda005_v1",
        "model_name": MODEL_NAME,
        "horizon": HORIZON,
        "score_formula": "final_score = alpha_score + 0.05 * reliability_zscore",
        "lambda": 0.05,
        "source_experiment": str(SOURCE_RUN),
        "source_dataset": str(SOURCE_RUN / "signal_dataset.csv"),
        "training_sample": "Experiment 8 train split only",
        "target": TARGET,
        "actual_return_column": ACTUAL,
        "feature_count": len(feature_cols),
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "no_live_retraining": True,
    }
    (OUTPUT_DIR / "freeze_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"frozen reliability model -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
