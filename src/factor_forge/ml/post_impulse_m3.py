from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import Field

from .config import StrictModel
from .post_impulse_runner import (
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET,
    _daily_equal_weights,
    _neutralize_score,
    _ranking_metrics,
    _linear_regressor,
    _success_classifier,
)


ENGINE_VERSION = "post_impulse_m3_diagnostic_v1"

ABSORPTION_GROUPS = {
    "impact": [
        "absorb__impact_level", "absorb__impact_resilience", "absorb__impact_slope",
        "absorb__impact_decay_rank", "absorb__impact_observed_days", "absorb__impact_missing",
    ],
    "low": [
        "absorb__low_slope_atr", "absorb__low_slope_rank", "absorb__low_monotonicity",
    ],
    "close": [
        "absorb__close_location_slope", "absorb__close_vwap_slope",
        "absorb__close_monotonicity", "absorb__close_slope_rank",
    ],
    "range": ["absorb__range_slope_atr", "absorb__range_contraction_rank"],
    "summary": ["absorb__path_confirmation"],
}

VARIANT_GROUPS = {
    "m2": [],
    "m3_impact": ["impact"],
    "m3_low": ["low"],
    "m3_close": ["close"],
    "m3_range": ["range"],
    "m3_path_core": ["low", "close", "range"],
    "m3_summary": ["summary"],
    "m3_full": ["impact", "low", "close", "range", "summary"],
}


class PostImpulseM3Config(StrictModel):
    version: int = 1
    name: str = "post_impulse_m3_diagnostic_v1"
    source_run: Path
    ridge_alpha: float = Field(default=1000.0, gt=0)
    logistic_c: float = Field(default=1.0, gt=0)
    minimum_train_events: int = Field(default=300, ge=20)
    minimum_daily_events: int = Field(default=5, ge=3)
    output_root: Path = Path("artifacts/post_impulse_m3_runs")


def load_post_impulse_m3_config(path: str | Path) -> PostImpulseM3Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        return PostImpulseM3Config.model_validate(yaml.safe_load(handle) or {})


def _base_m2_features(events: pd.DataFrame) -> list[str]:
    prefixes = ("coord__", "event__", "pressure__")
    return sorted(
        column for column in events.columns
        if column.startswith(prefixes) and column != "pressure__present"
    )


def _variant_features(events: pd.DataFrame, variant: str) -> list[str]:
    columns = _base_m2_features(events)
    for group in VARIANT_GROUPS[variant]:
        columns.extend(ABSORPTION_GROUPS[group])
    missing = set(columns) - set(events.columns)
    if missing:
        raise ValueError(f"M3 variant {variant} missing columns: {', '.join(sorted(missing))}")
    return list(dict.fromkeys(columns))


def _coefficients(model, input_features: list[str], *, variant: str, task: str) -> pd.DataFrame:
    names = model.named_steps["imputer"].get_feature_names_out(input_features)
    values = model.named_steps["model"].coef_
    if np.ndim(values) > 1:
        values = np.asarray(values)[0]
    return pd.DataFrame({
        "variant": variant,
        "task": task,
        "feature": names,
        "standardized_coefficient": np.asarray(values, dtype=float),
    })


class PostImpulseM3DiagnosticRunner:
    """Fixed absorption sub-block diagnostic on the already-frozen M2 sample."""

    def run(self, config_path: str | Path) -> dict:
        import joblib
        from sklearn.metrics import brier_score_loss, roc_auc_score

        config_path = Path(config_path)
        cfg = load_post_impulse_m3_config(config_path)
        source_summary = json.loads((cfg.source_run / "summary.json").read_text(encoding="utf-8"))
        events = pd.read_parquet(cfg.source_run / "event_dataset.parquet")
        events["signal_date"] = pd.to_datetime(events["signal_date"])
        modeling = events.loc[
            events["pressure__present"].eq(1.0)
            & events["split"].isin(["train", "valid", "test"])
        ].copy()
        train = modeling["split"].eq("train")
        if train.sum() < cfg.minimum_train_events:
            raise ValueError(
                f"M3 training events {train.sum()} < minimum {cfg.minimum_train_events}"
            )

        digest = hashlib.sha256(
            config_path.read_bytes()
            + str(source_summary["run_id"]).encode()
            + ENGINE_VERSION.encode()
        ).hexdigest()[:16]
        run_id = f"post_impulse_m3_{digest}"
        output = cfg.output_root / run_id
        summary_path = output / "summary.json"
        if summary_path.exists():
            return {**json.loads(summary_path.read_text(encoding="utf-8")), "cached": True}
        output.mkdir(parents=True, exist_ok=False)
        (output / "models").mkdir()
        (output / "config.yaml").write_bytes(config_path.read_bytes())

        metrics, predictions, coefficients, feature_rows = [], [], [], []
        for variant in VARIANT_GROUPS:
            features = _variant_features(modeling, variant)
            x = modeling[features].apply(pd.to_numeric, errors="coerce").astype(float)
            x = x.mask(~np.isfinite(x))
            feature_rows.extend(
                {"variant": variant, "feature": feature} for feature in features
            )

            train_reg = train & modeling[REGRESSION_TARGET].notna()
            reg_weights = _daily_equal_weights(modeling.loc[train_reg, "signal_date"])
            regression = _linear_regressor(cfg.ridge_alpha)
            regression.fit(
                x.loc[train_reg], modeling.loc[train_reg, REGRESSION_TARGET],
                model__sample_weight=reg_weights,
            )
            joblib.dump(regression, output / "models" / f"{variant}_ridge.joblib")
            coefficients.append(_coefficients(
                regression, features, variant=variant, task="regression"
            ))
            reg_score = pd.Series(regression.predict(x), index=modeling.index)
            reg_neutral = _neutralize_score(modeling, reg_score)
            for split in ["train", "valid", "test"]:
                mask = modeling["split"].eq(split) & modeling[REGRESSION_TARGET].notna()
                for score_type, score in [("raw", reg_score), ("neutral", reg_neutral)]:
                    metrics.append({
                        "variant": variant, "task": "regression", "split": split,
                        "score_type": score_type,
                        **_ranking_metrics(
                            modeling.loc[mask], score.loc[mask], REGRESSION_TARGET,
                            minimum_daily_events=cfg.minimum_daily_events,
                        ),
                    })
            predictions.append(pd.DataFrame({
                "event_id": modeling["event_id"], "signal_date": modeling["signal_date"],
                "ts_code": modeling["ts_code"], "split": modeling["split"],
                "variant": variant, "task": "regression",
                "target": modeling[REGRESSION_TARGET], "score_raw": reg_score,
                "score_neutral": reg_neutral,
            }))

            train_cls = train & modeling[CLASSIFICATION_TARGET].notna()
            cls_weights = _daily_equal_weights(modeling.loc[train_cls, "signal_date"])
            classifier = _success_classifier(cfg.logistic_c)
            classifier.fit(
                x.loc[train_cls], modeling.loc[train_cls, CLASSIFICATION_TARGET],
                model__sample_weight=cls_weights,
            )
            joblib.dump(classifier, output / "models" / f"{variant}_logistic.joblib")
            coefficients.append(_coefficients(
                classifier, features, variant=variant, task="classification"
            ))
            probability = pd.Series(classifier.predict_proba(x)[:, 1], index=modeling.index)
            cls_neutral = _neutralize_score(modeling, probability)
            for split in ["train", "valid", "test"]:
                mask = modeling["split"].eq(split) & modeling[CLASSIFICATION_TARGET].notna()
                y, p = modeling.loc[mask, CLASSIFICATION_TARGET], probability.loc[mask]
                raw_metrics = _ranking_metrics(
                    modeling.loc[mask], p, CLASSIFICATION_TARGET,
                    minimum_daily_events=cfg.minimum_daily_events,
                )
                metrics.append({
                    "variant": variant, "task": "classification", "split": split,
                    "score_type": "raw", **raw_metrics,
                    "auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else np.nan,
                    "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
                })
                metrics.append({
                    "variant": variant, "task": "classification", "split": split,
                    "score_type": "neutral",
                    **_ranking_metrics(
                        modeling.loc[mask], cls_neutral.loc[mask], CLASSIFICATION_TARGET,
                        minimum_daily_events=cfg.minimum_daily_events,
                    ),
                    "auc": np.nan, "brier": np.nan,
                })
            predictions.append(pd.DataFrame({
                "event_id": modeling["event_id"], "signal_date": modeling["signal_date"],
                "ts_code": modeling["ts_code"], "split": modeling["split"],
                "variant": variant, "task": "classification",
                "target": modeling[CLASSIFICATION_TARGET], "score_raw": probability,
                "score_neutral": cls_neutral,
            }))

        metric_frame = pd.DataFrame(metrics)
        coefficient_frame = pd.concat(coefficients, ignore_index=True)
        metric_frame = self._with_incremental_deltas(metric_frame)
        metric_frame.to_csv(output / "m3_metrics.csv", index=False, encoding="utf-8-sig")
        coefficient_frame.to_csv(
            output / "standardized_coefficients.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(feature_rows).to_csv(
            output / "variant_features.csv", index=False, encoding="utf-8-sig"
        )
        pd.concat(predictions, ignore_index=True).to_parquet(
            output / "predictions.parquet", index=False
        )
        (output / "report.md").write_text(
            self._report(modeling, metric_frame, coefficient_frame), encoding="utf-8"
        )
        summary = {
            "run_id": run_id,
            "status": "COMPLETED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_run": source_summary["run_id"],
            "event_count": int(len(modeling)),
            "variants": list(VARIANT_GROUPS),
            "diagnostic_only": True,
            "test_period_previously_opened": True,
            "output_path": str(output.resolve()),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary

    @staticmethod
    def _with_incremental_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
        result = metrics.copy()
        keys = ["task", "split", "score_type"]
        baseline = result.loc[result["variant"].eq("m2"), [
            *keys, "rank_ic_mean", "top_bottom_mean", "auc", "brier",
        ]].rename(columns={
            "rank_ic_mean": "m2_rank_ic", "top_bottom_mean": "m2_top_bottom",
            "auc": "m2_auc", "brier": "m2_brier",
        })
        result = result.merge(baseline, on=keys, how="left", validate="many_to_one")
        result["delta_rank_ic_vs_m2"] = result["rank_ic_mean"] - result["m2_rank_ic"]
        result["delta_top_bottom_vs_m2"] = result["top_bottom_mean"] - result["m2_top_bottom"]
        result["delta_auc_vs_m2"] = result["auc"] - result["m2_auc"]
        result["delta_brier_vs_m2"] = result["brier"] - result["m2_brier"]
        return result

    @staticmethod
    def _report(
        events: pd.DataFrame, metrics: pd.DataFrame, coefficients: pd.DataFrame
    ) -> str:
        regression = metrics.loc[
            metrics["task"].eq("regression")
            & metrics["score_type"].eq("neutral")
            & metrics["split"].isin(["valid", "test"])
        ]
        classification = metrics.loc[
            metrics["task"].eq("classification")
            & metrics["score_type"].eq("raw")
            & metrics["split"].isin(["valid", "test"])
        ]
        lines = [
            "# M3 absorption-path diagnostic",
            "",
            f"- Common pressure-qualified events: {len(events):,}.",
            "- Fixed Ridge/Logit hyperparameters; no event threshold or window search.",
            "- M2 is the common coordinate + event + pressure baseline.",
            "- This is a post-hoc diagnostic on an already inspected test interval.",
            "",
            "## Continuous industry-excess return (risk-neutral score)",
            "",
            "|Variant|Split|Rank IC|Delta vs M2|Top-bottom|",
            "|---|---|---:|---:|---:|",
        ]
        for row in regression.itertuples():
            lines.append(
                f"|{row.variant}|{row.split}|{row.rank_ic_mean:.4f}|"
                f"{row.delta_rank_ic_vs_m2:+.4f}|{row.top_bottom_mean:.4%}|"
            )
        lines += [
            "",
            "## Second-wave success classification",
            "",
            "|Variant|Split|AUC|Delta AUC|Brier|",
            "|---|---|---:|---:|---:|",
        ]
        for row in classification.itertuples():
            lines.append(
                f"|{row.variant}|{row.split}|{row.auc:.4f}|"
                f"{row.delta_auc_vs_m2:+.4f}|{row.brier:.4f}|"
            )
        added = coefficients.loc[
            coefficients["variant"].ne("m2")
            & coefficients["feature"].str.startswith("absorb__")
        ].copy()
        if len(added):
            strongest = added.assign(
                absolute=added["standardized_coefficient"].abs()
            ).sort_values(["task", "absolute"], ascending=[True, False]).groupby(
                "task", sort=False
            ).head(8)
            lines += [
                "", "## Largest standardized absorption coefficients", "",
                strongest[["task", "variant", "feature", "standardized_coefficient"]].to_markdown(
                    index=False, floatfmt=".6f"
                ), "",
            ]
        return "\n".join(lines)

