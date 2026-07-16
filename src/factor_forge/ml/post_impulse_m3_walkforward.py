from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import Field, model_validator

from factor_forge.config import load_project
from factor_forge.data import DataVersionRepository

from .config import StrictModel
from .post_impulse_m3 import _base_m2_features
from .post_impulse_runner import (
    CLASSIFICATION_TARGET,
    _daily_equal_weights,
    _neutralize_score,
    _ranking_metrics,
)


ENGINE_VERSION = "post_impulse_m3_minimal_walkforward_v1"
MINIMAL_M3_FEATURES = [
    "absorb__impact_observed_days",
    "absorb__impact_slope",
    "absorb__low_slope_atr",
    "absorb__close_vwap_slope",
]
EXPECTED_SIGNS = {
    "absorb__impact_observed_days": -1,
    "absorb__impact_slope": -1,
    "absorb__low_slope_atr": 1,
    "absorb__close_vwap_slope": 1,
}


class WalkForwardFold(StrictModel):
    id: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    @model_validator(mode="after")
    def chronological(self):
        if not (self.train_start <= self.train_end < self.test_start <= self.test_end):
            raise ValueError("walk-forward fold must satisfy train < test")
        return self


class MinimalM3Gate(StrictModel):
    minimum_positive_folds: int = Field(default=3, ge=1)
    minimum_oof_delta_rank_ic: float = 0.005
    minimum_sign_consistent_folds: int = Field(default=3, ge=1)
    require_nonnegative_auc_delta: bool = True
    require_nonpositive_brier_delta: bool = True
    require_positive_top_bottom_delta: bool = True


class PostImpulseM3WalkForwardConfig(StrictModel):
    version: Literal[1] = 1
    name: str = "post_impulse_m3_minimal_walkforward_v1"
    source_run: Path
    project_config: Path = Path("configs/project.yaml")
    purge_trading_days: Literal[11] = 11
    logistic_c: Literal[1.0] = 1.0
    calibration_bins: Literal[5] = 5
    minimum_train_events: int = Field(default=300, ge=20)
    minimum_daily_events: int = Field(default=5, ge=3)
    folds: list[WalkForwardFold]
    gate: MinimalM3Gate = Field(default_factory=MinimalM3Gate)
    output_root: Path = Path("artifacts/post_impulse_m3_walkforward_runs")

    @model_validator(mode="after")
    def expanding_folds(self):
        if len(self.folds) != 4:
            raise ValueError("minimal M3 walk-forward is frozen to four folds")
        if len({fold.id for fold in self.folds}) != len(self.folds):
            raise ValueError("walk-forward fold ids must be unique")
        for previous, current in zip(self.folds, self.folds[1:]):
            if current.train_start != previous.train_start:
                raise ValueError("walk-forward training must share a fixed origin")
            if current.train_end <= previous.train_end:
                raise ValueError("walk-forward training end must expand")
            if current.test_start <= previous.test_start:
                raise ValueError("walk-forward test intervals must advance")
        if self.gate.minimum_positive_folds > len(self.folds):
            raise ValueError("minimum_positive_folds exceeds fold count")
        if self.gate.minimum_sign_consistent_folds > len(self.folds):
            raise ValueError("minimum_sign_consistent_folds exceeds fold count")
        return self


def load_post_impulse_m3_walkforward_config(
    path: str | Path,
) -> PostImpulseM3WalkForwardConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return PostImpulseM3WalkForwardConfig.model_validate(yaml.safe_load(handle) or {})


def _classifier(c: float):
    """Unbalanced Logistic keeps probabilities interpretable for Brier/calibration."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("imputer", SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        )),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=c, max_iter=1000, class_weight=None)),
    ])


def _calendar_ordinals(panel_path: Path, start: str, end: str) -> pd.Index:
    dates = pd.read_parquet(
        panel_path,
        columns=["trade_date"],
        filters=[
            ("trade_date", ">=", pd.Timestamp(start)),
            ("trade_date", "<=", pd.Timestamp(end)),
        ],
    )["trade_date"]
    return pd.Index(sorted(pd.to_datetime(dates).unique()))


def _purged_train_mask(
    events: pd.DataFrame,
    fold: WalkForwardFold,
    calendar: pd.Index,
    purge_days: int,
) -> tuple[pd.Series, dict]:
    dates = pd.to_datetime(events["signal_date"])
    train = dates.between(pd.Timestamp(fold.train_start), pd.Timestamp(fold.train_end))
    test_start_ordinal = int(
        np.searchsorted(calendar.values, np.datetime64(fold.test_start), side="left")
    )
    ordinal = pd.Series(np.arange(len(calendar), dtype=int), index=calendar)
    event_ordinal = dates.map(ordinal)
    before = int(train.sum())
    overlap = train & event_ordinal.gt(test_start_ordinal - purge_days - 1)
    train &= ~overlap
    return train, {
        "train_before_purge": before,
        "train_after_purge": int(train.sum()),
        "purged_events": int(overlap.sum()),
        "last_train_signal": (
            str(dates.loc[train].max().date()) if train.any() else None
        ),
    }


def _coefficient_frame(model, input_features: list[str], fold_id: str, variant: str) -> pd.DataFrame:
    names = model.named_steps["imputer"].get_feature_names_out(input_features)
    values = np.asarray(model.named_steps["model"].coef_)[0]
    return pd.DataFrame({
        "fold": fold_id,
        "variant": variant,
        "feature": names,
        "standardized_coefficient": values.astype(float),
    })


def _calibration_table(
    predictions: pd.DataFrame, bins: int
) -> pd.DataFrame:
    rows = []
    for variant, group in predictions.groupby("variant", sort=False):
        sample = group.dropna(subset=["target", "probability"]).copy()
        sample["probability_bin"] = pd.qcut(
            sample["probability"], bins, labels=False, duplicates="drop"
        )
        table = sample.groupby("probability_bin", as_index=False).agg(
            event_count=("target", "size"),
            predicted_probability=("probability", "mean"),
            realized_success_rate=("target", "mean"),
        )
        table["variant"] = variant
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


class PostImpulseM3WalkForwardRunner:
    """Four-fold OOF comparison of M2 versus one frozen minimal M3 classifier."""

    def run(self, config_path: str | Path) -> dict:
        import joblib
        from sklearn.metrics import brier_score_loss, roc_auc_score

        config_path = Path(config_path)
        cfg = load_post_impulse_m3_walkforward_config(config_path)
        source_summary = json.loads((cfg.source_run / "summary.json").read_text(encoding="utf-8"))
        events = pd.read_parquet(cfg.source_run / "event_dataset.parquet")
        events["signal_date"] = pd.to_datetime(events["signal_date"])
        events = events.loc[
            events["pressure__present"].eq(1.0)
            & events[CLASSIFICATION_TARGET].notna()
        ].copy()
        missing = set(MINIMAL_M3_FEATURES) - set(events.columns)
        if missing:
            raise ValueError("minimal M3 source missing: " + ", ".join(sorted(missing)))

        project = load_project(cfg.project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        data_version = source_summary["data_version"]
        panel_path = (
            Path(project.paths.data_root) / "versions" / data_version
            / "curated" / "stock_daily_panel.parquet"
        )
        calendar = _calendar_ordinals(
            panel_path, cfg.folds[0].train_start, cfg.folds[-1].test_end
        )

        digest = hashlib.sha256(
            config_path.read_bytes()
            + str(source_summary["run_id"]).encode()
            + ENGINE_VERSION.encode()
        ).hexdigest()[:16]
        run_id = f"post_impulse_m3_walkforward_{digest}"
        output = cfg.output_root / run_id
        summary_path = output / "summary.json"
        if summary_path.exists():
            return {**json.loads(summary_path.read_text(encoding="utf-8")), "cached": True}
        output.mkdir(parents=True, exist_ok=False)
        (output / "models").mkdir()
        (output / "config.yaml").write_bytes(config_path.read_bytes())

        base = _base_m2_features(events)
        variants = {"c0_m2": base, "c1_m3_minimal": [*base, *MINIMAL_M3_FEATURES]}
        oof_rows, fold_metrics, coefficients, fold_audit = [], [], [], []
        for fold in cfg.folds:
            train, audit = _purged_train_mask(
                events, fold, calendar, cfg.purge_trading_days
            )
            test = events["signal_date"].between(
                pd.Timestamp(fold.test_start), pd.Timestamp(fold.test_end)
            )
            if train.sum() < cfg.minimum_train_events:
                raise ValueError(
                    f"fold {fold.id} train events {train.sum()} < {cfg.minimum_train_events}"
                )
            if not test.any():
                raise ValueError(f"fold {fold.id} has no test events")
            audit.update({
                "fold": fold.id,
                "test_events": int(test.sum()),
                "test_start": str(events.loc[test, "signal_date"].min().date()),
                "test_end": str(events.loc[test, "signal_date"].max().date()),
            })
            fold_audit.append(audit)
            train_weights = _daily_equal_weights(events.loc[train, "signal_date"])
            for variant, features in variants.items():
                x = events[features].apply(pd.to_numeric, errors="coerce").astype(float)
                x = x.mask(~np.isfinite(x))
                model = _classifier(cfg.logistic_c)
                model.fit(
                    x.loc[train], events.loc[train, CLASSIFICATION_TARGET],
                    model__sample_weight=train_weights,
                )
                joblib.dump(model, output / "models" / f"{fold.id}_{variant}.joblib")
                coefficients.append(_coefficient_frame(model, features, fold.id, variant))
                probability = pd.Series(
                    model.predict_proba(x.loc[test])[:, 1], index=events.index[test]
                )
                neutral = _neutralize_score(events.loc[test], probability)
                y = events.loc[test, CLASSIFICATION_TARGET]
                raw = _ranking_metrics(
                    events.loc[test], probability, CLASSIFICATION_TARGET,
                    minimum_daily_events=cfg.minimum_daily_events,
                )
                neutral_metrics = _ranking_metrics(
                    events.loc[test], neutral, CLASSIFICATION_TARGET,
                    minimum_daily_events=cfg.minimum_daily_events,
                )
                fold_metrics.append({
                    "fold": fold.id, "variant": variant,
                    "auc": float(roc_auc_score(y, probability)),
                    "brier": float(brier_score_loss(y, probability)),
                    "raw_rank_ic": raw["rank_ic_mean"],
                    "raw_top_bottom": raw["top_bottom_mean"],
                    "neutral_rank_ic": neutral_metrics["rank_ic_mean"],
                    "neutral_top_bottom": neutral_metrics["top_bottom_mean"],
                    "rank_ic_days": neutral_metrics["rank_ic_days"],
                })
                oof_rows.append(pd.DataFrame({
                    "event_id": events.loc[test, "event_id"],
                    "signal_date": events.loc[test, "signal_date"],
                    "ts_code": events.loc[test, "ts_code"],
                    "industry_l1_code": events.loc[test, "industry_l1_code"],
                    "fold": fold.id, "variant": variant,
                    "target": y, "probability": probability,
                    "neutral_score": neutral,
                }))

        oof = pd.concat(oof_rows, ignore_index=True)
        fold_frame = self._fold_deltas(pd.DataFrame(fold_metrics))
        coefficient_frame = pd.concat(coefficients, ignore_index=True)
        calibration = _calibration_table(oof, cfg.calibration_bins)
        aggregate = self._aggregate_metrics(oof, cfg.minimum_daily_events)
        sign_audit = self._sign_audit(coefficient_frame)
        gate = self._gate(cfg, fold_frame, aggregate, sign_audit)

        oof.to_parquet(output / "oof_predictions.parquet", index=False)
        fold_frame.to_csv(output / "fold_metrics.csv", index=False, encoding="utf-8-sig")
        aggregate.to_csv(output / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")
        calibration.to_csv(output / "calibration.csv", index=False, encoding="utf-8-sig")
        coefficient_frame.to_csv(
            output / "standardized_coefficients.csv", index=False, encoding="utf-8-sig"
        )
        sign_audit.to_csv(output / "coefficient_sign_audit.csv", index=False, encoding="utf-8-sig")
        (output / "fold_audit.json").write_text(
            json.dumps(fold_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output / "gate.json").write_text(
            json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output / "report.md").write_text(
            self._report(fold_frame, aggregate, calibration, sign_audit, gate), encoding="utf-8"
        )
        summary = {
            "run_id": run_id,
            "status": "COMPLETED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_run": source_summary["run_id"],
            "data_version": data_version,
            "oof_event_count": int(oof.loc[oof["variant"].eq("c0_m2")].shape[0]),
            "gate_passed": bool(gate["passed"]),
            "saved_next_action": gate["next_action"],
            "historical_diagnostic_only": True,
            "output_path": str(output.resolve()),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary

    @staticmethod
    def _fold_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
        baseline = metrics.loc[metrics["variant"].eq("c0_m2")].set_index("fold")
        result = metrics.copy()
        for column in [
            "auc", "brier", "raw_rank_ic", "raw_top_bottom",
            "neutral_rank_ic", "neutral_top_bottom",
        ]:
            result[f"delta_{column}_vs_c0"] = [
                value - baseline.loc[fold, column]
                for fold, value in result[["fold", column]].itertuples(index=False, name=None)
            ]
        return result

    @staticmethod
    def _aggregate_metrics(oof: pd.DataFrame, minimum_daily_events: int) -> pd.DataFrame:
        from sklearn.metrics import brier_score_loss, roc_auc_score

        rows = []
        for variant, group in oof.groupby("variant", sort=False):
            raw = _ranking_metrics(
                group.rename(columns={"target": CLASSIFICATION_TARGET}),
                group["probability"], CLASSIFICATION_TARGET,
                minimum_daily_events=minimum_daily_events,
            )
            neutral = _ranking_metrics(
                group.rename(columns={"target": CLASSIFICATION_TARGET}),
                group["neutral_score"], CLASSIFICATION_TARGET,
                minimum_daily_events=minimum_daily_events,
            )
            rows.append({
                "variant": variant,
                "event_count": int(len(group)),
                "auc": float(roc_auc_score(group["target"], group["probability"])),
                "brier": float(brier_score_loss(group["target"], group["probability"])),
                "raw_rank_ic": raw["rank_ic_mean"],
                "raw_top_bottom": raw["top_bottom_mean"],
                "neutral_rank_ic": neutral["rank_ic_mean"],
                "neutral_top_bottom": neutral["top_bottom_mean"],
                "rank_ic_days": neutral["rank_ic_days"],
            })
        result = pd.DataFrame(rows)
        baseline = result.loc[result["variant"].eq("c0_m2")].iloc[0]
        for column in [
            "auc", "brier", "raw_rank_ic", "raw_top_bottom",
            "neutral_rank_ic", "neutral_top_bottom",
        ]:
            result[f"delta_{column}_vs_c0"] = result[column] - baseline[column]
        return result

    @staticmethod
    def _sign_audit(coefficients: pd.DataFrame) -> pd.DataFrame:
        sample = coefficients.loc[
            coefficients["variant"].eq("c1_m3_minimal")
            & coefficients["feature"].isin(MINIMAL_M3_FEATURES)
        ].copy()
        sample["expected_sign"] = sample["feature"].map(EXPECTED_SIGNS)
        sample["observed_sign"] = np.sign(sample["standardized_coefficient"]).astype(int)
        sample["sign_matches"] = sample["expected_sign"].eq(sample["observed_sign"])
        summary = sample.groupby("feature", as_index=False).agg(
            expected_sign=("expected_sign", "first"),
            matching_folds=("sign_matches", "sum"),
            total_folds=("fold", "nunique"),
            mean_coefficient=("standardized_coefficient", "mean"),
            min_coefficient=("standardized_coefficient", "min"),
            max_coefficient=("standardized_coefficient", "max"),
        )
        return summary

    @staticmethod
    def _gate(
        cfg: PostImpulseM3WalkForwardConfig,
        folds: pd.DataFrame,
        aggregate: pd.DataFrame,
        signs: pd.DataFrame,
    ) -> dict:
        candidate_folds = folds.loc[folds["variant"].eq("c1_m3_minimal")]
        candidate = aggregate.loc[aggregate["variant"].eq("c1_m3_minimal")].iloc[0]
        checks = {
            "positive_fold_count": int(candidate_folds["delta_neutral_rank_ic_vs_c0"].gt(0).sum()),
            "minimum_positive_folds": cfg.gate.minimum_positive_folds,
            "oof_delta_neutral_rank_ic": float(candidate["delta_neutral_rank_ic_vs_c0"]),
            "minimum_oof_delta_rank_ic": cfg.gate.minimum_oof_delta_rank_ic,
            "oof_delta_auc": float(candidate["delta_auc_vs_c0"]),
            "oof_delta_brier": float(candidate["delta_brier_vs_c0"]),
            "oof_delta_neutral_top_bottom": float(candidate["delta_neutral_top_bottom_vs_c0"]),
            "minimum_sign_matching_folds": int(signs["matching_folds"].min()),
            "required_sign_matching_folds": cfg.gate.minimum_sign_consistent_folds,
        }
        passed = (
            checks["positive_fold_count"] >= cfg.gate.minimum_positive_folds
            and checks["oof_delta_neutral_rank_ic"] >= cfg.gate.minimum_oof_delta_rank_ic
            and checks["minimum_sign_matching_folds"] >= cfg.gate.minimum_sign_consistent_folds
            and (
                not cfg.gate.require_nonnegative_auc_delta or checks["oof_delta_auc"] >= 0
            )
            and (
                not cfg.gate.require_nonpositive_brier_delta or checks["oof_delta_brier"] <= 0
            )
            and (
                not cfg.gate.require_positive_top_bottom_delta
                or checks["oof_delta_neutral_top_bottom"] > 0
            )
        )
        return {
            "passed": bool(passed),
            "checks": checks,
            "next_action": (
                "test_m2_ranking_with_m3_probability_gate"
                if passed else "retain_m2_and_stop_m3_expansion"
            ),
        }

    @staticmethod
    def _report(
        folds: pd.DataFrame,
        aggregate: pd.DataFrame,
        calibration: pd.DataFrame,
        signs: pd.DataFrame,
        gate: dict,
    ) -> str:
        candidate_folds = folds.loc[folds["variant"].eq("c1_m3_minimal")]
        lines = [
            "# Minimal M3 expanding walk-forward",
            "",
            "- C0: M2 Logistic; C1: C0 plus four frozen minimal M3 features.",
            "- Four expanding folds, 11-trading-day purge, daily-equal training weights.",
            "- Imputation and scaling are fit inside each fold; probabilities are unbalanced Logistic outputs.",
            "- Historical diagnostic only: all covered dates were previously inspected.",
            "",
            "## Fold deltas, C1 minus C0",
            "",
            "|Fold|Delta neutral IC|Delta AUC|Delta Brier|Delta top-bottom|",
            "|---|---:|---:|---:|---:|",
        ]
        for row in candidate_folds.itertuples():
            lines.append(
                f"|{row.fold}|{row.delta_neutral_rank_ic_vs_c0:+.4f}|"
                f"{row.delta_auc_vs_c0:+.4f}|{row.delta_brier_vs_c0:+.4f}|"
                f"{row.delta_neutral_top_bottom_vs_c0:+.4%}|"
            )
        lines += [
            "", "## Aggregate OOF", "",
            aggregate.to_markdown(index=False, floatfmt=".6f"),
            "", "## Coefficient sign stability", "",
            signs.to_markdown(index=False, floatfmt=".6f"),
            "", "## Probability calibration", "",
            calibration.to_markdown(index=False, floatfmt=".6f"),
            "", "## Deterministic diagnostic Gate", "",
            f"- Passed: `{gate['passed']}`.",
            f"- Saved next action: `{gate['next_action']}`.",
            "",
        ]
        return "\n".join(lines)

