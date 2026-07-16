from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data import DataVersionRepository

from .post_impulse_config import PostImpulseMLConfig, load_post_impulse_ml_config
from .post_impulse_dataset import (
    REQUIRED_COLUMNS,
    assign_purged_splits,
    build_post_impulse_dataset,
)


ARM_BLOCKS = {
    "m0": ["coord"],
    "m1": ["coord", "event"],
    "m2": ["coord", "event", "pressure"],
    "m3": ["coord", "event", "pressure", "absorb"],
    "m4": ["coord", "event", "pressure", "absorb", "regime"],
    "m5": ["coord", "event", "pressure", "absorb", "regime", "interaction"],
}
REGRESSION_TARGET = "label__industry_excess_10d"
CLASSIFICATION_TARGET = "label__success"
ENGINE_VERSION = "post_impulse_path_ml_v1"


def _daily_equal_weights(dates: pd.Series) -> pd.Series:
    count = dates.groupby(dates, sort=False).transform("count").astype(float)
    weights = 1.0 / count
    return weights / weights.mean()


def _arm_features(blocks: dict[str, list[str]], arm: str) -> list[str]:
    return list(dict.fromkeys(
        column for block in ARM_BLOCKS[arm] for column in blocks.get(block, [])
    ))


def _neutralize_score(frame: pd.DataFrame, score: pd.Series) -> pd.Series:
    """Remove same-close industry/size/beta/liquidity/volatility exposure."""
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    controls = [
        "coord__log_circ_mv", "coord__beta_60d",
        "coord__volatility_20d_ts_pct", "coord__liquidity_20d_ts_pct",
    ]
    working = frame[["signal_date", "industry_l1_code", *controls]].copy()
    working["score"] = score
    for _, indexes in working.groupby("signal_date", sort=False).groups.items():
        sample = working.loc[indexes].dropna(subset=["score"])
        if len(sample) < 5:
            continue
        numeric = sample[controls].copy()
        numeric = numeric.fillna(numeric.median())
        std = numeric.std(ddof=0).replace(0.0, np.nan)
        numeric = ((numeric - numeric.mean()) / std).fillna(0.0)
        industries = pd.get_dummies(
            sample["industry_l1_code"].astype("string").fillna("UNKNOWN"),
            drop_first=True,
            dtype=float,
        )
        design = pd.concat([
            pd.Series(1.0, index=sample.index, name="intercept"), numeric, industries,
        ], axis=1)
        if len(sample) <= design.shape[1] + 4:
            output.loc[sample.index] = sample["score"] - sample["score"].mean()
            continue
        x, y = design.to_numpy(float), sample["score"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        output.loc[sample.index] = y - x @ beta
    return output


def _daily_rank_series(
    frame: pd.DataFrame, score: pd.Series, label: str, minimum: int
) -> pd.Series:
    sample = frame[["signal_date", label]].copy()
    sample["score"] = score
    values = {}
    for date, group in sample.dropna().groupby("signal_date", sort=True):
        if (
            len(group) >= minimum
            and group["score"].nunique() > 1
            and group[label].nunique() > 1
        ):
            values[date] = group["score"].corr(group[label], method="spearman")
    return pd.Series(values, dtype=float)


def _daily_top_bottom(
    frame: pd.DataFrame, score: pd.Series, label: str, minimum: int
) -> pd.Series:
    sample = frame[["signal_date", label]].copy()
    sample["score"] = score
    values = {}
    for date, group in sample.dropna().groupby("signal_date", sort=True):
        if len(group) < minimum or group["score"].nunique() < 2:
            continue
        count = max(1, math.ceil(len(group) * 0.2))
        ordered = group.sort_values("score")
        values[date] = float(
            ordered.tail(count)[label].mean() - ordered.head(count)[label].mean()
        )
    return pd.Series(values, dtype=float)


def _ranking_metrics(
    frame: pd.DataFrame,
    score: pd.Series,
    label: str,
    *,
    minimum_daily_events: int,
) -> dict:
    rank_ic = _daily_rank_series(frame, score, label, minimum_daily_events)
    spread = _daily_top_bottom(frame, score, label, minimum_daily_events)
    return {
        "sample_count": int(score.notna().sum()),
        "rank_ic_days": int(len(rank_ic)),
        "rank_ic_mean": float(rank_ic.mean()) if len(rank_ic) else np.nan,
        "rank_ic_positive_ratio": float((rank_ic > 0).mean()) if len(rank_ic) else np.nan,
        "top_bottom_days": int(len(spread)),
        "top_bottom_mean": float(spread.mean()) if len(spread) else np.nan,
    }


def _linear_regressor(alpha: float):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("imputer", SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        )),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def _success_classifier(c: float):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("imputer", SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        )),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=c, max_iter=1000, class_weight="balanced")),
    ])


def _lightgbm_regressor(cfg):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
        raise ImportError(
            'LightGBM is requested; install the optional ML dependencies with pip install -e ".[ml]"'
        ) from exc
    params = cfg.model_dump()
    params["verbosity"] = -1
    return LGBMRegressor(**params)


def _split_audit(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for split, group in frame.groupby("split", sort=False):
        rows.append({
            "split": str(split),
            "events": int(len(group)),
            "start": str(group["signal_date"].min().date()) if len(group) else None,
            "end": str(group["signal_date"].max().date()) if len(group) else None,
            "pressure_events": int(group["pressure__present"].eq(1.0).sum()),
            "mature_regression_labels": int(group[REGRESSION_TARGET].notna().sum()),
            "mature_success_labels": int(group[CLASSIFICATION_TARGET].notna().sum()),
        })
    return rows


class PostImpulseMLRunner:
    """Build the PIT event dataset and run fixed feature-block ablations."""

    def run(self, config_path: str | Path) -> dict:
        import joblib
        from sklearn.metrics import brier_score_loss, roc_auc_score

        config_path = Path(config_path)
        cfg = load_post_impulse_ml_config(config_path)
        project = load_project(cfg.project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        data_version, manifest = repository.load_manifest(cfg.data_version)
        panel_path = (
            Path(project.paths.data_root) / "versions" / data_version
            / "curated" / "stock_daily_panel.parquet"
        )
        panel = pd.read_parquet(
            panel_path,
            columns=REQUIRED_COLUMNS,
            filters=[("trade_date", ">=", pd.Timestamp(cfg.history_start_date))],
        )
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        built = build_post_impulse_dataset(panel, cfg.features, cfg.labels)
        events = assign_purged_splits(
            built.events, cfg.segments, pd.Index(panel["trade_date"].unique()),
            horizon=cfg.labels.horizon,
        )
        events = events.loc[
            events["signal_date"].between(
                pd.Timestamp(cfg.segments.train.start), pd.Timestamp(cfg.segments.test.end)
            )
            | events["split"].eq("purged")
        ].copy()
        if cfg.training.sample_scope == "pressure_events":
            modeling = events.loc[events["pressure__present"].eq(1.0)].copy()
        else:
            modeling = events.copy()
        if len(modeling.loc[modeling["split"].eq("train")]) < cfg.training.minimum_train_events:
            raise ValueError(
                "post-impulse training sample below minimum after pressure gate and purge: "
                f"{len(modeling.loc[modeling['split'].eq('train')])} < "
                f"{cfg.training.minimum_train_events}"
            )

        digest = hashlib.sha256(
            config_path.read_bytes() + data_version.encode() + ENGINE_VERSION.encode()
        ).hexdigest()[:16]
        run_id = f"post_impulse_ml_{digest}"
        output = cfg.output_root / run_id
        summary_path = output / "summary.json"
        if summary_path.exists():
            return {**json.loads(summary_path.read_text(encoding="utf-8")), "cached": True}
        output.mkdir(parents=True, exist_ok=False)
        (output / "models").mkdir()
        (output / "config.yaml").write_bytes(config_path.read_bytes())
        built.path.to_parquet(output / "event_path.parquet", index=False)
        events.to_parquet(output / "event_dataset.parquet", index=False)
        (output / "feature_manifest.json").write_text(
            json.dumps(built.feature_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        split_audit = _split_audit(events)
        (output / "split_audit.json").write_text(
            json.dumps(split_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        train = modeling["split"].eq("train")
        score_rows, metric_rows, feature_rows = [], [], []
        for arm in cfg.training.arms:
            features = _arm_features(built.feature_blocks, arm)
            if not features:
                raise ValueError(f"feature arm {arm} is empty")
            # Inf is never a valid missing encoding. Median imputation is fit on train only.
            x = modeling[features].apply(pd.to_numeric, errors="coerce").astype(float)
            x = x.mask(~np.isfinite(x))
            train_reg = train & modeling[REGRESSION_TARGET].notna()
            weights_reg = _daily_equal_weights(modeling.loc[train_reg, "signal_date"])
            for model_name in cfg.training.regression_models:
                model = (
                    _linear_regressor(cfg.training.ridge_alpha)
                    if model_name == "ridge"
                    else _lightgbm_regressor(cfg.training.lightgbm)
                )
                fit_kwargs = {"sample_weight": weights_reg}
                if model_name == "ridge":
                    fit_kwargs = {"model__sample_weight": weights_reg}
                model.fit(x.loc[train_reg], modeling.loc[train_reg, REGRESSION_TARGET], **fit_kwargs)
                joblib.dump(model, output / "models" / f"{arm}_{model_name}_regression.joblib")
                predicted = pd.Series(np.nan, index=modeling.index, dtype=float)
                eligible = modeling["split"].isin(["train", "valid", "test"])
                predicted.loc[eligible] = model.predict(x.loc[eligible])
                neutral = _neutralize_score(modeling, predicted)
                for split in ["train", "valid", "test"]:
                    mask = modeling["split"].eq(split) & modeling[REGRESSION_TARGET].notna()
                    for score_type, values in [("raw", predicted), ("neutral", neutral)]:
                        metric_rows.append({
                            "arm": arm, "model": model_name, "task": "regression",
                            "split": split, "score_type": score_type,
                            **_ranking_metrics(
                                modeling.loc[mask], values.loc[mask], REGRESSION_TARGET,
                                minimum_daily_events=cfg.training.minimum_daily_events,
                            ),
                        })
                score_rows.append(pd.DataFrame({
                    "event_id": modeling["event_id"], "signal_date": modeling["signal_date"],
                    "ts_code": modeling["ts_code"], "split": modeling["split"],
                    "arm": arm, "model": model_name, "task": "regression",
                    "target": modeling[REGRESSION_TARGET], "score_raw": predicted,
                    "score_neutral": neutral,
                }))
                feature_rows.extend(
                    {"arm": arm, "model": model_name, "feature": column}
                    for column in features
                )

            if cfg.training.run_success_classifier:
                train_cls = train & modeling[CLASSIFICATION_TARGET].notna()
                if modeling.loc[train_cls, CLASSIFICATION_TARGET].nunique() < 2:
                    raise ValueError("training success label has fewer than two classes")
                weights_cls = _daily_equal_weights(modeling.loc[train_cls, "signal_date"])
                classifier = _success_classifier(cfg.training.logistic_c)
                classifier.fit(
                    x.loc[train_cls], modeling.loc[train_cls, CLASSIFICATION_TARGET],
                    model__sample_weight=weights_cls,
                )
                joblib.dump(classifier, output / "models" / f"{arm}_logistic_success.joblib")
                probability = pd.Series(np.nan, index=modeling.index, dtype=float)
                eligible = modeling["split"].isin(["train", "valid", "test"])
                probability.loc[eligible] = classifier.predict_proba(x.loc[eligible])[:, 1]
                neutral = _neutralize_score(modeling, probability)
                for split in ["train", "valid", "test"]:
                    mask = modeling["split"].eq(split) & modeling[CLASSIFICATION_TARGET].notna()
                    y, p = modeling.loc[mask, CLASSIFICATION_TARGET], probability.loc[mask]
                    ranking = _ranking_metrics(
                        modeling.loc[mask], p, CLASSIFICATION_TARGET,
                        minimum_daily_events=cfg.training.minimum_daily_events,
                    )
                    metric_rows.append({
                        "arm": arm, "model": "logistic", "task": "classification",
                        "split": split, "score_type": "raw", **ranking,
                        "auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else np.nan,
                        "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
                    })
                    metric_rows.append({
                        "arm": arm, "model": "logistic", "task": "classification",
                        "split": split, "score_type": "neutral",
                        **_ranking_metrics(
                            modeling.loc[mask], neutral.loc[mask], CLASSIFICATION_TARGET,
                            minimum_daily_events=cfg.training.minimum_daily_events,
                        ),
                        "auc": np.nan, "brier": np.nan,
                    })
                score_rows.append(pd.DataFrame({
                    "event_id": modeling["event_id"], "signal_date": modeling["signal_date"],
                    "ts_code": modeling["ts_code"], "split": modeling["split"],
                    "arm": arm, "model": "logistic", "task": "classification",
                    "target": modeling[CLASSIFICATION_TARGET], "score_raw": probability,
                    "score_neutral": neutral,
                }))

        predictions = pd.concat(score_rows, ignore_index=True)
        metrics = pd.DataFrame(metric_rows)
        predictions.to_parquet(output / "predictions.parquet", index=False)
        metrics.to_csv(output / "ablation_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(feature_rows).to_csv(
            output / "arm_features.csv", index=False, encoding="utf-8-sig"
        )
        (output / "report.md").write_text(
            self._report(cfg, data_version, events, modeling, metrics), encoding="utf-8"
        )
        summary = {
            "run_id": run_id,
            "status": "COMPLETED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data_version": data_version,
            "data_end": manifest["end_date"],
            "event_count": int(len(events)),
            "modeling_event_count": int(len(modeling)),
            "sample_scope": cfg.training.sample_scope,
            "test_period_opened": True,
            "output_path": str(output.resolve()),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary

    @staticmethod
    def _report(
        cfg: PostImpulseMLConfig,
        data_version: str,
        events: pd.DataFrame,
        modeling: pd.DataFrame,
        metrics: pd.DataFrame,
    ) -> str:
        test = metrics.loc[
            metrics["split"].eq("test")
            & metrics["task"].eq("regression")
            & metrics["score_type"].eq("neutral")
        ]
        lines = [
            "# Post-impulse path ML ablation",
            "",
            f"- Data version: `{data_version}`.",
            f"- Events before sample gate: {len(events):,}; modeling events: {len(modeling):,}.",
            f"- Sample scope: `{cfg.training.sample_scope}`; labels start at the next open.",
            f"- Purge: {cfg.labels.horizon + 1} trading days before validation and test.",
            "- All median imputation, missing indicators and linear scaling are fit on training rows only.",
            "- The test interval is inspected by this run and cannot remain a clean future hold-out.",
            "",
            "## Test regression ablation (risk-neutral score)",
            "",
            "|Arm|Model|Features added through|Rank IC|Top-bottom|Days|",
            "|---|---|---|---:|---:|---:|",
        ]
        for row in test.itertuples():
            blocks = "+".join(ARM_BLOCKS[row.arm])
            lines.append(
                f"|{row.arm}|{row.model}|{blocks}|{row.rank_ic_mean:.4f}|"
                f"{row.top_bottom_mean:.4%}|{row.rank_ic_days}|"
            )
        lines += [
            "",
            "Interpretation is incremental: a path block is useful only when it improves the prior arm "
            "across validation and test without relying on risk exposure.",
            "",
        ]
        return "\n".join(lines)
