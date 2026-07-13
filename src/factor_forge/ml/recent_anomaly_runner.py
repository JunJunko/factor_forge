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
from factor_forge.evaluation.l1 import _newey_west_stats
from factor_forge.experiments.artifacts import json_default
from factor_forge.radar.templates import (
    filter_required_fields,
    load_radar_template,
)

from .mamba_state_trainer import encode_sequences, fit_reference_encoder
from .recent_anomaly_config import load_recent_anomaly_structure_config
from .recent_anomaly_dataset import build_recent_anomaly_dataset
from .recent_anomaly_structure import assert_fold_label_maturity, build_walk_forward_folds


ENGINE_VERSION = "recent_anomaly_structure_ranker_v1"


class RecentAnomalyStructureRunner:
    """PIT rolling-OOS comparison of static and recent-state event rankers."""

    def run(self, config_path: str | Path) -> dict:
        started = datetime.now(timezone.utc)
        config_path = Path(config_path)
        cfg = load_recent_anomaly_structure_config(config_path)
        scan = json.loads(cfg.scan_summary.read_text(encoding="utf-8"))
        templates = [load_radar_template(path) for path in cfg.event_templates]
        _validate_scan_contract(scan, templates, cfg)
        project = load_project(cfg.project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        version, _ = repository.load_manifest(cfg.data_version)
        if version != scan["data_version"]:
            raise ValueError("scan and recent-structure config use different immutable data versions")
        as_of = pd.Timestamp(cfg.as_of_date)

        source_events = {
            row["template_id"]: pd.read_parquet(row["events_path"])
            for row in scan["events"] if row["template_id"] in {template.id for template in templates}
        }
        earliest_event = min(
            pd.to_datetime(frame["trade_date"]).min() for frame in source_events.values()
        )
        warmup_rows = max(cfg.recent_structure.sequence_length, max(cfg.features.windows))
        start = earliest_event - pd.Timedelta(days=math.ceil(warmup_rows * 7 / 5) + 30)
        panel_path = project.paths.data_root / "versions" / version / "curated" / "stock_daily_panel.parquet"
        columns = list(dict.fromkeys([
            "trade_date", "ts_code", "adj_open", "adj_high", "adj_low", "adj_close",
            "volume_shares", "amount_cny", "turnover_rate", "log_total_mv", "log_circ_mv",
            "industry_l1_code", "is_tradeable", "is_liquid",
            *(field for template in templates for field in filter_required_fields(template)),
            *(field for template in templates for field in template.data.required_fields),
        ]))
        panel = pd.read_parquet(
            panel_path, columns=columns,
            filters=[("trade_date", ">=", start), ("trade_date", "<=", as_of)],
        )
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        dataset = build_recent_anomaly_dataset(
            panel, templates, cfg, as_of_date=as_of, source_events=source_events,
        )
        _validate_live_sources(scan, dataset.live_events, templates, as_of)

        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        identity = hashlib.sha256(json.dumps({
            "engine": ENGINE_VERSION, "data_version": version,
            "template_hashes": dataset.template_hashes, "config_hash": config_hash,
        }, sort_keys=True).encode()).hexdigest()[:16]
        run_id = f"recent_anomaly_structure_{identity}"
        output = cfg.output_root / run_id
        summary_path = output / "summary.json"
        if summary_path.exists():
            return {**json.loads(summary_path.read_text(encoding="utf-8")), "cached": True}
        output.mkdir(parents=True, exist_ok=False)
        (output / "folds").mkdir()

        calendar = dataset.calendar[-cfg.recent_structure.history_trading_days:]
        folds = build_walk_forward_folds(
            calendar, training_days=cfg.walk_forward.training_days,
            validation_days=cfg.walk_forward.validation_days,
            test_days=cfg.walk_forward.test_days, step_days=cfg.walk_forward.step_days,
            horizon=cfg.primary_horizon,
        )[-cfg.walk_forward.maximum_folds:]
        if not folds:
            raise ValueError("recent-structure configuration produced no walk-forward folds")
        predictions, importances, fold_summaries = [], [], []
        for fold in folds:
            fold_output = output / "folds" / f"fold_{fold.fold:02d}"
            fold_output.mkdir()
            result = _run_fold(dataset, fold, cfg, fold_output)
            predictions.append(result["predictions"])
            importances.append(result["importance"])
            fold_summaries.append(result["summary"])
        prediction = pd.concat(predictions, ignore_index=True)
        prediction.to_parquet(output / "oos_predictions.parquet", index=False)
        daily = _daily_paired_ic(prediction)
        daily.to_parquet(output / "paired_daily_rank_ic.parquet", index=False)
        stats = _paired_stats(daily, cfg.primary_horizon)
        factor_importance = _aggregate_importance(importances)
        factor_importance.to_csv(output / "conditional_feature_weights.csv", index=False, encoding="utf-8-sig")
        live, live_audit = _fit_live_models(dataset, cfg, output / "live", as_of)
        live.to_parquet(output / "live_ranking.parquet", index=False)
        live.to_csv(output / "live_ranking.csv", index=False, encoding="utf-8-sig")

        passed = (
            stats.get("nw_t_value") is not None
            and stats["nw_t_value"] >= cfg.gate_min_nw_t
            and stats.get("positive_ratio", 0) > cfg.gate_min_positive_ratio
            and stats.get("dates", 0) >= cfg.walk_forward.minimum_test_dates
        )
        decision = "OBSERVE_FORWARD" if passed else "NO_VALIDATED_RECENT_STATE_INCREMENT"
        summary = {
            "run_id": run_id, "engine_version": ENGINE_VERSION,
            "data_version": version, "scan_id": scan["scan_id"],
            "as_of_date": cfg.as_of_date,
            "idea_id": "idea_recent_anomaly_structure_v1",
            "plan_id": "plan_recent_anomaly_structure_walkforward_v1",
            "template_ids": [template.id for template in templates],
            "historical_episode_rows": int(len(dataset.episodes)),
            "live_event_template_rows": int(len(dataset.live_events)),
            "live_unique_stocks": int(live["instrument"].nunique()),
            "folds": fold_summaries, "primary_metric": cfg.primary_metric,
            "paired_daily_rank_ic_delta": stats,
            "gate": {"min_nw_t": cfg.gate_min_nw_t,
                     "min_positive_ratio": cfg.gate_min_positive_ratio,
                     "decision": decision},
            "top_conditional_features": factor_importance.head(20).to_dict(orient="records"),
            "live_fit_audit": live_audit,
            "top_live": live.head(20).to_dict(orient="records"),
            "interpretation_boundary": (
                "One pre-registered historical validation peek. OBSERVE_FORWARD is not Alpha; "
                "today's events have no labels and are excluded from fitting."
            ),
            "run_dir": str(output.resolve()), "cached": False,
        }
        summary_path.write_text(json.dumps(
            summary, ensure_ascii=False, indent=2, default=json_default
        ), encoding="utf-8")
        (output / "manifest.json").write_text(json.dumps({
            "run_id": run_id, "runner_type": "recent_anomaly_structure_runs",
            "status": "COMPLETED", "data_version": version,
            "scan_id": scan["scan_id"], "template_hashes": dataset.template_hashes,
            "contains_unmatured_labels_in_features": False,
            "today_events_in_fit_rows": 0,
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "report.md").write_text(_report(summary), encoding="utf-8")
        return summary


def _run_fold(dataset, fold, cfg, output):
    episodes = dataset.episodes.copy()
    assert_fold_label_maturity(episodes, fold)
    masks = {
        "train": episodes["trade_date"].between(fold.train_start, fold.train_end),
        "valid": episodes["trade_date"].between(fold.valid_start, fold.valid_end),
        "test": episodes["trade_date"].between(fold.test_start, fold.test_end),
    }
    for name in masks:
        masks[name] &= episodes["target"].notna() & episodes["is_liquid"].eq(True)
    if masks["train"].sum() < cfg.walk_forward.minimum_train_events:
        raise ValueError(f"fold {fold.fold} has too few train events")
    if any(not masks[name].any() for name in ("valid", "test")):
        raise ValueError(f"fold {fold.fold} contains an empty validation or test sample")
    train_positions = np.unique(episodes.loc[masks["train"], "sample_position"].to_numpy(int))
    valid_positions = np.unique(episodes.loc[masks["valid"], "sample_position"].to_numpy(int))
    fit = fit_reference_encoder(
        dataset.store, train_positions, valid_positions, cfg.encoder, cfg.training,
        seed=17, checkpoint_path=output / "encoder_seed_17.pt",
    )
    selected = episodes.loc[masks["train"] | masks["valid"] | masks["test"]].copy()
    positions = np.unique(selected["sample_position"].to_numpy(int))
    embedding = encode_sequences(
        fit.model, dataset.store, positions,
        batch_size=cfg.training.batch_size, device=fit.device,
    )
    selected = _attach_embeddings(selected, positions, embedding)
    static_features, adaptive_features = _feature_sets(dataset, embedding.shape[1])
    usable = selected[[*static_features, "target"]].replace([np.inf, -np.inf], np.nan)
    selected = selected.loc[usable[static_features].notna().all(axis=1) & usable["target"].notna()].copy()
    split_masks = {
        name: selected["trade_date"].between(getattr(fold, f"{name}_start"), getattr(fold, f"{name}_end"))
        for name in ("train", "valid", "test")
    }
    weights = _sample_weights(selected.loc[split_masks["train"]], fold.train_end)
    static = _fit_lgbm(
        selected.loc[split_masks["train"]], selected.loc[split_masks["valid"]],
        static_features, cfg, weights,
    )
    adaptive = _fit_lgbm(
        selected.loc[split_masks["train"]], selected.loc[split_masks["valid"]],
        adaptive_features, cfg, weights,
    )
    test = selected.loc[split_masks["test"], [
        "episode_id", "trade_date", "ts_code", "template_id", "target",
    ]].copy()
    test["score_static"] = static.predict(selected.loc[split_masks["test"], static_features])
    test["score_adaptive"] = adaptive.predict(selected.loc[split_masks["test"], adaptive_features])
    test["fold"] = fold.fold
    test.to_parquet(output / "predictions.parquet", index=False)
    importance = _model_contributions(
        adaptive, selected.loc[split_masks["test"], adaptive_features], adaptive_features, fold.fold
    )
    importance.to_csv(output / "conditional_feature_weights.csv", index=False, encoding="utf-8-sig")
    fit.history.to_parquet(output / "encoder_history.parquet", index=False)
    return {
        "predictions": test, "importance": importance,
        "summary": {
            "fold": fold.fold, "train_start": fold.train_start, "train_end": fold.train_end,
            "valid_start": fold.valid_start, "valid_end": fold.valid_end,
            "test_start": fold.test_start, "test_end": fold.test_end,
            "train_rows": int(split_masks["train"].sum()),
            "valid_rows": int(split_masks["valid"].sum()),
            "test_rows": int(split_masks["test"].sum()),
            "checkpoint_sha256": fit.checkpoint_sha256,
        },
    }


def _fit_live_models(dataset, cfg, output, as_of):
    output.mkdir()
    episodes = dataset.episodes.copy()
    mature = (
        episodes["target"].notna() & episodes["label_available_date"].le(as_of)
        & episodes["is_liquid"].eq(True)
    )
    calendar = dataset.calendar
    as_of_index = int(calendar.get_loc(as_of))
    train_start = calendar[max(0, as_of_index - cfg.walk_forward.training_days + 1)]
    history = episodes.loc[mature & episodes["trade_date"].ge(train_start)].copy()
    dates = pd.DatetimeIndex(sorted(history["trade_date"].unique()))
    valid_start = dates[max(1, len(dates) - cfg.walk_forward.validation_days)]
    encoder_train = history["trade_date"].lt(valid_start)
    encoder_valid = history["trade_date"].ge(valid_start)
    fit = fit_reference_encoder(
        dataset.store,
        np.unique(history.loc[encoder_train, "sample_position"].to_numpy(int)),
        np.unique(history.loc[encoder_valid, "sample_position"].to_numpy(int)),
        cfg.encoder, cfg.training, seed=17,
        checkpoint_path=output / "encoder_seed_17.pt",
    )
    combined = pd.concat([history, dataset.live_events], ignore_index=True, sort=False)
    positions = np.unique(combined["sample_position"].to_numpy(int))
    embedding = encode_sequences(
        fit.model, dataset.store, positions,
        batch_size=cfg.training.batch_size, device=fit.device,
    )
    combined = _attach_embeddings(combined, positions, embedding)
    static_features, adaptive_features = _feature_sets(dataset, embedding.shape[1])
    train = combined["target"].notna() & ~combined["episode_id"].astype(str).str.startswith("live_")
    live_mask = combined["episode_id"].astype(str).str.startswith("live_")
    common = combined[static_features].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    train &= common
    live_mask &= common
    weights = _sample_weights(combined.loc[train], as_of)
    static = _fit_lgbm(combined.loc[train], combined.loc[train], static_features, cfg, weights)
    adaptive = _fit_lgbm(combined.loc[train], combined.loc[train], adaptive_features, cfg, weights)
    live_rows = combined.loc[live_mask, [
        "trade_date", "ts_code", "template_id", "severity",
        "template_target_mean_20", "template_target_mean_60", "template_target_velocity",
    ]].copy()
    live_rows["score_static"] = static.predict(combined.loc[live_mask, static_features])
    live_rows["score_adaptive"] = adaptive.predict(combined.loc[live_mask, adaptive_features])
    live = live_rows.groupby(["trade_date", "ts_code"], as_index=False).agg(
        templates_triggered=("template_id", lambda values: ",".join(sorted(set(values)))),
        severity_max=("severity", "max"),
        score_static=("score_static", "mean"), score_adaptive=("score_adaptive", "mean"),
        recent_mean_20=("template_target_mean_20", "mean"),
        recent_mean_60=("template_target_mean_60", "mean"),
        recent_velocity=("template_target_velocity", "mean"),
    ).rename(columns={"trade_date": "datetime", "ts_code": "instrument"})
    live["rank_static"] = live["score_static"].rank(ascending=False, method="min").astype(int)
    live["rank_adaptive"] = live["score_adaptive"].rank(ascending=False, method="min").astype(int)
    live = live.sort_values("rank_adaptive").reset_index(drop=True)
    return live, {
        "fit_rows": int(train.sum()), "today_fit_rows": 0,
        "live_event_template_rows_scored": int(live_mask.sum()),
        "live_unique_stocks_scored": int(len(live)),
        "checkpoint_sha256": fit.checkpoint_sha256,
    }


def _feature_sets(dataset, state_dim):
    static = ["severity", *dataset.raw_feature_names, *dataset.template_feature_names]
    state = [f"recent_state_{index:02d}" for index in range(state_dim)]
    return static, [*static, *state, *dataset.recent_feature_names]


def _attach_embeddings(frame, positions, embedding):
    result = frame.copy()
    lookup = {int(position): row for row, position in enumerate(positions)}
    rows = np.array([lookup[int(position)] for position in result["sample_position"]], dtype=int)
    for index in range(embedding.shape[1]):
        result[f"recent_state_{index:02d}"] = embedding[rows, index]
    return result


def _sample_weights(train, reference_date):
    dates = pd.to_datetime(train["trade_date"])
    age = (pd.Timestamp(reference_date) - dates).dt.days
    decay = np.power(0.5, age / 126.0)
    date_count = train.groupby("trade_date")["episode_id"].transform("count")
    duplicate_count = train.groupby(["trade_date", "ts_code"])["episode_id"].transform("count")
    weight = decay / date_count / duplicate_count
    return weight / weight.mean()


def _fit_lgbm(train, valid, features, cfg, weights):
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(**cfg.lightgbm.model_dump(), verbosity=-1)
    model.fit(
        train[features], train["target"], sample_weight=weights,
        eval_set=[(valid[features], valid["target"])],
    )
    return model


def _model_contributions(model, frame, features, fold):
    contributions = model.booster_.predict(frame, pred_contrib=True)[:, :-1]
    rows = []
    for index, feature in enumerate(features):
        values = pd.to_numeric(frame[feature], errors="coerce")
        contrib = pd.Series(contributions[:, index], index=frame.index)
        valid = values.notna() & contrib.notna()
        direction = values.loc[valid].corr(contrib.loc[valid], method="spearman") if valid.sum() >= 3 else np.nan
        rows.append({
            "fold": fold, "feature": feature,
            "mean_abs_contribution": float(np.mean(np.abs(contributions[:, index]))),
            "mean_contribution": float(np.mean(contributions[:, index])),
            "conditional_direction": float(direction) if pd.notna(direction) else np.nan,
        })
    return pd.DataFrame(rows)


def _aggregate_importance(frames):
    data = pd.concat(frames, ignore_index=True)
    return data.groupby("feature", as_index=False).agg(
        folds=("fold", "nunique"),
        mean_abs_contribution=("mean_abs_contribution", "mean"),
        contribution_std=("mean_abs_contribution", "std"),
        mean_conditional_direction=("conditional_direction", "mean"),
        direction_positive_ratio=("conditional_direction", lambda x: float(pd.Series(x).gt(0).mean())),
    ).sort_values("mean_abs_contribution", ascending=False).reset_index(drop=True)


def _daily_paired_ic(prediction):
    stock = prediction.groupby(["fold", "trade_date", "ts_code"], as_index=False).agg(
        target=("target", "first"), score_static=("score_static", "mean"),
        score_adaptive=("score_adaptive", "mean"),
    )
    rows = []
    for (fold, date), group in stock.groupby(["fold", "trade_date"], sort=True):
        if len(group) < 3:
            continue
        left = group["score_adaptive"].corr(group["target"], method="spearman")
        right = group["score_static"].corr(group["target"], method="spearman")
        rows.append({"fold": fold, "trade_date": date, "rank_ic_adaptive": left,
                     "rank_ic_static": right, "delta": left - right,
                     "stocks": len(group)})
    return pd.DataFrame(rows)


def _paired_stats(daily, horizon):
    delta = daily["delta"].dropna()
    if delta.empty:
        return {"dates": 0, "mean": None, "positive_ratio": None}
    return {
        "dates": int(len(delta)), "mean": float(delta.mean()),
        "positive_ratio": float(delta.gt(0).mean()),
        **_newey_west_stats(delta, max_lags=horizon - 1),
    }


def _validate_scan_contract(scan, templates, cfg):
    if scan["as_of_date"] != cfg.as_of_date:
        raise ValueError("scan and config as_of_date differ")
    quality_ids = {row["template_id"] for row in scan["events"] if row["quality_gate_passed"]}
    configured = {template.id for template in templates}
    if configured != quality_ids:
        raise ValueError(f"configured templates must equal quality-passing scan templates: {quality_ids}")


def _validate_live_sources(scan, live, templates, as_of):
    rows = {row["template_id"]: row for row in scan["events"]}
    for template in templates:
        source = pd.read_parquet(rows[template.id]["events_path"], columns=["trade_date", "ts_code"])
        expected = set(source.loc[pd.to_datetime(source["trade_date"]).eq(as_of), "ts_code"].astype(str))
        actual = set(live.loc[live["template_id"].eq(template.id), "ts_code"].astype(str))
        if expected != actual:
            raise ValueError(f"live source mismatch for {template.id}: {len(expected)} != {len(actual)}")


def _report(summary):
    stats = summary["paired_daily_rank_ic_delta"]
    lines = [
        "# Recent Anomaly Structure Ranker", "",
        f"- Scan: `{summary['scan_id']}` as of `{summary['as_of_date']}`",
        f"- OOS paired RankIC delta: `{stats.get('mean')}`",
        f"- Newey-West t: `{stats.get('nw_t_value')}`",
        f"- Decision: `{summary['gate']['decision']}`", "",
        "## Current event ranking", "",
    ]
    for row in summary["top_live"][:10]:
        lines.append(
            f"- `{row['instrument']}` rank={row['rank_adaptive']} "
            f"events={row['templates_triggered']} score={row['score_adaptive']:.6g}"
        )
    lines.extend(["", "> " + summary["interpretation_boundary"], ""])
    return "\n".join(lines)
