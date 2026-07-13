from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data import DataVersionRepository
from factor_forge.evaluation.l1 import _newey_west_stats
from factor_forge.experiments.artifacts import json_default
from factor_forge.radar.templates import filter_required_fields, load_radar_template

from .event_factor_dataset import build_event_factor_dataset
from .event_factor_sensitivity_config import load_event_factor_sensitivity_config
from .event_mamba_trainer import encode_event_features, fit_event_mamba


ENGINE_VERSION = "event_factor_sensitivity_oof_v1"
ARMS = ("e0_controls", "e1_named_factors", "e2_oof_sensitivity", "e3_oof_embedding")


@dataclass(frozen=True)
class OOFBlock:
    block: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    oof_start: pd.Timestamp
    oof_end: pd.Timestamp


class EventFactorSensitivityRunner:
    def run(self, config_path: str | Path) -> dict:
        started = datetime.now(timezone.utc)
        config_path = Path(config_path)
        cfg = load_event_factor_sensitivity_config(config_path)
        scan = json.loads(cfg.scan_summary.read_text(encoding="utf-8"))
        templates = [load_radar_template(path) for path in cfg.event_templates]
        _validate_scan(scan, templates, cfg)
        project = load_project(cfg.project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        version, _ = repository.load_manifest(cfg.data_version)
        if version != scan["data_version"]:
            raise ValueError("scan and config data versions differ")
        as_of = pd.Timestamp(cfg.as_of_date)
        template_ids = {template.id for template in templates}
        source_events = {
            row["template_id"]: pd.read_parquet(row["events_path"])
            for row in scan["events"] if row["template_id"] in template_ids
        }
        earliest = min(pd.to_datetime(frame["trade_date"]).min() for frame in source_events.values())
        warmup = max(cfg.event.sequence_length, max(cfg.features.windows), 20)
        start = earliest - pd.Timedelta(days=math.ceil(warmup * 7 / 5) + 30)
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
        dataset = build_event_factor_dataset(
            panel, templates, cfg, as_of_date=as_of, source_events=source_events,
        )
        _validate_live_sources(scan, dataset.live_events, templates, as_of)

        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        identity = hashlib.sha256(json.dumps({
            "engine": ENGINE_VERSION, "data_version": version,
            "template_hashes": dataset.template_hashes, "config_hash": config_hash,
        }, sort_keys=True).encode()).hexdigest()[:16]
        run_id = f"event_factor_sensitivity_{identity}"
        output = cfg.output_root / run_id
        summary_path = output / "summary.json"
        if summary_path.exists():
            return {**json.loads(summary_path.read_text(encoding="utf-8")), "cached": True}
        output.mkdir(parents=True, exist_ok=False)
        (output / "oof_blocks").mkdir()
        calendar = dataset.calendar[-cfg.event.history_trading_days:]
        blocks = _build_blocks(calendar, cfg)
        if len(blocks) <= cfg.oof.evaluation_blocks:
            raise ValueError("OOF contract needs training blocks before evaluation blocks")
        oof_frames, block_audits = [], []
        for block in blocks:
            block_dir = output / "oof_blocks" / f"block_{block.block:02d}"
            block_dir.mkdir()
            frame, audit = _generate_oof_block(dataset, block, cfg, block_dir)
            oof_frames.append(frame)
            block_audits.append(audit)
        oof = pd.concat(oof_frames, ignore_index=True)
        oof.to_parquet(output / "historical_oof_features.parquet", index=False)
        oof_audit = _audit_oof(oof, blocks)
        (output / "oof_audit.json").write_text(json.dumps(
            oof_audit, ensure_ascii=False, indent=2, default=json_default
        ), encoding="utf-8")

        evaluation_ids = {block.block for block in blocks[-cfg.oof.evaluation_blocks:]}
        predictions = []
        feature_sets = _feature_sets(dataset, cfg)
        for block in blocks:
            if block.block not in evaluation_ids:
                continue
            current = oof.loc[oof["oof_block"].eq(block.block) & oof["target"].notna()].copy()
            prior = oof.loc[
                oof["oof_block"].lt(block.block)
                & oof["label_available_date"].le(block.oof_start)
                & oof["target"].notna()
            ].copy()
            if len(prior) < cfg.oof.minimum_prior_oof_rows:
                raise ValueError(f"block {block.block} has only {len(prior)} prior OOF rows")
            block_prediction = current[[
                "episode_id", "trade_date", "ts_code", "template_id", "target", "oof_block",
            ]].copy()
            for arm in ARMS:
                model = _fit_lgbm(prior, feature_sets[arm], cfg)
                block_prediction[f"score_{arm}"] = model.predict(current[feature_sets[arm]])
            predictions.append(block_prediction)
        prediction = pd.concat(predictions, ignore_index=True)
        prediction.to_parquet(output / "stacked_oos_predictions.parquet", index=False)
        daily = _daily_arm_ic(prediction)
        daily.to_parquet(output / "daily_rank_ic.parquet", index=False)
        deltas = {
            "e2_minus_e1": _delta_stats(daily, "e2_oof_sensitivity", "e1_named_factors", cfg),
            "e3_minus_e2": _delta_stats(daily, "e3_oof_embedding", "e2_oof_sensitivity", cfg),
        }
        (output / "paired_deltas.json").write_text(json.dumps(
            deltas, ensure_ascii=False, indent=2, default=json_default
        ), encoding="utf-8")
        live, live_audit = _fit_live(dataset, oof, cfg, output / "live", as_of)
        live.to_parquet(output / "live_event_ranking.parquet", index=False)
        live.to_csv(output / "live_event_ranking.csv", index=False, encoding="utf-8-sig")
        beta_summary = _beta_summary(oof, dataset.factor_names)
        beta_summary.to_csv(output / "oof_factor_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

        primary = deltas["e2_minus_e1"]
        passed = (
            primary.get("nw_t_value") is not None
            and primary["nw_t_value"] >= cfg.gate_min_nw_t
            and primary.get("positive_ratio", 0) > cfg.gate_min_positive_ratio
        )
        decision = "OBSERVE_FORWARD" if passed else "NO_VALIDATED_OOF_SENSITIVITY_INCREMENT"
        summary = {
            "run_id": run_id, "engine_version": ENGINE_VERSION,
            "idea_id": "idea_event_factor_sensitivity_oof_v1",
            "plan_id": "plan_event_factor_sensitivity_oof_v1",
            "data_version": version, "scan_id": scan["scan_id"], "as_of_date": cfg.as_of_date,
            "template_ids": dataset.template_ids, "factor_basis": dataset.factor_names,
            "historical_episode_rows": int(len(dataset.episodes)),
            "oof_rows": int(len(oof)), "oof_blocks": block_audits,
            "evaluation_blocks": sorted(evaluation_ids),
            "primary_metric": cfg.primary_metric, "paired_deltas": deltas,
            "gate": {"min_nw_t": cfg.gate_min_nw_t,
                     "min_positive_ratio": cfg.gate_min_positive_ratio,
                     "decision": decision},
            "oof_audit": oof_audit,
            "factor_sensitivity_summary": beta_summary.to_dict(orient="records"),
            "live_fit_audit": live_audit, "top_live": _json_records(live.head(20)),
            "interpretation_boundary": (
                "Historical Event-Mamba features are chronological OOF. A passed gate permits "
                "forward observation only; live ranks are not validated Alpha or trading instructions."
            ),
            "run_dir": str(output.resolve()), "cached": False,
        }
        summary_path.write_text(json.dumps(
            summary, ensure_ascii=False, indent=2, default=json_default
        ), encoding="utf-8")
        (output / "manifest.json").write_text(json.dumps({
            "run_id": run_id, "runner_type": "event_factor_sensitivity_runs",
            "status": "COMPLETED", "data_version": version, "scan_id": scan["scan_id"],
            "strict_oof_embeddings": True, "today_events_in_fit_rows": 0,
            "template_hashes": dataset.template_hashes,
            "started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary


def _build_blocks(calendar, cfg):
    dates = pd.DatetimeIndex(calendar)
    embargo = cfg.primary_horizon + 1
    first_oof = cfg.oof.training_days + cfg.oof.validation_days + 2 * embargo
    blocks, index, block_id = [], first_oof, 0
    while index + cfg.oof.block_days <= len(dates):
        valid_end_i = index - embargo - 1
        valid_start_i = valid_end_i - cfg.oof.validation_days + 1
        train_end_i = valid_start_i - embargo - 1
        train_start_i = train_end_i - cfg.oof.training_days + 1
        if train_start_i >= 0:
            blocks.append(OOFBlock(
                block=block_id, train_start=dates[train_start_i], train_end=dates[train_end_i],
                valid_start=dates[valid_start_i], valid_end=dates[valid_end_i],
                oof_start=dates[index], oof_end=dates[index + cfg.oof.block_days - 1],
            ))
            block_id += 1
        index += cfg.oof.block_days
    return blocks


def _generate_oof_block(dataset, block, cfg, output):
    events = dataset.episodes.copy()
    finite = events[[*dataset.raw_feature_names, *dataset.factor_names]].replace(
        [np.inf, -np.inf], np.nan
    ).notna().all(axis=1)
    train = events["trade_date"].between(block.train_start, block.train_end)
    valid = events["trade_date"].between(block.valid_start, block.valid_end)
    oof = events["trade_date"].between(block.oof_start, block.oof_end)
    common = finite & events["is_liquid"].eq(True)
    train &= common & events["target"].notna() & events["label_available_date"].lt(block.valid_start)
    valid &= common & events["target"].notna() & events["label_available_date"].lt(block.oof_start)
    oof &= common
    if train.sum() < cfg.oof.minimum_train_events or not valid.any() or not oof.any():
        raise ValueError(f"OOF block {block.block} has insufficient train/valid/oof rows")
    train_frame = _with_weights(events.loc[train].copy(), block.train_end)
    valid_frame = _with_weights(events.loc[valid].copy(), block.valid_end)
    fit = fit_event_mamba(
        dataset.store, train_frame, valid_frame, cfg,
        checkpoint_path=output / "event_mamba.pt",
    )
    oof_frame = events.loc[oof].copy().reset_index(drop=True)
    encoded = encode_event_features(fit.model, dataset.store, oof_frame, cfg, device=fit.device)
    oof_frame = _attach_encoded(oof_frame, encoded, dataset.factor_names)
    oof_frame["oof_block"] = block.block
    oof_frame["encoder_train_end"] = block.train_end
    oof_frame["encoder_valid_end"] = block.valid_end
    oof_frame = _standardize_residual_by_date(oof_frame)
    oof_frame.to_parquet(output / "oof_features.parquet", index=False)
    fit.history.to_parquet(output / "training_history.parquet", index=False)
    return oof_frame, {
        **{key: value for key, value in asdict(block).items()},
        "train_rows": int(train.sum()), "valid_rows": int(valid.sum()),
        "oof_rows": int(oof.sum()), "checkpoint_sha256": fit.checkpoint_sha256,
    }


def _fit_live(dataset, oof, cfg, output, as_of):
    output.mkdir()
    events = dataset.episodes.copy()
    finite = events[[*dataset.raw_feature_names, *dataset.factor_names]].replace(
        [np.inf, -np.inf], np.nan
    ).notna().all(axis=1)
    mature = finite & events["is_liquid"].eq(True) & events["target"].notna()
    mature &= events["label_available_date"].le(as_of)
    dates = pd.DatetimeIndex(sorted(events.loc[mature, "trade_date"].unique()))
    train_start = dates[max(0, len(dates) - cfg.oof.training_days - cfg.oof.validation_days)]
    valid_start = dates[max(1, len(dates) - cfg.oof.validation_days)]
    train = mature & events["trade_date"].between(train_start, valid_start, inclusive="left")
    valid = mature & events["trade_date"].ge(valid_start)
    train_frame = _with_weights(events.loc[train].copy(), valid_start)
    valid_frame = _with_weights(events.loc[valid].copy(), as_of)
    fit = fit_event_mamba(
        dataset.store, train_frame, valid_frame, cfg,
        checkpoint_path=output / "event_mamba.pt",
    )
    live = dataset.live_events.copy()
    live_finite = live[[*dataset.raw_feature_names, *dataset.factor_names]].replace(
        [np.inf, -np.inf], np.nan
    ).notna().all(axis=1) & live["is_liquid"].eq(True)
    live = live.loc[live_finite].copy().reset_index(drop=True)
    encoded = encode_event_features(fit.model, dataset.store, live, cfg, device=fit.device)
    live = _standardize_residual_by_date(_attach_encoded(live, encoded, dataset.factor_names))
    prior = oof.loc[oof["target"].notna() & oof["label_available_date"].le(as_of)].copy()
    feature_sets = _feature_sets(dataset, cfg)
    for arm in ARMS:
        model = _fit_lgbm(prior, feature_sets[arm], cfg)
        live[f"score_{arm}"] = model.predict(live[feature_sets[arm]])
    live["rank_e3"] = live["score_e3_oof_embedding"].rank(ascending=False, method="min").astype(int)
    live = live.sort_values("rank_e3").reset_index(drop=True)
    live.to_parquet(output / "event_rows.parquet", index=False)
    return live, {
        "event_mamba_train_rows": int(train.sum()),
        "event_mamba_valid_rows": int(valid.sum()),
        "lightgbm_oof_train_rows": int(len(prior)),
        "today_fit_rows": 0, "live_event_rows_scored": int(len(live)),
        "checkpoint_sha256": fit.checkpoint_sha256,
    }


def _feature_sets(dataset, cfg):
    beta = [f"beta_{name}" for name in dataset.factor_names]
    gated = [f"gated_{name}" for name in dataset.factor_names]
    residual = [f"event_embedding_{i:02d}" for i in range(cfg.event_mamba.residual_embedding_dim)]
    e0 = ["severity", *dataset.raw_feature_names, *dataset.template_feature_names]
    e1 = [*e0, *dataset.factor_names]
    e2 = [*e1, *beta, *gated, "event_mamba_prediction"]
    return {"e0_controls": e0, "e1_named_factors": e1,
            "e2_oof_sensitivity": e2, "e3_oof_embedding": [*e2, *residual]}


def _attach_encoded(frame, encoded, factors):
    result = frame.copy()
    result["event_mamba_prediction"] = encoded["prediction"]
    for index, factor in enumerate(factors):
        result[f"beta_{factor}"] = encoded["beta"][:, index]
        result[f"gated_{factor}"] = encoded["gated"][:, index]
    for index in range(encoded["residual_embedding"].shape[1]):
        result[f"event_embedding_{index:02d}"] = encoded["residual_embedding"][:, index]
    return result


def _standardize_residual_by_date(frame):
    result = frame.copy()
    columns = [name for name in result if name.startswith("event_embedding_")]
    for name in columns:
        group = result.groupby("trade_date")[name]
        result[name] = (result[name] - group.transform("mean")) / group.transform("std").replace(0, np.nan)
        result[name] = result[name].fillna(0.0)
    return result


def _with_weights(frame, reference):
    age = (pd.Timestamp(reference) - pd.to_datetime(frame["trade_date"])).dt.days
    date_count = frame.groupby("trade_date")["episode_id"].transform("count")
    duplicate = frame.groupby(["trade_date", "ts_code"])["episode_id"].transform("count")
    weight = np.power(0.5, age / 126.0) / date_count / duplicate
    frame["sample_weight"] = weight / weight.mean()
    return frame


def _fit_lgbm(train, features, cfg):
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(**cfg.lightgbm.model_dump(), verbosity=-1)
    weights = _with_weights(train.copy(), train["trade_date"].max())["sample_weight"]
    model.fit(train[features], train["target"], sample_weight=weights)
    return model


def _daily_arm_ic(prediction):
    score_columns = [f"score_{arm}" for arm in ARMS]
    stock = prediction.groupby(["oof_block", "trade_date", "ts_code"], as_index=False).agg(
        {"target": "first", **{column: "mean" for column in score_columns}}
    )
    rows = []
    for (block, date), group in stock.groupby(["oof_block", "trade_date"], sort=True):
        if len(group) < 3:
            continue
        row = {"oof_block": block, "trade_date": date, "stocks": len(group)}
        for arm in ARMS:
            row[f"rank_ic_{arm}"] = group[f"score_{arm}"].corr(group["target"], method="spearman")
        rows.append(row)
    return pd.DataFrame(rows)


def _delta_stats(daily, left, right, cfg):
    delta = (daily[f"rank_ic_{left}"] - daily[f"rank_ic_{right}"]).dropna()
    if delta.empty:
        return {"dates": 0, "mean": None, "positive_ratio": None}
    return {"dates": int(len(delta)), "mean": float(delta.mean()),
            "positive_ratio": float(delta.gt(0).mean()),
            **_newey_west_stats(delta, max_lags=cfg.primary_horizon - 1)}


def _beta_summary(oof, factors):
    rows = []
    for template, group in oof.groupby("template_id"):
        for factor in factors:
            beta = pd.to_numeric(group[f"beta_{factor}"], errors="coerce")
            rows.append({"template_id": template, "factor": factor,
                         "events": int(beta.notna().sum()), "mean_beta": float(beta.mean()),
                         "mean_abs_beta": float(beta.abs().mean()),
                         "positive_ratio": float(beta.gt(0).mean())})
    return pd.DataFrame(rows).sort_values(["template_id", "mean_abs_beta"], ascending=[True, False])


def _audit_oof(oof, blocks):
    violations = int((pd.to_datetime(oof["encoder_valid_end"]) >= pd.to_datetime(oof["trade_date"])).sum())
    duplicates = int(oof.duplicated(["episode_id", "oof_block"]).sum())
    return {"rows": int(len(oof)), "blocks": int(oof["oof_block"].nunique()),
            "encoder_date_violations": violations, "duplicate_episode_block_rows": duplicates,
            "passed": violations == 0 and duplicates == 0}


def _validate_scan(scan, templates, cfg):
    if scan["as_of_date"] != cfg.as_of_date:
        raise ValueError("scan and config as_of_date differ")
    quality = {row["template_id"] for row in scan["events"] if row["quality_gate_passed"]}
    configured = {template.id for template in templates}
    if quality != configured:
        raise ValueError("configured templates must equal quality-passing scan templates")


def _validate_live_sources(scan, live, templates, as_of):
    rows = {row["template_id"]: row for row in scan["events"]}
    for template in templates:
        source = pd.read_parquet(rows[template.id]["events_path"], columns=["trade_date", "ts_code"])
        expected = set(source.loc[pd.to_datetime(source["trade_date"]).eq(as_of), "ts_code"].astype(str))
        actual = set(live.loc[live["template_id"].eq(template.id), "ts_code"].astype(str))
        if expected != actual:
            raise ValueError(f"live source mismatch for {template.id}")


def finalize_incomplete_event_factor_run(run_dir: str | Path, config_path: str | Path) -> dict:
    """Finalize an already-computed run that failed only during JSON serialization."""
    output = Path(run_dir)
    cfg = load_event_factor_sensitivity_config(config_path)
    required = [
        "historical_oof_features.parquet", "stacked_oos_predictions.parquet",
        "daily_rank_ic.parquet", "paired_deltas.json", "oof_audit.json",
        "oof_factor_sensitivity_summary.csv", "live_event_ranking.parquet",
    ]
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        raise ValueError(f"incomplete run is missing computed artifacts: {missing}")
    if (output / "summary.json").exists():
        return json.loads((output / "summary.json").read_text(encoding="utf-8"))
    scan = json.loads(cfg.scan_summary.read_text(encoding="utf-8"))
    oof = pd.read_parquet(output / "historical_oof_features.parquet")
    prediction = pd.read_parquet(output / "stacked_oos_predictions.parquet")
    deltas = json.loads((output / "paired_deltas.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "oof_audit.json").read_text(encoding="utf-8"))
    beta = pd.read_csv(output / "oof_factor_sensitivity_summary.csv")
    live = pd.read_parquet(output / "live_event_ranking.parquet")
    primary = deltas["e2_minus_e1"]
    passed = (
        primary.get("nw_t_value") is not None
        and primary["nw_t_value"] >= cfg.gate_min_nw_t
        and primary.get("positive_ratio", 0) > cfg.gate_min_positive_ratio
    )
    checkpoints = sorted((output / "oof_blocks").glob("block_*/event_mamba.pt"))
    block_rows = oof.groupby("oof_block").size().to_dict()
    block_audits = [{
        "block": int(block), "oof_rows": int(rows),
        "checkpoint_sha256": hashlib.sha256(checkpoints[int(block)].read_bytes()).hexdigest()
        if int(block) < len(checkpoints) else None,
    } for block, rows in block_rows.items()]
    summary = {
        "run_id": output.name, "engine_version": ENGINE_VERSION,
        "idea_id": "idea_event_factor_sensitivity_oof_v1",
        "plan_id": "plan_event_factor_sensitivity_oof_v1",
        "data_version": cfg.data_version, "scan_id": scan["scan_id"],
        "as_of_date": cfg.as_of_date,
        "template_ids": [load_radar_template(path).id for path in cfg.event_templates],
        "factor_basis": cfg.event.factor_basis, "oof_rows": int(len(oof)),
        "oof_blocks": block_audits,
        "evaluation_blocks": sorted(map(int, prediction["oof_block"].unique())),
        "primary_metric": cfg.primary_metric, "paired_deltas": deltas,
        "gate": {"min_nw_t": cfg.gate_min_nw_t,
                 "min_positive_ratio": cfg.gate_min_positive_ratio,
                 "decision": "OBSERVE_FORWARD" if passed else "NO_VALIDATED_OOF_SENSITIVITY_INCREMENT"},
        "oof_audit": audit,
        "factor_sensitivity_summary": _json_records(beta),
        "live_fit_audit": {"today_fit_rows": 0, "live_event_rows_scored": int(len(live)),
                           "recovered_from_completed_numerical_artifacts": True},
        "top_live": _json_records(live.head(20)),
        "interpretation_boundary": (
            "Historical Event-Mamba features are chronological OOF. A passed gate permits "
            "forward observation only; live ranks are not validated Alpha or trading instructions."
        ),
        "run_dir": str(output.resolve()), "cached": False,
        "finalization_recovery": "NaT JSON serialization only; no numerical recomputation",
    }
    (output / "summary.json").write_text(json.dumps(
        summary, ensure_ascii=False, indent=2, default=json_default
    ), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "run_id": output.name, "runner_type": "event_factor_sensitivity_runs",
        "status": "COMPLETED", "data_version": cfg.data_version, "scan_id": scan["scan_id"],
        "strict_oof_embeddings": True, "today_events_in_fit_rows": 0,
        "finalized_from_existing_numerical_artifacts": True,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _json_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))
