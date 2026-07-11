from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data import DataVersionRepository
from factor_forge.evaluation.l1 import _forward_open_return
from factor_forge.research_control.models import utc_now
from factor_forge.research_control import ArtifactIndexer, ResearchControlStore

from .drift_models import DriftCard, DriftQuality, RelationDriftEvidence
from .drift_templates import (
    FeatureReturnDriftTemplate,
    RelationDriftTemplate,
    VariableRelationDriftTemplate,
    drift_required_trading_rows,
    load_drift_template,
)


class RelationDriftRunner:
    def run(
        self,
        template_path: str | Path,
        *,
        project_config: str | Path = "configs/project.yaml",
        data_version: str = "latest",
        as_of_date: str | None = None,
        output_root: str | Path = "artifacts/radar_drifts",
        research_db: str | Path | None = None,
    ) -> dict:
        template = load_drift_template(template_path)
        project = load_project(project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        resolved, manifest = repository.load_manifest(data_version)
        cutoff = pd.Timestamp(as_of_date or manifest["end_date"])
        drift_id = self._drift_id(template, resolved, cutoff)
        output = Path(output_root) / drift_id
        store = ResearchControlStore(research_db or project.paths.data_root / "research.sqlite3")
        store.initialize()
        cached = self._cached(output, template, resolved, cutoff)
        if cached is not None:
            cached_card_path = output / "drift_card.json"
            cached_card = DriftCard.model_validate_json(cached_card_path.read_text(encoding="utf-8"))
            store.register_drift_card(
                drift_id=cached_card.drift_id, template_id=cached_card.template_id,
                definition_hash=cached_card.definition_hash, data_version=cached_card.data_version,
                scan_date=cached_card.scan_date, artifact_path=output,
                card_sha256=hashlib.sha256(cached_card_path.read_bytes()).hexdigest(),
                discovered_at=cached_card.discovered_at, drift_count=cached_card.drift_count,
            )
            ArtifactIndexer(store, Path(output_root).parent).index()
            return cached
        rows_required = drift_required_trading_rows(template)
        calendar_days = math.ceil(rows_required * 7 / 5) + 120
        start = cutoff - pd.Timedelta(days=calendar_days)
        panel_path = project.paths.data_root / "versions" / resolved / "curated" / "stock_daily_panel.parquet"
        columns = [
            "trade_date", "ts_code", "adj_open", "adj_high", "adj_low", "adj_close",
            "amount_cny", "turnover_rate", "industry_l1_code", template.universe_field,
        ]
        panel = pd.read_parquet(
            panel_path, columns=columns,
            filters=[("trade_date", ">=", start), ("trade_date", "<=", cutoff)],
        )
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        panel = panel.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
        features, market = self._features(panel, template.universe_field)
        if isinstance(template, FeatureReturnDriftTemplate):
            card, series = self._feature_return_card(features, market, template, resolved, cutoff, drift_id)
        else:
            card, series = self._variable_relation_card(features, market, template, resolved, cutoff, drift_id)
        self._write(output, card, series)
        card_path = output / "drift_card.json"
        store.register_drift_card(
            drift_id=card.drift_id, template_id=card.template_id,
            definition_hash=card.definition_hash, data_version=card.data_version,
            scan_date=card.scan_date, artifact_path=output,
            card_sha256=hashlib.sha256(card_path.read_bytes()).hexdigest(),
            discovered_at=card.discovered_at, drift_count=card.drift_count,
        )
        ArtifactIndexer(store, Path(output_root).parent).index()
        return {
            "drift_id": drift_id, "artifact_path": str(output.resolve()),
            "template_id": template.id, "drift_count": card.drift_count,
            "relations": [item.model_dump(mode="json") for item in card.relations],
            "cached": False,
        }

    @staticmethod
    def _features(panel: pd.DataFrame, universe_field: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        data = panel.copy()
        close = pd.to_numeric(data["adj_close"], errors="coerce")
        grouped_close = close.groupby(data["ts_code"], sort=False)
        data["stock_return_1d"] = grouped_close.pct_change(1, fill_method=None)
        data["abs_return_1d"] = data["stock_return_1d"].abs()
        data["short_reversal_1d"] = -data["stock_return_1d"]
        data["volatility_20d"] = data["stock_return_1d"].groupby(
            data["ts_code"], sort=False
        ).transform(lambda x: x.rolling(20, min_periods=10).std(ddof=0))
        data["return_5d"] = grouped_close.pct_change(5, fill_method=None)
        open_ = pd.to_numeric(data["adj_open"], errors="coerce")
        high = pd.to_numeric(data["adj_high"], errors="coerce")
        low = pd.to_numeric(data["adj_low"], errors="coerce")
        span = (high - low).replace(0, np.nan)
        data["lower_shadow_ratio"] = (np.minimum(open_, close) - low) / span
        log_amount = np.log(pd.to_numeric(data["amount_cny"], errors="coerce").where(lambda x: x > 0))
        data["volume_price_efficiency"] = _daily_cross_section_residual(
            data["abs_return_1d"], log_amount, data["trade_date"], min_samples=100
        )
        valid = data["return_5d"].notna() & data["industry_l1_code"].notna()
        keys = [data["trade_date"], data["industry_l1_code"]]
        total = data["return_5d"].where(valid).groupby(keys).transform("sum")
        count = data["return_5d"].where(valid).groupby(keys).transform("count")
        industry_loo_5 = (total - data["return_5d"]) / (count - 1).replace(0, np.nan)
        data["industry_relative_return_5d"] = data["return_5d"] - industry_loo_5
        valid1 = data["stock_return_1d"].notna() & data["industry_l1_code"].notna()
        total1 = data["stock_return_1d"].where(valid1).groupby(keys).transform("sum")
        count1 = data["stock_return_1d"].where(valid1).groupby(keys).transform("count")
        data["industry_return_1d"] = (
            total1 - data["stock_return_1d"]
        ) / (count1 - 1).replace(0, np.nan)

        liquid = data.loc[data[universe_field].fillna(False).astype(bool)]
        daily = liquid.groupby("trade_date").agg(
            market_daily_return=("stock_return_1d", "mean"),
            market_breadth=("stock_return_1d", lambda x: float((x.dropna() > 0).mean()) if len(x.dropna()) else np.nan),
            market_amount=("amount_cny", "sum"),
        ).sort_index()
        daily["market_return_20"] = daily["market_daily_return"].rolling(20, min_periods=10).sum()
        daily["market_volatility"] = daily["market_daily_return"].rolling(20, min_periods=10).std(ddof=0)
        amount_log = np.log(daily["market_amount"].where(daily["market_amount"] > 0))
        daily["liquidity_regime"] = (
            amount_log - amount_log.shift(1).rolling(252, min_periods=60).mean()
        ) / amount_log.shift(1).rolling(252, min_periods=60).std(ddof=0)
        daily["market_direction"] = np.sign(daily["market_return_20"])
        return data, daily.reset_index()

    def _feature_return_card(self, data, market, template, version, cutoff, drift_id):
        eligible = data[template.universe_field].fillna(False).astype(bool)
        records, series_frames = [], []
        incomplete = 0
        for relation in template.relations:
            target = _forward_open_return(data, relation.target_horizon)
            frame = data.loc[eligible, ["trade_date", relation.predictor]].copy()
            frame["target"] = target.loc[eligible]
            daily = _daily_rank_correlation(
                frame, relation.predictor, "target", template.quality_gate.min_cross_section_size
            ).rename("relation_value").reset_index()
            mature_cutoff = daily.loc[daily["relation_value"].notna(), "trade_date"].max()
            if pd.isna(mature_cutoff):
                evidence = _empty_evidence(relation.id, relation.predictor,
                                           f"forward_return_{relation.target_horizon}", relation.metric, cutoff)
                records.append(evidence)
                continue
            incomplete = max(incomplete, int(data.loc[data["trade_date"].gt(mature_cutoff), "trade_date"].nunique()))
            daily = daily.merge(market, on="trade_date", how="left")
            residual = _regime_residualize(
                daily, template.windows.baseline, template.windows.recent,
                template.residualize_by,
            )
            daily["monitored_value"] = residual
            evidence = _summarize_relation(
                daily, relation.id, relation.predictor,
                f"forward_return_{relation.target_horizon}", relation.metric,
                template, regime_residualized=True,
            )
            records.append(evidence)
            daily["relation_id"] = relation.id
            series_frames.append(daily)
        return self._card(template, version, cutoff, drift_id, records, incomplete, True), _concat(series_frames)

    def _variable_relation_card(self, data, market, template, version, cutoff, drift_id):
        eligible = data[template.universe_field].fillna(False).astype(bool)
        records, series_frames = [], []
        for relation in template.relations:
            frame = data.loc[eligible, ["trade_date", relation.x, relation.y]].copy()
            daily = _daily_rank_correlation(
                frame, relation.x, relation.y, template.quality_gate.min_cross_section_size
            ).rename("relation_value").reset_index()
            daily["monitored_value"] = daily["relation_value"]
            evidence = _summarize_relation(
                daily, relation.id, relation.x, relation.y, relation.metric,
                template, regime_residualized=False,
            )
            records.append(evidence)
            daily["relation_id"] = relation.id
            series_frames.append(daily)
        return self._card(template, version, cutoff, drift_id, records, 0, False), _concat(series_frames)

    @staticmethod
    def _card(template, version, cutoff, drift_id, records, incomplete, labels):
        failures = [
            f"{item.relation_id}:valid_days_recent<{template.quality_gate.min_valid_days_recent}"
            for item in records if item.valid_days_recent < template.quality_gate.min_valid_days_recent
        ]
        return DriftCard(
            drift_id=drift_id, template_id=template.id, template_kind=template.kind,
            definition_hash=template.definition_hash(), discovered_at=utc_now(),
            data_version=version, scan_date=cutoff.strftime("%Y-%m-%d"),
            relations=records, drift_count=sum(item.is_drift for item in records),
            quality=DriftQuality(
                label_maturity_enforced=labels,
                future_incomplete_days_excluded=incomplete,
                temporal_audit_passed=True,
                quality_gate_passed=not failures,
                quality_gate_failures=failures,
            ),
        )

    @staticmethod
    def _drift_id(template, version, cutoff):
        digest = hashlib.sha256(
            f"{template.definition_hash()}|{version}|{cutoff:%Y-%m-%d}".encode("utf-8")
        ).hexdigest()[:16]
        return f"drift_{template.id}_{digest}"

    @staticmethod
    def _write(output: Path, card: DriftCard, series: pd.DataFrame):
        if output.exists():
            existing_card = output / "drift_card.json"
            existing_manifest = output / "manifest.json"
            existing_series = output / "relation_series.parquet"
            if all(path.exists() for path in (existing_card, existing_manifest, existing_series)):
                frozen = DriftCard.model_validate_json(existing_card.read_text(encoding="utf-8"))
                if (
                    frozen.drift_id == card.drift_id
                    and frozen.definition_hash == card.definition_hash
                    and frozen.data_version == card.data_version
                    and frozen.scan_date == card.scan_date
                ):
                    return
            raise FileExistsError(
                f"immutable drift artifact exists but is incomplete or has another identity: {output}"
            )
        output.mkdir(parents=True)
        card_path, series_path = output / "drift_card.json", output / "relation_series.parquet"
        card_path.write_text(card.model_dump_json(indent=2), encoding="utf-8")
        series.to_parquet(series_path, index=False)
        manifest = {
            "run_id": card.drift_id, "runner_type": "radar_drifts", "status": "COMPLETED",
            "data_version": card.data_version, "template_id": card.template_id,
            "definition_hash": card.definition_hash, "scan_date": card.scan_date,
            "drift_count": card.drift_count,
            "card_sha256": hashlib.sha256(card_path.read_bytes()).hexdigest(),
            "series_sha256": hashlib.sha256(series_path.read_bytes()).hexdigest(),
            "started_at": card.discovered_at, "finished_at": card.discovered_at,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _cached(output, template, version, cutoff):
        card_path, manifest_path, series_path = (
            output / "drift_card.json", output / "manifest.json", output / "relation_series.parquet"
        )
        if not all(path.exists() for path in (card_path, manifest_path, series_path)):
            return None
        card = DriftCard.model_validate_json(card_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            card.definition_hash != template.definition_hash()
            or card.data_version != version
            or card.scan_date != cutoff.strftime("%Y-%m-%d")
            or hashlib.sha256(card_path.read_bytes()).hexdigest() != manifest["card_sha256"]
            or hashlib.sha256(series_path.read_bytes()).hexdigest() != manifest["series_sha256"]
        ):
            raise ValueError(f"drift cache validation failed: {output}")
        return {
            "drift_id": card.drift_id, "artifact_path": str(output.resolve()),
            "template_id": template.id, "drift_count": card.drift_count,
            "relations": [item.model_dump(mode="json") for item in card.relations],
            "cached": True,
        }


def _daily_cross_section_residual(y, x, dates, min_samples):
    result = pd.Series(np.nan, index=y.index, dtype=float)
    frame = pd.DataFrame({"y": y, "x": x, "date": dates})
    for _, group in frame.groupby("date", sort=False):
        usable = group.dropna()
        if len(usable) < min_samples:
            continue
        matrix = np.column_stack([np.ones(len(usable)), usable["x"].to_numpy(float)])
        target = usable["y"].to_numpy(float)
        result.loc[usable.index] = target - matrix @ np.linalg.lstsq(matrix, target, rcond=None)[0]
    return result


def _daily_rank_correlation(frame, x, y, min_size):
    def correlation(group):
        usable = group[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        return usable[x].corr(usable[y], method="spearman") if len(usable) >= min_size else np.nan
    return frame.groupby("trade_date", sort=True).apply(correlation, include_groups=False)


def _regime_residualize(daily, baseline_days, recent_days, controls):
    usable = daily[["relation_value", *controls]].replace([np.inf, -np.inf], np.nan).dropna()
    baseline = usable.iloc[-(baseline_days + recent_days):-recent_days]
    result = pd.Series(np.nan, index=daily.index, dtype=float)
    if len(baseline) < max(60, len(controls) + 2):
        return daily["relation_value"]
    x = np.column_stack([np.ones(len(baseline)), baseline[controls].to_numpy(float)])
    beta = np.linalg.lstsq(x, baseline["relation_value"].to_numpy(float), rcond=None)[0]
    all_usable = daily[["relation_value", *controls]].replace([np.inf, -np.inf], np.nan).dropna()
    all_x = np.column_stack([np.ones(len(all_usable)), all_usable[controls].to_numpy(float)])
    result.loc[all_usable.index] = all_usable["relation_value"].to_numpy(float) - all_x @ beta
    return result


def _summarize_relation(daily, relation_id, predictor, target, metric, template, regime_residualized):
    values = daily.dropna(subset=["monitored_value"]).sort_values("trade_date")
    recent = values.tail(template.windows.recent)
    baseline = values.iloc[-(template.windows.baseline + template.windows.recent):-template.windows.recent]
    medium = values.tail(template.windows.medium)
    if recent.empty or baseline.empty:
        return _empty_evidence(relation_id, predictor, target, metric,
                               values["trade_date"].max() if len(values) else pd.Timestamp.today())
    base_values = baseline["monitored_value"]
    center = float(base_values.median())
    mad = float((base_values - center).abs().median()) * 1.4826
    scale = mad if np.isfinite(mad) and mad > 1e-12 else float(base_values.std(ddof=0))
    delta = float(recent["monitored_value"].mean() - base_values.mean())
    robust_z = delta / (scale / math.sqrt(len(recent))) if np.isfinite(scale) and scale > 0 else None
    standardized = (recent["monitored_value"] - float(base_values.mean())) / scale if scale and np.isfinite(scale) else pd.Series(dtype=float)
    cusum = float(standardized.cumsum().abs().max() / math.sqrt(len(standardized))) if len(standardized) else None
    rolling = recent["monitored_value"].rolling(10, min_periods=10).mean()
    threshold = scale / math.sqrt(10) if np.isfinite(scale) and scale > 0 else np.inf
    sign = 1 if delta >= 0 else -1
    condition = (rolling - float(base_values.mean())) * sign > threshold
    persistence = _trailing_true(condition)
    score = robust_z if template.detector.method == "robust_delta_zscore" else cusum
    is_drift = bool(
        score is not None and abs(score) >= template.detector.threshold
        and persistence >= template.detector.min_persistence_days
        and len(recent) >= template.quality_gate.min_valid_days_recent
    )
    return RelationDriftEvidence(
        relation_id=relation_id, predictor=predictor, target=target, metric=metric,
        effective_as_of_date=pd.Timestamp(values["trade_date"].max()).strftime("%Y-%m-%d"),
        baseline_days=len(baseline), recent_days=template.windows.recent,
        baseline_mean=float(base_values.mean()), medium_mean=float(medium["monitored_value"].mean()),
        recent_mean=float(recent["monitored_value"].mean()), delta=delta,
        robust_delta_zscore=robust_z, cusum_score=cusum,
        persistence_days=persistence, valid_days_recent=len(recent), is_drift=is_drift,
        direction="strengthening" if delta > 0 else "weakening" if delta < 0 else "none",
        regime_residualized=regime_residualized,
    )


def _empty_evidence(relation_id, predictor, target, metric, date):
    return RelationDriftEvidence(
        relation_id=relation_id, predictor=predictor, target=target, metric=metric,
        effective_as_of_date=pd.Timestamp(date).strftime("%Y-%m-%d"),
        baseline_days=0, recent_days=0,
    )


def _trailing_true(values):
    count = 0
    for value in reversed(values.fillna(False).tolist()):
        if not value:
            break
        count += 1
    return count


def _concat(frames):
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
