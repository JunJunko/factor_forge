from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from factor_forge.radar.scanner import RelationAnomalyScanner
from factor_forge.radar.templates import RadarTemplate, load_radar_template

from .config import FeatureConfig, LabelConfig
from .dataset import build_dataset


@dataclass(frozen=True)
class HistoricalEventChannels:
    frame: pd.DataFrame
    channel_names: list[str]
    template_hashes: dict[str, str]
    schema_hash: str


@dataclass(frozen=True)
class StateFeatureFrame:
    frame: pd.DataFrame
    raw_feature_names: list[str]
    state_feature_names: list[str]
    event_channel_names: list[str]
    template_hashes: dict[str, str]
    feature_schema_hash: str


def build_historical_event_channels(
    panel: pd.DataFrame,
    templates: Sequence[RadarTemplate],
    *,
    as_of_date: str | pd.Timestamp | None = None,
) -> HistoricalEventChannels:
    """Build dense label-free event channels from explicitly frozen templates."""
    if not templates:
        empty = panel[["trade_date", "ts_code"]].copy()
        empty["trade_date"] = pd.to_datetime(empty["trade_date"])
        return HistoricalEventChannels(empty, [], {}, _schema_hash({}, []))
    ids = [template.id for template in templates]
    if len(ids) != len(set(ids)):
        raise ValueError("historical event templates contain duplicate ids")
    for template in templates:
        if template.data.date_field != "trade_date" or template.data.entity_field != "ts_code":
            raise ValueError("Mamba pilot requires trade_date/ts_code radar template keys")

    base = panel[["trade_date", "ts_code"]].copy()
    base["trade_date"] = pd.to_datetime(base["trade_date"])
    base["ts_code"] = base["ts_code"].astype(str)
    if base.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("historical event panel has duplicate trade_date/ts_code rows")
    base = base.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)

    scanner = RelationAnomalyScanner()
    channels: list[str] = []
    hashes: dict[str, str] = {}
    result = base
    for template in templates:
        measured = scanner.measure_event_channels(panel, template, as_of_date=as_of_date)
        names = [column for column in measured.columns if column not in {"trade_date", "ts_code"}]
        result = result.merge(
            measured, on=["trade_date", "ts_code"], how="left", validate="one_to_one",
        )
        channels.extend(names)
        hashes[template.id] = template.definition_hash()
    forbidden = [name for name in result if name.lower().startswith(("forward_", "future_"))]
    if forbidden or any(name.lower() in {"label", "target"} for name in result):
        raise ValueError(f"future-label fields escaped into event channels: {forbidden}")
    return HistoricalEventChannels(
        frame=result,
        channel_names=channels,
        template_hashes=hashes,
        schema_hash=_schema_hash(hashes, channels),
    )


def build_state_feature_frame(
    panel: pd.DataFrame,
    features: FeatureConfig,
    label: LabelConfig,
    *,
    event_template_paths: Sequence[str | Path] = (),
    as_of_date: str | pd.Timestamp | None = None,
) -> StateFeatureFrame:
    """Build the flat PIT table consumed by sequence indexing and LightGBM."""
    dataset, raw_features = build_dataset(panel, features, label)
    templates = [load_radar_template(path) for path in event_template_paths]
    historical = build_historical_event_channels(panel, templates, as_of_date=as_of_date)
    events = historical.frame.rename(columns={"trade_date": "datetime", "ts_code": "instrument"})
    frame = dataset.merge(events, on=["datetime", "instrument"], how="left", validate="one_to_one")
    state_features = [*raw_features, *historical.channel_names]
    payload = {
        "raw_features": raw_features,
        "event_schema_hash": historical.schema_hash,
        "state_features": state_features,
        "feature_config": features.model_dump(mode="json"),
        "label_config": label.model_dump(mode="json"),
    }
    schema_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return StateFeatureFrame(
        frame=frame.sort_values(["instrument", "datetime"], kind="mergesort").reset_index(drop=True),
        raw_feature_names=raw_features,
        state_feature_names=state_features,
        event_channel_names=historical.channel_names,
        template_hashes=historical.template_hashes,
        feature_schema_hash=schema_hash,
    )


def _schema_hash(template_hashes: dict[str, str], channel_names: list[str]) -> str:
    return hashlib.sha256(json.dumps({
        "template_hashes": template_hashes,
        "channel_names": channel_names,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
