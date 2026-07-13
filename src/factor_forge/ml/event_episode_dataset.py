from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from factor_forge.event_study.labels import build_point_in_time_features_and_labels
from factor_forge.event_study.matching import match_episode_anchors
from factor_forge.radar.scanner import RelationAnomalyScanner

from .config import LabelConfig
from .dataset import build_dataset
from .mamba_state_dataset import SequenceStore, build_sequence_store


@dataclass(frozen=True)
class EventEpisodeDataset:
    store: SequenceStore
    episodes: pd.DataFrame
    live_events: pd.DataFrame
    matched_pairs: pd.DataFrame
    raw_feature_names: list[str]
    state_feature_names: list[str]
    event_channel_names: list[str]
    template_id: str
    definition_hash: str
    match_rate_primary: float


def deduplicate_event_episodes(
    channels: pd.DataFrame,
    *,
    template_id: str,
    definition_hash: str = "",
    dedup_trading_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_col, severity_col = f"{template_id}__event", f"{template_id}__severity"
    raw = channels.loc[channels[event_col].eq(1), ["trade_date", "ts_code", severity_col]].copy()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    raw = raw.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    calendar = pd.Index(sorted(pd.to_datetime(channels["trade_date"]).unique()))
    ordinal = pd.Series(np.arange(len(calendar), dtype=int), index=calendar)
    raw["trade_ordinal"] = raw["trade_date"].map(ordinal).astype(int)
    assignments = []
    for code, group in raw.groupby("ts_code", sort=False):
        anchor_date, anchor_ordinal = None, None
        episode_number = -1
        for row in group.itertuples(index=False):
            if anchor_ordinal is None or row.trade_ordinal - anchor_ordinal >= dedup_trading_days:
                anchor_date = pd.Timestamp(row.trade_date)
                anchor_ordinal = row.trade_ordinal
                episode_number += 1
            assignments.append({
                "trade_date": pd.Timestamp(row.trade_date), "ts_code": str(code),
                "anchor_date": anchor_date, "episode_number": episode_number,
            })
    assigned = raw.merge(pd.DataFrame(assignments), on=["trade_date", "ts_code"], validate="one_to_one")
    if assigned.empty:
        return assigned, pd.DataFrame(columns=["trade_date", "ts_code", "severity"])
    grouped = assigned.groupby(["ts_code", "anchor_date", "episode_number"], as_index=False).agg(
        severity=(severity_col, "first"),
        diagnostic_max_severity=(severity_col, "max"),
        diagnostic_trigger_count=("trade_date", "size"),
        diagnostic_last_trigger_date=("trade_date", "max"),
    )
    grouped = grouped.rename(columns={"anchor_date": "trade_date"})
    grouped["episode_id"] = grouped.apply(
        lambda row: "episode_" + hashlib.sha256(
            f"{definition_hash}|{template_id}|{row.ts_code}|{pd.Timestamp(row.trade_date):%Y-%m-%d}".encode()
        ).hexdigest()[:16], axis=1,
    )
    anchors = grouped[[
        "episode_id", "trade_date", "ts_code", "severity",
        "diagnostic_max_severity", "diagnostic_trigger_count", "diagnostic_last_trigger_date",
    ]].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    return assigned, anchors


def build_event_episode_dataset(panel, template, config, *, as_of_date) -> EventEpisodeDataset:
    panel = panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    channels = RelationAnomalyScanner().measure_event_channels(
        panel, template, as_of_date=as_of_date
    )
    _, anchors_all = deduplicate_event_episodes(
        channels, template_id=template.id,
        definition_hash=template.definition_hash(),
        dedup_trading_days=config.episode.dedup_trading_days,
    )
    dates = sorted(pd.to_datetime(panel["trade_date"]).unique())
    output_dates = set(dates[-config.episode.history_trading_days:])
    anchors = anchors_all.loc[anchors_all["trade_date"].isin(output_dates)].copy()

    enriched = build_point_in_time_features_and_labels(panel, config.horizons)
    raw_keys = channels.loc[channels[f"{template.id}__event"].eq(1), ["trade_date", "ts_code"]]
    anchor_keys = anchors[["trade_date", "ts_code", "severity"]]
    enriched = enriched.merge(
        raw_keys.assign(is_raw_event=True), on=["trade_date", "ts_code"], how="left",
        validate="one_to_one",
    ).merge(
        anchor_keys.assign(is_episode_anchor=True), on=["trade_date", "ts_code"], how="left",
        validate="one_to_one",
    )
    enriched["is_raw_event"] = enriched["is_raw_event"].eq(True)
    enriched["is_episode_anchor"] = enriched["is_episode_anchor"].eq(True)
    enriched["severity"] = pd.to_numeric(enriched["severity"], errors="coerce")
    pairs = match_episode_anchors(enriched, config.matching, config.horizons)
    labels = _aggregate_matched_labels(pairs, config.horizons)
    episodes = anchors.merge(labels, on=["trade_date", "ts_code"], how="left", validate="one_to_one")

    flat, raw_features = build_dataset(
        panel, config.features,
        LabelConfig(horizon=config.primary_horizon, price="adj_open", excess_over_universe=True),
    )
    event_columns = [
        f"{template.id}__eligible", f"{template.id}__event",
        f"{template.id}__severity", f"{template.id}__valid",
    ]
    event_frame = channels.rename(columns={"trade_date": "datetime", "ts_code": "instrument"})
    flat = flat.merge(
        event_frame[["datetime", "instrument", *event_columns]],
        on=["datetime", "instrument"], how="left", validate="one_to_one",
    )
    state_features = [*raw_features, *event_columns]
    store = build_sequence_store(
        flat, state_features, length=config.episode.sequence_length,
        min_valid_days=config.episode.min_valid_days, validity_feature_names=raw_features,
    )
    sample_lookup = store.samples.reset_index().set_index(["datetime", "instrument"])["index"]
    episodes["sample_position"] = [
        sample_lookup.get((pd.Timestamp(date), str(code)), np.nan)
        for date, code in episodes[["trade_date", "ts_code"]].itertuples(index=False, name=None)
    ]
    episodes = episodes.dropna(subset=["sample_position"]).copy()
    episodes["sample_position"] = episodes["sample_position"].astype(int)

    live = channels.loc[
        channels["trade_date"].eq(pd.Timestamp(as_of_date))
        & channels[f"{template.id}__event"].eq(1),
        ["trade_date", "ts_code", f"{template.id}__severity"],
    ].rename(columns={f"{template.id}__severity": "severity"})
    live["sample_position"] = [
        sample_lookup.get((pd.Timestamp(date), str(code)), np.nan)
        for date, code in live[["trade_date", "ts_code"]].itertuples(index=False, name=None)
    ]
    live = live.dropna(subset=["sample_position"]).copy()
    live["sample_position"] = live["sample_position"].astype(int)
    raw_count = len(anchors)
    matched_count = int(episodes[f"matched_excess_{config.primary_horizon}"].notna().sum())
    return EventEpisodeDataset(
        store=store, episodes=episodes, live_events=live, matched_pairs=pairs,
        raw_feature_names=raw_features, state_feature_names=state_features,
        event_channel_names=event_columns, template_id=template.id,
        definition_hash=template.definition_hash(),
        match_rate_primary=matched_count / raw_count if raw_count else 0.0,
    )


def _aggregate_matched_labels(pairs: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    frames = []
    for horizon in horizons:
        sample = pairs.loc[pairs["horizon"].eq(horizon)].copy() if len(pairs) else pd.DataFrame()
        if sample.empty:
            continue
        grouped = sample.groupby(["trade_date", "event_code"], as_index=False).agg(
            event_return=("event_return", "first"), control_return=("control_return", "mean"),
            control_count=("control_code", "nunique"), mean_match_distance=("match_distance", "mean"),
        ).rename(columns={"event_code": "ts_code"})
        grouped[f"matched_excess_{horizon}"] = grouped["event_return"] - grouped["control_return"]
        grouped = grouped.rename(columns={
            "event_return": f"event_return_{horizon}",
            "control_return": f"control_return_{horizon}",
            "control_count": f"control_count_{horizon}",
            "mean_match_distance": f"mean_match_distance_{horizon}",
        })
        frames.append(grouped)
    if not frames:
        return pd.DataFrame(columns=["trade_date", "ts_code"])
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on=["trade_date", "ts_code"], how="outer", validate="one_to_one")
    return result
