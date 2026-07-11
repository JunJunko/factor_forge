from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .models import ObservationArtifact, RadarScanResult


class ObservationWriter:
    def __init__(self, root: str | Path = "artifacts/radar_observations"):
        self.root = Path(root)

    def write(self, result: RadarScanResult) -> ObservationArtifact:
        card = result.card
        output = self.root / card.observation_id
        card_path = output / "observation_card.json"
        events_path = output / "events.parquet"
        manifest_path = output / "manifest.json"
        semantic = card.model_dump(mode="json")
        semantic.pop("discovered_at", None)
        semantic_sha = self._json_hash(semantic)
        if output.exists():
            if not card_path.exists() or not manifest_path.exists() or not events_path.exists():
                raise FileExistsError(f"incomplete immutable observation artifact: {output}")
            existing_card = json.loads(card_path.read_text(encoding="utf-8"))
            existing_card.pop("discovered_at", None)
            if self._json_hash(existing_card) != semantic_sha:
                raise FileExistsError(f"observation id collision with different content: {output}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if hashlib.sha256(card_path.read_bytes()).hexdigest() != manifest.get("card_sha256"):
                raise ValueError(f"cached observation card hash mismatch: {card_path}")
            if hashlib.sha256(events_path.read_bytes()).hexdigest() != manifest.get("events_sha256"):
                raise ValueError(f"cached observation events hash mismatch: {events_path}")
            existing_events = pd.read_parquet(events_path)
            if self._frame_hash(existing_events) != self._frame_hash(result.events):
                raise FileExistsError(f"observation id collision with different events: {output}")
            return ObservationArtifact(
                observation_id=card.observation_id, artifact_path=output,
                card_path=card_path, events_path=events_path, manifest_path=manifest_path,
                card_sha256=manifest["card_sha256"], event_count=int(manifest["event_count"]),
            )

        output.mkdir(parents=True, exist_ok=False)
        card_bytes = json.dumps(
            card.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8")
        card_path.write_bytes(card_bytes)
        result.events.to_parquet(events_path, index=False)
        events_sha = hashlib.sha256(events_path.read_bytes()).hexdigest()
        card_sha = hashlib.sha256(card_bytes).hexdigest()
        manifest = {
            "run_id": card.observation_id,
            "runner_type": "radar_observations",
            "status": "COMPLETED",
            "started_at": card.discovered_at,
            "finished_at": card.discovered_at,
            "data_version": card.data_version,
            "observation_id": card.observation_id,
            "observation_type": card.observation_type,
            "definition_id": card.definition.id,
            "definition_hash": card.definition.definition_hash,
            "as_of_date": card.as_of_date,
            "card_sha256": card_sha,
            "events_sha256": events_sha,
            "event_count": len(result.events),
            "contains_future_labels": False,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        return ObservationArtifact(
            observation_id=card.observation_id, artifact_path=output,
            card_path=card_path, events_path=events_path, manifest_path=manifest_path,
            card_sha256=card_sha, event_count=len(result.events),
        )

    @staticmethod
    def _json_hash(value: dict) -> str:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _frame_hash(frame: pd.DataFrame) -> str:
        digest = hashlib.sha256()
        digest.update("\x1f".join(map(str, frame.columns)).encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
        return digest.hexdigest()
