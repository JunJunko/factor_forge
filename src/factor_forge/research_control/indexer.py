from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .models import IndexedArtifact, utc_now
from .store import ResearchControlStore


class ResearchRunEnvelope(BaseModel):
    """Small common projection over heterogeneous immutable run manifests."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    runner_type: str
    status: str
    manifest_path: Path
    artifact_path: Path
    manifest_sha256: str
    data_version: str | None = None
    code_version: str | None = None
    factor_name: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    idea_id: str | None = None

    def indexed(self) -> IndexedArtifact:
        payload = self.model_dump(mode="json")
        payload.update({
            "manifest_path": str(self.manifest_path.resolve()),
            "artifact_path": str(self.artifact_path.resolve()),
            "indexed_at": utc_now(),
        })
        return IndexedArtifact.model_validate(payload)


class ArtifactIndexer:
    def __init__(self, store: ResearchControlStore, artifacts_root: str | Path = "artifacts"):
        self.store = store
        self.artifacts_root = Path(artifacts_root)

    def index(self) -> dict:
        manifests = sorted(self.artifacts_root.rglob("manifest.json")) if self.artifacts_root.exists() else []
        indexed = 0
        unchanged = 0
        errors: list[dict] = []
        for path in manifests:
            try:
                envelope = self.read_envelope(path)
                existing_hash = self._existing_hash(path)
                if existing_hash == envelope.manifest_sha256:
                    unchanged += 1
                    continue
                self.store.upsert_artifact(envelope.indexed())
                indexed += 1
            except Exception as exc:
                errors.append({"manifest": str(path.resolve()), "error": str(exc)})
        return {
            "artifacts_root": str(self.artifacts_root.resolve()),
            "found": len(manifests),
            "indexed": indexed,
            "unchanged": unchanged,
            "errors": errors,
            "summary": self.store.artifact_summary(),
        }

    def read_envelope(self, path: str | Path) -> ResearchRunEnvelope:
        path = Path(path)
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("manifest root must be an object")
        relative = path.resolve().relative_to(self.artifacts_root.resolve())
        runner_type = relative.parts[0] if len(relative.parts) > 1 else "artifacts"
        research = payload.get("research") if isinstance(payload.get("research"), dict) else {}
        return ResearchRunEnvelope(
            run_id=str(payload.get("run_id") or path.parent.name),
            runner_type=str(payload.get("runner_type") or runner_type),
            status=str(payload.get("status") or "UNKNOWN"),
            manifest_path=path,
            artifact_path=path.parent,
            manifest_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            data_version=self._first(payload, "data_version", "dataset_version"),
            code_version=self._first(payload, "code_version", "git_commit"),
            factor_name=self._first(payload, "factor_name", "model_name", "name"),
            started_at=self._first(payload, "started_at", "created_at"),
            finished_at=self._first(payload, "finished_at", "completed_at"),
            idea_id=research.get("idea_id") or payload.get("idea_id"),
        )

    def _existing_hash(self, path: Path) -> str | None:
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT manifest_sha256 FROM artifact_index WHERE manifest_path=?",
                (str(path.resolve()),),
            ).fetchone()
        return row[0] if row else None

    @staticmethod
    def _first(payload: dict, *keys: str) -> str | None:
        for key in keys:
            value = payload.get(key)
            if value is not None and not isinstance(value, (dict, list)):
                return str(value)
        return None
