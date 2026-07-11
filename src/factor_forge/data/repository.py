from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from factor_forge.exceptions import DataQualityError
from .metadata import MetadataStore
from .quality import DataQualityValidator, has_blocking_issues


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_complete_manifest(manifest: dict) -> bool:
    """Recognize full-history versions, including manifests created before this field existed."""
    kind = manifest.get("version_kind")
    if kind is not None:
        return kind == "complete"
    return (
        int(manifest.get("row_count", 0)) > 1_000_000
        and str(manifest.get("start_date", "9999-12-31")) <= "2017-01-01"
    )


class DataVersionRepository:
    def __init__(self, data_root: str | Path, metadata_db: str | Path):
        self.root = Path(data_root)
        self.metadata = MetadataStore(metadata_db)
        self.metadata.initialize()

    def publish(
        self,
        panel: pd.DataFrame,
        raw_datasets: dict[str, pd.DataFrame] | None = None,
        source: str = "tushare",
        version_kind: str = "complete",
    ) -> str:
        if version_kind not in {"complete", "incremental"}:
            raise ValueError("version_kind must be 'complete' or 'incremental'")
        issues = DataQualityValidator().validate(panel)
        if has_blocking_issues(issues):
            detail = "; ".join(f"{item.rule_name}: {item.detail}" for item in issues)
            raise DataQualityError(f"Data version not published: {detail}")
        now = datetime.now(timezone.utc)
        ordered = panel.sort_values(["trade_date", "ts_code"])
        content_digest = hashlib.sha256()
        chunk_size = 250_000
        for start in range(0, len(ordered), chunk_size):
            hashes = pd.util.hash_pandas_object(
                ordered.iloc[start:start + chunk_size], index=False
            ).values
            content_digest.update(hashes.tobytes())
        suffix = content_digest.hexdigest()[:8]
        version = f"data_v1_{now:%Y%m%dT%H%M%SZ}_{suffix}"
        final_dir = self.root / "versions" / version
        if final_dir.exists():
            raise DataQualityError(f"Immutable data version already exists: {version}")
        temporary = self.root / "versions" / f".{version}.{uuid4().hex}.tmp"
        try:
            curated = temporary / "curated"
            curated.mkdir(parents=True, exist_ok=False)
            panel_path = curated / "stock_daily_panel.parquet"
            ordered.to_parquet(panel_path, index=False)
            files = {panel_path.relative_to(temporary).as_posix(): sha256_file(panel_path)}
            for name, frame in (raw_datasets or {}).items():
                path = temporary / "raw" / source / f"{name}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(path, index=False)
                files[path.relative_to(temporary).as_posix()] = sha256_file(path)
            dates = pd.to_datetime(panel["trade_date"])
            manifest = {
                "data_version": version,
                "contract_version": "1.0.0",
                "created_at": now.isoformat(),
                "start_date": dates.min().date().isoformat(),
                "end_date": dates.max().date().isoformat(),
                "row_count": len(panel),
                "source": source,
                "version_kind": version_kind,
                "files": files,
                "quality_issues": [item.to_dict() for item in issues],
            }
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest_hash = sha256_file(manifest_path)
            temporary.rename(final_dir)
            with self.metadata.connect() as connection:
                connection.execute(
                    "INSERT INTO meta_data_version VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (version, now.isoformat(), manifest["start_date"], manifest["end_date"],
                     str(final_dir / "manifest.json"), manifest_hash, "PASSED"),
                )
                for issue in issues:
                    connection.execute(
                        "INSERT INTO meta_quality_issue(data_version, rule_name, severity, detail) VALUES (?, ?, ?, ?)",
                        (version, issue.rule_name, issue.severity, issue.detail),
                    )
            return version
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    def resolve(self, version: str) -> str:
        if version == "latest":
            resolved = self.latest_complete_version()
            if not resolved:
                raise FileNotFoundError("No complete published data version exists")
            return resolved
        if version == "latest_any":
            resolved = self.metadata.latest_version()
            if not resolved:
                raise FileNotFoundError("No published data version exists")
            return resolved
        return version

    def latest_complete_version(self) -> str | None:
        with self.metadata.connect() as connection:
            rows = connection.execute(
                "SELECT data_version, manifest_path FROM meta_data_version "
                "WHERE quality_status='PASSED' ORDER BY created_at DESC"
            ).fetchall()
        for row in rows:
            path = Path(row["manifest_path"])
            if not path.exists():
                continue
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if is_complete_manifest(manifest):
                return str(row["data_version"])
        return None

    def load_panel(self, version: str = "latest") -> tuple[str, pd.DataFrame]:
        resolved = self.resolve(version)
        path = self.root / "versions" / resolved / "curated" / "stock_daily_panel.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing panel for {resolved}: {path}")
        return resolved, pd.read_parquet(path)

    def load_manifest(self, version: str = "latest") -> tuple[str, dict]:
        resolved = self.resolve(version)
        path = self.root / "versions" / resolved / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing manifest for {resolved}: {path}")
        return resolved, json.loads(path.read_text(encoding="utf-8"))

    def load_raw_dataset(self, version: str, name: str, source: str = "tushare") -> pd.DataFrame | None:
        resolved = self.resolve(version)
        path = self.root / "versions" / resolved / "raw" / source / f"{name}.parquet"
        return pd.read_parquet(path) if path.exists() else None
