from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from factor_forge.exceptions import FactorCombinationError


ENGINE_VERSION = "factor-engine-v1"


class AtomicFactorCache:
    """Content-addressed cache containing raw atomic factor output only."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @staticmethod
    def yaml_hash(path: str | Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def key(self, path: str | Path, context: dict) -> tuple[str, dict]:
        metadata = {
            "factor_yaml_hash": self.yaml_hash(path),
            "data_version": str(context.get("data_version", "unknown")),
            "start_date": str(context.get("start_date", "")),
            "end_date": str(context.get("end_date", "")),
            "adjustment_mode": str(context.get("adjustment_mode", "forward_adjusted")),
            "base_universe": str(context.get("base_universe", "default")),
            "membership_version": str(context.get("membership_version", context.get("data_version", "unknown"))),
            "factor_engine_version": ENGINE_VERSION,
        }
        payload = json.dumps(metadata, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest(), metadata

    def load(self, key: str, expected: dict) -> pd.DataFrame | None:
        data_path, meta_path = self.root / f"{key}.parquet", self.root / f"{key}.json"
        if not data_path.exists() and not meta_path.exists():
            return None
        if not data_path.exists() or not meta_path.exists():
            raise FactorCombinationError("Atomic factor cache is incomplete; remove the damaged cache entry and rerun")
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            frame = pd.read_parquet(data_path)
        except Exception as exc:
            raise FactorCombinationError("Atomic factor cache is corrupt; remove the damaged cache entry and rerun") from exc
        if metadata != expected:
            return None
        self.validate(frame)
        return frame

    def save(self, key: str, metadata: dict, frame: pd.DataFrame) -> None:
        self.validate(frame)
        self.root.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(self.root / f"{key}.parquet", index=False)
        (self.root / f"{key}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @staticmethod
    def validate(frame: pd.DataFrame) -> None:
        required = {"trade_date", "ts_code", "factor_value"}
        if not required <= set(frame):
            raise FactorCombinationError(f"Atomic factor output/cache is missing columns: {sorted(required - set(frame))}")
        if frame.duplicated(["trade_date", "ts_code"]).any():
            raise FactorCombinationError("Atomic factor output/cache has duplicate trade_date + ts_code keys")
