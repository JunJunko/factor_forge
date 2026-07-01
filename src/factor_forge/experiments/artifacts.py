from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _json_default(value):
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)): return value.isoformat()
    if isinstance(value, Path): return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


class RunArtifacts:
    def __init__(self, root: str | Path, run_id: str):
        self.path = Path(root) / run_id
        self.path.mkdir(parents=True, exist_ok=False)

    def json(self, relative: str, value) -> None:
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    def yaml(self, relative: str, value) -> None:
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def parquet(self, relative: str, frame: pd.DataFrame) -> None:
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    def text(self, relative: str, value: str) -> None:
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
