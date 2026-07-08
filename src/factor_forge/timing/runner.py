from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from factor_forge.experiments.artifacts import json_default
from .config import TimingBuildConfig, load_timing_build_config
from .dataset import TimingInputData, build_timing_dataset


class TimingDatasetBuildRunner:
    def run(self, config_path: str | Path) -> dict:
        config_path = Path(config_path)
        cfg = load_timing_build_config(config_path)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + hashlib.sha256(config_path.read_bytes()).hexdigest()[:8]
        output = cfg.output_dir / f"{cfg.name}_{run_id}"
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.yaml").write_bytes(config_path.read_bytes())
        inputs = self._load_inputs(cfg)
        result = build_timing_dataset(inputs, cfg.features.to_dataclass())
        dataset_path = output / "timing_dataset.parquet"
        features_path = output / "feature_names.json"
        groups_path = output / "feature_groups.json"
        result.dataset.to_parquet(dataset_path, index=False)
        features_path.write_text(json.dumps(result.feature_names, ensure_ascii=False, indent=2), encoding="utf-8")
        groups_path.write_text(json.dumps(result.feature_groups, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = {
            "status": "SUCCESS",
            "run_dir": str(output),
            "dataset_path": str(dataset_path),
            "rows": len(result.dataset),
            "columns": len(result.dataset.columns),
            "feature_count": len(result.feature_names),
            "label_name": result.label_name,
            "feature_groups": {key: len(value) for key, value in result.feature_groups.items()},
            "date_start": result.dataset["trade_date"].min(),
            "date_end": result.dataset["trade_date"].max(),
        }
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
        return summary

    @staticmethod
    def _load_inputs(cfg: TimingBuildConfig) -> TimingInputData:
        values = {}
        for field, path in cfg.inputs.model_dump().items():
            if path is None:
                values[field] = None
                continue
            if field != "index_daily" and not Path(path).exists():
                values[field] = None
                continue
            values[field] = _read_table(path)
        return TimingInputData(**values)


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"input table does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"unsupported table format for {path}; use parquet/csv/xlsx")
