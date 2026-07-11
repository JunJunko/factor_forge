from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from conftest import make_panel
from factor_forge.data import DataVersionRepository
from factor_forge.radar.drift import RelationDriftRunner
from factor_forge.radar.drift_models import DriftCard
from factor_forge.radar.drift_templates import load_drift_template


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _project(tmp_path: Path, panel: pd.DataFrame) -> tuple[Path, str]:
    data_root = tmp_path / "data"
    metadata_db = tmp_path / "metadata.sqlite3"
    version = DataVersionRepository(data_root, metadata_db).publish(panel, source="test")
    project = tmp_path / "project.yaml"
    _write_yaml(project, {
        "project_name": "drift_test", "timezone": "Asia/Shanghai",
        "paths": {
            "data_root": str(data_root), "metadata_db": str(metadata_db),
            "artifacts_root": str(tmp_path / "factor_runs"),
        },
    })
    return project, version


def _base(kind: str) -> dict:
    return {
        "version": 1, "id": f"{kind}_test", "kind": kind,
        "entity": "market_relation",
        "windows": {"recent": 20, "medium": 60, "baseline": 120},
        "detector": {
            "method": "robust_delta_zscore" if kind == "feature_return_relation_drift" else "cusum",
            "threshold": 2.0, "min_persistence_days": 5,
        },
        "quality_gate": {"min_cross_section_size": 20, "min_valid_days_recent": 10},
        "universe_field": "is_liquid",
    }


def test_repository_contains_two_distinct_drift_templates():
    paths = sorted(Path("configs/drift").glob("*_v1.yaml"))
    assert len(paths) == 2
    templates = [load_drift_template(path) for path in paths]
    assert {item.kind for item in templates} == {
        "feature_return_relation_drift", "variable_relation_drift"
    }


def test_feature_return_drift_enforces_label_maturity_and_is_immutable(tmp_path):
    panel = make_panel(days=330, stocks=30)
    project, version = _project(tmp_path, panel)
    config = _base("feature_return_relation_drift")
    config.update({
        "relations": [{
            "id": "industry_relative_to_forward_5d",
            "predictor": "industry_relative_return_5d",
            "target_horizon": 5,
            "metric": "daily_rank_ic",
        }],
        "residualize_by": ["market_direction", "market_volatility", "market_breadth", "liquidity_regime"],
    })
    path = tmp_path / "feature_return.yaml"
    _write_yaml(path, config)
    output = tmp_path / "drifts"
    first = RelationDriftRunner().run(
        path, project_config=project, data_version=version, output_root=output,
    )
    second = RelationDriftRunner().run(
        path, project_config=project, data_version=version, output_root=output,
    )
    card = DriftCard.model_validate_json(
        (Path(first["artifact_path"]) / "drift_card.json").read_text(encoding="utf-8")
    )
    assert card.quality.label_maturity_enforced is True
    assert card.quality.future_incomplete_days_excluded >= 5
    assert pd.Timestamp(card.relations[0].effective_as_of_date) < pd.Timestamp(card.scan_date)
    assert second["cached"] is True


def test_variable_relation_drift_has_no_future_label_dependency(tmp_path):
    panel = make_panel(days=320, stocks=30)
    project, version = _project(tmp_path, panel)
    config = _base("variable_relation_drift")
    config["relations"] = [{
        "id": "stock_industry_relation",
        "x": "stock_return_1d", "y": "industry_return_1d", "metric": "daily_spearman",
    }]
    path = tmp_path / "variable.yaml"
    _write_yaml(path, config)
    result = RelationDriftRunner().run(
        path, project_config=project, data_version=version,
        output_root=tmp_path / "drifts",
    )
    card = DriftCard.model_validate_json(
        (Path(result["artifact_path"]) / "drift_card.json").read_text(encoding="utf-8")
    )
    assert card.quality.label_maturity_enforced is False
    assert card.quality.future_incomplete_days_excluded == 0
    assert card.relations[0].target == "industry_return_1d"

