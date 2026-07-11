from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DriftWindows(StrictModel):
    recent: int = Field(ge=20)
    medium: int = Field(ge=60)
    baseline: int = Field(ge=120)


class DriftDetector(StrictModel):
    method: Literal["robust_delta_zscore", "cusum"]
    threshold: float = Field(gt=0)
    min_persistence_days: int = Field(ge=5)


class DriftQualityGate(StrictModel):
    min_cross_section_size: int = Field(default=500, ge=20)
    min_valid_days_recent: int = Field(default=40, ge=10)


class FeatureReturnRelation(StrictModel):
    id: str
    predictor: Literal["lower_shadow_ratio", "volume_price_efficiency", "industry_relative_return_5d"]
    target_horizon: Literal[5, 10]
    metric: Literal["daily_rank_ic"] = "daily_rank_ic"


class VariableRelation(StrictModel):
    id: str
    x: Literal["turnover_rate", "stock_return_1d", "volatility_20d"]
    y: Literal["abs_return_1d", "industry_return_1d", "short_reversal_1d"]
    metric: Literal["daily_spearman"] = "daily_spearman"


class BaseDriftTemplate(StrictModel):
    version: Literal[1]
    id: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    entity: Literal["market_relation"]
    windows: DriftWindows
    detector: DriftDetector
    quality_gate: DriftQualityGate
    universe_field: Literal["is_liquid"] = "is_liquid"

    def definition_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class FeatureReturnDriftTemplate(BaseDriftTemplate):
    kind: Literal["feature_return_relation_drift"]
    relations: list[FeatureReturnRelation] = Field(min_length=1)
    residualize_by: list[Literal[
        "market_direction", "market_volatility", "market_breadth", "liquidity_regime"
    ]]


class VariableRelationDriftTemplate(BaseDriftTemplate):
    kind: Literal["variable_relation_drift"]
    relations: list[VariableRelation] = Field(min_length=1)


RelationDriftTemplate: TypeAlias = Annotated[
    FeatureReturnDriftTemplate | VariableRelationDriftTemplate,
    Field(discriminator="kind"),
]
DRIFT_TEMPLATE_ADAPTER = TypeAdapter(RelationDriftTemplate)


def load_drift_template(path: str | Path) -> RelationDriftTemplate:
    with Path(path).open("r", encoding="utf-8") as handle:
        return DRIFT_TEMPLATE_ADAPTER.validate_python(yaml.safe_load(handle))


def drift_required_trading_rows(template: RelationDriftTemplate) -> int:
    horizon = (
        max((r.target_horizon for r in template.relations), default=0)
        if isinstance(template, FeatureReturnDriftTemplate)
        else 0
    )
    return template.windows.baseline + template.windows.medium + template.windows.recent + horizon + 80
