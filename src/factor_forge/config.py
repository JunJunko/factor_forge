from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field, model_validator


class BaseModel(PydanticBaseModel):
    """Project-wide strict configuration model.

    A misspelled YAML key must fail validation instead of silently selecting a
    default value.
    """

    model_config = ConfigDict(extra="forbid")


class PathsConfig(BaseModel):
    data_root: Path = Path("data")
    metadata_db: Path = Path("data/metadata.sqlite3")
    artifacts_root: Path = Path("artifacts/runs")


class LiquidityConfig(BaseModel):
    window: int = 20
    min_avg_amount_cny: float = 30_000_000
    min_traded_days: int = 18
    max_participation_rate: float = 0.02


class ProjectDataConfig(BaseModel):
    start_date: str = "20150101"
    exchanges: list[str] = Field(default_factory=lambda: ["SSE", "SZSE"])
    boards: list[str] = Field(default_factory=lambda: ["main", "chinext", "star"])
    industry_standard: str = "SW2021"
    industry_level: str = "L1"
    industry_min_group_size: int = 10
    listing_age_days: int = 60
    liquidity: LiquidityConfig = Field(default_factory=LiquidityConfig)


class ProjectConfig(BaseModel):
    project_name: str = "factor_forge"
    timezone: str = "Asia/Shanghai"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: ProjectDataConfig = Field(default_factory=ProjectDataConfig)


class FactorMeta(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str
    description: str
    hypothesis: str
    direction: Literal["positive", "negative", "unknown"]
    expected_shape: Literal["monotonic", "top_tail", "bottom_tail", "unknown"] = "unknown"


class FactorData(BaseModel):
    frequency: Literal["daily"] = "daily"
    required_fields: list[str]
    lookback_days: int = Field(ge=0)


class FactorScope(BaseModel):
    universe: Literal["default", "tradeable", "liquid"] = "default"
    cross_section: Literal["market", "industry"] = "market"
    group_field: str = "industry_l1_code"
    industry_standard: str = "SW2021"
    industry_level: Literal["L1", "L2"] = "L1"
    point_in_time_required: bool = True
    min_group_size: int = Field(default=10, ge=2)


class RobustParameter(BaseModel):
    value: int | float
    robustness_neighbors: list[int | float] = Field(default_factory=list)


class FactorCalculation(BaseModel):
    features: dict[str, str] = Field(default_factory=dict)
    formula: str
    parameters: dict[str, RobustParameter] = Field(default_factory=dict)
    missing_policy: Literal["skip"] = "skip"
    winsorize: Literal["none", "mad"] = "none"
    mad_scale: float = 5.0
    standardize: Literal["none", "zscore"] = "none"


class FactorOutput(BaseModel):
    value_field: Literal["factor_value"] = "factor_value"


class FactorSpec(BaseModel):
    version: int = 1
    kind: Literal["factor"] = "factor"
    factor: FactorMeta
    data: FactorData
    scope: FactorScope
    calculation: FactorCalculation
    output: FactorOutput = Field(default_factory=FactorOutput)


class CombinationSource(BaseModel):
    type: Literal["yaml"] = "yaml"
    path: Path


class CombinationComponent(BaseModel):
    id: str = Field(min_length=1)
    source: CombinationSource
    direction: Literal["positive", "negative"] = "positive"
    weight: float = 1.0


class WinsorizeConfig(BaseModel):
    enabled: bool = False
    method: Literal["percentile"] = "percentile"
    lower: float = 0.01
    upper: float = 0.99

    @model_validator(mode="after")
    def valid_quantiles(self):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("winsorize requires 0 <= lower < upper <= 1")
        return self


class NormalizationConfig(BaseModel):
    method: Literal["cs_zscore", "cs_rank", "cs_percentile"] = "cs_zscore"
    scope: Literal["trade_date"] = "trade_date"
    min_samples: int = Field(default=2, ge=2)


class MissingValueConfig(BaseModel):
    method: Literal["intersection", "require_minimum_components", "zero_after_normalization"] = "require_minimum_components"
    minimum_valid_components: int = Field(default=2, ge=1)
    missing_score_after_normalization: float = 0.0


class CombinationPreprocessing(BaseModel):
    winsorize: WinsorizeConfig = Field(default_factory=WinsorizeConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    missing_value: MissingValueConfig = Field(default_factory=MissingValueConfig)


class CombinationMethod(BaseModel):
    method: Literal["weighted_sum", "equal_weight"] = "weighted_sum"
    normalize_weights: bool = True


class FilterCondition(BaseModel):
    operator: Literal["gt", "gte", "lt", "lte"]
    value: float


class FilterAction(BaseModel):
    type: Literal["exclude", "score_penalty"]
    value: float | None = None

    @model_validator(mode="after")
    def penalty_has_value(self):
        if self.type == "score_penalty" and self.value is None:
            raise ValueError("score_penalty requires action.value")
        return self


class CombinationFilter(BaseModel):
    id: str = Field(min_length=1)
    source: CombinationSource
    preprocessing: CombinationPreprocessing = Field(default_factory=CombinationPreprocessing)
    condition: FilterCondition
    action: FilterAction


class CombinationVariant(BaseModel):
    id: str = Field(min_length=1)
    components: list[str] = Field(min_length=1)
    filters: list[str] = Field(default_factory=list)


class CombinationOutput(BaseModel):
    direction: Literal["positive"] = "positive"
    column: Literal["factor_value"] = "factor_value"


class FactorCombinationBody(BaseModel):
    id: str = Field(min_length=1)
    name: str
    description: str = ""
    components: list[CombinationComponent] = Field(min_length=2)
    preprocessing: CombinationPreprocessing = Field(default_factory=CombinationPreprocessing)
    combination: CombinationMethod = Field(default_factory=CombinationMethod)
    filters: list[CombinationFilter] = Field(default_factory=list)
    variants: list[CombinationVariant] = Field(default_factory=list)
    output: CombinationOutput = Field(default_factory=CombinationOutput)

    @model_validator(mode="after")
    def validate_references(self):
        component_ids = [item.id for item in self.components]
        filter_ids = [item.id for item in self.filters]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("component ids must be unique")
        if len(filter_ids) != len(set(filter_ids)):
            raise ValueError("filter ids must be unique")
        if self.combination.method == "weighted_sum" and not any(item.weight != 0 for item in self.components):
            raise ValueError("weighted_sum component weights cannot all be zero")
        minimum = self.preprocessing.missing_value.minimum_valid_components
        if minimum > len(self.components):
            raise ValueError("minimum_valid_components cannot exceed component count")
        variant_ids = [item.id for item in self.variants]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("variant ids must be unique")
        for variant in self.variants:
            unknown_components = set(variant.components) - set(component_ids)
            unknown_filters = set(variant.filters) - set(filter_ids)
            if unknown_components:
                raise ValueError(f"variant {variant.id} references unknown components: {sorted(unknown_components)}")
            if unknown_filters:
                raise ValueError(f"variant {variant.id} references unknown filters: {sorted(unknown_filters)}")
        return self


class FactorCombinationSpec(BaseModel):
    version: Literal[1] = 1
    kind: Literal["factor_combination"]
    factor_combination: FactorCombinationBody


class L0Config(BaseModel):
    min_coverage: float = 0.70
    max_missing_rate: float = 0.30
    min_daily_cross_section: int = 100
    min_unique_ratio: float = 0.01
    min_valid_groups_per_day: int = 5
    future_data_violation: int = 0


class L1Config(BaseModel):
    forward_horizons: list[int] = Field(default_factory=lambda: [1, 3, 5, 10, 15])
    quantile_groups: int = 10
    min_cross_section: int = 100
    universes: list[Literal["tradeable", "liquid"]] = Field(
        default_factory=lambda: ["tradeable", "liquid"]
    )
    targets: list[Literal["stock_return", "stock_minus_sw_l1_return"]] = Field(
        default_factory=lambda: ["stock_return"]
    )


class IndustrySelectorOverrides(BaseModel):
    short_ema: int = Field(default=3, ge=1)
    long_ema: int = Field(default=10, ge=2)
    breadth_change_window: int = Field(default=5, ge=1)
    ridge_alpha: float = Field(default=1.0, ge=0)
    minimum_industry_members: int = Field(default=8, ge=2)


class IndustrySelectorConfig(BaseModel):
    preset: Literal["sw_l1_neutralized_rotation_v1"] = "sw_l1_neutralized_rotation_v1"
    overrides: IndustrySelectorOverrides = Field(default_factory=IndustrySelectorOverrides)


class IndustrySliceDiagnostics(BaseModel):
    evaluate_industry_selector: bool = True
    save_industry_intermediate: bool = True


class IndustrySliceConfig(BaseModel):
    enabled: bool = False
    industry_standard: Literal["sw"] = "sw"
    industry_level: Literal["l1"] = "l1"
    membership_mode: Literal["point_in_time"] = "point_in_time"
    selector: IndustrySelectorConfig = Field(default_factory=IndustrySelectorConfig)
    scopes: list[Literal["all", "top2", "top5", "top10", "bottom5"]] = Field(
        default_factory=lambda: ["all", "top5", "bottom5"]
    )
    diagnostics: IndustrySliceDiagnostics = Field(default_factory=IndustrySliceDiagnostics)


class ExecutionConstraints(BaseModel):
    exclude_suspended: bool = True
    cannot_buy_limit_up: bool = True
    cannot_sell_limit_down: bool = True
    exclude_st: bool = True
    exclude_delisting_period: bool = True
    min_listing_days: int = 60


class CostModel(BaseModel):
    commission_bps_per_side: float = 3
    slippage_bps_per_side: float = 5
    stamp_duty_bps_sell: float = 5


class L2Config(BaseModel):
    rebalance_frequency: Literal["1D"] = "1D"
    holding_periods: list[int] = Field(default_factory=lambda: [1, 3, 5, 10, 15])
    top_n: list[int] = Field(default_factory=lambda: [2, 5, 10, 20])
    universes: list[Literal["tradeable", "liquid"]] = Field(default_factory=lambda: ["liquid"])
    weighting: Literal["equal"] = "equal"
    signal_time: Literal["close_t"] = "close_t"
    entry_price: Literal["open_t1"] = "open_t1"
    overlapping_portfolios: Literal[True] = True
    initial_cash: float = 1_000_000
    lot_size: int = 100
    no_fill_policy: Literal["keep_cash"] = "keep_cash"
    execution_constraints: ExecutionConstraints = Field(default_factory=ExecutionConstraints)
    cost_model: CostModel = Field(default_factory=CostModel)
    cost_scenarios_bps: list[int] = Field(default_factory=lambda: [0, 10, 20])
    benchmarks: list[str] = Field(default_factory=lambda: ["universe_equal_weight"])


class L3Config(BaseModel):
    enabled: bool = True
    robustness_tests: list[str] = Field(default_factory=lambda: [
        "year_by_year", "rolling_window", "walk_forward", "topn_curve", "holding_decay",
        "cost_sensitivity", "parameter_neighborhood", "contribution_concentration",
        "removal_of_best_trades", "industry_exposure", "size_exposure",
    ])


class OutputConfig(BaseModel):
    store_trade_details: bool = True
    store_daily_positions: bool = True
    store_daily_factor_values: bool = True
    store_experiment_manifest: bool = True


class ExperimentSpec(BaseModel):
    version: int = 1
    name: str
    factor_config: Path = Path("__cli_factor__.yaml")
    project_config: Path = Path("configs/project.yaml")
    scoring_config: Path = Path("configs/contracts/alpha_scoring_v1.yaml")
    backtest_contract: Path = Path("configs/contracts/backtest_contract_v1.yaml")
    data_version: str = "latest"
    sample_start_date: str | None = None
    sample_end_date: str | None = None
    research_mode: Literal["exploratory"] = "exploratory"
    stage_l0: L0Config = Field(default_factory=L0Config)
    stage_l1: L1Config = Field(default_factory=L1Config)
    stage_l2: L2Config = Field(default_factory=L2Config)
    stage_l3: L3Config = Field(default_factory=L3Config)
    output: OutputConfig = Field(default_factory=OutputConfig)
    industry_slice: IndustrySliceConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def accept_two_yaml_protocol_shape(cls, value):
        """Accept the concise public shape while preserving every legacy key."""
        if not isinstance(value, dict):
            return value
        data = dict(value)
        identity = data.pop("experiment", None)
        if "name" not in data and isinstance(identity, dict) and identity.get("id"):
            data["name"] = identity["id"]
        evaluation = data.pop("evaluation", None)
        if isinstance(evaluation, dict):
            stage_l1 = dict(data.get("stage_l1") or {})
            if "horizons" in evaluation and "forward_horizons" not in stage_l1:
                stage_l1["forward_horizons"] = evaluation["horizons"]
            if "targets" in evaluation and "targets" not in stage_l1:
                stage_l1["targets"] = evaluation["targets"]
            data["stage_l1"] = stage_l1
        return data

    @model_validator(mode="after")
    def fixed_first_version_space(self) -> "ExperimentSpec":
        allowed_h = {1, 3, 5, 10, 15}
        allowed_n = {2, 5, 10, 20}
        if not set(self.stage_l1.forward_horizons) <= allowed_h:
            raise ValueError("V1 forward horizons are limited to 1/3/5/10/15")
        if not set(self.stage_l2.holding_periods) <= allowed_h:
            raise ValueError("V1 holding periods are limited to 1/3/5/10/15")
        if not set(self.stage_l2.top_n) <= allowed_n:
            raise ValueError("V1 TopN values are limited to 2/5/10/20")
        return self


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_project(path: str | Path) -> ProjectConfig:
    return ProjectConfig.model_validate(load_yaml(path))


def load_factor(path: str | Path) -> FactorSpec:
    return FactorSpec.model_validate(load_yaml(path))


def factor_source_kind(path: str | Path) -> str:
    from factor_forge.exceptions import UnsupportedFactorKindError
    kind = load_yaml(path).get("kind", "factor")
    if kind not in {"factor", "factor_combination"}:
        raise UnsupportedFactorKindError(f"Unsupported factor kind: {kind!r}")
    return kind


def load_factor_combination(path: str | Path) -> FactorCombinationSpec:
    return FactorCombinationSpec.model_validate(load_yaml(path))


def load_experiment(path: str | Path) -> ExperimentSpec:
    return ExperimentSpec.model_validate(load_yaml(path))
