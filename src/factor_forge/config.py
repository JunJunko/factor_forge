from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


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
    factor: FactorMeta
    data: FactorData
    scope: FactorScope
    calculation: FactorCalculation
    output: FactorOutput = Field(default_factory=FactorOutput)


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
    factor_config: Path
    project_config: Path = Path("configs/project.yaml")
    scoring_config: Path = Path("configs/contracts/alpha_scoring_v1.yaml")
    data_version: str = "latest"
    research_mode: Literal["exploratory"] = "exploratory"
    stage_l0: L0Config = Field(default_factory=L0Config)
    stage_l1: L1Config = Field(default_factory=L1Config)
    stage_l2: L2Config = Field(default_factory=L2Config)
    stage_l3: L3Config = Field(default_factory=L3Config)
    output: OutputConfig = Field(default_factory=OutputConfig)

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


def load_experiment(path: str | Path) -> ExperimentSpec:
    return ExperimentSpec.model_validate(load_yaml(path))
