from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PercentileWindow(StrictModel):
    window: int = Field(ge=2)
    min_periods: int = Field(ge=1)

    @model_validator(mode="after")
    def min_periods_not_above_window(self):
        if self.min_periods > self.window:
            raise ValueError("min_periods cannot exceed window")
        return self


class TemplateData(StrictModel):
    required_fields: list[str] = Field(min_length=1)
    universe_field: str = "is_liquid"
    entity_field: str = "ts_code"
    date_field: str = "trade_date"
    industry_field: str = "industry_l1_code"


class ScanWindow(StrictModel):
    discovery_window_days: int = Field(default=252, ge=20)
    recent_window_days: int = Field(default=20, ge=5)

    @model_validator(mode="after")
    def recent_inside_discovery(self):
        if self.recent_window_days >= self.discovery_window_days:
            raise ValueError("recent_window_days must be lower than discovery_window_days")
        return self


class EventFilters(StrictModel):
    min_listing_days: int = Field(default=0, ge=0)
    exclude_st: bool = False
    exclude_suspended: bool = False
    exclude_limit_locked: bool = False


class QualityGate(StrictModel):
    min_events: int = Field(default=0, ge=0)
    min_unique_stocks: int = Field(default=0, ge=0)
    min_unique_industries: int = Field(default=0, ge=0)
    max_industry_share: float = Field(default=1.0, ge=0, le=1)
    max_single_stock_share: float = Field(default=1.0, ge=0, le=1)


class PriceDropParameters(StrictModel):
    return_horizon: int = Field(default=3, ge=1)
    return_history: PercentileWindow
    volume_history: PercentileWindow
    return_percentile_lte: float = Field(ge=0, le=1)
    volume_percentile_lte: float = Field(ge=0, le=1)


class VolumeSurgeParameters(StrictModel):
    return_horizon: Literal[1] = 1
    abs_return_history: PercentileWindow
    volume_history: PercentileWindow
    volume_percentile_gte: float = Field(ge=0, le=1)
    abs_return_percentile_lte: float = Field(ge=0, le=1)


class HighTurnoverLowDisplacementParameters(StrictModel):
    turnover_history: PercentileWindow
    residual_history: PercentileWindow
    turnover_percentile_gte: float = Field(ge=0, le=1)
    residual_percentile_lte: float = Field(ge=0, le=1)
    min_cross_section: int = Field(default=100, ge=10)


class LowLiquidityLargeDisplacementParameters(StrictModel):
    abs_return_history: PercentileWindow
    amount_history: PercentileWindow
    abs_return_percentile_gte: float = Field(ge=0, le=1)
    amount_percentile_lte: float = Field(ge=0, le=1)


class LongLowerWickStrongCloseParameters(StrictModel):
    wick_history: PercentileWindow
    weak_return_history: PercentileWindow
    lower_wick_percentile_gte: float = Field(ge=0, le=1)
    close_position_gte: float = Field(ge=0, le=1)
    weak_return_percentile_lte: float = Field(ge=0, le=1)


class StockIndustryDivergenceParameters(StrictModel):
    return_horizon: int = Field(default=5, ge=1)
    volatility_window: int = Field(default=20, ge=5)
    upper_percentile_gte: float = Field(ge=0, le=1)
    lower_percentile_lte: float = Field(ge=0, le=1)
    min_industry_size: int = Field(default=10, ge=5)


class VolatilityCompressionBreakoutParameters(StrictModel):
    atr_window: int = Field(default=14, ge=2)
    range_window: int = Field(default=5, ge=2)
    breakout_window: int = Field(default=20, ge=5)
    state_history: PercentileWindow
    breakout_history: PercentileWindow
    amount_history: PercentileWindow
    compression_percentile_lte: float = Field(ge=0, le=1)
    breakout_percentile_gte: float = Field(ge=0, le=1)
    expanded_volume_percentile_gte: float = Field(ge=0, le=1)
    contracted_volume_percentile_lte: float = Field(ge=0, le=1)


class TrendExhaustionParameters(StrictModel):
    long_horizon: int = Field(default=10, ge=5)
    short_horizon: int = Field(default=3, ge=1)
    history: PercentileWindow
    strong_percentile_gte: float = Field(ge=0, le=1)
    weak_percentile_lte: float = Field(ge=0, le=1)
    acceleration_extreme: float = Field(ge=0, le=0.5)


CompositeRecipe = Literal[
    "index_up_breadth_down",
    "turnover_concentration",
    "turnover_displacement_close_context",
    "enhanced_stock_industry_residual",
    "price_volume_conditional_residual",
    "leader_industry_median_decoupling",
    "failed_breakout",
    "momentum_participation_deterioration",
    "percentile_rapid_migration",
    "short_long_percentile_conflict",
    "turnover_residual_concentration",
]


class CompositeAnomalyParameters(StrictModel):
    """Parameters shared by declarative, label-free composite anomaly recipes."""

    recipe: CompositeRecipe
    history: PercentileWindow
    short_window: int = Field(default=5, ge=2)
    long_window: int = Field(default=20, ge=5)
    upper_percentile: float = Field(default=0.90, ge=0.5, le=1)
    lower_percentile: float = Field(default=0.10, ge=0, le=0.5)
    change_threshold: float = Field(default=0.30, ge=0, le=1)
    min_cross_section: int = Field(default=20, ge=5)
    liquidity_min_periods: int = Field(default=10, ge=2)


class BaseTemplate(StrictModel):
    version: Literal[1]
    id: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    description: str
    observation_type: Literal["relation_anomaly"]
    data: TemplateData
    scan: ScanWindow
    filters: EventFilters = Field(default_factory=EventFilters)
    quality_gate: QualityGate = Field(default_factory=QualityGate)

    def definition_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PriceDropWithoutVolumeTemplate(BaseTemplate):
    kind: Literal["price_drop_without_volume_confirmation"]
    parameters: PriceDropParameters


class VolumeSurgeWithoutImpactTemplate(BaseTemplate):
    kind: Literal["volume_surge_without_price_impact"]
    parameters: VolumeSurgeParameters


class HighTurnoverLowDisplacementTemplate(BaseTemplate):
    kind: Literal["high_turnover_low_displacement"]
    parameters: HighTurnoverLowDisplacementParameters


class LowLiquidityLargeDisplacementTemplate(BaseTemplate):
    kind: Literal["low_liquidity_large_displacement"]
    parameters: LowLiquidityLargeDisplacementParameters


class LongLowerWickStrongCloseTemplate(BaseTemplate):
    kind: Literal["long_lower_wick_strong_close"]
    parameters: LongLowerWickStrongCloseParameters


class StockIndustryDivergenceTemplate(BaseTemplate):
    kind: Literal["stock_industry_divergence"]
    parameters: StockIndustryDivergenceParameters


class VolatilityCompressionBreakoutTemplate(BaseTemplate):
    kind: Literal["volatility_compression_breakout"]
    parameters: VolatilityCompressionBreakoutParameters


class TrendExhaustionTemplate(BaseTemplate):
    kind: Literal["trend_exhaustion"]
    parameters: TrendExhaustionParameters


class CompositeAnomalyTemplate(BaseTemplate):
    kind: Literal["composite_anomaly"]
    parameters: CompositeAnomalyParameters


RadarTemplate: TypeAlias = Annotated[
    PriceDropWithoutVolumeTemplate
    | VolumeSurgeWithoutImpactTemplate
    | HighTurnoverLowDisplacementTemplate
    | LowLiquidityLargeDisplacementTemplate
    | LongLowerWickStrongCloseTemplate
    | StockIndustryDivergenceTemplate
    | VolatilityCompressionBreakoutTemplate
    | TrendExhaustionTemplate
    | CompositeAnomalyTemplate,
    Field(discriminator="kind"),
]
RADAR_TEMPLATE_ADAPTER = TypeAdapter(RadarTemplate)


def load_radar_template(path: str | Path) -> RadarTemplate:
    with Path(path).open("r", encoding="utf-8") as handle:
        return RADAR_TEMPLATE_ADAPTER.validate_python(yaml.safe_load(handle))


def required_trading_rows(template: RadarTemplate) -> int:
    if isinstance(template, PriceDropWithoutVolumeTemplate):
        history = max(
            template.parameters.return_history.window + template.parameters.return_horizon,
            template.parameters.volume_history.window,
        )
    elif isinstance(template, VolumeSurgeWithoutImpactTemplate):
        history = max(
            template.parameters.abs_return_history.window + template.parameters.return_horizon,
            template.parameters.volume_history.window,
        )
    elif isinstance(template, HighTurnoverLowDisplacementTemplate):
        history = max(template.parameters.turnover_history.window,
                      template.parameters.residual_history.window) + 2
    elif isinstance(template, LowLiquidityLargeDisplacementTemplate):
        history = max(template.parameters.abs_return_history.window,
                      template.parameters.amount_history.window) + 2
    elif isinstance(template, LongLowerWickStrongCloseTemplate):
        history = max(template.parameters.wick_history.window,
                      template.parameters.weak_return_history.window) + 5
    elif isinstance(template, StockIndustryDivergenceTemplate):
        history = template.parameters.return_horizon + template.parameters.volatility_window + 5
    elif isinstance(template, VolatilityCompressionBreakoutTemplate):
        history = max(
            template.parameters.state_history.window + template.parameters.atr_window,
            template.parameters.breakout_history.window + template.parameters.breakout_window,
            template.parameters.amount_history.window,
        ) + 5
    elif isinstance(template, TrendExhaustionTemplate):
        history = template.parameters.history.window + template.parameters.long_horizon + 5
    else:
        history = template.parameters.history.window + template.parameters.long_window + 10
    return template.scan.discovery_window_days + history + 5


def filter_required_fields(template: RadarTemplate) -> list[str]:
    fields = []
    if template.filters.min_listing_days:
        fields.append("listing_trade_days")
    if template.filters.exclude_st:
        fields.append("is_st")
    if template.filters.exclude_suspended:
        fields.append("is_suspended")
    if template.filters.exclude_limit_locked:
        fields.extend(["is_limit_up_open", "is_limit_down_open"])
    return fields
