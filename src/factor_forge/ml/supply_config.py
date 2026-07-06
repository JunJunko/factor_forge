"""Pydantic configuration models for the low-volume-rise supply-contraction pipeline.

The research thesis (see ``低量上涨供给收缩因子与LightGBM特征计算规范.md``): stocks that
rise versus their industry on below-normal turnover are being lifted by supply squeeze
(holder lockup / 惜售), not demand, and that structure may predict forward returns.  V1
measures it via an A/B ablation -- Model A (control variables only) vs Model B (controls
plus the supply-contraction core) -- where the only verdict that matters is
``Incremental Alpha = Performance(Model B) - Performance(Model A)``.

These models are the strict (extra="forbid") contract for the YAML under
``configs/ml/``.  ``Segment``/``Segments`` are reused from :mod:`factor_forge.ml.config`
so the disjoint-ordered ``train < valid < test`` invariant stays in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from .config import Segment, Segments, StrictModel


class SupplyFeatureConfig(StrictModel):
    """Window / parameter overrides for the ~20-25 V1 features (document sec. 3, 14)."""

    excess_return_windows: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    volatility_window: int = Field(default=20, ge=2)
    volatility_ddof: Literal[0, 1] = Field(default=1, description="document '样本标准差' → ddof=1")
    turnover_windows: list[int] = Field(default_factory=lambda: [20, 60])
    amount_window: int = Field(default=20, ge=2)
    volume_residual_window: int = Field(default=120, ge=20)
    volume_residual_min_periods: int = Field(default=80, ge=10)
    price_impact_windows: list[int] = Field(default_factory=lambda: [1, 5])
    price_impact_denom_floor: float = Field(default=0.005, gt=0)
    amihud_window: int = Field(default=20, ge=2)
    zero_return_window: int = Field(default=20, ge=2)
    log_amount_window: int = Field(default=20, ge=2)
    effective_ticks_window: int = Field(default=5, ge=1)
    persistence_window: int = Field(default=5, ge=1)
    # tick_size is uniform 0.01 CNY across modern A-share boards; exposed for completeness.
    tick_size: float = Field(default=0.01, gt=0)
    # V2 stable-baseline + no-volume-rise structure (handoff doc sec. 3.2/3.6/3.7/3.8).
    # Baseline window = t-29..t-2 (baseline_window bars) + event window = t-1..t
    # (event_window bars); the event days NEVER enter the baseline mean/std (handoff 4.1.3).
    baseline_window: int = Field(default=28, ge=2)
    event_window: int = Field(default=2, ge=1, le=10)
    z_clip_lower: float = Field(default=-3.0, lt=0, description="recent_volume_z clip lower (handoff 3.8)")
    z_clip_upper: float = Field(default=3.0, gt=0, description="recent_volume_z clip upper; [-4,4] is a sensitivity variant")
    std_floor_method: Literal["cross_section_quantile", "train_period_fixed"] = "cross_section_quantile"
    std_floor_quantile: float = Field(default=0.10, ge=0, lt=0.5, description="daily cross-section quantile of baseline_std_28 (handoff 3.7)")
    winsor_quantile: float = Field(default=0.01, ge=0, lt=0.5)
    cross_sectional_zscore: bool = True
    min_listing_days: int = Field(default=60, ge=1, description="sample filter, document sec. 16")
    # Sample weights (document sec. 8.6 / 9.5 / 9.6) -- training weights, not model inputs.
    use_sample_weight: bool = True
    price_weight_lambda: float = Field(default=2.0, ge=0, description="lambda in 1/(1+lambda*tick_noise)")
    liquidity_weight_low_quantile: float = Field(default=0.1, ge=0, lt=0.5)
    liquidity_weight_full_quantile: float = Field(default=0.5, gt=0, le=1.0)


class SupplyLabelConfig(StrictModel):
    horizon: int = Field(default=5, ge=1, le=60)
    # V1 default open-to-open aligns with the Qlib deal_price=("$open","$open") and the
    # project BacktestEngine's adj_open mark-to-market.  ``open_to_close`` keeps the
    # document's sec. 15.1 definition selectable for an A/B follow-up.
    label_method: Literal["open_to_open", "open_to_close"] = "open_to_open"
    industry_neutralize: bool = True
    industry_aggregation: Literal["equal_weight", "leave_one_out"] = "leave_one_out"


class QlibLGBConfig(StrictModel):
    """Parameters passed to ``qlib.contrib.model.gbdt.LGBModel`` (mirrors breakout_qlib)."""

    loss: Literal["mse", "binary"] = "mse"
    learning_rate: float = Field(default=0.03, gt=0)
    num_leaves: int = Field(default=31, ge=2)
    max_depth: int = -1
    num_boost_round: int = Field(default=800, ge=1)
    early_stopping_rounds: int = Field(default=60, ge=1)
    min_data_in_leaf: int = Field(default=80, ge=1)
    feature_fraction: float = Field(default=0.8, gt=0, le=1)
    bagging_fraction: float = Field(default=0.8, gt=0, le=1)
    bagging_freq: int = Field(default=1, ge=0)
    lambda_l1: float = Field(default=0.1, ge=0)
    lambda_l2: float = Field(default=1.0, ge=0)
    num_threads: int = -1
    seed: int = 42


class SupplyBacktestConfig(StrictModel):
    """Qlib backtest portfolio knobs (TopkDropoutStrategy + SimulatorExecutor)."""

    topk: int = Field(default=50, ge=1)
    n_drop: int = Field(default=5, ge=0)
    hold_thresh: int = Field(default=1, ge=1, description=">=1 enforces A-share T+1 selling")
    initial_cash: float = Field(default=10_000_000, gt=0)
    round_trip_cost_bps: float = Field(default=20.0, ge=0, description="split half buy / half sell")
    trade_unit: int = Field(default=100, ge=1)


class CrossCheckConfig(StrictModel):
    """Optional project-BacktestEngine sanity check on Model B predictions.

    Qlib's native backtest is a first in this repo; running the proven
    :class:`factor_forge.backtest.engine.BacktestEngine` on the same signal is a cheap
    invariant safety net (it is NOT the primary backtest -- Qlib is).
    """

    enabled: bool = True
    universe: Literal["tradeable", "liquid"] = "tradeable"
    top_n: int = Field(default=50, ge=1)
    holding_days: int = Field(default=5, ge=1, le=60)
    cost_bps: float = Field(default=20, ge=0)


class ModelSpec(StrictModel):
    """One ablation arm: a name plus the feature-group names it may use.

    ``feature_groups`` resolves via :data:`supply_dataset.FEATURE_GROUP_REGISTRY`;
    ``features`` is an optional explicit list merged in (handy for minimal-arm
    ablations like "controls + just scarcity + scarcity_slope_5").
    """

    name: str
    feature_groups: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)


class AblationSpec(StrictModel):
    """Model A (controls) vs Model B (controls + supply_core) -- document sec. 17."""

    models: list[ModelSpec]

    @model_validator(mode="after")
    def at_least_two(self):
        if len(self.models) < 2:
            raise ValueError("ablation needs at least two models to measure incremental alpha")
        names = [m.name for m in self.models]
        if len(set(names)) != len(names):
            raise ValueError(f"ablation model names must be unique, got {names}")
        return self


class SupplyPipelineConfig(StrictModel):
    version: int = 1
    name: str = "supply_contraction_qlib_v1"
    project_config: Path = Path("configs/project.yaml")
    data_version: str = "latest"
    require_full_segment_coverage: bool = True
    segments: Segments
    features: SupplyFeatureConfig = Field(default_factory=SupplyFeatureConfig)
    label: SupplyLabelConfig = Field(default_factory=SupplyLabelConfig)
    model: QlibLGBConfig = Field(default_factory=QlibLGBConfig)
    backtest: SupplyBacktestConfig = Field(default_factory=SupplyBacktestConfig)
    crosscheck: CrossCheckConfig = Field(default_factory=CrossCheckConfig)
    ablation: AblationSpec
    qlib_provider_root: Path = Path("artifacts/qlib_bin_cache")
    output_root: Path = Path("artifacts/supply_contraction_runs")
    # Optional liquidity-based universe cap to keep the per-stock volume_residual build and
    # the Qlib backtest tractable on the full A-share history.  None = use every stock.
    universe_top_n: int | None = None


def load_supply_config(path: str | Path) -> SupplyPipelineConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return SupplyPipelineConfig.model_validate(yaml.safe_load(handle) or {})
