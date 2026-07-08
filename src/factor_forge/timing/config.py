from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .dataset import TimingFeatureConfig


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TimingInputPaths(StrictModel):
    index_daily: Path
    stock_daily: Path | None = None
    index_dailybasic: Path | None = None
    bond_yield: Path | None = None
    margin: Path | None = None
    option_basic: Path | None = None
    option_daily: Path | None = None
    option_iv_daily: Path | None = None
    futures_basic: Path | None = None
    futures_daily: Path | None = None
    futures_holding: Path | None = None
    moneyflow: Path | None = None
    cpi: Path | None = None
    pmi: Path | None = None
    epu: Path | None = None


class TimingBuildConfig(StrictModel):
    version: int = 1
    name: str = "timing_factor_library_v1"
    inputs: TimingInputPaths
    features: TimingFeatureConfigModel = Field(default_factory=lambda: TimingFeatureConfigModel())
    output_dir: Path = Path("artifacts/timing_features")


class TimingFeatureConfigModel(StrictModel):
    index_code: str = "000300.SH"
    benchmark_code: str | None = None
    future_prefix: str | None = None
    fallback_bond_10y_yield: float | None = Field(default=None, ge=0, le=1)
    horizon: int = Field(default=20, ge=1, le=120)
    horizons: tuple[int, ...] | None = None
    data_lag: int = Field(default=1, ge=0, le=10)
    z_window: int = Field(default=252, ge=20)
    pct_window: int = Field(default=756, ge=20)
    change_windows: tuple[int, ...] = (5, 20, 60)
    boll_windows: tuple[int, ...] = (20, 60)
    annualization_days: int = Field(default=365, ge=200, le=366)
    basis_roll_days: int = Field(default=5, ge=0, le=30)
    option_min_days_to_expiry: int = Field(default=7, ge=0, le=60)
    option_atm_moneyness: float = Field(default=0.05, gt=0, le=0.5)
    option_min_amount: float = Field(default=0.0, ge=0)
    option_price_field: str = "close"
    option_risk_free_rate: float = Field(default=0.02, ge=-0.1, le=0.2)
    iv_bounds: tuple[float, float] = (0.01, 1.50)
    z_clip: float = Field(default=3.0, gt=0)
    pct_clip: tuple[float, float] = (0.01, 0.99)
    extreme_low: tuple[float, ...] = (0.05, 0.10)
    extreme_high: tuple[float, ...] = (0.90, 0.95)

    def to_dataclass(self) -> TimingFeatureConfig:
        return TimingFeatureConfig(**self.model_dump())


def load_timing_build_config(path: str | Path) -> TimingBuildConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return TimingBuildConfig.model_validate(yaml.safe_load(handle) or {})
