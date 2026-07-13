from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from factor_forge.research_control.models import utc_now

from .models import (
    ObservationCard,
    ObservationDefinition,
    ObservationEvidence,
    ObservationQuality,
    RadarScanResult,
)
from .percentiles import pit_rolling_percentile, temporal_prefix_audit
from .concentration_features import build_turnover_residual_concentration_features
from .templates import (
    CompositeAnomalyTemplate,
    HighTurnoverLowDisplacementTemplate,
    LongLowerWickStrongCloseTemplate,
    LowLiquidityLargeDisplacementTemplate,
    PriceDropWithoutVolumeTemplate,
    RadarTemplate,
    StockIndustryDivergenceTemplate,
    TrendExhaustionTemplate,
    VolatilityCompressionBreakoutTemplate,
    VolumeSurgeWithoutImpactTemplate,
    filter_required_fields,
)


def observation_id_for(template: RadarTemplate, data_version: str, as_of_date: str | pd.Timestamp) -> str:
    as_of_text = pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
    digest = hashlib.sha256(
        json.dumps(
            {
                "template": template.definition_hash(),
                "data_version": data_version,
                "as_of_date": as_of_text,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"obs_{template.id}_{digest}"


@dataclass(frozen=True)
class RadarMeasurementResult:
    """Dense point-in-time measurements before discovery-window card aggregation."""

    frame: pd.DataFrame
    condition_values: dict
    measurement_fields: list[str]
    audit_specs: list[tuple[str, int, int]]
    duplicate_keys: int
    as_of_date: pd.Timestamp


class RelationAnomalyScanner:
    def scan(
        self,
        panel: pd.DataFrame,
        template: RadarTemplate,
        *,
        data_version: str,
        as_of_date: str | pd.Timestamp | None = None,
    ) -> RadarScanResult:
        measurement = self._measure(panel, template, as_of_date=as_of_date)
        return self._build_result(
            measurement.frame, template, data_version=data_version,
            as_of_date=measurement.as_of_date,
            condition_values=measurement.condition_values,
            measurement_fields=measurement.measurement_fields,
            duplicate_keys=measurement.duplicate_keys,
            audit_specs=measurement.audit_specs,
        )

    def measure_event_channels(
        self,
        panel: pd.DataFrame,
        template: RadarTemplate,
        *,
        as_of_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return only whitelisted, label-free dense channels for sequence models."""
        measurement = self._measure(panel, template, as_of_date=as_of_date)
        data = template.data
        measured = measurement.frame
        eligible = self._eligible(measured, template)
        severity = pd.to_numeric(measured["severity"], errors="coerce")
        valid = eligible & severity.notna()
        event = valid & measured["is_event"].fillna(False).astype(bool)
        prefix = template.id
        return pd.DataFrame({
            data.date_field: pd.to_datetime(measured[data.date_field]),
            data.entity_field: measured[data.entity_field].astype(str),
            f"{prefix}__eligible": eligible.astype("float32"),
            f"{prefix}__event": event.astype("float32"),
            f"{prefix}__severity": severity.where(event, 0.0).where(valid).astype("float32"),
            f"{prefix}__valid": valid.astype("float32"),
        }).reset_index(drop=True)

    def _measure(
        self,
        panel: pd.DataFrame,
        template: RadarTemplate,
        *,
        as_of_date: str | pd.Timestamp | None = None,
    ) -> RadarMeasurementResult:
        """Prepare and measure one template; shared by cards and sequence channels."""
        data = template.data
        required = set(data.required_fields) | set(filter_required_fields(template)) | {
            data.entity_field, data.date_field, data.industry_field, data.universe_field,
        }
        missing = required - set(panel.columns)
        if missing:
            raise KeyError(f"radar input missing columns: {sorted(missing)}")
        duplicate_keys = int(panel.duplicated([data.entity_field, data.date_field]).sum())
        if duplicate_keys:
            raise ValueError(f"radar input has {duplicate_keys} duplicate entity/date rows")

        working = panel.copy()
        working[data.date_field] = pd.to_datetime(working[data.date_field])
        cutoff = pd.Timestamp(as_of_date) if as_of_date is not None else working[data.date_field].max()
        working = working.loc[working[data.date_field].le(cutoff)].copy()
        if working.empty:
            raise ValueError("radar input is empty at or before as_of_date")
        working = working.sort_values([data.entity_field, data.date_field], kind="mergesort")

        if isinstance(template, PriceDropWithoutVolumeTemplate):
            measured, condition_values, measurement_fields, audit_specs = self._price_drop_without_volume(
                working, template
            )
        elif isinstance(template, VolumeSurgeWithoutImpactTemplate):
            measured, condition_values, measurement_fields, audit_specs = self._volume_surge_without_impact(
                working, template
            )
        elif isinstance(template, HighTurnoverLowDisplacementTemplate):
            measured, condition_values, measurement_fields, audit_specs = self._high_turnover_low_displacement(
                working, template
            )
        elif isinstance(template, LowLiquidityLargeDisplacementTemplate):
            measured, condition_values, measurement_fields, audit_specs = self._low_liquidity_large_displacement(
                working, template
            )
        elif isinstance(template, LongLowerWickStrongCloseTemplate):
            measured, condition_values, measurement_fields, audit_specs = self._long_lower_wick_strong_close(
                working, template
            )
        elif isinstance(template, StockIndustryDivergenceTemplate):
            measured, condition_values, measurement_fields, audit_specs = self._stock_industry_divergence(
                working, template
            )
        elif isinstance(template, VolatilityCompressionBreakoutTemplate):
            measured, condition_values, measurement_fields, audit_specs = self._volatility_compression_breakout(
                working, template
            )
        elif isinstance(template, TrendExhaustionTemplate):
            measured, condition_values, measurement_fields, audit_specs = self._trend_exhaustion(
                working, template
            )
        elif isinstance(template, CompositeAnomalyTemplate):
            measured, condition_values, measurement_fields, audit_specs = self._composite_anomaly(
                working, template
            )
        else:  # pragma: no cover - the discriminated template union prevents this
            raise TypeError(f"unsupported radar template: {type(template).__name__}")

        return RadarMeasurementResult(
            frame=measured,
            condition_values=condition_values,
            measurement_fields=measurement_fields,
            audit_specs=audit_specs,
            duplicate_keys=duplicate_keys,
            as_of_date=cutoff,
        )

    @staticmethod
    def _price_drop_without_volume(
        frame: pd.DataFrame, template: PriceDropWithoutVolumeTemplate
    ) -> tuple[pd.DataFrame, dict, list[str], list[tuple[str, int, int]]]:
        data, parameters = template.data, template.parameters
        result = frame.copy()
        close = pd.to_numeric(result["adj_close"], errors="coerce")
        result["return_horizon"] = close.groupby(result[data.entity_field], sort=False).pct_change(
            parameters.return_horizon, fill_method=None
        )
        result["volume_value"] = pd.to_numeric(result["volume_shares"], errors="coerce")
        result["return_history_percentile"] = pit_rolling_percentile(
            result, "return_horizon", entity_column=data.entity_field, date_column=data.date_field,
            window=parameters.return_history.window, min_periods=parameters.return_history.min_periods,
        )
        result["volume_history_percentile"] = pit_rolling_percentile(
            result, "volume_value", entity_column=data.entity_field, date_column=data.date_field,
            window=parameters.volume_history.window, min_periods=parameters.volume_history.min_periods,
        )
        eligible = RelationAnomalyScanner._eligible(result, template)
        event = (
            eligible
            & result["return_history_percentile"].le(parameters.return_percentile_lte)
            & result["volume_history_percentile"].le(parameters.volume_percentile_lte)
        )
        return_scale = max(parameters.return_percentile_lte, 1e-9)
        volume_scale = max(parameters.volume_percentile_lte, 1e-9)
        result["severity"] = (
            (parameters.return_percentile_lte - result["return_history_percentile"]) / return_scale
            + (parameters.volume_percentile_lte - result["volume_history_percentile"]) / volume_scale
        ).clip(lower=0)
        result["is_event"] = event
        conditions = {
            "return_horizon": parameters.return_horizon,
            "return_history_percentile_lte": parameters.return_percentile_lte,
            "volume_history_percentile_lte": parameters.volume_percentile_lte,
            "strict_prior_history": "true",
        }
        measurements = [
            "return_horizon", "volume_value", "return_history_percentile",
            "volume_history_percentile", "severity",
        ]
        audits = [
            ("return_horizon", parameters.return_history.window, parameters.return_history.min_periods),
            ("volume_value", parameters.volume_history.window, parameters.volume_history.min_periods),
        ]
        return result, conditions, measurements, audits

    @staticmethod
    def _volume_surge_without_impact(
        frame: pd.DataFrame, template: VolumeSurgeWithoutImpactTemplate
    ) -> tuple[pd.DataFrame, dict, list[str], list[tuple[str, int, int]]]:
        data, parameters = template.data, template.parameters
        result = frame.copy()
        close = pd.to_numeric(result["adj_close"], errors="coerce")
        result["abs_return_1d"] = close.groupby(result[data.entity_field], sort=False).pct_change(
            parameters.return_horizon, fill_method=None
        ).abs()
        result["volume_value"] = pd.to_numeric(result["volume_shares"], errors="coerce")
        result["abs_return_history_percentile"] = pit_rolling_percentile(
            result, "abs_return_1d", entity_column=data.entity_field, date_column=data.date_field,
            window=parameters.abs_return_history.window,
            min_periods=parameters.abs_return_history.min_periods,
        )
        result["volume_history_percentile"] = pit_rolling_percentile(
            result, "volume_value", entity_column=data.entity_field, date_column=data.date_field,
            window=parameters.volume_history.window, min_periods=parameters.volume_history.min_periods,
        )
        eligible = RelationAnomalyScanner._eligible(result, template)
        event = (
            eligible
            & result["volume_history_percentile"].ge(parameters.volume_percentile_gte)
            & result["abs_return_history_percentile"].le(parameters.abs_return_percentile_lte)
        )
        volume_scale = max(1 - parameters.volume_percentile_gte, 1e-9)
        impact_scale = max(parameters.abs_return_percentile_lte, 1e-9)
        result["severity"] = (
            (result["volume_history_percentile"] - parameters.volume_percentile_gte) / volume_scale
            + (parameters.abs_return_percentile_lte - result["abs_return_history_percentile"]) / impact_scale
        ).clip(lower=0)
        result["is_event"] = event
        conditions = {
            "return_horizon": parameters.return_horizon,
            "volume_history_percentile_gte": parameters.volume_percentile_gte,
            "abs_return_history_percentile_lte": parameters.abs_return_percentile_lte,
            "strict_prior_history": "true",
        }
        measurements = [
            "abs_return_1d", "volume_value", "abs_return_history_percentile",
            "volume_history_percentile", "severity",
        ]
        audits = [
            ("abs_return_1d", parameters.abs_return_history.window,
             parameters.abs_return_history.min_periods),
            ("volume_value", parameters.volume_history.window, parameters.volume_history.min_periods),
        ]
        return result, conditions, measurements, audits

    @staticmethod
    def _high_turnover_low_displacement(
        frame: pd.DataFrame, template: HighTurnoverLowDisplacementTemplate
    ) -> tuple[pd.DataFrame, dict, list[str], list[tuple[str, int, int]]]:
        data, p = template.data, template.parameters
        result = frame.copy()
        close = pd.to_numeric(result["adj_close"], errors="coerce")
        result["abs_return_1d"] = close.groupby(result[data.entity_field], sort=False).pct_change(
            1, fill_method=None
        ).abs()
        result["turnover_value"] = pd.to_numeric(result["turnover_rate"], errors="coerce")
        result["turnover_history_percentile"] = pit_rolling_percentile(
            result, "turnover_value", entity_column=data.entity_field, date_column=data.date_field,
            window=p.turnover_history.window, min_periods=p.turnover_history.min_periods,
        )
        result["displacement_given_turnover_residual"] = RelationAnomalyScanner._daily_residual(
            result, "abs_return_1d", ["turnover_history_percentile"], p.min_cross_section,
            [data.date_field],
        )
        result["residual_history_percentile"] = pit_rolling_percentile(
            result, "displacement_given_turnover_residual",
            entity_column=data.entity_field, date_column=data.date_field,
            window=p.residual_history.window, min_periods=p.residual_history.min_periods,
        )
        eligible = RelationAnomalyScanner._eligible(result, template)
        result["is_event"] = (
            eligible
            & result["turnover_history_percentile"].ge(p.turnover_percentile_gte)
            & result["residual_history_percentile"].le(p.residual_percentile_lte)
        )
        result["severity"] = (
            (result["turnover_history_percentile"] - p.turnover_percentile_gte)
            / max(1 - p.turnover_percentile_gte, 1e-9)
            + (p.residual_percentile_lte - result["residual_history_percentile"])
            / max(p.residual_percentile_lte, 1e-9)
        ).clip(lower=0)
        return result, {
            "turnover_history_percentile_gte": p.turnover_percentile_gte,
            "conditional_residual_percentile_lte": p.residual_percentile_lte,
            "residual_model": "daily_abs_return_on_turnover_percentile",
        }, [
            "abs_return_1d", "turnover_value", "turnover_history_percentile",
            "displacement_given_turnover_residual", "residual_history_percentile", "severity",
        ], [
            ("turnover_value", p.turnover_history.window, p.turnover_history.min_periods),
            ("displacement_given_turnover_residual", p.residual_history.window,
             p.residual_history.min_periods),
        ]

    @staticmethod
    def _low_liquidity_large_displacement(
        frame: pd.DataFrame, template: LowLiquidityLargeDisplacementTemplate
    ) -> tuple[pd.DataFrame, dict, list[str], list[tuple[str, int, int]]]:
        data, p = template.data, template.parameters
        result = frame.copy()
        close = pd.to_numeric(result["adj_close"], errors="coerce")
        result["abs_return_1d"] = close.groupby(result[data.entity_field], sort=False).pct_change(
            1, fill_method=None
        ).abs()
        result["amount_value"] = pd.to_numeric(result["amount_cny"], errors="coerce")
        result["abs_return_history_percentile"] = pit_rolling_percentile(
            result, "abs_return_1d", entity_column=data.entity_field, date_column=data.date_field,
            window=p.abs_return_history.window, min_periods=p.abs_return_history.min_periods,
        )
        result["amount_history_percentile"] = pit_rolling_percentile(
            result, "amount_value", entity_column=data.entity_field, date_column=data.date_field,
            window=p.amount_history.window, min_periods=p.amount_history.min_periods,
        )
        eligible = RelationAnomalyScanner._eligible(result, template)
        result["is_event"] = (
            eligible
            & result["abs_return_history_percentile"].ge(p.abs_return_percentile_gte)
            & result["amount_history_percentile"].le(p.amount_percentile_lte)
        )
        result["severity"] = (
            (result["abs_return_history_percentile"] - p.abs_return_percentile_gte)
            / max(1 - p.abs_return_percentile_gte, 1e-9)
            + (p.amount_percentile_lte - result["amount_history_percentile"])
            / max(p.amount_percentile_lte, 1e-9)
        ).clip(lower=0)
        return result, {
            "abs_return_history_percentile_gte": p.abs_return_percentile_gte,
            "amount_history_percentile_lte": p.amount_percentile_lte,
        }, [
            "abs_return_1d", "amount_value", "abs_return_history_percentile",
            "amount_history_percentile", "severity",
        ], [
            ("abs_return_1d", p.abs_return_history.window, p.abs_return_history.min_periods),
            ("amount_value", p.amount_history.window, p.amount_history.min_periods),
        ]

    @staticmethod
    def _long_lower_wick_strong_close(
        frame: pd.DataFrame, template: LongLowerWickStrongCloseTemplate
    ) -> tuple[pd.DataFrame, dict, list[str], list[tuple[str, int, int]]]:
        data, p = template.data, template.parameters
        result = frame.copy()
        open_ = pd.to_numeric(result["adj_open"], errors="coerce")
        high = pd.to_numeric(result["adj_high"], errors="coerce")
        low = pd.to_numeric(result["adj_low"], errors="coerce")
        close = pd.to_numeric(result["adj_close"], errors="coerce")
        span = (high - low).replace(0, np.nan)
        result["lower_wick_ratio"] = (np.minimum(open_, close) - low) / span
        result["close_position_in_bar"] = (close - low) / span
        result["return_3d"] = close.groupby(result[data.entity_field], sort=False).pct_change(
            3, fill_method=None
        )
        result["lower_wick_history_percentile"] = pit_rolling_percentile(
            result, "lower_wick_ratio", entity_column=data.entity_field, date_column=data.date_field,
            window=p.wick_history.window, min_periods=p.wick_history.min_periods,
        )
        result["return_3d_history_percentile"] = pit_rolling_percentile(
            result, "return_3d", entity_column=data.entity_field, date_column=data.date_field,
            window=p.weak_return_history.window, min_periods=p.weak_return_history.min_periods,
        )
        eligible = RelationAnomalyScanner._eligible(result, template)
        result["is_event"] = (
            eligible
            & result["lower_wick_history_percentile"].ge(p.lower_wick_percentile_gte)
            & result["close_position_in_bar"].ge(p.close_position_gte)
            & result["return_3d_history_percentile"].le(p.weak_return_percentile_lte)
        )
        result["severity"] = (
            (result["lower_wick_history_percentile"] - p.lower_wick_percentile_gte)
            / max(1 - p.lower_wick_percentile_gte, 1e-9)
            + (result["close_position_in_bar"] - p.close_position_gte)
            / max(1 - p.close_position_gte, 1e-9)
            + (p.weak_return_percentile_lte - result["return_3d_history_percentile"])
            / max(p.weak_return_percentile_lte, 1e-9)
        ).clip(lower=0)
        return result, {
            "lower_wick_history_percentile_gte": p.lower_wick_percentile_gte,
            "close_position_in_bar_gte": p.close_position_gte,
            "return_3d_history_percentile_lte": p.weak_return_percentile_lte,
        }, [
            "lower_wick_ratio", "close_position_in_bar", "return_3d",
            "lower_wick_history_percentile", "return_3d_history_percentile", "severity",
        ], [
            ("lower_wick_ratio", p.wick_history.window, p.wick_history.min_periods),
            ("return_3d", p.weak_return_history.window, p.weak_return_history.min_periods),
        ]

    @staticmethod
    def _stock_industry_divergence(
        frame: pd.DataFrame, template: StockIndustryDivergenceTemplate
    ) -> tuple[pd.DataFrame, dict, list[str], list[tuple[str, int, int]]]:
        data, p = template.data, template.parameters
        result = frame.copy()
        close = pd.to_numeric(result["adj_close"], errors="coerce")
        grouped = close.groupby(result[data.entity_field], sort=False)
        result["return_5d"] = grouped.pct_change(p.return_horizon, fill_method=None)
        daily = grouped.pct_change(1, fill_method=None)
        result["volatility_20d"] = daily.groupby(result[data.entity_field], sort=False).transform(
            lambda values: values.rolling(p.volatility_window, min_periods=max(5, p.volatility_window // 2)).std(ddof=0)
        )
        result["industry_controlled_return_residual"] = RelationAnomalyScanner._daily_residual(
            result, "return_5d", ["log_total_mv", "volatility_20d"],
            p.min_industry_size, [data.date_field, data.industry_field],
        )
        valid_group = result.groupby([data.date_field, data.industry_field], dropna=False)[
            "industry_controlled_return_residual"
        ].transform("count").ge(p.min_industry_size)
        result["industry_residual_percentile"] = result.groupby(
            [data.date_field, data.industry_field], dropna=False
        )["industry_controlled_return_residual"].rank(pct=True, method="average").where(valid_group)
        eligible = RelationAnomalyScanner._eligible(result, template)
        strong = result["industry_residual_percentile"].ge(p.upper_percentile_gte)
        weak = result["industry_residual_percentile"].le(p.lower_percentile_lte)
        result["is_event"] = eligible & (strong | weak)
        result["event_subtype"] = np.select(
            [strong, weak], ["stock_strong_industry_weak", "stock_weak_industry_strong"],
            default="none",
        )
        result["severity"] = np.maximum(
            (result["industry_residual_percentile"] - p.upper_percentile_gte)
            / max(1 - p.upper_percentile_gte, 1e-9),
            (p.lower_percentile_lte - result["industry_residual_percentile"])
            / max(p.lower_percentile_lte, 1e-9),
        ).clip(lower=0)
        return result, {
            "upper_industry_residual_percentile_gte": p.upper_percentile_gte,
            "lower_industry_residual_percentile_lte": p.lower_percentile_lte,
            "controls": "industry_intercept+log_total_mv+volatility_20d",
        }, [
            "return_5d", "volatility_20d", "industry_controlled_return_residual",
            "industry_residual_percentile", "event_subtype", "severity",
        ], []

    @staticmethod
    def _volatility_compression_breakout(
        frame: pd.DataFrame, template: VolatilityCompressionBreakoutTemplate
    ) -> tuple[pd.DataFrame, dict, list[str], list[tuple[str, int, int]]]:
        data, p = template.data, template.parameters
        result = frame.copy()
        high = pd.to_numeric(result["adj_high"], errors="coerce")
        low = pd.to_numeric(result["adj_low"], errors="coerce")
        close = pd.to_numeric(result["adj_close"], errors="coerce")
        grouped_close = close.groupby(result[data.entity_field], sort=False)
        previous_close = grouped_close.shift(1)
        true_range = pd.concat([
            high - low, (high - previous_close).abs(), (low - previous_close).abs()
        ], axis=1).max(axis=1)
        result["normalized_atr_14"] = true_range.groupby(
            result[data.entity_field], sort=False
        ).transform(lambda values: values.rolling(p.atr_window, min_periods=p.atr_window).mean()) / close
        rolling_high = high.groupby(result[data.entity_field], sort=False).transform(
            lambda values: values.rolling(p.range_window, min_periods=p.range_window).max()
        )
        rolling_low = low.groupby(result[data.entity_field], sort=False).transform(
            lambda values: values.rolling(p.range_window, min_periods=p.range_window).min()
        )
        result["range_5d_normalized"] = (rolling_high - rolling_low) / close
        prior_high = close.groupby(result[data.entity_field], sort=False).transform(
            lambda values: values.shift(1).rolling(p.breakout_window, min_periods=p.breakout_window).max()
        )
        result["breakout_strength"] = close / prior_high - 1.0
        result["amount_value"] = pd.to_numeric(result["amount_cny"], errors="coerce")
        for field, output in [
            ("normalized_atr_14", "atr_history_percentile"),
            ("range_5d_normalized", "range_history_percentile"),
        ]:
            result[output] = pit_rolling_percentile(
                result, field, entity_column=data.entity_field, date_column=data.date_field,
                window=p.state_history.window, min_periods=p.state_history.min_periods,
            )
        result["breakout_history_percentile"] = pit_rolling_percentile(
            result, "breakout_strength", entity_column=data.entity_field, date_column=data.date_field,
            window=p.breakout_history.window, min_periods=p.breakout_history.min_periods,
        )
        result["amount_history_percentile"] = pit_rolling_percentile(
            result, "amount_value", entity_column=data.entity_field, date_column=data.date_field,
            window=p.amount_history.window, min_periods=p.amount_history.min_periods,
        )
        eligible = RelationAnomalyScanner._eligible(result, template)
        compression = (
            result["atr_history_percentile"].le(p.compression_percentile_lte)
            & result["range_history_percentile"].le(p.compression_percentile_lte)
        )
        breakout = result["breakout_history_percentile"].ge(p.breakout_percentile_gte)
        result["is_event"] = eligible & compression & breakout
        result["event_subtype"] = np.select(
            [
                result["amount_history_percentile"].ge(p.expanded_volume_percentile_gte),
                result["amount_history_percentile"].le(p.contracted_volume_percentile_lte),
            ],
            ["expanded_volume_breakout", "contracted_volume_breakout"], default="neutral_volume_breakout",
        )
        result["severity"] = (
            (p.compression_percentile_lte - result["atr_history_percentile"])
            / max(p.compression_percentile_lte, 1e-9)
            + (p.compression_percentile_lte - result["range_history_percentile"])
            / max(p.compression_percentile_lte, 1e-9)
            + (result["breakout_history_percentile"] - p.breakout_percentile_gte)
            / max(1 - p.breakout_percentile_gte, 1e-9)
        ).clip(lower=0)
        return result, {
            "compression_percentile_lte": p.compression_percentile_lte,
            "breakout_history_percentile_gte": p.breakout_percentile_gte,
            "volume_subtypes": "expanded|contracted|neutral",
        }, [
            "normalized_atr_14", "range_5d_normalized", "breakout_strength",
            "atr_history_percentile", "range_history_percentile",
            "breakout_history_percentile", "amount_history_percentile",
            "event_subtype", "severity",
        ], [
            ("normalized_atr_14", p.state_history.window, p.state_history.min_periods),
            ("range_5d_normalized", p.state_history.window, p.state_history.min_periods),
            ("breakout_strength", p.breakout_history.window, p.breakout_history.min_periods),
        ]

    @staticmethod
    def _trend_exhaustion(
        frame: pd.DataFrame, template: TrendExhaustionTemplate
    ) -> tuple[pd.DataFrame, dict, list[str], list[tuple[str, int, int]]]:
        data, p = template.data, template.parameters
        result = frame.copy()
        close = pd.to_numeric(result["adj_close"], errors="coerce")
        grouped = close.groupby(result[data.entity_field], sort=False)
        result["return_long"] = grouped.pct_change(p.long_horizon, fill_method=None)
        result["return_short"] = grouped.pct_change(p.short_horizon, fill_method=None)
        result["return_acceleration"] = (
            result["return_short"] - (p.short_horizon / p.long_horizon) * result["return_long"]
        )
        for field, output in [
            ("return_long", "return_long_history_percentile"),
            ("return_acceleration", "acceleration_history_percentile"),
        ]:
            result[output] = pit_rolling_percentile(
                result, field, entity_column=data.entity_field, date_column=data.date_field,
                window=p.history.window, min_periods=p.history.min_periods,
            )
        eligible = RelationAnomalyScanner._eligible(result, template)
        up = (
            result["return_long_history_percentile"].ge(p.strong_percentile_gte)
            & result["return_short"].gt(0)
            & result["acceleration_history_percentile"].le(p.acceleration_extreme)
        )
        down = (
            result["return_long_history_percentile"].le(p.weak_percentile_lte)
            & result["return_short"].lt(0)
            & result["acceleration_history_percentile"].ge(1 - p.acceleration_extreme)
        )
        result["is_event"] = eligible & (up | down)
        result["event_subtype"] = np.select(
            [up, down], ["uptrend_exhaustion", "downtrend_deceleration"], default="none"
        )
        result["severity"] = np.maximum(
            (result["return_long_history_percentile"] - p.strong_percentile_gte)
            / max(1 - p.strong_percentile_gte, 1e-9)
            + (p.acceleration_extreme - result["acceleration_history_percentile"])
            / max(p.acceleration_extreme, 1e-9),
            (p.weak_percentile_lte - result["return_long_history_percentile"])
            / max(p.weak_percentile_lte, 1e-9)
            + (result["acceleration_history_percentile"] - (1 - p.acceleration_extreme))
            / max(p.acceleration_extreme, 1e-9),
        ).clip(lower=0)
        return result, {
            "strong_return_percentile_gte": p.strong_percentile_gte,
            "weak_return_percentile_lte": p.weak_percentile_lte,
            "acceleration_tail": p.acceleration_extreme,
        }, [
            "return_long", "return_short", "return_acceleration",
            "return_long_history_percentile", "acceleration_history_percentile",
            "event_subtype", "severity",
        ], [
            ("return_long", p.history.window, p.history.min_periods),
            ("return_acceleration", p.history.window, p.history.min_periods),
        ]

    @staticmethod
    def _composite_anomaly(
        frame: pd.DataFrame, template: CompositeAnomalyTemplate
    ) -> tuple[pd.DataFrame, dict, list[str], list[tuple[str, int, int]]]:
        """Execute one of the frozen composite recipes without future labels."""
        data, p = template.data, template.parameters
        result = frame.copy()
        entity, date, industry = data.entity_field, data.date_field, data.industry_field
        close = pd.to_numeric(result["adj_close"], errors="coerce")
        grouped_close = close.groupby(result[entity], sort=False)
        result["return_1d"] = grouped_close.pct_change(1, fill_method=None)
        result["return_short"] = grouped_close.pct_change(p.short_window, fill_method=None)
        result["return_long"] = grouped_close.pct_change(p.long_window, fill_method=None)
        result["abs_return_1d"] = result["return_1d"].abs()
        eligible = RelationAnomalyScanner._eligible(result, template)
        recipe = p.recipe
        audits: list[tuple[str, int, int]] = []
        measurements: list[str] = []

        def pit(source: str, output: str, window=None, min_periods=None):
            window = window or p.history.window
            min_periods = min_periods or p.history.min_periods
            result[output] = pit_rolling_percentile(
                result, source, entity_column=entity, date_column=date,
                window=window, min_periods=min_periods,
            )
            audits.append((source, window, min_periods))

        if recipe == "turnover_residual_concentration":
            result = build_turnover_residual_concentration_features(
                result,
                universe_field=data.universe_field,
                liquidity_window=p.long_window,
                liquidity_min_periods=p.liquidity_min_periods,
                history_window=p.history.window,
                history_min_periods=p.history.min_periods,
                persistence_window=p.short_window,
                contributor_percentile=p.upper_percentile,
                concentration_percentile=p.upper_percentile,
                min_cross_section=p.min_cross_section,
            )
            event = (
                result["is_residual_contributor"]
                & result["residual_concentration_history_percentile"].ge(p.upper_percentile)
            )
            result["severity"] = (
                result["amount_residual_cross_section_percentile"] - p.upper_percentile
                + result["residual_concentration_history_percentile"] - p.upper_percentile
            ).clip(lower=0)
            measurements = [
                "top5_positive_amount_residual_mass_share",
                "residual_concentration_history_percentile",
                "amount_residual_cross_section_percentile",
                "industry_residual_mass_hhi",
                "top_size_decile_residual_mass_share",
                "contributor_return_sign_coherence",
                "contributor_volume_price_efficiency",
                "residual_concentration_persistence_5d",
                "severity",
            ]
            audits.append((
                "top5_positive_amount_residual_mass_share",
                p.history.window,
                p.history.min_periods,
            ))

        elif recipe == "index_up_breadth_down":
            amount = pd.to_numeric(result["amount_cny"], errors="coerce").clip(lower=0)
            result["return_contribution_proxy"] = result["return_short"] * amount
            result["contribution_percentile"] = result.groupby(date)[
                "return_contribution_proxy"
            ].rank(pct=True)
            daily = result.groupby(date, sort=True).agg(
                market_return_short=("return_short", "median"),
                advancing_ratio=("return_1d", lambda x: float(x.gt(0).mean())),
            )
            daily["breadth_change"] = daily["advancing_ratio"].diff(p.short_window)
            result = result.join(daily, on=date)
            pit("market_return_short", "market_return_percentile")
            event = (
                result["market_return_percentile"].ge(p.upper_percentile)
                & result["breadth_change"].le(-p.change_threshold)
                & result["contribution_percentile"].ge(0.90)
            )
            result["severity"] = (
                result["market_return_percentile"] - p.upper_percentile
                + (-result["breadth_change"] - p.change_threshold)
            ).clip(lower=0)
            measurements = ["market_return_short", "advancing_ratio", "breadth_change",
                            "return_contribution_proxy", "contribution_percentile",
                            "market_return_percentile", "severity"]

        elif recipe == "turnover_concentration":
            amount = pd.to_numeric(result["amount_cny"], errors="coerce").clip(lower=0)
            result["amount_value"] = amount
            result["amount_cross_section_percentile"] = result.groupby(date)["amount_value"].rank(pct=True)
            top_amount = amount.where(result["amount_cross_section_percentile"].ge(0.95), 0.0)
            totals = amount.groupby(result[date]).transform("sum").replace(0, np.nan)
            result["top5_amount_share"] = top_amount.groupby(result[date]).transform("sum") / totals
            pit("top5_amount_share", "concentration_history_percentile")
            event = (
                result["concentration_history_percentile"].ge(p.upper_percentile)
                & result["amount_cross_section_percentile"].ge(0.95)
            )
            result["severity"] = (
                result["concentration_history_percentile"] - p.upper_percentile
                + result["amount_cross_section_percentile"] - 0.95
            ).clip(lower=0)
            measurements = ["amount_value", "amount_cross_section_percentile", "top5_amount_share",
                            "concentration_history_percentile", "severity"]

        elif recipe == "turnover_displacement_close_context":
            result["turnover_value"] = pd.to_numeric(result["turnover_rate"], errors="coerce")
            high = pd.to_numeric(result["adj_high"], errors="coerce")
            low = pd.to_numeric(result["adj_low"], errors="coerce")
            result["close_position"] = (close - low) / (high - low).replace(0, np.nan)
            pit("turnover_value", "turnover_percentile")
            pit("abs_return_1d", "displacement_percentile")
            event = result["turnover_percentile"].ge(p.upper_percentile) & result[
                "displacement_percentile"
            ].le(p.lower_percentile)
            result["event_subtype"] = np.where(
                result["close_position"].ge(0.7), "strong_close_absorption", "weak_close_pressure"
            )
            result["severity"] = (
                result["turnover_percentile"] - p.upper_percentile
                + p.lower_percentile - result["displacement_percentile"]
            ).clip(lower=0)
            measurements = ["turnover_value", "abs_return_1d", "close_position",
                            "turnover_percentile", "displacement_percentile", "event_subtype", "severity"]

        elif recipe == "enhanced_stock_industry_residual":
            daily_return = result["return_1d"]
            result["volatility"] = daily_return.groupby(result[entity], sort=False).transform(
                lambda x: x.rolling(p.long_window, min_periods=max(5, p.long_window // 2)).std(ddof=0)
            )
            result["enhanced_industry_residual"] = RelationAnomalyScanner._daily_residual(
                result, "return_short", ["log_total_mv", "volatility"],
                p.min_cross_section, [date, industry],
            )
            result["residual_cross_section_percentile"] = result.groupby(
                [date, industry], dropna=False
            )["enhanced_industry_residual"].rank(pct=True)
            upper = result["residual_cross_section_percentile"].ge(p.upper_percentile)
            lower = result["residual_cross_section_percentile"].le(p.lower_percentile)
            event = upper | lower
            result["event_subtype"] = np.select([upper, lower], ["positive_residual", "negative_residual"], default="none")
            result["severity"] = np.maximum(
                result["residual_cross_section_percentile"] - p.upper_percentile,
                p.lower_percentile - result["residual_cross_section_percentile"],
            ).clip(lower=0)
            measurements = ["return_short", "volatility", "enhanced_industry_residual",
                            "residual_cross_section_percentile", "event_subtype", "severity"]

        elif recipe == "price_volume_conditional_residual":
            result["log_volume"] = np.log1p(pd.to_numeric(result["volume_shares"], errors="coerce").clip(lower=0))
            result["volume_conditional_residual"] = RelationAnomalyScanner._daily_residual(
                result, "log_volume", ["abs_return_1d", "log_total_mv"],
                p.min_cross_section, [date],
            )
            pit("abs_return_1d", "price_change_percentile")
            pit("volume_conditional_residual", "volume_residual_percentile")
            low_confirmation = result["price_change_percentile"].ge(p.upper_percentile) & result[
                "volume_residual_percentile"
            ].le(p.lower_percentile)
            high_churn = result["price_change_percentile"].le(p.lower_percentile) & result[
                "volume_residual_percentile"
            ].ge(p.upper_percentile)
            event = low_confirmation | high_churn
            result["event_subtype"] = np.select(
                [low_confirmation, high_churn], ["large_move_low_volume_residual", "high_volume_low_displacement"], default="none"
            )
            result["severity"] = np.maximum(
                result["price_change_percentile"] - result["volume_residual_percentile"],
                result["volume_residual_percentile"] - result["price_change_percentile"],
            ).clip(lower=0)
            measurements = ["abs_return_1d", "log_volume", "volume_conditional_residual",
                            "price_change_percentile", "volume_residual_percentile", "event_subtype", "severity"]

        elif recipe == "leader_industry_median_decoupling":
            result["size_rank_in_industry"] = result.groupby([date, industry])["log_total_mv"].rank(pct=True)
            result["industry_median_return"] = result.groupby([date, industry])["return_short"].transform("median")
            result["leader_median_gap"] = result["return_short"] - result["industry_median_return"]
            result["industry_advancing_ratio"] = result["return_1d"].gt(0).groupby(
                [result[date], result[industry]]
            ).transform("mean")
            pit("leader_median_gap", "leader_gap_percentile")
            event = (
                result["size_rank_in_industry"].ge(0.90)
                & result["leader_gap_percentile"].ge(p.upper_percentile)
                & result["industry_advancing_ratio"].le(0.50 - p.change_threshold / 2)
            )
            result["severity"] = (
                result["leader_gap_percentile"] - p.upper_percentile
                + (0.50 - result["industry_advancing_ratio"])
            ).clip(lower=0)
            measurements = ["return_short", "industry_median_return", "leader_median_gap",
                            "size_rank_in_industry", "industry_advancing_ratio", "leader_gap_percentile", "severity"]

        elif recipe == "failed_breakout":
            prior_high = close.groupby(result[entity], sort=False).transform(
                lambda x: x.shift(1).rolling(p.long_window, min_periods=p.long_window).max()
            )
            result["breakout_strength"] = close / prior_high - 1
            result["recent_breakout_strength"] = result["breakout_strength"].groupby(
                result[entity], sort=False
            ).transform(lambda x: x.shift(1).rolling(p.short_window, min_periods=1).max())
            result["fallback_depth"] = close / prior_high - 1
            event = result["recent_breakout_strength"].gt(0) & result["fallback_depth"].lt(0)
            result["severity"] = (-result["fallback_depth"] + result["recent_breakout_strength"]).clip(lower=0)
            measurements = ["breakout_strength", "recent_breakout_strength", "fallback_depth", "severity"]

        elif recipe == "momentum_participation_deterioration":
            positive = result["return_1d"].gt(0).astype(float)
            result["positive_day_ratio"] = positive.groupby(result[entity], sort=False).transform(
                lambda x: x.rolling(p.long_window, min_periods=p.long_window).mean()
            )
            result["participation_change"] = result["positive_day_ratio"] - result[
                "positive_day_ratio"
            ].groupby(result[entity], sort=False).shift(p.short_window)
            pit("return_long", "momentum_percentile")
            event = result["momentum_percentile"].ge(p.upper_percentile) & result[
                "participation_change"
            ].le(-p.change_threshold)
            result["severity"] = (
                result["momentum_percentile"] - p.upper_percentile
                + (-result["participation_change"] - p.change_threshold)
            ).clip(lower=0)
            measurements = ["return_long", "positive_day_ratio", "participation_change",
                            "momentum_percentile", "severity"]

        elif recipe == "percentile_rapid_migration":
            high = pd.to_numeric(result["adj_high"], errors="coerce")
            low = pd.to_numeric(result["adj_low"], errors="coerce")
            previous = grouped_close.shift(1)
            true_range = pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)
            result["normalized_atr"] = true_range.groupby(result[entity], sort=False).transform(
                lambda x: x.rolling(p.long_window, min_periods=p.long_window).mean()
            ) / close
            pit("normalized_atr", "atr_percentile")
            result["percentile_velocity"] = result["atr_percentile"] - result[
                "atr_percentile"
            ].groupby(result[entity], sort=False).shift(p.short_window)
            event = result["percentile_velocity"].abs().ge(p.change_threshold)
            result["event_subtype"] = np.where(result["percentile_velocity"].ge(0), "rapid_rise", "rapid_fall")
            result["severity"] = (result["percentile_velocity"].abs() - p.change_threshold).clip(lower=0)
            measurements = ["normalized_atr", "atr_percentile", "percentile_velocity", "event_subtype", "severity"]

        else:  # short_long_percentile_conflict
            result["amount_value"] = pd.to_numeric(result["amount_cny"], errors="coerce")
            short_history = max(20, p.short_window * 12)
            short_min = max(10, short_history // 2)
            pit("amount_value", "short_amount_percentile", short_history, short_min)
            pit("amount_value", "long_amount_percentile")
            result["short_long_percentile_gap"] = result["short_amount_percentile"] - result["long_amount_percentile"]
            event = result["short_long_percentile_gap"].abs().ge(p.change_threshold)
            result["event_subtype"] = np.where(result["short_long_percentile_gap"].ge(0), "recent_only_extreme", "long_term_only_extreme")
            result["severity"] = (result["short_long_percentile_gap"].abs() - p.change_threshold).clip(lower=0)
            measurements = ["amount_value", "short_amount_percentile", "long_amount_percentile",
                            "short_long_percentile_gap", "event_subtype", "severity"]

        result["is_event"] = eligible & event.fillna(False)
        return result, {
            "recipe": recipe,
            "upper_percentile": p.upper_percentile,
            "lower_percentile": p.lower_percentile,
            "change_threshold": p.change_threshold,
            "strict_prior_history": "true",
        }, measurements, audits

    @staticmethod
    def _eligible(frame: pd.DataFrame, template: RadarTemplate) -> pd.Series:
        mask = frame[template.data.universe_field].fillna(False).astype(bool)
        filters = template.filters
        if filters.min_listing_days:
            mask &= pd.to_numeric(frame["listing_trade_days"], errors="coerce").ge(filters.min_listing_days)
        if filters.exclude_st:
            mask &= ~frame["is_st"].fillna(True).astype(bool)
        if filters.exclude_suspended:
            mask &= ~frame["is_suspended"].fillna(True).astype(bool)
        if filters.exclude_limit_locked:
            mask &= ~(
                frame["is_limit_up_open"].fillna(False).astype(bool)
                | frame["is_limit_down_open"].fillna(False).astype(bool)
            )
        return mask

    @staticmethod
    def _daily_residual(
        frame: pd.DataFrame,
        target: str,
        controls: list[str],
        min_samples: int,
        group_fields: list[str],
    ) -> pd.Series:
        output = pd.Series(np.nan, index=frame.index, dtype=float)
        for _, group in frame.groupby(group_fields, dropna=False, sort=False):
            usable = group[[target, *controls]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(usable) < max(min_samples, len(controls) + 2):
                continue
            x = np.column_stack([np.ones(len(usable)), usable[controls].to_numpy(dtype=float)])
            y = usable[target].to_numpy(dtype=float)
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            output.loc[usable.index] = y - x @ beta
        return output

    def _build_result(
        self,
        measured: pd.DataFrame,
        template: RadarTemplate,
        *,
        data_version: str,
        as_of_date: pd.Timestamp,
        condition_values: dict,
        measurement_fields: list[str],
        duplicate_keys: int,
        audit_specs: list[tuple[str, int, int]],
    ) -> RadarScanResult:
        data = template.data
        unique_dates = sorted(pd.to_datetime(measured[data.date_field]).unique())
        discovery_dates = unique_dates[-template.scan.discovery_window_days:]
        recent_dates = discovery_dates[-template.scan.recent_window_days:]
        historical_dates = discovery_dates[:-template.scan.recent_window_days]
        discovery_mask = measured[data.date_field].isin(discovery_dates)
        eligible = self._eligible(measured, template)
        analysis_mask = discovery_mask & eligible
        recent_mask = measured[data.date_field].isin(recent_dates) & eligible
        historical_mask = measured[data.date_field].isin(historical_dates) & eligible
        event_mask = analysis_mask & measured["is_event"].fillna(False)
        events = measured.loc[event_mask].copy()

        event_count = len(events)
        all_industries = measured.loc[analysis_mask, data.industry_field].dropna().nunique()
        event_industries = events[data.industry_field].dropna().nunique()
        recent_rate = self._rate(measured.loc[recent_mask, "is_event"])
        historical_rate = self._rate(measured.loc[historical_mask, "is_event"])
        ratio = recent_rate / historical_rate if recent_rate is not None and historical_rate not in (None, 0) else None
        scan_date = discovery_dates[-1]
        scan_mask = measured[data.date_field].eq(scan_date) & eligible
        scan_count = int(measured.loc[scan_mask, "is_event"].fillna(False).astype(bool).sum())
        scan_rate = self._rate(measured.loc[scan_mask, "is_event"])
        daily_rates = (
            measured.loc[analysis_mask]
            .groupby(data.date_field)["is_event"]
            .mean()
            .sort_index()
        )
        prior_rates = daily_rates.iloc[:-1].tail(60)
        rate_std = float(prior_rates.std(ddof=0)) if len(prior_rates) >= 20 else np.nan
        rolling_rate_z = (
            float((daily_rates.iloc[-1] - prior_rates.mean()) / rate_std)
            if len(daily_rates) and np.isfinite(rate_std) and rate_std > 0 else None
        )
        entity_shares = events[data.entity_field].value_counts(normalize=True) if event_count else pd.Series(dtype=float)
        industry_shares = events[data.industry_field].value_counts(normalize=True) if event_count else pd.Series(dtype=float)

        audit_sample = measured.loc[
            measured[data.entity_field].isin(measured[data.entity_field].drop_duplicates().head(20))
        ]
        audit_passed = all(
            temporal_prefix_audit(
                audit_sample, field, entity_column=data.entity_field, date_column=data.date_field,
                window=window, min_periods=min_periods,
            )
            for field, window, min_periods in audit_specs
        )
        missing_rates = {
            field: float(measured.loc[analysis_mask, field].isna().mean()) if analysis_mask.any() else 1.0
            for field in measurement_fields
        }
        definition_hash = template.definition_hash()
        as_of_text = as_of_date.strftime("%Y-%m-%d")
        observation_id = observation_id_for(template, data_version, as_of_text)

        event_fields = [
            data.date_field, data.entity_field, data.industry_field,
            *measurement_fields, "is_event", "template_id",
        ]
        events["template_id"] = template.id
        events = events[event_fields].reset_index(drop=True)
        severity = pd.to_numeric(events["severity"], errors="coerce") if event_count else pd.Series(dtype=float)
        dates = pd.to_datetime(events[data.date_field]) if event_count else pd.Series(dtype="datetime64[ns]")
        evidence = ObservationEvidence(
            event_count=event_count,
            unique_entities=int(events[data.entity_field].nunique()),
            unique_industries=int(event_industries),
            industry_coverage=float(event_industries / all_industries) if all_industries else None,
            recent_event_rate=recent_rate,
            historical_event_rate=historical_rate,
            event_rate_ratio=ratio,
            severity_median=float(severity.median()) if severity.notna().any() else None,
            severity_p90=float(severity.quantile(0.9)) if severity.notna().any() else None,
            max_entity_share=float(entity_shares.iloc[0]) if not entity_shares.empty else None,
            max_industry_share=float(industry_shares.iloc[0]) if not industry_shares.empty else None,
            event_date_start=dates.min().strftime("%Y-%m-%d") if event_count else None,
            event_date_end=dates.max().strftime("%Y-%m-%d") if event_count else None,
            scan_date_event_count=scan_count,
            scan_date_event_rate=scan_rate,
            rolling_event_rate_zscore=rolling_rate_z,
        )
        quality_failures = self._quality_gate_failures(evidence, template)
        card = ObservationCard(
            observation_id=observation_id,
            definition=ObservationDefinition(
                id=template.id, version=template.version, kind=template.kind,
                description=template.description, definition_hash=definition_hash,
            ),
            discovered_at=utc_now(), data_version=data_version, as_of_date=as_of_text,
            universe=data.universe_field,
            discovery_window_days=template.scan.discovery_window_days,
            recent_window_days=template.scan.recent_window_days,
            conditions=condition_values,
            evidence=evidence,
            quality=ObservationQuality(
                input_rows=len(measured), eligible_rows=int(analysis_mask.sum()),
                duplicate_keys=duplicate_keys, measurement_missing_rates=missing_rates,
                temporal_audit_passed=audit_passed,
                quality_gate_passed=not quality_failures,
                quality_gate_failures=quality_failures,
            ),
            event_fields=event_fields,
        )
        return RadarScanResult(card, events)

    @staticmethod
    def _rate(values: pd.Series) -> float | None:
        return float(values.fillna(False).astype(bool).mean()) if len(values) else None

    @staticmethod
    def _quality_gate_failures(evidence: ObservationEvidence, template: RadarTemplate) -> list[str]:
        gate = template.quality_gate
        failures = []
        if evidence.event_count < gate.min_events:
            failures.append(f"event_count<{gate.min_events}")
        if evidence.unique_entities < gate.min_unique_stocks:
            failures.append(f"unique_stocks<{gate.min_unique_stocks}")
        if evidence.unique_industries < gate.min_unique_industries:
            failures.append(f"unique_industries<{gate.min_unique_industries}")
        if evidence.max_industry_share is not None and evidence.max_industry_share > gate.max_industry_share:
            failures.append(f"max_industry_share>{gate.max_industry_share}")
        if evidence.max_entity_share is not None and evidence.max_entity_share > gate.max_single_stock_share:
            failures.append(f"max_single_stock_share>{gate.max_single_stock_share}")
        return failures
