from __future__ import annotations

import numpy as np
import pandas as pd

from .models import (
    ActiveBox,
    BoxState,
    BreakoutConfig,
    BreakoutRunResult,
    ColumnMap,
    FactorStage,
    OperatorContext,
)
from .operators import OperatorRegistry, _slope, default_operator_registry


class BreakoutProcessEngine:
    """Detect frozen consolidation boxes and emit point-in-time breakout events."""

    def __init__(
        self,
        config: BreakoutConfig | None = None,
        *,
        columns: ColumnMap | None = None,
        operators: OperatorRegistry | None = None,
    ) -> None:
        self.config = config or BreakoutConfig()
        self.columns = columns or ColumnMap()
        self.operators = operators or default_operator_registry()

    def run(self, panel: pd.DataFrame) -> BreakoutRunResult:
        normalized = self._normalize(panel)
        boxes: list[dict] = []
        daily: list[dict] = []
        events: list[dict] = []

        for security, frame in normalized.groupby(self.columns.security, sort=False):
            security_boxes, security_daily, security_events = self._run_security(
                str(security), frame.reset_index(drop=True)
            )
            boxes.extend(security_boxes)
            daily.extend(security_daily)
            events.extend(security_events)

        return BreakoutRunResult(
            boxes=pd.DataFrame(boxes),
            daily_features=pd.DataFrame(daily),
            events=pd.DataFrame(events),
        )

    def _normalize(self, panel: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.columns.required()) - set(panel.columns))
        if missing:
            raise ValueError(f"breakout process panel is missing columns: {missing}")
        data = panel.loc[:, list(self.columns.required())].copy()
        data[self.columns.date] = pd.to_datetime(data[self.columns.date])
        numeric = (
            self.columns.open,
            self.columns.high,
            self.columns.low,
            self.columns.close,
            self.columns.volume,
        )
        for column in numeric:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[list(numeric)].isna().any().any():
            raise ValueError("breakout process input contains missing or non-numeric OHLCV values")
        if (data[[self.columns.open, self.columns.high, self.columns.low, self.columns.close]] <= 0).any().any():
            raise ValueError("breakout process prices must be positive")
        if (data[self.columns.volume] < 0).any():
            raise ValueError("breakout process volume cannot be negative")
        data = data.sort_values([self.columns.security, self.columns.date], kind="stable")
        duplicates = data.duplicated([self.columns.security, self.columns.date])
        if duplicates.any():
            raise ValueError("breakout process input has duplicate security/date rows")
        return data.reset_index(drop=True)

    def _run_security(
        self, security: str, frame: pd.DataFrame
    ) -> tuple[list[dict], list[dict], list[dict]]:
        boxes: list[dict] = []
        daily: list[dict] = []
        events: list[dict] = []
        active: ActiveBox | None = None

        for position in range(len(frame)):
            current = frame.iloc[position]
            current_date = pd.Timestamp(current[self.columns.date])
            history = frame.iloc[:position]

            if active is None:
                active = self._try_create_box(security, history, current_date, position)
                if active is None:
                    continue

            age = position - active.created_position
            pre_context = OperatorContext(
                history=history,
                box=active,
                columns=self.columns,
                config=self.config,
                as_of=pd.Timestamp(history.iloc[-1][self.columns.date]),
                box_age=age,
            )
            pre_features = self.operators.compute(FactorStage.PRE_BREAKOUT, pre_context)

            close = float(current[self.columns.close])
            upper_trigger = active.upper + self.config.breakout_buffer_atr * active.frozen_atr
            lower_trigger = active.lower - self.config.failure_buffer_atr * active.frozen_atr

            if close > upper_trigger:
                active.state = BoxState.TRIGGERED
                active.closed_at = current_date
                active.close_reason = "breakout"
                event_context = OperatorContext(
                    history=history,
                    current=current,
                    box=active,
                    columns=self.columns,
                    config=self.config,
                    as_of=current_date,
                    box_age=age,
                )
                event_features = self.operators.compute(FactorStage.BREAKOUT, event_context)
                events.append(
                    {
                        **self._identity(active),
                        "event_id": f"{active.box_id}:breakout",
                        "event_time": current_date,
                        "observation_time": current_date,
                        "available_time": current_date,
                        "pre_window_end": pre_context.as_of,
                        **active.setup_features,
                        **pre_features,
                        **event_features,
                    }
                )
                boxes.append(self._box_record(active))
                active = None
                continue

            if close < lower_trigger:
                active.state = BoxState.CLOSED
                active.closed_at = current_date
                active.close_reason = "downside_failure"
                boxes.append(self._box_record(active))
                active = None
                continue

            if age >= self.config.max_active_days:
                active.state = BoxState.CLOSED
                active.closed_at = current_date
                active.close_reason = "expired"
                boxes.append(self._box_record(active))
                active = None
                continue

            end_of_day_history = frame.iloc[: position + 1]
            daily_context = OperatorContext(
                history=end_of_day_history,
                box=active,
                columns=self.columns,
                config=self.config,
                as_of=current_date,
                box_age=age,
            )
            daily.append(
                {
                    **self._identity(active),
                    "observation_time": current_date,
                    "available_time": current_date,
                    "state": active.state.value,
                    **active.setup_features,
                    **self.operators.compute(FactorStage.PRE_BREAKOUT, daily_context),
                }
            )

        if active is not None:
            boxes.append(self._box_record(active))
        return boxes, daily, events

    def _try_create_box(
        self,
        security: str,
        history: pd.DataFrame,
        current_date: pd.Timestamp,
        position: int,
    ) -> ActiveBox | None:
        required = max(
            self.config.box_lookback,
            self.config.atr_window + 1,
            self.config.volatility_long_window + 1,
        )
        if len(history) < required:
            return None

        source = history.iloc[-self.config.box_lookback :]
        atr = self._atr(history)
        if not np.isfinite(atr) or atr <= 0:
            return None
        upper = float(source[self.columns.high].max())
        lower = float(source[self.columns.low].min())
        width_atr = (upper - lower) / atr
        closes = source[self.columns.close].to_numpy(dtype=float)
        abs_slope_atr = abs(_slope(closes)) / atr
        volatility_ratio = self._volatility_ratio(history)

        if width_atr > self.config.max_box_width_atr:
            return None
        if abs_slope_atr > self.config.max_abs_slope_atr:
            return None
        maximum_ratio = self.config.max_volatility_ratio
        if maximum_ratio is not None and volatility_ratio > maximum_ratio:
            return None

        source_start = pd.Timestamp(source.iloc[0][self.columns.date])
        source_end = pd.Timestamp(source.iloc[-1][self.columns.date])
        box_id = f"{security}:{source_end.strftime('%Y%m%d')}"
        box = ActiveBox(
            box_id=box_id,
            security=security,
            upper=upper,
            lower=lower,
            frozen_atr=atr,
            source_start=source_start,
            source_end=source_end,
            created_at=current_date,
            created_position=position,
        )
        setup_history_size = max(
            self.config.box_lookback,
            self.config.volatility_long_window + 1,
        )
        setup_context = OperatorContext(
            history=history.iloc[-setup_history_size:],
            box=box,
            columns=self.columns,
            config=self.config,
            as_of=source_end,
        )
        box.setup_features.update(self.operators.compute(FactorStage.SETUP, setup_context))
        return box

    def _atr(self, history: pd.DataFrame) -> float:
        window = self.config.atr_window
        recent = history.iloc[-(window + 1) :]
        high = recent[self.columns.high].to_numpy(dtype=float)[1:]
        low = recent[self.columns.low].to_numpy(dtype=float)[1:]
        previous_close = recent[self.columns.close].to_numpy(dtype=float)[:-1]
        true_range = np.maximum.reduce(
            (high - low, np.abs(high - previous_close), np.abs(low - previous_close))
        )
        return float(np.mean(true_range))

    def _volatility_ratio(self, history: pd.DataFrame) -> float:
        long_window = self.config.volatility_long_window
        short_window = self.config.volatility_short_window
        closes = history[self.columns.close].iloc[-(long_window + 1) :]
        returns = closes.pct_change().dropna().to_numpy(dtype=float)
        short_vol = float(np.std(returns[-short_window:], ddof=0))
        long_vol = float(np.std(returns[-long_window:], ddof=0))
        if long_vol == 0:
            return 0.0 if short_vol == 0 else np.inf
        return short_vol / long_vol

    @staticmethod
    def _identity(box: ActiveBox) -> dict:
        return {
            "box_id": box.box_id,
            "ts_code": box.security,
            "upper": box.upper,
            "lower": box.lower,
            "frozen_atr": box.frozen_atr,
            "box_source_start": box.source_start,
            "box_source_end": box.source_end,
            "box_created_at": box.created_at,
        }

    def _box_record(self, box: ActiveBox) -> dict:
        record = self._identity(box)
        record.update(
            {
                "state": box.state.value,
                "closed_at": box.closed_at,
                "close_reason": box.close_reason,
                **box.setup_features,
            }
        )
        return record
