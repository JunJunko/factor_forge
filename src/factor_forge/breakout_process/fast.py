from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

import numpy as np
import pandas as pd

from .models import BoxState, BreakoutConfig, BreakoutRunResult, ColumnMap
from .operators import _slope


def _rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=float)
    if len(values) < window:
        return result
    x = np.arange(window, dtype=float)
    x -= x.mean()
    denominator = float(np.dot(x, x))
    result[window - 1 :] = np.correlate(values, x, mode="valid") / denominator
    return result


class BreakoutEventBuilder:
    """Fast default-factor event builder for full-market research scheduling.

    Unlike :class:`BreakoutProcessEngine`, this builder intentionally emits no daily
    active-box snapshots and does not support custom operators. Its event and box
    formulas match the default registry.
    """

    def __init__(
        self,
        config: BreakoutConfig | None = None,
        *,
        columns: ColumnMap | None = None,
        workers: int = 1,
    ) -> None:
        self.config = config or BreakoutConfig()
        self.columns = columns or ColumnMap()
        if workers <= 0:
            raise ValueError("workers must be positive")
        self.workers = int(workers)

    def run(self, panel: pd.DataFrame) -> BreakoutRunResult:
        data = self._normalize(panel)
        boxes: list[dict] = []
        events: list[dict] = []
        groups = data.groupby(self.columns.security, sort=False)
        if self.workers == 1:
            for security, frame in groups:
                security_boxes, security_events = self._run_security(
                    str(security), frame.reset_index(drop=True)
                )
                boxes.extend(security_boxes)
                events.extend(security_events)
        else:
            self._run_parallel(groups, boxes, events)
        return BreakoutRunResult(
            boxes=pd.DataFrame(boxes),
            daily_features=pd.DataFrame(),
            events=pd.DataFrame(events),
        )

    def _run_parallel(self, groups, boxes: list[dict], events: list[dict]) -> None:
        pending: set[Future] = set()
        maximum_pending = self.workers * 2
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            for security, frame in groups:
                pending.add(
                    executor.submit(
                        self._run_security,
                        str(security),
                        frame.reset_index(drop=True),
                    )
                )
                if len(pending) >= maximum_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        security_boxes, security_events = future.result()
                        boxes.extend(security_boxes)
                        events.extend(security_events)
            for future in pending:
                security_boxes, security_events = future.result()
                boxes.extend(security_boxes)
                events.extend(security_events)

    def _normalize(self, panel: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.columns.required()) - set(panel.columns))
        if missing:
            raise ValueError(f"breakout event panel is missing columns: {missing}")
        data = panel.loc[:, list(self.columns.required())].copy()
        data[self.columns.date] = pd.to_datetime(data[self.columns.date])
        numeric = list(self.columns.required()[2:])
        for column in numeric:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna(subset=numeric)
        prices = [self.columns.open, self.columns.high, self.columns.low, self.columns.close]
        data = data.loc[(data[prices] > 0).all(axis=1) & (data[self.columns.volume] >= 0)]
        data = data.sort_values([self.columns.security, self.columns.date], kind="stable")
        if data.duplicated([self.columns.security, self.columns.date]).any():
            raise ValueError("breakout event input has duplicate security/date rows")
        return data.reset_index(drop=True)

    def _run_security(self, security: str, frame: pd.DataFrame) -> tuple[list[dict], list[dict]]:
        c = self.columns
        cfg = self.config
        dates = pd.to_datetime(frame[c.date]).to_numpy()
        open_price = frame[c.open].to_numpy(dtype=float)
        high = frame[c.high].to_numpy(dtype=float)
        low = frame[c.low].to_numpy(dtype=float)
        close = frame[c.close].to_numpy(dtype=float)
        volume = frame[c.volume].to_numpy(dtype=float)
        count = len(frame)
        if count == 0:
            return [], []

        true_range = np.full(count, np.nan)
        true_range[1:] = np.maximum.reduce(
            (high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        atr = pd.Series(true_range).rolling(cfg.atr_window, min_periods=cfg.atr_window).mean().to_numpy()
        rolling_upper = pd.Series(high).rolling(cfg.box_lookback, min_periods=cfg.box_lookback).max().to_numpy()
        rolling_lower = pd.Series(low).rolling(cfg.box_lookback, min_periods=cfg.box_lookback).min().to_numpy()
        box_slope = _rolling_slope(close, cfg.box_lookback)
        returns = pd.Series(close).pct_change(fill_method=None)
        short_vol = returns.rolling(
            cfg.volatility_short_window, min_periods=cfg.volatility_short_window
        ).std(ddof=0).to_numpy()
        long_vol = returns.rolling(
            cfg.volatility_long_window, min_periods=cfg.volatility_long_window
        ).std(ddof=0).to_numpy()
        volatility_ratio = np.divide(
            short_vol,
            long_vol,
            out=np.full(count, np.nan),
            where=long_vol > 0,
        )
        both_zero = (short_vol == 0) & (long_vol == 0)
        volatility_ratio[both_zero] = 0.0

        required = max(cfg.box_lookback, cfg.atr_window + 1, cfg.volatility_long_window + 1)
        boxes: list[dict] = []
        events: list[dict] = []
        active: dict | None = None

        for position in range(required, count):
            if active is None:
                end = position - 1
                frozen_atr = atr[end]
                if not np.isfinite(frozen_atr) or frozen_atr <= 0:
                    continue
                width_atr = (rolling_upper[end] - rolling_lower[end]) / frozen_atr
                abs_slope_atr = abs(box_slope[end]) / frozen_atr
                ratio = volatility_ratio[end]
                if not np.isfinite(width_atr) or not np.isfinite(abs_slope_atr):
                    continue
                if width_atr > cfg.max_box_width_atr or abs_slope_atr > cfg.max_abs_slope_atr:
                    continue
                if cfg.max_volatility_ratio is not None:
                    if not np.isfinite(ratio) or ratio > cfg.max_volatility_ratio:
                        continue
                source_start = pd.Timestamp(dates[position - cfg.box_lookback])
                source_end = pd.Timestamp(dates[end])
                upper = float(rolling_upper[end])
                lower = float(rolling_lower[end])
                if ratio == 0:
                    contraction = np.inf
                elif np.isfinite(ratio):
                    contraction = float(-np.log(ratio))
                else:
                    contraction = np.nan
                active = {
                    "box_id": f"{security}:{source_end.strftime('%Y%m%d')}",
                    "ts_code": security,
                    "upper": upper,
                    "lower": lower,
                    "frozen_atr": float(frozen_atr),
                    "box_source_start": source_start,
                    "box_source_end": source_end,
                    "box_created_at": pd.Timestamp(dates[position]),
                    "created_position": position,
                    "range_compactness": -float(width_atr),
                    "volatility_contraction": contraction,
                    "trend_flatness": -float(abs_slope_atr),
                }

            age = position - int(active["created_position"])
            upper_trigger = active["upper"] + cfg.breakout_buffer_atr * active["frozen_atr"]
            lower_trigger = active["lower"] - cfg.failure_buffer_atr * active["frozen_atr"]
            identity = {key: value for key, value in active.items() if key != "created_position"}

            if close[position] > upper_trigger:
                pre = self._pre_features(close, position, active, age)
                event = self._event_features(
                    open_price, close, volume, position, active
                )
                event_time = pd.Timestamp(dates[position])
                events.append(
                    {
                        **identity,
                        "event_id": f"{active['box_id']}:breakout",
                        "event_time": event_time,
                        "observation_time": event_time,
                        "available_time": event_time,
                        "pre_window_end": pd.Timestamp(dates[position - 1]),
                        **pre,
                        **event,
                    }
                )
                boxes.append(
                    {
                        **identity,
                        "state": BoxState.TRIGGERED.value,
                        "closed_at": event_time,
                        "close_reason": "breakout",
                    }
                )
                active = None
            elif close[position] < lower_trigger:
                boxes.append(
                    {
                        **identity,
                        "state": BoxState.CLOSED.value,
                        "closed_at": pd.Timestamp(dates[position]),
                        "close_reason": "downside_failure",
                    }
                )
                active = None
            elif age >= cfg.max_active_days:
                boxes.append(
                    {
                        **identity,
                        "state": BoxState.CLOSED.value,
                        "closed_at": pd.Timestamp(dates[position]),
                        "close_reason": "expired",
                    }
                )
                active = None

        if active is not None:
            identity = {key: value for key, value in active.items() if key != "created_position"}
            boxes.append(
                {
                    **identity,
                    "state": BoxState.ACTIVE.value,
                    "closed_at": pd.NaT,
                    "close_reason": None,
                }
            )
        return boxes, events

    def _pre_features(
        self, close: np.ndarray, position: int, box: dict, age: int
    ) -> dict[str, float]:
        cfg = self.config
        scale = box["frozen_atr"]
        process = close[position - cfg.process_window : position]
        velocity = _slope(process) / scale
        k = cfg.acceleration_window
        differences = np.diff(close[position - (2 * k + 1) : position])
        acceleration = (float(np.mean(differences[k:])) - float(np.mean(differences[:k]))) / k / scale
        persistence_values = np.diff(close[position - (cfg.process_window + 1) : position])
        return {
            "approach_velocity": float(velocity),
            "pre_acceleration": float(acceleration),
            "direction_persistence": float(np.mean(persistence_values > 0)),
            "consolidation_age": float(cfg.box_lookback + age),
        }

    def _event_features(
        self,
        open_price: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        position: int,
        box: dict,
    ) -> dict[str, float]:
        cfg = self.config
        scale = box["frozen_atr"]
        velocity = (close[position] - close[position - 1]) / scale
        k = cfg.acceleration_window
        recent_velocity = float(np.mean(np.diff(close[position - (k + 1) : position]))) / scale
        baseline_volume = float(np.median(volume[position - cfg.volume_window : position]))
        relative = (
            float(np.log(volume[position] / baseline_volume))
            if baseline_volume > 0 and volume[position] > 0
            else np.nan
        )
        return {
            "breakout_strength": float((close[position] - box["upper"]) / scale),
            "breakout_velocity": float(velocity),
            "breakout_acceleration": float(velocity - recent_velocity),
            "relative_volume": relative,
            "gap_atr": float((open_price[position] - close[position - 1]) / scale),
        }
