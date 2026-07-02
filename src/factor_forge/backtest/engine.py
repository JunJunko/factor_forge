from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from factor_forge.config import CostModel, ExecutionConstraints


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    metrics: dict[str, Any]


@dataclass
class _Position:
    ts_code: str
    shares: int
    entry_date: pd.Timestamp
    due_date: pd.Timestamp | None
    entry_raw_open: float
    entry_adj_open: float
    signal_date: pd.Timestamp
    condition_quantile: int | None = None


@dataclass
class _Sleeve:
    cash: float
    positions: list[_Position]


class BacktestEngine:
    """T-close signals, T+1-open fills, fixed-duration overlapping cash sleeves."""

    def run(
        self,
        panel: pd.DataFrame,
        factor_values: pd.DataFrame,
        *,
        universe: str,
        top_n: int,
        holding_days: int,
        initial_cash: float,
        lot_size: int,
        constraints: ExecutionConstraints,
        cost_model: CostModel,
        cost_scenario_bps: float | None = None,
        market_benchmark: pd.DataFrame | None = None,
        selection_membership: pd.DataFrame | None = None,
    ) -> BacktestResult:
        data = panel.merge(
            factor_values[["trade_date", "ts_code", "factor_value"]],
            on=["trade_date", "ts_code"], how="left",
        )
        if selection_membership is not None:
            required = {"trade_date", "ts_code", "selection_eligible", "condition_quantile"}
            if not required <= set(selection_membership.columns):
                raise ValueError(
                    "selection_membership is missing columns: "
                    + ", ".join(sorted(required - set(selection_membership.columns)))
                )
            if selection_membership.duplicated(["trade_date", "ts_code"]).any():
                raise ValueError("selection_membership keys must be unique")
            data = data.merge(
                selection_membership[list(required)],
                on=["trade_date", "ts_code"], how="left",
            )
            data["selection_eligible"] = data["selection_eligible"].eq(True)
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        data = data.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        dates = list(pd.Index(data["trade_date"].unique()).sort_values())
        by_date = {date: frame.set_index("ts_code") for date, frame in data.groupby("trade_date")}
        mark_prices = self._mark_price_lookup(data)
        sleeves = [_Sleeve(initial_cash / holding_days, []) for _ in range(holding_days)]
        trades: list[dict] = []
        positions: list[dict] = []
        daily_rows: list[dict] = []
        generated = executed = 0
        for date_index, date in enumerate(dates):
            today = by_date[date]
            # Due sells happen before a sleeve is considered for its next batch.
            for sleeve_id, sleeve in enumerate(sleeves):
                remaining = []
                for position in sleeve.positions:
                    if position.due_date is None or date < position.due_date or not self._can_sell(today, position.ts_code, constraints):
                        remaining.append(position)
                        continue
                    value = self._position_value(position, date, mark_prices)
                    sell_cost = self._cost(value, "SELL", cost_model, cost_scenario_bps)
                    sleeve.cash += value - sell_cost
                    executed += 1
                    trades.append(self._trade_row(position, date, sleeve_id, "SELL", value, sell_cost, today))
                sleeve.positions = remaining
            if date_index >= 1:
                sleeve_id = (date_index - 1) % holding_days
                sleeve = sleeves[sleeve_id]
                signal_date = dates[date_index - 1]
                if not sleeve.positions:
                    signals = by_date[signal_date]
                    universe_column = f"is_{universe}"
                    candidate_mask = (
                        signals[universe_column].fillna(False).astype(bool)
                        & signals["factor_value"].notna()
                    )
                    if selection_membership is not None:
                        candidate_mask &= signals["selection_eligible"]
                    candidates = signals[candidate_mask].sort_values(
                        ["factor_value"], ascending=False
                    ).head(top_n)
                    generated += len(candidates)
                    target = sleeve.cash / top_n if top_n else 0.0
                    for ts_code in candidates.index:
                        if not self._can_buy(today, ts_code, constraints):
                            continue
                        row = today.loc[ts_code]
                        if isinstance(row, pd.DataFrame):
                            row = row.iloc[0]
                        price = float(row["raw_open"])
                        shares = int(target // (price * lot_size)) * lot_size
                        if shares <= 0:
                            continue
                        gross = shares * price
                        buy_cost = self._cost(gross, "BUY", cost_model, cost_scenario_bps)
                        if gross + buy_cost > sleeve.cash:
                            continue
                        sleeve.cash -= gross + buy_cost
                        due_index = date_index + holding_days
                        position = _Position(
                            ts_code, shares, date, dates[due_index] if due_index < len(dates) else None, price,
                            float(row["adj_open"]), signal_date,
                            int(candidates.loc[ts_code, "condition_quantile"])
                            if selection_membership is not None else None,
                        )
                        sleeve.positions.append(position)
                        executed += 1
                        trades.append(self._trade_row(position, date, sleeve_id, "BUY", gross, buy_cost, today))
            nav = 0.0
            gross_exposure = 0.0
            for sleeve_id, sleeve in enumerate(sleeves):
                nav += sleeve.cash
                for position in sleeve.positions:
                    value = self._position_value(position, date, mark_prices)
                    nav += value
                    gross_exposure += value
                    positions.append({
                        "trade_date": date, "sleeve_id": sleeve_id, "ts_code": position.ts_code,
                        "shares": position.shares, "market_value": value, "due_date": position.due_date,
                    })
            daily_rows.append({"trade_date": date, "nav": nav, "gross_exposure": gross_exposure})
        daily = pd.DataFrame(daily_rows)
        daily["return"] = daily["nav"].pct_change().fillna(0.0)
        if selection_membership is not None:
            daily["benchmark_return"] = self._benchmark_returns(
                data, universe, dates, eligibility_column="selection_eligible"
            )
            daily["universe_benchmark_return"] = self._benchmark_returns(data, universe, dates)
        else:
            daily["benchmark_return"] = self._benchmark_returns(data, universe, dates)
        if market_benchmark is not None and not market_benchmark.empty:
            daily["market_index_return"] = self._market_index_returns(market_benchmark, dates)
        daily["excess_return"] = daily["return"] - daily["benchmark_return"]
        trades_frame = pd.DataFrame(trades)
        positions_frame = pd.DataFrame(positions)
        metrics = self._metrics(daily, trades_frame, generated, executed)
        metrics["benchmark_scope"] = (
            "condition_equal_weight" if selection_membership is not None else "universe_equal_weight"
        )
        return BacktestResult(daily, trades_frame, positions_frame, metrics)

    @staticmethod
    def _mark_price_lookup(data: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], float]:
        ordered = data[["ts_code", "trade_date", "adj_open", "adj_close"]].sort_values(["ts_code", "trade_date"])
        base = ordered["adj_open"].fillna(ordered["adj_close"])
        ordered["mark"] = base.groupby(ordered["ts_code"]).ffill()
        return {(row.trade_date, row.ts_code): row.mark for row in ordered.itertuples()}

    @staticmethod
    def _position_value(position: _Position, date: pd.Timestamp, prices: dict) -> float:
        mark = prices.get((date, position.ts_code), np.nan)
        if not np.isfinite(mark):
            mark = position.entry_adj_open
        return position.shares * position.entry_raw_open * mark / position.entry_adj_open

    @staticmethod
    def _can_buy(today: pd.DataFrame, ts_code: str, constraints: ExecutionConstraints) -> bool:
        if ts_code not in today.index:
            return False
        row = today.loc[ts_code]
        if isinstance(row, pd.DataFrame): row = row.iloc[0]
        return bool(
            np.isfinite(row.get("raw_open", np.nan))
            and (not constraints.exclude_suspended or not row.get("is_suspended", True))
            and (not constraints.cannot_buy_limit_up or not row.get("is_limit_up_open", False))
            and (not constraints.exclude_st or not row.get("is_st", False))
            and (not constraints.exclude_delisting_period or not row.get("is_delisting_period", False))
            and row.get("listing_trade_days", 0) >= constraints.min_listing_days
        )

    @staticmethod
    def _can_sell(today: pd.DataFrame, ts_code: str, constraints: ExecutionConstraints) -> bool:
        if ts_code not in today.index:
            return False
        row = today.loc[ts_code]
        if isinstance(row, pd.DataFrame): row = row.iloc[0]
        return bool(
            np.isfinite(row.get("raw_open", np.nan))
            and (not constraints.exclude_suspended or not row.get("is_suspended", True))
            and (not constraints.cannot_sell_limit_down or not row.get("is_limit_down_open", False))
        )

    @staticmethod
    def _cost(value: float, side: str, model: CostModel, scenario: float | None) -> float:
        if scenario is not None:
            return value * (scenario / 2.0) / 10_000.0
        bps = model.commission_bps_per_side + model.slippage_bps_per_side
        if side == "SELL": bps += model.stamp_duty_bps_sell
        return value * bps / 10_000.0

    @staticmethod
    def _trade_row(position, date, sleeve_id, side, gross, cost, today) -> dict:
        row = today.loc[position.ts_code]
        if isinstance(row, pd.DataFrame): row = row.iloc[0]
        return {
            "trade_date": date, "signal_date": position.signal_date, "sleeve_id": sleeve_id,
            "ts_code": position.ts_code, "side": side, "shares": position.shares,
            "raw_open": row.get("raw_open"), "gross_value": gross, "cost": cost,
            "condition_quantile": position.condition_quantile,
        }

    @staticmethod
    def _benchmark_returns(
        data: pd.DataFrame,
        universe: str,
        dates: list,
        eligibility_column: str | None = None,
    ) -> list[float]:
        columns = ["ts_code", "trade_date", "adj_open", f"is_{universe}"]
        if eligibility_column is not None:
            columns.append(eligibility_column)
        ordered = data[columns].sort_values(["ts_code", "trade_date"])
        grouped = ordered.groupby("ts_code")["adj_open"]
        ordered["tradable_forward_return"] = grouped.shift(-2) / grouped.shift(-1) - 1
        benchmark_mask = ordered[f"is_{universe}"].fillna(False).astype(bool)
        if eligibility_column is not None:
            benchmark_mask &= ordered[eligibility_column].fillna(False).astype(bool)
        means = ordered[benchmark_mask].groupby("trade_date")["tradable_forward_return"].mean()
        # A universe known at T close enters T+1 open; its first return is visible at T+2 open.
        mapped = {dates[i + 2]: means.get(dates[i], 0.0) for i in range(len(dates) - 2)}
        return [float(mapped.get(date, 0.0) or 0.0) for date in dates]

    @staticmethod
    def _market_index_returns(benchmark: pd.DataFrame, dates: list) -> list[float]:
        frame = benchmark.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame = frame.sort_values("trade_date")
        frame["return"] = frame["open"].pct_change()
        values = frame.set_index("trade_date")["return"]
        return [float(values.get(date, 0.0) or 0.0) for date in dates]

    @staticmethod
    def _metrics(daily: pd.DataFrame, trades: pd.DataFrame, generated: int, executed: int) -> dict:
        count = max(len(daily) - 1, 1)
        total = daily["nav"].iloc[-1] / daily["nav"].iloc[0] - 1 if len(daily) else 0.0
        annual = (1 + total) ** (252 / count) - 1 if total > -1 else -1.0
        benchmark_total = float((1 + daily["benchmark_return"]).prod() - 1)
        benchmark_annual = (1 + benchmark_total) ** (252 / count) - 1 if benchmark_total > -1 else -1.0
        returns = daily["return"]
        volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
        running_max = daily["nav"].cummax()
        drawdown = daily["nav"] / running_max - 1
        max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
        buy_count = int((trades["side"] == "BUY").sum()) if not trades.empty else 0
        metrics = {
            "total_return": float(total), "annualized_return": float(annual),
            "benchmark_annualized_return": float(benchmark_annual),
            "annualized_excess_return": float(annual - benchmark_annual),
            "annualized_volatility": volatility,
            "sharpe": float(annual / volatility) if volatility > 0 else None,
            "max_drawdown": max_drawdown,
            "calmar": float(annual / abs(max_drawdown)) if max_drawdown < 0 else None,
            "generated_signals": int(generated), "executed_buys": buy_count,
            "execution_rate": float(buy_count / generated) if generated else 0.0,
            "turnover_notional": float(trades["gross_value"].sum()) if not trades.empty else 0.0,
        }
        if "universe_benchmark_return" in daily:
            universe_total = float((1 + daily["universe_benchmark_return"]).prod() - 1)
            universe_annual = (
                (1 + universe_total) ** (252 / count) - 1 if universe_total > -1 else -1.0
            )
            metrics["universe_benchmark_annualized_return"] = universe_annual
            metrics["annualized_excess_return_vs_universe"] = float(annual - universe_annual)
        if "market_index_return" in daily:
            index_total = float((1 + daily["market_index_return"]).prod() - 1)
            metrics["market_index_annualized_return"] = (
                (1 + index_total) ** (252 / count) - 1 if index_total > -1 else -1.0
            )
        return metrics
