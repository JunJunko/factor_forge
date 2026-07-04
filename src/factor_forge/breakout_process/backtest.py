from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from factor_forge.config import load_project
from factor_forge.data import DataVersionRepository


@dataclass(frozen=True)
class EventBacktestConfig:
    research_run: str
    project_config: str = "configs/project.yaml"
    initial_cash: float = 1_000_000.0
    holding_days: int = 10
    top_n: tuple[int, ...] = (5, 10, 20)
    cost_scenarios_bps: tuple[float, ...] = (0.0, 10.0, 20.0)
    lot_size: int = 100
    min_listing_days: int = 60
    output_subdir: str = "backtests"
    strategies: tuple[str, ...] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EventBacktestConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if "top_n" in raw:
            raw["top_n"] = tuple(int(value) for value in raw["top_n"])
        if "cost_scenarios_bps" in raw:
            raw["cost_scenarios_bps"] = tuple(float(value) for value in raw["cost_scenarios_bps"])
        if "strategies" in raw and raw["strategies"] is not None:
            raw["strategies"] = tuple(str(value) for value in raw["strategies"])
        return cls(**raw)


@dataclass
class _Holding:
    ts_code: str
    shares: int
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    due_index: int
    raw_entry: float
    adj_entry: float
    last_mark: float


@dataclass
class _Sleeve:
    cash: float
    holdings: list[_Holding] = field(default_factory=list)


STRATEGIES = {
    "pair_strong": {
        "score": "pair:approach_velocity+continuous_move",
        "pool": "strong",
        "all_candidates": False,
    },
    "approach_only_strong": {
        "score": "single:approach_velocity",
        "pool": "strong",
        "all_candidates": False,
    },
    "continuous_only_strong": {
        "score": "single:continuous_move",
        "pool": "strong",
        "all_candidates": False,
    },
    "strong_breakout_equal_weight": {
        "score": None,
        "pool": "strong",
        "all_candidates": True,
    },
    "pair_all_breakouts": {
        "score": "pair:approach_velocity+continuous_move",
        "pool": "all",
        "all_candidates": False,
    },
    "pair_shallow_breakouts": {
        "score": "pair:approach_velocity+continuous_move",
        "pool": "shallow",
        "all_candidates": False,
    },
    "pair_exclude_top20_strength": {
        "score": "pair:approach_velocity+continuous_move",
        "pool": "exclude_top20",
        "all_candidates": False,
    },
    "pair_middle60_strength": {
        "score": "pair:approach_velocity+continuous_move",
        "pool": "middle60",
        "all_candidates": False,
    },
    "pair_bottom20_strength": {
        "score": "pair:approach_velocity+continuous_move",
        "pool": "bottom20",
        "all_candidates": False,
    },
    "all_breakouts_equal_weight": {
        "score": None,
        "pool": "all",
        "all_candidates": True,
    },
    "shallow_breakouts_equal_weight": {
        "score": None,
        "pool": "shallow",
        "all_candidates": True,
    },
    "exclude_top20_equal_weight": {
        "score": None,
        "pool": "exclude_top20",
        "all_candidates": True,
    },
    "middle60_equal_weight": {
        "score": None,
        "pool": "middle60",
        "all_candidates": True,
    },
    "bottom20_equal_weight": {
        "score": None,
        "pool": "bottom20",
        "all_candidates": True,
    },
    "heat_adjusted_all": {
        "score": "heat_adjusted_score",
        "pool": "all",
        "all_candidates": False,
    },
    "heat_adjusted_market_up": {
        "score": "heat_adjusted_score",
        "pool": "market_up",
        "all_candidates": False,
    },
    "base_market_up": {
        "score": "pair:approach_velocity+continuous_move",
        "pool": "market_up",
        "all_candidates": False,
    },
    "heat_adjusted_market_low_vol": {
        "score": "heat_adjusted_score",
        "pool": "market_low_vol",
        "all_candidates": False,
    },
    "heat_adjusted_low_heat_market_up": {
        "score": "heat_adjusted_score",
        "pool": "low_heat_market_up",
        "all_candidates": False,
    },
    "market_up_equal_weight": {
        "score": None,
        "pool": "market_up",
        "all_candidates": True,
    },
    "market_low_vol_equal_weight": {
        "score": None,
        "pool": "market_low_vol",
        "all_candidates": True,
    },
    "low_heat_market_up_equal_weight": {
        "score": None,
        "pool": "low_heat_market_up",
        "all_candidates": True,
    },
}


class EventBacktestRunner:
    def run(self, config_path: str | Path) -> dict:
        config_path = Path(config_path)
        config = EventBacktestConfig.from_yaml(config_path)
        research_path = Path(config.research_run)
        research_manifest = json.loads((research_path / "manifest.json").read_text(encoding="utf-8"))
        events = pd.read_parquet(research_path / "events_with_scores.parquet")
        events["trade_date"] = pd.to_datetime(events["trade_date"])
        events["strong_breakout"] = events["breakout_strength"] >= events.groupby(
            "trade_date"
        )["breakout_strength"].transform("median")
        events["strength_percentile"] = events.groupby("trade_date")[
            "breakout_strength"
        ].rank(method="average", pct=True)
        events["breakout_acceleration_percentile"] = events.groupby("trade_date")[
            "breakout_acceleration"
        ].rank(method="average", pct=True)
        component_ranks = events.groupby("trade_date")[[
            "approach_velocity",
            "continuous_move",
        ]].rank(method="average", pct=True)
        events["heat_adjusted_score"] = pd.concat(
            [
                component_ranks["approach_velocity"],
                component_ranks["continuous_move"],
                1.0 - events["strength_percentile"],
                1.0 - events["breakout_acceleration_percentile"],
            ],
            axis=1,
        ).mean(axis=1, skipna=False)

        project = load_project(config.project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        data_version = repository.resolve(research_manifest["data_version"])
        panel_path = (
            Path(project.paths.data_root)
            / "versions"
            / data_version
            / "curated"
            / "stock_daily_panel.parquet"
        )
        panel = self._load_market(panel_path, events["trade_date"].min())
        dates = list(pd.Index(panel["trade_date"].unique()).sort_values())
        by_date = {
            pd.Timestamp(date): frame.set_index("ts_code")
            for date, frame in panel.groupby("trade_date", sort=True)
        }

        summaries: list[dict] = []
        daily_outputs: list[pd.DataFrame] = []
        trade_outputs: list[pd.DataFrame] = []
        requested = set(config.strategies) if config.strategies is not None else set(STRATEGIES)
        unknown = requested - set(STRATEGIES)
        if unknown:
            raise ValueError(f"unknown breakout backtest strategies: {sorted(unknown)}")
        for strategy_name, strategy in STRATEGIES.items():
            if strategy_name not in requested:
                continue
            top_values = (0,) if strategy["all_candidates"] else config.top_n
            for top_n in top_values:
                selections = self._selections(events, strategy, top_n)
                for cost in config.cost_scenarios_bps:
                    daily, trades, metrics = self._simulate(
                        dates,
                        by_date,
                        selections,
                        holding_days=config.holding_days,
                        initial_cash=config.initial_cash,
                        lot_size=config.lot_size,
                        min_listing_days=config.min_listing_days,
                        cost_bps=cost,
                        allocation_count=None if strategy["all_candidates"] else top_n,
                    )
                    run_key = f"{strategy_name}:top{top_n or 'all'}:cost{cost:g}"
                    summaries.append(
                        {
                            "run_key": run_key,
                            "strategy": strategy_name,
                            "pool": strategy["pool"],
                            "top_n": "all" if top_n == 0 else top_n,
                            "cost_bps": cost,
                            **metrics,
                        }
                    )
                    daily.insert(0, "run_key", run_key)
                    trades.insert(0, "run_key", run_key)
                    daily_outputs.append(daily)
                    trade_outputs.append(trades)

        summary = pd.DataFrame(summaries)
        baseline = summary.loc[
            summary["top_n"].eq("all"),
            ["pool", "cost_bps", "annualized_return"],
        ].rename(columns={"annualized_return": "pool_baseline_annualized_return"})
        summary = summary.merge(baseline, on=["pool", "cost_bps"], how="left")
        summary["annualized_excess_vs_pool"] = (
            summary["annualized_return"] - summary["pool_baseline_annualized_return"]
        )
        summary["positive_after_cost"] = summary["annualized_return"] > 0
        # Compatibility name retained for the first-round report reader.
        summary["annualized_excess_vs_strong_breakout"] = (
            summary["annualized_return"] - summary["baseline_annualized_return"]
        ) if "baseline_annualized_return" in summary else np.nan
        summary = summary.sort_values(
            ["cost_bps", "annualized_return"], ascending=[True, False]
        ).reset_index(drop=True)

        output = research_path / config.output_subdir / datetime.now(timezone.utc).strftime(
            "backtest_%Y%m%dT%H%M%SZ"
        )
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.yaml").write_text(
            config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
        pd.concat(daily_outputs, ignore_index=True).to_parquet(output / "daily.parquet", index=False)
        pd.concat(trade_outputs, ignore_index=True).to_parquet(output / "trades.parquet", index=False)
        (output / "report.md").write_text(self._report(summary), encoding="utf-8")
        manifest = {
            "status": "COMPLETED",
            "research_run": str(research_path.resolve()),
            "data_version": data_version,
            "backtest_count": int(len(summary)),
            "output_path": str(output.resolve()),
            "best_20bps_run": summary.loc[summary["cost_bps"] == 20].iloc[0]["run_key"],
            "positive_20bps_count": int(
                summary.loc[summary["cost_bps"] == 20, "positive_after_cost"].sum()
            ),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    @staticmethod
    def _load_market(path: Path, start_date: pd.Timestamp) -> pd.DataFrame:
        available = set(pq.ParquetFile(path).schema.names)
        required = [
            "trade_date",
            "ts_code",
            "raw_open",
            "adj_open",
            "adj_close",
            "is_suspended",
            "is_limit_up_open",
            "is_limit_down_open",
            "is_st",
            "is_delisting_period",
            "listing_trade_days",
        ]
        missing = sorted(set(required) - available)
        if missing:
            raise ValueError(f"backtest market panel missing columns: {missing}")
        panel = pd.read_parquet(path, columns=required)
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        # Keep one prior date so the first event can still enter on the following session.
        dates = pd.Index(panel["trade_date"].unique()).sort_values()
        prior = dates[max(int(dates.searchsorted(start_date)) - 1, 0)]
        return panel.loc[panel["trade_date"] >= prior].sort_values(
            ["trade_date", "ts_code"], kind="stable"
        ).reset_index(drop=True)

    @staticmethod
    def _selections(events: pd.DataFrame, strategy: dict, top_n: int) -> dict[pd.Timestamp, list[str]]:
        pool = strategy["pool"]
        masks = {
            "all": pd.Series(True, index=events.index),
            "strong": events["strong_breakout"],
            "shallow": ~events["strong_breakout"],
            "exclude_top20": events["strength_percentile"] <= 0.80,
            "middle60": events["strength_percentile"].between(0.20, 0.80),
            "bottom20": events["strength_percentile"] <= 0.20,
            "market_up": events["market_trend_20"] >= 0,
            "market_low_vol": (
                events["market_volatility_20"] < events["market_volatility_reference"]
            ),
            "low_heat_market_up": (
                (events["strength_percentile"] <= 0.50)
                & (events["breakout_acceleration_percentile"] <= 0.50)
                & (events["market_trend_20"] >= 0)
            ),
        }
        sample = events.loc[masks[pool]]
        selections: dict[pd.Timestamp, list[str]] = {}
        for trade_date, daily in sample.groupby("trade_date", sort=True):
            if strategy["all_candidates"]:
                selected = daily.sort_values("ts_code")
            else:
                selected = daily.dropna(subset=[strategy["score"]]).sort_values(
                    [strategy["score"], "ts_code"], ascending=[False, True]
                ).head(top_n)
            selections[pd.Timestamp(trade_date)] = selected["ts_code"].tolist()
        return selections

    def _simulate(
        self,
        dates: list,
        by_date: dict[pd.Timestamp, pd.DataFrame],
        selections: dict[pd.Timestamp, list[str]],
        *,
        holding_days: int,
        initial_cash: float,
        lot_size: int,
        min_listing_days: int,
        cost_bps: float,
        allocation_count: int | None,
        exposure_by_signal_date: dict[pd.Timestamp, float] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        sleeves = [_Sleeve(initial_cash / holding_days) for _ in range(holding_days)]
        daily_rows: list[dict] = []
        trade_rows: list[dict] = []
        generated = blocked = 0
        for date_index, date in enumerate(dates):
            date = pd.Timestamp(date)
            today = by_date[date]
            for sleeve_id, sleeve in enumerate(sleeves):
                remaining: list[_Holding] = []
                for holding in sleeve.holdings:
                    row = self._row(today, holding.ts_code)
                    if date_index < holding.due_index or not self._can_sell(row):
                        remaining.append(holding)
                        continue
                    mark = self._mark(row, holding.last_mark)
                    gross = holding.shares * holding.raw_entry * mark / holding.adj_entry
                    cost = gross * (cost_bps / 2.0) / 10_000.0
                    sleeve.cash += gross - cost
                    trade_rows.append(
                        self._trade_row(holding, date, sleeve_id, "SELL", gross, cost)
                    )
                sleeve.holdings = remaining

            if date_index >= 1:
                signal_date = pd.Timestamp(dates[date_index - 1])
                candidates = selections.get(signal_date, [])
                sleeve_id = (date_index - 1) % holding_days
                sleeve = sleeves[sleeve_id]
                if candidates and not sleeve.holdings:
                    generated += len(candidates)
                    denominator = allocation_count or len(candidates)
                    exposure = (
                        float(exposure_by_signal_date.get(signal_date, 0.0))
                        if exposure_by_signal_date is not None
                        else 1.0
                    )
                    exposure = min(max(exposure, 0.0), 1.0)
                    target = sleeve.cash * exposure / denominator if denominator else 0.0
                    for code in candidates:
                        row = self._row(today, code)
                        if not self._can_buy(row, min_listing_days):
                            blocked += 1
                            continue
                        raw_open = float(row["raw_open"])
                        adj_open = float(row["adj_open"])
                        shares = int(target // (raw_open * lot_size)) * lot_size
                        if shares <= 0:
                            continue
                        gross = shares * raw_open
                        cost = gross * (cost_bps / 2.0) / 10_000.0
                        if gross + cost > sleeve.cash:
                            continue
                        sleeve.cash -= gross + cost
                        holding = _Holding(
                            ts_code=code,
                            shares=shares,
                            signal_date=signal_date,
                            entry_date=date,
                            due_index=date_index + holding_days,
                            raw_entry=raw_open,
                            adj_entry=adj_open,
                            last_mark=adj_open,
                        )
                        sleeve.holdings.append(holding)
                        trade_rows.append(
                            self._trade_row(holding, date, sleeve_id, "BUY", gross, cost)
                        )

            nav = 0.0
            exposure = 0.0
            for sleeve in sleeves:
                nav += sleeve.cash
                for holding in sleeve.holdings:
                    row = self._row(today, holding.ts_code)
                    holding.last_mark = self._mark(row, holding.last_mark)
                    value = (
                        holding.shares
                        * holding.raw_entry
                        * holding.last_mark
                        / holding.adj_entry
                    )
                    nav += value
                    exposure += value
            daily_rows.append({"trade_date": date, "nav": nav, "gross_exposure": exposure})

        daily = pd.DataFrame(daily_rows)
        daily["return"] = daily["nav"].pct_change().fillna(0.0)
        trades = pd.DataFrame(trade_rows)
        returns = daily["return"]
        total_return = float(daily["nav"].iloc[-1] / daily["nav"].iloc[0] - 1)
        years = max((len(daily) - 1) / 252, 1 / 252)
        annual = float((1 + total_return) ** (1 / years) - 1) if total_return > -1 else -1.0
        volatility = float(returns.std(ddof=1) * math.sqrt(252))
        drawdown = daily["nav"] / daily["nav"].cummax() - 1
        yearly_returns = daily.assign(year=daily["trade_date"].dt.year).groupby("year")[
            "return"
        ].apply(lambda values: float((1.0 + values).prod() - 1.0))
        buy_count = int((trades["side"] == "BUY").sum()) if len(trades) else 0
        metrics = {
            "total_return": total_return,
            "annualized_return": annual,
            "annualized_volatility": volatility,
            "sharpe": annual / volatility if volatility > 0 else np.nan,
            "max_drawdown": float(drawdown.min()),
            "generated_signals": generated,
            "executed_buys": buy_count,
            "blocked_buys": blocked,
            "execution_rate": buy_count / generated if generated else 0.0,
            "turnover_notional": float(trades["gross_value"].sum()) if len(trades) else 0.0,
            "positive_year_ratio": float((yearly_returns > 0).mean()) if len(yearly_returns) else np.nan,
            "worst_year_return": float(yearly_returns.min()) if len(yearly_returns) else np.nan,
            "best_year_return": float(yearly_returns.max()) if len(yearly_returns) else np.nan,
        }
        return daily, trades, metrics

    @staticmethod
    def _row(today: pd.DataFrame, code: str) -> pd.Series | None:
        if code not in today.index:
            return None
        row = today.loc[code]
        return row.iloc[0] if isinstance(row, pd.DataFrame) else row

    @staticmethod
    def _can_buy(row: pd.Series | None, min_listing_days: int) -> bool:
        if row is None:
            return False
        return bool(
            np.isfinite(row.get("raw_open", np.nan))
            and np.isfinite(row.get("adj_open", np.nan))
            and not row.get("is_suspended", True)
            and not row.get("is_limit_up_open", False)
            and not row.get("is_st", False)
            and not row.get("is_delisting_period", False)
            and row.get("listing_trade_days", 0) >= min_listing_days
        )

    @staticmethod
    def _can_sell(row: pd.Series | None) -> bool:
        return bool(
            row is not None
            and np.isfinite(row.get("raw_open", np.nan))
            and not row.get("is_suspended", True)
            and not row.get("is_limit_down_open", False)
        )

    @staticmethod
    def _mark(row: pd.Series | None, fallback: float) -> float:
        if row is None:
            return fallback
        value = row.get("adj_open", np.nan)
        if not np.isfinite(value):
            value = row.get("adj_close", np.nan)
        return float(value) if np.isfinite(value) else fallback

    @staticmethod
    def _trade_row(
        holding: _Holding,
        date: pd.Timestamp,
        sleeve_id: int,
        side: str,
        gross: float,
        cost: float,
    ) -> dict:
        return {
            "trade_date": date,
            "signal_date": holding.signal_date,
            "sleeve_id": sleeve_id,
            "ts_code": holding.ts_code,
            "side": side,
            "shares": holding.shares,
            "gross_value": gross,
            "cost": cost,
        }

    @staticmethod
    def _report(summary: pd.DataFrame) -> str:
        rows = summary.loc[summary["cost_bps"] == 20].sort_values(
            "annualized_return", ascending=False
        )
        lines = [
            "# 突破过程组合回测",
            "",
            "信号于 T 日收盘形成，T+1 开盘执行，持有 10 个交易日，使用独立重叠资金袖套。",
            "下表为总往返成本 20 bps 的结果。每个排序策略与同事件池的等权组合比较。",
            "",
            "|策略|事件池|TopN|年化收益|年化波动|Sharpe|最大回撤|相对池基准年化超额|执行率|",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows.itertuples():
            lines.append(
                f"|{row.strategy}|{row.pool}|{row.top_n}|{row.annualized_return:.2%}|"
                f"{row.annualized_volatility:.2%}|{row.sharpe:.2f}|{row.max_drawdown:.2%}|"
                f"{row.annualized_excess_vs_pool:.2%}|{row.execution_rate:.2%}|"
            )
        return "\n".join(lines) + "\n"
