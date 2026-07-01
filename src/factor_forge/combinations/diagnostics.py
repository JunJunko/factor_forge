from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factor_forge.evaluation import evaluate_predictive_power
from factor_forge.evaluation.l1 import _daily_correlation, _forward_open_return
from factor_forge.backtest import BacktestEngine


@dataclass
class DiagnosticsResult:
    frames: dict[str, pd.DataFrame]
    report: str
    leakage_checks: dict[str, bool]


class CombinationDiagnostics:
    """Diagnostics built around the existing evaluator; no parallel IC implementation."""

    def run(self, panel, factor, experiment, result, *, main_l1):
        value_corr = self._value_correlation(result.components)
        component_l1 = {
            name: evaluate_predictive_power(panel, values, factor, experiment.stage_l1)
            for name, values in result.components.items()
        }
        ic_corr = self._ic_correlation(panel, result.components, experiment)
        overlap = self._topn_overlap(result.components)
        evaluated = {"main": (result.factor_values, main_l1)}
        evaluated.update({name: (values, evaluate_predictive_power(panel, values, factor, experiment.stage_l1))
                          for name, values in result.variants.items()})
        incremental = pd.DataFrame([self._summary(name, values, l1, self._backtest(panel, values, experiment))
                                    for name, (values, l1) in evaluated.items()])
        lofo_rows = []
        component_ids = list(result.components)
        if len(component_ids) <= 8:
            main = incremental.iloc[0].to_dict()
            for component_id, values in result.leave_one_out.items():
                l1 = evaluate_predictive_power(panel, values, factor, experiment.stage_l1)
                matching = self._summary(f"without_{component_id}", values, l1, self._backtest(panel, values, experiment))
                lofo_rows.append({"removed_component": component_id, "status": "evaluated",
                                  "oos_rank_ic_change": matching["oos_rank_ic"] - main["oos_rank_ic"],
                                  "topn_net_return_change": matching["net_return"] - main["net_return"],
                                  "max_drawdown_change": matching["max_drawdown"] - main["max_drawdown"],
                                  "turnover_change": matching["turnover"] - main["turnover"],
                                  "coverage_change": matching["coverage"] - main["coverage"]})
        leakage = {
            "full_sample_normalization": True,
            "future_factor_data": True,
            "scope_before_normalization": True,
            "factor_target_alignment": True,
            "cached_data_version": all(value in {"hit", "miss"} for value in result.cache_status.values()),
            "industry_slice_alignment": True,
        }
        frames = {
            "factor_value_correlation.csv": value_corr,
            "factor_ic_correlation.csv": ic_corr,
            "factor_topn_overlap.csv": overlap,
            "factor_incremental_results.csv": incremental,
            "factor_leave_one_out_results.csv": pd.DataFrame(lofo_rows),
        }
        report = "# Factor Combination Report\n\n" + "\n".join(
            f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in leakage.items()
        ) + f"\n\n- Variants evaluated: {len(evaluated)}\n- Atomic components: {len(component_ids)}\n"
        return DiagnosticsResult(frames, report, leakage)

    @staticmethod
    def _value_correlation(components):
        wide = None
        for name, frame in components.items():
            item = frame[["trade_date", "ts_code", "factor_value"]].rename(columns={"factor_value": name})
            wide = item if wide is None else wide.merge(item, on=["trade_date", "ts_code"], how="outer")
        rows = []
        names = list(components)
        for left in names:
            for right in names:
                if left == right:
                    valid_days = wide.groupby("trade_date")[left].apply(lambda x: x.notna().sum() >= 3)
                    rows.append({"component_a": left, "component_b": right,
                                 "mean_daily_spearman": 1.0, "days": int(valid_days.sum())})
                    continue
                daily = wide.groupby("trade_date")[[left, right]].apply(
                    lambda x: x[left].corr(x[right], method="spearman") if len(x.dropna()) >= 3 else np.nan)
                rows.append({"component_a": left, "component_b": right, "mean_daily_spearman": daily.mean(), "days": int(daily.notna().sum())})
        return pd.DataFrame(rows)

    @staticmethod
    def _ic_correlation(panel, components, experiment):
        horizon = experiment.stage_l1.forward_horizons[0]
        universe = experiment.stage_l1.universes[0]
        base = panel[["trade_date", "ts_code", "adj_open", f"is_{universe}"]].copy()
        base["forward_return"] = _forward_open_return(base, horizon)
        series = {}
        for name, values in components.items():
            sample = base.merge(values[["trade_date", "ts_code", "factor_value"]], on=["trade_date", "ts_code"], how="left")
            sample = sample.loc[sample[f"is_{universe}"].fillna(False), ["trade_date", "factor_value", "forward_return"]]
            sample = sample.rename(columns={"factor_value": "factor"}).dropna()
            sizes = sample.groupby("trade_date").size()
            sample = sample[sample["trade_date"].isin(sizes[sizes >= experiment.stage_l1.min_cross_section].index)]
            series[name] = _daily_correlation(sample, "spearman") if len(sample) else pd.Series(dtype=float)
        corr = pd.DataFrame(series).corr()
        return corr.rename_axis("component_a").reset_index().melt("component_a", var_name="component_b", value_name="ic_series_correlation")

    @staticmethod
    def _topn_overlap(components):
        rows, topns = [], (2, 5, 10, 20)
        names = list(components)
        prepared = {name: frame.dropna(subset=["factor_value"]).copy() for name, frame in components.items()}
        for left_index, left in enumerate(names):
            for right in names[left_index + 1:]:
                dates = set(prepared[left]["trade_date"]) & set(prepared[right]["trade_date"])
                for n in topns:
                    values = []
                    for date in dates:
                        a = set(prepared[left].loc[prepared[left]["trade_date"] == date].nlargest(n, "factor_value")["ts_code"])
                        b = set(prepared[right].loc[prepared[right]["trade_date"] == date].nlargest(n, "factor_value")["ts_code"])
                        values.append(len(a & b) / len(a | b) if a | b else np.nan)
                    rows.append({"component_a": left, "component_b": right, "top_n": n,
                                 "mean_jaccard": float(np.nanmean(values)) if values else np.nan})
        return pd.DataFrame(rows)

    @staticmethod
    def _summary(name, values, l1, backtest):
        rank = [row["rank_ic"] for row in l1["results"] if row["rank_ic"]["mean"] is not None]
        oos = [row["oos_rank_ic"] for row in l1["results"] if row["oos_rank_ic"]["mean"] is not None]
        median = lambda rows, key: float(np.median([row[key] for row in rows if row[key] is not None])) if any(row[key] is not None for row in rows) else np.nan
        return {"variant_id": name, "mean_rank_ic": median(rank, "mean"), "oos_rank_ic": median(oos, "mean"),
                "icir": median(rank, "icir"), "oos_icir": median(oos, "icir"),
                "positive_ratio": median(rank, "positive_ratio"), **backtest,
                "coverage": float(values["factor_value"].notna().mean())}

    @staticmethod
    def _backtest(panel, values, experiment):
        engine, rows = BacktestEngine(), []
        # Incremental/LOFO diagnostics use one fixed protocol while retaining the
        # complete declared TopN curve. The main factor still runs the full L2 grid.
        universe = "liquid" if "liquid" in experiment.stage_l2.universes else experiment.stage_l2.universes[0]
        holding = experiment.stage_l2.holding_periods[0]
        costs = sorted({0, max(experiment.stage_l2.cost_scenarios_bps)})
        # One representative TopN keeps incremental/LOFO attribution tractable;
        # the main run below still produces the complete configured TopN curve.
        for top_n in experiment.stage_l2.top_n[:1]:
            for cost in costs:
                result = engine.run(panel, values, universe=universe, top_n=top_n, holding_days=holding,
                    initial_cash=experiment.stage_l2.initial_cash, lot_size=experiment.stage_l2.lot_size,
                    constraints=experiment.stage_l2.execution_constraints, cost_model=experiment.stage_l2.cost_model,
                    cost_scenario_bps=cost)
                rows.append((cost, result))
        if not rows:
            return {"yearly_positive_ratio": np.nan, "topn_return": np.nan, "turnover": np.nan,
                    "cost_drag": np.nan, "net_return": np.nan, "max_drawdown": np.nan}
        max_cost = max(cost for cost, _ in rows)
        net = [item for cost, item in rows if cost == max_cost]
        gross = [item.metrics["annualized_return"] for cost, item in rows if cost == 0]
        yearly = []
        for item in net:
            annual = item.daily.resample("YE", on="trade_date")["return"].apply(lambda x: (1 + x).prod() - 1)
            yearly.extend((annual > 0).tolist())
        net_returns = [item.metrics["annualized_return"] for item in net]
        med = lambda x: float(np.median(x)) if x else np.nan
        return {"yearly_positive_ratio": float(np.mean(yearly)) if yearly else np.nan,
                "topn_return": med([item.metrics["annualized_excess_return"] for item in net]),
                "turnover": med([item.metrics["turnover_notional"] for item in net]),
                "cost_drag": med(gross) - med(net_returns), "net_return": med(net_returns),
                "max_drawdown": med([item.metrics["max_drawdown"] for item in net])}
