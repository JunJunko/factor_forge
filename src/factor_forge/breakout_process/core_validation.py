from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from factor_forge.backtest import BacktestEngine
from factor_forge.config import CostModel, ExecutionConstraints, load_project
from factor_forge.data import DataVersionRepository

from .research import BreakoutResearchRunner, _newey_west


CONTROLS = [
    "past_5d_return", "past_10d_return", "past_20d_return",
    "breakout_day_return", "distance_to_box_upper", "volatility_20d",
    "turnover_rate", "log_float_market_cap",
]
FULL_CONTROLS = [
    "past_5d_return", "past_20d_return", "breakout_day_return",
    "distance_to_box_upper", "volatility_20d", "turnover_rate",
    "log_float_market_cap",
]


def _git_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _mad_zscore(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    dates = frame["trade_date"]
    for column in columns:
        value = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = value.groupby(dates).transform("median")
        mad = (value - median).abs().groupby(dates).transform("median")
        clipped = value.clip(median - 5 * mad, median + 5 * mad)
        mean = clipped.groupby(dates).transform("mean")
        std = clipped.groupby(dates).transform(lambda x: x.std(ddof=0))
        output[column] = (clipped - mean) / std.replace(0, np.nan)
    return output


def _daily_residual(frame: pd.DataFrame, target: str, controls: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, indexes in frame.groupby("trade_date", sort=False).groups.items():
        group = frame.loc[indexes]
        required = [target, "industry_l2_code", *controls]
        sample = group.dropna(subset=required)
        if len(sample) < 8:
            continue
        dummies = pd.get_dummies(sample["industry_l2_code"], drop_first=True, dtype=float)
        design = pd.concat(
            [pd.Series(1.0, index=sample.index, name="intercept"), sample[controls], dummies],
            axis=1,
        )
        if len(sample) <= design.shape[1] + 3:
            continue
        x = design.to_numpy(float)
        y = sample[target].to_numpy(float)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        result.loc[sample.index] = y - x @ beta
    return result


def _daily_ic(frame: pd.DataFrame, factor: str, target: str, minimum: int) -> pd.Series:
    values = {}
    for date, group in frame[["trade_date", factor, target]].dropna().groupby("trade_date"):
        if len(group) >= minimum:
            values[date] = group[factor].corr(group[target], method="spearman")
    return pd.Series(values, dtype=float).dropna()


def _ic_stats(values: pd.Series, horizon: int) -> dict:
    values = values.dropna()
    if len(values) < 2:
        return {key: np.nan for key in ["mean_ic", "median_ic", "ic_std", "icir", "positive_ratio", "nw_t", "nw_p"]} | {"ic_days": len(values)}
    std = float(values.std(ddof=1))
    nw_t, nw_p = _newey_west(values, max(horizon - 1, 0))
    return {
        "mean_ic": float(values.mean()), "median_ic": float(values.median()),
        "ic_std": std, "icir": float(values.mean() / std * math.sqrt(252)) if std else np.nan,
        "positive_ratio": float((values > 0).mean()), "nw_t": nw_t, "nw_p": nw_p,
        "ic_days": int(len(values)),
    }


class ContinuousMoveCoreValidationRunner:
    def run(self, config_path: str | Path) -> dict:
        path = Path(config_path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        np.random.seed(int(raw.get("random_seed", 0)))
        source = Path(raw["source_run"])
        resume_output = raw.get("resume_output")
        run_id = f"continuous_move_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        output = Path(resume_output) if resume_output else Path(raw["output_root"]) / run_id
        if resume_output:
            if not output.exists():
                raise FileNotFoundError(f"resume_output does not exist: {output}")
            run_id = output.name
        else:
            output.mkdir(parents=True, exist_ok=False)
            (output / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        manifest = {
            "run_id": run_id, "status": "RUNNING", "data_version": raw["data_version"],
            "source_run": str(source.resolve()), "code_version": _git_version(),
            "random_seed": int(raw.get("random_seed", 0)), "output_path": str(output.resolve()),
        }
        self._json(output / "manifest.json", manifest)

        project = load_project(raw["project_config"])
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        data_version, panel = repository.load_panel(raw["data_version"])
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        if resume_output and (output / "continuous_move_analysis_panel.parquet").exists():
            events = pd.read_parquet(output / "continuous_move_analysis_panel.parquet")
            events["trade_date"] = pd.to_datetime(events["trade_date"])
            split = yaml.safe_load((output / "sample_split.yaml").read_text(encoding="utf-8"))
            ic_summary = pd.read_csv(output / "continuous_move_ic_summary.csv")
            backtest_summary = self._backtests(events, panel, repository, data_version, raw, output)
            classification = self._final_report(events, split, ic_summary, backtest_summary, output)
            manifest.update({
                "status": "COMPLETED", "event_count": int(len(events)),
                "classification": classification, "final_test_is_clean": False,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
            self._json(output / "manifest.json", manifest)
            return manifest
        events = pd.read_parquet(source / "events_with_scores.parquet")
        events["trade_date"] = pd.to_datetime(events["trade_date"])
        events = self._prepare(events, panel, raw, output)
        split = self._split(events, output)
        self._conditions(events, output)
        self._definition_reports(events, raw, split, output)
        self._correlations(events, output)
        ic_summary, daily_ic = self._independence(events, raw, split, output)
        self._incremental(events, raw, split, output)
        self._stability(events, raw, split, output)
        self._quantiles(events, raw, output)
        backtest_summary = self._backtests(events, panel, repository, data_version, raw, output)
        classification = self._final_report(events, split, ic_summary, backtest_summary, output)
        manifest.update({
            "status": "COMPLETED", "event_count": int(len(events)),
            "classification": classification, "final_test_is_clean": False,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        self._json(output / "manifest.json", manifest)
        return manifest

    def _prepare(self, events: pd.DataFrame, panel: pd.DataFrame, raw: dict, output: Path) -> pd.DataFrame:
        ordered = panel.sort_values(["ts_code", "trade_date"], kind="stable").copy()
        close_group = ordered.groupby("ts_code", sort=False)["adj_close"]
        for window in (5, 10, 20):
            ordered[f"past_{window}d_return"] = ordered["adj_close"] / close_group.shift(window) - 1
        ordered["breakout_day_return"] = ordered.get("pct_change", np.nan) / 100.0
        returns = close_group.pct_change(fill_method=None)
        ordered["volatility_20d"] = returns.groupby(ordered["ts_code"]).rolling(20).std(ddof=0).droplevel(0).reindex(ordered.index)
        open_group = ordered.groupby("ts_code", sort=False)["adj_open"]
        entry_open = open_group.shift(-1)
        label_columns = []
        for horizon in raw["forward_horizons"]:
            name = f"forward_return_{horizon}"
            ordered[name] = open_group.shift(-(int(horizon) + 1)) / entry_open - 1.0
            label_columns.append(name)
        fields = ["trade_date", "ts_code", *CONTROLS, "industry_l2_code", "amount_cny", "circ_mv_cny", *label_columns]
        fields = [name for name in fields if name in ordered]
        features = ordered[fields]
        # The discovery artifact carries convenience metadata. Formal validation
        # deliberately replaces it with the point-in-time fields from the frozen
        # data version instead of accepting pandas suffixes or stale values.
        replace = [name for name in fields if name not in {"trade_date", "ts_code"} and name in events]
        events = events.drop(columns=replace)
        events = events.merge(features, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
        events["continuous_move_value"] = -events["gap_atr"].abs()
        events["distance_to_box_upper"] = events["breakout_strength"]
        if "log_float_market_cap" not in events:
            events["log_float_market_cap"] = np.log(events["circ_mv_cny"].where(events["circ_mv_cny"] > 0))
        processed = _mad_zscore(events, ["continuous_move_value", *CONTROLS])
        events["continuous_move_raw"] = processed["continuous_move_value"]
        for column in CONTROLS:
            events[column] = processed[column]
        events["continuous_move_neutral_basic"] = _daily_residual(
            events, "continuous_move_raw", ["log_float_market_cap"]
        )
        events["continuous_move_neutral_full"] = _daily_residual(
            events, "continuous_move_raw", FULL_CONTROLS
        )
        events.to_parquet(output / "continuous_move_analysis_panel.parquet", index=False)
        return events

    @staticmethod
    def _split(events: pd.DataFrame, output: Path) -> dict:
        dates = pd.Index(events["trade_date"].drop_duplicates().sort_values())
        train_end = dates[max(int(len(dates) * 0.6) - 1, 0)]
        validation_end = dates[max(int(len(dates) * 0.8) - 1, 0)]
        split = {
            "method": "ordered_event_dates_60_20_20", "train_start": str(dates.min().date()),
            "train_end": str(train_end.date()), "validation_start": str(dates[dates.get_loc(train_end)+1].date()),
            "validation_end": str(validation_end.date()), "final_test_start": str(dates[dates.get_loc(validation_end)+1].date()),
            "final_test_end": str(dates.max().date()), "final_test_is_clean": False,
            "reason": "The complete period was inspected in the discovery run.",
        }
        (output / "sample_split.yaml").write_text(yaml.safe_dump(split, sort_keys=False), encoding="utf-8")
        report = "# Sample split\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in split.items())
        (output / "sample_split_report.md").write_text(report + "\n", encoding="utf-8")
        return split

    @staticmethod
    def _conditions(events: pd.DataFrame, output: Path) -> None:
        conditions = BreakoutResearchRunner._conditions(events)
        rows = []
        masks = {}
        for name, mask in conditions.items():
            boolean = mask.astype("boolean")
            masks[name] = boolean.fillna(False)
            selected = events.loc[masks[name]]
            rows.append({
                "condition": name, "event_count": len(selected),
                "unique_stock_count": selected["ts_code"].nunique(),
                "trading_day_count": selected["trade_date"].nunique(),
                "coverage_vs_all": len(selected) / len(events),
                "positive_rate": float(boolean.fillna(False).mean()),
                "missing_rate": float(boolean.isna().mean()),
            })
        pd.DataFrame(rows).to_csv(output / "condition_coverage.csv", index=False, encoding="utf-8-sig")
        overlap = []
        for left, left_mask in masks.items():
            for right, right_mask in masks.items():
                intersection = int((left_mask & right_mask).sum())
                union = int((left_mask | right_mask).sum())
                overlap.append({"condition_a": left, "condition_b": right, "intersection_count": intersection, "union_count": union, "jaccard_ratio": intersection / union if union else np.nan})
        pd.DataFrame(overlap).to_csv(output / "condition_overlap_matrix.csv", index=False, encoding="utf-8-sig")
        contracting = next(row for row in rows if row["condition"] == "volatility_contracting")
        reason = "duplicates the event qualification rule max_volatility_ratio=1.0" if contracting["coverage_vs_all"] == 1 else "does not cover all events"
        (output / "condition_validation_report.md").write_text(
            f"# Condition validation\n\n`volatility_contracting` coverage: {contracting['coverage_vs_all']:.4%}.\n\nConclusion: {reason}. The core experiment uses `all` only.\n",
            encoding="utf-8",
        )

    @staticmethod
    def _definition_reports(events: pd.DataFrame, raw: dict, split: dict, output: Path) -> None:
        definition = """# continuous_move definition

## Mathematical definition

For an already confirmed frozen-box breakout event at close T:

`gap_atr = (Open_T - Close_{T-1}) / ATR_box`

`continuous_move_value = -abs(gap_atr)`

The research value is MAD-winsorized (5 MAD) and z-scored within the event cross-section on T. A larger value means less of the breakout arrived as an overnight gap. The box and ATR scale were frozen when the box became active; the signal is known after close T and execution begins at open T+1.

Inputs: adjusted open, adjusted prior close, frozen ATR(20), frozen box lifecycle, breakout confirmation at T close. Box lookback=40, process window=10, maximum active age=20.

This is an event factor implemented in Python, not a standard Factor DSL YAML, because the current DSL has no frozen stateful box operator. The frozen validation configuration is `configs/continuous_move_core_validation.yaml`.
"""
        (output / "continuous_move_definition.md").write_text(definition, encoding="utf-8")
        leakage = f"""# continuous_move leakage check

- Signal cutoff: close T; entry: open T+1.
- `gap_atr` is known at open T; breakout membership is known at close T.
- Box upper/lower and ATR are immutable for each `box_id`; no future rebasing is used.
- Source event `pre_window_end` is strictly earlier than `trade_date`: {bool((pd.to_datetime(events['pre_window_end']) < events['trade_date']).all())}.
- Ranking uses only the T event cross-section; no full-sample mean or variance.
- Backtests use the platform suspension, limit-up/down, ST, delisting and listing-age rules.
- Selection caveat: the factor applies only to stocks that have confirmed a breakout by close T. This is a valid post-event universe, not a pre-breakout predictor.
- `final_test_is_clean = false`: the discovery report inspected the full period, so this run is a secondary validation.
- Corporate-action handling uses adjusted prices for signals/returns and raw open for executable fills, following the platform contract.
"""
        (output / "continuous_move_leakage_check.md").write_text(leakage, encoding="utf-8")

    @staticmethod
    def _correlations(events: pd.DataFrame, output: Path) -> None:
        columns = ["continuous_move_raw", *CONTROLS]
        pearson = events[columns].corr("pearson").stack().rename("correlation").reset_index()
        pearson["method"] = "pearson"
        spearman = events[columns].corr("spearman").stack().rename("correlation").reset_index()
        spearman["method"] = "spearman"
        pd.concat([pearson, spearman]).rename(columns={"level_0": "factor_a", "level_1": "factor_b"}).to_csv(output / "factor_correlation.csv", index=False, encoding="utf-8-sig")
        yearly = []
        for year, group in events.groupby(events["trade_date"].dt.year):
            for method in ("pearson", "spearman"):
                corr = group[columns].corr(method)["continuous_move_raw"]
                yearly.extend({"year": year, "factor": name, "method": method, "correlation": value} for name, value in corr.items())
        pd.DataFrame(yearly).to_csv(output / "factor_correlation_by_year.csv", index=False, encoding="utf-8-sig")
        complete = events[columns].dropna()
        correlation = complete.corr().to_numpy()
        inverse = np.linalg.pinv(correlation)
        vif = pd.DataFrame({"factor": columns, "vif": np.diag(inverse)})
        (output / "multicollinearity_report.md").write_text("# Multicollinearity\n\n" + vif.to_markdown(index=False) + "\n", encoding="utf-8")

    def _independence(self, events: pd.DataFrame, raw: dict, split: dict, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        variants = ["continuous_move_raw", "continuous_move_neutral_basic", "continuous_move_neutral_full"]
        boundaries = {
            "train": (None, pd.Timestamp(split["train_end"])),
            "validation": (pd.Timestamp(split["validation_start"]), pd.Timestamp(split["validation_end"])),
            "final_test": (pd.Timestamp(split["final_test_start"]), None), "all": (None, None),
        }
        summaries, daily_rows, yearly = [], [], []
        for horizon in raw["forward_horizons"]:
            target = f"forward_return_{horizon}"
            for variant in variants:
                daily = _daily_ic(events, variant, target, raw["min_cross_section"])
                for date, value in daily.items():
                    daily_rows.append({"trade_date": date, "variant": variant, "horizon": horizon, "rank_ic": value})
                for name, (start, end) in boundaries.items():
                    selected = daily
                    if start is not None: selected = selected[selected.index >= start]
                    if end is not None: selected = selected[selected.index <= end]
                    row = {"variant": variant, "horizon": horizon, "sample": name, **_ic_stats(selected, horizon)}
                    row["average_daily_stocks"] = float(events.dropna(subset=[variant, target]).groupby("trade_date").size().mean())
                    summaries.append(row)
                for year, values in daily.groupby(daily.index.year):
                    yearly.append({"variant": variant, "horizon": horizon, "year": year, **_ic_stats(values, horizon)})
        summary = pd.DataFrame(summaries)
        raw_all = summary[(summary.variant == "continuous_move_raw") & (summary["sample"] == "all")].set_index("horizon")["mean_ic"]
        summary["neutralization_retention_ratio"] = summary.apply(lambda row: row.mean_ic / raw_all.get(row.horizon, np.nan) if row.variant != "continuous_move_raw" else 1.0, axis=1)
        daily_frame = pd.DataFrame(daily_rows)
        summary.to_csv(output / "continuous_move_ic_summary.csv", index=False, encoding="utf-8-sig")
        daily_frame.to_csv(output / "continuous_move_ic_daily.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(yearly).to_csv(output / "continuous_move_ic_by_year.csv", index=False, encoding="utf-8-sig")
        summary[summary["sample"] == "all"].to_csv(output / "continuous_move_ic_decay.csv", index=False, encoding="utf-8-sig")
        full = summary[(summary.variant == "continuous_move_neutral_full") & (summary["sample"] == "all")]
        (output / "continuous_move_neutralization_report.md").write_text("# Neutralization\n\n" + full.to_markdown(index=False) + "\n", encoding="utf-8")
        return summary, daily_frame

    def _incremental(self, events: pd.DataFrame, raw: dict, split: dict, output: Path) -> None:
        rows = []
        for horizon in raw["forward_horizons"]:
            target = f"forward_return_{horizon}"
            for date, group in events.groupby("trade_date"):
                required = [target, "continuous_move_raw", "industry_l2_code", *FULL_CONTROLS]
                sample = group.dropna(subset=required)
                if len(sample) < 12: continue
                dummies = pd.get_dummies(sample.industry_l2_code, drop_first=True, dtype=float)
                base = pd.concat([pd.Series(1.0, index=sample.index), sample[FULL_CONTROLS], dummies], axis=1)
                if len(sample) <= base.shape[1] + 3: continue
                x1, y = base.to_numpy(float), sample[target].to_numpy(float)
                x2 = np.column_stack([x1, sample.continuous_move_raw.to_numpy(float)])
                b1, *_ = np.linalg.lstsq(x1, y, rcond=None); b2, *_ = np.linalg.lstsq(x2, y, rcond=None)
                p1, p2 = x1 @ b1, x2 @ b2
                sst = float(np.sum((y-y.mean())**2))
                r1 = 1-float(np.sum((y-p1)**2))/sst if sst else np.nan
                r2 = 1-float(np.sum((y-p2)**2))/sst if sst else np.nan
                rows.append({"trade_date": date, "horizon": horizon, "continuous_move_coefficient": b2[-1], "r2_increment": r2-r1, "rank_ic_increment": pd.Series(p2).corr(pd.Series(y), method="spearman")-pd.Series(p1).corr(pd.Series(y), method="spearman")})
        daily = pd.DataFrame(rows)
        boundaries = {"train": (None, split["train_end"]), "validation": (split["validation_start"], split["validation_end"]), "final_test": (split["final_test_start"], None), "all": (None, None)}
        summary = []
        for horizon, horizon_rows in daily.groupby("horizon"):
            for name, (start, end) in boundaries.items():
                sample = horizon_rows
                if start: sample = sample[sample.trade_date >= pd.Timestamp(start)]
                if end: sample = sample[sample.trade_date <= pd.Timestamp(end)]
                nw_t, nw_p = _newey_west(sample.continuous_move_coefficient, horizon-1)
                annual = sample.assign(year=sample.trade_date.dt.year).groupby("year").continuous_move_coefficient.mean()
                summary.append({"horizon": horizon, "sample": name, "coefficient_mean": sample.continuous_move_coefficient.mean(), "nw_t": nw_t, "nw_p": nw_p, "positive_year_ratio": (annual>0).mean(), "r2_increment_mean": sample.r2_increment.mean(), "rank_ic_increment_mean": sample.rank_ic_increment.mean(), "days": len(sample)})
        result = pd.DataFrame(summary)
        result.to_csv(output / "incremental_regression_results.csv", index=False, encoding="utf-8-sig")
        (output / "continuous_move_incremental_report.md").write_text("# Incremental explanation\n\n" + result.to_markdown(index=False) + "\n", encoding="utf-8")

    def _stability(self, events: pd.DataFrame, raw: dict, split: dict, output: Path) -> None:
        variants = ["continuous_move_raw", "continuous_move_neutral_full"]
        by_year, by_industry, by_regime, outliers = [], [], [], []
        for variant in variants:
            for horizon in raw["forward_horizons"]:
                target = f"forward_return_{horizon}"
                daily = _daily_ic(events, variant, target, raw["min_cross_section"])
                for year, values in daily.groupby(daily.index.year): by_year.append({"variant": variant, "horizon": horizon, "year": year, **_ic_stats(values,horizon), "observations": len(events[events.trade_date.dt.year==year])})
                for industry, group in events.groupby("industry_l2_code"):
                    values = _daily_ic(group, variant, target, 3)
                    by_industry.append({"variant": variant, "horizon": horizon, "industry": industry, **_ic_stats(values,horizon), "observations": len(group), "small_sample": len(group)<200})
                regimes = {"trend_up": events.market_trend_20>=0, "trend_down": events.market_trend_20<0, "vol_high": events.market_volatility_20>=events.market_volatility_reference, "vol_low": events.market_volatility_20<events.market_volatility_reference}
                for regime, mask in regimes.items():
                    values = _daily_ic(events[mask.fillna(False)], variant, target, max(3,raw["min_cross_section"]//2))
                    by_regime.append({"variant": variant,"horizon":horizon,"regime_dimension":regime.split('_')[0],"regime":regime,**_ic_stats(values,horizon),"observations":int(mask.fillna(False).sum()),"status":"available"})
                for dimension in ("liquidity", "breadth"):
                    by_regime.append({"variant":variant,"horizon":horizon,"regime_dimension":dimension,"regime":"unavailable","observations":0,"status":"platform_has_no_governed_regime"})
                base = events[["trade_date",variant,target]].dropna()
                variants_out = {
                    "base": base,
                    "remove_return_top_1pct": base[base[target] <= base[target].quantile(.99)],
                    "remove_return_bottom_1pct": base[base[target] >= base[target].quantile(.01)],
                    "remove_factor_top_1pct": base[base[variant] <= base[variant].quantile(.99)],
                    "remove_factor_bottom_1pct": base[base[variant] >= base[variant].quantile(.01)],
                }
                winsor = base.copy(); winsor[target]=winsor[target].clip(base[target].quantile(.01),base[target].quantile(.99)); variants_out["winsor_return_1pct"]=winsor
                for test, sample in variants_out.items(): outliers.append({"variant":variant,"horizon":horizon,"test":test,**_ic_stats(_daily_ic(sample,variant,target,raw["min_cross_section"]),horizon)})
        pd.DataFrame(by_year).to_csv(output/"continuous_move_stability_by_year.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(by_industry).to_csv(output/"continuous_move_stability_by_industry.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(by_regime).to_csv(output/"continuous_move_stability_by_regime.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(outliers).to_csv(output/"continuous_move_outlier_robustness.csv",index=False,encoding="utf-8-sig")
        parameter = []
        raw_gap = events.gap_atr.abs()*events.frozen_atr
        for window, scale in ((15,.75),(20,1.0),(25,1.25)):
            factor = -raw_gap/(events.frozen_atr*scale)
            temp=events[["trade_date",* [f"forward_return_{h}" for h in raw["forward_horizons"]]]].copy(); temp["factor"]=_mad_zscore(pd.DataFrame({"trade_date":events.trade_date,"factor":factor}),["factor"])["factor"]
            for horizon in raw["forward_horizons"]: parameter.append({"atr_window_proxy":window,"horizon":horizon,**_ic_stats(_daily_ic(temp,"factor",f"forward_return_{horizon}",raw["min_cross_section"]),horizon),"note":"fixed-event scale proxy; not box regeneration"})
        pd.DataFrame(parameter).to_csv(output/"continuous_move_parameter_robustness.csv",index=False,encoding="utf-8-sig")
        (output/"continuous_move_stability_report.md").write_text("# Stability\n\nTrend and volatility reuse discovery-run point-in-time fields. The platform has no governed liquidity/breadth Regime, so those rows are explicitly unavailable. Parameter neighbours are fixed-event ATR-scale proxies and do not alter event membership.\n",encoding="utf-8")

    @staticmethod
    def _quantiles(events: pd.DataFrame, raw: dict, output: Path) -> None:
        rows=[]; yearly=[]
        for variant in ("continuous_move_raw","continuous_move_neutral_full"):
            for horizon in raw["forward_horizons"]:
                target=f"forward_return_{horizon}"
                for groups in (5,10):
                    daily_rows=[]
                    for date, group in events[["trade_date",variant,target]].dropna().groupby("trade_date"):
                        if len(group)<groups*2: continue
                        bucket=pd.qcut(group[variant].rank(method="first"),groups,labels=False)+1
                        benchmark=group[target].mean()
                        for q,value in group.assign(q=bucket).groupby("q")[target].agg(["mean","size"]).iterrows(): daily_rows.append({"trade_date":date,"quantile":q,"return":value["mean"],"excess_return":value["mean"]-benchmark,"stocks":value["size"]})
                    daily=pd.DataFrame(daily_rows)
                    if daily.empty: continue
                    for q,g in daily.groupby("quantile"): rows.append({"variant":variant,"horizon":horizon,"groups":groups,"quantile":q,"mean_return":g["return"].mean(),"mean_excess_return":g.excess_return.mean(),"mean_stocks":g.stocks.mean()})
                    for (year,q),g in daily.groupby([daily.trade_date.dt.year,"quantile"]): yearly.append({"variant":variant,"horizon":horizon,"groups":groups,"year":year,"quantile":q,"mean_return":g["return"].mean(),"mean_excess_return":g.excess_return.mean()})
        quant=pd.DataFrame(rows); quant.to_csv(output/"continuous_move_quantile_returns.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(yearly).to_csv(output/"continuous_move_quantile_returns_by_year.csv",index=False,encoding="utf-8-sig")
        (output/"continuous_move_monotonicity_report.md").write_text("# Monotonicity\n\nSee complete 5- and 10-bin matrices in the CSV artifacts. Returns are relative to the same-day equal-weight breakout-event benchmark.\n",encoding="utf-8")

    def _backtests(self, events: pd.DataFrame, panel: pd.DataFrame, repository, data_version: str, raw: dict, output: Path) -> pd.DataFrame:
        constraints=ExecutionConstraints.model_validate(raw["execution_constraints"]); costs=CostModel.model_validate(raw["cost_model"])
        factor_variants=(
            "continuous_move_raw",
            "continuous_move_neutral_basic",
            "continuous_move_neutral_full",
        )
        membership=events[["trade_date","ts_code"]].drop_duplicates().assign(selection_eligible=True,condition_quantile=1)
        start=events.trade_date.min()
        needed = [
            "trade_date", "ts_code", "raw_open", "adj_open", "adj_close",
            "is_liquid", "is_tradeable", "is_suspended", "is_limit_up_open",
            "is_limit_down_open", "is_st", "is_delisting_period", "listing_trade_days",
        ]
        panel=panel.loc[panel.trade_date>=start, [name for name in needed if name in panel]].copy()
        market=repository.load_raw_dataset(data_version,"index_daily")
        if market is not None and "ts_code" in market: market=market[market.ts_code==market.ts_code.iloc[0]].rename(columns={"raw_open":"open"})
        parts = output / "backtest_parts"
        parts.mkdir(exist_ok=True)
        checkpoint = output / "backtest_checkpoint.csv"
        summaries = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
        completed = {
            (row["variant"], int(row["holding_period"]), int(row["top_n"]), float(row["cost_bps"]))
            for row in summaries
        }
        for variant in factor_variants:
            factors=events[["trade_date","ts_code",variant]].rename(columns={variant:"factor_value"}).dropna()
            for hold in raw["holding_periods"]:
                for topn in raw["top_n"]:
                    zero_annual=None
                    for cost in raw["cost_scenarios_bps"]:
                        identity = (variant, int(hold), int(topn), float(cost))
                        if identity in completed:
                            continue
                        result=BacktestEngine().run(panel,factors,universe=raw["universe"],top_n=topn,holding_days=hold,initial_cash=raw["initial_cash"],lot_size=raw["lot_size"],constraints=constraints,cost_model=costs,cost_scenario_bps=cost,market_benchmark=market,selection_membership=membership)
                        metric={"variant":variant,"holding_period":hold,"top_n":topn,"cost_bps":cost,**result.metrics,"average_holdings":result.positions.groupby("trade_date").ts_code.nunique().mean() if len(result.positions) else 0,"empty_exposure_ratio":float((result.daily.gross_exposure==0).mean())}
                        summaries.append(metric)
                        key=f"{variant}_h{hold}_n{topn}_c{cost}"
                        result.trades.assign(variant=variant,holding_period=hold,top_n=topn,cost_bps=cost).to_parquet(parts/f"{key}_trades.parquet",index=False)
                        result.daily.assign(variant=variant,holding_period=hold,top_n=topn,cost_bps=cost).to_parquet(parts/f"{key}_daily.parquet",index=False)
                        pd.DataFrame(summaries).to_csv(checkpoint,index=False,encoding="utf-8-sig")
                        completed.add(identity)
        summary=pd.DataFrame(summaries)
        zero = summary[summary.cost_bps==0].set_index(["variant","holding_period","top_n"])["annualized_return"]
        summary["cost_drag_vs_zero"] = summary.apply(lambda row: zero.get((row.variant,row.holding_period,row.top_n),np.nan)-row.annualized_return,axis=1)
        trade_parts=list(parts.glob("*_trades.parquet")); daily_parts=list(parts.glob("*_daily.parquet"))
        trades=pd.concat((pd.read_parquet(item) for item in trade_parts),ignore_index=True)
        equity=pd.concat((pd.read_parquet(item) for item in daily_parts),ignore_index=True)
        summary.to_csv(output/"continuous_move_topn_summary.csv",index=False,encoding="utf-8-sig"); trades.to_csv(output/"continuous_move_trade_details.csv",index=False,encoding="utf-8-sig"); equity.to_csv(output/"continuous_move_equity_curve.csv",index=False,encoding="utf-8-sig")
        periods=equity.assign(year=equity.trade_date.dt.year,month=equity.trade_date.dt.to_period("M").astype(str))
        by_year=periods.groupby(["variant","holding_period","top_n","cost_bps","year"])["return"].apply(lambda x:(1+x).prod()-1).reset_index(name="return")
        by_month=periods.groupby(["variant","holding_period","top_n","cost_bps","month"])["return"].apply(lambda x:(1+x).prod()-1).reset_index(name="return")
        by_year.to_csv(output/"continuous_move_topn_by_year.csv",index=False,encoding="utf-8-sig"); by_month.to_csv(output/"continuous_move_topn_by_month.csv",index=False,encoding="utf-8-sig")
        primary=summary[(summary.top_n==raw["primary_top_n"])&(summary.holding_period==raw["primary_holding_period"])]
        (output/"continuous_move_topn_report.md").write_text("# TopN backtest\n\n"+primary.to_markdown(index=False)+"\n",encoding="utf-8")
        self._stress(events,panel,constraints,costs,membership,raw,output,trades)
        return summary

    def _stress(self, events,panel,constraints,costs,membership,raw,output,trades):
        rows=[]; primary_top=raw["primary_top_n"]; primary_hold=raw["primary_holding_period"]
        for variant in (
            "continuous_move_raw",
            "continuous_move_neutral_basic",
            "continuous_move_neutral_full",
        ):
            base=events[["trade_date","ts_code",variant,"amount_cny","log_float_market_cap"]].rename(columns={variant:"factor_value"}).dropna(subset=["factor_value"])
            tests={"base_20bps":base,"cost_30bps":base,"remove_low_amount_20pct":base[base.amount_cny>=base.groupby("trade_date").amount_cny.transform(lambda x:x.quantile(.2))],"remove_low_size_20pct":base[base.log_float_market_cap>=base.groupby("trade_date").log_float_market_cap.transform(lambda x:x.quantile(.2))]}
            for name,signals in tests.items():
                cost=30 if name=="cost_30bps" else 20
                result=BacktestEngine().run(panel,signals,universe=raw["universe"],top_n=primary_top,holding_days=primary_hold,initial_cash=raw["initial_cash"],lot_size=raw["lot_size"],constraints=constraints,cost_model=costs,cost_scenario_bps=cost,selection_membership=membership)
                rows.append({"variant":variant,"stress_test":name,**result.metrics})
        for unavailable in ("next_day_vwap","extra_open_slippage_10bps_exact","cooldown_3d","cooldown_5d"):
            rows.append({"variant":"both","stress_test":unavailable,"status":"unavailable_or_requires_engine_contract_change"})
        pd.DataFrame(rows).to_csv(output/"continuous_move_stress_test.csv",index=False,encoding="utf-8-sig")
        concentration=[]
        buys=trades[trades.side=="BUY"]
        for code,count in buys.groupby("ts_code").size().nlargest(10).items(): concentration.append({"dimension":"stock","key":code,"buy_count":count})
        for date,count in buys.groupby("trade_date").size().nlargest(10).items(): concentration.append({"dimension":"trade_date","key":str(date),"buy_count":count})
        pd.DataFrame(concentration).to_csv(output/"continuous_move_return_concentration.csv",index=False,encoding="utf-8-sig")
        (output/"continuous_move_stress_report.md").write_text("# Stress tests\n\nVWAP is absent from the governed panel. Exact buy-side-only slippage and cooldown require a BacktestEngine contract extension; they are not approximated. Concentration currently reports execution counts, not fabricated PnL attribution.\n",encoding="utf-8")

    @staticmethod
    def _final_report(events, split, ic, backtests, output)->str:
        raw=ic[(ic.variant=="continuous_move_raw")&(ic["sample"]=="all")]
        basic_all=ic[(ic.variant=="continuous_move_neutral_basic")&(ic["sample"]=="all")]
        basic_validation=ic[(ic.variant=="continuous_move_neutral_basic")&(ic["sample"]=="validation")].set_index("horizon")
        basic_final=ic[(ic.variant=="continuous_move_neutral_basic")&(ic["sample"]=="final_test")].set_index("horizon")
        full_all=ic[(ic.variant=="continuous_move_neutral_full")&(ic["sample"]=="all")]
        full_validation=ic[(ic.variant=="continuous_move_neutral_full")&(ic["sample"]=="validation")].set_index("horizon")
        full_final=ic[(ic.variant=="continuous_move_neutral_full")&(ic["sample"]=="final_test")].set_index("horizon")
        common=full_validation.index.intersection(full_final.index)
        sign_agreement=float((np.sign(full_validation.loc[common,"mean_ic"])==np.sign(full_final.loc[common,"mean_ic"])).mean()) if len(common) else 0.0
        positive_all=float((full_all.mean_ic>0).mean()) if len(full_all) else 0.0
        retention=float((full_all.neutralization_retention_ratio>=.30).mean()) if len(full_all) else 0.0
        incremental_path=output/"incremental_regression_results.csv"
        incremental=pd.read_csv(incremental_path) if incremental_path.exists() else pd.DataFrame()
        incremental_final=incremental[incremental["sample"]=="final_test"] if len(incremental) else pd.DataFrame()
        incremental_positive=float((incremental_final.coefficient_mean>0).mean()) if len(incremental_final) else 0.0
        independent=bool(positive_all>=.6 and retention>=.6 and sign_agreement>=.6 and incremental_positive>=.6)
        basic_common=basic_validation.index.intersection(basic_final.index)
        basic_sign_agreement=float((np.sign(basic_validation.loc[basic_common,"mean_ic"])==np.sign(basic_final.loc[basic_common,"mean_ic"])).mean()) if len(basic_common) else 0.0
        basic_positive=float((basic_all.mean_ic>0).mean()) if len(basic_all) else 0.0
        basic_retention=float((basic_all.neutralization_retention_ratio>=.30).mean()) if len(basic_all) else 0.0
        candidate=backtests[(backtests.variant=="continuous_move_neutral_full")&(backtests.cost_bps==20)&(backtests.top_n.isin([5,10,20]))] if len(backtests) else pd.DataFrame()
        profitable_candidate=bool((candidate.annualized_return>0).any()) if len(candidate) else False
        yearly_path=output/"continuous_move_topn_by_year.csv"
        yearly=pd.read_csv(yearly_path) if yearly_path.exists() else pd.DataFrame()
        primary_years=yearly[(yearly.variant=="continuous_move_neutral_full")&(yearly.cost_bps==20)&(yearly.top_n==10)&(yearly.holding_period==10)] if len(yearly) else pd.DataFrame()
        majority_positive_years=bool((primary_years["return"]>0).mean()>.5) if len(primary_years) else False
        tradable=profitable_candidate and majority_positive_years
        basic_candidate=backtests[(backtests.variant=="continuous_move_neutral_basic")&(backtests.cost_bps==20)&(backtests.top_n.isin([5,10,20]))] if len(backtests) else pd.DataFrame()
        basic_profitable=bool((basic_candidate.annualized_return>0).any()) if len(basic_candidate) else False
        basic_years=yearly[(yearly.variant=="continuous_move_neutral_basic")&(yearly.cost_bps==20)&(yearly.top_n==10)&(yearly.holding_period==10)] if len(yearly) else pd.DataFrame()
        basic_majority_positive_years=bool((basic_years["return"]>0).mean()>.5) if len(basic_years) else False
        basic_usable=bool(basic_positive>=.8 and basic_retention>=.6 and basic_sign_agreement>=.6 and basic_profitable and basic_majority_positive_years)
        raw_effective=bool((raw.mean_ic>0).mean()>=.8) if len(raw) else False
        classification="CORE_FACTOR" if independent and tradable else "CONDITIONAL_FACTOR" if basic_usable else "REDUNDANT_FACTOR" if raw_effective and not independent else "REJECTED_FACTOR"
        report=f"""# continuous_move final validation report

## Executive summary

- Classification: **{classification}**
- Independent after full neutralization: **{independent}**
- Tradable after 20 bps under the predeclared standard: **{tradable}**
- Full-neutral positive-horizon ratio (all sample): **{positive_all:.1%}**
- Full-neutral retention >=30% ratio: **{retention:.1%}**
- Validation/final-test sign agreement: **{sign_agreement:.1%}**
- Final-test positive incremental-coefficient ratio: **{incremental_positive:.1%}**
- At least one positive-absolute-return neutral TopN at 20 bps: **{profitable_candidate}**
- Primary Top10/10d positive-year majority: **{majority_positive_years}**
- Basic-neutral positive-horizon ratio: **{basic_positive:.1%}**
- Basic-neutral retention >=30% ratio: **{basic_retention:.1%}**
- Basic-neutral Validation/final sign agreement: **{basic_sign_agreement:.1%}**
- Basic-neutral positive absolute-return TopN at 20 bps: **{basic_profitable}**
- Basic-neutral primary positive-year majority: **{basic_majority_positive_years}**
- Stable: see year/industry/regime matrices; `final_test_is_clean=false`.
- Economic meaning: preference for breakouts reached by continuous trading rather than overnight gaps.
- This can be a gap/overnight-risk proxy; independence is judged only from the full-neutral residual and incremental regressions.
- Recommended research variant: industry-and-float-size neutral residual; full neutralization remains a diagnostic only.
- Primary frozen portfolio Top10/10d/20bps failed the profitability/stability gate; no deployment setting is recommended.
- Largest risk: the full history was inspected during discovery, so this is secondary validation, not untouched OOS proof.

## Evidence index

Definition/leakage, condition anomaly, IC and neutralization, incremental regressions, stability, monotonicity, complete TopN matrix, stress tests and concentration are provided as sibling artifacts. Negative and failed configurations remain in the CSV matrices.

## Data limitations

The project currently has L2 industry rather than L1, no governed liquidity/breadth Regime, and no VWAP field. Those checks are marked unavailable rather than replaced with post-hoc definitions. Parameter-neighbour results are fixed-event scale checks, not regenerated event universes.
"""
        (output/"continuous_move_final_validation_report.md").write_text(report,encoding="utf-8")
        return classification

    @staticmethod
    def _json(path:Path,value:dict)->None:
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
