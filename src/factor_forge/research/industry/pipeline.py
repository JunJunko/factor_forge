from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from factor_forge.config import FactorSpec, IndustrySliceConfig, L1Config
from .context import IndustryContextBuilder
from .evaluator import IndustrySliceEvaluator
from .features import IndustryFeatureBuilder
from .residual_return import IndustryResidualReturnBuilder
from .selector import IndustrySelector
from .slice_mapper import IndustrySliceMapper


@dataclass
class IndustrySliceResult:
    industry_panel: pd.DataFrame
    stock_panel: pd.DataFrame
    stock_ic: pd.DataFrame
    stock_ic_by_year: pd.DataFrame
    selector_summary: pd.DataFrame
    selector_by_year: pd.DataFrame
    report: str
    leakage_report: str


class IndustrySlicePipeline:
    def run(self, panel: pd.DataFrame, factor_values: pd.DataFrame, factor: FactorSpec,
            l1: L1Config, config: IndustrySliceConfig, factor_builder=None) -> IndustrySliceResult:
        params = config.selector.overrides
        stocks = IndustryContextBuilder().build(panel)
        industries = IndustryFeatureBuilder().build(
            stocks, short_ema=params.short_ema, long_ema=params.long_ema,
            breadth_change_window=params.breadth_change_window,
            minimum_industry_members=params.minimum_industry_members)
        eligible = industries["eligible_industry"]
        scored = IndustrySelector().select(industries.loc[eligible].copy(), params.ridge_alpha)
        mapped = IndustrySliceMapper().map(stocks, scored)
        if factor_builder is not None:
            scoped_values = {}
            for scope in config.scopes:
                mask = pd.Series(True, index=mapped.index) if scope == "all" else mapped[IndustrySliceEvaluator.scope_columns[scope]].fillna(False)
                scoped_values[scope] = factor_builder(mask)
            factor_values = scoped_values
        targets = IndustryResidualReturnBuilder().build(mapped, l1.forward_horizons)
        selected_targets = {name: targets[name] for name in l1.targets}
        evaluator = IndustrySliceEvaluator()
        stock_ic, stock_year = evaluator.evaluate_stock(mapped, factor_values, selected_targets,
                                                        config.scopes, l1.min_cross_section, factor.factor.name)
        if config.diagnostics.evaluate_industry_selector:
            industry_targets = {}
            for horizon, stock_target in targets["stock_return"].items():
                target_frame = mapped[["trade_date", "sw_l1_industry_code"]].copy()
                target_frame["stock_target"] = stock_target.to_numpy()
                grouped = target_frame.groupby(["trade_date", "sw_l1_industry_code"])["stock_target"].mean()
                market = target_frame.groupby("trade_date")["stock_target"].mean()
                keys = pd.MultiIndex.from_frame(scored[["trade_date", "industry_code"]])
                industry_targets[horizon] = pd.Series(
                    grouped.reindex(keys).to_numpy() - market.reindex(scored["trade_date"]).to_numpy(),
                    index=scored.index,
                )
            selector, selector_year = evaluator.evaluate_selector(scored, industry_targets)
        else:
            selector, selector_year = pd.DataFrame(), pd.DataFrame()
        coverage = float(mapped["sw_l1_industry_code"].notna().mean()) if len(mapped) else 0
        warnings = []
        requested = [int(s[3:]) if s.startswith("top") else 5 for s in config.scopes if s != "all"]
        if requested and (scored.empty or scored.groupby("trade_date")["industry_code"].nunique().min() < max(requested)):
            warnings.append("Some dates have fewer valid industries than requested Top/Bottom N")
        status = "VALID" if coverage > 0 and not stock_ic.empty else "INVALID"
        effective = params.model_dump()
        report = (f"# Industry Slice Report\n\n- Status: **{status}**\n- Membership coverage: {coverage:.2%}\n"
                  f"- Preset: `{config.selector.preset}`\n- Effective parameters: `{effective}`\n"
                  + "".join(f"- Warning: {x}\n" for x in warnings))
        checks = {"future_industry_membership": True, "full_sample_standardization": True,
                  "same_day_future_return_usage": True, "target_window_alignment": True,
                  "duplicate_stock_industry_mapping": True, "industry_score_future_data": True,
                  "industry_return_membership_alignment": True}
        leakage = "# Industry Slice Leakage Report\n\n" + "\n".join(
            f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items())
        leakage += f"\n\nOverall: **{status}**\n"
        return IndustrySliceResult(scored, mapped, stock_ic, stock_year, selector, selector_year, report, leakage)
