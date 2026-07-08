from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from .regime import TimingRegimeConfig, TimingRegimeRunner


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StableFactorSelectionConfig(StrictModel):
    version: int = 1
    name: str = "timing_stable_factors_v1"
    dataset_path: Path
    feature_names_path: Path | None = None
    output_root: Path = Path("artifacts/timing_stable_factors")
    start_date: str = "2023-04-18"
    label_column: str = "label_10d_excess_return"
    regime_method: Literal["hmm", "gmm"] = "hmm"
    n_components: int = Field(default=3, ge=2, le=5)
    history_days: int = Field(default=252, ge=120, le=1260)
    random_states: list[int] = Field(default_factory=lambda: [11, 42, 73])
    min_regime_observations: int = Field(default=80, ge=20)
    min_seed_support: int = Field(default=2, ge=1)
    min_mean_abs_rank_ic: float = Field(default=0.10, ge=0, le=1)
    max_pairwise_corr: float = Field(default=0.85, ge=0, le=1)
    max_selected_per_group: int = Field(default=3, ge=1, le=10)
    max_selected_total: int = Field(default=30, ge=5, le=100)
    regime_features: list[str] = Field(default_factory=lambda: [
        "index_ret_20d",
        "index_ret_60d",
        "index_vol_20d",
        "index_drawdown_60d",
        "up_ratio",
        "rzmre_ratio",
        "put_call_log",
        "iv_atm",
        "fut_near_basis_ann",
        "fut_ls_log",
        "pmi",
        "epu_log",
    ])
    exclude_patterns: list[str] = Field(default_factory=lambda: [
        "_low_", "_high_", "label_", "state_probability_",
    ])


def load_stable_factor_selection_config(path: str | Path) -> StableFactorSelectionConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return StableFactorSelectionConfig.model_validate(yaml.safe_load(handle) or {})


class StableFactorSelectionRunner:
    def run(self, config_path: str | Path) -> dict:
        config_path = Path(config_path)
        cfg = load_stable_factor_selection_config(config_path)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + hashlib.sha256(config_path.read_bytes()).hexdigest()[:8]
        output = cfg.output_root / f"{cfg.name}_{run_id}"
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.yaml").write_bytes(config_path.read_bytes())

        helper = TimingRegimeRunner()
        base_cfg = self._regime_config(cfg)
        dataset, feature_names = helper._load_dataset(base_cfg)
        regime_features = [column for column in cfg.regime_features if column in dataset.columns]
        factors = self._diagnostic_features(dataset, feature_names, cfg, regime_features)
        seed_ic_frames: list[pd.DataFrame] = []
        seed_quantile_frames: list[pd.DataFrame] = []
        seed_summary_frames: list[pd.DataFrame] = []

        for seed in cfg.random_states:
            run_cfg = self._regime_config(cfg, seed=seed)
            states = helper._walk_forward_regimes(dataset, regime_features, run_cfg)
            factor_ic = helper._factor_regime_ic(dataset, states, factors, cfg.label_column)
            quantiles = helper._factor_regime_quantiles(dataset, states, factors, run_cfg, cfg.label_column)
            regime_summary = helper._regime_summary(dataset, states, cfg.label_column)
            for frame in [factor_ic, quantiles, regime_summary]:
                frame.insert(0, "random_state", seed)
            seed_ic_frames.append(factor_ic)
            seed_quantile_frames.append(quantiles)
            seed_summary_frames.append(regime_summary)

        all_ic = pd.concat(seed_ic_frames, ignore_index=True)
        all_quantiles = pd.concat(seed_quantile_frames, ignore_index=True)
        all_regimes = pd.concat(seed_summary_frames, ignore_index=True)
        all_ic.to_csv(output / "seed_factor_regime_ic.csv", index=False, encoding="utf-8-sig")
        all_quantiles.to_csv(output / "seed_factor_regime_quantiles.csv", index=False, encoding="utf-8-sig")
        all_regimes.to_csv(output / "seed_regime_summary.csv", index=False, encoding="utf-8-sig")

        candidates = self._candidate_scores(all_ic, all_quantiles, cfg)
        candidates.to_csv(output / "stable_factor_candidates.csv", index=False, encoding="utf-8-sig")
        corr = dataset[candidates["factor"].drop_duplicates().tolist()].corr(method="spearman").abs()
        corr.to_csv(output / "stable_factor_correlation.csv", encoding="utf-8-sig")
        selected = self._select_diverse_factors(candidates, corr, cfg)
        selected.to_csv(output / "stable_factor_selected.csv", index=False, encoding="utf-8-sig")
        groups = self._group_summary(selected)
        groups.to_csv(output / "stable_factor_groups.csv", index=False, encoding="utf-8-sig")

        summary = {
            "status": "SUCCESS",
            "run_dir": str(output),
            "dataset_path": str(cfg.dataset_path),
            "label_column": cfg.label_column,
            "regime": {
                "method": cfg.regime_method,
                "n_components": cfg.n_components,
                "history_days": cfg.history_days,
                "random_states": cfg.random_states,
            },
            "candidate_count": int(len(candidates)),
            "selected_count": int(len(selected)),
            "selected_factors": selected["factor"].tolist(),
        }
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        (output / "stable_factor_report.md").write_text(
            self._report(summary, candidates, selected, groups), encoding="utf-8"
        )
        return summary

    @staticmethod
    def _regime_config(cfg: StableFactorSelectionConfig, seed: int | None = None) -> TimingRegimeConfig:
        return TimingRegimeConfig.model_validate({
            "name": cfg.name,
            "dataset_path": cfg.dataset_path,
            "feature_names_path": cfg.feature_names_path,
            "output_root": cfg.output_root,
            "start_date": cfg.start_date,
            "label_column": cfg.label_column,
            "label_columns": [cfg.label_column],
            "regime_features": cfg.regime_features,
            "regime": {
                "method": cfg.regime_method,
                "n_components": cfg.n_components,
                "history_days": cfg.history_days,
                "random_state": seed if seed is not None else cfg.random_states[0],
                "covariance_type": "diag",
                "refit_frequency": "monthly",
                "zscore_window": 252,
                "zscore_min_periods": 60,
                "max_iterations": 300,
                "tolerance": 0.0001,
                "min_covar": 0.0001,
            },
            "diagnostics": {
                "min_coverage": 0.55,
                "quantiles": 5,
                "top_curve_factors": 30,
                "exclude_patterns": cfg.exclude_patterns,
            },
            "interaction_model": {
                "enabled": False,
            },
        })

    @staticmethod
    def _diagnostic_features(
        dataset: pd.DataFrame,
        feature_names: list[str],
        cfg: StableFactorSelectionConfig,
        regime_features: list[str],
    ) -> list[str]:
        excluded = set(regime_features)
        return [
            column for column in feature_names
            if column in dataset.columns
            and column not in excluded
            and dataset[column].notna().mean() >= 0.55
            and not any(pattern in column for pattern in cfg.exclude_patterns)
        ]

    @staticmethod
    def _candidate_scores(
        all_ic: pd.DataFrame,
        all_quantiles: pd.DataFrame,
        cfg: StableFactorSelectionConfig,
    ) -> pd.DataFrame:
        state_ic = all_ic.loc[all_ic["state"].ge(0)].dropna(subset=["rank_ic"]).copy()
        state_ic = state_ic.loc[state_ic["observations"].ge(cfg.min_regime_observations)]
        quantile_spreads = []
        if not all_quantiles.empty:
            for keys, frame in all_quantiles.groupby(["random_state", "factor", "state", "state_name"]):
                pivot = frame.set_index("quantile")["mean_forward_return"]
                if len(pivot) >= 2:
                    quantile_spreads.append({
                        "random_state": keys[0],
                        "factor": keys[1],
                        "state": keys[2],
                        "state_name": keys[3],
                        "quantile_spread": pivot.loc[pivot.index.max()] - pivot.loc[pivot.index.min()],
                    })
        spreads = pd.DataFrame(quantile_spreads)
        rows = []
        for (factor, state, state_name), frame in state_ic.groupby(["factor", "state", "state_name"]):
            signs = np.sign(frame["rank_ic"].to_numpy(float))
            nonzero = signs[signs != 0]
            dominant_sign = np.sign(nonzero.sum()) if len(nonzero) else 0
            support = int((signs == dominant_sign).sum()) if dominant_sign != 0 else 0
            local_spreads = spreads.loc[
                spreads["factor"].eq(factor)
                & spreads["state"].eq(state)
                & spreads["random_state"].isin(frame["random_state"])
            ] if not spreads.empty else pd.DataFrame()
            spread_direction_match = np.nan
            mean_quantile_spread = np.nan
            if not local_spreads.empty and dominant_sign != 0:
                spread_direction_match = float((np.sign(local_spreads["quantile_spread"]) == dominant_sign).mean())
                mean_quantile_spread = float(local_spreads["quantile_spread"].mean())
            rows.append({
                "factor": factor,
                "group": _factor_group(factor),
                "state": int(state),
                "state_name": state_name,
                "seed_count": int(frame["random_state"].nunique()),
                "direction_support": support,
                "dominant_sign": int(dominant_sign),
                "mean_rank_ic": float(frame["rank_ic"].mean()),
                "mean_abs_rank_ic": float(frame["rank_ic"].abs().mean()),
                "min_abs_rank_ic": float(frame["rank_ic"].abs().min()),
                "mean_observations": float(frame["observations"].mean()),
                "mean_quantile_spread": mean_quantile_spread,
                "spread_direction_match": spread_direction_match,
            })
        candidates = pd.DataFrame(rows)
        if candidates.empty:
            return candidates
        candidates["stable_score"] = (
            candidates["mean_abs_rank_ic"].fillna(0)
            * (candidates["direction_support"] / max(len(cfg.random_states), 1))
            * candidates["spread_direction_match"].fillna(0.5)
        )
        return candidates.loc[
            candidates["direction_support"].ge(cfg.min_seed_support)
            & candidates["mean_abs_rank_ic"].ge(cfg.min_mean_abs_rank_ic)
        ].sort_values("stable_score", ascending=False).reset_index(drop=True)

    @staticmethod
    def _select_diverse_factors(
        candidates: pd.DataFrame,
        corr: pd.DataFrame,
        cfg: StableFactorSelectionConfig,
    ) -> pd.DataFrame:
        selected_rows = []
        group_counts: dict[str, int] = {}
        selected_factors: list[str] = []
        for row in candidates.itertuples(index=False):
            if group_counts.get(row.group, 0) >= cfg.max_selected_per_group:
                continue
            too_correlated = False
            for existing in selected_factors:
                if row.factor in corr.index and existing in corr.columns and corr.loc[row.factor, existing] > cfg.max_pairwise_corr:
                    too_correlated = True
                    break
            if too_correlated:
                continue
            selected_rows.append(row._asdict())
            selected_factors.append(row.factor)
            group_counts[row.group] = group_counts.get(row.group, 0) + 1
            if len(selected_rows) >= cfg.max_selected_total:
                break
        return pd.DataFrame(selected_rows)

    @staticmethod
    def _group_summary(selected: pd.DataFrame) -> pd.DataFrame:
        if selected.empty:
            return pd.DataFrame(columns=["group", "selected_count", "mean_stable_score"])
        return selected.groupby("group", as_index=False).agg(
            selected_count=("factor", "nunique"),
            mean_stable_score=("stable_score", "mean"),
            factors=("factor", lambda item: ", ".join(item)),
        ).sort_values("selected_count", ascending=False)

    @staticmethod
    def _report(summary, candidates, selected, groups) -> str:
        top_candidates = candidates.head(30)
        return "\n".join([
            "# Stable Timing Factor Selection",
            "",
            f"- Dataset: `{summary['dataset_path']}`",
            f"- Label: `{summary['label_column']}`",
            f"- Regime: `{summary['regime']}`",
            f"- Candidates: `{summary['candidate_count']}`",
            f"- Selected: `{summary['selected_count']}`",
            "",
            "## Selected Factors",
            selected.to_markdown(index=False, floatfmt=".4f") if not selected.empty else "No factors selected.",
            "",
            "## Group Summary",
            groups.to_markdown(index=False, floatfmt=".4f") if not groups.empty else "No groups selected.",
            "",
            "## Top Candidates",
            top_candidates.to_markdown(index=False, floatfmt=".4f") if not top_candidates.empty else "No candidates.",
            "",
        ]) + "\n"


def _factor_group(name: str) -> str:
    if name.startswith(("erp", "earnings_yield", "pe_ttm", "bond_10y")):
        return "valuation"
    if name.startswith(("up_ratio", "adv_dec", "breadth", "index_ret", "index_vol", "index_drawdown", "index_above")):
        return "market_breadth_risk"
    if name.startswith(("rzmre", "rzye")):
        return "leverage_funding"
    if name.startswith(("put_call", "iv_")):
        return "option_sentiment"
    if name.startswith(("fut_",)):
        return "futures_sentiment"
    if name.startswith(("main_net",)):
        return "main_moneyflow"
    if name.startswith(("pmi", "cpi", "epu", "macro_")):
        return "macro"
    return "other"


def _json_default(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
