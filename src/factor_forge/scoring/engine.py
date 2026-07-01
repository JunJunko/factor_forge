from __future__ import annotations

import numpy as np

from factor_forge.config import FactorSpec


def _band(value: float | None, bands: list[tuple[float, int]]) -> int:
    if value is None or not np.isfinite(value):
        return 0
    for threshold, score in bands:
        if value >= threshold:
            return score
    return 0


class AlphaScorer:
    """V1 100-point profile. Missing evidence earns zero, never an invented estimate."""

    def __init__(self, profile: dict | None = None):
        self.profile = profile or {}

    def score(self, factor: FactorSpec, l0: dict, l1: dict, l2: list[dict]) -> dict:
        hard_flags = {
            "future_data": l0["metrics"].get("future_data_violations", 0) > 0,
            "insufficient_coverage": not l0["checks"].get("coverage", False),
            "invalid_execution_price": False,
            "survivorship_bias": False,
            "severe_oos_reversal": False,
            "cost_failure": False,
        }
        primary = [r for r in l2 if r["universe"] == "liquid" and r["top_n"] == 2 and r["cost_bps"] == 20]
        gross = [r for r in l2 if r["universe"] == "liquid" and r["top_n"] == 2 and r["cost_bps"] == 0]
        oos = self._oos(primary, l1)
        tradability = self._tradability(primary, gross)
        stability = self._stability(primary)
        topn = self._topn(l2)
        statistics = self._statistics(l1, factor.factor.expected_shape)
        independence = {
            "score": 2 if factor.factor.hypothesis.strip() else 0,
            "factor_correlation": {"score": 0, "status": "UNAVAILABLE_NO_FACTOR_LIBRARY"},
            "incremental_value": {"score": 0, "status": "UNAVAILABLE_NO_COMPOSITE_PORTFOLIO"},
            "economic_rationale": {"score": 2 if factor.factor.hypothesis.strip() else 0},
        }
        dimension_scores = {
            "oos_effectiveness": oos["score"], "tradable_performance": tradability["score"],
            "stability": stability["score"], "topn_structure": topn["score"],
            "statistical_evidence": statistics["score"], "independence_explanation": independence["score"],
        }
        hard_flags["severe_oos_reversal"] = oos.get("severe_reversal", False)
        hard_flags["cost_failure"] = bool(primary) and np.median([x["metrics"]["annualized_excess_return"] for x in primary]) <= 0
        total = int(sum(dimension_scores.values()))
        classification_config = self.profile.get("classification", {})
        rejected_max = classification_config.get("rejected", {}).get("max_score", 49)
        watch_max = classification_config.get("watchlist", {}).get("max_score", 64)
        candidate_max = classification_config.get("candidate", {}).get("max_score", 79)
        approved = classification_config.get("approved", {})
        approved_minimums = approved.get("minimum_dimension_scores", {
            "oos_effectiveness": 17, "tradable_performance": 17, "stability": 12,
        })
        if any(hard_flags[k] for k in ["future_data", "insufficient_coverage", "invalid_execution_price", "survivorship_bias"]):
            classification = "INVALID"
        elif hard_flags["severe_oos_reversal"]:
            classification = "REJECTED"
        elif total <= rejected_max:
            classification = "REJECTED"
        elif total <= watch_max:
            classification = "WATCHLIST"
        elif total <= candidate_max:
            classification = "CANDIDATE"
        elif all(dimension_scores.get(name, 0) >= minimum for name, minimum in approved_minimums.items()):
            classification = "APPROVED"
        else:
            classification = "CANDIDATE"
        return {
            "profile": self.profile.get("profile", "short_term_topn_v1"), "total_score": total,
            "classification": classification, "dimension_scores": dimension_scores,
            "hard_flags": hard_flags,
            "details": {"oos": oos, "tradability": tradability, "stability": stability,
                        "topn_structure": topn, "statistics": statistics, "independence": independence},
        }

    @staticmethod
    def _oos(primary: list[dict], l1: dict) -> dict:
        if not primary:
            return {"score": 0, "status": "NO_PRIMARY_RESULT"}
        oos_returns, is_returns, positive_windows = [], [], []
        for row in primary:
            daily = row["daily"]
            split = max(int(len(daily) * 0.8), 1)
            is_excess = daily["excess_return"].iloc[:split]
            oos_excess = daily["excess_return"].iloc[split:]
            annual = lambda x: float(x.mean() * 252) if len(x) else 0.0
            is_returns.append(annual(is_excess)); oos_returns.append(annual(oos_excess))
            oos_daily = daily.iloc[split:]
            positive_windows.append(
                float((oos_daily.resample("YE", on="trade_date")["excess_return"].sum() > 0).mean())
                if len(oos_daily) else 0.0
            )
        is_value, oos_value = float(np.median(is_returns)), float(np.median(oos_returns))
        retention = oos_value / is_value if is_value > 0 else (1.0 if oos_value > 0 else -1.0)
        net_score = 8 if oos_value > 0 else 0
        retention_score = _band(retention, [(0.7, 6), (0.4, 5), (0.2, 3), (0.0, 1)])
        walk_score = _band(float(np.mean(positive_windows)), [(0.8, 7), (0.6, 5), (0.4, 3), (0.2, 1)])
        l1_oos = [r["oos_rank_ic"]["mean"] for r in l1["results"] if r["oos_rank_ic"]["mean"] is not None]
        direction = float(np.median(l1_oos)) if l1_oos else 0.0
        direction_score = 4 if direction > 0 else 0
        return {"score": net_score + retention_score + walk_score + direction_score,
                "oos_annualized_excess": oos_value, "is_annualized_excess": is_value,
                "retention_ratio": retention, "oos_ic": direction,
                "severe_reversal": is_value > 0 and oos_value < 0 and direction < 0}

    @staticmethod
    def _tradability(primary: list[dict], gross: list[dict]) -> dict:
        if not primary:
            return {"score": 0, "status": "NO_PRIMARY_RESULT"}
        excess = float(np.median([x["metrics"]["annualized_excess_return"] for x in primary]))
        calmar_values = [x["metrics"]["calmar"] for x in primary if x["metrics"]["calmar"] is not None]
        calmar = float(np.median(calmar_values)) if calmar_values else 0.0
        execution = float(np.median([x["metrics"]["execution_rate"] for x in primary]))
        gross_excess = float(np.median([x["metrics"]["annualized_excess_return"] for x in gross])) if gross else excess
        resilience = excess / gross_excess if gross_excess > 0 else 0.0
        score = _band(excess, [(0.15, 10), (0.10, 8), (0.05, 6), (0.02, 3), (0.0, 1)])
        score += _band(calmar, [(1.0, 6), (0.7, 5), (0.4, 3), (0.2, 1)])
        score += _band(resilience, [(0.8, 5), (0.6, 4), (0.4, 2), (0.2, 1)])
        score += _band(execution, [(0.95, 4), (0.85, 3), (0.70, 2), (0.50, 1)])
        return {"score": score, "annualized_excess": excess, "calmar": calmar,
                "cost_resilience": resilience, "execution_rate": execution}

    @staticmethod
    def _stability(primary: list[dict]) -> dict:
        if not primary:
            return {"score": 0, "status": "NO_PRIMARY_RESULT"}
        yearly, rolling, concentration = [], [], []
        for row in primary:
            daily = row["daily"].copy()
            annual = daily.resample("YE", on="trade_date")["excess_return"].sum()
            yearly.append(float((annual > 0).mean()) if len(annual) else 0.0)
            roll = daily["excess_return"].rolling(252, min_periods=63).sum().dropna()
            rolling.append(float((roll > 0).mean()) if len(roll) else 0.0)
            positive = daily["excess_return"].clip(lower=0).sort_values(ascending=False)
            concentration.append(float(positive.head(10).sum() / positive.sum()) if positive.sum() else 1.0)
        year_value, rolling_value, conc = map(float, map(np.median, [yearly, rolling, concentration]))
        score = _band(year_value, [(0.7, 6), (0.6, 5), (0.5, 3), (0.4, 1)])
        score += _band(rolling_value, [(0.7, 5), (0.6, 4), (0.5, 3), (0.4, 1)])
        # Parameter neighborhood is intentionally unavailable until neighbors are declared and run.
        score += _band(1.0 - conc, [(0.8, 4), (0.6, 3), (0.4, 2), (0.2, 1)])
        return {"score": score, "year_positive_ratio": year_value,
                "rolling_positive_ratio": rolling_value, "top10_day_concentration": conc,
                "parameter_neighborhood": "UNAVAILABLE_NOT_DECLARED"}

    @staticmethod
    def _topn(rows: list[dict]) -> dict:
        sample = [r for r in rows if r["universe"] == "liquid" and r["cost_bps"] == 20]
        curve = {}
        for n in [2, 5, 10, 20]:
            values = [r["metrics"]["annualized_excess_return"] for r in sample if r["top_n"] == n]
            curve[n] = float(np.median(values)) if values else None
        valid = [curve[n] for n in [2, 5, 10, 20] if curve[n] is not None]
        if len(valid) < 2 or curve[2] is None or curve[2] <= 0:
            score = 0
        elif all(a >= b for a, b in zip(valid, valid[1:])) and all(x > 0 for x in valid):
            score = 10
        elif curve[5] is not None and curve[5] > 0:
            score = 7
        else:
            score = 5
        return {"score": score, "annualized_excess_curve": curve}

    @staticmethod
    def _statistics(l1: dict, shape: str) -> dict:
        results = l1["results"]
        ic = [r["rank_ic"]["mean"] for r in results if r["rank_ic"]["mean"] is not None]
        icir = [r["rank_ic"]["icir"] for r in results if r["rank_ic"]["icir"] is not None]
        q_values = [r["fdr_q"] for r in results if r["fdr_q"] is not None]
        structure = [r["monotonicity"] for r in results if r["monotonicity"] is not None]
        mean_ic = float(np.median(ic)) if ic else 0.0
        mean_icir = float(np.median(icir)) if icir else 0.0
        q = float(np.median(q_values)) if q_values else 1.0
        structure_value = float(np.median(structure)) if structure else 0.0
        score = _band(mean_ic, [(0.05, 3), (0.03, 2), (0.01, 1)])
        score += _band(mean_icir, [(1.0, 3), (0.5, 2), (0.2, 1)])
        score += 2 if q <= 0.05 else (1 if q <= 0.10 else 0)
        score += _band(structure_value, [(0.8, 2), (0.5, 1)])
        return {"score": score, "median_rank_ic": mean_ic, "median_icir": mean_icir,
                "median_fdr_q": q, "distribution_structure": structure_value, "expected_shape": shape}
