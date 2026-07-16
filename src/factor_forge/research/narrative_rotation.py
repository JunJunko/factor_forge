from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np
import pandas as pd

from factor_forge.research.concept_portfolio_backtest import TargetBuildResult, cap_and_redistribute


NARRATIVE_KEYWORDS = {
    "spring_risk_on": ["半导体", "商业航天", "新能源", "AI芯片", "人工智能"],
    "shock_and_repair": ["煤炭", "电力", "银行", "能源", "半导体", "CPO", "PCB", "AI芯片"],
    "technology_main_wave": ["CPO", "PCB", "存储", "半导体", "先进封装", "新材料", "AI芯片"],
    "broadening": ["创新药", "消费", "金融", "机器人", "周期", "CPO", "PCB", "存储", "半导体"],
}


@dataclass(frozen=True)
class NarrativeRules:
    holding_days: int = 5
    concepts: int = 3
    leader_slots: int = 2
    successor_slots: int = 1
    stocks_per_concept: int = 3
    maximum_stock_weight: float = 0.10
    minimum_adv20_cny: float = 20_000_000.0
    jaccard_limit: float = 0.80


def attach_rotation_signals(features: pd.DataFrame) -> pd.DataFrame:
    result = features.sort_values(["concept_code", "trade_date"]).copy()
    grouped = result.groupby("concept_code", sort=False)
    result["concept_amount_ma5"] = grouped["concept_amount"].transform(
        lambda values: values.rolling(5, min_periods=4).mean()
    )
    result["concept_amount_ma20"] = grouped["concept_amount"].transform(
        lambda values: values.rolling(20, min_periods=18).mean()
    )
    result["concept_amount_ratio"] = result["concept_amount_ma5"] / result["concept_amount_ma20"]
    rank_columns = {
        "rs_20d": "rotation_rs_rank",
        "rs_momentum_5d": "rotation_rs_momentum_rank",
        "concept_return_5d": "rotation_return5_rank",
        "concept_return_20d": "rotation_return20_rank",
        "concept_amount_ratio": "rotation_amount_rank",
    }
    for source, target in rank_columns.items():
        result[target] = result.groupby("trade_date", observed=True)[source].rank(pct=True)
    leader = (
        result["eligible_concept"].fillna(False)
        & result["rotation_rs_rank"].ge(0.85)
        & result["rs_momentum_5d"].gt(0)
        & result["concept_return_5d"].gt(0)
        & result["rotation_amount_rank"].ge(0.50)
    )
    successor = (
        result["eligible_concept"].fillna(False)
        & result["rotation_rs_momentum_rank"].ge(0.90)
        & result["common_delta_rank"].ge(0.70)
        & result["rotation_return5_rank"].ge(0.60)
        & result["rotation_rs_rank"].lt(0.85)
    )
    result["rotation_leader_score"] = (
        0.40 * result["rotation_rs_rank"]
        + 0.25 * result["rotation_rs_momentum_rank"]
        + 0.20 * result["rotation_amount_rank"]
        + 0.15 * result["common_delta_rank"]
    ).where(leader)
    result["rotation_successor_score"] = (
        0.40 * result["rotation_rs_momentum_rank"]
        + 0.25 * result["common_delta_rank"]
        + 0.20 * result["rotation_amount_rank"]
        + 0.15 * result["rotation_return5_rank"]
    ).where(successor)
    result["rotation_momentum_score"] = result["rotation_return20_rank"].where(
        result["eligible_concept"].fillna(False) & result["concept_return_20d"].gt(0)
    )
    return result.sort_values(["trade_date", "concept_code"]).reset_index(drop=True)


def narrative_stage(date: pd.Timestamp) -> str | None:
    if pd.Timestamp("2026-01-01") <= date <= pd.Timestamp("2026-02-28"):
        return "spring_risk_on"
    if pd.Timestamp("2026-03-01") <= date <= pd.Timestamp("2026-04-30"):
        return "shock_and_repair"
    if pd.Timestamp("2026-05-01") <= date <= pd.Timestamp("2026-06-30"):
        return "technology_main_wave"
    if pd.Timestamp("2026-07-01") <= date <= pd.Timestamp("2026-07-14"):
        return "broadening"
    return None


def build_narrative_rotation_targets(
    features: pd.DataFrame,
    members: pd.DataFrame,
    stock_panel: pd.DataFrame,
    regimes: pd.DataFrame,
    *,
    variant: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    offset: int,
    rules: NarrativeRules = NarrativeRules(),
) -> TargetBuildResult:
    variants = {"momentum_baseline", "causal_leader", "causal_leader_successor", "narrative_assisted"}
    if variant not in variants:
        raise KeyError(variant)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    calendar = pd.Index(sorted(stock_panel.loc[stock_panel["trade_date"].between(start, end), "trade_date"].unique()))
    if not 0 <= offset < rules.holding_days:
        raise ValueError("offset outside holding period")
    feature_days = {pd.Timestamp(d): day.set_index("concept_code") for d, day in features.loc[
        features["trade_date"].isin(calendar)
    ].groupby("trade_date", observed=True)}
    member_days = {pd.Timestamp(d): day[["concept_code", "ts_code"]] for d, day in members.loc[
        members["trade_date"].isin(calendar)
    ].groupby("trade_date", observed=True)}
    stock_days = {pd.Timestamp(d): day.set_index("ts_code") for d, day in stock_panel.loc[
        stock_panel["trade_date"].isin(calendar)
    ].groupby("trade_date", observed=True)}
    regime_map = regimes.set_index("trade_date")["regime"].to_dict()
    rows: list[dict] = []
    selections: list[dict] = []
    entry_dates: list[pd.Timestamp] = []
    exit_dates: list[pd.Timestamp] = []
    for position in range(offset, len(calendar), rules.holding_days):
        entry_position = position + 1
        exit_position = entry_position + rules.holding_days
        if exit_position >= len(calendar):
            break
        date = pd.Timestamp(calendar[position])
        entry_date, exit_date = pd.Timestamp(calendar[entry_position]), pd.Timestamp(calendar[exit_position])
        day = feature_days.get(date)
        day_members = member_days.get(date, pd.DataFrame(columns=["concept_code", "ts_code"]))
        stocks = stock_days.get(date)
        if day is None or stocks is None:
            continue
        regime = str(regime_map.get(date, "neutral"))
        selected = _select_concepts(day, day_members, variant, date, regime, rules)
        if not selected:
            entry_dates.append(entry_date); exit_dates.append(exit_date); continue
        strongest = max(item[2] for item in selected)
        if variant == "narrative_assisted":
            exposure = 0.90
        elif regime == "overheat":
            exposure = 0.60
        elif regime == "retreat":
            exposure = 0.80 if any(item[1] == "leader" and item[2] >= 0.85 for item in selected) else 0.30
        else:
            exposure = 0.90
        contributions: list[dict] = []
        total_score = sum(max(item[2], 0.01) for item in selected)
        concept_counts = day_members.groupby("ts_code", observed=True)["concept_code"].nunique()
        for code, role, score in selected:
            picks = _core_stocks(code, day.loc[code], day_members, stocks, rules)
            if not picks:
                continue
            concept_weight = score / total_score
            for stock in picks:
                contributions.append({
                    "ts_code": stock, "concept_code": code,
                    "raw_weight": concept_weight / len(picks) / math.sqrt(max(concept_counts.get(stock, 1), 1)),
                })
            selections.append({
                "variant": variant, "signal_date": date, "entry_date": entry_date,
                "concept_code": code, "concept_name": day.loc[code, "concept_name"],
                "role": role, "score": score, "regime": regime,
                "narrative_stage": narrative_stage(date), "stocks": len(picks),
            })
        if contributions:
            target = pd.DataFrame(contributions).groupby("ts_code", observed=True).agg(
                raw_weight=("raw_weight", "sum"), supporting_concepts=("concept_code", "nunique")
            ).reset_index()
            target["target_weight"] = cap_and_redistribute(
                target["raw_weight"], min(rules.maximum_stock_weight / exposure, 1.0)
            ).to_numpy() * exposure
            for row in target.itertuples(index=False):
                rows.append({
                    "variant": variant, "signal_date": date, "entry_date": entry_date,
                    "ts_code": row.ts_code, "target_weight": row.target_weight,
                    "supporting_concepts": row.supporting_concepts, "regime": regime,
                    "strongest_concept_score": strongest,
                })
        entry_dates.append(entry_date); exit_dates.append(exit_date)
    targets = pd.DataFrame(rows)
    if targets.empty:
        targets = pd.DataFrame(columns=[
            "variant", "signal_date", "entry_date", "ts_code", "target_weight",
            "supporting_concepts", "regime", "strongest_concept_score",
        ])
    return TargetBuildResult(
        targets, pd.DataFrame(selections),
        sorted(set(entry_dates + exit_dates)), sorted(set(entry_dates)),
    )


def _select_concepts(day: pd.DataFrame, members: pd.DataFrame, variant: str,
                     date: pd.Timestamp, regime: str, rules: NarrativeRules) -> list[tuple[str, str, float]]:
    candidates: list[tuple[str, str, float]] = []
    if variant == "momentum_baseline":
        ranked = day.dropna(subset=["rotation_momentum_score"]).sort_values("rotation_momentum_score", ascending=False)
        candidates = [(str(code), "momentum", float(row["rotation_momentum_score"])) for code, row in ranked.iterrows()]
    elif variant == "causal_leader":
        ranked = day.dropna(subset=["rotation_leader_score"]).sort_values("rotation_leader_score", ascending=False)
        candidates = [(str(code), "leader", float(row["rotation_leader_score"])) for code, row in ranked.iterrows()]
    elif variant == "causal_leader_successor":
        leaders = day.dropna(subset=["rotation_leader_score"]).sort_values("rotation_leader_score", ascending=False)
        successors = day.dropna(subset=["rotation_successor_score"]).sort_values("rotation_successor_score", ascending=False)
        candidates = [(str(code), "leader", float(row["rotation_leader_score"])) for code, row in leaders.iterrows()]
        if regime != "retreat":
            successor_items = [(str(code), "successor", float(row["rotation_successor_score"])) for code, row in successors.iterrows()]
            candidates = candidates[: rules.leader_slots] + successor_items + candidates[rules.leader_slots :]
    else:
        stage = narrative_stage(date)
        if stage is None:
            return []
        pattern = re.compile("|".join(re.escape(item) for item in NARRATIVE_KEYWORDS[stage]))
        allowed = day.loc[day["concept_name"].astype(str).map(lambda value: bool(pattern.search(value)))].copy()
        allowed = allowed.loc[allowed["eligible_concept"].fillna(False)]
        allowed["oracle_score"] = (
            0.60 * allowed["rotation_return5_rank"] + 0.40 * allowed["rotation_rs_momentum_rank"]
        )
        ranked = allowed.dropna(subset=["oracle_score"]).sort_values("oracle_score", ascending=False)
        candidates = [(str(code), "narrative", float(row["oracle_score"])) for code, row in ranked.iterrows()]
    return _deduplicate(candidates, members, rules)


def _deduplicate(candidates: list[tuple[str, str, float]], members: pd.DataFrame,
                 rules: NarrativeRules) -> list[tuple[str, str, float]]:
    sets = members.groupby("concept_code", observed=True)["ts_code"].agg(lambda values: frozenset(values))
    selected: list[tuple[str, str, float]] = []
    selected_sets: list[frozenset] = []
    roles = {"leader": 0, "successor": 0}
    for code, role, score in candidates:
        if role == "leader" and roles["leader"] >= rules.leader_slots and any(item[1] == "successor" for item in candidates):
            continue
        if role == "successor" and roles["successor"] >= rules.successor_slots:
            continue
        member_set = sets.get(code, frozenset())
        if not member_set or any(len(member_set & existing) / len(member_set | existing) > rules.jaccard_limit for existing in selected_sets):
            continue
        selected.append((code, role, score)); selected_sets.append(member_set)
        if role in roles:
            roles[role] += 1
        if len(selected) >= rules.concepts:
            break
    return selected


def _core_stocks(concept_code: str, concept: pd.Series, members: pd.DataFrame,
                 stocks: pd.DataFrame, rules: NarrativeRules) -> list[str]:
    codes = members.loc[members["concept_code"].eq(concept_code), "ts_code"].astype(str)
    available = stocks.loc[stocks.index.intersection(codes)].copy()
    available = available.loc[
        available["is_tradeable"].fillna(False)
        & available["amount_ma20"].ge(rules.minimum_adv20_cny)
        & available["stock_return_20d"].gt(0)
    ]
    if available.empty:
        return []
    relative = available["stock_return_20d"] - float(concept.get("concept_return_20d", 0.0))
    available["core_score"] = (
        np.log(available["amount_ma20"].clip(lower=1)).rank(pct=True) * 0.45
        + available["stock_return_20d"].rank(pct=True) * 0.35
        + relative.rank(pct=True) * 0.20
    )
    return available.sort_values(["core_score", "amount_ma20"], ascending=False).head(
        rules.stocks_per_concept
    ).index.astype(str).tolist()
