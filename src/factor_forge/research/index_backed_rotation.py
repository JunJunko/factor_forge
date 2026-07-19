from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd


DEFAULT_THEME_KEYWORDS = (
    "人工智能", "机器人", "芯片", "半导体", "计算机", "软件", "云计算", "大数据",
    "通信", "5G", "电子", "互联网", "游戏", "传媒", "光伏", "新能源", "电池",
    "储能", "风电", "电力", "电网", "汽车", "智能车", "创新药", "医药", "医疗",
    "生物", "白酒", "食品", "消费", "农业", "养殖", "稀土", "有色", "黄金",
    "煤炭", "钢铁", "化工", "建材", "军工", "航空", "航天", "机械", "装备",
    "机床", "证券", "银行", "保险", "房地产", "基建", "物流", "旅游", "家电",
    "港口", "油气",
)

CLUSTER_RULES = (
    ("ai_robotics", ("人工智能", "机器人", "智能")),
    ("semiconductors", ("芯片", "半导体")),
    ("software_media", ("软件", "云计算", "大数据", "互联网", "游戏", "传媒")),
    ("electronics_communication", ("电子", "通信", "5G")),
    ("solar_wind", ("光伏", "风电")),
    ("battery_storage", ("电池", "储能", "新能源")),
    ("power_grid", ("电力", "电网")),
    ("automotive", ("汽车", "智能车")),
    ("healthcare", ("创新药", "医药", "医疗", "生物")),
    ("consumer", ("白酒", "食品", "消费", "旅游", "家电")),
    ("agriculture", ("农业", "养殖")),
    ("materials", ("稀土", "有色", "黄金", "钢铁", "化工", "建材")),
    ("energy", ("煤炭", "油气")),
    ("defense", ("军工", "航空", "航天")),
    ("industrials", ("机械", "装备", "机床")),
    ("financials", ("证券", "银行", "保险")),
    ("real_estate_infra", ("房地产", "基建")),
    ("transportation", ("物流", "港口")),
)


def classify_theme_cluster(name: str) -> str:
    value = str(name or "")
    for cluster, keywords in CLUSTER_RULES:
        if any(keyword in value for keyword in keywords):
            return cluster
    return "other_theme"


def official_theme_etf_candidates(
    etf_basic: pd.DataFrame,
    *,
    as_of: str,
    theme_keywords: Iterable[str] = DEFAULT_THEME_KEYWORDS,
    allowed_index_suffixes: Iterable[str] = (".CSI", ".SH", ".SZ"),
) -> pd.DataFrame:
    """Return all domestic thematic ETFs without using future liquidity data."""
    basic = etf_basic.copy()
    basic["list_date_parsed"] = pd.to_datetime(basic["list_date"], errors="coerce")
    keywords = tuple(theme_keywords)
    suffixes = tuple(allowed_index_suffixes)
    index_name = basic["index_name"].fillna("").astype(str)
    thematic = index_name.map(lambda value: any(keyword in value for keyword in keywords))
    result = basic.loc[
        basic["etf_type"].eq("纯境内")
        & basic["index_code"].fillna("").astype(str).str.endswith(suffixes)
        & basic["list_date_parsed"].le(pd.Timestamp(as_of))
        & thematic
    ].copy()
    result["cluster"] = result["index_name"].map(classify_theme_cluster)
    return result.sort_values(["index_code", "list_date_parsed", "ts_code"]).reset_index(drop=True)


def recover_delisted_theme_etf_candidates(
    etf_basic: pd.DataFrame,
    fund_basic: pd.DataFrame,
    etf_indexes: pd.DataFrame,
    *,
    as_of: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recover delisted ETF tracking indexes from exact normalized benchmarks.

    No fuzzy match is accepted. Ambiguous or non-thematic matches remain excluded
    and are retained in the audit table.
    """
    delisted = etf_basic.loc[etf_basic["list_status"].eq("D")].copy()
    funds = fund_basic[[
        column for column in ("ts_code", "benchmark", "delist_date")
        if column in fund_basic
    ]].drop_duplicates("ts_code", keep="last")
    delisted = delisted.merge(funds, on="ts_code", how="left", validate="one_to_one")
    names: dict[str, set[tuple[str, str]]] = {}
    domestic = etf_indexes.loc[
        etf_indexes["ts_code"].fillna("").astype(str).str.endswith((".CSI", ".SH", ".SZ"))
    ]
    for item in domestic.itertuples(index=False):
        for raw_name in (getattr(item, "indx_name", None), getattr(item, "indx_csname", None)):
            normalized = _normalize_index_benchmark(raw_name)
            if normalized:
                names.setdefault(normalized, set()).add((str(item.ts_code), str(item.indx_name)))
    rows = []
    for item in delisted.itertuples(index=False):
        normalized = _normalize_index_benchmark(getattr(item, "benchmark", None))
        matches = sorted(names.get(normalized, set()))
        unique = matches[0] if len(matches) == 1 else None
        index_code = unique[0] if unique else None
        index_name = unique[1] if unique else None
        cluster = classify_theme_cluster(index_name) if unique else None
        thematic = bool(unique and cluster != "other_theme")
        rows.append({
            **item._asdict(),
            "benchmark_normalized": normalized,
            "recovered_index_code": index_code,
            "recovered_index_name": index_name,
            "recovered_cluster": cluster,
            "exact_benchmark_matches": len(matches),
            "recovery_pass": thematic,
            "recovery_reason": (
                "exact_normalized_benchmark" if thematic
                else "ambiguous_or_non_theme_or_unmatched"
            ),
        })
    audit = pd.DataFrame(rows)
    recovered = audit.loc[
        audit["recovery_pass"]
        & pd.to_datetime(audit["list_date"], errors="coerce").le(pd.Timestamp(as_of))
    ].copy()
    if recovered.empty:
        return recovered, audit
    recovered["index_code"] = recovered["recovered_index_code"]
    recovered["index_name"] = recovered["recovered_index_name"]
    recovered["cluster"] = recovered["recovered_cluster"]
    recovered["list_date_parsed"] = pd.to_datetime(recovered["list_date"], errors="coerce")
    recovered["mapping_source"] = "delisted_fund_benchmark_exact"
    return recovered.reset_index(drop=True), audit.reset_index(drop=True)


def _normalize_index_benchmark(value: object) -> str:
    text = str(value or "").upper()
    for token in (
        "收益率", "价格指数", "全收益指数", "指数", "100%", "×", "*",
        "（", "）", "(", ")", "人民币", "CNY",
    ):
        text = text.replace(token, "")
    return re.sub(r"[^0-9A-Z\u4e00-\u9fff]", "", text)


def select_exact_index_etf_mapping(
    etf_basic: pd.DataFrame,
    recent_daily: pd.DataFrame,
    recent_share: pd.DataFrame,
    *,
    selection_cutoff: str,
    minimum_adv_cny: float,
    minimum_aum_cny: float,
    minimum_observations: int,
    theme_keywords: Iterable[str] = DEFAULT_THEME_KEYWORDS,
    allowed_index_suffixes: Iterable[str] = (".CSI", ".SH", ".SZ"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one liquid domestic ETF for each exact official tracking index."""
    candidate = official_theme_etf_candidates(
        etf_basic, as_of=selection_cutoff, theme_keywords=theme_keywords,
        allowed_index_suffixes=allowed_index_suffixes,
    )
    candidate = candidate.loc[candidate["list_status"].eq("L")].copy()

    daily = recent_daily.copy()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"].astype(str))
    daily = daily.loc[daily["trade_date"].le(pd.Timestamp(selection_cutoff))]
    daily["amount_cny"] = pd.to_numeric(daily["amount"], errors="coerce") * 1000.0
    stats = daily.groupby("ts_code", observed=True).agg(
        observations=("trade_date", "nunique"),
        adv_cny=("amount_cny", "mean"),
        last_close=("close", "last"),
        last_trade_date=("trade_date", "max"),
    ).reset_index()
    shares = recent_share.copy()
    shares["trade_date"] = pd.to_datetime(shares["trade_date"].astype(str))
    shares["fd_share"] = pd.to_numeric(shares["fd_share"], errors="coerce")
    latest_share = shares.loc[
        shares["trade_date"].le(pd.Timestamp(selection_cutoff))
    ].sort_values("trade_date").groupby("ts_code", observed=True).tail(1)
    stats = stats.merge(latest_share[["ts_code", "fd_share"]], on="ts_code", how="left")
    stats["aum_cny"] = stats["last_close"] * stats["fd_share"] * 10000.0
    audit = candidate.merge(stats, on="ts_code", how="left")
    audit["eligible_liquidity"] = (
        audit["observations"].fillna(0).ge(minimum_observations)
        & audit["adv_cny"].fillna(0).ge(minimum_adv_cny)
        & audit["aum_cny"].fillna(0).ge(minimum_aum_cny)
    )
    audit["cluster"] = audit["index_name"].map(classify_theme_cluster)
    audit = audit.sort_values(
        ["index_code", "eligible_liquidity", "adv_cny", "aum_cny"],
        ascending=[True, False, False, False],
    )
    selected = audit.loc[audit["eligible_liquidity"]].groupby(
        "index_code", observed=True,
    ).head(1).copy()
    mapping = selected.rename(columns={
        "index_code": "concept_code",
        "index_name": "concept_name",
        "ts_code": "etf_code",
        "csname": "etf_name",
    })
    mapping["match_type"] = "exact_official_tracking_index"
    mapping["selected"] = True
    mapping["selection_reason"] = (
        "highest recent ADV among exact-index ETFs passing ADV/AUM/history gates"
    )
    columns = [
        "concept_code", "concept_name", "cluster", "match_type", "etf_code", "etf_name",
        "adv_cny", "aum_cny", "selected", "selection_reason",
    ]
    return mapping[columns].reset_index(drop=True), audit.reset_index(drop=True)


def filter_mapping_by_weight_coverage(
    mapping: pd.DataFrame,
    index_weights: pd.DataFrame,
    *,
    minimum_weight_months: int,
    minimum_members: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = index_weights.copy()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"].astype(str))
    coverage = weights.groupby("index_code", observed=True).agg(
        weight_months=("trade_date", "nunique"),
        first_weight_date=("trade_date", "min"),
        last_weight_date=("trade_date", "max"),
        unique_members=("con_code", "nunique"),
    ).reset_index().rename(columns={"index_code": "concept_code"})
    audited = mapping.merge(coverage, on="concept_code", how="left")
    audited["weight_coverage_pass"] = (
        audited["weight_months"].fillna(0).ge(minimum_weight_months)
        & audited["unique_members"].fillna(0).ge(minimum_members)
    )
    return (
        audited.loc[audited["weight_coverage_pass"]].copy().reset_index(drop=True),
        audited.reset_index(drop=True),
    )


def expand_monthly_index_membership(
    index_weights: pd.DataFrame,
    mapping: pd.DataFrame,
    trade_dates: Iterable[pd.Timestamp],
    *,
    lag_sessions: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand monthly index weights to causal daily memberships.

    The snapshot used on a signal date must be at least ``lag_sessions`` trading
    sessions old. This prevents a month-end constituent snapshot from being used
    at the same close at which it first becomes observable.
    """
    if lag_sessions < 1:
        raise ValueError("lag_sessions must be at least one")
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(list(trade_dates)).unique()))
    if calendar.empty:
        raise ValueError("trade date calendar is empty")
    weights = index_weights.copy()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"].astype(str))
    weights = weights.loc[weights["index_code"].isin(mapping["concept_code"])].copy()
    weights = weights.drop_duplicates(["index_code", "trade_date", "con_code"], keep="last")
    date_frame = pd.DataFrame({"trade_date": calendar})
    member_parts: list[pd.DataFrame] = []
    index_parts: list[pd.DataFrame] = []
    metadata = mapping.drop_duplicates("concept_code").set_index("concept_code")
    for code, group in weights.groupby("index_code", sort=True):
        snapshots = pd.DataFrame({
            "snapshot_date": sorted(group["trade_date"].unique()),
        })
        lookup = date_frame.copy()
        lookup["available_cutoff"] = lookup["trade_date"].shift(lag_sessions)
        lookup = lookup.dropna(subset=["available_cutoff"])
        lookup = pd.merge_asof(
            lookup.sort_values("available_cutoff"),
            snapshots.sort_values("snapshot_date"),
            left_on="available_cutoff", right_on="snapshot_date", direction="backward",
        ).dropna(subset=["snapshot_date"])
        if lookup.empty:
            continue
        expanded = lookup[["trade_date", "snapshot_date"]].merge(
            group.rename(columns={"trade_date": "snapshot_date"}),
            on="snapshot_date", how="left", validate="many_to_many",
        )
        expanded = expanded.rename(columns={"index_code": "concept_code", "con_code": "ts_code"})
        member_parts.append(expanded[["trade_date", "concept_code", "ts_code"]])
        index_part = lookup[["trade_date"]].copy()
        index_part["concept_code"] = code
        index_part["concept_name"] = metadata.loc[code, "concept_name"]
        index_parts.append(index_part)
    if not member_parts:
        return pd.DataFrame(), pd.DataFrame()
    index = pd.concat(index_parts, ignore_index=True).drop_duplicates(
        ["trade_date", "concept_code"], keep="last",
    )
    members = pd.concat(member_parts, ignore_index=True).drop_duplicates(
        ["trade_date", "concept_code", "ts_code"], keep="last",
    )
    return (
        index.sort_values(["trade_date", "concept_code"]).reset_index(drop=True),
        members.sort_values(["trade_date", "concept_code", "ts_code"]).reset_index(drop=True),
    )


def build_monthly_index_history_eligibility(
    index_weights: pd.DataFrame,
    trade_dates: Iterable[pd.Timestamp],
    *,
    minimum_weight_months: int,
    minimum_members: int,
    availability_lag_sessions: int = 1,
) -> pd.DataFrame:
    """Evaluate index history gates separately at every prior month end."""
    if availability_lag_sessions < 1:
        raise ValueError("availability_lag_sessions must be at least one")
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(list(trade_dates)).unique()))
    if calendar.empty:
        return pd.DataFrame()
    monthly_last = pd.Series(calendar, index=calendar).groupby(calendar.to_period("M")).max()
    positions = {pd.Timestamp(date): position for position, date in enumerate(calendar)}
    weights = index_weights.copy()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"].astype(str))
    rows = []
    for selection_date in monthly_last:
        position = positions[pd.Timestamp(selection_date)]
        cutoff_position = position - availability_lag_sessions
        if cutoff_position < 0 or position + 1 >= len(calendar):
            continue
        cutoff = pd.Timestamp(calendar[cutoff_position])
        available = weights.loc[weights["trade_date"].le(cutoff)]
        if available.empty:
            continue
        history = available.groupby("index_code", observed=True)["trade_date"].nunique()
        latest_dates = available.groupby("index_code", observed=True)["trade_date"].max()
        latest = available.merge(
            latest_dates.rename("latest_snapshot_date"),
            left_on="index_code", right_index=True, how="inner",
        )
        latest = latest.loc[latest["trade_date"].eq(latest["latest_snapshot_date"])]
        member_count = latest.groupby("index_code", observed=True)["con_code"].nunique()
        for index_code in history.index:
            months = int(history.loc[index_code])
            members = int(member_count.get(index_code, 0))
            rows.append({
                "selection_date": pd.Timestamp(selection_date),
                "effective_start": pd.Timestamp(calendar[position + 1]),
                "concept_code": index_code,
                "available_cutoff": cutoff,
                "available_weight_months": months,
                "latest_snapshot_date": pd.Timestamp(latest_dates.loc[index_code]),
                "latest_member_count": members,
                "index_history_pass": bool(
                    months >= minimum_weight_months and members >= minimum_members
                ),
            })
    return pd.DataFrame(rows).sort_values(
        ["selection_date", "concept_code"],
    ).reset_index(drop=True)


def build_monthly_pit_etf_mapping(
    etfs: pd.DataFrame,
    candidates: pd.DataFrame,
    concept_features: pd.DataFrame,
    trade_dates: Iterable[pd.Timestamp],
    *,
    minimum_listing_sessions: int = 60,
    liquidity_window: int = 20,
    minimum_liquidity_observations: int = 18,
    minimum_adv_cny: float = 50_000_000,
    minimum_aum_cny: float = 1_000_000_000,
    correlation_window: int = 120,
    minimum_correlation_observations: int = 60,
    minimum_mapping_correlation: float = 0.80,
    index_history_eligibility: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the most liquid exact-index ETF using only prior month-end data."""
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(list(trade_dates)).unique()))
    if calendar.empty:
        raise ValueError("trade date calendar is empty")
    monthly_last = pd.Series(calendar, index=calendar).groupby(calendar.to_period("M")).max()
    calendar_positions = {date: position for position, date in enumerate(calendar)}
    selection_pairs = []
    for selection_date in monthly_last:
        position = calendar_positions[pd.Timestamp(selection_date)]
        if position + 1 >= len(calendar):
            continue
        effective_start = pd.Timestamp(calendar[position + 1])
        selection_pairs.append((pd.Timestamp(selection_date), effective_start))
    if not selection_pairs:
        return pd.DataFrame(), pd.DataFrame()

    candidate_columns = [
        "ts_code", "index_code", "index_name", "csname", "list_date_parsed", "cluster",
    ]
    metadata = candidates[candidate_columns].drop_duplicates("ts_code")
    frame = etfs.merge(metadata, on="ts_code", how="inner", validate="many_to_one")
    frame = frame.sort_values(["ts_code", "trade_date"]).copy()
    grouped = frame.groupby("ts_code", sort=False)
    frame["listing_sessions"] = grouped.cumcount() + 1
    prehistory_listing = frame["list_date_parsed"].lt(calendar.min())
    frame.loc[prehistory_listing, "listing_sessions"] += minimum_listing_sessions
    frame["adv_cny_pit"] = grouped["amount_cny"].transform(
        lambda values: values.rolling(
            liquidity_window, min_periods=minimum_liquidity_observations,
        ).mean()
    )
    frame["liquidity_observations"] = grouped["amount_cny"].transform(
        lambda values: values.rolling(liquidity_window, min_periods=1).count()
    )
    returns = concept_features[[
        "trade_date", "concept_code", "concept_return_1d",
    ]].rename(columns={"concept_code": "index_code"})
    frame = frame.merge(
        returns, on=["trade_date", "index_code"], how="left", validate="many_to_one",
    ).sort_values(["ts_code", "trade_date"])
    frame["mapping_correlation_pit"] = frame.groupby(
        "ts_code", sort=False, group_keys=False,
    ).apply(
        lambda group: group["etf_return_1d"].rolling(
            correlation_window, min_periods=minimum_correlation_observations,
        ).corr(group["concept_return_1d"]),
        include_groups=False,
    ).reset_index(level=0, drop=True).sort_index()
    paired = frame["etf_return_1d"].notna() & frame["concept_return_1d"].notna()
    frame["correlation_observations"] = paired.groupby(frame["ts_code"], sort=False).transform(
        lambda values: values.rolling(correlation_window, min_periods=1).sum()
    )

    audit_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    for selection_date, effective_start in selection_pairs:
        month = frame.loc[frame["trade_date"].eq(selection_date)].copy()
        if month.empty:
            continue
        month["selection_date"] = selection_date
        month["effective_start"] = effective_start
        month["effective_month"] = str(effective_start.to_period("M"))
        month["listing_pass"] = month["listing_sessions"].ge(minimum_listing_sessions)
        month["liquidity_pass"] = (
            month["liquidity_observations"].ge(minimum_liquidity_observations)
            & month["adv_cny_pit"].ge(minimum_adv_cny)
        )
        month["aum_pass"] = month["aum_cny"].ge(minimum_aum_cny)
        month["correlation_pass"] = (
            month["correlation_observations"].ge(minimum_correlation_observations)
            & month["mapping_correlation_pit"].ge(minimum_mapping_correlation)
        )
        if index_history_eligibility is not None:
            history = index_history_eligibility.loc[
                index_history_eligibility["selection_date"].eq(selection_date),
                [
                    "concept_code", "available_cutoff", "available_weight_months",
                    "latest_snapshot_date", "latest_member_count", "index_history_pass",
                ],
            ].rename(columns={"concept_code": "index_code"})
            month = month.merge(history, on="index_code", how="left", validate="many_to_one")
            month["index_history_pass"] = month["index_history_pass"].fillna(False)
        else:
            month["index_history_pass"] = True
        month["eligible_pit"] = month[[
            "listing_pass", "liquidity_pass", "aum_pass", "correlation_pass",
            "index_history_pass",
        ]].all(axis=1)
        audit_parts.append(month)
        chosen = month.loc[month["eligible_pit"]].sort_values(
            ["index_code", "adv_cny_pit", "aum_cny"],
            ascending=[True, False, False],
        ).groupby("index_code", observed=True).head(1).copy()
        if not chosen.empty:
            selected_parts.append(chosen)
    audit = pd.concat(audit_parts, ignore_index=True) if audit_parts else pd.DataFrame()
    if not selected_parts:
        return pd.DataFrame(), audit
    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.rename(columns={
        "index_code": "concept_code", "index_name": "concept_name",
        "ts_code": "etf_code", "csname": "etf_name",
    })
    selected["match_type"] = "exact_official_tracking_index_pit"
    selected["selected"] = True
    columns = [
        "selection_date", "effective_start", "effective_month", "concept_code",
        "concept_name", "cluster", "match_type", "etf_code", "etf_name", "adv_cny_pit",
        "aum_cny", "mapping_correlation_pit", "correlation_observations", "selected",
    ]
    for column in (
        "available_cutoff", "available_weight_months", "latest_snapshot_date",
        "latest_member_count", "index_history_pass",
    ):
        if column in selected:
            columns.append(column)
    return selected[columns].sort_values(
        ["effective_start", "concept_code"],
    ).reset_index(drop=True), audit


def build_dynamic_etf_signal_panel(
    concept_features: pd.DataFrame,
    etfs: pd.DataFrame,
    mapping_schedule: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the prior-month PIT mapping and compute executable ETF scores."""
    frame = etfs.copy()
    frame["effective_month"] = frame["trade_date"].dt.to_period("M").astype(str)
    metadata = candidates[[
        "ts_code", "index_code", "index_name", "csname", "cluster",
    ]].drop_duplicates("ts_code").rename(columns={
        "index_code": "concept_code", "index_name": "concept_name", "csname": "etf_name",
    })
    panel = frame.merge(metadata, on="ts_code", how="inner", validate="many_to_one")
    selected = mapping_schedule.rename(columns={"etf_code": "ts_code"})[[
        "effective_month", "concept_code", "ts_code", "selection_date", "effective_start",
        "match_type", "mapping_correlation_pit", "correlation_observations",
    ]].copy()
    selected["mapping_pass"] = True
    panel = panel.merge(
        selected, on=["effective_month", "concept_code", "ts_code"], how="left",
        validate="many_to_one",
    )
    wanted = [
        "trade_date", "concept_code", "concept_return_1d", "eligible_concept",
        "breadth_float", "common_delta_rank", "rs_momentum_5d", "rrg_quadrant",
        "signal_rrg_only", "common_breadth_delta_smooth5",
    ]
    panel = panel.merge(
        concept_features[wanted], on=["trade_date", "concept_code"],
        how="left", validate="many_to_one",
    )
    panel["mapping_pass"] = panel["mapping_pass"].astype("boolean").fillna(False).astype(bool)
    panel["match_type"] = panel["match_type"].fillna("exact_official_tracking_index_pit")
    panel["mapping_correlation"] = panel["mapping_correlation_pit"]
    panel["score_etf_momentum"] = np.nan
    selected_rows = panel["mapping_pass"]
    panel.loc[selected_rows, "score_etf_momentum"] = (
        0.6 * panel.loc[selected_rows].groupby("trade_date")["etf_momentum_20d"].transform(_zscore)
        + 0.4 * panel.loc[selected_rows].groupby("trade_date")["etf_momentum_60d"].transform(_zscore)
    )
    return panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def next_open_trade_date(calendar: Iterable[pd.Timestamp], after: str | pd.Timestamp) -> pd.Timestamp:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(calendar)).unique()))
    future = dates[dates > pd.Timestamp(after)]
    if future.empty:
        raise ValueError(f"calendar has no trading date after {after}")
    return pd.Timestamp(future.min())


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = numeric.std(ddof=0)
    if not pd.notna(std) or std <= 0:
        return pd.Series(0.0, index=values.index)
    return (numeric - numeric.mean()) / std
