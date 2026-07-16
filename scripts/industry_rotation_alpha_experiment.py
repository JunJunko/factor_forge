from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository
from factor_forge.data.tushare_provider import TushareProvider
from factor_forge.research.industry_rotation_alpha import (
    PRIMARY_HORIZONS,
    attach_pit_membership, attach_stitched_pit_membership,
    audit_rotation_panel,
    build_rotation_dataset,
    conditional_payoff_matrix,
    evaluate_rotation_signals,
    fama_macbeth_breadth_increment,
)


BASE_COLUMNS = [
    "trade_date", "ts_code", "raw_open", "adj_open", "adj_close", "amount_cny",
    "circ_mv_cny", "is_suspended", "is_limit_up_open", "is_limit_down_open",
    "is_st", "is_delisting_period", "listing_trade_days", "is_tradeable", "is_liquid",
]

SPLITS = {
    "exploratory_retro_taxonomy": ("2016-01-01", "2020-12-31"),
    "valid_taxonomy_validation": ("2021-11-01", "2023-12-31"),
    "final_holdout": ("2024-01-01", "2026-06-30"),
}


def main() -> None:
    args = parse_args()
    project = load_project(args.project)
    snapshot = refresh_membership(Path(args.snapshot_root), "SW2021") if args.refresh_membership else latest_snapshot(Path(args.snapshot_root), "SW2021")
    legacy_snapshot = None
    if args.stitch_classifications:
        legacy_snapshot = refresh_membership(Path(args.snapshot_root), "SW2014") if args.refresh_legacy_membership else latest_snapshot(Path(args.snapshot_root), "SW2014")
    increment_path = Path(args.increment_staging)
    if args.refresh_increment:
        increment_path = refresh_daily_increment(
            Path(args.increment_root), args.increment_start, args.increment_end
        )
    if args.refresh_only:
        print(json.dumps({"snapshot": str(snapshot), "legacy_snapshot": str(legacy_snapshot) if legacy_snapshot else None, "increment": str(increment_path)}, ensure_ascii=False))
        return
    membership = pd.read_parquet(snapshot / "sw_l2_membership.parquet")
    repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version = repository.resolve(args.data_version)
    panel_path = project.paths.data_root / "versions" / version / "curated" / "stock_daily_panel.parquet"
    panel = pd.read_parquet(panel_path, columns=BASE_COLUMNS)
    increment = load_staged_increment(increment_path, panel) if str(increment_path) else pd.DataFrame()
    if not increment.empty:
        panel = pd.concat([panel, increment], ignore_index=True)
        panel = panel.sort_values(["trade_date", "ts_code"]).drop_duplicates(
            ["trade_date", "ts_code"], keep="last"
        ).reset_index(drop=True)
    if legacy_snapshot is not None:
        legacy_membership = pd.read_parquet(legacy_snapshot / "sw_l2_membership.parquet")
        panel = attach_stitched_pit_membership(panel, legacy_membership, membership)
    else:
        panel = attach_pit_membership(panel, membership)
    audit = audit_rotation_panel(panel, membership)
    failures = data_gate_failures(audit.to_dict(), stitched=legacy_snapshot is not None)
    if failures:
        raise RuntimeError("rotation data gate failed: " + "; ".join(failures))

    run_id = datetime.now(timezone.utc).strftime("industry_rotation_%Y%m%dT%H%M%SZ")
    output = Path(args.output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "data_audit.json").write_text(
        json.dumps({"data_version": version, "membership_snapshot": str(snapshot), "legacy_membership_snapshot": str(legacy_snapshot) if legacy_snapshot else None, **audit.to_dict()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stocks, groups = build_rotation_dataset(panel, horizons=PRIMARY_HORIZONS, minimum_members=8)
    groups.to_parquet(output / "industry_daily_features.parquet", index=False)
    evaluation, daily = evaluate_rotation_signals(groups, splits=SPLITS)
    evaluation.to_csv(output / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output / "daily_signal_payoffs.csv", index=False, encoding="utf-8-sig")
    conditional = conditional_payoff_matrix(groups, horizon=5)
    conditional.to_csv(output / "conditional_payoff_matrix_5d.csv", index=False, encoding="utf-8-sig")
    fama_rows = []
    for split, (start, end) in SPLITS.items():
        period = groups.loc[groups["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        fama = fama_macbeth_breadth_increment(period, horizon=5)
        fama.insert(0, "split", split)
        fama_rows.append(fama)
    fama = pd.concat(fama_rows, ignore_index=True)
    fama.to_csv(output / "fama_macbeth_5d.csv", index=False, encoding="utf-8-sig")
    report = render_report(version, snapshot, audit.to_dict(), evaluation, conditional, fama)
    (output / "research_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"run_dir": str(output), "data_version": version, "audit": audit.to_dict()}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PIT SW L2 breadth + RRG alpha experiment")
    parser.add_argument("--project", type=Path, default=Path("configs/project_sw_l2.yaml"))
    parser.add_argument("--data-version", default="latest")
    parser.add_argument("--snapshot-root", default="data/industry_rotation/snapshots")
    parser.add_argument("--refresh-membership", action="store_true")
    parser.add_argument("--refresh-legacy-membership", action="store_true")
    parser.add_argument("--stitch-classifications", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--refresh-only", action="store_true")
    parser.add_argument("--increment-staging", default="data/staging/tushare_20260713_20260714")
    parser.add_argument("--refresh-increment", action="store_true")
    parser.add_argument("--increment-root", default="data/industry_rotation/increments")
    parser.add_argument("--increment-start", default="20260713")
    parser.add_argument("--increment-end", default="20260714")
    parser.add_argument("--output-root", default="artifacts/industry_rotation_alpha")
    return parser.parse_args()


def refresh_membership(root: Path, standard: str) -> Path:
    provider = TushareProvider()
    classification = provider.query("index_classify", level="L2", src=standard)
    frames = []
    codes = classification["index_code"].dropna().astype(str).unique().tolist()
    for position, code in enumerate(codes, start=1):
        for state in ("Y", "N"):
            frame = provider.query("index_member_all", l2_code=code, is_new=state)
            if not frame.empty:
                frames.append(frame)
        if position == 1 or position % 10 == 0 or position == len(codes):
            print(f"membership {position}/{len(codes)} l2_code={code}", flush=True)
    membership = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["ts_code", "l2_code", "in_date", "out_date"], keep="last"
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / f"sw_l2_{standard.lower()}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    classification.to_parquet(output / "sw_l2_classification.parquet", index=False)
    membership.to_parquet(output / "sw_l2_membership.parquet", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "tushare",
        "standard": standard,
        "level": "L2",
        "classification_rows": int(len(classification)),
        "membership_rows": int(len(membership)),
        "stocks": int(membership["ts_code"].nunique()),
        "current_rows": int(membership["out_date"].isna().sum()),
        "historical_rows": int(membership["out_date"].notna().sum()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def latest_snapshot(root: Path, standard: str) -> Path:
    snapshots = []
    for path in root.glob("sw_l2_*"):
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("standard") == standard:
            snapshots.append(path)
    snapshots.sort()
    if not snapshots:
        raise FileNotFoundError(f"no {standard} L2 snapshot; request a refresh")
    return snapshots[-1]


def refresh_daily_increment(root: Path, start: str, end: str) -> Path:
    provider = TushareProvider()
    calendar = provider.query("trade_cal", exchange="SSE", start_date=start, end_date=end)
    dates = calendar.loc[pd.to_numeric(calendar["is_open"], errors="coerce").eq(1), "cal_date"].astype(str).tolist()
    output = root / f"tushare_{start}_{end}"
    output.mkdir(parents=True, exist_ok=True)
    endpoints = {
        "daily": "daily", "adj_factor": "adj_factor", "daily_basic": "daily_basic",
        "stk_limit": "stk_limit", "suspend": "suspend_d", "st_status": "stock_st",
    }
    for date in dates:
        for name, endpoint in endpoints.items():
            frame = provider.query(endpoint, trade_date=date)
            if "trade_date" not in frame:
                frame["trade_date"] = date
            directory = output / name
            directory.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(directory / f"trade_date={date}.parquet", index=False)
        print(f"daily increment fetched trade_date={date}", flush=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "source": "tushare",
        "start": start, "end": end, "open_dates": dates, "endpoints": list(endpoints),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def load_staged_increment(staging: Path, base: pd.DataFrame) -> pd.DataFrame:
    if not staging.exists():
        return pd.DataFrame(columns=BASE_COLUMNS)
    def read_parts(name: str) -> pd.DataFrame:
        paths = sorted((staging / name).glob("trade_date=*.parquet"))
        frames = [pd.read_parquet(path) for path in paths]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    daily = read_parts("daily")
    if daily.empty:
        return pd.DataFrame(columns=BASE_COLUMNS)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"].astype(str))
    daily = daily.rename(columns={"open": "raw_open", "close": "raw_close", "amount": "source_amount"})
    adj = read_parts("adj_factor")
    basic = read_parts("daily_basic")
    limits = read_parts("stk_limit")
    status = read_parts("st_status")
    for frame in (adj, basic, limits, status):
        if not frame.empty:
            frame["trade_date"] = pd.to_datetime(frame["trade_date"].astype(str))
    result = daily.merge(adj[["trade_date", "ts_code", "adj_factor"]], on=["trade_date", "ts_code"], how="left")
    result = result.merge(basic[["trade_date", "ts_code", "circ_mv"]], on=["trade_date", "ts_code"], how="left")
    if not limits.empty:
        result = result.merge(limits[["trade_date", "ts_code", "up_limit", "down_limit"]], on=["trade_date", "ts_code"], how="left")
    else:
        result["up_limit"] = np.nan
        result["down_limit"] = np.nan
    result["amount_cny"] = pd.to_numeric(result["source_amount"], errors="coerce") * 1_000
    result["circ_mv_cny"] = pd.to_numeric(result["circ_mv"], errors="coerce") * 10_000
    result["adj_open"] = result["raw_open"] * result["adj_factor"]
    result["adj_close"] = result["raw_close"] * result["adj_factor"]
    result["is_suspended"] = result["raw_open"].isna()
    result["is_limit_up_open"] = result["raw_open"].notna() & result["up_limit"].notna() & result["raw_open"].ge(result["up_limit"] - 0.001)
    result["is_limit_down_open"] = result["raw_open"].notna() & result["down_limit"].notna() & result["raw_open"].le(result["down_limit"] + 0.001)
    st_keys = set(zip(status.get("trade_date", []), status.get("ts_code", [])))
    result["is_st"] = [key in st_keys for key in zip(result["trade_date"], result["ts_code"])]
    result["is_delisting_period"] = False
    last_listing = base.groupby("ts_code")["listing_trade_days"].max()
    result = result.sort_values(["ts_code", "trade_date"])
    result["listing_trade_days"] = result.groupby("ts_code").cumcount() + 1 + result["ts_code"].map(last_listing).fillna(0)
    result["is_tradeable"] = (
        result["listing_trade_days"].ge(60) & ~result["is_suspended"] & ~result["is_st"]
        & result["adj_close"].notna()
    )
    tail = base.sort_values(["ts_code", "trade_date"]).groupby("ts_code", sort=False).tail(19)[
        ["trade_date", "ts_code", "amount_cny", "is_tradeable"]
    ]
    liquidity = pd.concat([
        tail,
        result[["trade_date", "ts_code", "amount_cny", "is_tradeable"]],
    ]).sort_values(["ts_code", "trade_date"])
    liquidity["amount_ma20"] = liquidity.groupby("ts_code")["amount_cny"].transform(
        lambda values: values.rolling(20, min_periods=18).mean()
    )
    latest_liquidity = liquidity.merge(
        result[["trade_date", "ts_code"]], on=["trade_date", "ts_code"], how="inner"
    )
    latest_liquidity["rank"] = latest_liquidity.where(latest_liquidity["is_tradeable"])["amount_ma20"].groupby(latest_liquidity["trade_date"]).rank(method="first", ascending=False)
    liquid_keys = latest_liquidity.set_index(["trade_date", "ts_code"])["rank"].le(1000)
    keys = pd.MultiIndex.from_frame(result[["trade_date", "ts_code"]])
    result["is_liquid"] = liquid_keys.reindex(keys, fill_value=False).to_numpy()
    return result.reindex(columns=BASE_COLUMNS).sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def data_gate_failures(audit: dict, *, stitched: bool) -> list[str]:
    failures = []
    if audit["duplicate_keys"]:
        failures.append("duplicate stock-day keys")
    if audit["overlapping_intervals"]:
        failures.append("overlapping membership intervals")
    if audit["current_membership_violations"]:
        failures.append("multiple current memberships")
    if audit["tradeable_membership_coverage"] < 0.95:
        failures.append("tradeable membership coverage below 95%")
    lower = 90 if stitched else 120
    if not (lower <= audit["median_daily_groups"] <= 140):
        failures.append(f"daily L2 group count outside {lower}-140")
    if audit["min_daily_groups"] < lower:
        failures.append(f"minimum daily L2 group count below {lower}")
    return failures


def render_report(version: str, snapshot: Path, audit: dict, evaluation: pd.DataFrame,
                  conditional: pd.DataFrame, fama: pd.DataFrame) -> str:
    primary = evaluation.loc[
        evaluation["signal"].isin(["breadth_rrg", "momentum_20d", "breadth_delta", "rrg_only"])
        & evaluation["horizon"].eq(5)
    ].copy()
    final = primary.loc[primary["split"].eq("final_holdout")]
    fm = fama.loc[(fama["split"].eq("final_holdout")) & (fama["term"].eq("breadth_delta_5d"))]
    return f"""# 申万二级行业扩散 + RRG Alpha 实验

## 数据 Gate

- 数据版本：`{version}`
- 行业快照：`{snapshot}`
- 时间范围：{audit['start_date']} 至 {audit['end_date']}
- 股票日行数：{audit['rows']:,}
- 行业数量：{audit['group_count']}
- 行业归属覆盖率：{audit['membership_coverage']:.2%}
- 可交易股票行业覆盖率：{audit['tradeable_membership_coverage']:.2%}
- 每日行业数量（最小/中位/最大）：{audit['min_daily_groups']} / {audit['median_daily_groups']:.0f} / {audit['max_daily_groups']}
- 重复键/区间重叠/多重当前归属：{audit['duplicate_keys']} / {audit['overlapping_intervals']} / {audit['current_membership_violations']}

## 预注册主周期：5日

```text
{final.to_string(index=False)}
```

## 最终盲测：控制相对动量后的扩散增量

```text
{fm.to_string(index=False)}
```

## 解释

`breadth_rrg` 只有在20日平滑自由流通市值扩散度大于50%、5日扩散变化位于当日前30%，且相对动量继续改善时才产生信号。标签使用信号日成分冻结，并从T+1开盘计算到T+h+1开盘。
完整消融、条件收益矩阵与逐日序列见同目录 CSV/Parquet 文件。
"""


if __name__ == "__main__":
    main()
