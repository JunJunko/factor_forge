from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "web_app" / "static"
TASKS: dict[str, dict[str, Any]] = {}
TASK_LOCK = threading.Lock()
ATR_RUN_DIR = (
    ROOT
    / "artifacts"
    / "atr_reversion_runs"
    / "atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z"
)
FROZEN_SENSITIVITY_DIR = ATR_RUN_DIR / "event_badtrade_iteration_20260707T111609Z"
FROZEN_AUDIT_DIR = FROZEN_SENSITIVITY_DIR
LIVE_OPS_ROOT = ROOT / "artifacts" / "atr_reversion_live_ops"
POSITION_STATE_PATH = LIVE_OPS_ROOT / "position_state.json"
SHADOW_PORTFOLIO_PATH = LIVE_OPS_ROOT / "shadow_portfolio.json"


app = FastAPI(title="Factor Forge Control Panel")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SyncRequest(BaseModel):
    start: str = Field(pattern=r"^\d{8}$")
    end: str = Field(pattern=r"^\d{8}$")
    merge_full_history: bool = True


class SignalRequest(BaseModel):
    signal_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class SellAdviceRequest(BaseModel):
    signal_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    holdings_text: str = ""


class DailyChainRequest(BaseModel):
    signal_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    update_data: bool = True
    force_regenerate_signals: bool = False
    start: str | None = Field(default=None, pattern=r"^\d{8}$")
    end: str | None = Field(default=None, pattern=r"^\d{8}$")
    merge_full_history: bool = True
    holdings_text: str = ""


class ShadowAddRequest(BaseModel):
    signal_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    ts_codes: list[str]


class ShadowPositionActionRequest(BaseModel):
    position_id: str
    action_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    note: str = ""


def _task(status: str = "queued") -> tuple[str, dict[str, Any]]:
    task_id = uuid.uuid4().hex[:12]
    item = {
        "id": task_id,
        "status": status,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": None,
        "finished_at": None,
        "logs": [],
        "result": None,
        "error": None,
    }
    with TASK_LOCK:
        TASKS[task_id] = item
    return task_id, item


def _append_log(task_id: str, line: str) -> None:
    with TASK_LOCK:
        task = TASKS[task_id]
        task["logs"].append(line.rstrip())
        task["logs"] = task["logs"][-800:]


def _set_task(task_id: str, **updates: Any) -> None:
    with TASK_LOCK:
        TASKS[task_id].update(updates)


def _run_subprocess(task_id: str, args: list[str]) -> int:
    _append_log(task_id, "> " + " ".join(args))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        args,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _append_log(task_id, line)
    return proc.wait()


def _load_repo():
    from factor_forge.config import load_project
    from factor_forge.data.repository import DataVersionRepository

    project = load_project(ROOT / "configs" / "project.yaml")
    return DataVersionRepository(project.paths.data_root, project.paths.metadata_db)


def _is_complete_manifest(manifest: dict[str, Any]) -> bool:
    return int(manifest.get("row_count", 0)) > 1_000_000 and str(manifest.get("start_date", "")) <= "2017-01-01"


def _find_previous_complete_version(exclude: str | None = None) -> str | None:
    repo = _load_repo()
    with repo.metadata.connect() as conn:
        rows = conn.execute(
            "SELECT data_version FROM meta_data_version WHERE quality_status='PASSED' ORDER BY created_at DESC"
        ).fetchall()
    for row in rows:
        version = row["data_version"]
        if version == exclude:
            continue
        try:
            _, manifest = repo.load_manifest(version)
        except FileNotFoundError:
            continue
        if _is_complete_manifest(manifest):
            return version
    return None


def _append_increment_to_complete(task_id: str, increment_version: str) -> str:
    repo = _load_repo()
    _, inc_manifest = repo.load_manifest(increment_version)
    if _is_complete_manifest(inc_manifest):
        _append_log(task_id, f"latest version is already complete: {increment_version}")
        return increment_version
    base_version = _find_previous_complete_version(exclude=increment_version)
    if not base_version:
        raise RuntimeError("No previous complete data version found to append increment into.")
    _append_log(task_id, f"merge increment {increment_version} into base {base_version}")
    _, base = repo.load_panel(base_version)
    _, inc = repo.load_panel(increment_version)
    panel = pd.concat([base, inc], ignore_index=True)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = (
        panel.sort_values(["trade_date", "ts_code"])
        .drop_duplicates(["trade_date", "ts_code"], keep="last")
        .sort_values(["ts_code", "trade_date"])
        .reset_index(drop=True)
    )
    g = panel.groupby("ts_code", sort=False)
    panel["listing_trade_days"] = g.cumcount() + 1
    panel["is_factor_eligible"] = (
        panel["is_tradeable"].fillna(False).astype(bool)
        & ~panel["is_st"].fillna(False).astype(bool)
        & ~panel["is_delisting_period"].fillna(False).astype(bool)
        & ~panel["is_suspended"].fillna(False).astype(bool)
        & panel["listing_trade_days"].ge(60)
    )
    panel["is_tradeable"] = (
        panel["raw_open"].notna()
        & panel["adj_open"].notna()
        & ~panel["is_suspended"].fillna(True).astype(bool)
    )
    rolling_amount = (
        panel["amount_cny"].where(panel["amount_cny"] > 0)
        .groupby(panel["ts_code"], sort=False)
        .rolling(20, min_periods=18)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(panel.index)
    )
    rank = rolling_amount.where(panel["is_tradeable"]).groupby(panel["trade_date"], sort=False).rank(
        method="first", ascending=False
    )
    panel["is_liquid"] = rank.le(1000).fillna(False)
    published = repo.publish(panel, raw_datasets=None, source="tushare_append")
    _append_log(task_id, f"published complete version {published}")
    return published


def _latest_data_status() -> dict[str, Any]:
    repo = _load_repo()
    version, manifest = repo.load_manifest("latest")
    return {
        "version": version,
        "start_date": manifest.get("start_date"),
        "end_date": manifest.get("end_date"),
        "row_count": manifest.get("row_count"),
        "source": manifest.get("source"),
        "complete": _is_complete_manifest(manifest),
    }


def _latest_panel_signal_day_status(signal_date: pd.Timestamp) -> dict[str, Any]:
    repo = _load_repo()
    version, panel = repo.load_panel("latest")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    target = signal_date.normalize()
    day = panel.loc[panel["trade_date"].eq(target)]
    tradeable_rows = 0
    if not day.empty and "is_tradeable" in day:
        tradeable_rows = int(day["is_tradeable"].fillna(False).astype(bool).sum())
    return {
        "version": version,
        "panel_end": panel["trade_date"].max().normalize() if len(panel) else pd.NaT,
        "has_signal_date": not day.empty,
        "rows": int(len(day)),
        "tradeable_rows": tradeable_rows,
    }


def _ensure_signal_date_data(task_id: str, signal_date: str, merge_full_history: bool = True) -> dict[str, Any]:
    signal_ts = pd.Timestamp(signal_date).normalize()
    status = _latest_panel_signal_day_status(signal_ts)
    if status["has_signal_date"] and status["tradeable_rows"] > 0:
        _append_log(
            task_id,
            "[update_data] signal-date data ready "
            f"version={status['version']} panel_end={status['panel_end'].date()} "
            f"rows={status['rows']} tradeable_rows={status['tradeable_rows']}",
        )
        return _latest_data_status()

    compact = signal_ts.strftime("%Y%m%d")
    reason = (
        "missing signal date"
        if not status["has_signal_date"]
        else f"signal date has no tradeable rows rows={status['rows']}"
    )
    _append_log(task_id, f"[update_data] auto-fill required for {signal_date}: {reason}")
    data_status = _sync_data_inline(task_id, compact, compact, merge_full_history)
    status = _latest_panel_signal_day_status(signal_ts)
    if not status["has_signal_date"]:
        raise RuntimeError(f"signal_date {signal_date} is still missing from latest panel after auto-fill")
    if status["tradeable_rows"] <= 0:
        raise RuntimeError(
            f"signal_date {signal_date} has no tradeable rows after auto-fill "
            f"(rows={status['rows']}, version={status['version']})"
        )
    _append_log(
        task_id,
        "[update_data] auto-fill verified "
        f"version={status['version']} rows={status['rows']} tradeable_rows={status['tradeable_rows']}",
    )
    return data_status


def _latest_signal_dir() -> Path | None:
    root = ROOT / "artifacts" / "atr_reversion_live_signals"
    if not root.exists():
        return None
    dirs = [
        p
        for p in root.iterdir()
        if p.is_dir() and (p / "signal_summary.json").exists() and (p / "top_recommendations.csv").exists()
    ]
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None


def _read_signal(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "signal_summary.json"
    top_path = run_dir / "top_recommendations.csv"
    if not summary_path.exists() or not top_path.exists():
        raise FileNotFoundError(f"Missing signal outputs in {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    top = pd.read_csv(top_path).replace({np.nan: None})
    return {
        "run_dir": str(run_dir.relative_to(ROOT)),
        "summary": summary,
        "top": top.to_dict("records"),
        "files": {
            "summary": str(summary_path.relative_to(ROOT)),
            "top_recommendations": str(top_path.relative_to(ROOT)),
            "top100_candidates": str((run_dir / "top100_candidates.csv").relative_to(ROOT)),
            "run_log": str((run_dir / "run.log").relative_to(ROOT)),
        },
    }


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _file_link(path: Path, label: str) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return {
        "label": label,
        "path": _rel(path),
        "size": path.stat().st_size,
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
    }


def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _frozen_sensitivity_row() -> dict[str, Any] | None:
    path = FROZEN_SENSITIVITY_DIR / "event_iteration_summary.csv"
    if not path.exists():
        return None
    try:
        data = pd.read_csv(path)
    except Exception:
        return None
    row = data.loc[data["variant"].eq("cluster_alpha_payoff_gate_top5")]
    if row.empty:
        return None
    return row.iloc[0].replace({np.nan: None}).to_dict()


def _frozen_latest_year() -> dict[str, Any] | None:
    path = FROZEN_AUDIT_DIR / "event_iteration_yearly.csv"
    if not path.exists():
        return None
    try:
        data = pd.read_csv(path)
    except Exception:
        return None
    if "variant" in data.columns:
        data = data[data["variant"].eq("cluster_alpha_payoff_gate_top5")]
    if data.empty:
        return None
    return data.sort_values("year").iloc[-1].replace({np.nan: None}).to_dict()


def _build_dashboard(data_status: dict[str, Any], signal: dict[str, Any] | None) -> dict[str, Any]:
    summary = (signal or {}).get("summary") or {}
    top = (signal or {}).get("top") or []
    fit_quality = summary.get("fit_quality") or {}
    risk_inputs = summary.get("risk_gate_inputs") or {}
    final_exposure = float(summary.get("final_exposure", 0.0) or 0.0) if summary else None
    signal_date = str(summary.get("signal_date", ""))[:10] if summary.get("signal_date") else ""
    data_end = str(data_status.get("end_date") or "")
    target_positions = [row for row in top if float(row.get("target_weight") or 0.0) > 0.0]
    signal_day_blocks = [
        row
        for row in top
        if bool(row.get("is_suspended")) or bool(row.get("is_limit_up_open")) or bool(row.get("is_st"))
    ]
    warnings: list[str] = []
    if signal and signal_date and data_end and signal_date != data_end:
        warnings.append(f"最新信号日 {signal_date} 与数据截止日 {data_end} 不一致，请确认是否要重新生成。")
    if final_exposure is not None and final_exposure <= 0.0:
        warnings.append("最终仓位为 0%，当前建议只观察不建仓。")
    if bool(fit_quality.get("flipped")):
        warnings.append("fit-quality 冻结规则正在反向排序，说明近期原始因子方向偏弱。")
    if signal_day_blocks:
        warnings.append("候选股里存在信号日停牌、ST或涨停开盘标记，次日执行前要复核。")

    trade_audit = _read_json_file(FROZEN_SENSITIVITY_DIR / "trade_audit_summary.json") or {}
    frozen_year = _frozen_latest_year()
    sensitivity = _frozen_sensitivity_row()
    files = [
        _file_link(FROZEN_SENSITIVITY_DIR / "report.md", "事件版冻结策略报告"),
        _file_link(FROZEN_SENSITIVITY_DIR / "oos_2025_2026_focus" / "report.md", "2025/2026 OOS重点报告"),
        _file_link(FROZEN_SENSITIVITY_DIR / "event_iteration_summary.csv", "事件版策略矩阵汇总"),
        _file_link(FROZEN_SENSITIVITY_DIR / "trade_execution_audit.csv", "交易执行审计"),
    ]
    display_status = "未生成" if not summary else ("观察" if final_exposure is not None and final_exposure <= 0.0 else "可执行")
    return {
        "asof": datetime.now().isoformat(timespec="seconds"),
        "status": display_status,
        "warnings": warnings,
        "workflow": [
            {"name": "数据", "state": "完成" if data_status.get("complete") else "需检查", "detail": f"截止 {data_end or '-'}"},
            {"name": "信号", "state": "完成" if signal else "未生成", "detail": signal_date or "-"},
            {
                "name": "仓位",
                "state": "观察" if final_exposure is not None and final_exposure <= 0.0 else "待执行",
                "detail": "" if final_exposure is None else f"{final_exposure:.0%}",
            },
            {
                "name": "审计",
                "state": "通过" if int(trade_audit.get("blocking_trade_issues", 0) or 0) == 0 else "红灯",
                "detail": f"blocking={trade_audit.get('blocking_trade_issues', '-')}",
            },
        ],
        "execution": {
            "signal_date": signal_date,
            "intended_execution": summary.get("intended_execution"),
            "final_exposure": final_exposure,
            "target_position_count": len(target_positions),
            "candidate_count": len(top),
            "predictable_candidates": summary.get("predictable_candidates"),
            "next_day_note": summary.get("next_day_fillability_note"),
            "signal_day_block_count": len(signal_day_blocks),
        },
        "risk": {
            "model": summary.get("model"),
            "algorithm": summary.get("signal_algorithm"),
            "train_start": str(summary.get("train_start_actual", ""))[:10] if summary.get("train_start_actual") else "",
            "train_end": str(summary.get("train_end_actual", ""))[:10] if summary.get("train_end_actual") else "",
            "train_rows": summary.get("train_rows"),
            "hmm_state": summary.get("hmm_predicted_state"),
            "hmm_exposure": summary.get("hmm_exposure"),
            "risk_gate": summary.get("risk_gate"),
            "fit_quality": fit_quality,
            "risk_gate_inputs": risk_inputs,
        },
        "research_audit": {
            "trade_audit": trade_audit,
            "frozen_latest_year": frozen_year,
            "frozen_sensitivity": sensitivity,
            "files": [item for item in files if item],
        },
    }


def _daily_output_dir(signal_date: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = LIVE_OPS_ROOT / f"daily_chain_{signal_date.replace('-', '')}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _load_position_state() -> dict[str, Any] | None:
    return _read_json_file(POSITION_STATE_PATH)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_ready(payload: Any) -> Any:
    return json.loads(json.dumps(payload, ensure_ascii=False, default=_json_default))


def _json_default(obj):
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _compute_health(signal: dict[str, Any]) -> dict[str, Any]:
    summary = signal.get("summary") or {}
    fit = summary.get("fit_quality") or {}
    risk_inputs = summary.get("risk_gate_inputs") or {}
    rank_ic = _as_float(fit.get("rank_ic_rolling"))
    spread = _as_float(fit.get("decile_spread_rolling"))
    top5_excess = _as_float(fit.get("top5_excess_rolling"))
    top5_hit = _as_float(fit.get("top5_hit_rolling"))
    fit_obs = int(fit.get("fit_obs") or 0)
    min_obs = int(fit.get("min_obs") or 15)
    risk_gate = _as_float(summary.get("risk_gate"))
    hmm_exposure = _as_float(summary.get("hmm_exposure"))
    final_exposure = _as_float(summary.get("final_exposure"))
    components = []
    if fit_obs < min_obs:
        components.append({"name": "fit_quality", "state": "UNKNOWN", "reason": f"样本不足 {fit_obs}/{min_obs}"})
    elif rank_ic < 0 and spread < 0:
        components.append({"name": "fit_quality", "state": "INVERTED", "reason": "RankIC 与 decile spread 同时为负，启用反向排序"})
    elif rank_ic > 0 and spread > 0:
        components.append({"name": "fit_quality", "state": "HEALTHY", "reason": "RankIC 与 decile spread 同向为正"})
    else:
        components.append({"name": "fit_quality", "state": "MIXED", "reason": "方向指标不一致"})

    gate_type = str(risk_inputs.get("gate_type") or "market_payoff_gate")
    if risk_gate <= 0:
        components.append({"name": "payoff_gate", "state": "RISK_OFF", "reason": f"{gate_type}=0"})
    else:
        components.append({"name": "payoff_gate", "state": "PASS", "reason": f"{gate_type}={risk_gate:.2f}"})

    if hmm_exposure <= 0:
        components.append({"name": "hmm_regime", "state": "RISK_OFF", "reason": "HMM exposure=0"})
    elif hmm_exposure < 1:
        components.append({"name": "hmm_regime", "state": "REDUCED", "reason": f"HMM exposure={hmm_exposure:.2f}"})
    else:
        components.append({"name": "hmm_regime", "state": "FULL", "reason": "HMM exposure=1"})

    if risk_gate <= 0 or final_exposure <= 0:
        overall = "RISK_OFF"
    elif any(item["state"] == "INVERTED" for item in components):
        overall = "CAUTIOUS_ALPHA"
    elif any(item["state"] in {"UNKNOWN", "MIXED", "REDUCED"} for item in components):
        overall = "WATCH"
    else:
        overall = "HEALTHY"
    return {
        "signal_date": str(summary.get("signal_date", ""))[:10],
        "overall": overall,
        "components": components,
        "metrics": {
            "rank_ic_rolling": rank_ic,
            "decile_spread_rolling": spread,
            "top5_excess_rolling": top5_excess,
            "top5_hit_rolling": top5_hit,
            "fit_obs": fit_obs,
            "risk_gate": risk_gate,
            "hmm_exposure": hmm_exposure,
            "final_exposure": final_exposure,
            "top5_excess_5round": _as_float(risk_inputs.get("top5_excess_5round")),
            "payoff_mean_net_10d": _as_float(risk_inputs.get("payoff_mean_net_10d")),
            "payoff_lcb_net_10d": _as_float(risk_inputs.get("payoff_lcb_net_10d")),
            "payoff_effective_obs": _as_float(risk_inputs.get("payoff_effective_obs")),
        },
    }


def _decide_position_state(signal: dict[str, Any], health: dict[str, Any]) -> dict[str, Any]:
    summary = signal.get("summary") or {}
    previous = _load_position_state() or {}
    final_exposure = _as_float(summary.get("final_exposure"))
    if not np.isfinite(final_exposure):
        final_exposure = 0.0
    health_state = health.get("overall")
    if not signal.get("summary"):
        state = "OBSERVE"
        target = 0.0
        allow_buys = False
        reason = "没有可用信号"
    elif health_state == "RISK_OFF" or final_exposure <= 0:
        state = "FLAT"
        target = 0.0
        allow_buys = False
        reason = "风险门控或最终仓位为 0"
    elif health_state == "CAUTIOUS_ALPHA":
        state = "CAUTIOUS"
        target = min(final_exposure, 0.5)
        allow_buys = target > 0
        reason = "Alpha 方向反转，允许小心执行"
    elif health_state == "WATCH":
        state = "REDUCE"
        target = min(final_exposure, 0.5)
        allow_buys = target > 0
        reason = "健康度处于观察区，限制仓位"
    else:
        state = "NORMAL"
        target = final_exposure
        allow_buys = target > 0
        reason = "健康度通过，按模型仓位执行"
    output = {
        "signal_date": str(summary.get("signal_date", ""))[:10],
        "state": state,
        "previous_state": previous.get("state"),
        "target_exposure": float(np.clip(target, 0.0, 1.0)),
        "allow_new_buys": bool(allow_buys),
        "reason": reason,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    POSITION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_json(POSITION_STATE_PATH, output)
    return output


def _build_orders(signal: dict[str, Any], position_state: dict[str, Any], holdings_text: str) -> pd.DataFrame:
    rows = []
    signal_date = position_state.get("signal_date") or str((signal.get("summary") or {}).get("signal_date", ""))[:10]
    target_exposure = float(position_state.get("target_exposure") or 0.0)
    allow_buys = bool(position_state.get("allow_new_buys"))
    top = pd.DataFrame(signal.get("top") or [])
    if allow_buys and not top.empty:
        suspended = top["is_suspended"].fillna(False).astype(bool) if "is_suspended" in top else pd.Series(False, index=top.index)
        st = top["is_st"].fillna(False).astype(bool) if "is_st" in top else pd.Series(False, index=top.index)
        limit_up = top["is_limit_up_open"].fillna(False).astype(bool) if "is_limit_up_open" in top else pd.Series(False, index=top.index)
        tradable = top[~suspended & ~st & ~limit_up].head(5)
        weight = target_exposure / len(tradable) if len(tradable) else 0.0
        for row in tradable.to_dict("records"):
            rows.append(
                {
                    "signal_date": signal_date,
                    "side": "BUY",
                    "ts_code": row.get("ts_code"),
                    "name": row.get("name"),
                    "target_weight": weight,
                    "reason": position_state.get("reason"),
                    "execution": "next_trade_day_open",
                    "status": "DRAFT",
                }
            )
    elif not allow_buys:
        rows.append(
            {
                "signal_date": signal_date,
                "side": "NO_BUY",
                "ts_code": "",
                "name": "",
                "target_weight": 0.0,
                "reason": position_state.get("reason"),
                "execution": "no_new_position",
                "status": "INFO",
            }
        )

    if holdings_text.strip():
        advice = _sell_advice(SellAdviceRequest(signal_date=signal_date, holdings_text=holdings_text))
        for item in advice.get("items", []):
            if item.get("action") == "SELL":
                rows.append(
                    {
                        "signal_date": signal_date,
                        "side": "SELL",
                        "ts_code": item.get("ts_code"),
                        "name": item.get("name"),
                        "target_weight": 0.0,
                        "reason": item.get("reason"),
                        "execution": "next_trade_day_open_if_sellable",
                        "status": "DRAFT",
                    }
                )
    return pd.DataFrame(rows)


def _execution_audit(signal: dict[str, Any], orders: pd.DataFrame) -> dict[str, Any]:
    top = pd.DataFrame(signal.get("top") or [])
    blockers = []
    if not top.empty:
        for col, label in [("is_suspended", "信号日停牌"), ("is_st", "ST"), ("is_limit_up_open", "信号日涨停开盘")]:
            if col in top:
                hits = top[top[col].fillna(False).astype(bool)]
                for row in hits.to_dict("records"):
                    blockers.append({"ts_code": row.get("ts_code"), "name": row.get("name"), "issue": label})
    return {
        "blocking_trade_issues": len(blockers),
        "blockers": blockers,
        "order_count": int(len(orders)),
        "buy_count": int((orders.get("side") == "BUY").sum()) if not orders.empty and "side" in orders else 0,
        "sell_count": int((orders.get("side") == "SELL").sum()) if not orders.empty and "side" in orders else 0,
        "note": "这里只能审计信号日已知的停牌/ST/涨停开盘等信息；下一交易日开盘是否可成交，要等下一交易日数据确认。",
    }


def _shadow_report(
    out: Path,
    data_status: dict[str, Any],
    signal: dict[str, Any],
    health: dict[str, Any],
    position_state: dict[str, Any],
    orders: pd.DataFrame,
    audit: dict[str, Any],
) -> dict[str, Any]:
    top = signal.get("top") or []
    files = {
        "health": out / "health.json",
        "position_state": out / "position_state.json",
        "orders": out / "orders.csv",
        "execution_audit": out / "execution_audit.json",
        "shadow_report": out / "shadow_report.md",
        "shadow_report_json": out / "shadow_report.json",
    }
    _write_json(files["health"], health)
    _write_json(files["position_state"], position_state)
    orders.to_csv(files["orders"], index=False, encoding="utf-8-sig")
    _write_json(files["execution_audit"], audit)
    report = {
        "run_dir": _rel(out),
        "data": data_status,
        "signal_run": signal.get("run_dir"),
        "health": health,
        "position_state": position_state,
        "execution_audit": audit,
        "top": top[:10],
        "files": {name: _rel(path) for name, path in files.items()},
    }
    _write_json(files["shadow_report_json"], report)
    files["shadow_report"].write_text(_render_shadow_markdown(report), encoding="utf-8")
    return report


def _render_shadow_markdown(report: dict[str, Any]) -> str:
    state = report["position_state"]
    health = report["health"]
    audit = report["execution_audit"]
    lines = [
        "# ATR Reversion Daily Shadow Report",
        "",
        f"- 信号日：{state.get('signal_date')}",
        f"- 仓位状态：{state.get('state')}",
        f"- 目标仓位：{state.get('target_exposure', 0):.0%}",
        f"- 健康度：{health.get('overall')}",
        f"- 执行审计红灯：{audit.get('blocking_trade_issues')}",
        f"- 决策原因：{state.get('reason')}",
        "",
        "## Health Components",
    ]
    for item in health.get("components", []):
        lines.append(f"- {item.get('name')}: {item.get('state')}，{item.get('reason')}")
    lines.extend(["", "## Top Candidates"])
    for row in report.get("top", [])[:5]:
        lines.append(f"- {row.get('rank')}. {row.get('ts_code')} {row.get('name')} weight={row.get('target_weight')}")
    return "\n".join(lines) + "\n"


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _load_shadow_portfolio() -> dict[str, Any]:
    data = _read_json_file(SHADOW_PORTFOLIO_PATH)
    if not data:
        return {"positions": []}
    data.setdefault("positions", [])
    return data


def _save_shadow_portfolio(book: dict[str, Any]) -> None:
    book["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(SHADOW_PORTFOLIO_PATH, book)


def _add_shadow_positions(req: ShadowAddRequest) -> dict[str, Any]:
    codes = [code.strip().upper() for code in req.ts_codes if code.strip()]
    if not codes:
        raise HTTPException(status_code=400, detail="没有选择候选股")
    run_dir = _latest_signal_dir_for_date(pd.Timestamp(req.signal_date))
    if run_dir is None:
        raise HTTPException(status_code=404, detail=f"找不到 {req.signal_date} 的有效信号")
    signal = _read_signal(run_dir)
    top = pd.DataFrame(signal.get("top") or [])
    if top.empty:
        raise HTTPException(status_code=404, detail="信号候选为空")
    selected = top[top["ts_code"].astype(str).isin(codes)].copy()
    missing = sorted(set(codes) - set(selected["ts_code"].astype(str)))
    book = _load_shadow_portfolio()
    existing = {
        (str(p.get("signal_date")), str(p.get("ts_code")), str(p.get("status", "OPEN")))
        for p in book["positions"]
        if str(p.get("status", "OPEN")) not in {"CLOSED", "REMOVED"}
    }
    added = []
    for row in selected.to_dict("records"):
        key = (req.signal_date, str(row.get("ts_code")), "OPEN")
        if key in existing:
            continue
        position = {
            "id": uuid.uuid4().hex[:10],
            "status": "OPEN",
            "signal_date": req.signal_date,
            "ts_code": row.get("ts_code"),
            "name": row.get("name"),
            "rank": row.get("rank"),
            "industry_l1_name": row.get("industry_l1_name"),
            "factor_value": row.get("factor_value"),
            "target_weight": row.get("target_weight"),
            "signal_raw_close": row.get("raw_close"),
            "signal_amount_cny": row.get("amount_cny"),
            "source_run": signal.get("run_dir"),
            "selected_at": datetime.now().isoformat(timespec="seconds"),
            "entry_policy": "next_trade_day_open",
        }
        book["positions"].append(position)
        added.append(position)
    _save_shadow_portfolio(book)
    evaluated = _evaluate_shadow_portfolio()
    return {"added": added, "missing": missing, "portfolio": evaluated}


def _delete_shadow_position(req: ShadowPositionActionRequest) -> dict[str, Any]:
    book = _load_shadow_portfolio()
    before = len(book.get("positions", []))
    kept = [p for p in book.get("positions", []) if str(p.get("id")) != req.position_id]
    if len(kept) == before:
        raise HTTPException(status_code=404, detail="shadow position not found")
    book["positions"] = kept
    book.setdefault("audit_log", []).append(
        {
            "action": "DELETE",
            "position_id": req.position_id,
            "note": req.note,
            "at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _save_shadow_portfolio(book)
    return {"position_id": req.position_id, "portfolio": _evaluate_shadow_portfolio()}


def _sell_shadow_position(req: ShadowPositionActionRequest) -> dict[str, Any]:
    book = _load_shadow_portfolio()
    pos = next((p for p in book.get("positions", []) if str(p.get("id")) == req.position_id), None)
    if pos is None:
        raise HTTPException(status_code=404, detail="shadow position not found")
    if str(pos.get("status", "OPEN")) == "CLOSED":
        raise HTTPException(status_code=400, detail="shadow position is already closed")

    repo = _load_repo()
    version, panel = repo.load_panel("latest")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    code = str(pos.get("ts_code"))
    latest_date = panel["trade_date"].max().normalize()
    action_date = pd.Timestamp(req.action_date).normalize() if req.action_date else latest_date
    day = panel.loc[panel["trade_date"].eq(action_date) & panel["ts_code"].astype(str).eq(code)]
    if day.empty:
        raise HTTPException(status_code=404, detail=f"no market row for {code} on {action_date.date()}")
    mark = day.iloc[-1]
    if bool(mark.get("is_suspended", False)):
        raise HTTPException(status_code=400, detail=f"{code} is suspended on {action_date.date()}")

    sell_adj = _as_float(mark.get("adj_close"))
    sell_raw = _as_float(mark.get("raw_close"))
    entry_adj = _as_float(pos.get("entry_adj_open"))
    entry_raw = _as_float(pos.get("entry_raw_open"))
    entry_date = pos.get("entry_date")
    if not np.isfinite(entry_adj):
        evaluated = _evaluate_shadow_portfolio()
        match = next((p for p in evaluated.get("positions", []) if str(p.get("id")) == req.position_id), None)
        entry_adj = _as_float((match or {}).get("entry_adj_open"))
        entry_raw = _as_float((match or {}).get("entry_raw_open"))
        entry_date = (match or {}).get("entry_date")
    if not np.isfinite(entry_adj):
        raise HTTPException(status_code=400, detail="shadow position has no simulated entry fill to sell")
    if entry_date and action_date < pd.Timestamp(entry_date).normalize():
        raise HTTPException(status_code=400, detail="sell date is before simulated entry date")
    realized_return = sell_adj / entry_adj - 1.0 if np.isfinite(entry_adj) and entry_adj else float("nan")

    pos.update(
        {
            "status": "CLOSED",
            "closed_at": datetime.now().isoformat(timespec="seconds"),
            "entry_date": entry_date,
            "entry_raw_open": entry_raw,
            "entry_adj_open": entry_adj,
            "exit_date": action_date,
            "exit_raw_close": sell_raw,
            "exit_adj_close": sell_adj,
            "realized_return": realized_return,
            "close_data_version": version,
            "close_note": req.note,
        }
    )
    book.setdefault("audit_log", []).append(
        {
            "action": "SELL",
            "position_id": req.position_id,
            "ts_code": code,
            "action_date": action_date,
            "exit_raw_close": sell_raw,
            "realized_return": realized_return,
            "note": req.note,
            "at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _save_shadow_portfolio(book)
    return {"position": pos, "portfolio": _evaluate_shadow_portfolio()}


def _evaluate_shadow_portfolio() -> dict[str, Any]:
    book = _load_shadow_portfolio()
    repo = _load_repo()
    version, panel = repo.load_panel("latest")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    dates = list(pd.Index(panel["trade_date"].drop_duplicates()).sort_values())
    latest_date = dates[-1] if dates else pd.NaT
    date_pos = {d.normalize(): i for i, d in enumerate(dates)}
    panel_idx = panel.set_index(["trade_date", "ts_code"], drop=False)
    latest_signal = _signal_for_date(latest_date) if pd.notna(latest_date) else None
    final_exposure = None
    if latest_signal:
        final_exposure = float(latest_signal["summary"].get("final_exposure", np.nan))
    hazard = _sell_impact_hazard()
    hazard_today = hazard[hazard["trade_date"].eq(latest_date)].set_index("ts_code") if pd.notna(latest_date) else pd.DataFrame()
    rows = []
    open_returns = []
    for pos in book.get("positions", []):
        row = dict(pos)
        if str(pos.get("status", "OPEN")) == "CLOSED":
            realized = _as_float(pos.get("realized_return"))
            row.update(
                {
                    "mark_date": pos.get("exit_date"),
                    "mark_raw_close": pos.get("exit_raw_close"),
                    "mark_adj_close": pos.get("exit_adj_close"),
                    "shadow_return": realized,
                    "eval_status": "CLOSED",
                    "eval_note": pos.get("close_note") or "closed shadow position",
                    "sell_action": "CLOSED",
                    "sell_reason": pos.get("close_note") or "closed shadow position",
                    "sell_impact_efficiency": None,
                    "sell_impact_deviation_60d": None,
                    "hazard_strict": False,
                }
            )
            rows.append(row)
            continue
        signal_ts = pd.Timestamp(pos.get("signal_date")).normalize()
        signal_i = date_pos.get(signal_ts)
        if signal_i is None:
            row.update({"eval_status": "NO_SIGNAL_DATE", "eval_note": "信号日不在 latest 面板中"})
            rows.append(row)
            continue
        entry_i = signal_i + 1
        if entry_i >= len(dates):
            row.update({"eval_status": "PENDING_ENTRY", "eval_note": "下一交易日数据尚不可见"})
            rows.append(row)
            continue
        entry_date = dates[entry_i].normalize()
        code = str(pos.get("ts_code"))
        try:
            entry = panel_idx.loc[(entry_date, code)]
        except KeyError:
            row.update({"entry_date": entry_date, "eval_status": "NO_ENTRY_ROW", "eval_note": "下一交易日无该股票行情"})
            rows.append(row)
            continue
        if bool(entry.get("is_suspended", False)):
            row.update({"entry_date": entry_date, "eval_status": "BLOCKED_BUY", "eval_note": "下一交易日停牌，模拟未成交"})
            rows.append(row)
            continue
        if bool(entry.get("is_limit_up_open", False)):
            row.update({"entry_date": entry_date, "eval_status": "BLOCKED_BUY", "eval_note": "下一交易日涨停开盘，模拟未成交"})
            rows.append(row)
            continue
        entry_adj = _as_float(entry.get("adj_open"))
        entry_raw = _as_float(entry.get("raw_open"))
        latest_for_code = panel[panel["ts_code"].astype(str).eq(code) & panel["trade_date"].le(latest_date)].sort_values("trade_date")
        if latest_for_code.empty:
            row.update({"entry_date": entry_date, "eval_status": "NO_MARK_ROW", "eval_note": "无可评估行情"})
            rows.append(row)
            continue
        mark = latest_for_code.iloc[-1]
        mark_adj = _as_float(mark.get("adj_close"))
        mark_raw = _as_float(mark.get("raw_close"))
        ret = mark_adj / entry_adj - 1.0 if entry_adj and np.isfinite(entry_adj) and np.isfinite(mark_adj) else np.nan
        held_days = date_pos.get(pd.Timestamp(mark["trade_date"]).normalize(), entry_i) - entry_i + 1
        hrow = hazard_today.loc[code] if code in hazard_today.index else pd.Series(dtype=object)
        strict_hazard = bool(hrow.get("hazard_dev_q5_eff_q5", False))
        sell_reasons = []
        sell_action = "HOLD"
        if final_exposure is not None and np.isfinite(final_exposure) and final_exposure <= 0:
            sell_action = "SELL"
            sell_reasons.append("final_exposure=0")
        if held_days >= 10:
            sell_action = "SELL"
            sell_reasons.append("holding>=10d")
        if strict_hazard:
            sell_action = "SELL"
            sell_reasons.append("sell_impact_hazard")
        if not sell_reasons:
            sell_reasons.append("no_sell_rule_triggered")
        row.update(
            {
                "entry_date": entry_date,
                "entry_raw_open": entry_raw,
                "entry_adj_open": entry_adj,
                "mark_date": pd.Timestamp(mark["trade_date"]).normalize(),
                "mark_raw_close": mark_raw,
                "mark_adj_close": mark_adj,
                "holding_trade_days": int(max(held_days, 0)),
                "shadow_return": ret,
                "eval_status": "OPEN_EVALUATED",
                "sell_action": sell_action,
                "sell_reason": "; ".join(sell_reasons),
                "sell_impact_efficiency": None if pd.isna(hrow.get("sell_impact_efficiency", np.nan)) else float(hrow.get("sell_impact_efficiency")),
                "sell_impact_deviation_60d": None if pd.isna(hrow.get("sell_impact_deviation_60d", np.nan)) else float(hrow.get("sell_impact_deviation_60d")),
                "hazard_strict": strict_hazard,
                "eval_note": "按下一交易日开盘模拟入场，按最新收盘估值",
            }
        )
        if np.isfinite(ret):
            open_returns.append(ret)
        rows.append(row)
    summary = {
        "data_version": version,
        "latest_date": latest_date,
        "position_count": len(rows),
        "open_count": sum(1 for r in rows if str(r.get("status", "OPEN")) == "OPEN"),
        "closed_count": sum(1 for r in rows if str(r.get("status", "OPEN")) == "CLOSED"),
        "evaluated_count": sum(1 for r in rows if r.get("eval_status") == "OPEN_EVALUATED"),
        "pending_count": sum(1 for r in rows if r.get("eval_status") == "PENDING_ENTRY"),
        "blocked_count": sum(1 for r in rows if r.get("eval_status") == "BLOCKED_BUY"),
        "sell_count": sum(1 for r in rows if r.get("sell_action") == "SELL"),
        "avg_shadow_return": float(np.mean(open_returns)) if open_returns else None,
        "win_rate": float(np.mean([r > 0 for r in open_returns])) if open_returns else None,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    out = {"summary": summary, "positions": rows, "file": _rel(SHADOW_PORTFOLIO_PATH)}
    LIVE_OPS_ROOT.mkdir(parents=True, exist_ok=True)
    (LIVE_OPS_ROOT / "shadow_portfolio_latest_eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return _json_ready(out)


@lru_cache(maxsize=1)
def _stock_names() -> dict[str, str]:
    for path in sorted((ROOT / "data" / "versions").glob("*/raw/tushare/stock_basic.parquet"), reverse=True):
        try:
            data = pd.read_parquet(path, columns=["ts_code", "name"])
        except Exception:
            continue
        return dict(zip(data["ts_code"].astype(str), data["name"].astype(str)))
    return {}


@lru_cache(maxsize=1)
def _sell_impact_hazard() -> pd.DataFrame:
    run = ROOT / "artifacts" / "runs" / "sell_impact_efficiency_v1__20260702T032703Z__be62d97d"
    eff = pd.read_parquet(run / "factor_values.parquet")
    dev = pd.read_parquet(run / "conditioning_factor_values.parquet")
    eff["trade_date"] = pd.to_datetime(eff["trade_date"])
    dev["trade_date"] = pd.to_datetime(dev["trade_date"])
    eff = eff.rename(columns={"factor_value": "sell_impact_efficiency", "factor_valid": "eff_valid"})
    dev = dev.rename(columns={"factor_value": "sell_impact_deviation_60d", "factor_valid": "dev_valid"})
    data = eff[["trade_date", "ts_code", "sell_impact_efficiency", "eff_valid"]].merge(
        dev[["trade_date", "ts_code", "sell_impact_deviation_60d", "dev_valid"]],
        on=["trade_date", "ts_code"],
        how="inner",
    )
    valid_eff = data["eff_valid"].fillna(False) & data["sell_impact_efficiency"].notna()
    valid_dev = data["dev_valid"].fillna(False) & data["sell_impact_deviation_60d"].notna()
    data["eff_q80"] = data["sell_impact_efficiency"] >= data["sell_impact_efficiency"].where(valid_eff).groupby(data["trade_date"]).transform(lambda s: s.quantile(0.8))
    data["dev_q80"] = data["sell_impact_deviation_60d"] >= data["sell_impact_deviation_60d"].where(valid_dev).groupby(data["trade_date"]).transform(lambda s: s.quantile(0.8))
    data["hazard_dev_q5_eff_q5"] = valid_eff & valid_dev & data["eff_q80"].fillna(False) & data["dev_q80"].fillna(False)
    return data[[
        "trade_date",
        "ts_code",
        "sell_impact_efficiency",
        "sell_impact_deviation_60d",
        "hazard_dev_q5_eff_q5",
    ]]


def _signal_for_date(signal_date: pd.Timestamp) -> dict[str, Any] | None:
    root = ROOT / "artifacts" / "atr_reversion_live_signals"
    if not root.exists():
        return None
    candidates = []
    for run_dir in root.iterdir():
        summary_path = run_dir / "signal_summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if pd.Timestamp(summary.get("signal_date")).normalize() == signal_date.normalize():
            candidates.append((run_dir.stat().st_mtime, summary, run_dir))
    if not candidates:
        return None
    _mtime, summary, run_dir = max(candidates, key=lambda item: item[0])
    return {"summary": summary, "run_dir": str(run_dir.relative_to(ROOT))}


def _latest_signal_dir_for_date(signal_date: pd.Timestamp) -> Path | None:
    root = ROOT / "artifacts" / "atr_reversion_live_signals"
    if not root.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for run_dir in root.iterdir():
        summary_path = run_dir / "signal_summary.json"
        top_path = run_dir / "top_recommendations.csv"
        if not summary_path.exists() or not top_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if pd.Timestamp(summary.get("signal_date")).normalize() == signal_date.normalize():
                candidates.append((run_dir.stat().st_mtime, run_dir))
        except Exception:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _parse_holdings(text: str) -> list[dict[str, str]]:
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in re.split(r"[,，\\s]+", line) if p.strip()]
        if not parts:
            continue
        code = parts[0].upper()
        entry_date = parts[1] if len(parts) > 1 else ""
        out.append({"ts_code": code, "entry_date": entry_date})
    return out


def _sell_advice(req: SellAdviceRequest) -> dict[str, Any]:
    signal_date = pd.Timestamp(req.signal_date)
    holdings = _parse_holdings(req.holdings_text)
    if not holdings:
        return {"signal_date": req.signal_date, "items": [], "note": "未输入持仓。"}
    repo = _load_repo()
    version, panel = repo.load_panel("latest")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    dates = list(pd.Index(panel["trade_date"].drop_duplicates()).sort_values())
    if signal_date not in set(dates):
        raise HTTPException(status_code=400, detail=f"signal_date {req.signal_date} not found in latest panel")
    signal_info = _signal_for_date(signal_date)
    final_exposure = None
    if signal_info:
        final_exposure = float(signal_info["summary"].get("final_exposure", np.nan))
    hazard = _sell_impact_hazard()
    hazard_today = hazard[hazard["trade_date"].eq(signal_date)].set_index("ts_code")
    panel_today = panel[panel["trade_date"].eq(signal_date)].set_index("ts_code")
    names = _stock_names()
    date_pos = {d: i for i, d in enumerate(dates)}
    signal_i = date_pos[signal_date]
    items = []
    for h in holdings:
        code = h["ts_code"]
        entry_raw = h.get("entry_date") or ""
        entry_ts = pd.NaT
        held_days = None
        maturity_sell = False
        if entry_raw:
            entry_ts = pd.Timestamp(entry_raw)
            if entry_ts in date_pos:
                held_days = signal_i - date_pos[entry_ts] + 1
                maturity_sell = held_days >= 10
        hrow = hazard_today.loc[code] if code in hazard_today.index else pd.Series(dtype=object)
        strict_hazard = bool(hrow.get("hazard_dev_q5_eff_q5", False))
        prow = panel_today.loc[code] if code in panel_today.index else pd.Series(dtype=object)
        signal_day_block = bool(prow.get("is_suspended", False)) or bool(prow.get("is_limit_down_open", False))
        reasons = []
        action = "HOLD"
        if final_exposure is not None and np.isfinite(final_exposure) and final_exposure <= 0:
            action = "SELL"
            reasons.append("组合风控 final_exposure=0")
        if maturity_sell:
            action = "SELL"
            reasons.append("持有满10个交易日")
        if strict_hazard:
            action = "SELL"
            reasons.append("卖压避雷触发：deviation前20%且efficiency前20%")
        if not reasons:
            reasons.append("未触发卖出规则")
        items.append({
            "ts_code": code,
            "name": names.get(code, ""),
            "entry_date": entry_raw,
            "holding_trade_days": held_days,
            "action": action,
            "reason": "；".join(reasons),
            "sell_impact_efficiency": None if pd.isna(hrow.get("sell_impact_efficiency", np.nan)) else float(hrow.get("sell_impact_efficiency")),
            "sell_impact_deviation_60d": None if pd.isna(hrow.get("sell_impact_deviation_60d", np.nan)) else float(hrow.get("sell_impact_deviation_60d")),
            "hazard_strict": strict_hazard,
            "signal_day_sell_block": signal_day_block,
            "execution_note": "按T日收盘信号，T+1开盘卖出；T+1跌停/停牌需顺延，当前无法预知下一交易日可卖性。",
        })
    return {
        "signal_date": req.signal_date,
        "data_version": version,
        "final_exposure": final_exposure,
        "signal_run": signal_info["run_dir"] if signal_info else None,
        "items": items,
    }


def _run_sync_task(task_id: str, req: SyncRequest) -> None:
    _set_task(task_id, status="running", started_at=datetime.now().isoformat(timespec="seconds"))
    try:
        status = _sync_data_inline(task_id, req.start, req.end, req.merge_full_history)
        _set_task(task_id, status="succeeded", finished_at=datetime.now().isoformat(timespec="seconds"), result=status)
    except Exception as exc:
        _append_log(task_id, f"ERROR: {exc}")
        _set_task(task_id, status="failed", finished_at=datetime.now().isoformat(timespec="seconds"), error=str(exc))


def _sync_data_inline(task_id: str, start: str, end: str, merge_full_history: bool) -> dict[str, Any]:
    _append_log(task_id, f"[update_data] start={start} end={end}")
    rc = _run_subprocess(
        task_id,
        [
            sys.executable,
            "-m",
            "factor_forge.cli",
            "data",
            "ingest",
            "--config",
            "configs/project.yaml",
            "--start",
            start,
            "--end",
            end,
        ],
    )
    if rc != 0:
        raise RuntimeError(f"data ingest failed with exit code {rc}")
    status = _latest_data_status()
    if merge_full_history and not status["complete"]:
        status["version"] = _append_increment_to_complete(task_id, status["version"])
        status = _latest_data_status()
    _append_log(task_id, f"[update_data] done version={status.get('version')} end={status.get('end_date')}")
    return status


def _run_signal_task(task_id: str, req: SignalRequest) -> None:
    _set_task(task_id, status="running", started_at=datetime.now().isoformat(timespec="seconds"))
    try:
        _ensure_signal_date_data(task_id, req.signal_date)
        rc = _run_subprocess(task_id, [sys.executable, "scripts/atr_reversion_live_signal_rolling2y.py", req.signal_date])
        if rc != 0:
            raise RuntimeError(f"signal generation failed with exit code {rc}")
        run_dir = _latest_signal_dir()
        if run_dir is None:
            raise RuntimeError("No signal output directory found.")
        _set_task(
            task_id,
            status="succeeded",
            finished_at=datetime.now().isoformat(timespec="seconds"),
            result=_read_signal(run_dir),
        )
    except Exception as exc:
        _append_log(task_id, f"ERROR: {exc}")
        _set_task(task_id, status="failed", finished_at=datetime.now().isoformat(timespec="seconds"), error=str(exc))


def _generate_signal_inline(task_id: str, signal_date: str, force: bool = False) -> dict[str, Any]:
    _append_log(task_id, f"[generate_all_model_signals] signal_date={signal_date}")
    _append_log(task_id, "[generate_all_model_signals] active production model: event_alpha_payoff_gate_top5_frozen_20260707")
    existing = _latest_signal_dir_for_date(pd.Timestamp(signal_date))
    if existing is not None and not force:
        signal = _read_signal(existing)
        _append_log(task_id, f"[generate_all_model_signals] reused existing run={signal.get('run_dir')}")
        return signal
    _ensure_signal_date_data(task_id, signal_date)
    rc = _run_subprocess(task_id, [sys.executable, "scripts/atr_reversion_live_signal_rolling2y.py", signal_date])
    if rc != 0:
        raise RuntimeError(f"signal generation failed with exit code {rc}")
    run_dir = _latest_signal_dir()
    if run_dir is None:
        raise RuntimeError("No signal output directory found.")
    signal = _read_signal(run_dir)
    _append_log(task_id, f"[generate_all_model_signals] done run={signal.get('run_dir')}")
    return signal


def _run_daily_chain_task(task_id: str, req: DailyChainRequest) -> None:
    _set_task(task_id, status="running", started_at=datetime.now().isoformat(timespec="seconds"))
    try:
        signal_compact = req.signal_date.replace("-", "")
        start = req.start or signal_compact
        end = req.end or signal_compact
        out = _daily_output_dir(req.signal_date)
        _append_log(task_id, f"[daily_chain] output={_rel(out)}")

        if req.update_data:
            data_status = _sync_data_inline(task_id, start, end, req.merge_full_history)
        else:
            _append_log(task_id, "[update_data] skipped")
            data_status = _latest_data_status()

        signal = _generate_signal_inline(task_id, req.signal_date, force=req.force_regenerate_signals)

        _append_log(task_id, "[compute_health] computing health from latest signal summary")
        health = _compute_health(signal)
        _append_log(task_id, f"[compute_health] overall={health.get('overall')}")

        _append_log(task_id, "[decide_position_state] applying state policy")
        position_state = _decide_position_state(signal, health)
        _append_log(
            task_id,
            f"[decide_position_state] state={position_state.get('state')} target={position_state.get('target_exposure'):.0%}",
        )

        _append_log(task_id, "[build_orders] drafting orders")
        orders = _build_orders(signal, position_state, req.holdings_text)
        _append_log(task_id, f"[build_orders] rows={len(orders)}")

        _append_log(task_id, "[execution_audit] checking signal-day execution constraints")
        audit = _execution_audit(signal, orders)
        _append_log(task_id, f"[execution_audit] blocking_trade_issues={audit.get('blocking_trade_issues')}")

        _append_log(task_id, "[shadow_report] writing report files")
        report = _shadow_report(out, data_status, signal, health, position_state, orders, audit)
        _append_log(task_id, f"[shadow_report] done report={report['files']['shadow_report']}")
        _set_task(task_id, status="succeeded", finished_at=datetime.now().isoformat(timespec="seconds"), result=report)
    except Exception as exc:
        _append_log(task_id, f"ERROR: {exc}")
        _set_task(task_id, status="failed", finished_at=datetime.now().isoformat(timespec="seconds"), error=str(exc))


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status() -> dict[str, Any]:
    signal = None
    latest_dir = _latest_signal_dir()
    if latest_dir is not None:
        try:
            signal = _read_signal(latest_dir)
        except Exception:
            signal = {"run_dir": str(latest_dir.relative_to(ROOT))}
    data_status = _latest_data_status()
    return {
        "data": data_status,
        "default_signal_date": data_status.get("end_date"),
        "latest_signal": signal,
        "dashboard": _build_dashboard(data_status, signal),
    }


@app.post("/api/sync")
def sync(req: SyncRequest) -> dict[str, str]:
    task_id, _ = _task()
    threading.Thread(target=_run_sync_task, args=(task_id, req), daemon=True).start()
    return {"task_id": task_id}


@app.post("/api/signal")
def signal(req: SignalRequest) -> dict[str, str]:
    task_id, _ = _task()
    threading.Thread(target=_run_signal_task, args=(task_id, req), daemon=True).start()
    return {"task_id": task_id}


@app.post("/api/daily-chain")
def daily_chain(req: DailyChainRequest) -> dict[str, str]:
    task_id, _ = _task()
    threading.Thread(target=_run_daily_chain_task, args=(task_id, req), daemon=True).start()
    return {"task_id": task_id}


@app.post("/api/sell-advice")
def sell_advice(req: SellAdviceRequest) -> dict[str, Any]:
    return _sell_advice(req)


@app.get("/api/shadow-portfolio")
def shadow_portfolio() -> dict[str, Any]:
    return _evaluate_shadow_portfolio()


@app.post("/api/shadow-portfolio/add")
def shadow_portfolio_add(req: ShadowAddRequest) -> dict[str, Any]:
    return _json_ready(_add_shadow_positions(req))


@app.post("/api/shadow-portfolio/sell")
def shadow_portfolio_sell(req: ShadowPositionActionRequest) -> dict[str, Any]:
    return _json_ready(_sell_shadow_position(req))


@app.post("/api/shadow-portfolio/delete")
def shadow_portfolio_delete(req: ShadowPositionActionRequest) -> dict[str, Any]:
    return _json_ready(_delete_shadow_position(req))


@app.get("/api/tasks/{task_id}")
def task_status(task_id: str) -> dict[str, Any]:
    with TASK_LOCK:
        task = TASKS.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="task not found")
        return dict(task)


@app.get("/api/file")
def file(path: str):
    rel = Path(path)
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(status_code=400, detail="invalid path")
    target = (ROOT / rel).resolve()
    if ROOT not in target.parents and target != ROOT:
        raise HTTPException(status_code=400, detail="invalid path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(target)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app.server:app", host="127.0.0.1", port=8765, reload=False)
