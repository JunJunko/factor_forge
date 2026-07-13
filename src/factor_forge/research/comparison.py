"""Immutable, plan-driven comparison of compact daily L1 IC artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from factor_forge.evaluation import compare_daily_rank_ic


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _daily_ic_path(run_or_artifact: Path) -> Path:
    path = run_or_artifact / "l1_daily_rank_ic.parquet" if run_or_artifact.is_dir() else run_or_artifact
    if path.name != "l1_daily_rank_ic.parquet" or not path.is_file():
        raise ValueError(f"Expected an existing l1_daily_rank_ic.parquet artifact, got: {run_or_artifact}")
    return path


def _select_daily_ic(frame: pd.DataFrame, selector: dict[str, Any], label: str) -> pd.DataFrame:
    required = {"trade_date", "target", "variant", "universe", "horizon", "rank_ic"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} artifact is missing columns: {sorted(missing)}")
    selected = frame.copy()
    for field in ("target", "variant", "universe", "horizon"):
        if field not in selector:
            raise ValueError(f"composition selector is missing {field}")
        selected = selected.loc[selected[field] == selector[field]]
    if selected.empty:
        raise ValueError(f"{label} artifact has no rows matching the frozen selector")
    if selected.duplicated("trade_date").any():
        raise ValueError(f"{label} artifact has duplicate dates for the frozen selector")
    return selected[["trade_date", "rank_ic"]].sort_values("trade_date").reset_index(drop=True)


def create_composition_comparison(
    plan_path: Path,
    market_run_or_artifact: Path,
    industry_run_or_artifact: Path,
    *,
    output_root: Path = Path("artifacts/research_comparisons"),
) -> dict[str, Any]:
    """Create one immutable comparison artifact from two completed L1 runs.

    The command consumes only compact daily IC outputs.  It does not access market
    tables and it refuses ambiguous or pre-existing artifact paths.
    """
    if not plan_path.is_file():
        raise ValueError(f"Plan does not exist: {plan_path}")
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    comparison = plan.get("composition_comparison") or {}
    selector = (comparison.get("inputs") or {}).get("selector")
    thresholds = comparison.get("thresholds") or {}
    if not isinstance(selector, dict) or not isinstance(thresholds, dict):
        raise ValueError("Plan must define composition_comparison inputs.selector and thresholds")

    market_path = _daily_ic_path(market_run_or_artifact)
    industry_path = _daily_ic_path(industry_run_or_artifact)
    market = _select_daily_ic(pd.read_parquet(market_path), selector, "market")
    industry = _select_daily_ic(pd.read_parquet(industry_path), selector, "industry")
    result = compare_daily_rank_ic(
        market,
        industry,
        horizon=int(selector["horizon"]),
        min_overlap_days=int(thresholds["min_overlap_days"]),
        min_retention_ratio=float(thresholds["min_retention_ratio"]),
        max_p_value=float(thresholds["max_p_value"]),
    )
    artifact_key = hashlib.sha256(
        (f"{_sha256(plan_path)}:{_sha256(market_path)}:{_sha256(industry_path)}").encode()
    ).hexdigest()[:16]
    artifact_dir = output_root / f"composition_comparison_{artifact_key}"
    if artifact_dir.exists():
        raise FileExistsError(f"Immutable comparison artifact already exists: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=False)
    aligned = market.rename(columns={"rank_ic": "market_rank_ic"}).merge(
        industry.rename(columns={"rank_ic": "industry_rank_ic"}), on="trade_date", how="inner", validate="one_to_one"
    ).dropna().sort_values("trade_date")
    aligned.to_parquet(artifact_dir / "aligned_daily_rank_ic.parquet", index=False)
    payload = {
        "kind": "composition_daily_rank_ic_comparison",
        "plan_id": plan.get("plan_id"),
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": _sha256(plan_path),
        "inputs": {
            "market_daily_ic": {"path": str(market_path.resolve()), "sha256": _sha256(market_path)},
            "industry_daily_ic": {"path": str(industry_path.resolve()), "sha256": _sha256(industry_path)},
            "selector": selector,
        },
        "result": result,
    }
    (artifact_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {**payload, "artifact_path": str(artifact_dir.resolve())}
