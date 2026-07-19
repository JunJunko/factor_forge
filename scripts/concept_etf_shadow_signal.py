from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from factor_forge.research.concept_etf_shadow import latest_target_table


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    panel_path = resolve_panel(Path(args.signal_panel))
    panel = pd.read_parquet(panel_path)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    as_of = pd.Timestamp(args.as_of) if args.as_of else panel["trade_date"].max()
    if not args.preview and as_of.weekday() != 4:
        raise RuntimeError(
            "committed weekly shadow signals require a Friday signal date; use --preview for diagnostics "
            "or pass the holiday-shortened week explicitly after adding a calendar exception"
        )
    targets = latest_target_table(
        panel, as_of=as_of, top_n=int(config["signal"]["top_n"]), universe=config["universe"],
    )
    status = "PREVIEW_NOT_FOR_EXECUTION" if args.preview else "COMMITTED_SHADOW_RECORD"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_root) / f"shadow_signal_{as_of:%Y%m%d}_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    targets.to_csv(output / "targets.csv", index=False, encoding="utf-8-sig")
    targets.to_parquet(output / "targets.parquet", index=False)
    manifest = {
        "status": status, "created_at": datetime.now(timezone.utc).isoformat(),
        "signal_date": str(as_of.date()), "execute_at": "next_trade_open",
        "config": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_panel": str(panel_path.resolve()), "source_panel_last_date": str(panel["trade_date"].max().date()),
        "portfolios": config["portfolios"], "target_rows": len(targets),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(output), **manifest}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an immutable concept ETF shadow signal record")
    parser.add_argument("--config", default="configs/research/concept_etf_forward_v1.yaml")
    parser.add_argument("--signal-panel", default="artifacts/concept_etf_rotation")
    parser.add_argument("--as-of")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--output-root", default="artifacts/concept_etf_shadow_signals")
    return parser.parse_args()


def resolve_panel(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = list(path.glob("concept_etf_rotation_*/etf_signal_panel.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no ETF signal panel below {path}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


if __name__ == "__main__":
    main()
