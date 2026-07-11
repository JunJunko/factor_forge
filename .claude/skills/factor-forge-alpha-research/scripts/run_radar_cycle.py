from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_TEMPLATES = (
    "configs/radar/price_drop_without_volume_confirmation_v1.yaml",
    "configs/radar/volume_surge_without_price_impact_v1.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Factor Forge label-free Radar templates as one deterministic cycle."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-version", default="latest")
    parser.add_argument("--as-of", help="YYYYMMDD; defaults to the selected data version end")
    parser.add_argument("--template", action="append", dest="templates")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_workspace(workspace: Path) -> Path:
    root = workspace.resolve()
    required = [root / "pyproject.toml", root / "src" / "factor_forge", root / "configs" / "radar"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"not a Factor Forge workspace; missing: {missing}")
    return root


def command_for(args: argparse.Namespace, root: Path, template: str) -> list[str]:
    command = [
        args.python,
        "-m",
        "factor_forge.cli",
        "radar",
        "scan",
        "--template",
        str((root / template).resolve()),
        "--data-version",
        args.data_version,
    ]
    if args.as_of:
        command.extend(["--as-of", args.as_of])
    return command


def main() -> int:
    args = parse_args()
    root = validate_workspace(args.workspace)
    templates = tuple(args.templates or DEFAULT_TEMPLATES)
    commands = [command_for(args, root, template) for template in templates]
    if args.dry_run:
        print(json.dumps({"workspace": str(root), "commands": commands}, ensure_ascii=False, indent=2))
        return 0

    cycle_root = root / "artifacts" / "radar_cycles"
    cycle_root.mkdir(parents=True, exist_ok=True)
    lock_path = cycle_root / ".cycle.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SystemExit(f"another Radar cycle is active or left a stale lock: {lock_path}") from exc
    started = datetime.now(timezone.utc)
    try:
        os.write(descriptor, f"pid={os.getpid()} started={started.isoformat()}".encode("utf-8"))
        os.close(descriptor)
        results = []
        for template, command in zip(templates, commands, strict=True):
            completed = subprocess.run(
                command, cwd=root, text=True, capture_output=True, encoding="utf-8"
            )
            record = {
                "template": template,
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
            results.append(record)
            if completed.returncode != 0:
                break
        finished = datetime.now(timezone.utc)
        identity = hashlib.sha256(
            json.dumps(
                {"started": started.isoformat(), "templates": templates, "data_version": args.data_version},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        summary = {
            "cycle_id": f"radar_cycle_{started:%Y%m%dT%H%M%SZ}_{identity}",
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "workspace": str(root),
            "data_version": args.data_version,
            "as_of": args.as_of,
            "status": "COMPLETED" if all(item["returncode"] == 0 for item in results) else "FAILED",
            "results": results,
        }
        output = cycle_root / f"{summary['cycle_id']}.json"
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({**summary, "summary_path": str(output.resolve())}, ensure_ascii=False, indent=2))
        return 0 if summary["status"] == "COMPLETED" else 1
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
