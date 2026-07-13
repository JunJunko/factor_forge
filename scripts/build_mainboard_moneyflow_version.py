from __future__ import annotations

import argparse
import json

from factor_forge.config import load_project
from factor_forge.data.moneyflow_enrichment import MainboardMoneyflowEnricher
from factor_forge.data.tushare_provider import TushareProvider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-project", default="configs/project.yaml")
    parser.add_argument("--target-project", default="configs/project_mainboard_moneyflow.yaml")
    parser.add_argument("--base-version", default="latest")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    def progress(position: int, total: int, date: str) -> None:
        print(f"moneyflow {position}/{total} {date}", flush=True)

    result = MainboardMoneyflowEnricher(
        load_project(args.base_project), load_project(args.target_project), TushareProvider()
    ).run(
        base_version=args.base_version,
        start_date=args.start,
        end_date=args.end,
        publish=args.publish,
        min_coverage=args.min_coverage,
        progress=progress,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
