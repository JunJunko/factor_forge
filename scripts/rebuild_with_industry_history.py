from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.ingestion import TushareIngestor
from factor_forge.data.panel import DailyPanelBuilder
from factor_forge.data.repository import DataVersionRepository
from factor_forge.data.tushare_provider import TushareProvider


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild a data version after refreshing point-in-time industry history"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-version", required=True)
    args = parser.parse_args()

    project = load_project(args.config)
    raw_dir = project.paths.data_root / "versions" / args.base_version / "raw" / "tushare"
    if not raw_dir.is_dir():
        raise FileNotFoundError(raw_dir)

    datasets = {
        path.stem: pd.read_parquet(path)
        for path in sorted(raw_dir.glob("*.parquet"))
    }
    ingestor = TushareIngestor(project, TushareProvider())
    industry = ingestor._fetch_industry_reference()
    datasets.update(industry)
    members = industry["industry_membership"]
    print(
        f"industry rows={len(members)} stocks={members['ts_code'].nunique()} "
        f"historical={members['out_date'].notna().sum()}"
    )

    repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    with pd.option_context("mode.copy_on_write", True):
        panel = DailyPanelBuilder(project).build(datasets)
        version = repository.publish(panel, datasets)
    ingestor._persist_dimensions(datasets, version)
    print(version)


if __name__ == "__main__":
    main()
