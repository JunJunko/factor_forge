from __future__ import annotations

import json
from pathlib import Path

import typer

from factor_forge.config import load_factor, load_project
from factor_forge.data.ingestion import TushareIngestor
from factor_forge.data.metadata import MetadataStore
from factor_forge.data.tushare_provider import TushareProvider
from factor_forge.experiments import ExperimentRunner


app = typer.Typer(help="Reusable A-share daily factor research platform")
data_app = typer.Typer(help="Build and inspect immutable local data versions")
factor_app = typer.Typer(help="Validate declarative factor definitions")
experiment_app = typer.Typer(help="Run staged factor experiments")
app.add_typer(data_app, name="data")
app.add_typer(factor_app, name="factor")
app.add_typer(experiment_app, name="experiment")


@data_app.command("init")
def data_init(config: Path = typer.Option(Path("configs/project.yaml"))):
    project = load_project(config)
    MetadataStore(project.paths.metadata_db).initialize()
    typer.echo(f"Initialized {project.paths.metadata_db}")


@data_app.command("check-permissions")
def check_permissions(config: Path = typer.Option(Path("configs/project.yaml"))):
    project = load_project(config)
    report = TushareIngestor(project, TushareProvider()).check_permissions()
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@data_app.command("ingest")
def ingest(
    start: str = typer.Option(..., help="YYYYMMDD"),
    end: str = typer.Option(..., help="YYYYMMDD"),
    config: Path = typer.Option(Path("configs/project.yaml")),
):
    project = load_project(config)
    def progress(done: int, total: int, date: str) -> None:
        typer.echo(f"fetch {done}/{total} ({done / total:.1%}) trade_date={date}")

    version = TushareIngestor(project, TushareProvider(), progress=progress).ingest(start, end)
    typer.echo(version)


@factor_app.command("validate")
def factor_validate(path: Path):
    spec = load_factor(path)
    typer.echo(f"VALID: {spec.factor.name} (contract v{spec.version})")


@experiment_app.command("run")
def experiment_run(path: Path):
    result = ExperimentRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
