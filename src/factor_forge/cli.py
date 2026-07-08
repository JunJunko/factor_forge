from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from factor_forge.config import load_factor, load_project
from factor_forge.data.ingestion import TushareIngestor
from factor_forge.data.metadata import MetadataStore
from factor_forge.data.repository import DataVersionRepository
from factor_forge.data.tushare_provider import TushareProvider
from factor_forge.experiments import ExperimentRunner
from factor_forge.experiments.artifacts import json_default


app = typer.Typer(help="Reusable A-share daily factor research platform")
data_app = typer.Typer(help="Build and inspect immutable local data versions")
factor_app = typer.Typer(help="Validate declarative factor definitions")
experiment_app = typer.Typer(help="Run staged factor experiments")
ml_app = typer.Typer(help="Run cross-sectional LightGBM experiments")
app.add_typer(data_app, name="data")
app.add_typer(factor_app, name="factor")
app.add_typer(experiment_app, name="experiment")
app.add_typer(ml_app, name="ml")


@ml_app.command("run")
def ml_run(path: Path):
    from factor_forge.ml import MLExperimentRunner
    result = MLExperimentRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("value-run")
def value_ml_run(path: Path):
    """Train the point-in-time 5/10/20-day value-recovery ensemble."""
    from factor_forge.ml.value_regression import ValueRegressionRunner
    result = ValueRegressionRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("timing-build")
def timing_build(path: Path):
    """Build daily market-timing ML features from local Tushare-style tables."""
    from factor_forge.timing.runner import TimingDatasetBuildRunner
    result = TimingDatasetBuildRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("timing-regime")
def timing_regime(path: Path):
    """Run timing HMM/GMM regime diagnostics and regime-factor interaction model."""
    from factor_forge.timing.regime import TimingRegimeRunner
    result = TimingRegimeRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("timing-regime-grid")
def timing_regime_grid(path: Path):
    """Run timing HMM/GMM regime parameter stability grid."""
    from factor_forge.timing.regime import TimingRegimeGridRunner
    result = TimingRegimeGridRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("timing-stable-factors")
def timing_stable_factors(path: Path):
    """Select stable timing factors across fixed-regime random seeds."""
    from factor_forge.timing.stable_factors import StableFactorSelectionRunner
    result = StableFactorSelectionRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("timing-position-model")
def timing_position_model(path: Path):
    """Train a regime-aware timing position model and backtest daily exposure."""
    from factor_forge.timing.position_model import TimingPositionModelRunner
    result = TimingPositionModelRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("value-diagnostics")
def value_diagnostics(path: Path):
    """Run portfolio, decile and price-vs-full diagnostics for a value model."""
    from factor_forge.ml.value_diagnostics import ValueDiagnosticsRunner
    result = ValueDiagnosticsRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("value-fixed-validation")
def value_fixed_validation(path: Path):
    """Run the frozen full-model Top5/10-day validation protocol."""
    from factor_forge.ml.fixed_validation import FixedValidationRunner
    result = FixedValidationRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("value-hmm-regime")
def value_hmm_regime(path: Path):
    """Validate leakage-safe HMM market regimes on the frozen value model."""
    from factor_forge.ml.value_hmm_regime import ValueHMMRegimeRunner
    result = ValueHMMRegimeRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("value-style-attribution")
def value_style_attribution(path: Path):
    """Decompose value strategy excess return into style-factor exposures + residual alpha."""
    from factor_forge.ml.value_style_attribution import StyleAttributionRunner
    result = StyleAttributionRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("supply-run")
def supply_run(path: Path):
    """Train + backtest the low-volume-rise supply-contraction factor via Qlib (A/B ablation)."""
    from factor_forge.ml.supply_runner import SupplyContractionRunner
    result = SupplyContractionRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@data_app.command("init")
def data_init(config: Path = typer.Option(Path("configs/project.yaml"))):
    project = load_project(config)
    MetadataStore(project.paths.metadata_db).initialize()
    typer.echo(f"Initialized {project.paths.metadata_db}")


@data_app.command("check-permissions")
def check_permissions(config: Path = typer.Option(Path("configs/project.yaml"))):
    project = load_project(config)
    report = TushareIngestor(project, TushareProvider()).check_permissions()
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2, default=json_default))


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


@data_app.command("ingest-fundamentals")
def ingest_fundamentals(
    config: Path = typer.Option(Path("configs/project.yaml")),
    data_version: str = typer.Option("latest"),
    start_year: int = typer.Option(2014),
    output: Path = typer.Option(Path("data/fundamentals_pit.parquet")),
):
    """Fetch Tushare financial statements and build conservative PIT snapshots."""
    from factor_forge.data.fundamentals import TushareFundamentalIngestor

    project = load_project(config)
    repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, manifest = repository.load_manifest(data_version)
    calendar = repository.load_raw_dataset(version, "trade_calendar")
    securities = repository.load_raw_dataset(version, "stock_basic")
    if calendar is None or securities is None:
        raise typer.BadParameter("data version has no trade_calendar or stock_basic raw dataset")
    open_dates = calendar.loc[pd.to_numeric(calendar["is_open"], errors="coerce").eq(1), "cal_date"]
    codes = set(securities["ts_code"].dropna().astype(str))

    def progress(done: int, total: int, endpoint: str, period: str, rows: int) -> None:
        typer.echo(f"fundamentals {done}/{total} {endpoint} period={period} rows={rows}")

    result = TushareFundamentalIngestor(
        TushareProvider(), project.paths.data_root, progress=progress
    ).ingest(
        start_year=start_year,
        end_date=manifest["end_date"],
        trading_dates=open_dates,
        securities=codes,
        output_path=output,
    )
    typer.echo(json.dumps(result.__dict__, ensure_ascii=False, indent=2, default=json_default))


@data_app.command("timing-ingest")
def timing_ingest(
    start: str = typer.Option(..., help="YYYYMMDD"),
    end: str = typer.Option(..., help="YYYYMMDD"),
    index_code: str = typer.Option("000300.SH", help="Timing target index"),
    config: Path = typer.Option(Path("configs/project.yaml")),
    data_version: str = typer.Option("latest", help="Stock panel version for market breadth"),
    output_dir: Path = typer.Option(Path("data/timing")),
    overwrite: bool = typer.Option(False, help="Overwrite existing timing parquet files"),
    include_options: bool = typer.Option(True, help="Fetch opt_basic/opt_daily"),
    include_futures: bool = typer.Option(True, help="Fetch fut_basic/fut_daily/fut_holding"),
    include_moneyflow: bool = typer.Option(True, help="Fetch moneyflow_mkt_dc when permitted"),
):
    """Fetch Tushare-style raw tables for the timing factor library."""
    from factor_forge.timing.ingestion import TushareTimingIngestor

    def progress(stage: str, done: int, total: int, detail: str) -> None:
        typer.echo(f"timing-ingest {stage} {done}/{total}: {detail}")

    result = TushareTimingIngestor(
        TushareProvider(), output_dir=output_dir, progress=progress
    ).ingest(
        start_date=start,
        end_date=end,
        index_code=index_code,
        project_config=config,
        data_version=data_version,
        include_options=include_options,
        include_futures=include_futures,
        include_moneyflow=include_moneyflow,
        overwrite=overwrite,
    )
    typer.echo(json.dumps(result.__dict__, ensure_ascii=False, indent=2, default=json_default))


@factor_app.command("validate")
def factor_validate(path: Path):
    spec = load_factor(path)
    typer.echo(f"VALID: {spec.factor.name} (contract v{spec.version})")


@experiment_app.command("run")
def experiment_run(path: Path):
    result = ExperimentRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@experiment_app.command("run-with-timing")
def experiment_run_with_timing(
    path: Path,
    timing_daily: Path = typer.Option(..., "--timing-daily", help="timing_position_daily.csv, parquet, or its run directory"),
    factor: Path | None = typer.Option(None, "--factor", help="Override factor YAML, same as root run --factor"),
    position_column: str = typer.Option("target_position", "--position-column", help="Column used as entry cash multiplier"),
    date_column: str = typer.Option("trade_date", "--date-column", help="Trading date column in timing daily file"),
):
    multiplier = _load_position_multiplier(timing_daily, position_column, date_column)
    result = ExperimentRunner().run(
        path,
        factor_path=factor,
        position_multiplier=multiplier,
        position_multiplier_source=str(timing_daily),
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


def _load_position_multiplier(path: Path, position_column: str, date_column: str) -> pd.Series:
    source = Path(path)
    if source.is_dir():
        source = source / "timing_position_daily.csv"
    if not source.exists():
        raise typer.BadParameter(f"timing daily file does not exist: {source}")
    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source)
    else:
        frame = pd.read_csv(source)
    missing = {date_column, position_column} - set(frame.columns)
    if missing:
        raise typer.BadParameter(
            f"timing daily file missing columns: {', '.join(sorted(missing))}; "
            f"available={', '.join(map(str, frame.columns))}"
        )
    multiplier = pd.Series(
        pd.to_numeric(frame[position_column], errors="coerce").to_numpy(),
        index=pd.to_datetime(frame[date_column]),
        name="position_multiplier",
    )
    multiplier = multiplier.dropna().sort_index().clip(0.0, 1.0)
    if multiplier.empty:
        raise typer.BadParameter("position multiplier is empty after parsing")
    return multiplier


@app.command("run")
def run(
    factor: Path = typer.Option(..., "--factor", help="Stock factor YAML"),
    experiment: Path = typer.Option(..., "--experiment", help="Experiment YAML"),
):
    """Run with the stable two-YAML user protocol."""
    result = ExperimentRunner().run(experiment, factor_path=factor)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    app()
