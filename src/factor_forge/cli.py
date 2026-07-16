from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
import yaml

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
research_app = typer.Typer(help="Track research lineage, budgets, decisions, and run artifacts")
feature_registry_app = typer.Typer(help="Validate auditable, non-executable research feature entries")
radar_app = typer.Typer(help="Scan label-free market observations with frozen templates")
event_study_app = typer.Typer(help="Run matched event studies on frozen observations")
drift_app = typer.Typer(help="Scan recent market relation drift with frozen templates")
anomaly_scan_app = typer.Typer(help="Run the frozen latest-market anomaly scan bundle")
app.add_typer(data_app, name="data")
app.add_typer(factor_app, name="factor")
app.add_typer(experiment_app, name="experiment")
app.add_typer(ml_app, name="ml")
app.add_typer(research_app, name="research")
app.add_typer(feature_registry_app, name="feature-registry")
app.add_typer(radar_app, name="radar")
app.add_typer(event_study_app, name="event-study")
app.add_typer(drift_app, name="drift")
app.add_typer(anomaly_scan_app, name="anomaly-scan")


@anomaly_scan_app.command("latest")
def anomaly_scan_latest(
    config: Path = typer.Option(
        Path("configs/radar/latest_market_scan_v1.yaml"), "--config"
    ),
    data_version: str = typer.Option("latest", "--data-version"),
    as_of: str | None = typer.Option(None, "--as-of", help="YYYYMMDD; defaults to data end"),
    sync: bool = typer.Option(
        True, "--sync/--no-sync",
        help="Synchronize missing completed trading days before resolving latest",
    ),
):
    """Discover and run all configured event and relationship-drift templates."""
    from factor_forge.radar.batch import MarketAnomalyScanRunner

    _echo_model(MarketAnomalyScanRunner().run(
        config, data_version=data_version, as_of_date=as_of, sync=sync,
    ))


@drift_app.command("validate-template")
def drift_validate_template(path: Path):
    from factor_forge.radar.drift_templates import load_drift_template

    template = load_drift_template(path)
    typer.echo(
        f"VALID: {template.id} kind={template.kind} definition_hash={template.definition_hash()}"
    )


@drift_app.command("scan")
def drift_scan(
    template: Path = typer.Option(..., "--template"),
    project_config: Path = typer.Option(Path("configs/project.yaml"), "--project-config"),
    data_version: str = typer.Option("latest", "--data-version"),
    as_of: str | None = typer.Option(None, "--as-of"),
    output_root: Path = typer.Option(Path("artifacts/radar_drifts"), "--output-root"),
    research_db: Path | None = typer.Option(None, "--research-db"),
):
    from factor_forge.radar.drift import RelationDriftRunner

    _echo_model(RelationDriftRunner().run(
        template, project_config=project_config, data_version=data_version,
        as_of_date=as_of, output_root=output_root, research_db=research_db,
    ))


@event_study_app.command("validate")
def event_study_validate(path: Path):
    from factor_forge.event_study import load_event_study_config

    config = load_event_study_config(path)
    typer.echo(
        f"VALID: {config.name} primary={config.inference.primary_stage}/"
        f"{config.inference.primary_horizon}D"
    )


@event_study_app.command("run")
def event_study_run(path: Path):
    from factor_forge.event_study import EventStudyRunner

    _echo_model(EventStudyRunner().run(path))


@radar_app.command("validate-template")
def radar_validate_template(path: Path):
    from factor_forge.radar import load_radar_template

    template = load_radar_template(path)
    typer.echo(
        f"VALID: {template.id} kind={template.kind} definition_hash={template.definition_hash()}"
    )


@radar_app.command("scan")
def radar_scan(
    template: Path = typer.Option(..., "--template"),
    project_config: Path = typer.Option(Path("configs/project.yaml"), "--project-config"),
    data_version: str = typer.Option("latest", "--data-version"),
    as_of: str | None = typer.Option(None, "--as-of", help="YYYYMMDD; defaults to data end"),
    output_root: Path = typer.Option(Path("artifacts/radar_observations"), "--output-root"),
    research_db: Path | None = typer.Option(None, "--research-db"),
):
    from factor_forge.radar import RadarRunner

    result = RadarRunner().run(
        template, project_config=project_config, data_version=data_version,
        as_of_date=as_of, output_root=output_root, research_db=research_db,
    )
    _echo_model(result)


def _research_store(config: Path, db: Path | None):
    from factor_forge.research_control import ResearchControlStore

    path = db if db is not None else load_project(config).paths.data_root / "research.sqlite3"
    store = ResearchControlStore(path)
    store.initialize()
    return store


def _echo_model(value) -> None:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))


@research_app.command("init")
def research_init(
    config: Path = typer.Option(Path("configs/project.yaml")),
    db: Path | None = typer.Option(None, help="Override research SQLite path"),
):
    store = _research_store(config, db)
    typer.echo(f"Initialized {store.path}")


@feature_registry_app.command("validate")
def feature_registry_validate(path: Path):
    """Validate one entry or a registry index without executing a feature."""
    from factor_forge.feature_registry import load_feature_entry, load_feature_registry

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if raw.get("kind") == "feature_registry":
        entries = load_feature_registry(path)
        typer.echo(f"VALID: {path} entries={len(entries)}")
    else:
        entry = load_feature_entry(path)
        typer.echo(f"VALID: {entry.id} lifecycle={entry.lifecycle}")


@research_app.command("index-artifacts")
def research_index_artifacts(
    artifacts_root: Path = typer.Option(Path("artifacts")),
    config: Path = typer.Option(Path("configs/project.yaml")),
    db: Path | None = typer.Option(None, help="Override research SQLite path"),
):
    from factor_forge.research_control import ArtifactIndexer

    result = ArtifactIndexer(_research_store(config, db), artifacts_root).index()
    _echo_model(result)


@research_app.command("artifact-summary")
def research_artifact_summary(
    config: Path = typer.Option(Path("configs/project.yaml")),
    db: Path | None = typer.Option(None, help="Override research SQLite path"),
):
    _echo_model(_research_store(config, db).artifact_summary())


@research_app.command("idea-create")
def research_idea_create(
    title: str = typer.Option(...),
    thesis: str = typer.Option(...),
    family_id: str = typer.Option(...),
    target_horizon: int | None = typer.Option(None),
    idea_id: str | None = typer.Option(None),
    max_trials: int = typer.Option(5),
    max_revisions: int = typer.Option(2),
    max_validation_peeks: int = typer.Option(2),
    config: Path = typer.Option(Path("configs/project.yaml")),
    db: Path | None = typer.Option(None),
):
    idea = _research_store(config, db).create_idea(
        title=title, thesis=thesis, family_id=family_id,
        target_horizon=target_horizon, idea_id=idea_id,
        max_trials=max_trials, max_revisions=max_revisions,
        max_validation_peeks=max_validation_peeks,
    )
    _echo_model(idea)


@research_app.command("hypothesis-add")
def research_hypothesis_add(
    idea_id: str,
    statement: str = typer.Option(...),
    alternative_to: str | None = typer.Option(None),
    hypothesis_id: str | None = typer.Option(None),
    config: Path = typer.Option(Path("configs/project.yaml")),
    db: Path | None = typer.Option(None),
):
    hypothesis = _research_store(config, db).add_hypothesis(
        idea_id, statement, alternative_to=alternative_to, hypothesis_id=hypothesis_id
    )
    _echo_model(hypothesis)


@research_app.command("plan-create")
def research_plan_create(
    idea_id: str,
    name: str = typer.Option(...),
    primary_metric: str = typer.Option(...),
    hypothesis_id: str | None = typer.Option(None),
    config_path: Path | None = typer.Option(None),
    plan_id: str | None = typer.Option(None),
    project_config: Path = typer.Option(Path("configs/project.yaml"), "--project-config"),
    db: Path | None = typer.Option(None),
):
    plan = _research_store(project_config, db).create_plan(
        idea_id, name, primary_metric, hypothesis_id=hypothesis_id,
        config_path=config_path, plan_id=plan_id,
    )
    _echo_model(plan)


@research_app.command("compare-daily-ic")
def research_compare_daily_ic(
    plan: Path = typer.Option(..., "--plan", help="Frozen alpha research plan YAML"),
    market_run: Path = typer.Option(..., "--market-run", help="Market trial run directory or daily IC artifact"),
    industry_run: Path = typer.Option(..., "--industry-run", help="Industry trial run directory or daily IC artifact"),
    output_root: Path = typer.Option(Path("artifacts/research_comparisons"), "--output-root"),
):
    """Materialize the frozen cross-Trial composition comparison after T2 and T3."""
    from factor_forge.research.comparison import create_composition_comparison

    _echo_model(create_composition_comparison(
        plan, market_run, industry_run, output_root=output_root
    ))


@research_app.command("trial-record")
def research_trial_record(
    plan_id: str,
    data_role: str = typer.Option(..., help="discovery|validation|forward"),
    status: str = typer.Option(..., help="queued|running|success|failed|cancelled"),
    external_run_id: str | None = typer.Option(None),
    artifact_path: Path | None = typer.Option(None),
    validation_peek: bool = typer.Option(False),
    revision: bool = typer.Option(False),
    trial_id: str | None = typer.Option(None),
    config: Path = typer.Option(Path("configs/project.yaml")),
    db: Path | None = typer.Option(None),
):
    from factor_forge.research_control.models import DataRole, TrialStatus

    trial = _research_store(config, db).record_trial(
        plan_id=plan_id,
        data_role=DataRole(data_role.lower()),
        status=TrialStatus(status.upper()),
        external_run_id=external_run_id,
        artifact_path=artifact_path,
        validation_peek=validation_peek,
        revision=revision,
        trial_id=trial_id,
    )
    _echo_model(trial)


@research_app.command("decision-save")
def research_decision_save(
    trial_id: str,
    action: str = typer.Option(...),
    reason: str = typer.Option(...),
    decided_by: str = typer.Option(...),
    decision_id: str | None = typer.Option(None),
    config: Path = typer.Option(Path("configs/project.yaml")),
    db: Path | None = typer.Option(None),
):
    from factor_forge.research_control.models import DecisionAction

    decision = _research_store(config, db).save_decision(
        trial_id, DecisionAction(action.lower()), reason, decided_by, decision_id=decision_id
    )
    _echo_model(decision)


@research_app.command("idea-show")
def research_idea_show(
    idea_id: str,
    config: Path = typer.Option(Path("configs/project.yaml")),
    db: Path | None = typer.Option(None),
):
    _echo_model(_research_store(config, db).idea_summary(idea_id))


@research_app.command("idea-status")
def research_idea_status(
    idea_id: str,
    status: str = typer.Option(..., help="draft|active|paused|closed"),
    config: Path = typer.Option(Path("configs/project.yaml")),
    db: Path | None = typer.Option(None),
):
    from factor_forge.research_control.models import IdeaStatus

    _echo_model(_research_store(config, db).set_idea_status(idea_id, IdeaStatus(status.upper())))


@research_app.command("trial-status")
def research_trial_status(
    trial_id: str,
    status: str = typer.Option(..., help="queued|running|success|failed|cancelled"),
    config: Path = typer.Option(Path("configs/project.yaml")),
    db: Path | None = typer.Option(None),
):
    from factor_forge.research_control.models import TrialStatus

    _echo_model(_research_store(config, db).set_trial_status(trial_id, TrialStatus(status.upper())))


@research_app.command("plan-status")
def research_plan_status(
    plan_id: str,
    status: str = typer.Option(..., help="cancelled"),
    config: Path = typer.Option(Path("configs/project.yaml")),
    db: Path | None = typer.Option(None),
):
    from factor_forge.research_control.models import PlanStatus

    _echo_model(_research_store(config, db).set_plan_status(plan_id, PlanStatus(status.upper())))


@research_app.command("sealed-access-record")
def research_sealed_access_record(
    idea_id: str,
    requested_by: str = typer.Option(...),
    approved_by: str = typer.Option(...),
    reason: str = typer.Option(...),
    data_start: str = typer.Option(..., help="YYYYMMDD"),
    data_end: str = typer.Option(..., help="YYYYMMDD"),
    artifact_path: Path | None = typer.Option(None),
    config: Path = typer.Option(Path("configs/project.yaml")),
    db: Path | None = typer.Option(None),
):
    audit = _research_store(config, db).record_sealed_access(
        idea_id, requested_by, approved_by, reason, data_start, data_end, artifact_path
    )
    _echo_model(audit)


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


@ml_app.command("mamba-state-run")
def mamba_state_run(path: Path):
    """Run the frozen Raw / State / Raw+State cross-sectional pilot."""
    from factor_forge.ml.mamba_state_runner import MambaStateLightGBMRunner

    result = MambaStateLightGBMRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("mamba-anomaly-demo")
def mamba_anomaly_demo(
    scan_summary: Path = typer.Option(..., "--scan-summary"),
    project_config: Path = typer.Option(Path("configs/project.yaml"), "--project-config"),
    output_root: Path = typer.Option(Path("artifacts/mamba_anomaly_demos"), "--output-root"),
):
    """Run a bounded CPU state-ranking demo driven by a frozen anomaly scan."""
    from factor_forge.ml.mamba_anomaly_demo import TodayAnomalyStateDemoRunner

    result = TodayAnomalyStateDemoRunner().run(
        scan_summary, project_config=project_config, output_root=output_root,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("event-episode-run")
def event_episode_run(path: Path):
    """Run the frozen single-template Event Episode E0-E3 ranker."""
    from factor_forge.ml.event_episode_runner import EventEpisodeRankerRunner

    result = EventEpisodeRankerRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("post-impulse-run")
def post_impulse_run(path: Path):
    """Build PIT post-impulse path features and run fixed ML block ablations."""
    from factor_forge.ml.post_impulse_runner import PostImpulseMLRunner

    result = PostImpulseMLRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("post-impulse-m3")
def post_impulse_m3(path: Path):
    """Diagnose fixed M3 absorption sub-blocks against the common M2 sample."""
    from factor_forge.ml.post_impulse_m3 import PostImpulseM3DiagnosticRunner

    result = PostImpulseM3DiagnosticRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("post-impulse-m3-walkforward")
def post_impulse_m3_walkforward(path: Path):
    """Run the frozen four-fold OOF minimal-M3 classifier comparison."""
    from factor_forge.ml.post_impulse_m3_walkforward import PostImpulseM3WalkForwardRunner

    result = PostImpulseM3WalkForwardRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("post-impulse-m2-backtest")
def post_impulse_m2_backtest(path: Path):
    """Run the frozen M1/M2 OOF Ridge comparison and executable backtest."""
    from factor_forge.ml.post_impulse_m2_walkforward import PostImpulseM2WalkForwardRunner

    result = PostImpulseM2WalkForwardRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("post-impulse-m21")
def post_impulse_m21(path: Path):
    """Run the compressed-pressure reranker and corrected cohort backtest."""
    from factor_forge.ml.post_impulse_m21 import PostImpulseM21Runner

    result = PostImpulseM21Runner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("post-impulse-m2-path")
def post_impulse_m2_path(path: Path):
    """Trace OOF Top-5 pressure reranking through fixed executable exit horizons."""
    from factor_forge.ml.post_impulse_path import PostImpulseM2PathRunner

    result = PostImpulseM2PathRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("event-episode-oos")
def event_episode_oos(
    run_dir: Path = typer.Option(..., "--run-dir"),
    project_config: Path = typer.Option(Path("configs/project.yaml"), "--project-config"),
    top_n: int = typer.Option(10, "--top-n"),
    holding_days: int = typer.Option(5, "--holding-days"),
    cost_bps: float = typer.Option(15.0, "--cost-bps"),
):
    """Backtest held-out Event Episode predictions with T+1-open execution."""
    from factor_forge.ml.event_episode_oos import EventEpisodeOOSBacktestRunner

    result = EventEpisodeOOSBacktestRunner().run(
        run_dir, project_config=project_config, top_n=top_n,
        holding_days=holding_days, cost_bps=cost_bps,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("recent-anomaly-structure-run")
def recent_anomaly_structure_run(path: Path):
    """Run the frozen PIT rolling-OOS recent anomaly structure ranker."""
    from factor_forge.ml.recent_anomaly_runner import RecentAnomalyStructureRunner

    result = RecentAnomalyStructureRunner().run(path)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


@ml_app.command("event-factor-sensitivity-run")
def event_factor_sensitivity_run(path: Path):
    """Run Event-Mamba named sensitivities with strict chronological OOF stacking."""
    from factor_forge.ml.event_factor_sensitivity_runner import EventFactorSensitivityRunner

    result = EventFactorSensitivityRunner().run(path)
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
