from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Annotated, Any, NoReturn

import duckdb
import pandas as pd
import typer

from alpha_lab.baseline.pipeline import run_baseline
from alpha_lab.data.normalize import to_qlib_instrument
from alpha_lab.data.pipeline import run_ingestion
from alpha_lab.data.qlib_export import export_qlib
from alpha_lab.data.quality import build_quality_report
from alpha_lab.database.catalog import (
    check_database,
    initialize_database,
    sync_exposure_snapshot,
    sync_repository_metadata,
    sync_research_snapshot,
)
from alpha_lab.evaluation.pipeline import evaluate_factor
from alpha_lab.factors.registry import FactorRegistry
from alpha_lab.mining.pipeline import (
    initialize_mining_run,
    render_mining_report,
    run_mining_loop,
    run_mining_round,
)
from alpha_lab.research_data.pipeline import (
    probe_research_data,
    run_research_data_pipeline,
)
from alpha_lab.research_data.snapshot import validate_research_snapshot
from alpha_lab.research_data.universe import universe_as_of
from alpha_lab.robustness.approval import (
    approve_test_request,
    create_test_request,
)
from alpha_lab.robustness.exposure_data import probe_exposure_capabilities
from alpha_lab.robustness.exposure_snapshot import build_exposure_snapshot
from alpha_lab.robustness.final_test import run_final_test
from alpha_lab.robustness.freeze import freeze_candidate
from alpha_lab.robustness.report import evaluate_frozen_candidate

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="A-Share Alpha Lab data and reproducible baseline commands.",
)


def _render(value: dict[str, Any]) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _phase6_error(operation: str, error: BaseException) -> NoReturn:
    """Render a credential-safe error without echoing provider exception text."""
    typer.echo(
        json.dumps(
            {
                "error_type": type(error).__name__,
                "operation": operation,
                "status": "error",
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        err=True,
    )
    raise typer.Exit(code=1) from error


def _phase6_artifact(experiments_dir: Path, artifact_id: str, filename: str) -> Path:
    """Resolve exactly one canonical Phase 6 artifact by its document ID."""
    prefix = {
        "freeze.json": "freeze",
        "test_request.json": "request",
        "approval.json": "approval",
    }.get(filename)
    if prefix is None:  # pragma: no cover - internal programming error
        raise ValueError("unsupported Phase 6 artifact type")
    if re.fullmatch(rf"{prefix}-[0-9a-f]{{64}}", artifact_id) is None:
        raise ValueError(f"invalid Phase 6 {prefix} ID")
    phase6 = experiments_dir / "phase6"
    if filename == "freeze.json":
        candidates = [phase6 / artifact_id / filename]
    elif filename == "test_request.json":
        candidates = sorted(phase6.glob(f"freeze-*/{filename}"))
    else:  # approval.json
        candidates = sorted(phase6.glob(f"freeze-*/approvals/{artifact_id}.json"))
    matches: list[Path] = []
    identity_key = {
        "freeze.json": "freeze_id",
        "test_request.json": "request_id",
        "approval.json": "approval_id",
    }[filename]
    for candidate in candidates:
        if not candidate.is_file() or candidate.is_symlink():
            continue
        try:
            document = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(document, dict) and document.get(identity_key) == artifact_id:
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one registered {identity_key}")
    return matches[0]


def _latest_snapshot(data_dir: Path, snapshot: str | None) -> str:
    if snapshot:
        return snapshot
    state_path = data_dir / "state" / "latest_snapshot.txt"
    if not state_path.is_file():
        raise typer.BadParameter(
            "no latest snapshot; run make data-bootstrap first", param_hint="snapshot"
        )
    return state_path.read_text(encoding="utf-8").strip()


def _latest_research_snapshot(data_dir: Path, snapshot: str | None) -> str:
    if snapshot:
        return snapshot
    state_path = data_dir / "state" / "latest_research_snapshot.txt"
    if not state_path.is_file():
        raise typer.BadParameter(
            "no latest research snapshot; run make research-data-bootstrap first",
            param_hint="snapshot",
        )
    return state_path.read_text(encoding="utf-8").strip()


def _ingest(config_dir: Path, data_dir: Path, end_date: date | None) -> None:
    try:
        result = run_ingestion(config_dir, data_dir, end_date=end_date)
    except (RuntimeError, ValueError) as error:
        typer.echo(f"data ingestion failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    _render(
        {
            "snapshot_id": result.snapshot.snapshot_id,
            "quality_status": result.snapshot.quality_status,
            "network_requests": result.network_requests,
            "cache_hits": result.cache_hits,
            "raw_artifact_count": result.raw_artifact_count,
            "selected_provider": result.selected_provider,
            "fallback_reason": result.fallback_reason,
            "manifest": str(result.snapshot.manifest_path),
        }
    )
    if result.snapshot.quality_status == "error":
        raise typer.Exit(code=2)


@app.command("data-bootstrap")
def data_bootstrap(
    config_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("config"),
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
) -> None:
    """Fetch the fixed Phase 1 sample and materialize a snapshot."""
    _ingest(config_dir, data_dir, end_date=None)


@app.command("data-update")
def data_update(
    config_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("config"),
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
    end_date: Annotated[
        str | None,
        typer.Option(metavar="YYYY-MM-DD", help="Optional inclusive range extension."),
    ] = None,
) -> None:
    """Reuse cached intervals and fetch only a missing date tail."""
    parsed_end_date: date | None = None
    if end_date is not None:
        try:
            parsed_end_date = date.fromisoformat(end_date)
        except ValueError as error:
            raise typer.BadParameter(
                "end-date must use YYYY-MM-DD", param_hint="end-date"
            ) from error
    _ingest(config_dir, data_dir, end_date=parsed_end_date)


@app.command("data-validate")
def data_validate(
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
    snapshot: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Recalculate and display deterministic quality checks."""
    snapshot_id = _latest_snapshot(data_dir, snapshot)
    silver_path = data_dir / "silver" / snapshot_id / "daily.parquet"
    if not silver_path.is_file():
        raise typer.BadParameter(f"silver snapshot does not exist: {silver_path}")
    manifest_path = data_dir / "manifests" / snapshot_id / "manifest.json"
    if not manifest_path.is_file():
        raise typer.BadParameter(f"snapshot manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        to_qlib_instrument(item["code"]) for item in manifest["universe"]["symbols"]
    }
    report = build_quality_report(
        pd.read_parquet(silver_path), expected_instruments=expected
    )
    _render({"snapshot_id": snapshot_id, **report})
    if report["status"] == "error":
        raise typer.Exit(code=2)


@app.command("qlib-export")
def qlib_export(
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
    snapshot: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Export one immutable silver snapshot to Qlib file storage."""
    snapshot_id = _latest_snapshot(data_dir, snapshot)
    silver_path = data_dir / "silver" / snapshot_id / "daily.parquet"
    if not silver_path.is_file():
        raise typer.BadParameter(f"silver snapshot does not exist: {silver_path}")
    result = export_qlib(
        silver_path,
        data_dir / "qlib" / snapshot_id,
        snapshot_id,
    )
    _render(
        {
            "snapshot_id": result.snapshot_id,
            "output": str(result.output_path),
            "content_sha256": result.content_sha256,
            "file_count": result.file_count,
        }
    )


@app.command("research-data-probe")
def research_data_probe(
    config_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("config"),
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
) -> None:
    """Probe bounded Tushare Phase 5 capabilities without building a snapshot."""
    try:
        report = probe_research_data(config_dir, data_dir)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"research data probe failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    _render(asdict(report))


def _run_research_ingestion(
    config_dir: Path, data_dir: Path, end_date: date | None
) -> None:
    try:
        result = run_research_data_pipeline(config_dir, data_dir, end_date=end_date)
        sync_research_snapshot(
            data_dir / "metadata.duckdb",
            data_dir,
            result.snapshot.manifest_path,
        )
    except (duckdb.Error, OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"research data ingestion failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    _render(
        {
            "snapshot_id": result.snapshot.snapshot_id,
            "quality_status": result.snapshot.quality_status,
            "historical_symbol_count": result.historical_symbol_count,
            "membership_method": result.membership_method,
            "network_requests": result.network_requests,
            "cache_hits": result.cache_hits,
            "raw_artifact_count": result.raw_artifact_count,
            "manifest": str(result.snapshot.manifest_path),
        }
    )


@app.command("research-data-bootstrap")
def research_data_bootstrap(
    config_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("config"),
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
) -> None:
    """Build the bounded CSI 300 point-in-time Phase 5 snapshot."""
    _run_research_ingestion(config_dir, data_dir, None)


@app.command("research-data-update")
def research_data_update(
    end_date: Annotated[str, typer.Option("--end-date", metavar="YYYY-MM-DD")],
    config_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("config"),
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
) -> None:
    """Append missing raw intervals and build a new immutable research snapshot."""
    try:
        parsed = date.fromisoformat(end_date)
    except ValueError as error:
        raise typer.BadParameter(
            "end-date must use YYYY-MM-DD", param_hint="end-date"
        ) from error
    _run_research_ingestion(config_dir, data_dir, parsed)


@app.command("research-data-validate")
def research_data_validate(
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
    snapshot: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Verify manifest identity and every Phase 5 artifact checksum."""
    snapshot_id = _latest_research_snapshot(data_dir, snapshot)
    try:
        report = validate_research_snapshot(data_dir, snapshot_id)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"research data validation failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    _render(report)
    if not report["healthy"]:
        raise typer.Exit(code=2)


@app.command("universe-asof")
def universe_asof(
    as_of: Annotated[str, typer.Option("--date", metavar="YYYY-MM-DD")],
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
    snapshot: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Return the historically knowable CSI 300 universe for one date."""
    try:
        selected_date = date.fromisoformat(as_of)
    except ValueError as error:
        raise typer.BadParameter(
            "date must use YYYY-MM-DD", param_hint="date"
        ) from error
    snapshot_id = _latest_research_snapshot(data_dir, snapshot)
    manifest_path = data_dir / "manifests" / snapshot_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = {item["name"]: item for item in manifest["artifacts"]}
    security = pd.read_parquet(data_dir / artifacts["security_master.parquet"]["path"])
    membership = pd.read_parquet(
        data_dir / artifacts["index_membership.parquet"]["path"]
    )
    selected = universe_as_of(security, membership, selected_date)
    members = [
        {
            "security_id": str(row.security_id),
            "ts_code": str(getattr(row, "ts_code", "")),
            "name": str(getattr(row, "name", "")),
            "weight": (
                None if pd.isna(getattr(row, "weight", None)) else float(row.weight)
            ),
        }
        for row in selected.itertuples(index=False)
    ]
    _render(
        {
            "snapshot_id": snapshot_id,
            "date": selected_date.isoformat(),
            "index_id": "CN:INDEX:000300.SH",
            "member_count": len(members),
            "members": members,
        }
    )


@app.command("db-init")
def db_init(
    config_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("config"),
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
    database: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
) -> None:
    """Initialize the DuckDB catalog and sync repository manifests."""
    database_path = database or data_dir / "metadata.duckdb"
    initialized = initialize_database(database_path)
    synced = sync_repository_metadata(database_path, config_dir, data_dir)
    report = check_database(database_path, data_dir)
    _render(
        {
            **report,
            "migration_sha256": initialized.migration_sha256,
            "securities_synced": synced.securities_synced,
            "snapshots_synced": synced.snapshots_synced,
            "artifacts_synced": synced.artifacts_synced,
            "quality_results_synced": synced.quality_results_synced,
        }
    )
    if not report["healthy"]:
        raise typer.Exit(code=2)


@app.command("db-check")
def db_check(
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
    database: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
) -> None:
    """Validate catalog schema, logical references, and artifact paths."""
    database_path = database or data_dir / "metadata.duckdb"
    report = check_database(database_path, data_dir)
    _render(report)
    if not report["healthy"]:
        raise typer.Exit(code=2)


@app.command("baseline")
def baseline(
    config_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("config"),
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
    output_dir: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "artifacts/baseline"
    ),
    snapshot: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Run the deterministic Phase 2 Alpha158 + LightGBM engineering baseline."""
    try:
        result = run_baseline(config_dir, data_dir, output_dir, snapshot_id=snapshot)
    except (RuntimeError, ValueError) as error:
        typer.echo(f"baseline failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    _render(
        {
            "run_id": result.run_id,
            "output": str(result.output_dir),
            "manifest": str(result.manifest_path),
            "markdown_report": str(result.markdown_report_path),
            "html_report": str(result.html_report_path),
            "reproducibility_sha256": result.reproducibility_sha256,
        }
    )


@app.command("factor-list")
def factor_list(
    config_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("config"),
) -> None:
    """List registered Phase 3 factor candidates and their audit status."""
    registry = FactorRegistry(
        Path(__file__).parent / "factors" / "candidates",
        config_dir / "factor_registry.yaml",
    )
    _render(
        {
            "factors": [
                {
                    "factor_id": item.metadata.factor_id,
                    "name": item.metadata.name,
                    "family": item.metadata.family,
                    "status": item.metadata.status,
                    "source_sha256": item.source_sha256,
                    "metadata_sha256": item.metadata_sha256,
                }
                for item in registry.all()
            ],
            "accepted_factor_ids": sorted(registry.accepted_factor_ids),
        }
    )


@app.command("factor-eval")
def factor_eval(
    factor_id: Annotated[str, typer.Option("--id", help="Registered factor ID.")],
    config_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("config"),
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
    output_dir: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "artifacts/factors"
    ),
    snapshot: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Run the locked Phase 3 structured evaluation for one factor."""
    try:
        result = evaluate_factor(
            factor_id,
            config_dir,
            data_dir,
            output_dir,
            snapshot_id=snapshot,
        )
    except (
        duckdb.Error,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        typer.echo(f"factor evaluation failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    _render(
        {
            "factor_id": result.factor_id,
            "run_id": result.run_id,
            "output": str(result.output_dir),
            "factor_result": str(result.result_path),
            "factor_values": str(result.values_path),
            "result_sha256": result.result_sha256,
            "eligible_for_review": result.eligible_for_review,
        }
    )


@app.command("mining-init")
def mining_init(
    run_id: Annotated[str, typer.Option("--run", help="Mining run ID.")],
    rounds: Annotated[int, typer.Option(min=1)] = 5,
    config_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("config"),
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
    experiments_dir: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "experiments"
    ),
) -> None:
    """Create or reopen a Phase 4 manifest and research brief."""
    try:
        run_dir = initialize_mining_run(
            run_id,
            rounds,
            repo_root=Path.cwd(),
            config_dir=config_dir,
            data_dir=data_dir,
            experiments_dir=experiments_dir,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"mining initialization failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    _render({"run_id": run_id, "run_dir": str(run_dir), "rounds": rounds})


@app.command("mining-round")
def mining_round(
    run_id: Annotated[str, typer.Option("--run", help="Mining run ID.")],
    config_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("config"),
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
    experiments_dir: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "experiments"
    ),
    artifacts_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("artifacts"),
    proposal: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
) -> None:
    """Evaluate one staged Phase 4 factor proposal and retain its decision."""
    try:
        result = run_mining_round(
            run_id,
            repo_root=Path.cwd(),
            config_dir=config_dir,
            data_dir=data_dir,
            experiments_dir=experiments_dir,
            artifacts_dir=artifacts_dir,
            proposal_path=proposal,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"mining round failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    _render(
        {
            "run_id": result.run_id,
            "round_number": result.round_number,
            "factor_id": result.factor_id,
            "decision": result.decision,
            "round_dir": str(result.round_dir),
            "decision_path": str(result.decision_path),
        }
    )


@app.command("mining-loop")
def mining_loop(
    run_id: Annotated[str, typer.Option("--run", help="Mining run ID.")],
    rounds: Annotated[int, typer.Option(min=1)] = 5,
    config_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("config"),
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
    experiments_dir: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "experiments"
    ),
    artifacts_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("artifacts"),
    proposals_dir: Annotated[Path | None, typer.Option(file_okay=False)] = None,
) -> None:
    """Run or resume a bounded Phase 4 mining loop."""
    try:
        results = run_mining_loop(
            run_id,
            rounds,
            repo_root=Path.cwd(),
            config_dir=config_dir,
            data_dir=data_dir,
            experiments_dir=experiments_dir,
            artifacts_dir=artifacts_dir,
            proposals_dir=proposals_dir,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"mining loop failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    _render(
        {
            "run_id": run_id,
            "completed_rounds": len(results),
            "decisions": [item.decision for item in results],
            "report": str(experiments_dir / run_id / "final_report.md"),
        }
    )


@app.command("mining-report")
def mining_report(
    run_id: Annotated[str, typer.Option("--run", help="Mining run ID.")],
    experiments_dir: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "experiments"
    ),
) -> None:
    """Regenerate the small Markdown audit report for a mining run."""
    try:
        path = render_mining_report(run_id, experiments_dir)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"mining report failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    _render({"run_id": run_id, "report": str(path)})


@app.command("exposure-probe")
def exposure_probe(
    config_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("config"),
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
) -> None:
    """Probe bounded Phase 6 size and industry provider capabilities."""
    try:
        report = probe_exposure_capabilities(config_dir, data_dir)
    except (duckdb.Error, OSError, RuntimeError, TypeError, ValueError) as error:
        _phase6_error("exposure-probe", error)
    _render({"operation": "exposure-probe", "status": "ok", **report})


@app.command("exposure-bootstrap")
def exposure_bootstrap(
    config_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("config"),
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
) -> None:
    """Build, validate, and catalog one immutable Phase 6 exposure snapshot."""
    try:
        result = build_exposure_snapshot(config_dir, data_dir)
        sync_exposure_snapshot(
            data_dir / "metadata.duckdb", data_dir, result.manifest_path
        )
    except (duckdb.Error, OSError, RuntimeError, TypeError, ValueError) as error:
        _phase6_error("exposure-bootstrap", error)
    _render(
        {
            "manifest": str(result.manifest_path),
            "manifest_sha256": result.manifest_sha256,
            "operation": "exposure-bootstrap",
            "quality_status": result.quality_status,
            "snapshot_id": result.snapshot_id,
            "status": "ok",
        }
    )


@app.command("robustness-freeze")
def robustness_freeze(
    factor_id: Annotated[str, typer.Option("--id", help="F1002 or F1003.")],
    config_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("config"),
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
    experiments_dir: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "experiments"
    ),
) -> None:
    """Freeze one approved Phase 6 candidate and all current dependencies."""
    try:
        result = freeze_candidate(factor_id, config_dir, data_dir, experiments_dir)
    except (
        duckdb.Error,
        KeyError,
        OSError,
        PermissionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        _phase6_error("robustness-freeze", error)
    _render(
        {
            "factor_id": result.factor_id,
            "freeze": str(result.freeze_path),
            "freeze_id": result.freeze_id,
            "freeze_sha256": result.freeze_sha256,
            "operation": "robustness-freeze",
            "status": "ok",
        }
    )


@app.command("robustness-eval")
def robustness_eval(
    freeze: Annotated[str, typer.Option("--freeze", help="Freeze ID.")],
    config_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("config"),
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
    experiments_dir: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "experiments"
    ),
) -> None:
    """Run the five fixed pre-test folds without opening the locked test."""
    try:
        freeze_path = _phase6_artifact(experiments_dir, freeze, "freeze.json")
        result = evaluate_frozen_candidate(
            freeze_path, config_dir, data_dir, experiments_dir
        )
    except (
        duckdb.Error,
        KeyError,
        OSError,
        PermissionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        _phase6_error("robustness-eval", error)
    _render(
        {
            "freeze_id": result.freeze_id,
            "operation": "robustness-eval",
            "passed": result.passed,
            "report": str(result.report_path),
            "report_sha256": result.report_sha256,
            "status": "ok",
            "test_accessed": False,
        }
    )


@app.command("test-request")
def test_request(
    freeze: Annotated[str, typer.Option("--freeze", help="Freeze ID.")],
    experiments_dir: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "experiments"
    ),
) -> None:
    """Create a request only after replaying all pre-test evidence."""
    try:
        freeze_path = _phase6_artifact(experiments_dir, freeze, "freeze.json")
        request_path = create_test_request(freeze_path)
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (
        duckdb.Error,
        KeyError,
        OSError,
        PermissionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        _phase6_error("test-request", error)
    _render(
        {
            "freeze_id": request["freeze_id"],
            "operation": "test-request",
            "request": str(request_path),
            "request_id": request["request_id"],
            "status": "test_requested",
            "test_accessed": False,
        }
    )


@app.command("test-approve")
def test_approve(
    request: Annotated[str, typer.Option("--request", help="Request ID.")],
    approver: Annotated[str, typer.Option("--approver", help="Human identity.")],
    confirm: Annotated[
        str, typer.Option("--confirm", help="Exact confirmed freeze SHA256.")
    ],
    experiments_dir: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "experiments"
    ),
) -> None:
    """Record an explicit human approval for one request and freeze hash."""
    try:
        request_path = _phase6_artifact(experiments_dir, request, "test_request.json")
        approval_path = approve_test_request(request_path, approver, confirm)
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
    except (
        duckdb.Error,
        KeyError,
        OSError,
        PermissionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        _phase6_error("test-approve", error)
    _render(
        {
            "approval": str(approval_path),
            "approval_id": approval["approval_id"],
            "freeze_id": approval["freeze_id"],
            "operation": "test-approve",
            "request_id": approval["request_id"],
            "status": "approved",
            "test_accessed": False,
        }
    )


@app.command("final-test")
def final_test(
    approval: Annotated[str, typer.Option("--approval", help="Approval ID.")],
    config_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("config"),
    data_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data"),
    experiments_dir: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "experiments"
    ),
) -> None:
    """Run the locked test only from one explicit, cataloged approval."""
    try:
        approval_path = _phase6_artifact(experiments_dir, approval, "approval.json")
        result_path = run_final_test(
            approval_path, config_dir, data_dir, experiments_dir
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (
        duckdb.Error,
        KeyError,
        OSError,
        PermissionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        _phase6_error("final-test", error)
    _render(
        {
            "operation": "final-test",
            "result": str(result_path),
            "status": result["status"],
            "test_accessed": result["test_accessed"],
            "test_run_id": result["test_run_id"],
        }
    )


if __name__ == "__main__":
    app()
