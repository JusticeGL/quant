from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

from alpha_lab.data.normalize import to_qlib_instrument
from alpha_lab.data.pipeline import run_ingestion
from alpha_lab.data.qlib_export import export_qlib
from alpha_lab.data.quality import build_quality_report
from alpha_lab.database.catalog import (
    check_database,
    initialize_database,
    sync_repository_metadata,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="A-Share Alpha Lab Phase 1 data commands.",
)


def _render(value: dict[str, Any]) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _latest_snapshot(data_dir: Path, snapshot: str | None) -> str:
    if snapshot:
        return snapshot
    state_path = data_dir / "state" / "latest_snapshot.txt"
    if not state_path.is_file():
        raise typer.BadParameter(
            "no latest snapshot; run make data-bootstrap first", param_hint="snapshot"
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


if __name__ == "__main__":
    app()
