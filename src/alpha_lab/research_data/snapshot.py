from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alpha_lab.data.providers.base import file_sha256
from alpha_lab.research_data.config import ResearchDataConfig
from alpha_lab.research_data.contracts import ResearchTables
from alpha_lab.research_data.provider import TushareArtifact
from alpha_lab.research_data.quality import build_research_quality_report
from alpha_lab.research_data.universe import universe_as_of

RESEARCH_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResearchSnapshotResult:
    snapshot_id: str
    snapshot_dir: Path
    quality_report_path: Path
    manifest_path: Path
    manifest_sha256: str
    quality_status: str


def validate_research_snapshot(data_root: Path, snapshot_id: str) -> dict[str, Any]:
    manifest_path = data_root / "manifests" / snapshot_id / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"research snapshot manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("snapshot_id") != snapshot_id
        or manifest.get("snapshot_type") != "research_market"
    ):
        raise ValueError("research snapshot manifest identity mismatch")
    checked = 0
    failures: list[str] = []
    for item in [*manifest.get("raw_inputs", []), *manifest.get("artifacts", [])]:
        path = data_root / str(item["path"])
        checked += 1
        if not path.is_file():
            failures.append(f"missing:{item['path']}")
        elif file_sha256(path) != str(item["sha256"]):
            failures.append(f"sha256:{item['path']}")
    quality = manifest.get("quality_report")
    if quality:
        path = data_root / str(quality["path"])
        checked += 1
        if not path.is_file():
            failures.append(f"missing:{quality['path']}")
        elif file_sha256(path) != str(quality["sha256"]):
            failures.append(f"sha256:{quality['path']}")
    return {
        "snapshot_id": snapshot_id,
        "healthy": not failures and manifest.get("quality_status") != "error",
        "quality_status": manifest.get("quality_status"),
        "checked_artifact_count": checked,
        "failures": failures,
        "manifest_sha256": file_sha256(manifest_path),
    }


def materialize_research_snapshot(
    data_root: Path,
    config: ResearchDataConfig,
    tables: ResearchTables,
    raw_inputs: Sequence[TushareArtifact],
) -> ResearchSnapshotResult:
    quality = build_research_quality_report(tables, config)
    if quality["status"] == "error":
        raise ValueError("research data quality gates failed")

    research_root = data_root / "research"
    research_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".p5-build-", dir=research_root))
    try:
        universe_dates = _build_universe_dates(tables, config)
        artifacts = _write_tables(temporary, tables, universe_dates)
        identity = {
            "research_schema_version": RESEARCH_SCHEMA_VERSION,
            "config": config.model_dump(mode="json"),
            "raw_inputs": [
                _raw_identity(item)
                for item in sorted(
                    raw_inputs, key=lambda value: (value.api_name, value.request_sha256)
                )
            ],
            "artifacts": [
                {
                    "name": item["name"],
                    "sha256": item["sha256"],
                    "row_count": item["row_count"],
                }
                for item in artifacts
            ],
        }
        identity_hash = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
        snapshot_id = f"p5-{identity_hash[:20]}"
        snapshot_dir = research_root / snapshot_id
        if snapshot_dir.exists():
            _validate_existing_snapshot(snapshot_dir, temporary, artifacts)
        else:
            os.replace(temporary, snapshot_dir)

        manifest_dir = data_root / "manifests" / snapshot_id
        quality_path = manifest_dir / "quality_report.json"
        manifest_path = manifest_dir / "manifest.json"
        _write_immutable(quality_path, _canonical_bytes(quality))
        manifest_artifacts = [
            {
                **item,
                "path": (Path("research") / snapshot_id / str(item["name"])).as_posix(),
            }
            for item in artifacts
        ]
        manifest = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "snapshot_type": "research_market",
            "identity_sha256": identity_hash,
            "quality_status": quality["status"],
            "source": {"provider": "tushare", "credential_redacted": True},
            "scope": quality["scope"],
            "summary": quality["summary"],
            "raw_inputs": [
                {
                    **_raw_identity(item),
                    "path": _portable_path(data_root, item.parquet_path),
                }
                for item in sorted(
                    raw_inputs, key=lambda value: (value.api_name, value.request_sha256)
                )
            ],
            "artifacts": manifest_artifacts,
            "quality_report": {
                "path": _portable_path(data_root, quality_path),
                "sha256": hashlib.sha256(_canonical_bytes(quality)).hexdigest(),
            },
        }
        _write_immutable(manifest_path, _canonical_bytes(manifest))
        latest = data_root / "state" / "latest_research_snapshot.txt"
        _write_mutable(latest, f"{snapshot_id}\n".encode())
        return ResearchSnapshotResult(
            snapshot_id=snapshot_id,
            snapshot_dir=snapshot_dir,
            quality_report_path=quality_path,
            manifest_path=manifest_path,
            manifest_sha256=file_sha256(manifest_path),
            quality_status=str(quality["status"]),
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _write_tables(
    root: Path, tables: ResearchTables, universe_dates: pd.DataFrame
) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    for name, frame, sort_keys in (
        ("security_master.parquet", tables.security_master, ["security_id"]),
        (
            "security_name_history.parquet",
            tables.security_name_history,
            ["security_id", "effective_from"],
        ),
        (
            "trading_calendar.parquet",
            tables.trading_calendar,
            ["calendar_date", "exchange"],
        ),
        (
            "index_membership.parquet",
            tables.index_membership,
            ["index_id", "security_id", "effective_from"],
        ),
        (
            "suspension.parquet",
            tables.suspension,
            ["security_id", "effective_from"],
        ),
        (
            "universe_dates.parquet",
            universe_dates,
            ["as_of_date", "security_id"],
        ),
    ):
        artifacts.append(_write_frame(root, name, frame, sort_keys))
    for dataset, frame, sort_keys in (
        ("daily_bar", tables.daily_bar, ["trade_date", "security_id"]),
        (
            "adjustment_factor",
            tables.adjustment_factor,
            ["trade_date", "security_id", "factor_type"],
        ),
        ("daily_status", tables.daily_status, ["trade_date", "security_id"]),
    ):
        if frame.empty:
            artifacts.append(
                _write_frame(root, f"{dataset}/part.parquet", frame, sort_keys)
            )
            continue
        years = pd.to_datetime(frame["trade_date"], errors="raise").dt.year
        for year in sorted(years.unique()):
            part = frame.loc[years == year].copy()
            artifacts.append(
                _write_frame(
                    root,
                    f"{dataset}/year={int(year)}/part.parquet",
                    part,
                    sort_keys,
                )
            )
    return sorted(artifacts, key=lambda item: str(item["name"]))


def _write_frame(
    root: Path, name: str, frame: pd.DataFrame, sort_keys: list[str]
) -> dict[str, object]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    usable_keys = [key for key in sort_keys if key in frame.columns]
    ordered = (
        frame.sort_values(usable_keys, kind="stable").reset_index(drop=True)
        if usable_keys
        else frame.reset_index(drop=True)
    )
    table = pa.Table.from_pandas(ordered, preserve_index=False)
    pq.write_table(  # type: ignore[no-untyped-call]
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
    )
    return {
        "name": name,
        "format": "parquet",
        "sha256": file_sha256(path),
        "row_count": len(ordered),
    }


def _build_universe_dates(
    tables: ResearchTables, config: ResearchDataConfig
) -> pd.DataFrame:
    calendar = tables.trading_calendar
    if calendar.empty:
        return pd.DataFrame(columns=["as_of_date", "index_id", "security_id", "weight"])
    dates = pd.to_datetime(
        calendar.loc[calendar["is_open"].fillna(False), "calendar_date"]
    )
    dates = dates.loc[
        (dates.dt.date >= config.start_date) & (dates.dt.date <= config.end_date)
    ]
    parts: list[pd.DataFrame] = []
    for value in sorted(dates.dt.date.unique()):
        selected = universe_as_of(
            tables.security_master,
            tables.index_membership,
            value,
        )
        columns = [
            item
            for item in ("as_of_date", "index_id", "security_id", "weight")
            if item in selected.columns
        ]
        parts.append(selected.loc[:, columns])
    if not parts:
        return pd.DataFrame(columns=["as_of_date", "index_id", "security_id", "weight"])
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["as_of_date", "security_id"], kind="stable")
        .reset_index(drop=True)
    )


def _validate_existing_snapshot(
    destination: Path,
    temporary: Path,
    artifacts: list[dict[str, object]],
) -> None:
    for item in artifacts:
        relative = Path(str(item["name"]))
        existing = destination / relative
        candidate = temporary / relative
        if not existing.is_file() or file_sha256(existing) != file_sha256(candidate):
            raise RuntimeError(f"immutable snapshot differs: {destination}")


def _raw_identity(artifact: TushareArtifact) -> dict[str, object]:
    value = asdict(artifact)
    return {
        "api_name": value["api_name"],
        "request_sha256": value["request_sha256"],
        "sha256": value["sha256"],
        "row_count": value["row_count"],
        "params": value["params"],
        "fields": value["fields"],
    }


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable artifact differs: {path}")
        return
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_mutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _portable_path(data_root: Path, path: Path) -> str:
    try:
        return path.relative_to(data_root).as_posix()
    except ValueError:
        return path.name
