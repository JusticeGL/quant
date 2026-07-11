from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alpha_lab.data.config import DataSourceConfig, UniverseConfig
from alpha_lab.data.normalize import to_qlib_instrument
from alpha_lab.data.quality import build_quality_report

NORMALIZATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RawInput:
    provider: str
    endpoint: str
    symbol: str
    path: Path
    sha256: str
    row_count: int
    requested_start: str
    requested_end: str


@dataclass(frozen=True)
class SnapshotResult:
    snapshot_id: str
    bronze_path: Path
    silver_path: Path
    quality_report_path: Path
    manifest_path: Path
    manifest_sha256: str
    quality_status: str


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_bytes(path: Path, content: bytes, *, immutable: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if immutable and path.read_bytes() != content:
            raise RuntimeError(f"immutable artifact would change: {path}")
        if immutable:
            return
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_parquet_immutable(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(  # type: ignore[no-untyped-call]
            table,
            temporary,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_snapshot(
    data_root: Path,
    frame: pd.DataFrame,
    *,
    source: DataSourceConfig,
    universe: UniverseConfig,
    raw_inputs: list[RawInput],
) -> SnapshotResult:
    ordered = frame.sort_values(
        ["trade_date", "instrument"], kind="stable"
    ).reset_index(drop=True)
    identity = {
        "normalization_schema_version": NORMALIZATION_SCHEMA_VERSION,
        "source": source.model_dump(mode="json"),
        "universe": universe.model_dump(mode="json"),
        "raw_inputs": [
            {key: value for key, value in asdict(item).items() if key != "path"}
            for item in sorted(
                raw_inputs,
                key=lambda value: (
                    value.symbol,
                    value.requested_start,
                    value.requested_end,
                    value.sha256,
                ),
            )
        ],
    }
    identity_hash = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    snapshot_id = f"p1-{identity_hash[:20]}"

    bronze_path = data_root / "bronze" / snapshot_id / "daily.parquet"
    silver_path = data_root / "silver" / snapshot_id / "daily.parquet"
    report_path = data_root / "manifests" / snapshot_id / "quality_report.json"
    manifest_path = data_root / "manifests" / snapshot_id / "manifest.json"
    _write_parquet_immutable(bronze_path, ordered)
    _write_parquet_immutable(silver_path, ordered)

    quality = build_quality_report(
        ordered,
        expected_instruments={
            to_qlib_instrument(symbol.code) for symbol in universe.symbols
        },
    )
    _atomic_write_bytes(report_path, _canonical_bytes(quality), immutable=True)
    manifest = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "identity_sha256": identity_hash,
        "source": source.model_dump(mode="json"),
        "universe": universe.model_dump(mode="json"),
        "summary": {
            "row_count": len(ordered),
            "instrument_count": int(ordered["instrument"].nunique()),
            "date_start": quality["date_range"]["start"],
            "date_end": quality["date_range"]["end"],
            "quality_status": quality["status"],
        },
        "raw_inputs": [
            {
                **{key: value for key, value in asdict(item).items() if key != "path"},
                "path": _portable_path(data_root, item.path),
            }
            for item in sorted(
                raw_inputs, key=lambda value: (value.symbol, value.sha256)
            )
        ],
        "artifacts": {
            "bronze": {
                "path": _portable_path(data_root, bronze_path),
                "sha256": _sha256(bronze_path),
            },
            "silver": {
                "path": _portable_path(data_root, silver_path),
                "sha256": _sha256(silver_path),
            },
            "quality_report": {
                "path": _portable_path(data_root, report_path),
                "sha256": _sha256(report_path),
            },
        },
    }
    _atomic_write_bytes(manifest_path, _canonical_bytes(manifest), immutable=True)
    latest_path = data_root / "state" / "latest_snapshot.txt"
    _atomic_write_bytes(latest_path, f"{snapshot_id}\n".encode(), immutable=False)

    return SnapshotResult(
        snapshot_id=snapshot_id,
        bronze_path=bronze_path,
        silver_path=silver_path,
        quality_report_path=report_path,
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        quality_status=str(quality["status"]),
    )


def _portable_path(data_root: Path, path: Path) -> str:
    try:
        return path.relative_to(data_root).as_posix()
    except ValueError:
        return path.name
