from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

QLIB_FIELDS = ("open", "high", "low", "close", "volume", "amount")


@dataclass(frozen=True)
class QlibExportResult:
    output_path: Path
    snapshot_id: str
    content_sha256: str
    file_count: int


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_manifest(root: Path) -> list[dict[str, str]]:
    return [
        {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "export_manifest.json"
    ]


def _content_hash(files: list[dict[str, str]]) -> str:
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def export_qlib(
    silver_path: Path, output_path: Path, snapshot_id: str
) -> QlibExportResult:
    frame = pd.read_parquet(silver_path)
    required = {"trade_date", "instrument", *QLIB_FIELDS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"silver data is missing Qlib fields: {missing}")
    if frame.duplicated(["trade_date", "instrument"]).any():
        raise ValueError("silver data has duplicate trade_date/instrument keys")

    frame = frame.sort_values(["instrument", "trade_date"], kind="stable").copy()
    frame.loc[:, "trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    calendar = pd.DatetimeIndex(sorted(frame["trade_date"].unique()))
    if calendar.empty:
        raise ValueError("cannot export an empty Qlib calendar")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    )
    try:
        _write_qlib_files(frame, calendar, temporary)
        files = _content_manifest(temporary)
        content_sha256 = _content_hash(files)
        export_manifest = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "source_silver_sha256": _sha256(silver_path),
            "fields": list(QLIB_FIELDS),
            "content_sha256": content_sha256,
            "files": files,
        }
        (temporary / "export_manifest.json").write_text(
            json.dumps(
                export_manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )

        if output_path.exists():
            existing_files = _content_manifest(output_path)
            existing_hash = _content_hash(existing_files)
            if existing_hash != content_sha256:
                raise RuntimeError(
                    f"existing Qlib export differs for snapshot {snapshot_id}"
                )
            return QlibExportResult(
                output_path=output_path,
                snapshot_id=snapshot_id,
                content_sha256=existing_hash,
                file_count=len(existing_files),
            )

        os.replace(temporary, output_path)
        return QlibExportResult(
            output_path=output_path,
            snapshot_id=snapshot_id,
            content_sha256=content_sha256,
            file_count=len(files),
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _write_qlib_files(
    frame: pd.DataFrame, calendar: pd.DatetimeIndex, root: Path
) -> None:
    calendars_dir = root / "calendars"
    instruments_dir = root / "instruments"
    features_dir = root / "features"
    calendars_dir.mkdir(parents=True)
    instruments_dir.mkdir(parents=True)
    features_dir.mkdir(parents=True)

    (calendars_dir / "day.txt").write_text(
        "".join(f"{day.strftime('%Y-%m-%d')}\n" for day in calendar),
        encoding="utf-8",
    )
    instrument_lines: list[str] = []
    calendar_positions = {day: position for position, day in enumerate(calendar)}

    for instrument, part in frame.groupby("instrument", sort=True):
        part = part.sort_values("trade_date", kind="stable").set_index("trade_date")
        start = pd.Timestamp(part.index.min())
        end = pd.Timestamp(part.index.max())
        start_index = calendar_positions[start]
        end_index = calendar_positions[end]
        aligned_calendar = calendar[start_index : end_index + 1]
        instrument_name = str(instrument)
        instrument_lines.append(
            f"{instrument_name}\t{start.strftime('%Y-%m-%d')}\t{end.strftime('%Y-%m-%d')}\n"
        )
        instrument_dir = features_dir / instrument_name.lower()
        instrument_dir.mkdir(parents=True)
        for field in QLIB_FIELDS:
            values = (
                pd.to_numeric(part[field], errors="coerce")
                .reindex(aligned_calendar)
                .to_numpy(dtype="<f4")
            )
            payload = np.concatenate(
                (np.asarray([start_index], dtype="<f4"), values)
            ).astype("<f4", copy=False)
            payload.tofile(instrument_dir / f"{field}.day.bin")

    (instruments_dir / "all.txt").write_text(
        "".join(instrument_lines), encoding="utf-8"
    )
