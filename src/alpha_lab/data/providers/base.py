from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RawArtifact:
    provider: str
    endpoint: str
    symbol: str
    parquet_path: Path
    metadata_path: Path
    sha256: str
    row_count: int
    requested_start: date
    requested_end: date
    ingested_at: str


@dataclass(frozen=True)
class RawRange:
    frame: pd.DataFrame
    artifacts: tuple[RawArtifact, ...]
    cache_hits: int
    network_requests: int


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def intervals_overlap(
    left_start: date, left_end: date, right_start: date, right_end: date
) -> bool:
    return left_start <= right_end and right_start <= left_end


def missing_intervals(
    start: date, end: date, covered: list[tuple[date, date]]
) -> list[tuple[date, date]]:
    missing: list[tuple[date, date]] = []
    cursor = start
    for interval_start, interval_end in sorted(covered):
        if interval_end < cursor or interval_start > end:
            continue
        clipped_start = max(interval_start, start)
        clipped_end = min(interval_end, end)
        if clipped_start > cursor:
            missing.append((cursor, clipped_start - timedelta(days=1)))
        cursor = max(cursor, clipped_end + timedelta(days=1))
        if cursor > end:
            break
    if cursor <= end:
        missing.append((cursor, end))
    return missing
