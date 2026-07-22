from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class ExposureTables:
    market_cap: pd.DataFrame
    industry_definition: pd.DataFrame
    industry_membership: pd.DataFrame


@dataclass(frozen=True)
class ExposureSnapshotResult:
    snapshot_id: str
    snapshot_dir: Path
    quality_report_path: Path
    manifest_path: Path
    manifest_sha256: str
    quality_status: str


@dataclass(frozen=True)
class FrozenCandidate:
    freeze_id: str
    factor_id: Literal["F1002", "F1003"]
    freeze_path: Path
    freeze_sha256: str


@dataclass(frozen=True)
class RobustnessResult:
    freeze_id: str
    output_dir: Path
    report_path: Path
    report_sha256: str
    passed: bool
