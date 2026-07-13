from __future__ import annotations

import hashlib
import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from alpha_lab.data.normalize import to_qlib_instrument

LOCKED_TEST_START = date(2026, 1, 1)
_MARKET_DATASETS = ("daily_bar", "adjustment_factor", "daily_status")


def read_pretest_market(
    data_dir: Path, snapshot_id: str, end_before: date
) -> pd.DataFrame:
    _require_pretest_boundary(end_before)
    manifest = _read_manifest(data_dir, snapshot_id, "research_market")
    selected = _pretest_partition_artifacts(
        data_dir,
        manifest,
        snapshot_id,
        root="research",
        datasets=_MARKET_DATASETS,
    )
    daily_bar = _read_partition_set(selected["daily_bar"], end_before)
    if daily_bar.empty:
        raise ValueError("Phase 5 pre-test daily_bar partitions are missing")
    adjustment = _read_partition_set(selected["adjustment_factor"], end_before)
    status = _read_partition_set(selected["daily_status"], end_before)
    return _market_contract(daily_bar, adjustment, status)


def read_pretest_exposures(
    data_dir: Path, snapshot_id: str, end_before: date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_pretest_boundary(end_before)
    manifest = _read_manifest(data_dir, snapshot_id, "point_in_time_exposure")
    selected = _pretest_partition_artifacts(
        data_dir,
        manifest,
        snapshot_id,
        root="exposures",
        datasets=("market_cap",),
    )
    market_cap = _read_partition_set(selected["market_cap"], end_before)
    membership_artifact = _fixed_artifact(
        data_dir,
        manifest,
        snapshot_id,
        root="exposures",
        name="industry_membership.parquet",
    )
    membership = pd.read_parquet(membership_artifact)
    _require_columns(
        membership,
        {
            "industry_id",
            "security_id",
            "effective_from",
            "effective_to",
            "known_at",
        },
        "industry membership",
    )
    bounded = membership.copy()
    bounded["effective_from"] = pd.to_datetime(
        bounded["effective_from"], errors="raise"
    ).dt.normalize()
    bounded["effective_to"] = pd.to_datetime(
        bounded["effective_to"], errors="raise"
    ).dt.normalize()
    bounded["known_at"] = pd.to_datetime(bounded["known_at"], errors="raise", utc=True)
    if bounded[["effective_from", "known_at"]].isna().any().any():
        raise ValueError("industry membership has missing point-in-time dates")
    cutoff = pd.Timestamp(end_before)
    locked_cutoff = pd.Timestamp(LOCKED_TEST_START)
    known_cutoff = pd.Timestamp(end_before, tz="UTC")
    locked_known_cutoff = pd.Timestamp(LOCKED_TEST_START, tz="UTC")
    bounded = bounded.loc[
        (bounded["effective_from"] < cutoff)
        & (bounded["effective_from"] < locked_cutoff)
        & (bounded["known_at"] < known_cutoff)
        & (bounded["known_at"] < locked_known_cutoff)
    ].copy()
    maximum_effective_to = pd.Timestamp(end_before - timedelta(days=1))
    bounded["effective_to"] = bounded["effective_to"].where(
        bounded["effective_to"].isna()
        | (bounded["effective_to"] <= maximum_effective_to),
        maximum_effective_to,
    )
    bounded = bounded.sort_values(
        ["security_id", "effective_from", "industry_id"], kind="stable"
    ).reset_index(drop=True)
    return market_cap, bounded


def _require_pretest_boundary(end_before: date) -> None:
    if end_before >= LOCKED_TEST_START:
        raise PermissionError(
            "pre-test readers cannot reach the locked test boundary 2026-01-01"
        )


def _read_manifest(
    data_dir: Path, snapshot_id: str, expected_type: str
) -> dict[str, Any]:
    if re.fullmatch(r"p(?:5|6x)-[A-Za-z0-9]+", snapshot_id) is None:
        raise ValueError("invalid snapshot ID")
    path = data_dir / "manifests" / snapshot_id / "manifest.json"
    if not path.is_file():
        raise ValueError(f"snapshot manifest is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("snapshot manifest is malformed") from error
    if not isinstance(value, dict):
        raise ValueError("snapshot manifest must be a mapping")
    if (
        value.get("snapshot_id") != snapshot_id
        or value.get("snapshot_type") != expected_type
    ):
        raise ValueError("snapshot manifest identity mismatch")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not all(
        isinstance(item, dict) for item in artifacts
    ):
        raise ValueError("snapshot manifest artifacts are malformed")
    names = [item.get("name") for item in artifacts]
    if any(not isinstance(name, str) for name in names) or len(names) != len(
        set(names)
    ):
        raise ValueError("snapshot manifest artifact names are invalid")
    return value


def _pretest_partition_artifacts(
    data_dir: Path,
    manifest: dict[str, Any],
    snapshot_id: str,
    *,
    root: str,
    datasets: tuple[str, ...],
) -> dict[str, list[Path]]:
    selected: dict[str, list[Path]] = {dataset: [] for dataset in datasets}
    patterns = {
        dataset: re.compile(rf"^{re.escape(dataset)}/year=([0-9]{{4}})/part[.]parquet$")
        for dataset in datasets
    }
    for item in manifest["artifacts"]:
        name = str(item["name"])
        dataset = next(
            (value for value in datasets if name.startswith(f"{value}/")), None
        )
        if dataset is None:
            continue
        match = patterns[dataset].fullmatch(name)
        if match is None:
            raise ValueError(f"non-canonical {dataset} artifact: {name}")
        year = int(match.group(1))
        if year >= LOCKED_TEST_START.year:
            continue
        selected[dataset].append(
            _verified_artifact_path(
                data_dir, item, snapshot_id, root=root, expected_name=name
            )
        )
    return selected


def _fixed_artifact(
    data_dir: Path,
    manifest: dict[str, Any],
    snapshot_id: str,
    *,
    root: str,
    name: str,
) -> Path:
    matches = [item for item in manifest["artifacts"] if item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"snapshot manifest must declare exactly one {name}")
    return _verified_artifact_path(
        data_dir, matches[0], snapshot_id, root=root, expected_name=name
    )


def _verified_artifact_path(
    data_dir: Path,
    artifact: dict[str, Any],
    snapshot_id: str,
    *,
    root: str,
    expected_name: str,
) -> Path:
    expected_relative = Path(root) / snapshot_id / expected_name
    if artifact.get("path") != expected_relative.as_posix():
        raise ValueError(f"non-canonical artifact path for {expected_name}")
    path = data_dir / expected_relative
    current = path
    while current != data_dir:
        if current.is_symlink():
            raise ValueError(f"artifact path contains symlink: {expected_name}")
        current = current.parent
    snapshot_root = (data_dir / root / snapshot_id).resolve()
    try:
        path.resolve().relative_to(snapshot_root)
    except ValueError as error:
        raise ValueError(f"artifact escapes snapshot root: {expected_name}") from error
    if not path.is_file():
        raise ValueError(f"snapshot artifact is missing: {expected_name}")
    expected_sha256 = artifact.get("sha256")
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or _sha256(path) != expected_sha256
    ):
        raise ValueError(f"snapshot artifact checksum mismatch: {expected_name}")
    return path


def _read_partition_set(paths: list[Path], end_before: date) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for path in paths:
        part = pd.read_parquet(path)
        _require_columns(
            part, {"trade_date", "security_id", "known_at"}, "dated snapshot"
        )
        if part["known_at"].isna().any():
            raise ValueError(f"dated snapshot known_at contains nulls: {path.name}")
        try:
            part["known_at"] = pd.to_datetime(
                part["known_at"], errors="raise", utc=True
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"dated snapshot known_at is invalid: {path.name}"
            ) from error
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"], errors="raise"
    ).dt.normalize()
    known_at = frame["known_at"]
    bounded = frame.loc[
        (frame["trade_date"].dt.date < end_before)
        & (frame["trade_date"].dt.date < LOCKED_TEST_START)
        & (known_at < pd.Timestamp(end_before, tz="UTC"))
        & (known_at < pd.Timestamp(LOCKED_TEST_START, tz="UTC"))
    ].copy()
    return bounded.sort_values(
        ["trade_date", "security_id"], kind="stable"
    ).reset_index(drop=True)


def _market_contract(
    daily_bar: pd.DataFrame,
    adjustment: pd.DataFrame,
    status: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["trade_date", "security_id"]
    _require_columns(
        daily_bar,
        {
            *keys,
            "open",
            "high",
            "low",
            "close",
            "volume_shares",
            "amount_cny",
        },
        "daily_bar",
    )
    if daily_bar.duplicated(keys).any():
        raise ValueError("daily_bar has duplicate keys")
    frame = daily_bar.copy()
    if not adjustment.empty:
        _require_columns(
            adjustment, {*keys, "factor_type", "adj_factor"}, "adjustment_factor"
        )
        adjustment = adjustment.loc[
            adjustment["factor_type"].astype(str) == "tushare_adj",
            [*keys, "adj_factor"],
        ]
        if adjustment.duplicated(keys).any():
            raise ValueError("adjustment_factor has duplicate keys")
        frame = frame.merge(adjustment, on=keys, how="left", validate="one_to_one")
    else:
        frame["adj_factor"] = pd.NA
    if not status.empty:
        _require_columns(status, {*keys, "is_suspended", "is_st"}, "daily_status")
        selected_status = status[[*keys, "is_suspended", "is_st"]]
        if selected_status.duplicated(keys).any():
            raise ValueError("daily_status has duplicate keys")
        frame = frame.merge(selected_status, on=keys, how="left", validate="one_to_one")
    else:
        frame["is_suspended"] = pd.NA
        frame["is_st"] = pd.NA
    frame["instrument"] = frame["security_id"].map(_to_instrument)
    frame = frame.rename(
        columns={
            "volume_shares": "volume",
            "amount_cny": "amount",
            "is_suspended": "suspend",
        }
    )
    for column in ("limit_up", "limit_down"):
        frame[column] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    for column in ("list_date", "delist_date"):
        frame[column] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    frame["ingested_at"] = frame.get("known_at", pd.NaT)
    columns = [
        "trade_date",
        "instrument",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "adj_factor",
        "suspend",
        "limit_up",
        "limit_down",
        "is_st",
        "list_date",
        "delist_date",
        "source",
        "ingested_at",
    ]
    _require_columns(frame, set(columns), "market contract")
    return (
        frame.loc[:, columns]
        .sort_values(["trade_date", "instrument"], kind="stable")
        .reset_index(drop=True)
    )


def _to_instrument(security_id: object) -> str:
    parts = str(security_id).split(":")
    if len(parts) != 3 or parts[0] != "CN":
        raise ValueError(f"unsupported security ID: {security_id}")
    expected_prefix = {"SSE": "SH", "SZSE": "SZ", "BSE": "BJ"}.get(parts[1])
    try:
        instrument = to_qlib_instrument(parts[2])
    except ValueError as error:
        raise ValueError(f"unsupported security ID: {security_id}") from error
    if expected_prefix is None or not instrument.startswith(expected_prefix):
        raise ValueError(f"unsupported security ID: {security_id}")
    return instrument


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
