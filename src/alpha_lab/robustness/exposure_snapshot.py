from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alpha_lab.data.providers.base import file_sha256
from alpha_lab.research_data.provider import TushareArtifact
from alpha_lab.robustness.config import RobustnessConfig, load_robustness_config
from alpha_lab.robustness.contracts import ExposureSnapshotResult, ExposureTables
from alpha_lab.robustness.exposure_data import (
    _load_phase5_exposure_context,
    _provider_from_environment,
    acquire_exposure_tables,
    validate_exposure_tables,
)

EXPOSURE_SCHEMA_VERSION = 1


def build_exposure_snapshot(config_dir: Path, data_dir: Path) -> ExposureSnapshotResult:
    config, policy_sha256 = load_robustness_config(config_dir / "robustness.yaml")
    phase5_manifest = (
        data_dir / "manifests" / config.phase5_snapshot_id / "manifest.json"
    )
    provider = _provider_from_environment(data_dir, config)
    tables, raw_inputs = acquire_exposure_tables(data_dir, config, provider)
    return materialize_exposure_snapshot(
        data_dir,
        config,
        policy_sha256,
        phase5_manifest,
        tables,
        raw_inputs,
    )


def materialize_exposure_snapshot(
    data_dir: Path,
    config: RobustnessConfig,
    policy_sha256: str,
    phase5_manifest_path: Path,
    tables: ExposureTables,
    raw_inputs: Sequence[TushareArtifact],
) -> ExposureSnapshotResult:
    _read_phase5_manifest(phase5_manifest_path, config)
    phase5_context = _load_phase5_exposure_context(
        data_dir,
        config.phase5_snapshot_id,
        config.warmup.start,
        config.test.end,
    )
    known_security_ids = set(phase5_context["security"]["security_id"].astype(str))
    expected_security_ids = set(
        phase5_context["observations"]["security_id"].astype(str)
    )
    expected_industry_ids = set(tables.industry_definition["industry_id"].astype(str))
    quality = validate_exposure_tables(
        tables,
        known_security_ids,
        expected_security_ids=expected_security_ids,
        expected_industry_ids=expected_industry_ids,
        expected_market_observations=phase5_context["observations"],
        minimum_temporal_coverage=config.minimum_fold_coverage,
        market_start_date=config.warmup.start,
        market_end_date=config.test.end,
    )
    if quality["status"] == "error":
        raise ValueError("exposure data quality gates failed")

    exposure_root = data_dir / "exposures"
    exposure_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".p6x-build-", dir=exposure_root))
    try:
        artifacts = _write_tables(temporary, tables)
        quality_bytes = _canonical_bytes(quality)
        quality_sha256 = hashlib.sha256(quality_bytes).hexdigest()
        raw_identities = [
            _raw_identity(item)
            for item in sorted(
                raw_inputs, key=lambda value: (value.api_name, value.request_sha256)
            )
        ]
        phase5_sha256 = file_sha256(phase5_manifest_path)
        coverage_scope = {
            "start_date": config.warmup.start.isoformat(),
            "end_date": config.test.end.isoformat(),
            "minimum_temporal_coverage": config.minimum_fold_coverage,
        }
        identity = {
            "exposure_schema_version": EXPOSURE_SCHEMA_VERSION,
            "phase5_manifest_sha256": phase5_sha256,
            "policy_sha256": policy_sha256,
            "quality_report_sha256": quality_sha256,
            "coverage_scope": coverage_scope,
            "raw_request_identities": raw_identities,
            "artifacts": [
                {
                    "name": item["name"],
                    "sha256": item["sha256"],
                    "row_count": item["row_count"],
                }
                for item in artifacts
            ],
        }
        identity_sha256 = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
        snapshot_id = f"p6x-{identity_sha256[:20]}"
        snapshot_dir = exposure_root / snapshot_id
        if snapshot_dir.exists():
            _validate_existing(snapshot_dir, temporary, artifacts)
        else:
            os.replace(temporary, snapshot_dir)

        manifest_dir = data_dir / "manifests" / snapshot_id
        quality_path = manifest_dir / "quality_report.json"
        manifest_path = manifest_dir / "manifest.json"
        _write_immutable(quality_path, quality_bytes)
        manifest_artifacts = [
            {
                **item,
                "path": (
                    Path("exposures") / snapshot_id / str(item["name"])
                ).as_posix(),
            }
            for item in artifacts
        ]
        manifest = {
            "schema_version": EXPOSURE_SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "snapshot_type": "point_in_time_exposure",
            "identity_sha256": identity_sha256,
            "phase5_snapshot_id": config.phase5_snapshot_id,
            "phase5_manifest_sha256": phase5_sha256,
            "policy_sha256": policy_sha256,
            "quality_status": quality["status"],
            "coverage_scope": coverage_scope,
            "source": {
                "provider": "tushare",
                "classification_standard": "SW2021",
                "credential_redacted": True,
            },
            "raw_inputs": [
                {**item, "path": _portable_path(data_dir, artifact.parquet_path)}
                for item, artifact in zip(
                    raw_identities,
                    sorted(
                        raw_inputs,
                        key=lambda value: (value.api_name, value.request_sha256),
                    ),
                    strict=True,
                )
            ],
            "artifacts": manifest_artifacts,
            "quality_report": {
                "path": _portable_path(data_dir, quality_path),
                "sha256": quality_sha256,
            },
        }
        _write_immutable(manifest_path, _canonical_bytes(manifest))
        validation = validate_exposure_snapshot(data_dir, snapshot_id)
        if not validation["healthy"]:
            failures = validation["failures"]
            raise ValueError(
                f"published exposure snapshot failed validation: {failures}"
            )
        _write_mutable(
            data_dir / "state" / "latest_exposure_snapshot.txt",
            f"{snapshot_id}\n".encode(),
        )
        return ExposureSnapshotResult(
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


def validate_exposure_snapshot(data_dir: Path, snapshot_id: str) -> dict[str, object]:
    manifest_path = data_dir / "manifests" / snapshot_id / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"exposure snapshot manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("snapshot_id") != snapshot_id
        or manifest.get("snapshot_type") != "point_in_time_exposure"
        or not snapshot_id.startswith("p6x-")
    ):
        raise ValueError("exposure snapshot manifest identity mismatch")
    checked = 0
    failures: list[str] = []
    quality_reference = manifest.get("quality_report")
    identity = {
        "exposure_schema_version": manifest.get("schema_version"),
        "phase5_manifest_sha256": manifest.get("phase5_manifest_sha256"),
        "policy_sha256": manifest.get("policy_sha256"),
        "quality_report_sha256": (
            quality_reference.get("sha256")
            if isinstance(quality_reference, dict)
            else None
        ),
        "coverage_scope": manifest.get("coverage_scope"),
        "raw_request_identities": [
            {
                key: item.get(key)
                for key in (
                    "api_name",
                    "request_sha256",
                    "sha256",
                    "row_count",
                    "params",
                    "fields",
                )
            }
            for item in manifest.get("raw_inputs", [])
        ],
        "artifacts": [
            {key: item.get(key) for key in ("name", "sha256", "row_count")}
            for item in manifest.get("artifacts", [])
        ],
    }
    actual_identity = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    if (
        manifest.get("identity_sha256") != actual_identity
        or snapshot_id != f"p6x-{actual_identity[:20]}"
    ):
        failures.append("identity_sha256")
    failures.extend(_phase5_dependency_failures(data_dir, manifest))
    failures.extend(_artifact_layout_failures(data_dir, manifest))
    for item in [*manifest.get("raw_inputs", []), *manifest.get("artifacts", [])]:
        path = data_dir / str(item["path"])
        checked += 1
        if not path.is_file():
            failures.append(f"missing:{item['path']}")
        elif file_sha256(path) != str(item["sha256"]):
            failures.append(f"sha256:{item['path']}")
    quality = quality_reference
    if not isinstance(quality, dict):
        failures.append("missing:quality_report")
    else:
        path = data_dir / str(quality["path"])
        checked += 1
        if not path.is_file():
            failures.append(f"missing:{quality['path']}")
        elif file_sha256(path) != str(quality["sha256"]):
            failures.append(f"sha256:{quality['path']}")
        else:
            try:
                quality_document = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                failures.append("quality_schema")
            else:
                failures.extend(_quality_report_failures(quality_document, manifest))
                try:
                    recomputed_quality = _recompute_quality_report(data_dir, manifest)
                except (KeyError, TypeError, ValueError, OSError):
                    failures.append("quality_recomputed")
                else:
                    if _canonical_bytes(recomputed_quality) != _canonical_bytes(
                        quality_document
                    ):
                        failures.append("quality_recomputed")
    return {
        "snapshot_id": snapshot_id,
        "healthy": not failures and manifest.get("quality_status") != "error",
        "quality_status": manifest.get("quality_status"),
        "checked_artifact_count": checked,
        "failures": failures,
        "manifest_sha256": file_sha256(manifest_path),
    }


def _write_tables(root: Path, tables: ExposureTables) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    market = tables.market_cap
    if market.empty:
        artifacts.append(
            _write_frame(
                root, "market_cap/part.parquet", market, ["trade_date", "security_id"]
            )
        )
    else:
        years = pd.to_datetime(market["trade_date"], errors="raise").dt.year
        for year in sorted(years.unique()):
            artifacts.append(
                _write_frame(
                    root,
                    f"market_cap/year={int(year)}/part.parquet",
                    market.loc[years == year].copy(),
                    ["trade_date", "security_id"],
                )
            )
    artifacts.append(
        _write_frame(
            root,
            "industry_definition.parquet",
            tables.industry_definition,
            ["industry_id"],
        )
    )
    artifacts.append(
        _write_frame(
            root,
            "industry_membership.parquet",
            tables.industry_membership,
            ["security_id", "effective_from", "industry_id"],
        )
    )
    return sorted(artifacts, key=lambda item: str(item["name"]))


def _artifact_layout_failures(data_dir: Path, manifest: dict[str, Any]) -> list[str]:
    try:
        scope = manifest["coverage_scope"]
        start_date = date.fromisoformat(str(scope["start_date"]))
        end_date = date.fromisoformat(str(scope["end_date"]))
        snapshot_id = str(manifest["snapshot_id"])
        phase5_snapshot_id = str(manifest["phase5_snapshot_id"])
        artifacts = manifest["artifacts"]
        if not isinstance(artifacts, list) or not all(
            isinstance(item, dict) for item in artifacts
        ):
            raise ValueError("invalid artifacts")
        names = [str(item["name"]) for item in artifacts]
        if len(names) != len(set(names)):
            raise ValueError("duplicate artifacts")
        for fixed_name in (
            "industry_definition.parquet",
            "industry_membership.parquet",
        ):
            if names.count(fixed_name) != 1:
                raise ValueError("missing fixed artifact")
        market_pattern = re.compile(r"^market_cap/year=([0-9]{4})/part[.]parquet$")
        market_items: list[tuple[int, dict[str, Any]]] = []
        for item, name in zip(artifacts, names, strict=True):
            expected_path = (Path("exposures") / snapshot_id / name).as_posix()
            if item.get("path") != expected_path:
                raise ValueError("artifact path mismatch")
            if name.startswith("market_cap/"):
                matched = market_pattern.fullmatch(name)
                if matched is None:
                    raise ValueError("invalid market partition grammar")
                year = int(matched.group(1))
                if year < start_date.year or year > end_date.year:
                    raise ValueError("market partition outside scope")
                market_items.append((year, item))
        if not market_items or len({year for year, _ in market_items}) != len(
            market_items
        ):
            raise ValueError("duplicate or missing market partition")
        context = _load_phase5_exposure_context(
            data_dir, phase5_snapshot_id, start_date, end_date
        )
        expected_years = set(
            pd.to_datetime(
                context["observations"]["trade_date"], errors="raise"
            ).dt.year
        )
        if {year for year, _ in market_items} != expected_years:
            raise ValueError("market partition years are incomplete")
        for year, item in market_items:
            frame = _read_verified_frame(data_dir, item)
            row_years = set(pd.to_datetime(frame["trade_date"], errors="raise").dt.year)
            if row_years != {year}:
                raise ValueError("market row year mismatches partition")
        snapshot_dir = data_dir / "exposures" / snapshot_id
        actual_files = {
            path.relative_to(snapshot_dir).as_posix()
            for path in snapshot_dir.rglob("*")
            if path.is_file()
        }
        if actual_files != set(names):
            raise ValueError("snapshot directory is not manifest-closed")
    except (KeyError, TypeError, ValueError, OSError):
        return ["artifact_layout"]
    return []


def _recompute_quality_report(
    data_dir: Path, manifest: dict[str, Any]
) -> dict[str, object]:
    scope = manifest.get("coverage_scope")
    if not isinstance(scope, dict) or set(scope) != {
        "start_date",
        "end_date",
        "minimum_temporal_coverage",
    }:
        raise ValueError("invalid coverage scope")
    start_date = date.fromisoformat(str(scope["start_date"]))
    end_date = date.fromisoformat(str(scope["end_date"]))
    threshold = float(scope["minimum_temporal_coverage"])
    if threshold != 0.70 or end_date < start_date:
        raise ValueError("invalid coverage policy")
    phase5_snapshot_id = manifest.get("phase5_snapshot_id")
    if not isinstance(phase5_snapshot_id, str):
        raise ValueError("invalid Phase 5 snapshot")
    context = _load_phase5_exposure_context(
        data_dir, phase5_snapshot_id, start_date, end_date
    )
    artifacts = {
        str(item["name"]): item
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict) and "name" in item
    }
    market_parts = [
        _read_verified_frame(data_dir, item)
        for name, item in sorted(artifacts.items())
        if name.startswith("market_cap/")
    ]
    if not market_parts:
        raise ValueError("market cap artifacts are missing")
    definitions = _read_verified_frame(
        data_dir, artifacts["industry_definition.parquet"]
    )
    membership = _read_verified_frame(
        data_dir, artifacts["industry_membership.parquet"]
    )
    tables = ExposureTables(
        market_cap=pd.concat(market_parts, ignore_index=True),
        industry_definition=definitions,
        industry_membership=membership,
    )
    observations = context["observations"]
    return validate_exposure_tables(
        tables,
        set(context["security"]["security_id"].astype(str)),
        expected_security_ids=set(observations["security_id"].astype(str)),
        expected_industry_ids=set(definitions["industry_id"].astype(str)),
        expected_market_observations=observations,
        minimum_temporal_coverage=threshold,
        market_start_date=start_date,
        market_end_date=end_date,
    )


def _read_verified_frame(data_dir: Path, artifact: dict[str, Any]) -> pd.DataFrame:
    path = data_dir / str(artifact["path"])
    if not path.is_file() or file_sha256(path) != str(artifact["sha256"]):
        raise ValueError("artifact checksum mismatch")
    return pd.read_parquet(path)


def _phase5_dependency_failures(
    data_dir: Path, exposure_manifest: dict[str, Any]
) -> list[str]:
    failures: list[str] = []
    phase5_snapshot_id = exposure_manifest.get("phase5_snapshot_id")
    if not isinstance(phase5_snapshot_id, str):
        return ["phase5_snapshot_id"]
    manifest_path = data_dir / "manifests" / phase5_snapshot_id / "manifest.json"
    if not manifest_path.is_file():
        return ["phase5:missing:manifest.json"]
    if file_sha256(manifest_path) != exposure_manifest.get("phase5_manifest_sha256"):
        failures.append("phase5_manifest_sha256")
    try:
        phase5 = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return [*failures, "phase5:manifest_schema"]
    if (
        phase5.get("snapshot_id") != phase5_snapshot_id
        or phase5.get("snapshot_type") != "research_market"
    ):
        failures.append("phase5:manifest_identity")
    artifacts = {
        item.get("name"): item
        for item in phase5.get("artifacts", [])
        if isinstance(item, dict)
    }
    for name in (
        "security_master.parquet",
        "index_membership.parquet",
        "universe_dates.parquet",
    ):
        artifact = artifacts.get(name)
        if not isinstance(artifact, dict):
            failures.append(f"phase5:missing:{name}")
            continue
        path = data_dir / str(artifact.get("path", ""))
        if not path.is_file():
            failures.append(f"phase5:missing:{name}")
        elif file_sha256(path) != artifact.get("sha256"):
            failures.append(f"phase5:sha256:{name}")
    daily_names = sorted(
        str(name)
        for name in artifacts
        if isinstance(name, str) and name.startswith("daily_bar/")
    )
    if not daily_names:
        failures.append("phase5:missing:daily_bar")
    for name in daily_names:
        artifact = artifacts[name]
        path = data_dir / str(artifact.get("path", ""))
        if not path.is_file():
            failures.append(f"phase5:missing:{name}")
        elif file_sha256(path) != artifact.get("sha256"):
            failures.append(f"phase5:sha256:{name}")
    return failures


def _quality_report_failures(quality: object, manifest: dict[str, Any]) -> list[str]:
    if not isinstance(quality, dict):
        return ["quality_schema"]
    required_top_level = {
        "schema_version",
        "policy",
        "status",
        "summary",
        "checks",
    }
    if set(quality) != required_top_level:
        return ["quality_schema"]
    if (
        quality.get("schema_version") != 1
        or quality.get("policy") != "phase6_exposure_quality_v1"
        or not isinstance(quality.get("summary"), dict)
        or not isinstance(quality.get("checks"), dict)
    ):
        return ["quality_schema"]
    expected_checks = {
        "empty_required_table",
        "duplicate_keys",
        "industry_interval_overlap",
        "unknown_security_reference",
        "unknown_industry_reference",
        "invalid_market_cap",
        "missing_security_coverage",
        "missing_industry_coverage",
        "insufficient_temporal_coverage",
        "undercovered_security",
        "market_cap_out_of_scope",
    }
    checks = quality["checks"]
    check_failure = set(checks) != expected_checks
    counts: list[int] = []
    if not check_failure:
        for item in checks.values():
            if not isinstance(item, dict) or set(item) != {
                "severity",
                "status",
                "count",
            }:
                check_failure = True
                break
            count = item.get("count")
            if (
                item.get("severity") != "error"
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
                or item.get("status") != ("pass" if count == 0 else "fail")
            ):
                check_failure = True
                break
            counts.append(count)
    failures = ["quality_checks"] if check_failure else []
    derived_status = "error" if any(counts) or check_failure else "pass"
    if (
        quality.get("status") != derived_status
        or quality.get("status") != manifest.get("quality_status")
        or quality.get("status") == "error"
    ):
        failures.append("quality_status")

    summary = quality["summary"]
    artifacts = {
        item.get("name"): item
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict)
    }
    market_count = sum(
        int(item.get("row_count", -1))
        for name, item in artifacts.items()
        if isinstance(name, str) and name.startswith("market_cap/")
    )
    expected_summary_counts = {
        "market_cap_count": market_count,
        "industry_definition_count": _artifact_row_count(
            artifacts, "industry_definition.parquet"
        ),
        "industry_membership_count": _artifact_row_count(
            artifacts, "industry_membership.parquet"
        ),
    }
    expected_observations = summary.get("expected_observation_count", -1)
    observed_observations = summary.get("observed_observation_count", -1)
    reported_ratio = summary.get("temporal_coverage_ratio")
    derived_ratio = (
        observed_observations / expected_observations
        if isinstance(expected_observations, int)
        and expected_observations > 0
        and isinstance(observed_observations, int)
        else 1.0
    )
    if (
        set(summary)
        != {
            *expected_summary_counts,
            "expected_security_count",
            "expected_industry_count",
            "expected_observation_count",
            "observed_observation_count",
            "temporal_coverage_ratio",
            "minimum_temporal_coverage",
        }
        or any(
            summary.get(key) != value for key, value in expected_summary_counts.items()
        )
        or any(
            not isinstance(summary.get(key), int) or summary[key] < 0
            for key in (
                "expected_security_count",
                "expected_industry_count",
                "expected_observation_count",
                "observed_observation_count",
            )
        )
        or not isinstance(summary.get("temporal_coverage_ratio"), (int, float))
        or not 0 <= summary["temporal_coverage_ratio"] <= 1
        or not isinstance(summary.get("minimum_temporal_coverage"), (int, float))
        or not 0 <= summary["minimum_temporal_coverage"] <= 1
        or summary.get("observed_observation_count", 0)
        > summary.get("expected_observation_count", 0)
        or not isinstance(reported_ratio, (int, float))
        or abs(float(reported_ratio) - derived_ratio) > 1e-12
    ):
        failures.append("quality_row_counts")
    return failures


def _artifact_row_count(artifacts: dict[object, dict[str, Any]], name: str) -> int:
    artifact = artifacts.get(name)
    if not isinstance(artifact, dict):
        return -1
    value = artifact.get("row_count")
    return value if isinstance(value, int) and not isinstance(value, bool) else -1


def _write_frame(
    root: Path, name: str, frame: pd.DataFrame, sort_keys: list[str]
) -> dict[str, object]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.sort_values(sort_keys, kind="stable").reset_index(drop=True)
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


def _read_phase5_manifest(path: Path, config: RobustnessConfig) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Phase 5 manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("snapshot_id") != config.phase5_snapshot_id
        or manifest.get("snapshot_type") != "research_market"
    ):
        raise ValueError("Phase 5 manifest identity mismatch")
    return cast(dict[str, Any], manifest)


def _known_security_ids(data_dir: Path, phase5_manifest: dict[str, Any]) -> set[str]:
    artifact = next(
        (
            item
            for item in phase5_manifest.get("artifacts", [])
            if item.get("name") == "security_master.parquet"
        ),
        None,
    )
    if artifact is None:
        raise ValueError("Phase 5 manifest missing security_master.parquet")
    path = data_dir / str(artifact["path"])
    if file_sha256(path) != str(artifact["sha256"]):
        raise ValueError("Phase 5 security master checksum mismatch")
    frame = pd.read_parquet(path, columns=["security_id"])
    return set(frame["security_id"].astype(str))


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


def _validate_existing(
    destination: Path,
    temporary: Path,
    artifacts: list[dict[str, object]],
) -> None:
    for item in artifacts:
        relative = Path(str(item["name"]))
        existing = destination / relative
        candidate = temporary / relative
        if not existing.is_file() or file_sha256(existing) != file_sha256(candidate):
            raise RuntimeError(f"immutable exposure snapshot differs: {destination}")


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


def _portable_path(data_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(data_dir).as_posix()
    except ValueError:
        return path.name
