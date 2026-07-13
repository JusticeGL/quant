from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, cast

import yaml

from alpha_lab.baseline.config import CostConfig
from alpha_lab.factors.contract import FactorMetadata
from alpha_lab.quality_contracts import (
    exposure_quality_failures,
    phase5_quality_failures,
)
from alpha_lab.research_data.config import load_research_data_config
from alpha_lab.robustness.config import load_robustness_config
from alpha_lab.robustness.contracts import FrozenCandidate

FREEZE_SCHEMA_VERSION = 1
FIXED_PHASE5_SNAPSHOT_ID = "p5-ecaa6e8aeae6b9f8fb25"


def freeze_candidate(
    factor_id: str,
    config_dir: Path,
    data_dir: Path,
    experiments_dir: Path,
) -> FrozenCandidate:
    repo_root = config_dir.parent
    _trusted_directory(repo_root, repo_root, "repository root")
    _trusted_directory(experiments_dir, repo_root, "experiments output", required=False)
    payload = _current_payload(factor_id, config_dir, data_dir)
    identity_sha256 = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    freeze_id = f"freeze-{identity_sha256}"
    document = {
        **payload,
        "freeze_id": freeze_id,
        "identity_sha256": identity_sha256,
    }
    content = _canonical_bytes(document, newline=True)
    path = experiments_dir / "phase6" / freeze_id / "freeze.json"
    _write_immutable(path, content, experiments_dir, repo_root)
    return FrozenCandidate(
        freeze_id=freeze_id,
        factor_id=cast(Any, factor_id),
        freeze_path=path,
        freeze_sha256=hashlib.sha256(content).hexdigest(),
    )


def validate_freeze(
    freeze_path: Path, config_dir: Path, data_dir: Path
) -> dict[str, object]:
    if freeze_path.name != "freeze.json":
        raise ValueError("freeze path must end in freeze.json")
    repo_root = config_dir.parent
    if freeze_path.parent.parent.name != "phase6":
        raise ValueError("freeze path must use the phase6 artifact layout")
    experiments_dir = freeze_path.parent.parent.parent
    _trusted_directory(repo_root, repo_root, "repository root")
    _trusted_directory(experiments_dir, repo_root, "experiments output")
    _trusted_file(freeze_path, experiments_dir, "freeze artifact")
    try:
        content = freeze_path.read_bytes()
        document = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("freeze artifact is missing or malformed") from error
    if not isinstance(document, dict):
        raise ValueError("freeze artifact must be a mapping")
    expected_keys = {
        "schema_version",
        "factor",
        "snapshots",
        "policies",
        "test",
        "git_commit",
        "freeze_id",
        "identity_sha256",
    }
    if set(document) != expected_keys or document.get("schema_version") != 1:
        raise ValueError("freeze artifact schema mismatch")
    if content != _canonical_bytes(document, newline=True):
        raise ValueError("freeze artifact is not canonical JSON")
    factor = _validate_freeze_document_schema(document)
    identity_fields = expected_keys - {"freeze_id", "identity_sha256"}
    payload = {key: document[key] for key in identity_fields}
    actual_identity = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    expected_freeze_id = f"freeze-{actual_identity}"
    if (
        document.get("identity_sha256") != actual_identity
        or document.get("freeze_id") != expected_freeze_id
        or freeze_path.parent.name != expected_freeze_id
    ):
        raise ValueError("freeze identity mismatch")
    current = _current_payload(str(factor.get("factor_id")), config_dir, data_dir)
    _require_current_dependencies(payload, current)
    return {
        **document,
        "healthy": True,
        "freeze_sha256": hashlib.sha256(content).hexdigest(),
    }


def _current_payload(
    factor_id: str, config_dir: Path, data_dir: Path
) -> dict[str, object]:
    repo_root = config_dir.parent
    _trusted_directory(repo_root, repo_root, "repository root")
    _trusted_directory(config_dir, repo_root, "configuration directory")
    _trusted_directory(data_dir, repo_root, "data directory")
    robustness_path = config_dir / "robustness.yaml"
    _trusted_file(robustness_path, config_dir, "robustness policy")
    robustness, robustness_sha256 = load_robustness_config(robustness_path)
    if robustness.phase5_snapshot_id != FIXED_PHASE5_SNAPSHOT_ID:
        raise ValueError(
            f"fixed Phase 5 snapshot must remain {FIXED_PHASE5_SNAPSHOT_ID}"
        )
    if factor_id not in robustness.factor_ids:
        raise PermissionError(
            f"{factor_id} is not an approved Phase 6 candidate; "
            "only F1002 and F1003 are allowed"
        )
    candidate_dir = repo_root / "src" / "alpha_lab" / "factors" / "candidates"
    _trusted_directory(candidate_dir, repo_root, "factor candidate directory")
    source_path = candidate_dir / f"{factor_id}.py"
    metadata_path = candidate_dir / f"{factor_id}.yaml"
    _trusted_file(source_path, repo_root, "factor source")
    _trusted_file(metadata_path, repo_root, "factor metadata")
    try:
        metadata_raw = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        metadata = FactorMetadata.model_validate(metadata_raw)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"factor metadata is invalid: {factor_id}") from error
    if metadata.factor_id != factor_id or metadata.status != "candidate":
        raise ValueError(f"factor metadata identity is invalid: {factor_id}")

    phase5_path = (
        data_dir / "manifests" / robustness.phase5_snapshot_id / "manifest.json"
    )
    _validate_phase5_manifest_structure(
        phase5_path,
        data_dir=data_dir,
        config_dir=config_dir,
        snapshot_id=robustness.phase5_snapshot_id,
    )
    phase5_sha256 = _sha256(phase5_path)

    exposure_id = _latest_exposure_snapshot_id(data_dir)
    exposure_path = data_dir / "manifests" / exposure_id / "manifest.json"
    exposure = _validate_exposure_manifest_structure(
        exposure_path,
        data_dir=data_dir,
        snapshot_id=exposure_id,
        phase5_snapshot_id=robustness.phase5_snapshot_id,
        phase5_manifest_sha256=phase5_sha256,
        policy_sha256=robustness_sha256,
        expected_scope={
            "start_date": robustness.warmup.start.isoformat(),
            "end_date": robustness.test.end.isoformat(),
            "minimum_temporal_coverage": robustness.minimum_fold_coverage,
        },
    )
    if (
        exposure.get("phase5_snapshot_id") != robustness.phase5_snapshot_id
        or exposure.get("phase5_manifest_sha256") != phase5_sha256
    ):
        raise ValueError("Phase 5 manifest drift: exposure snapshot dependency differs")
    if exposure.get("policy_sha256") != robustness_sha256:
        raise ValueError(
            "robustness policy drift: exposure snapshot dependency differs"
        )

    costs_path = config_dir / "costs.yaml"
    _trusted_file(costs_path, config_dir, "cost policy")
    costs_sha256 = _cost_policy_sha256(costs_path)
    commit = _git_commit(repo_root)
    _require_candidate_at_commit(repo_root, commit, source_path, metadata_path)
    return {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "factor": {
            "factor_id": factor_id,
            "source_path": source_path.relative_to(repo_root).as_posix(),
            "source_sha256": _sha256(source_path),
            "metadata_path": metadata_path.relative_to(repo_root).as_posix(),
            "metadata_sha256": _sha256(metadata_path),
        },
        "snapshots": {
            "phase5": {
                "snapshot_id": robustness.phase5_snapshot_id,
                "manifest_path": phase5_path.relative_to(data_dir).as_posix(),
                "manifest_sha256": phase5_sha256,
            },
            "exposure": {
                "snapshot_id": exposure_id,
                "manifest_path": exposure_path.relative_to(data_dir).as_posix(),
                "manifest_sha256": _sha256(exposure_path),
            },
        },
        "policies": {
            "robustness": {
                "path": (config_dir / "robustness.yaml")
                .relative_to(repo_root)
                .as_posix(),
                "sha256": robustness_sha256,
            },
            "costs": {
                "path": costs_path.relative_to(repo_root).as_posix(),
                "sha256": costs_sha256,
            },
        },
        "test": {
            "start": robustness.test.start.isoformat(),
            "end": robustness.test.end.isoformat(),
            "access": robustness.test.access,
        },
        "git_commit": commit,
    }


def _validate_phase5_manifest_structure(
    path: Path,
    *,
    data_dir: Path,
    config_dir: Path,
    snapshot_id: str,
) -> dict[str, Any]:
    label = "Phase 5 manifest"
    _trusted_file(path, data_dir, label)
    manifest = _read_canonical_json(path, label)
    _require_keys(
        manifest,
        {
            "schema_version",
            "snapshot_id",
            "snapshot_type",
            "identity_sha256",
            "quality_status",
            "source",
            "scope",
            "summary",
            "raw_inputs",
            "artifacts",
            "quality_report",
        },
        label,
    )
    if (
        manifest["schema_version"] != 1
        or manifest["snapshot_id"] != snapshot_id
        or manifest["snapshot_type"] != "research_market"
        or re.fullmatch(r"p5-[0-9a-f]{20}", snapshot_id) is None
        or manifest["quality_status"] not in {"pass", "warning"}
        or manifest["source"] != {"provider": "tushare", "credential_redacted": True}
    ):
        raise ValueError(f"{label} schema or identity is invalid")
    raw_inputs = _validate_raw_inputs(manifest["raw_inputs"], data_dir, label)
    artifacts = _validate_snapshot_artifacts(
        manifest["artifacts"],
        data_dir=data_dir,
        root="research",
        snapshot_id=snapshot_id,
        snapshot_kind="phase5",
        label=label,
    )
    research_config_path = config_dir / "research_data.yaml"
    _trusted_file(research_config_path, config_dir, "research data policy")
    config = load_research_data_config(config_dir)
    identity = {
        "research_schema_version": 1,
        "config": config.model_dump(mode="json"),
        "raw_inputs": raw_inputs,
        "artifacts": artifacts,
    }
    identity_sha256 = hashlib.sha256(
        _canonical_bytes(identity, newline=True)
    ).hexdigest()
    if (
        manifest["identity_sha256"] != identity_sha256
        or snapshot_id != f"p5-{identity_sha256[:20]}"
    ):
        raise ValueError(f"{label} content identity is invalid")
    quality = _validate_quality_reference(
        data_dir,
        snapshot_id,
        manifest["quality_report"],
        manifest=manifest,
        snapshot_kind="phase5",
        expected_status=str(manifest["quality_status"]),
        expected_policy="phase5_point_in_time_quality_v1",
        label=label,
    )
    if manifest["scope"] != quality.get("scope") or manifest["summary"] != quality.get(
        "summary"
    ):
        raise ValueError(f"{label} quality summary is detached")
    return manifest


def _validate_exposure_manifest_structure(
    path: Path,
    *,
    data_dir: Path,
    snapshot_id: str,
    phase5_snapshot_id: str,
    phase5_manifest_sha256: str,
    policy_sha256: str,
    expected_scope: dict[str, object],
) -> dict[str, Any]:
    label = "exposure manifest"
    _trusted_file(path, data_dir, label)
    manifest = _read_canonical_json(path, label)
    _require_keys(
        manifest,
        {
            "schema_version",
            "snapshot_id",
            "snapshot_type",
            "identity_sha256",
            "phase5_snapshot_id",
            "phase5_manifest_sha256",
            "policy_sha256",
            "quality_status",
            "coverage_scope",
            "source",
            "raw_inputs",
            "artifacts",
            "quality_report",
        },
        label,
    )
    if (
        manifest["schema_version"] != 1
        or manifest["snapshot_id"] != snapshot_id
        or manifest["snapshot_type"] != "point_in_time_exposure"
        or re.fullmatch(r"p6x-[0-9a-f]{20}", snapshot_id) is None
        or manifest["phase5_snapshot_id"] != phase5_snapshot_id
        or manifest["phase5_manifest_sha256"] != phase5_manifest_sha256
        or manifest["policy_sha256"] != policy_sha256
        or manifest["quality_status"] != "pass"
        or manifest["coverage_scope"] != expected_scope
        or manifest["source"]
        != {
            "provider": "tushare",
            "classification_standard": "SW2021",
            "credential_redacted": True,
        }
    ):
        raise ValueError(
            f"{label} schema, Phase 5/robustness policy links, "
            "or quality status is invalid"
        )
    raw_inputs = _validate_raw_inputs(manifest["raw_inputs"], data_dir, label)
    artifacts = _validate_snapshot_artifacts(
        manifest["artifacts"],
        data_dir=data_dir,
        root="exposures",
        snapshot_id=snapshot_id,
        snapshot_kind="exposure",
        label=label,
    )
    quality = _validate_quality_reference(
        data_dir,
        snapshot_id,
        manifest["quality_report"],
        manifest=manifest,
        snapshot_kind="exposure",
        expected_status="pass",
        expected_policy="phase6_exposure_quality_v1",
        label=label,
    )
    identity = {
        "exposure_schema_version": 1,
        "phase5_manifest_sha256": phase5_manifest_sha256,
        "policy_sha256": policy_sha256,
        "quality_report_sha256": manifest["quality_report"]["sha256"],
        "coverage_scope": expected_scope,
        "raw_request_identities": raw_inputs,
        "artifacts": artifacts,
    }
    identity_sha256 = hashlib.sha256(
        _canonical_bytes(identity, newline=True)
    ).hexdigest()
    if (
        manifest["identity_sha256"] != identity_sha256
        or snapshot_id != f"p6x-{identity_sha256[:20]}"
        or quality.get("status") != "pass"
    ):
        raise ValueError(f"{label} content identity is invalid")
    return manifest


def _read_canonical_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    try:
        content = path.read_bytes()
        value = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is malformed") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    if content != _canonical_bytes(value, newline=True):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _require_keys(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} schema is invalid")
    return value


def _validate_freeze_document_schema(
    document: dict[str, Any],
) -> dict[str, Any]:
    factor = _require_keys(
        document.get("factor"),
        {
            "factor_id",
            "source_path",
            "source_sha256",
            "metadata_path",
            "metadata_sha256",
        },
        "freeze factor",
    )
    factor_id = factor["factor_id"]
    if (
        not isinstance(factor_id, str)
        or not isinstance(factor["source_path"], str)
        or factor["source_path"] != f"src/alpha_lab/factors/candidates/{factor_id}.py"
        or not _is_sha256(factor["source_sha256"])
        or not isinstance(factor["metadata_path"], str)
        or factor["metadata_path"]
        != f"src/alpha_lab/factors/candidates/{factor_id}.yaml"
        or not _is_sha256(factor["metadata_sha256"])
    ):
        raise ValueError("freeze factor schema is invalid")

    snapshots = _require_keys(
        document.get("snapshots"), {"phase5", "exposure"}, "freeze snapshots"
    )
    phase5 = _require_keys(
        snapshots["phase5"],
        {"snapshot_id", "manifest_path", "manifest_sha256"},
        "freeze Phase 5 snapshot",
    )
    exposure = _require_keys(
        snapshots["exposure"],
        {"snapshot_id", "manifest_path", "manifest_sha256"},
        "freeze exposure snapshot",
    )
    _validate_frozen_snapshot(phase5, "p5", "freeze Phase 5 snapshot")
    _validate_frozen_snapshot(exposure, "p6x", "freeze exposure snapshot")

    policies = _require_keys(
        document.get("policies"), {"robustness", "costs"}, "freeze policies"
    )
    for name in ("robustness", "costs"):
        policy = _require_keys(
            policies[name], {"path", "sha256"}, f"freeze {name} policy"
        )
        filename = "robustness.yaml" if name == "robustness" else "costs.yaml"
        expected_path = f"config/{filename}"
        if policy["path"] != expected_path or not _is_sha256(policy["sha256"]):
            raise ValueError(f"freeze {name} policy schema is invalid")

    test = _require_keys(
        document.get("test"), {"start", "end", "access"}, "freeze test boundary"
    )
    if (
        not all(
            isinstance(test[key], str)
            and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", test[key])
            for key in ("start", "end")
        )
        or test["access"] != "human_approval_only"
    ):
        raise ValueError("freeze test boundary schema is invalid")
    if (
        not isinstance(document.get("git_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", document["git_commit"]) is None
    ):
        raise ValueError("freeze Git commit schema is invalid")
    if not _is_sha256(document.get("identity_sha256")) or not isinstance(
        document.get("freeze_id"), str
    ):
        raise ValueError("freeze identity schema is invalid")
    return factor


def _validate_frozen_snapshot(value: dict[str, Any], prefix: str, label: str) -> None:
    snapshot_id = value["snapshot_id"]
    if (
        not isinstance(snapshot_id, str)
        or re.fullmatch(rf"{prefix}-[0-9a-f]{{20}}", snapshot_id) is None
        or value["manifest_path"] != f"manifests/{snapshot_id}/manifest.json"
        or not _is_sha256(value["manifest_sha256"])
    ):
        raise ValueError(f"{label} schema is invalid")


def _validate_raw_inputs(
    value: object, data_dir: Path, label: str
) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} raw input schema is invalid")
    result: list[dict[str, object]] = []
    ordering: list[tuple[str, str]] = []
    for item in value:
        document = _require_keys(
            item,
            {
                "api_name",
                "request_sha256",
                "sha256",
                "row_count",
                "params",
                "fields",
                "path",
            },
            f"{label} raw input",
        )
        if (
            not isinstance(document["api_name"], str)
            or not _is_sha256(document["request_sha256"])
            or not _is_sha256(document["sha256"])
            or not _is_row_count(document["row_count"])
            or not isinstance(document["params"], dict)
            or not isinstance(document["fields"], list)
            or not document["fields"]
            or not all(isinstance(field, str) for field in document["fields"])
            or not isinstance(document["path"], str)
            or not document["path"].startswith("raw/")
        ):
            raise ValueError(f"{label} raw input schema is invalid")
        path = data_dir / document["path"]
        _trusted_file(path, data_dir, f"{label} raw input")
        if _sha256(path) != document["sha256"]:
            raise ValueError(f"{label} raw input checksum differs from manifest")
        ordering.append((document["api_name"], document["request_sha256"]))
        result.append(
            {
                key: document[key]
                for key in (
                    "api_name",
                    "request_sha256",
                    "sha256",
                    "row_count",
                    "params",
                    "fields",
                )
            }
        )
    if ordering != sorted(ordering) or len(ordering) != len(set(ordering)):
        raise ValueError(f"{label} raw input ordering is invalid")
    return result


def _validate_snapshot_artifacts(
    value: object,
    *,
    data_dir: Path,
    root: str,
    snapshot_id: str,
    snapshot_kind: str,
    label: str,
) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} artifact schema is invalid")
    names: list[str] = []
    result: list[dict[str, object]] = []
    for item in value:
        document = _require_keys(
            item,
            {"name", "format", "sha256", "row_count", "path"},
            f"{label} artifact",
        )
        name = document["name"]
        if (
            not isinstance(name, str)
            or document["format"] != "parquet"
            or not _is_sha256(document["sha256"])
            or not _is_row_count(document["row_count"])
            or document["path"] != f"{root}/{snapshot_id}/{name}"
        ):
            raise ValueError(f"{label} artifact schema is invalid")
        path = data_dir / str(document["path"])
        _trusted_file(path, data_dir, f"{label} artifact {name}")
        if _sha256(path) != document["sha256"]:
            raise ValueError(f"{label} artifact checksum differs from manifest: {name}")
        names.append(name)
        result.append({key: document[key] for key in ("name", "sha256", "row_count")})
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError(f"{label} artifact ordering is invalid")
    if snapshot_kind == "phase5":
        _validate_phase5_artifact_names(names, label)
    else:
        _validate_exposure_artifact_names(names, label)
    snapshot_dir = data_dir / root / snapshot_id
    _trusted_directory(snapshot_dir, data_dir, f"{label} artifact directory")
    discovered = list(snapshot_dir.rglob("*"))
    for discovered_path in discovered:
        _trusted_path(
            discovered_path,
            data_dir,
            f"{label} artifact namespace",
            kind=None,
        )
    actual = {
        path.relative_to(snapshot_dir).as_posix()
        for path in discovered
        if path.is_file()
    }
    if actual != set(names):
        raise ValueError(f"{label} artifact namespace is not closed")
    return result


def _validate_phase5_artifact_names(names: list[str], label: str) -> None:
    fixed = {
        "security_master.parquet",
        "security_name_history.parquet",
        "trading_calendar.parquet",
        "index_membership.parquet",
        "suspension.parquet",
        "universe_dates.parquet",
    }
    if not fixed.issubset(names):
        raise ValueError(f"{label} fixed artifact namespace is incomplete")
    allowed = set(fixed)
    for dataset in ("daily_bar", "adjustment_factor", "daily_status"):
        matching = [name for name in names if name.startswith(f"{dataset}/")]
        year_pattern = re.compile(
            rf"^{re.escape(dataset)}/year=[0-9]{{4}}/part[.]parquet$"
        )
        if matching == [f"{dataset}/part.parquet"] or (
            matching and all(year_pattern.fullmatch(name) for name in matching)
        ):
            allowed.update(matching)
        else:
            raise ValueError(f"{label} {dataset} artifact namespace is invalid")
    if set(names) != allowed:
        raise ValueError(f"{label} has an unexpected artifact")


def _validate_exposure_artifact_names(names: list[str], label: str) -> None:
    fixed = {"industry_definition.parquet", "industry_membership.parquet"}
    market = [name for name in names if name.startswith("market_cap/")]
    pattern = re.compile(r"^market_cap/year=[0-9]{4}/part[.]parquet$")
    if (
        not fixed.issubset(names)
        or not market
        or not all(pattern.fullmatch(name) for name in market)
        or set(names) != fixed | set(market)
    ):
        raise ValueError(f"{label} artifact namespace is invalid")


def _validate_quality_reference(
    data_dir: Path,
    snapshot_id: str,
    value: object,
    *,
    manifest: dict[str, Any],
    snapshot_kind: str,
    expected_status: str,
    expected_policy: str,
    label: str,
) -> dict[str, Any]:
    reference = _require_keys(value, {"path", "sha256"}, f"{label} quality")
    expected_path = f"manifests/{snapshot_id}/quality_report.json"
    if reference["path"] != expected_path or not _is_sha256(reference["sha256"]):
        raise ValueError(f"{label} quality reference is invalid")
    path = data_dir / expected_path
    _trusted_file(path, data_dir, f"{label} quality report")
    quality = _read_canonical_json(path, f"{label} quality report")
    if _sha256(path) != reference["sha256"]:
        raise ValueError(f"{label} quality report checksum differs")
    if (
        quality.get("schema_version") != 1
        or quality.get("policy") != expected_policy
        or quality.get("status") != expected_status
        or not isinstance(quality.get("summary"), dict)
        or not isinstance(quality.get("checks"), dict)
    ):
        raise ValueError(f"{label} quality report schema is invalid")
    failures = (
        phase5_quality_failures(quality, manifest)
        if snapshot_kind == "phase5"
        else exposure_quality_failures(quality, manifest)
    )
    if failures:
        raise ValueError(f"{label} quality contract is invalid: {', '.join(failures)}")
    return quality


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_row_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _latest_exposure_snapshot_id(data_dir: Path) -> str:
    path = data_dir / "state" / "latest_exposure_snapshot.txt"
    _trusted_file(path, data_dir, "latest exposure snapshot pointer")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("latest exposure snapshot pointer is invalid") from error
    if len(lines) != 1 or re.fullmatch(r"p6x-[A-Za-z0-9]+", lines[0]) is None:
        raise ValueError("latest exposure snapshot pointer is invalid")
    return lines[0]


def _cost_policy_sha256(path: Path) -> str:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        CostConfig.model_validate(raw)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError("cost policy is invalid") from error
    return hashlib.sha256(_canonical_bytes(raw)).hexdigest()


def _git_commit(repo_root: Path) -> str:
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout.strip():
            raise ValueError("cannot freeze or validate with a dirty Git tree")
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Git repository state is unavailable") from error
    value = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value) is None:
        raise ValueError("Git commit identity is invalid")
    return value


def _require_candidate_at_commit(
    repo_root: Path,
    commit: str,
    source_path: Path,
    metadata_path: Path,
) -> None:
    for path in (source_path, metadata_path):
        relative = path.relative_to(repo_root).as_posix()
        try:
            tracked = subprocess.run(
                ["git", "-C", str(repo_root), "ls-tree", commit, "--", relative],
                check=True,
                capture_output=True,
                text=True,
            )
            if not tracked.stdout.strip():
                raise ValueError(f"candidate files must be tracked at HEAD: {relative}")
            committed = subprocess.run(
                ["git", "-C", str(repo_root), "show", f"{commit}:{relative}"],
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError(
                f"candidate files must be tracked at HEAD: {relative}"
            ) from error
        if committed != path.read_bytes():
            raise ValueError(f"candidate file differs from tracked HEAD: {relative}")


def _require_current_dependencies(
    pinned: dict[str, object], current: dict[str, object]
) -> None:
    pinned_factor = cast(dict[str, object], pinned["factor"])
    current_factor = cast(dict[str, object], current["factor"])
    pinned_snapshots = cast(dict[str, dict[str, Any]], pinned["snapshots"])
    current_snapshots = cast(dict[str, dict[str, Any]], current["snapshots"])
    pinned_policies = cast(dict[str, dict[str, Any]], pinned["policies"])
    current_policies = cast(dict[str, dict[str, Any]], current["policies"])
    checks = (
        (
            "factor source",
            pinned_factor.get("source_sha256"),
            current_factor.get("source_sha256"),
        ),
        (
            "factor metadata",
            pinned_factor.get("metadata_sha256"),
            current_factor.get("metadata_sha256"),
        ),
        (
            "Phase 5 manifest",
            pinned_snapshots["phase5"]["manifest_sha256"],
            current_snapshots["phase5"]["manifest_sha256"],
        ),
        (
            "exposure manifest",
            pinned_snapshots["exposure"]["manifest_sha256"],
            current_snapshots["exposure"]["manifest_sha256"],
        ),
        (
            "robustness policy",
            pinned_policies["robustness"]["sha256"],
            current_policies["robustness"]["sha256"],
        ),
        (
            "cost policy",
            pinned_policies["costs"]["sha256"],
            current_policies["costs"]["sha256"],
        ),
        ("fixed test boundary", pinned.get("test"), current.get("test")),
        ("Git commit", pinned.get("git_commit"), current.get("git_commit")),
    )
    for label, expected, actual in checks:
        if expected != actual:
            raise ValueError(f"freeze dependency drift: {label}")
    if pinned != current:
        raise ValueError("freeze dependency drift: canonical payload")


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        + suffix
    ).encode("utf-8")


def _write_immutable(
    path: Path, content: bytes, experiments_dir: Path, repo_root: Path
) -> None:
    _trusted_directory(experiments_dir, repo_root, "experiments output", required=False)
    current = experiments_dir
    for component in ("phase6", path.parent.name):
        if not current.exists():
            current.mkdir()
        _trusted_directory(current, repo_root, "freeze output", required=True)
        current = current / component
    if not current.exists():
        current.mkdir()
    _trusted_directory(current, experiments_dir, "freeze output")
    _trusted_path(path, experiments_dir, "freeze artifact", kind="file", required=False)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable freeze differs: {path}")
        return
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _trusted_file(path: Path, root: Path, label: str) -> Path:
    return _trusted_path(path, root, label, kind="file")


def _trusted_directory(
    path: Path,
    root: Path,
    label: str,
    *,
    required: bool = True,
) -> Path:
    return _trusted_path(path, root, label, kind="directory", required=required)


def _trusted_path(
    path: Path,
    root: Path,
    label: str,
    *,
    kind: str | None,
    required: bool = True,
) -> Path:
    root_absolute = Path(os.path.abspath(root))
    path_absolute = Path(os.path.abspath(path))
    try:
        relative = path_absolute.relative_to(root_absolute)
    except ValueError as error:
        raise ValueError(f"{label} escapes its trusted root") from error

    current = root_absolute
    components = (current, *(current := current / part for part in relative.parts))
    for component in components:
        if component.is_symlink():
            raise ValueError(f"{label} path contains a symlink: {component}")

    exists = path_absolute.exists()
    if required and not exists:
        raise ValueError(f"{label} is missing: {path_absolute}")
    if exists:
        if kind == "file" and not path_absolute.is_file():
            raise ValueError(f"{label} is not a regular file: {path_absolute}")
        if kind == "directory" and not path_absolute.is_dir():
            raise ValueError(f"{label} is not a directory: {path_absolute}")
        try:
            path_absolute.resolve(strict=True).relative_to(
                root_absolute.resolve(strict=True)
            )
        except (OSError, ValueError) as error:
            raise ValueError(f"{label} escapes its trusted root") from error
    return path_absolute


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
