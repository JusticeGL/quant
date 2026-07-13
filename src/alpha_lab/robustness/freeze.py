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
    _write_immutable(path, content)
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
    factor = document.get("factor")
    if not isinstance(factor, dict) or set(factor) != {
        "factor_id",
        "source_path",
        "source_sha256",
        "metadata_path",
        "metadata_sha256",
    }:
        raise ValueError("freeze factor schema mismatch")
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
    robustness, robustness_sha256 = load_robustness_config(
        config_dir / "robustness.yaml"
    )
    if robustness.phase5_snapshot_id != FIXED_PHASE5_SNAPSHOT_ID:
        raise ValueError(
            f"fixed Phase 5 snapshot must remain {FIXED_PHASE5_SNAPSHOT_ID}"
        )
    if factor_id not in robustness.factor_ids:
        raise PermissionError(
            f"{factor_id} is not an approved Phase 6 candidate; "
            "only F1002 and F1003 are allowed"
        )
    repo_root = config_dir.parent
    candidate_dir = repo_root / "src" / "alpha_lab" / "factors" / "candidates"
    source_path = candidate_dir / f"{factor_id}.py"
    metadata_path = candidate_dir / f"{factor_id}.yaml"
    if not source_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"approved factor files are missing: {factor_id}")
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
    phase5 = _manifest_header(
        phase5_path, robustness.phase5_snapshot_id, "research_market"
    )
    del phase5
    phase5_sha256 = _sha256(phase5_path)

    exposure_id = _latest_exposure_snapshot_id(data_dir)
    exposure_path = data_dir / "manifests" / exposure_id / "manifest.json"
    exposure = _manifest_header(exposure_path, exposure_id, "point_in_time_exposure")
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
    costs_sha256 = _cost_policy_sha256(costs_path)
    commit = _git_commit(repo_root)
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


def _manifest_header(
    path: Path, snapshot_id: str, snapshot_type: str
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"snapshot manifest is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"snapshot manifest is malformed: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"snapshot manifest must be a mapping: {path}")
    if (
        value.get("snapshot_id") != snapshot_id
        or value.get("snapshot_type") != snapshot_type
    ):
        raise ValueError(f"snapshot manifest identity mismatch: {path}")
    return value


def _latest_exposure_snapshot_id(data_dir: Path) -> str:
    path = data_dir / "state" / "latest_exposure_snapshot.txt"
    if not path.is_file():
        raise ValueError("latest exposure snapshot pointer is missing")
    lines = path.read_text(encoding="utf-8").splitlines()
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
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value) is None:
        raise ValueError("Git commit identity is invalid")
    return value


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


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
