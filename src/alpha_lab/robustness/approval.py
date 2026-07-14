from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REQUEST_SCHEMA_VERSION = 1
APPROVAL_SCHEMA_VERSION = 1
_GATE_NAMES = {
    "direction_consistency",
    "fold_coverage",
    "double_cost_direction",
    "industry_neutral_retention",
}
_ROBUSTNESS_ARTIFACTS = (
    "walk_forward.json",
    "cost_sensitivity.json",
    "exposure_report.json",
    "robustness_report.md",
)


def create_test_request(freeze_path: Path) -> Path:
    freeze_dir = freeze_path.parent
    if freeze_dir.is_symlink():
        raise ValueError("freeze directory is untrusted")
    freeze = _read_canonical_json(freeze_path, "freeze")
    if freeze_path.name != "freeze.json" or freeze.get("freeze_id") != freeze_dir.name:
        raise ValueError("freeze layout or identity is invalid")
    freeze_sha256 = _sha256(freeze_path)
    artifacts = {
        name: {
            "path": name,
            "sha256": _sha256(_required_regular_file(freeze_dir / name, freeze_dir)),
        }
        for name in _ROBUSTNESS_ARTIFACTS
    }
    walk = _read_canonical_json(freeze_dir / "walk_forward.json", "walk-forward")
    gates = walk.get("gates")
    if (
        walk.get("freeze_id") != freeze.get("freeze_id")
        or walk.get("freeze_sha256") != freeze_sha256
        or walk.get("test_accessed") is not False
        or walk.get("passed") is not True
        or not isinstance(gates, dict)
        or set(gates) != _GATE_NAMES
        or any(value is not True for value in gates.values())
    ):
        raise PermissionError("robustness gate has not passed for this freeze")
    test = freeze.get("test")
    if test != {
        "access": "human_approval_only",
        "start": "2026-01-01",
        "end": "2026-07-11",
    }:
        raise ValueError("freeze locked-test boundary is invalid")
    identity = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "status": "test_requested",
        "freeze_id": freeze["freeze_id"],
        "freeze_sha256": freeze_sha256,
        "robustness_artifacts": artifacts,
        "gates": gates,
        "locked_test": test,
    }
    digest = _digest(identity)
    document = {**identity, "request_id": f"request-{digest}"}
    path = freeze_dir / "test_request.json"
    _write_immutable(path, canonical_bytes(document), "test request")
    return path


def approve_test_request(
    request_path: Path, approver: str, confirmed_freeze_sha256: str
) -> Path:
    _validate_approver(approver)
    request = validate_test_request(request_path)
    _validate_request_files(request_path, request)
    if confirmed_freeze_sha256 != request["freeze_sha256"]:
        raise PermissionError("freeze confirmation does not match the test request")
    approval_dir = request_path.parent / "approvals"
    if approval_dir.is_symlink():
        raise ValueError("approval directory is untrusted")
    if approval_dir.exists():
        for current in sorted(approval_dir.glob("*.json")):
            try:
                document = validate_approval(current)
            except ValueError as error:
                raise RuntimeError(f"immutable approval differs: {current}") from error
            if (
                document["request_id"] == request["request_id"]
                and document["request_sha256"] == _sha256(request_path)
                and document["freeze_sha256"] == confirmed_freeze_sha256
                and document["approver"] == approver
            ):
                return current
    identity = {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "status": "approved",
        "request_id": request["request_id"],
        "request_sha256": _sha256(request_path),
        "freeze_id": request["freeze_id"],
        "freeze_sha256": confirmed_freeze_sha256,
        "approver": approver,
        "approved_at": _utc_now(),
    }
    digest = _digest(identity)
    document = {**identity, "approval_id": f"approval-{digest}"}
    path = approval_dir / f"approval-{digest}.json"
    _write_immutable(path, canonical_bytes(document), "approval")
    return path


def validate_test_request(path: Path) -> dict[str, Any]:
    document = _read_canonical_json(path, "test request")
    required = {
        "schema_version",
        "status",
        "request_id",
        "freeze_id",
        "freeze_sha256",
        "robustness_artifacts",
        "gates",
        "locked_test",
    }
    identity = {key: value for key, value in document.items() if key != "request_id"}
    if (
        set(document) != required
        or document.get("schema_version") != REQUEST_SCHEMA_VERSION
        or document.get("status") != "test_requested"
        or document.get("request_id") != f"request-{_digest(identity)}"
        or not _freeze_id(document.get("freeze_id"))
        or not _sha(document.get("freeze_sha256"))
        or document.get("locked_test")
        != {
            "access": "human_approval_only",
            "start": "2026-01-01",
            "end": "2026-07-11",
        }
        or not isinstance(document.get("gates"), dict)
        or set(document["gates"]) != _GATE_NAMES
        or any(value is not True for value in document["gates"].values())
    ):
        raise ValueError("test request schema or identity is invalid")
    artifacts = document.get("robustness_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(_ROBUSTNESS_ARTIFACTS):
        raise ValueError("test request robustness artifacts are invalid")
    for name, reference in artifacts.items():
        if (
            not isinstance(reference, dict)
            or set(reference) != {"path", "sha256"}
            or reference.get("path") != name
            or not _sha(reference.get("sha256"))
        ):
            raise ValueError("test request robustness artifact reference is invalid")
    if path.name != "test_request.json" or path.parent.name != document["freeze_id"]:
        raise ValueError("test request path does not match its freeze identity")
    return document


def validate_approval(path: Path) -> dict[str, Any]:
    document = _read_canonical_json(path, "approval")
    required = {
        "schema_version",
        "status",
        "approval_id",
        "request_id",
        "request_sha256",
        "freeze_id",
        "freeze_sha256",
        "approver",
        "approved_at",
    }
    identity = {key: value for key, value in document.items() if key != "approval_id"}
    if (
        set(document) != required
        or document.get("schema_version") != APPROVAL_SCHEMA_VERSION
        or document.get("status") != "approved"
        or document.get("approval_id") != f"approval-{_digest(identity)}"
        or not re.fullmatch(r"request-[0-9a-f]{64}", str(document.get("request_id")))
        or not _sha(document.get("request_sha256"))
        or not _freeze_id(document.get("freeze_id"))
        or not _sha(document.get("freeze_sha256"))
        or not _timestamp(document.get("approved_at"))
    ):
        raise ValueError("approval schema or identity is invalid")
    _validate_approver(document.get("approver"))
    if path.name != f"{document['approval_id']}.json":
        raise ValueError("approval path does not match its identity")
    return document


def canonical_bytes(value: object) -> bytes:
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


def _read_canonical_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or untrusted")
    try:
        content = path.read_bytes()
        value = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is malformed") from error
    if not isinstance(value, dict) or content != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_immutable(path: Path, content: bytes, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError(f"immutable {label} differs: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise RuntimeError(f"immutable {label} differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise RuntimeError(f"immutable {label} differs: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _required_regular_file(path: Path, root: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required robustness artifact is missing: {path.name}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("robustness artifact escapes freeze directory") from error
    return path


def _validate_request_files(path: Path, request: dict[str, Any]) -> None:
    freeze_dir = path.parent
    freeze_path = freeze_dir / "freeze.json"
    if (
        freeze_path.is_symlink()
        or not freeze_path.is_file()
        or _sha256(freeze_path) != request["freeze_sha256"]
    ):
        raise ValueError("test request freeze hash has drifted")
    for name, reference in request["robustness_artifacts"].items():
        artifact = freeze_dir / name
        if (
            artifact.is_symlink()
            or not artifact.is_file()
            or _sha256(artifact) != reference["sha256"]
        ):
            raise ValueError(f"test request robustness hash has drifted: {name}")


def _validate_approver(value: object) -> None:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or len(value) > 128
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("approver must be a non-empty printable identity")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _freeze_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"freeze-[0-9a-f]{64}", value) is not None
    )
