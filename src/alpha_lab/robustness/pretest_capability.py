from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, cast

PRETEST_CAPABILITY_SCHEMA_VERSION = 1
PRETEST_CUTOFF = "2026-01-01"
_PHASE5_DATASETS = ("daily_bar", "adjustment_factor", "daily_status")
_EXPOSURE_FIXED = {
    "industry_definition.parquet",
    "industry_membership_pretest.parquet",
}


def build_pretest_capability(
    phase5_manifest: dict[str, Any], exposure_manifest: dict[str, Any]
) -> dict[str, Any]:
    artifacts: list[dict[str, object]] = []
    for item in phase5_manifest.get("artifacts", []):
        name = str(item.get("name"))
        if _safe_phase5_name(name):
            artifacts.append(_capability_artifact("phase5", item))
    for item in exposure_manifest.get("artifacts", []):
        name = str(item.get("name"))
        if name in _EXPOSURE_FIXED or _safe_partition(name, "market_cap"):
            artifacts.append(_capability_artifact("exposure", item))
    artifacts.sort(key=lambda item: (str(item["domain"]), str(item["name"])))
    if not all(
        any(
            item["domain"] == "phase5" and str(item["name"]).startswith(f"{dataset}/")
            for item in artifacts
        )
        for dataset in _PHASE5_DATASETS
    ):
        raise ValueError("pre-test capability is missing a Phase 5 market dataset")
    if not _EXPOSURE_FIXED.issubset(
        {str(item["name"]) for item in artifacts if item["domain"] == "exposure"}
    ) or not any(
        item["domain"] == "exposure" and str(item["name"]).startswith("market_cap/")
        for item in artifacts
    ):
        raise ValueError("pre-test capability is missing an exposure dataset")
    summary = {
        "artifact_count": len(artifacts),
        "row_count": sum(cast(int, item["row_count"]) for item in artifacts),
        "phase5_row_count": sum(
            cast(int, item["row_count"])
            for item in artifacts
            if item["domain"] == "phase5"
        ),
        "exposure_row_count": sum(
            cast(int, item["row_count"])
            for item in artifacts
            if item["domain"] == "exposure"
        ),
    }
    checks = {
        "artifact_namespace": {"status": "pass", "count": 0},
        "locked_boundary": {"status": "pass", "count": 0},
        "physical_hashes": {"status": "pass", "count": 0},
    }
    identity = {
        "schema_version": PRETEST_CAPABILITY_SCHEMA_VERSION,
        "capability_type": "phase6_pretest_data",
        "cutoff": PRETEST_CUTOFF,
        "phase5_parent": {
            "snapshot_id": phase5_manifest.get("snapshot_id"),
            "manifest_sha256": exposure_manifest.get("phase5_manifest_sha256"),
        },
        "policy_sha256": exposure_manifest.get("policy_sha256"),
        "artifacts": artifacts,
        "quality": {"status": "pass", "summary": summary, "checks": checks},
    }
    identity_sha256 = _digest(identity)
    return {
        **identity,
        "capability_id": f"pretest-{identity_sha256[:20]}",
        "identity_sha256": identity_sha256,
    }


def validate_pretest_capability(
    data_dir: Path, snapshot_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate only root metadata, capability bytes, and listed safe artifacts."""
    if re.fullmatch(r"p6x-[0-9a-f]{20}", snapshot_id) is None:
        raise ValueError("invalid exposure snapshot ID")
    manifest_path = data_dir / "manifests" / snapshot_id / "manifest.json"
    manifest = _canonical_document(manifest_path, "exposure root manifest")
    if (
        manifest.get("snapshot_id") != snapshot_id
        or manifest.get("snapshot_type") != "point_in_time_exposure"
    ):
        raise ValueError("exposure root manifest identity mismatch")
    reference = manifest.get("pretest_capability")
    if not isinstance(reference, dict) or set(reference) != {
        "path",
        "sha256",
        "capability_id",
    }:
        raise ValueError("pre-test capability reference is missing or malformed")
    if reference["path"] != "pretest_capability.json" or not _sha(reference["sha256"]):
        raise ValueError("pre-test capability reference is not canonical")
    capability_path = manifest_path.parent / "pretest_capability.json"
    capability_bytes = capability_path.read_bytes()
    if hashlib.sha256(capability_bytes).hexdigest() != reference["sha256"]:
        raise ValueError("pre-test capability checksum mismatch")
    capability = _canonical_document(capability_path, "pre-test capability")
    _validate_capability_document(capability, manifest, reference)
    _validate_root_identity(manifest)
    for item in capability["artifacts"]:
        root = "research" if item["domain"] == "phase5" else "exposures"
        parent = (
            capability["phase5_parent"]["snapshot_id"]
            if item["domain"] == "phase5"
            else snapshot_id
        )
        path = data_dir / root / parent / item["name"]
        if _contains_symlink(path, data_dir):
            raise ValueError(
                "pre-test capability artifact path contains a symlink: "
                f"{item['domain']}:{item['name']}"
            )
        if not path.is_file() or _file_digest(path) != item["sha256"]:
            raise ValueError(
                "pre-test capability artifact checksum mismatch: "
                f"{item['domain']}:{item['name']}"
            )
    return manifest, capability


def root_identity(manifest: dict[str, Any]) -> dict[str, object]:
    quality = manifest.get("quality_report")
    return {
        "exposure_schema_version": manifest.get("schema_version"),
        "phase5_manifest_sha256": manifest.get("phase5_manifest_sha256"),
        "policy_sha256": manifest.get("policy_sha256"),
        "quality_report_sha256": quality.get("sha256")
        if isinstance(quality, dict)
        else None,
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
        "pretest_capability": manifest.get("pretest_capability"),
    }


def _validate_root_identity(manifest: dict[str, Any]) -> None:
    actual = _digest(root_identity(manifest))
    if (
        manifest.get("identity_sha256") != actual
        or manifest.get("snapshot_id") != f"p6x-{actual[:20]}"
    ):
        raise ValueError("exposure root identity does not bind pre-test capability")


def _validate_capability_document(
    capability: dict[str, Any], manifest: dict[str, Any], reference: dict[str, Any]
) -> None:
    required = {
        "schema_version",
        "capability_type",
        "capability_id",
        "identity_sha256",
        "cutoff",
        "phase5_parent",
        "policy_sha256",
        "artifacts",
        "quality",
    }
    if (
        set(capability) != required
        or capability.get("schema_version") != 1
        or capability.get("capability_type") != "phase6_pretest_data"
        or capability.get("cutoff") != PRETEST_CUTOFF
    ):
        raise ValueError("pre-test capability schema is invalid")
    identity = {
        key: capability[key] for key in required - {"capability_id", "identity_sha256"}
    }
    digest = _digest(identity)
    if (
        capability.get("identity_sha256") != digest
        or capability.get("capability_id") != f"pretest-{digest[:20]}"
        or reference.get("capability_id") != capability.get("capability_id")
    ):
        raise ValueError("pre-test capability identity mismatch")
    parent = capability.get("phase5_parent")
    if parent != {
        "snapshot_id": manifest.get("phase5_snapshot_id"),
        "manifest_sha256": manifest.get("phase5_manifest_sha256"),
    } or capability.get("policy_sha256") != manifest.get("policy_sha256"):
        raise ValueError("pre-test capability parent binding mismatch")
    expected = build_pretest_capability(
        {
            "snapshot_id": manifest.get("phase5_snapshot_id"),
            "artifacts": [
                item
                for item in capability.get("artifacts", [])
                if item.get("domain") == "phase5"
            ],
        },
        {
            "phase5_manifest_sha256": manifest.get("phase5_manifest_sha256"),
            "policy_sha256": manifest.get("policy_sha256"),
            "artifacts": [
                item
                for item in capability.get("artifacts", [])
                if item.get("domain") == "exposure"
            ],
        },
    )
    if capability != expected:
        raise ValueError("pre-test capability quality or namespace is invalid")


def _capability_artifact(domain: str, item: dict[str, Any]) -> dict[str, object]:
    result = {
        "domain": domain,
        "name": item.get("name"),
        "sha256": item.get("sha256"),
        "row_count": item.get("row_count"),
    }
    if (
        not isinstance(result["name"], str)
        or not _sha(result["sha256"])
        or not isinstance(result["row_count"], int)
        or isinstance(result["row_count"], bool)
        or int(result["row_count"]) < 0
    ):
        raise ValueError("pre-test capability artifact metadata is invalid")
    return result


def _safe_phase5_name(name: str) -> bool:
    return any(_safe_partition(name, dataset) for dataset in _PHASE5_DATASETS)


def _safe_partition(name: str, dataset: str) -> bool:
    match = re.fullmatch(
        rf"{re.escape(dataset)}/year=([0-9]{{4}})/part[.]parquet", name
    )
    return match is not None and int(match.group(1)) < 2026


def _canonical_document(path: Path, label: str) -> dict[str, Any]:
    try:
        if _contains_symlink(path, path.parents[2]):
            raise ValueError(f"{label} path contains a symlink")
        content = path.read_bytes()
        value = json.loads(content)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} is missing or malformed") from error
    if not isinstance(value, dict) or content != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _contains_symlink(path: Path, root: Path) -> bool:
    current = path
    while current != root:
        if current.is_symlink():
            return True
        if current.parent == current:
            return True
        current = current.parent
    return root.is_symlink()


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
    ).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
