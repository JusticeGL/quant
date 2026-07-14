from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from alpha_lab.robustness.approval import (
    approve_test_request,
    create_test_request,
    validate_approval,
    validate_test_request,
)


def test_request_requires_all_pretest_gates(tmp_path: Path) -> None:
    freeze = _freeze_fixture(tmp_path, passed=False)
    with pytest.raises(PermissionError, match="robustness gate"):
        create_test_request(freeze)


def test_request_is_hash_linked_canonical_and_idempotent(tmp_path: Path) -> None:
    freeze = _freeze_fixture(tmp_path)
    first = create_test_request(freeze)
    original = first.read_bytes()
    second = create_test_request(freeze)
    document = validate_test_request(first)

    assert second == first
    assert first.read_bytes() == original
    assert re.fullmatch(r"request-[0-9a-f]{64}", document["request_id"])
    assert document["status"] == "test_requested"
    assert document["locked_test"] == {
        "access": "human_approval_only",
        "end": "2026-07-11",
        "start": "2026-01-01",
    }
    assert all(document["gates"].values())


def test_request_refuses_existing_conflicting_bytes(tmp_path: Path) -> None:
    freeze = _freeze_fixture(tmp_path)
    path = create_test_request(freeze)
    path.write_text("conflict\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="immutable test request"):
        create_test_request(freeze)


@pytest.mark.parametrize("approver", ["", "   ", "bad\nname", "x" * 129])
def test_approval_rejects_malformed_approver(tmp_path: Path, approver: str) -> None:
    request = create_test_request(_freeze_fixture(tmp_path))
    with pytest.raises(ValueError, match="approver"):
        approve_test_request(request, approver, "a" * 64)


def test_approval_requires_exact_freeze_confirmation(tmp_path: Path) -> None:
    request = create_test_request(_freeze_fixture(tmp_path))
    with pytest.raises(PermissionError, match="freeze confirmation"):
        approve_test_request(request, "Human Reviewer", "0" * 64)


@pytest.mark.parametrize("name", ["freeze.json", "robustness_report.md"])
def test_approval_rejects_request_input_hash_drift(tmp_path: Path, name: str) -> None:
    request = create_test_request(_freeze_fixture(tmp_path))
    confirmation = validate_test_request(request)["freeze_sha256"]
    (request.parent / name).write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash has drifted"):
        approve_test_request(request, "Human Reviewer", confirmation)


def test_approval_is_hash_linked_and_idempotent(tmp_path: Path) -> None:
    request = create_test_request(_freeze_fixture(tmp_path))
    requested = validate_test_request(request)
    confirmation = requested["freeze_sha256"]
    first = approve_test_request(request, "Human Reviewer", confirmation)
    second = approve_test_request(request, "Human Reviewer", confirmation)
    approval = validate_approval(first)

    assert second == first
    assert approval["status"] == "approved"
    assert approval["request_id"] == requested["request_id"]
    assert approval["request_sha256"] == _sha256(request)
    assert approval["freeze_sha256"] == confirmation


def test_approval_refuses_conflicting_existing_bytes(tmp_path: Path) -> None:
    request = create_test_request(_freeze_fixture(tmp_path))
    confirmation = validate_test_request(request)["freeze_sha256"]
    path = approve_test_request(request, "Human Reviewer", confirmation)
    path.write_text("conflict\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="immutable approval"):
        approve_test_request(request, "Human Reviewer", confirmation)


def _freeze_fixture(tmp_path: Path, *, passed: bool = True) -> Path:
    freeze_id = "freeze-" + "0" * 64
    root = tmp_path / "experiments" / "phase6" / freeze_id
    root.mkdir(parents=True)
    freeze = {
        "schema_version": 1,
        "freeze_id": freeze_id,
        "identity_sha256": "1" * 64,
        "factor": {
            "factor_id": "F1002",
            "source_path": "src/alpha_lab/factors/candidates/F1002.py",
            "source_sha256": "2" * 64,
            "metadata_path": "src/alpha_lab/factors/candidates/F1002.yaml",
            "metadata_sha256": "3" * 64,
        },
        "snapshots": {
            "phase5": {
                "snapshot_id": "p5-" + "4" * 20,
                "manifest_path": "manifests/p5-" + "4" * 20 + "/manifest.json",
                "manifest_sha256": "5" * 64,
            },
            "exposure": {
                "snapshot_id": "p6x-" + "6" * 20,
                "manifest_path": "manifests/p6x-" + "6" * 20 + "/manifest.json",
                "manifest_sha256": "7" * 64,
                "capability_id": "pretest-" + "8" * 20,
                "capability_sha256": "9" * 64,
            },
        },
        "policies": {
            "robustness": {"path": "config/robustness.yaml", "sha256": "b" * 64},
            "costs": {"path": "config/costs.yaml", "sha256": "c" * 64},
        },
        "test": {
            "access": "human_approval_only",
            "start": "2026-01-01",
            "end": "2026-07-11",
        },
        "git_commit": "d" * 40,
    }
    _write_json(root / "freeze.json", freeze)
    common = {
        "schema_version": 1,
        "freeze_id": freeze_id,
        "freeze_sha256": _sha256(root / "freeze.json"),
        "factor_id": "F1002",
        "test_accessed": False,
    }
    gates = {
        "direction_consistency": passed,
        "fold_coverage": True,
        "double_cost_direction": True,
        "industry_neutral_retention": True,
    }
    _write_json(root / "walk_forward.json", {**common, "gates": gates, "passed": passed})
    _write_json(root / "cost_sensitivity.json", common)
    _write_json(root / "exposure_report.json", common)
    (root / "robustness_report.md").write_text("# report\n", encoding="utf-8")
    return root / "freeze.json"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
