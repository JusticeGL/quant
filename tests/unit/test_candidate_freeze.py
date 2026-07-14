from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

from alpha_lab.research_data.config import load_research_data_config
from alpha_lab.robustness import freeze
from alpha_lab.robustness.freeze import freeze_candidate, validate_freeze

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def freeze_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    repo = tmp_path / "repo"
    config_dir = repo / "config"
    candidates = repo / "src" / "alpha_lab" / "factors" / "candidates"
    config_dir.mkdir(parents=True)
    candidates.mkdir(parents=True)
    for name in (
        "robustness.yaml",
        "costs.yaml",
        "factor_registry.yaml",
        "research_data.yaml",
    ):
        shutil.copyfile(ROOT / "config" / name, config_dir / name)
    for factor_id in ("F1001", "F1002", "F1003"):
        for suffix in (".py", ".yaml"):
            source = (
                ROOT
                / "src"
                / "alpha_lab"
                / "factors"
                / "candidates"
                / f"{factor_id}{suffix}"
            )
            shutil.copyfile(
                source,
                candidates / f"{factor_id}{suffix}",
            )

    data_dir = repo / "data"
    phase5_path = _write_phase5_snapshot(data_dir, config_dir)
    phase5_id = phase5_path.parent.name
    robustness_path = config_dir / "robustness.yaml"
    robustness = yaml.safe_load(robustness_path.read_text(encoding="utf-8"))
    robustness["phase5_snapshot_id"] = phase5_id
    robustness_path.write_text(
        yaml.safe_dump(robustness, sort_keys=False), encoding="utf-8"
    )
    phase5_sha256 = _sha256(phase5_path)
    _, policy_sha256 = freeze.load_robustness_config(robustness_path)
    exposure_path = _write_exposure_snapshot(
        data_dir,
        phase5_id=phase5_id,
        phase5_sha256=phase5_sha256,
        policy_sha256=policy_sha256,
    )
    exposure_id = exposure_path.parent.name
    pointer = data_dir / "state" / "latest_exposure_snapshot.txt"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(f"{exposure_id}\n", encoding="utf-8")
    monkeypatch.setattr(freeze, "FIXED_PHASE5_SNAPSHOT_ID", phase5_id)
    (repo / ".gitignore").write_text("data/\nexperiments/\n", encoding="utf-8")
    (repo / "runtime.txt").write_text("tracked runtime\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "task4@example.test")
    _git(repo, "config", "user.name", "Task 4 Test")
    _git(repo, "add", ".gitignore", "runtime.txt", "config", "src")
    _git(repo, "commit", "-m", "fixture")
    return {
        "repo": repo,
        "config": config_dir,
        "data": data_dir,
        "experiments": repo / "experiments",
        "phase5_manifest": phase5_path,
        "exposure_manifest": exposure_path,
        "source": candidates / "F1002.py",
        "metadata": candidates / "F1002.yaml",
    }


@pytest.mark.parametrize("factor_id", ["F1001", "F9999"])
def test_freeze_rejects_unapproved_or_unknown_candidate(
    freeze_fixture: dict[str, Path], factor_id: str
) -> None:
    with pytest.raises(PermissionError, match="approved Phase 6 candidate"):
        freeze_candidate(
            factor_id,
            freeze_fixture["config"],
            freeze_fixture["data"],
            freeze_fixture["experiments"],
        )


def test_freeze_pins_candidate_snapshots_policies_boundary_and_git(
    freeze_fixture: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    readers: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> object:
        readers.append("opened")
        raise AssertionError("freeze must not read Parquet or DuckDB")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(pq, "read_table", forbidden)
    monkeypatch.setattr(duckdb, "connect", forbidden)

    result = freeze_candidate(
        "F1002",
        freeze_fixture["config"],
        freeze_fixture["data"],
        freeze_fixture["experiments"],
    )
    document = json.loads(result.freeze_path.read_text(encoding="utf-8"))

    assert result.freeze_id == f"freeze-{document['identity_sha256']}"
    assert result.freeze_path == (
        freeze_fixture["experiments"] / "phase6" / result.freeze_id / "freeze.json"
    )
    assert document["factor"] == {
        "factor_id": "F1002",
        "metadata_path": "src/alpha_lab/factors/candidates/F1002.yaml",
        "metadata_sha256": _sha256(freeze_fixture["metadata"]),
        "source_path": "src/alpha_lab/factors/candidates/F1002.py",
        "source_sha256": _sha256(freeze_fixture["source"]),
    }
    assert document["snapshots"]["phase5"]["snapshot_id"] == (
        freeze_fixture["phase5_manifest"].parent.name
    )
    assert document["snapshots"]["phase5"]["manifest_sha256"] == _sha256(
        freeze_fixture["phase5_manifest"]
    )
    assert document["snapshots"]["exposure"]["manifest_sha256"] == _sha256(
        freeze_fixture["exposure_manifest"]
    )
    assert (
        document["policies"]["robustness"]["sha256"]
        == (
            freeze.load_robustness_config(freeze_fixture["config"] / "robustness.yaml")[
                1
            ]
        )
    )
    assert document["policies"]["costs"]["sha256"] == _canonical_yaml_hash(
        freeze_fixture["config"] / "costs.yaml"
    )
    assert document["test"] == {
        "access": "human_approval_only",
        "end": "2026-07-11",
        "start": "2026-01-01",
    }
    assert document["git_commit"] == _git(freeze_fixture["repo"], "rev-parse", "HEAD")
    assert readers == []


def test_freeze_is_byte_identical_on_repeat(freeze_fixture: dict[str, Path]) -> None:
    first = freeze_candidate(
        "F1002",
        freeze_fixture["config"],
        freeze_fixture["data"],
        freeze_fixture["experiments"],
    )
    original = first.freeze_path.read_bytes()
    second = freeze_candidate(
        "F1002",
        freeze_fixture["config"],
        freeze_fixture["data"],
        freeze_fixture["experiments"],
    )

    assert second == first
    assert second.freeze_path.read_bytes() == original


def test_freeze_allows_second_approved_candidate(
    freeze_fixture: dict[str, Path],
) -> None:
    result = freeze_candidate(
        "F1003",
        freeze_fixture["config"],
        freeze_fixture["data"],
        freeze_fixture["experiments"],
    )
    assert result.factor_id == "F1003"


def test_freeze_rejects_policy_retargeted_to_another_phase5_snapshot(
    freeze_fixture: dict[str, Path],
) -> None:
    robustness_path = freeze_fixture["config"] / "robustness.yaml"
    robustness = yaml.safe_load(robustness_path.read_text(encoding="utf-8"))
    alternate_id = "p5-alternatefixture"
    robustness["phase5_snapshot_id"] = alternate_id
    robustness_path.write_text(
        yaml.safe_dump(robustness, sort_keys=False), encoding="utf-8"
    )
    alternate_path = (
        freeze_fixture["data"] / "manifests" / alternate_id / "manifest.json"
    )
    alternate_path.parent.mkdir(parents=True)
    _write_json(
        alternate_path,
        {"snapshot_id": alternate_id, "snapshot_type": "research_market"},
    )
    exposure = json.loads(
        freeze_fixture["exposure_manifest"].read_text(encoding="utf-8")
    )
    exposure["phase5_snapshot_id"] = alternate_id
    exposure["phase5_manifest_sha256"] = _sha256(alternate_path)
    exposure["policy_sha256"] = freeze.load_robustness_config(robustness_path)[1]
    _write_json(freeze_fixture["exposure_manifest"], exposure)

    with pytest.raises(ValueError, match="fixed Phase 5"):
        freeze_candidate(
            "F1002",
            freeze_fixture["config"],
            freeze_fixture["data"],
            freeze_fixture["experiments"],
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "phase5_header_only",
        "phase5_identity",
        "exposure_header_only",
        "exposure_identity",
        "exposure_quality_error",
        "exposure_extra_artifact",
        "phase5_raw_checksum",
        "exposure_artifact_checksum",
    ],
)
def test_freeze_rejects_noncanonical_published_snapshot_manifests(
    freeze_fixture: dict[str, Path],
    corruption: str,
) -> None:
    phase5_path = freeze_fixture["phase5_manifest"]
    exposure_path = freeze_fixture["exposure_manifest"]
    phase5 = json.loads(phase5_path.read_text(encoding="utf-8"))
    exposure = json.loads(exposure_path.read_text(encoding="utf-8"))
    if corruption == "phase5_header_only":
        phase5 = {
            "snapshot_id": phase5["snapshot_id"],
            "snapshot_type": "research_market",
        }
        _write_json(phase5_path, phase5)
        exposure["phase5_manifest_sha256"] = _sha256(phase5_path)
        _write_json(exposure_path, exposure)
    elif corruption == "phase5_identity":
        phase5["identity_sha256"] = "0" * 64
        _write_json(phase5_path, phase5)
        exposure["phase5_manifest_sha256"] = _sha256(phase5_path)
        _write_json(exposure_path, exposure)
    elif corruption == "exposure_header_only":
        _write_json(
            exposure_path,
            {
                "snapshot_id": exposure["snapshot_id"],
                "snapshot_type": "point_in_time_exposure",
                "phase5_snapshot_id": exposure["phase5_snapshot_id"],
                "phase5_manifest_sha256": exposure["phase5_manifest_sha256"],
                "policy_sha256": exposure["policy_sha256"],
            },
        )
    elif corruption == "exposure_identity":
        exposure["identity_sha256"] = "0" * 64
        _write_json(exposure_path, exposure)
    elif corruption == "exposure_quality_error":
        exposure["quality_status"] = "error"
        _write_json(exposure_path, exposure)
    elif corruption == "exposure_extra_artifact":
        exposure["artifacts"].append(
            {
                "name": "unexpected.parquet",
                "format": "parquet",
                "sha256": "f" * 64,
                "row_count": 1,
                "path": f"exposures/{exposure['snapshot_id']}/unexpected.parquet",
            }
        )
        _write_json(exposure_path, exposure)
    elif corruption == "phase5_raw_checksum":
        raw_path = freeze_fixture["data"] / phase5["raw_inputs"][0]["path"]
        raw_path.write_bytes(raw_path.read_bytes() + b"corrupt")
    else:
        artifact_path = freeze_fixture["data"] / exposure["artifacts"][0]["path"]
        artifact_path.write_bytes(artifact_path.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="manifest"):
        freeze_candidate(
            "F1002",
            freeze_fixture["config"],
            freeze_fixture["data"],
            freeze_fixture["experiments"],
        )


@pytest.mark.parametrize("snapshot_kind", ["phase5", "exposure"])
@pytest.mark.parametrize(
    "corruption",
    [
        "empty_checks",
        "missing_check",
        "extra_check",
        "malformed_check",
        "false_pass",
        "summary_row_count_drift",
    ],
)
@pytest.mark.parametrize("operation", ["create", "validate"])
def test_freeze_creation_and_validation_reject_invalid_quality_contracts(
    freeze_fixture: dict[str, Path],
    snapshot_kind: str,
    corruption: str,
    operation: str,
) -> None:
    frozen = None
    if operation == "validate":
        frozen = freeze_candidate(
            "F1002",
            freeze_fixture["config"],
            freeze_fixture["data"],
            freeze_fixture["experiments"],
        )

    _corrupt_and_republish_quality(freeze_fixture, snapshot_kind, corruption)

    with pytest.raises(ValueError, match="quality"):
        if frozen is None:
            freeze_candidate(
                "F1002",
                freeze_fixture["config"],
                freeze_fixture["data"],
                freeze_fixture["experiments"],
            )
        else:
            validate_freeze(
                frozen.freeze_path,
                freeze_fixture["config"],
                freeze_fixture["data"],
            )


@pytest.mark.parametrize("operation", ["create", "validate"])
def test_freeze_creation_and_validation_reject_phase5_scope_retarget(
    freeze_fixture: dict[str, Path], operation: str
) -> None:
    frozen = None
    if operation == "validate":
        frozen = freeze_candidate(
            "F1002",
            freeze_fixture["config"],
            freeze_fixture["data"],
            freeze_fixture["experiments"],
        )

    _corrupt_and_republish_quality(freeze_fixture, "phase5", "coherent_scope_retarget")

    with pytest.raises(ValueError, match="quality.*scope"):
        if frozen is None:
            freeze_candidate(
                "F1002",
                freeze_fixture["config"],
                freeze_fixture["data"],
                freeze_fixture["experiments"],
            )
        else:
            validate_freeze(
                frozen.freeze_path,
                freeze_fixture["config"],
                freeze_fixture["data"],
            )


@pytest.mark.parametrize("operation", ["create", "validate"])
@pytest.mark.parametrize(
    "corruption",
    [
        "detached_minimum_coverage",
        "expected_industry_count_drift",
        "observed_observation_count_drift",
        "expected_security_count_drift",
    ],
)
def test_freeze_creation_and_validation_reject_detached_exposure_coverage(
    freeze_fixture: dict[str, Path], operation: str, corruption: str
) -> None:
    frozen = None
    if operation == "validate":
        frozen = freeze_candidate(
            "F1002",
            freeze_fixture["config"],
            freeze_fixture["data"],
            freeze_fixture["experiments"],
        )

    _corrupt_and_republish_quality(freeze_fixture, "exposure", corruption)

    with pytest.raises(ValueError, match="quality.*row_counts"):
        if frozen is None:
            freeze_candidate(
                "F1002",
                freeze_fixture["config"],
                freeze_fixture["data"],
                freeze_fixture["experiments"],
            )
        else:
            validate_freeze(
                frozen.freeze_path,
                freeze_fixture["config"],
                freeze_fixture["data"],
            )


@pytest.mark.parametrize(
    "target",
    [
        "source",
        "metadata",
        "robustness_policy",
        "cost_policy",
        "phase5_manifest",
        "exposure_manifest",
        "pointer",
    ],
)
def test_freeze_rejects_symlinked_dependency_path(
    freeze_fixture: dict[str, Path],
    tmp_path: Path,
    target: str,
) -> None:
    paths = {
        "source": freeze_fixture["source"],
        "metadata": freeze_fixture["metadata"],
        "robustness_policy": freeze_fixture["config"] / "robustness.yaml",
        "cost_policy": freeze_fixture["config"] / "costs.yaml",
        "phase5_manifest": freeze_fixture["phase5_manifest"],
        "exposure_manifest": freeze_fixture["exposure_manifest"],
        "pointer": freeze_fixture["data"] / "state" / "latest_exposure_snapshot.txt",
    }
    path = paths[target]
    outside = tmp_path / f"outside-{target}"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        freeze_candidate(
            "F1002",
            freeze_fixture["config"],
            freeze_fixture["data"],
            freeze_fixture["experiments"],
        )


@pytest.mark.parametrize("component", ["experiments", "phase6"])
def test_freeze_rejects_symlinked_output_component(
    freeze_fixture: dict[str, Path],
    tmp_path: Path,
    component: str,
) -> None:
    experiments = freeze_fixture["experiments"]
    outside = tmp_path / f"outside-{component}"
    outside.mkdir()
    if component == "experiments":
        experiments.symlink_to(outside, target_is_directory=True)
    else:
        experiments.mkdir()
        (experiments / "phase6").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        freeze_candidate(
            "F1002",
            freeze_fixture["config"],
            freeze_fixture["data"],
            experiments,
        )


@pytest.mark.parametrize("component", ["freeze_dir", "freeze_file"])
def test_validate_freeze_rejects_symlinked_artifact_component(
    freeze_fixture: dict[str, Path],
    tmp_path: Path,
    component: str,
) -> None:
    result = freeze_candidate(
        "F1002",
        freeze_fixture["config"],
        freeze_fixture["data"],
        freeze_fixture["experiments"],
    )
    if component == "freeze_dir":
        outside = tmp_path / "outside-freeze-dir"
        shutil.copytree(result.freeze_path.parent, outside)
        shutil.rmtree(result.freeze_path.parent)
        result.freeze_path.parent.symlink_to(outside, target_is_directory=True)
    else:
        outside = tmp_path / "outside-freeze.json"
        outside.write_bytes(result.freeze_path.read_bytes())
        result.freeze_path.unlink()
        result.freeze_path.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        validate_freeze(
            result.freeze_path,
            freeze_fixture["config"],
            freeze_fixture["data"],
        )


@pytest.mark.parametrize("staged", [False, True])
def test_freeze_rejects_tracked_dirty_tree(
    freeze_fixture: dict[str, Path], staged: bool
) -> None:
    runtime = freeze_fixture["repo"] / "runtime.txt"
    runtime.write_text("tracked runtime changed\n", encoding="utf-8")
    if staged:
        _git(freeze_fixture["repo"], "add", "runtime.txt")

    with pytest.raises(ValueError, match="dirty Git tree"):
        freeze_candidate(
            "F1002",
            freeze_fixture["config"],
            freeze_fixture["data"],
            freeze_fixture["experiments"],
        )


def test_validate_freeze_rejects_tracked_dirty_tree(
    freeze_fixture: dict[str, Path],
) -> None:
    result = freeze_candidate(
        "F1002",
        freeze_fixture["config"],
        freeze_fixture["data"],
        freeze_fixture["experiments"],
    )
    (freeze_fixture["repo"] / "runtime.txt").write_text(
        "tracked runtime changed\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="dirty Git tree"):
        validate_freeze(
            result.freeze_path,
            freeze_fixture["config"],
            freeze_fixture["data"],
        )


def test_freeze_rejects_candidate_untracked_at_head(
    freeze_fixture: dict[str, Path],
) -> None:
    repo = freeze_fixture["repo"]
    paths = [
        "src/alpha_lab/factors/candidates/F1002.py",
        "src/alpha_lab/factors/candidates/F1002.yaml",
    ]
    _git(repo, "rm", "--cached", *paths)
    _git(repo, "commit", "-m", "remove candidate from commit")

    with pytest.raises(ValueError, match="tracked.*HEAD"):
        freeze_candidate(
            "F1002",
            freeze_fixture["config"],
            freeze_fixture["data"],
            freeze_fixture["experiments"],
        )


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("source", "factor source"),
        ("metadata", "factor metadata"),
        ("phase5_manifest", "Phase 5 manifest"),
        ("exposure_manifest", "exposure manifest"),
        ("robustness_policy", "robustness policy"),
        ("cost_policy", "cost policy"),
        ("git", "Git commit"),
    ],
)
def test_validate_freeze_fails_closed_on_current_dependency_drift(
    freeze_fixture: dict[str, Path],
    target: str,
    expected: str,
) -> None:
    result = freeze_candidate(
        "F1002",
        freeze_fixture["config"],
        freeze_fixture["data"],
        freeze_fixture["experiments"],
    )
    if target in {
        "source",
        "metadata",
        "phase5_manifest",
        "exposure_manifest",
    }:
        path = freeze_fixture[target]
        path.write_bytes(path.read_bytes() + b" ")
    elif target == "robustness_policy":
        path = freeze_fixture["config"] / "robustness.yaml"
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        document["size_correlation_risk_threshold"] = 0.31
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    elif target == "cost_policy":
        path = freeze_fixture["config"] / "costs.yaml"
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        document["notes"] += " drift"
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    else:
        path = freeze_fixture["repo"] / "runtime.txt"
        path.write_text("new tracked runtime\n", encoding="utf-8")

    if target not in {"phase5_manifest", "exposure_manifest"}:
        relative = path.relative_to(freeze_fixture["repo"]).as_posix()
        _git(freeze_fixture["repo"], "add", relative)
        _git(freeze_fixture["repo"], "commit", "-m", f"drift {target}")

    with pytest.raises(ValueError, match=expected):
        validate_freeze(
            result.freeze_path,
            freeze_fixture["config"],
            freeze_fixture["data"],
        )


def test_validate_freeze_recomputes_identity_and_avoids_test_data_readers(
    freeze_fixture: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    result = freeze_candidate(
        "F1002",
        freeze_fixture["config"],
        freeze_fixture["data"],
        freeze_fixture["experiments"],
    )
    document = json.loads(result.freeze_path.read_text(encoding="utf-8"))
    document["identity_sha256"] = "0" * 64
    result.freeze_path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("freeze validation must not read test data")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(pq, "read_table", forbidden)
    monkeypatch.setattr(duckdb, "connect", forbidden)

    with pytest.raises(ValueError, match="identity"):
        validate_freeze(
            result.freeze_path,
            freeze_fixture["config"],
            freeze_fixture["data"],
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "snapshots_not_mapping",
        "snapshot_missing_key",
        "policies_not_mapping",
        "test_missing_key",
        "nested_hash_wrong_type",
        "git_commit_wrong_type",
    ],
)
def test_validate_freeze_rejects_malformed_nested_schema_as_value_error(
    freeze_fixture: dict[str, Path], corruption: str
) -> None:
    result = freeze_candidate(
        "F1002",
        freeze_fixture["config"],
        freeze_fixture["data"],
        freeze_fixture["experiments"],
    )
    document = json.loads(result.freeze_path.read_text(encoding="utf-8"))
    if corruption == "snapshots_not_mapping":
        document["snapshots"] = None
    elif corruption == "snapshot_missing_key":
        del document["snapshots"]["phase5"]["manifest_sha256"]
    elif corruption == "policies_not_mapping":
        document["policies"] = []
    elif corruption == "test_missing_key":
        del document["test"]["access"]
    elif corruption == "nested_hash_wrong_type":
        document["factor"]["source_sha256"] = True
    else:
        document["git_commit"] = 1
    malformed_path = _rewrite_freeze_identity(result.freeze_path, document)

    with pytest.raises(ValueError, match="schema"):
        validate_freeze(
            malformed_path,
            freeze_fixture["config"],
            freeze_fixture["data"],
        )


def _write_json(path: Path, document: object) -> None:
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _rewrite_freeze_identity(path: Path, document: dict[str, object]) -> Path:
    payload = {
        key: value
        for key, value in document.items()
        if key not in {"freeze_id", "identity_sha256"}
    }
    identity = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    freeze_id = f"freeze-{identity}"
    document["identity_sha256"] = identity
    document["freeze_id"] = freeze_id
    destination = path.parent.parent / freeze_id / "freeze.json"
    destination.parent.mkdir()
    _write_json(destination, document)
    return destination


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_yaml_hash(path: Path) -> str:
    return hashlib.sha256(
        json.dumps(
            yaml.safe_load(path.read_text(encoding="utf-8")),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def _write_phase5_snapshot(data_dir: Path, config_dir: Path) -> Path:
    quality = _phase5_quality()
    quality_bytes = _canonical_json_bytes(quality)
    raw_bytes = b"phase5 raw fixture"
    raw_identity = _raw_identity("a", raw_bytes)
    names = [
        "security_master.parquet",
        "security_name_history.parquet",
        "trading_calendar.parquet",
        "index_membership.parquet",
        "suspension.parquet",
        "universe_dates.parquet",
        "daily_bar/year=2025/part.parquet",
        "adjustment_factor/year=2025/part.parquet",
        "daily_status/year=2025/part.parquet",
    ]
    artifact_identities = [_artifact_identity(name) for name in sorted(names)]
    config = load_research_data_config(config_dir)
    identity = {
        "research_schema_version": 1,
        "config": config.model_dump(mode="json"),
        "raw_inputs": [raw_identity],
        "artifacts": artifact_identities,
    }
    identity_sha256 = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()
    snapshot_id = f"p5-{identity_sha256[:20]}"
    raw_path = data_dir / "raw" / "tushare" / "fixture" / "request.parquet"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(raw_bytes)
    artifacts = _write_fixture_artifacts(
        data_dir, "research", snapshot_id, artifact_identities
    )
    quality_path = data_dir / "manifests" / snapshot_id / "quality_report.json"
    quality_path.parent.mkdir(parents=True)
    quality_path.write_bytes(quality_bytes)
    manifest = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "snapshot_type": "research_market",
        "identity_sha256": identity_sha256,
        "quality_status": "pass",
        "source": {"provider": "tushare", "credential_redacted": True},
        "scope": quality["scope"],
        "summary": quality["summary"],
        "raw_inputs": [{**raw_identity, "path": "raw/tushare/fixture/request.parquet"}],
        "artifacts": artifacts,
        "quality_report": {
            "path": f"manifests/{snapshot_id}/quality_report.json",
            "sha256": hashlib.sha256(quality_bytes).hexdigest(),
        },
    }
    manifest_path = quality_path.with_name("manifest.json")
    manifest_path.write_bytes(_canonical_json_bytes(manifest))
    return manifest_path


def _write_exposure_snapshot(
    data_dir: Path,
    *,
    phase5_id: str,
    phase5_sha256: str,
    policy_sha256: str,
) -> Path:
    quality = _exposure_quality()
    quality_bytes = _canonical_json_bytes(quality)
    raw_bytes = b"phase6 raw fixture"
    raw_identity = _raw_identity("b", raw_bytes)
    artifact_identities = [
        _artifact_identity(name)
        for name in (
            "industry_definition.parquet",
            "industry_membership.parquet",
            "industry_membership_pretest.parquet",
            "market_cap/year=2025/part.parquet",
        )
    ]
    coverage_scope = {
        "start_date": "2020-01-01",
        "end_date": "2026-07-11",
        "minimum_temporal_coverage": 0.7,
    }
    identity = {
        "exposure_schema_version": 1,
        "phase5_manifest_sha256": phase5_sha256,
        "policy_sha256": policy_sha256,
        "quality_report_sha256": hashlib.sha256(quality_bytes).hexdigest(),
        "coverage_scope": coverage_scope,
        "raw_request_identities": [raw_identity],
        "artifacts": artifact_identities,
    }
    identity_sha256 = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()
    snapshot_id = f"p6x-{identity_sha256[:20]}"
    raw_path = data_dir / "raw" / "tushare" / "fixture" / "exposure.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw_bytes)
    artifacts = _write_fixture_artifacts(
        data_dir, "exposures", snapshot_id, artifact_identities
    )
    quality_path = data_dir / "manifests" / snapshot_id / "quality_report.json"
    quality_path.parent.mkdir(parents=True)
    quality_path.write_bytes(quality_bytes)
    manifest = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "snapshot_type": "point_in_time_exposure",
        "identity_sha256": identity_sha256,
        "phase5_snapshot_id": phase5_id,
        "phase5_manifest_sha256": phase5_sha256,
        "policy_sha256": policy_sha256,
        "quality_status": "pass",
        "coverage_scope": coverage_scope,
        "source": {
            "provider": "tushare",
            "classification_standard": "SW2021",
            "credential_redacted": True,
        },
        "raw_inputs": [
            {**raw_identity, "path": "raw/tushare/fixture/exposure.parquet"}
        ],
        "artifacts": artifacts,
        "quality_report": {
            "path": f"manifests/{snapshot_id}/quality_report.json",
            "sha256": hashlib.sha256(quality_bytes).hexdigest(),
        },
    }
    manifest_path = quality_path.with_name("manifest.json")
    manifest_path.write_bytes(_canonical_json_bytes(manifest))
    return manifest_path


def _write_fixture_artifacts(
    data_dir: Path,
    root: str,
    snapshot_id: str,
    identities: list[dict[str, object]],
) -> list[dict[str, object]]:
    artifacts = []
    for item in identities:
        name = str(item["name"])
        path = data_dir / root / snapshot_id / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_artifact_bytes(name))
        artifacts.append(
            {
                **item,
                "format": "parquet",
                "path": f"{root}/{snapshot_id}/{name}",
            }
        )
    return artifacts


def _corrupt_and_republish_quality(
    fixture: dict[str, Path], snapshot_kind: str, corruption: str
) -> None:
    data_dir = fixture["data"]
    manifest_path = fixture[f"{snapshot_kind}_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quality_path = data_dir / manifest["quality_report"]["path"]
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    checks = quality["checks"]
    first_check = next(iter(checks))
    if corruption == "empty_checks":
        quality["checks"] = {}
    elif corruption == "missing_check":
        del checks[first_check]
    elif corruption == "extra_check":
        checks["unexpected_check"] = {
            "severity": "error",
            "status": "pass",
            "count": 0,
        }
    elif corruption == "malformed_check":
        checks[first_check] = {"severity": "error", "status": "pass"}
    elif corruption == "false_pass":
        checks[first_check] = {
            "severity": "error",
            "status": "fail",
            "count": 1,
        }
    elif corruption == "coherent_scope_retarget":
        quality["scope"] = {
            "index_code": "RETARGETED",
            "start_date": "1990-01-01",
            "end_date": "2099-01-01",
        }
    elif corruption == "detached_minimum_coverage":
        quality["summary"]["minimum_temporal_coverage"] = 0.0
    elif corruption == "expected_industry_count_drift":
        quality["summary"]["expected_industry_count"] += 1
    elif corruption == "observed_observation_count_drift":
        quality["summary"]["observed_observation_count"] = 0
        quality["summary"]["temporal_coverage_ratio"] = 0.0
    elif corruption == "expected_security_count_drift":
        quality["summary"]["expected_security_count"] = 2
    else:
        count_name = (
            "daily_bar_count" if snapshot_kind == "phase5" else "market_cap_count"
        )
        quality["summary"][count_name] += 1

    if snapshot_kind == "phase5":
        quality_bytes = _canonical_json_bytes(quality)
        quality_path.write_bytes(quality_bytes)
        manifest["quality_report"]["sha256"] = hashlib.sha256(quality_bytes).hexdigest()
        manifest["scope"] = quality["scope"]
        manifest["summary"] = quality["summary"]
        manifest_path.write_bytes(_canonical_json_bytes(manifest))
        _republish_exposure_snapshot(
            fixture,
            phase5_manifest_sha256=_sha256(manifest_path),
        )
    else:
        _republish_exposure_snapshot(fixture, quality=quality)


def _republish_exposure_snapshot(
    fixture: dict[str, Path],
    *,
    quality: dict[str, object] | None = None,
    phase5_manifest_sha256: str | None = None,
) -> Path:
    data_dir = fixture["data"]
    old_manifest_path = fixture["exposure_manifest"]
    manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    old_snapshot_id = manifest["snapshot_id"]
    if quality is None:
        old_quality_path = data_dir / manifest["quality_report"]["path"]
        quality = json.loads(old_quality_path.read_text(encoding="utf-8"))
    quality_bytes = _canonical_json_bytes(quality)
    if phase5_manifest_sha256 is not None:
        manifest["phase5_manifest_sha256"] = phase5_manifest_sha256
    manifest["quality_report"]["sha256"] = hashlib.sha256(quality_bytes).hexdigest()
    identity = {
        "exposure_schema_version": manifest["schema_version"],
        "phase5_manifest_sha256": manifest["phase5_manifest_sha256"],
        "policy_sha256": manifest["policy_sha256"],
        "quality_report_sha256": manifest["quality_report"]["sha256"],
        "coverage_scope": manifest["coverage_scope"],
        "raw_request_identities": [
            {
                key: item[key]
                for key in (
                    "api_name",
                    "request_sha256",
                    "sha256",
                    "row_count",
                    "params",
                    "fields",
                )
            }
            for item in manifest["raw_inputs"]
        ],
        "artifacts": [
            {key: item[key] for key in ("name", "sha256", "row_count")}
            for item in manifest["artifacts"]
        ],
    }
    identity_sha256 = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()
    new_snapshot_id = f"p6x-{identity_sha256[:20]}"
    old_root = data_dir / "exposures" / old_snapshot_id
    new_root = data_dir / "exposures" / new_snapshot_id
    shutil.copytree(old_root, new_root)
    manifest["snapshot_id"] = new_snapshot_id
    manifest["identity_sha256"] = identity_sha256
    for item in manifest["artifacts"]:
        item["path"] = f"exposures/{new_snapshot_id}/{item['name']}"
    manifest["quality_report"]["path"] = (
        f"manifests/{new_snapshot_id}/quality_report.json"
    )
    new_manifest_dir = data_dir / "manifests" / new_snapshot_id
    new_manifest_dir.mkdir(parents=True)
    (new_manifest_dir / "quality_report.json").write_bytes(quality_bytes)
    new_manifest_path = new_manifest_dir / "manifest.json"
    new_manifest_path.write_bytes(_canonical_json_bytes(manifest))
    pointer = data_dir / "state" / "latest_exposure_snapshot.txt"
    pointer.write_text(f"{new_snapshot_id}\n", encoding="utf-8")
    return new_manifest_path


def _raw_identity(prefix: str, content: bytes) -> dict[str, object]:
    return {
        "api_name": "fixture",
        "request_sha256": prefix * 64,
        "sha256": hashlib.sha256(content).hexdigest(),
        "row_count": 1,
        "params": {},
        "fields": ["fixture"],
    }


def _artifact_identity(name: str) -> dict[str, object]:
    return {
        "name": name,
        "sha256": hashlib.sha256(_artifact_bytes(name)).hexdigest(),
        "row_count": 1,
    }


def _artifact_bytes(name: str) -> bytes:
    return f"fixture:{name}".encode()


def _phase5_quality() -> dict[str, object]:
    checks = {
        name: {"severity": severity, "status": "pass", "count": 0}
        for name, severity in (
            ("duplicate_keys", "error"),
            ("membership_overlap", "error"),
            ("name_history_overlap", "error"),
            ("suspension_overlap", "error"),
            ("unknown_security_reference", "error"),
            ("membership_lifecycle_violation", "error"),
            ("invalid_adjustment_factor", "error"),
            ("nullable_status", "warning"),
            ("missing_delist_date", "warning"),
        )
    }
    return {
        "schema_version": 1,
        "policy": "phase5_point_in_time_quality_v1",
        "status": "pass",
        "scope": {
            "index_code": "000300.SH",
            "start_date": "2020-01-01",
            "end_date": "2026-07-11",
        },
        "summary": {
            "security_count": 1,
            "delisted_security_count": 0,
            "membership_interval_count": 1,
            "daily_bar_count": 1,
            "adjustment_factor_count": 1,
            "daily_status_count": 1,
        },
        "checks": checks,
    }


def _exposure_quality() -> dict[str, object]:
    checks = {
        name: {"severity": "error", "status": "pass", "count": 0}
        for name in (
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
        )
    }
    return {
        "schema_version": 1,
        "policy": "phase6_exposure_quality_v1",
        "status": "pass",
        "summary": {
            "market_cap_count": 1,
            "industry_definition_count": 1,
            "industry_membership_count": 1,
            "industry_membership_pretest_count": 1,
            "expected_security_count": 1,
            "expected_industry_count": 1,
            "expected_observation_count": 1,
            "observed_observation_count": 1,
            "temporal_coverage_ratio": 1.0,
            "minimum_temporal_coverage": 0.7,
        },
        "checks": checks,
    }


def _canonical_json_bytes(value: object) -> bytes:
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


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
