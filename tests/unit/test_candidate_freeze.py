from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

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
    for name in ("robustness.yaml", "costs.yaml", "factor_registry.yaml"):
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
    phase5_id = "p5-ecaa6e8aeae6b9f8fb25"
    phase5_path = data_dir / "manifests" / phase5_id / "manifest.json"
    phase5_path.parent.mkdir(parents=True)
    _write_json(
        phase5_path,
        {"snapshot_id": phase5_id, "snapshot_type": "research_market"},
    )
    phase5_sha256 = _sha256(phase5_path)
    config, policy_sha256 = freeze.load_robustness_config(
        config_dir / "robustness.yaml"
    )
    exposure_id = "p6x-currentfixture00000"
    exposure_path = data_dir / "manifests" / exposure_id / "manifest.json"
    exposure_path.parent.mkdir(parents=True)
    _write_json(
        exposure_path,
        {
            "snapshot_id": exposure_id,
            "snapshot_type": "point_in_time_exposure",
            "phase5_snapshot_id": phase5_id,
            "phase5_manifest_sha256": phase5_sha256,
            "policy_sha256": policy_sha256,
            "artifacts": [],
        },
    )
    pointer = data_dir / "state" / "latest_exposure_snapshot.txt"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(f"{exposure_id}\n", encoding="utf-8")
    monkeypatch.setattr(freeze, "_git_commit", lambda repo_root: "a" * 40)
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
    assert document["snapshots"]["phase5"]["snapshot_id"] == ("p5-ecaa6e8aeae6b9f8fb25")
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
    assert document["git_commit"] == "a" * 40
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
    monkeypatch: pytest.MonkeyPatch,
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
        freeze_fixture[target].write_bytes(freeze_fixture[target].read_bytes() + b" ")
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
        monkeypatch.setattr(freeze, "_git_commit", lambda repo_root: "b" * 40)

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


def _write_json(path: Path, document: object) -> None:
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


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
