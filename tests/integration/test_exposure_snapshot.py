from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.research_data.provider import TushareArtifact
from alpha_lab.robustness.config import load_robustness_config
from alpha_lab.robustness.contracts import ExposureTables
from alpha_lab.robustness.exposure_data import (
    normalize_industry_definition,
    normalize_industry_membership,
    normalize_market_cap,
)
from alpha_lab.robustness.exposure_snapshot import (
    materialize_exposure_snapshot,
    validate_exposure_snapshot,
)

ROOT = Path(__file__).resolve().parents[2]


def test_exposure_snapshot_is_deterministic_and_validated_before_pointer(
    tmp_path: Path,
) -> None:
    phase5_manifest = _phase5_fixture(tmp_path)
    config, policy_sha256 = load_robustness_config(ROOT / "config" / "robustness.yaml")
    raw_inputs = [_raw_input(tmp_path)]

    first = materialize_exposure_snapshot(
        tmp_path,
        config,
        policy_sha256,
        phase5_manifest,
        _tables(),
        raw_inputs,
    )
    second = materialize_exposure_snapshot(
        tmp_path,
        config,
        policy_sha256,
        phase5_manifest,
        _tables(),
        raw_inputs,
    )

    assert first.snapshot_id.startswith("p6x-")
    assert first.snapshot_id == second.snapshot_id
    assert first.manifest_sha256 == second.manifest_sha256
    assert (
        tmp_path / "state" / "latest_exposure_snapshot.txt"
    ).read_text().strip() == first.snapshot_id
    validation = validate_exposure_snapshot(tmp_path, first.snapshot_id)
    assert validation["healthy"] is True
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert (
        manifest["phase5_manifest_sha256"]
        == hashlib.sha256(phase5_manifest.read_bytes()).hexdigest()
    )
    assert manifest["policy_sha256"] == policy_sha256


def test_exposure_snapshot_refuses_quality_error_without_publishing_pointer(
    tmp_path: Path,
) -> None:
    phase5_manifest = _phase5_fixture(tmp_path)
    config, policy_sha256 = load_robustness_config(ROOT / "config" / "robustness.yaml")
    tables = _tables()
    tables.market_cap.loc[0, "security_id"] = "CN:SSE:600999"

    with pytest.raises(ValueError, match="quality"):
        materialize_exposure_snapshot(
            tmp_path,
            config,
            policy_sha256,
            phase5_manifest,
            tables,
            [_raw_input(tmp_path)],
        )

    assert not (tmp_path / "state" / "latest_exposure_snapshot.txt").exists()


def test_validation_rejects_manifest_identity_tampering(tmp_path: Path) -> None:
    result = _materialize_fixture(tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["policy_sha256"] = "b" * 64
    result.manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    validation = validate_exposure_snapshot(tmp_path, result.snapshot_id)

    assert validation["healthy"] is False
    assert "identity_sha256" in validation["failures"]


def test_validation_requires_matching_quality_report(tmp_path: Path) -> None:
    result = _materialize_fixture(tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest.pop("quality_report")
    result.manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    validation = validate_exposure_snapshot(tmp_path, result.snapshot_id)

    assert validation["healthy"] is False
    assert "missing:quality_report" in validation["failures"]


@pytest.mark.parametrize(
    ("target", "expected_failure"),
    [
        ("manifest", "phase5_manifest_sha256"),
        ("security_master.parquet", "phase5:sha256:security_master.parquet"),
        ("index_membership.parquet", "phase5:sha256:index_membership.parquet"),
    ],
)
def test_validation_rejects_post_publication_phase5_tampering(
    tmp_path: Path, target: str, expected_failure: str
) -> None:
    result = _materialize_fixture(tmp_path)
    phase5_path = tmp_path / "manifests" / "p5-ecaa6e8aeae6b9f8fb25" / "manifest.json"
    if target == "manifest":
        phase5_path.write_bytes(phase5_path.read_bytes() + b" ")
    else:
        phase5 = json.loads(phase5_path.read_text(encoding="utf-8"))
        artifact = next(item for item in phase5["artifacts"] if item["name"] == target)
        path = tmp_path / artifact["path"]
        path.write_bytes(b"tampered")

    validation = validate_exposure_snapshot(tmp_path, result.snapshot_id)

    assert validation["healthy"] is False
    assert expected_failure in validation["failures"]


def test_validation_rejects_substituted_quality_document(tmp_path: Path) -> None:
    result = _materialize_fixture(tmp_path)
    _replace_quality(result.manifest_path, tmp_path, {"status": "pass"})

    validation = validate_exposure_snapshot(tmp_path, result.snapshot_id)

    assert validation["healthy"] is False
    assert "quality_schema" in validation["failures"]
    assert "identity_sha256" in validation["failures"]


def test_validation_rejects_internally_inconsistent_quality_check(
    tmp_path: Path,
) -> None:
    result = _materialize_fixture(tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    quality_path = tmp_path / manifest["quality_report"]["path"]
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["checks"]["duplicate_keys"] = {
        "severity": "error",
        "status": "pass",
        "count": 1,
    }
    _replace_quality(result.manifest_path, tmp_path, quality)

    validation = validate_exposure_snapshot(tmp_path, result.snapshot_id)

    assert validation["healthy"] is False
    assert "quality_checks" in validation["failures"]


def test_validation_rejects_quality_row_count_mismatch(tmp_path: Path) -> None:
    result = _materialize_fixture(tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    quality_path = tmp_path / manifest["quality_report"]["path"]
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["summary"]["market_cap_count"] = 999
    _replace_quality(result.manifest_path, tmp_path, quality)

    validation = validate_exposure_snapshot(tmp_path, result.snapshot_id)

    assert validation["healthy"] is False
    assert "quality_row_counts" in validation["failures"]


def _materialize_fixture(tmp_path: Path):
    phase5_manifest = _phase5_fixture(tmp_path)
    config, policy_sha256 = load_robustness_config(ROOT / "config" / "robustness.yaml")
    return materialize_exposure_snapshot(
        tmp_path,
        config,
        policy_sha256,
        phase5_manifest,
        _tables(),
        [_raw_input(tmp_path)],
    )


def _tables() -> ExposureTables:
    definitions = normalize_industry_definition(
        pd.DataFrame(
            [["801010.SI", "农林牧渔", "L1", "110000", "SW2021"]],
            columns=["index_code", "industry_name", "level", "industry_code", "src"],
        )
    )
    membership = pd.DataFrame(
        [
            [
                "801010.SI",
                "农林牧渔",
                "",
                "",
                "",
                "",
                "600000.SH",
                "浦发银行",
                "20210101",
                "",
                "Y",
            ]
        ],
        columns=[
            "l1_code",
            "l1_name",
            "l2_code",
            "l2_name",
            "l3_code",
            "l3_name",
            "ts_code",
            "name",
            "in_date",
            "out_date",
            "is_new",
        ],
    )
    market = pd.DataFrame(
        [["600000.SH", "20210104", 123.4, 100.0]],
        columns=["ts_code", "trade_date", "total_mv", "circ_mv"],
    )
    return ExposureTables(
        market_cap=normalize_market_cap(market),
        industry_definition=definitions,
        industry_membership=normalize_industry_membership(
            membership, set(definitions["source_index_code"])
        ),
    )


def _phase5_fixture(tmp_path: Path) -> Path:
    research = tmp_path / "research" / "p5-ecaa6e8aeae6b9f8fb25"
    research.mkdir(parents=True)
    security = research / "security_master.parquet"
    membership = research / "index_membership.parquet"
    pd.DataFrame([{"security_id": "CN:SSE:600000"}]).to_parquet(security, index=False)
    pd.DataFrame([{"security_id": "CN:SSE:600000"}]).to_parquet(membership, index=False)
    manifest_dir = tmp_path / "manifests" / "p5-ecaa6e8aeae6b9f8fb25"
    manifest_dir.mkdir(parents=True)
    manifest = {
        "snapshot_id": "p5-ecaa6e8aeae6b9f8fb25",
        "snapshot_type": "research_market",
        "artifacts": [
            {
                "name": artifact.name,
                "path": artifact.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "row_count": 1,
            }
            for artifact in (security, membership)
        ],
    }
    path = manifest_dir / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _raw_input(tmp_path: Path) -> TushareArtifact:
    raw = tmp_path / "raw" / "tushare" / "fixture" / "request.parquet"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"fixture")
    metadata = raw.with_suffix(".json")
    metadata.write_text("{}\n", encoding="utf-8")
    return TushareArtifact(
        api_name="fixture",
        request_sha256="a" * 64,
        parquet_path=raw,
        metadata_path=metadata,
        sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        row_count=1,
        params={},
        fields=("fixture",),
        ingested_at="2026-07-13T00:00:00Z",
    )


def _replace_quality(manifest_path: Path, data_dir: Path, quality: object) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quality_path = data_dir / manifest["quality_report"]["path"]
    quality_path.write_text(
        json.dumps(quality, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["quality_report"]["sha256"] = hashlib.sha256(
        quality_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
