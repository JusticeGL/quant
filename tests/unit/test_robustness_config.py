from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from datetime import date
from pathlib import Path

import pytest
import yaml

from alpha_lab.robustness.config import RobustnessConfig, load_robustness_config
from alpha_lab.robustness.contracts import (
    ExposureSnapshotResult,
    ExposureTables,
    FrozenCandidate,
    RobustnessResult,
)

ROOT = Path(__file__).parents[2]


def test_phase6_policy_has_locked_calendar_and_candidates() -> None:
    config, _ = load_robustness_config(ROOT / "config" / "robustness.yaml")

    assert config.factor_ids == ["F1002", "F1003"]
    assert config.warmup.start == date(2020, 1, 1)
    assert config.warmup.end == date(2020, 12, 31)
    assert [fold.fold_id for fold in config.walk_forward_folds] == [
        "wf_2021",
        "wf_2022",
        "wf_2023",
        "wf_2024",
        "wf_2025",
    ]
    assert len(config.walk_forward_folds) == 5
    assert config.test.start == date(2026, 1, 1)
    assert config.test.end == date(2026, 7, 11)
    assert config.test.access == "human_approval_only"
    assert config.cost_multipliers == [0.5, 1.0, 1.5, 2.0]


def test_phase6_policy_has_exact_gates_and_exposure_source() -> None:
    config, _ = load_robustness_config(ROOT / "config" / "robustness.yaml")

    assert config.minimum_fold_coverage == 0.70
    assert config.minimum_direction_consistent_folds == 4
    assert config.minimum_industry_neutral_ic_retention == 0.50
    assert config.minimum_industry_observation_coverage == 0.98
    assert config.size_correlation_risk_threshold == 0.30
    assert config.exposure_source.classification_standard == "SW2021"
    assert config.exposure_source.endpoints.market_cap == "daily_basic"
    assert config.exposure_source.endpoints.industry_classification == "index_classify"
    assert config.exposure_source.endpoints.industry_membership == "index_member_all"


def test_phase6_policy_rejects_fold_overlap_with_test() -> None:
    document = yaml.safe_load((ROOT / "config" / "robustness.yaml").read_text())
    document["walk_forward_folds"][-1]["end"] = "2026-01-02"

    with pytest.raises(ValueError, match="test boundary"):
        RobustnessConfig.model_validate(document)


@pytest.mark.parametrize(
    ("boundary", "value"),
    [("start", "2026-01-02"), ("end", "2027-07-11")],
)
def test_phase6_policy_rejects_locked_test_boundary_drift(
    boundary: str, value: str
) -> None:
    document = yaml.safe_load((ROOT / "config" / "robustness.yaml").read_text())
    document["test"][boundary] = value

    with pytest.raises(ValueError, match="locked test range"):
        RobustnessConfig.model_validate(document)


def test_phase6_policy_hashes_validated_canonical_content() -> None:
    config, digest = load_robustness_config(ROOT / "config" / "robustness.yaml")
    canonical = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    assert digest == hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize("value", [0, -0.01, 1.01])
def test_industry_observation_coverage_is_a_strict_unit_interval(value: float) -> None:
    document = yaml.safe_load((ROOT / "config" / "robustness.yaml").read_text())
    document["minimum_industry_observation_coverage"] = value

    with pytest.raises(ValueError):
        RobustnessConfig.model_validate(document)


def test_phase6_contracts_are_frozen_dataclasses() -> None:
    expected_fields = {
        ExposureTables: {
            "market_cap",
            "industry_definition",
            "industry_membership",
        },
        ExposureSnapshotResult: {
            "snapshot_id",
            "snapshot_dir",
            "quality_report_path",
            "manifest_path",
            "manifest_sha256",
            "quality_status",
        },
        FrozenCandidate: {
            "freeze_id",
            "factor_id",
            "freeze_path",
            "freeze_sha256",
        },
        RobustnessResult: {
            "freeze_id",
            "output_dir",
            "report_path",
            "report_sha256",
            "passed",
        },
    }

    for contract, names in expected_fields.items():
        assert is_dataclass(contract)
        assert contract.__dataclass_params__.frozen is True
        assert {field.name for field in fields(contract)} == names
