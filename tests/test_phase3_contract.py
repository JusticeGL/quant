from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_phase3_contract_files_and_make_targets_exist() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "\nfactor-list:" in makefile
    assert "\nfactor-eval:" in makefile
    assert (ROOT / "config" / "factor_evaluation.yaml").is_file()
    assert (ROOT / "config" / "factor_registry.yaml").is_file()


def test_evaluation_policy_is_locked() -> None:
    document = yaml.safe_load(
        (ROOT / "config" / "factor_evaluation.yaml").read_text(encoding="utf-8")
    )
    assert document["locked"] is True
    assert document["engineering_only"] is True


def test_three_reference_factors_have_code_metadata_and_tests() -> None:
    candidates = ROOT / "src" / "alpha_lab" / "factors" / "candidates"
    for factor_id in ("F0001", "F0002", "F0003"):
        assert (candidates / f"{factor_id}.py").is_file()
        assert (candidates / f"{factor_id}.yaml").is_file()
    assert (ROOT / "tests" / "unit" / "test_factor_candidates.py").is_file()


def test_phase3_evaluator_remains_separate_from_phase4_orchestration() -> None:
    assert (ROOT / "src" / "alpha_lab" / "evaluation" / "pipeline.py").is_file()
    assert (ROOT / "src" / "alpha_lab" / "mining" / "pipeline.py").is_file()
