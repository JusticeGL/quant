from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_phase4_delivery_files_and_stable_make_targets_exist() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("mining-round", "mining-loop", "report"):
        assert f"\n{target}:" in makefile

    assert (ROOT / ".agents" / "skills" / "factor-mine" / "SKILL.md").is_file()
    for name in ("hypothesis", "proposal", "decision"):
        assert (ROOT / "schemas" / f"{name}.schema.json").is_file()


def test_phase4_schemas_are_strict_json_objects() -> None:
    for path in sorted((ROOT / "schemas").glob("*.schema.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["type"] == "object"
        assert document["additionalProperties"] is False
        assert document["required"]


def test_mining_policy_requires_human_approval() -> None:
    document = yaml.safe_load(
        (ROOT / "config" / "mining.yaml").read_text(encoding="utf-8")
    )
    assert document["default_rounds"] == 5
    assert document["require_human_approval_for_acceptance"] is True


def test_small_audit_outputs_are_trackable_but_payloads_are_ignored() -> None:
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "experiments/**/*" in lines
    for pattern in (
        "!experiments/**/run_manifest.json",
        "!experiments/**/research_brief.md",
        "!experiments/**/test_report.json",
        "!experiments/**/decision.json",
        "!experiments/**/final_report.md",
    ):
        assert pattern in lines
