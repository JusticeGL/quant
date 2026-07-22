from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]
COMMANDS = (
    "exposure-probe",
    "exposure-bootstrap",
    "robustness-freeze",
    "robustness-eval",
    "test-request",
    "test-approve",
    "final-test",
)


def test_phase6_make_targets_exist_and_guard_required_variables() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in COMMANDS:
        assert f"\n{target}:" in makefile
    for target, variable in {
        "robustness-freeze": "ID",
        "robustness-eval": "FREEZE",
        "test-request": "FREEZE",
        "test-approve": "REQUEST",
        "final-test": "APPROVAL",
    }.items():
        recipe = makefile.split(f"\n{target}:", 1)[1].split("\n\n", 1)[0]
        assert f"$({variable})" in recipe
        assert "test -n" in recipe
    approval_recipe = makefile.split("\ntest-approve:", 1)[1].split("\n\n", 1)[0]
    for variable in ("REQUEST", "APPROVER", "CONFIRM"):
        assert f"$({variable})" in approval_recipe


def test_phase6_audit_allowlist_is_exact_and_large_outputs_stay_ignored(
    tmp_path: Path,
) -> None:
    del tmp_path
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    expected = (
        "!experiments/phase6/freeze-*/freeze.json",
        "!experiments/phase6/freeze-*/walk_forward.json",
        "!experiments/phase6/freeze-*/cost_sensitivity.json",
        "!experiments/phase6/freeze-*/exposure_report.json",
        "!experiments/phase6/freeze-*/robustness_report.md",
        "!experiments/phase6/freeze-*/test_request.json",
        "!experiments/phase6/freeze-*/approvals/approval-*.json",
        "!experiments/phase6/freeze-*/final/final-*/result.json",
        "!experiments/phase6/freeze-*/final/final-*/report.md",
    )
    assert all(rule in rules for rule in expected)
    phase6_negations = [
        rule for rule in rules if rule.startswith("!experiments/phase6/")
    ]
    assert all(rule.endswith(("/", ".json", ".md")) for rule in phase6_negations)
    assert "*.parquet" in rules
    assert "/data/" in rules
    assert ".env" in rules


def test_phase6_documentation_covers_safety_and_operating_contract() -> None:
    document = (ROOT / "docs/phase6_robustness.md").read_text(encoding="utf-8")
    for phrase in (
        "2020-01-01",
        "2021-01-01",
        "2025-12-31",
        "2026-01-01",
        "2026-07-11",
        "human approval",
        "research-only",
        "cooperative threat model",
        "provider",
        "warning",
    ):
        assert phrase in document
    for command in COMMANDS:
        assert f"make {command}" in document
