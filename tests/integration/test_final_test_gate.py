from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pytest

from alpha_lab.database import catalog
from alpha_lab.robustness import final_test
from alpha_lab.robustness.approval import canonical_bytes
from alpha_lab.robustness.final_test import run_final_test


def test_missing_approval_fails_before_locked_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(
        final_test,
        "_read_locked_market",
        lambda *_: opened.append("market"),
    )
    with pytest.raises(PermissionError, match="approval"):
        run_final_test(
            tmp_path / "missing.json",
            tmp_path / "config",
            tmp_path / "data",
            tmp_path / "experiments",
        )
    assert opened == []


@pytest.mark.parametrize(
    "stage",
    [
        "approval",
        "request",
        "freeze",
        "candidate",
        "policy",
        "cost",
        "phase5",
        "exposure",
    ],
)
def test_dependency_failure_order_is_before_locked_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    approval = tmp_path / "approval.json"
    approval.write_text("{}\n", encoding="utf-8")
    order: list[str] = []
    stages = [
        "approval",
        "request",
        "freeze",
        "candidate",
        "policy",
        "cost",
        "phase5",
        "exposure",
    ]
    for name in stages:
        def validator(*_: object, _name: str = name) -> dict[str, Any]:
            order.append(_name)
            if _name == stage:
                raise ValueError(f"{_name} invalid")
            return _validated_fixture(tmp_path)

        monkeypatch.setattr(final_test, f"_validate_{name}", validator)

    monkeypatch.setattr(
        final_test,
        "_read_locked_market",
        lambda *_: (_ for _ in ()).throw(AssertionError("locked read reached")),
    )
    with pytest.raises(ValueError, match=f"{stage} invalid"):
        run_final_test(approval, tmp_path / "config", tmp_path / "data", tmp_path)
    assert order == stages[: stages.index(stage) + 1]


def test_final_artifacts_are_idempotent_and_conflicts_are_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    approval = tmp_path / "approval.json"
    approval.write_text("{}\n", encoding="utf-8")
    validated = _validated_fixture(tmp_path)
    for name in (
        "approval",
        "request",
        "freeze",
        "candidate",
        "policy",
        "cost",
        "phase5",
        "exposure",
    ):
        monkeypatch.setattr(final_test, f"_validate_{name}", lambda *_, **__: validated)
    monkeypatch.setattr(final_test, "_read_locked_market", lambda *_: "LOCKED")
    monkeypatch.setattr(
        final_test,
        "_evaluate_locked_test",
        lambda *_: {"metrics": {"mean_rank_ic": -0.01}, "status": "completed"},
    )

    first = run_final_test(approval, tmp_path / "config", tmp_path / "data", tmp_path)
    original = first.read_bytes()
    second = run_final_test(approval, tmp_path / "config", tmp_path / "data", tmp_path)
    assert second == first
    assert second.read_bytes() == original
    report = first.with_name("report.md")
    assert report.is_file()
    assert "-0.01" in report.read_text(encoding="utf-8")

    first.write_text("conflict\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="immutable final-test artifact"):
        run_final_test(approval, tmp_path / "config", tmp_path / "data", tmp_path)


def test_authorized_execution_failure_keeps_immutable_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    approval = tmp_path / "approval.json"
    approval.write_text("{}\n", encoding="utf-8")
    validated = _validated_fixture(tmp_path)
    for name in (
        "approval",
        "request",
        "freeze",
        "candidate",
        "policy",
        "cost",
        "phase5",
        "exposure",
    ):
        monkeypatch.setattr(final_test, f"_validate_{name}", lambda *_, **__: validated)
    monkeypatch.setattr(final_test, "_read_locked_market", lambda *_: "LOCKED")
    monkeypatch.setattr(
        final_test,
        "_evaluate_locked_test",
        lambda *_: (_ for _ in ()).throw(RuntimeError("fixture failure")),
    )

    with pytest.raises(RuntimeError, match="fixture failure"):
        run_final_test(approval, tmp_path / "config", tmp_path / "data", tmp_path)
    result_path = next((tmp_path / "phase6").glob("*/final/*/result.json"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "test_failed"
    assert result["error"] == {"message": "fixture failure", "type": "RuntimeError"}
    assert result_path.with_name("report.md").is_file()


def test_final_test_run_catalog_audit_is_idempotent_and_conflict_safe(
    tmp_path: Path,
) -> None:
    state = _validated_fixture(tmp_path)
    database = tmp_path / "data/metadata.duckdb"
    _seed_final_catalog(database, state)
    state["catalog_database"] = database
    state["started_at"] = "2026-07-14T00:00:00+00:00"
    result = {
        "test_run_id": "final-" + "9" * 64,
        "status": "test_completed",
        "result": {"metrics": {"mean_rank_ic": -0.01}},
    }
    output = tmp_path / "result.json"
    output.write_bytes(canonical_bytes(result))
    output.with_name("report.md").write_text("report\n", encoding="utf-8")

    final_test._record_final_run(state, result, output)
    final_test._record_final_run(state, result, output)
    with duckdb.connect(str(database), read_only=True) as connection:
        row = connection.execute(
            "SELECT count(*), min(status) FROM research.final_test_run"
        ).fetchone()
    assert row == (1, "success")

    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "UPDATE research.final_test_run SET run_sha256 = repeat('0', 64)"
        )
    with pytest.raises(RuntimeError, match="catalog registration conflict"):
        final_test._record_final_run(state, result, output)


def _validated_fixture(tmp_path: Path) -> dict[str, Any]:
    return {
        "approval_id": "approval-" + "a" * 64,
        "approval_sha256": "b" * 64,
        "request_id": "request-" + "c" * 64,
        "request_sha256": "d" * 64,
        "freeze_id": "freeze-" + "e" * 64,
        "freeze_sha256": "f" * 64,
        "factor": {"factor_id": "F1002"},
        "snapshots": {
            "phase5": {"snapshot_id": "p5-" + "1" * 20},
            "exposure": {"snapshot_id": "p6x-" + "2" * 20},
        },
        "policies": {},
        "test": {"start": "2026-01-01", "end": "2026-07-11"},
        "experiments_dir": tmp_path,
    }


def _seed_final_catalog(database: Path, state: dict[str, Any]) -> None:
    catalog.initialize_database(database)
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            """
            INSERT INTO research.factor_definition
                (factor_id, name, family)
            VALUES ('F1002', 'fixture-factor', 'fixture')
            """
        )
        connection.execute(
            """
            INSERT INTO research.factor_version
                (factor_version_id, factor_id, formula, implementation_path,
                 code_sha256, metadata_sha256, lookback, direction)
            VALUES ('F1002-fixture', 'F1002', 'fixture', 'fixture.py',
                    repeat('1', 64), repeat('2', 64), 1, 1)
            """
        )
        connection.execute(
            """
            INSERT INTO research.factor_freeze
                (freeze_id, freeze_sha256, factor_version_id,
                 phase5_snapshot_id, exposure_snapshot_id,
                 robustness_policy_sha256, cost_policy_sha256, code_commit,
                 test_start, test_end, manifest_artifact_id)
            VALUES (?, ?, 'F1002-fixture', ?, ?, repeat('3', 64),
                    repeat('4', 64), 'fixture', DATE '2026-01-01',
                    DATE '2026-07-11', repeat('5', 64))
            """,
            [
                state["freeze_id"],
                state["freeze_sha256"],
                state["snapshots"]["phase5"]["snapshot_id"],
                state["snapshots"]["exposure"]["snapshot_id"],
            ],
        )
        connection.execute(
            """
            INSERT INTO research.test_request
                (request_id, request_sha256, freeze_id, freeze_sha256,
                 robustness_report_sha256, test_start, test_end)
            VALUES (?, ?, ?, ?, repeat('6', 64), DATE '2026-01-01',
                    DATE '2026-07-11')
            """,
            [
                state["request_id"],
                state["request_sha256"],
                state["freeze_id"],
                state["freeze_sha256"],
            ],
        )
        connection.execute(
            """
            INSERT INTO research.test_approval
                (approval_id, approval_sha256, request_id, freeze_id,
                 confirmed_freeze_sha256, test_start, test_end, approver)
            VALUES (?, ?, ?, ?, ?, DATE '2026-01-01', DATE '2026-07-11',
                    'fixture reviewer')
            """,
            [
                state["approval_id"],
                state["approval_sha256"],
                state["request_id"],
                state["freeze_id"],
                state["freeze_sha256"],
            ],
        )
