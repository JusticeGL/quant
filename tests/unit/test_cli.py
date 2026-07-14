from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import alpha_lab.cli as cli
from alpha_lab.cli import app

FREEZE_ID = f"freeze-{'a' * 64}"
REQUEST_ID = f"request-{'b' * 64}"
APPROVAL_ID = f"approval-{'c' * 64}"


def test_cli_registers_data_database_and_baseline_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    for command in (
        "data-bootstrap",
        "data-update",
        "data-validate",
        "qlib-export",
        "db-init",
        "db-check",
        "baseline",
        "factor-list",
        "factor-eval",
        "mining-init",
        "mining-round",
        "mining-loop",
        "mining-report",
        "research-data-probe",
        "research-data-bootstrap",
        "research-data-update",
        "research-data-validate",
        "universe-asof",
        "exposure-probe",
        "exposure-bootstrap",
        "robustness-freeze",
        "robustness-eval",
        "test-request",
        "test-approve",
        "final-test",
    ):
        assert command in result.output


def test_exposure_bootstrap_syncs_catalog_and_renders_json(
    monkeypatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "data/manifests/p6x-test/manifest.json"
    result = SimpleNamespace(
        snapshot_id="p6x-test",
        quality_status="pass",
        snapshot_dir=tmp_path / "data/exposures/p6x-test",
        manifest_path=manifest,
        manifest_sha256="a" * 64,
        quality_report_path=manifest.parent / "quality_report.json",
    )
    calls: list[tuple[Path, Path, Path]] = []
    monkeypatch.setattr(cli, "build_exposure_snapshot", lambda *_: result)
    monkeypatch.setattr(cli, "sync_exposure_snapshot", lambda *args: calls.append(args))

    invoked = CliRunner().invoke(
        app,
        [
            "exposure-bootstrap",
            "--config-dir",
            str(tmp_path / "config"),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert invoked.exit_code == 0, invoked.output
    assert '"snapshot_id": "p6x-test"' in invoked.output
    assert calls == [(tmp_path / "data/metadata.duckdb", tmp_path / "data", manifest)]


def test_phase6_failure_is_structured_and_does_not_leak_provider_message(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cli,
        "probe_exposure_capabilities",
        lambda *_: (_ for _ in ()).throw(RuntimeError("secret-token-value")),
    )

    invoked = CliRunner().invoke(
        app,
        [
            "exposure-probe",
            "--config-dir",
            str(tmp_path / "config"),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert invoked.exit_code == 1
    assert '"error_type": "RuntimeError"' in invoked.output
    assert "secret-token-value" not in invoked.output


def test_phase6_artifact_ids_reject_path_traversal(tmp_path: Path) -> None:
    invoked = CliRunner().invoke(
        app,
        [
            "robustness-eval",
            "--freeze",
            "..",
            "--experiments-dir",
            str(tmp_path / "experiments"),
        ],
    )

    assert invoked.exit_code == 1
    assert '"error_type": "ValueError"' in invoked.output


def test_test_approve_resolves_request_and_passes_explicit_confirmation(
    monkeypatch, tmp_path: Path
) -> None:
    experiments = tmp_path / "experiments"
    request = experiments / f"phase6/{FREEZE_ID}/test_request.json"
    request.parent.mkdir(parents=True)
    request.write_text(f'{{"request_id":"{REQUEST_ID}"}}\n', encoding="utf-8")
    approval = request.parent / f"approvals/{APPROVAL_ID}.json"

    def approve(*args) -> Path:
        approval.parent.mkdir(parents=True)
        approval.write_text(
            f'{{"approval_id":"{APPROVAL_ID}","freeze_id":"{FREEZE_ID}",'
            f'"request_id":"{REQUEST_ID}"}}\n',
            encoding="utf-8",
        )
        return approval

    monkeypatch.setattr(cli, "approve_test_request", approve)

    invoked = CliRunner().invoke(
        app,
        [
            "test-approve",
            "--request",
            REQUEST_ID,
            "--approver",
            "human@example.com",
            "--confirm",
            "b" * 64,
            "--experiments-dir",
            str(experiments),
        ],
    )

    assert invoked.exit_code == 0, invoked.output
    assert '"approval":' in invoked.output


def test_final_test_uses_only_explicit_approval_artifact(
    monkeypatch, tmp_path: Path
) -> None:
    experiments = tmp_path / "experiments"
    approval = experiments / f"phase6/{FREEZE_ID}/approvals/{APPROVAL_ID}.json"
    approval.parent.mkdir(parents=True)
    approval.write_text(f'{{"approval_id":"{APPROVAL_ID}"}}\n', encoding="utf-8")
    result = experiments / f"phase6/{FREEZE_ID}/final/final-a/result.json"

    def final(*args) -> Path:
        result.parent.mkdir(parents=True)
        result.write_text(
            '{"status":"test_completed","test_accessed":true,'
            '"test_run_id":"final-a"}\n',
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(cli, "run_final_test", final)

    invoked = CliRunner().invoke(
        app,
        [
            "final-test",
            "--approval",
            APPROVAL_ID,
            "--config-dir",
            str(tmp_path / "config"),
            "--data-dir",
            str(tmp_path / "data"),
            "--experiments-dir",
            str(experiments),
        ],
    )

    assert invoked.exit_code == 0, invoked.output
    assert '"result":' in invoked.output
