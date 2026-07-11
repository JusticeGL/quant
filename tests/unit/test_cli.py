from __future__ import annotations

from typer.testing import CliRunner

from alpha_lab.cli import app


def test_cli_registers_all_phase1_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    for command in (
        "data-bootstrap",
        "data-update",
        "data-validate",
        "qlib-export",
        "db-init",
        "db-check",
    ):
        assert command in result.output
