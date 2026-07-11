from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_phase5_files_and_stable_make_targets_exist() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in (
        "research-data-probe",
        "research-data-bootstrap",
        "research-data-update",
        "research-data-validate",
        "universe-asof",
    ):
        assert f"\n{target}:" in makefile
    assert (ROOT / "config" / "research_data.yaml").is_file()
    assert (
        ROOT / "src" / "alpha_lab" / "database" / "sql" / "002_research_data.sql"
    ).is_file()
    assert (ROOT / "docs" / "phase5_research_data.md").is_file()


def test_applied_initial_migration_is_unchanged() -> None:
    path = ROOT / "src" / "alpha_lab" / "database" / "sql" / "001_initial.sql"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "fce0387dbc42f386d52c4def7b6bd76090a61403a75d058e0fdc6b178657dd67"
    )


def test_compose_exposes_secret_safe_data_service() -> None:
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    service = compose["services"]["data"]
    assert service["environment"]["TUSHARE_TOKEN"] == "${TUSHARE_TOKEN:-}"
    assert "TUSHARE_HTTP_URL" in service["environment"]
    assert service["working_dir"] == "/workspace"


def test_example_environment_contains_no_token() -> None:
    document = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "TUSHARE_TOKEN=\n" in document
    assert "TUSHARE_HTTP_URL=https://api.tushare.pro" in document
