from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_makefile_exposes_phase1_targets() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    for target in ("data-bootstrap", "data-update", "data-validate", "qlib-export"):
        assert f"\n{target}:" in makefile


def test_makefile_exposes_database_targets() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    for target in ("db-init", "db-check"):
        assert f"\n{target}:" in makefile


def test_phase1_does_not_create_later_phase_modules() -> None:
    package = ROOT / "src" / "alpha_lab"

    for later_phase in ("backtest", "evaluation", "factors", "mining"):
        assert not (package / later_phase).exists()


def test_generated_data_ignore_is_anchored_to_repository_root() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert "/data/" in gitignore
    assert "data/" not in gitignore
    assert "/data" in dockerignore
    assert "data" not in dockerignore
