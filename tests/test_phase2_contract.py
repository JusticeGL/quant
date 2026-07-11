from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_phase2_configuration_and_make_target_exist() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "\nbaseline:" in makefile
    for name in ("baseline.yaml", "splits.yaml", "costs.yaml"):
        assert (ROOT / "config" / name).is_file()


def test_test_split_is_locked_and_human_approval_only() -> None:
    splits = yaml.safe_load(
        (ROOT / "config" / "splits.yaml").read_text(encoding="utf-8")
    )
    assert splits["locked"] is True
    assert splits["test"]["locked"] is True
    assert splits["test"]["access"] == "human_approval_only"


def test_project_does_not_create_automatic_mining_module() -> None:
    package = ROOT / "src" / "alpha_lab"
    assert not (package / "mining").exists()
