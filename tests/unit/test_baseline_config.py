from __future__ import annotations

from pathlib import Path

import pytest

from alpha_lab.baseline.config import load_phase2_config
from alpha_lab.baseline.features import alpha158_definition

ROOT = Path(__file__).resolve().parents[2]


def test_phase2_config_is_locked_and_hashes_are_stable() -> None:
    first = load_phase2_config(ROOT / "config")
    second = load_phase2_config(ROOT / "config")

    assert first.config_sha256 == second.config_sha256
    assert first.splits.locked is True
    assert first.splits.test.locked is True
    assert first.splits.validation.end < first.splits.test.start
    assert first.costs.rule_for(first.splits.validation.start).commission_assumption


def test_cost_policy_rejects_uncovered_date() -> None:
    config = load_phase2_config(ROOT / "config")

    with pytest.raises(ValueError, match="exactly one cost rule"):
        config.costs.rule_for(config.costs.rules[0].effective_from.replace(year=2020))


def test_pinned_qlib_exposes_exact_alpha158_contract() -> None:
    expressions, names = alpha158_definition()

    assert len(expressions) == 158
    assert len(names) == 158
    assert len(set(names)) == 158
    assert names[:3] == ["KMID", "KLEN", "KMID2"]
