from __future__ import annotations

from datetime import date
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


@pytest.mark.parametrize(
    ("trade_date", "stamp_sell", "transfer"),
    [
        (date(2021, 1, 1), 0.001, 0.00002),
        (date(2022, 4, 28), 0.001, 0.00002),
        (date(2022, 4, 29), 0.001, 0.00001),
        (date(2023, 8, 27), 0.001, 0.00001),
        (date(2023, 8, 28), 0.0005, 0.00001),
        (date(2025, 12, 31), 0.0005, 0.00001),
    ],
)
def test_locked_cost_policy_covers_walk_forward_and_boundaries(
    trade_date: date, stamp_sell: float, transfer: float
) -> None:
    config = load_phase2_config(ROOT / "config")
    rule = config.costs.rule_for(trade_date)

    assert rule.commission_rate == 0.0003
    assert rule.minimum_commission == 5.0
    assert rule.stamp_duty_rate_sell == stamp_sell
    assert rule.transfer_fee_rate_buy == transfer
    assert rule.transfer_fee_rate_sell == transfer


def test_cost_policy_rejects_pre_phase6_uncovered_date() -> None:
    config = load_phase2_config(ROOT / "config")

    with pytest.raises(ValueError, match="exactly one cost rule"):
        config.costs.rule_for(date(2020, 12, 31))


def test_pinned_qlib_exposes_exact_alpha158_contract() -> None:
    expressions, names = alpha158_definition()

    assert len(expressions) == 158
    assert len(names) == 158
    assert len(set(names)) == 158
    assert names[:3] == ["KMID", "KLEN", "KMID2"]
