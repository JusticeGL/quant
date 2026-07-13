from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from alpha_lab.baseline.config import CostConfig, CostRule
from alpha_lab.robustness.config import WalkForwardFold, load_robustness_config
from alpha_lab.robustness.walk_forward import (
    build_fold_labels,
    evaluate_gates,
    scale_costs,
)

ROOT = Path(__file__).resolve().parents[2]


def test_labels_do_not_cross_fold_end() -> None:
    market = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2025-12-29", "2025-12-30", "2025-12-31", "2026-01-02"]
            ),
            "instrument": ["SH600000"] * 4,
            "open": [10.0, 11.0, 12.0, 20.0],
        }
    )
    labels = build_fold_labels(
        market,
        WalkForwardFold(
            fold_id="wf_2025", start=date(2025, 1, 1), end=date(2025, 12, 31)
        ),
    )

    valid = labels.dropna(subset=["label"])
    assert valid["entry_date"].max().date() <= date(2025, 12, 31)
    assert valid["exit_date"].max().date() <= date(2025, 12, 31)
    assert valid.iloc[0]["label"] == 12.0 / 11.0 - 1.0


def test_cost_scaling_includes_every_cost_field() -> None:
    costs = CostConfig(
        schema_version=1,
        policy_id="fixture",
        locked=True,
        currency="CNY",
        notes="fixture",
        rules=[
            CostRule(
                effective_from=date(2020, 1, 1),
                effective_to=None,
                commission_rate=0.001,
                minimum_commission=5.0,
                stamp_duty_rate_buy=0.002,
                stamp_duty_rate_sell=0.003,
                transfer_fee_rate_buy=0.004,
                transfer_fee_rate_sell=0.005,
                commission_assumption=True,
                sources={"fixture": "fixture"},
            )
        ],
    )

    scaled = scale_costs(costs, 1.5)

    original = costs.rules[0]
    result = scaled.rules[0]
    assert result.commission_rate == original.commission_rate * 1.5
    assert result.minimum_commission == original.minimum_commission * 1.5
    assert result.stamp_duty_rate_buy == original.stamp_duty_rate_buy * 1.5
    assert result.stamp_duty_rate_sell == original.stamp_duty_rate_sell * 1.5
    assert result.transfer_fee_rate_buy == original.transfer_fee_rate_buy * 1.5
    assert result.transfer_fee_rate_sell == original.transfer_fee_rate_sell * 1.5
    assert costs.rules[0].minimum_commission == 5.0


def test_gate_boundaries_are_exact_and_size_risk_is_not_a_gate() -> None:
    config, _ = load_robustness_config(ROOT / "config" / "robustness.yaml")
    folds = [
        {"mean_rank_ic": value, "coverage": 0.70}
        for value in (0.01, 0.02, 0.03, 0.04, -0.01)
    ]
    costs = {
        "scenarios": {
            "1.0": {"metrics": {"total_return": 0.01}},
            "2.0": {"metrics": {"total_return": 0.0001}},
        }
    }
    exposures = {
        "industry": {"abs_rank_ic_retention": 0.50},
        "size": {"risk_flag": True},
    }

    gates = evaluate_gates(folds, costs, exposures, config)

    assert gates == {
        "direction_consistency": True,
        "fold_coverage": True,
        "double_cost_direction": True,
        "industry_neutral_retention": True,
    }

    costs["scenarios"]["2.0"]["metrics"]["total_return"] = 0.0
    assert (
        evaluate_gates(folds, costs, exposures, config)["double_cost_direction"]
        is False
    )
