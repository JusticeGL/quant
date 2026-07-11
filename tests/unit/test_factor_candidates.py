from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.evaluation.leakage import audit_factor
from alpha_lab.factors.contract import validate_factor_output
from alpha_lab.factors.registry import FactorRegistry

ROOT = Path(__file__).resolve().parents[2]


def _market() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2024-01-02", periods=50)
    for index, trade_date in enumerate(dates):
        for offset, instrument in enumerate(("A", "B", "C")):
            close = 10.0 + offset * 2 + index * (0.1 + offset * 0.01)
            open_price = close * (1.0 + ((index + offset) % 5 - 2) * 0.001)
            rows.append(
                {
                    "trade_date": trade_date,
                    "instrument": instrument,
                    "open": open_price,
                    "high": max(open_price, close) * 1.01,
                    "low": min(open_price, close) * 0.99,
                    "close": close,
                    "volume": 1_000_000.0 + index * 10_000.0 + offset * 1_000.0,
                    "amount": close
                    * (1_000_000.0 + index * 10_000.0 + offset * 1_000.0),
                }
            )
    return pd.DataFrame(rows)


def test_registry_keeps_three_references_and_loads_phase4_candidates() -> None:
    registry = FactorRegistry(
        ROOT / "src" / "alpha_lab" / "factors" / "candidates",
        ROOT / "config" / "factor_registry.yaml",
    )

    factors = registry.all()
    references = [
        item.metadata.factor_id
        for item in factors
        if item.metadata.status == "reference"
    ]
    candidates = [
        item.metadata.factor_id
        for item in factors
        if item.metadata.status == "candidate"
    ]

    assert references == [
        "F0001",
        "F0002",
        "F0003",
    ]
    assert candidates == ["F1000", "F1001", "F1002", "F1003", "F1004"]
    assert not registry.accepted_factor_ids


def test_registered_factors_satisfy_contract_and_leakage_invariance() -> None:
    registry = FactorRegistry(
        ROOT / "src" / "alpha_lab" / "factors" / "candidates",
        ROOT / "config" / "factor_registry.yaml",
    )
    market = _market()

    for candidate in registry.all():
        values = validate_factor_output(candidate, market)
        report = audit_factor(candidate, market)
        assert list(values.columns) == ["trade_date", "instrument", "value"]
        assert np.isfinite(values["value"].dropna()).all()
        assert report.passed, report.to_dict()
