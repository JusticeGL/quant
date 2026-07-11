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
            rows.append(
                {
                    "trade_date": trade_date,
                    "instrument": instrument,
                    "close": 10.0 + offset * 2 + index * (0.1 + offset * 0.01),
                }
            )
    return pd.DataFrame(rows)


def test_registry_loads_three_unique_reference_factors() -> None:
    registry = FactorRegistry(
        ROOT / "src" / "alpha_lab" / "factors" / "candidates",
        ROOT / "config" / "factor_registry.yaml",
    )

    assert [item.metadata.factor_id for item in registry.all()] == [
        "F0001",
        "F0002",
        "F0003",
    ]
    assert not registry.accepted_factor_ids


def test_reference_factors_satisfy_contract_and_leakage_invariance() -> None:
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
