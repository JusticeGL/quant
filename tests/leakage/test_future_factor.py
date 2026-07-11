from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from alpha_lab.evaluation.leakage import audit_factor, inspect_factor_source
from alpha_lab.factors.contract import FactorCandidate, FactorMetadata

ROOT = Path(__file__).resolve().parents[2]


def test_intentional_future_shift_factor_is_rejected() -> None:
    source_path = ROOT / "tests" / "fixtures" / "factors" / "F9999.py"
    metadata = FactorMetadata(
        factor_id="F9999",
        name="intentional_future_shift",
        hypothesis="This fixture must be rejected because it reads tomorrow's close.",
        formula="Ref(close, -1)",
        inputs=["close"],
        lookback=1,
        direction=1,
        family="test_fixture",
        author="test_fixture",
        parent_factor_ids=[],
        created_at=datetime.now(UTC),
        status="candidate",
    )

    def placeholder(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[["trade_date", "instrument"]].assign(value=0.0)

    candidate = FactorCandidate(
        metadata,
        placeholder,
        source_path,
        source_path,
        "a" * 64,
        "b" * 64,
    )
    market = pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2024-01-02", periods=10),
            "instrument": ["A"] * 10,
            "close": range(10),
        }
    )

    issues = inspect_factor_source(source_path, {"close"})
    report = audit_factor(candidate, market)

    assert any(issue.code == "future_shift" for issue in issues)
    assert report.passed is False
    assert report.prefix_invariant is False
    assert report.future_perturbation_invariant is False
