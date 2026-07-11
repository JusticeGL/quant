from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.evaluation.metrics import calculate_factor_metrics, factor_correlations


def test_factor_metrics_include_groups_turnover_and_stability() -> None:
    rows = []
    for day in pd.bdate_range("2024-04-15", periods=6):
        for index, instrument in enumerate(("A", "B", "C", "D", "E"), start=1):
            rows.append(
                {
                    "trade_date": day,
                    "instrument": instrument,
                    "value": float(index),
                    "score": float(index),
                    "label": float(index) / 100,
                }
            )
    frame = pd.DataFrame(rows)

    result = calculate_factor_metrics(
        frame, expected_rows=len(frame), group_count=5, annualization_days=252
    )

    assert result["coverage"] == 1.0
    assert result["mean_rank_ic"] == pytest.approx(1.0)
    assert result["top_minus_bottom_return"] == pytest.approx(0.04)
    assert result["group_monotonicity"] == pytest.approx(1.0)
    assert "2024-04" in result["stability"]["monthly"]
    assert result["factor_turnover"] == 0.0


def test_factor_correlations_align_on_date_and_instrument() -> None:
    base = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "instrument": ["A", "B"],
            "score": [1.0, 2.0],
        }
    )
    other = base.assign(score=[-1.0, -2.0])

    result = factor_correlations(base, {"F0002": other})

    assert result["F0002"] == pytest.approx(-1.0)
