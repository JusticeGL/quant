from __future__ import annotations

import pandas as pd

from alpha_lab.baseline.analysis import analyze_signals


def test_signal_analysis_uses_cross_sectional_daily_correlations() -> None:
    predictions = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"] * 3),
            "instrument": ["A", "B", "C"] * 2,
            "score": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "label": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }
    )

    result = analyze_signals(predictions, 252)

    assert result["coverage"] == 1.0
    assert result["mean_ic"] == 1.0
    assert result["mean_rank_ic"] == 1.0
    assert result["mean_top_bottom_spread"] == 2.0
