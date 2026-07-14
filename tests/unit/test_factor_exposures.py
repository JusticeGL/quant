from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.robustness.exposures import calculate_exposures


def test_exposures_are_point_in_time_and_leave_missing_rows_missing() -> None:
    dates = pd.to_datetime(["2025-01-02"] * 5)
    instruments = ["SH600000", "SH600001", "SZ000001", "SZ000002", "SZ000003"]
    scores = pd.DataFrame(
        {"trade_date": dates, "instrument": instruments, "score": range(1, 6)}
    )
    labels = scores[["trade_date", "instrument"]].assign(
        label=[0.01, 0.02, 0.03, 0.04, 0.05]
    )
    security_ids = [
        "CN:SSE:600000",
        "CN:SSE:600001",
        "CN:SZSE:000001",
        "CN:SZSE:000002",
    ]
    market_cap = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2025-01-02"] * 4),
            "security_id": security_ids,
            "total_market_cap_cny": np.exp([1.0, 2.0, 3.0, 4.0]),
            "known_at": pd.to_datetime(["2025-01-02T00:00:00Z"] * 4),
        }
    )
    industries = pd.DataFrame(
        {
            "industry_id": ["A", "A", "B", "B", "C"],
            "security_id": [*security_ids, "CN:SZSE:000003"],
            "effective_from": pd.to_datetime(["2024-01-01"] * 5),
            "effective_to": [pd.NaT] * 5,
            "known_at": pd.to_datetime(
                ["2024-01-01T00:00:00Z"] * 4 + ["2025-01-03T00:00:00Z"]
            ),
        }
    )

    report = calculate_exposures(
        scores,
        market_cap,
        industries,
        labels,
        size_risk_threshold=1.0,
    )

    assert report["industry"]["joined_rows"] == 4
    assert report["industry"]["input_rows"] == 5
    assert report["industry"]["matched_rows"] == 4
    assert report["industry"]["excluded_rows"] == 1
    assert report["industry"]["coverage"] == 0.8
    assert report["industry"]["neutral_rank_ic"] is not None
    assert report["industry"]["original_joined_rows"] == 4
    assert report["size"]["joined_rows"] == 4
    assert report["size"]["uses"] == "log(total_market_cap_cny)"
    assert report["size"]["risk_threshold"] == 1.0
    assert report["size"]["risk_flag"] is False
    assert report["missing"]["industry_rows"] == 1
    assert report["missing"]["size_rows"] == 1
    assert report["industry"]["by_industry"] == [
        {"industry_id": "A", "mean_score": 1.5, "observations": 2},
        {"industry_id": "B", "mean_score": 3.5, "observations": 2},
    ]
    assert report["industry"]["mean_score_dispersion"] == 1.0
