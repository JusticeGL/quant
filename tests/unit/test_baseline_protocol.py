from __future__ import annotations

from datetime import date

import pandas as pd

from alpha_lab.baseline.pipeline import _select_observations


def test_training_labels_are_purged_when_outcome_crosses_boundary() -> None:
    dates = [date(2024, 1, day) for day in range(2, 9)]
    dataset = pd.DataFrame(
        {
            "datetime": pd.to_datetime(dates),
            "instrument": ["A"] * len(dates),
            "LABEL": [0.01] * len(dates),
        }
    )

    selected = _select_observations(
        dataset,
        start=dates[0],
        signal_end=dates[4],
        outcome_end=dates[4],
        market_dates=dates,
        outcome_steps=2,
    )

    assert selected["datetime"].dt.date.tolist() == dates[:3]
