from __future__ import annotations

from datetime import date

import pandas as pd

from alpha_lab.research_data.universe import universe_as_of


def _securities() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "security_id": "CN:SSE:600001",
                "list_date": pd.Timestamp("2000-01-01"),
                "delist_date": pd.NaT,
            },
            {
                "security_id": "CN:SSE:600002",
                "list_date": pd.Timestamp("2000-01-01"),
                "delist_date": pd.Timestamp("2022-12-31"),
            },
        ]
    )


def _membership() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "index_id": "CN:INDEX:000300.SH",
                "security_id": "CN:SSE:600001",
                "effective_from": pd.Timestamp("2021-01-01"),
                "effective_to": pd.NaT,
                "known_at": pd.Timestamp("2021-01-05", tz="UTC"),
                "weight": 0.5,
            },
            {
                "index_id": "CN:INDEX:000300.SH",
                "security_id": "CN:SSE:600002",
                "effective_from": pd.Timestamp("2021-01-01"),
                "effective_to": pd.Timestamp("2022-12-31"),
                "known_at": pd.Timestamp("2021-01-01", tz="UTC"),
                "weight": 0.5,
            },
        ]
    )


def test_membership_announced_later_is_invisible() -> None:
    result = universe_as_of(_securities(), _membership(), date(2021, 1, 4))

    assert set(result["security_id"]) == {"CN:SSE:600002"}


def test_delisted_security_remains_historical_but_not_after_delisting() -> None:
    before = universe_as_of(_securities(), _membership(), date(2021, 6, 1))
    after = universe_as_of(_securities(), _membership(), date(2023, 6, 1))

    assert "CN:SSE:600002" in set(before["security_id"])
    assert "CN:SSE:600002" not in set(after["security_id"])


def test_query_returns_stable_sorted_rows() -> None:
    result = universe_as_of(_securities(), _membership(), date(2021, 6, 1))

    assert result["security_id"].tolist() == ["CN:SSE:600001", "CN:SSE:600002"]
