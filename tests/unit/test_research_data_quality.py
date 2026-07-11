from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_lab.research_data.config import load_research_data_config
from alpha_lab.research_data.contracts import ResearchTables
from alpha_lab.research_data.quality import build_research_quality_report

ROOT = Path(__file__).resolve().parents[2]


def _tables(*, overlapping: bool = False) -> ResearchTables:
    security = pd.DataFrame(
        [
            {
                "security_id": "CN:SSE:600001",
                "list_status": "D",
                "list_date": pd.Timestamp("2010-01-01"),
                "delist_date": pd.Timestamp("2022-12-31"),
            }
        ]
    )
    membership = pd.DataFrame(
        [
            {
                "index_id": "CN:INDEX:000300.SH",
                "security_id": "CN:SSE:600001",
                "effective_from": pd.Timestamp("2021-01-01"),
                "effective_to": pd.Timestamp("2021-12-31"),
                "known_at": pd.Timestamp("2021-01-01", tz="UTC"),
            }
        ]
    )
    if overlapping:
        membership = pd.concat(
            [
                membership,
                membership.assign(
                    effective_from=pd.Timestamp("2021-06-01"),
                    effective_to=pd.Timestamp("2022-01-01"),
                ),
            ],
            ignore_index=True,
        )
    return ResearchTables(
        security_master=security,
        security_name_history=pd.DataFrame(
            columns=["security_id", "effective_from", "effective_to"]
        ),
        trading_calendar=pd.DataFrame(
            [
                {
                    "exchange": "SSE",
                    "calendar_date": pd.Timestamp("2021-01-04"),
                    "is_open": True,
                }
            ]
        ),
        index_membership=membership,
        daily_bar=pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2021-01-04"),
                    "security_id": "CN:SSE:600001",
                }
            ]
        ),
        adjustment_factor=pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2021-01-04"),
                    "security_id": "CN:SSE:600001",
                    "factor_type": "tushare_adj",
                    "adj_factor": 1.2,
                }
            ]
        ),
        suspension=pd.DataFrame(
            columns=["security_id", "effective_from", "effective_to"]
        ),
        daily_status=pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2021-01-04"),
                    "security_id": "CN:SSE:600001",
                    "is_suspended": pd.NA,
                    "is_st": pd.NA,
                }
            ]
        ),
    )


def test_overlapping_membership_is_error() -> None:
    report = build_research_quality_report(
        _tables(overlapping=True), load_research_data_config(ROOT / "config")
    )

    assert report["status"] == "error"
    assert report["checks"]["membership_overlap"]["count"] == 1


def test_delisted_security_is_retained_and_nullable_status_is_warning() -> None:
    report = build_research_quality_report(
        _tables(), load_research_data_config(ROOT / "config")
    )

    assert report["status"] == "warning"
    assert report["summary"]["delisted_security_count"] == 1
    assert report["checks"]["nullable_status"]["count"] == 2


def test_nonpositive_adjustment_factor_is_error() -> None:
    tables = _tables()
    tables.adjustment_factor.loc[0, "adj_factor"] = 0.0

    report = build_research_quality_report(
        tables, load_research_data_config(ROOT / "config")
    )

    assert report["status"] == "error"
    assert report["checks"]["invalid_adjustment_factor"]["count"] == 1
