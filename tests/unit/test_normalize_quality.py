from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_lab.data.normalize import (
    MINIMUM_COLUMNS,
    normalize_akshare_daily,
    normalize_baostock_daily,
)
from alpha_lab.data.quality import build_quality_report

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "akshare_stock_zh_a_hist.csv"
)


def normalized_frame() -> pd.DataFrame:
    raw = pd.read_csv(FIXTURE, dtype={"股票代码": str})
    return normalize_akshare_daily(
        raw,
        symbol="600519",
        ingested_at="2026-07-10T00:00:00+00:00",
    )


def test_normalization_maps_to_minimum_schema_without_inventing_status() -> None:
    frame = normalized_frame()

    assert list(frame.columns) == list(MINIMUM_COLUMNS)
    assert frame["instrument"].unique().tolist() == ["SH600519"]
    assert frame["trade_date"].dt.strftime("%Y-%m-%d").tolist()[0] == "2024-01-02"
    assert frame["close"].tolist() == [1710.0, 1705.0, 1722.0]
    assert frame["volume"].tolist()[0] == 2_500_000.0
    for field in (
        "adj_factor",
        "suspend",
        "limit_up",
        "limit_down",
        "is_st",
        "list_date",
        "delist_date",
    ):
        assert frame[field].isna().all()


def test_quality_report_explicitly_marks_missing_status_fields() -> None:
    report = build_quality_report(normalized_frame())

    assert report["row_count"] == 3
    assert report["instrument_count"] == 1
    assert report["duplicates"]["count"] == 0
    assert report["invalid_rows"]["count"] == 0
    assert set(report["missing_status_fields"]) == {
        "adj_factor",
        "suspend",
        "limit_up",
        "limit_down",
        "is_st",
        "list_date",
        "delist_date",
    }
    assert report["status"] == "warning"


def test_quality_report_detects_duplicate_and_invalid_ohlc() -> None:
    frame = normalized_frame()
    bad = frame.iloc[[0]].copy()
    bad.loc[:, "high"] = bad["low"] - 1
    frame = pd.concat([frame, bad], ignore_index=True)

    report = build_quality_report(frame)

    assert report["duplicates"]["count"] == 2
    assert report["invalid_rows"]["count"] == 1
    assert report["status"] == "error"


def test_quality_report_detects_a_missing_configured_instrument() -> None:
    report = build_quality_report(
        normalized_frame(), expected_instruments={"SH600519", "SZ000001"}
    )

    assert report["missing_instruments"] == ["SZ000001"]
    assert report["status"] == "error"


def test_baostock_normalization_preserves_share_volume_and_known_status() -> None:
    raw = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "open": "54.63",
                "high": "55.05",
                "low": "54.20",
                "close": "54.56",
                "volume": "23905697",
                "amount": "1305162003.76",
                "tradestatus": "1",
                "isST": "0",
            }
        ]
    )

    frame = normalize_baostock_daily(
        raw, symbol="000333", ingested_at="2026-07-10T00:00:00+00:00"
    )

    assert frame.loc[0, "instrument"] == "SZ000333"
    assert frame.loc[0, "volume"] == 23_905_697.0
    assert frame.loc[0, "amount"] == 1_305_162_003.76
    assert frame.loc[0, "suspend"] == False  # noqa: E712
    assert frame.loc[0, "is_st"] == False  # noqa: E712
    assert frame.loc[0, "source"] == "baostock.query_history_k_data_plus"
