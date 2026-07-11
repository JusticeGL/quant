from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.research_data.normalize import (
    normalize_adjustment_factors,
    normalize_daily_bars,
    normalize_index_membership_intervals,
    normalize_name_history,
    normalize_security_master,
    normalize_suspensions,
    reconstruct_weight_membership,
    to_security_id,
)


def test_security_master_keeps_delisted_security() -> None:
    raw = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "symbol": "600001",
                "name": "示例退市",
                "market": "主板",
                "exchange": "SSE",
                "curr_type": "CNY",
                "list_status": "D",
                "list_date": "20000101",
                "delist_date": "20221231",
            }
        ]
    )

    result = normalize_security_master(raw, ingested_at="2026-07-11T00:00:00Z")

    assert result.loc[0, "security_id"] == "CN:SSE:600001"
    assert result.loc[0, "list_status"] == "D"
    assert result.loc[0, "delist_date"] == pd.Timestamp("2022-12-31")
    assert result.loc[0, "known_at"] == pd.Timestamp("2026-07-11", tz="UTC")


def test_name_history_derives_st_only_inside_effective_interval() -> None:
    raw = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "name": "*ST示例",
                "start_date": "20210105",
                "end_date": "20210630",
                "ann_date": "20210104",
                "change_reason": "特别处理",
            },
            {
                "ts_code": "600001.SH",
                "name": "示例股份",
                "start_date": "20210701",
                "end_date": None,
                "ann_date": "20210630",
                "change_reason": "撤销特别处理",
            },
        ]
    )

    result = normalize_name_history(raw)

    assert result["is_st"].tolist() == [True, False]
    assert result["known_at"].tolist() == [
        pd.Timestamp("2021-01-04", tz="UTC"),
        pd.Timestamp("2021-06-30", tz="UTC"),
    ]
    assert result.loc[0, "effective_from"] == pd.Timestamp("2021-01-05")


def test_membership_uses_announcement_or_conservative_effective_date() -> None:
    raw = pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": "600001.SH",
                "in_date": "20210105",
                "out_date": None,
                "ann_date": "20210104",
                "weight": 0.4,
            },
            {
                "index_code": "000300.SH",
                "con_code": "000001.SZ",
                "in_date": "20210106",
                "out_date": None,
                "ann_date": None,
                "weight": 0.3,
            },
        ]
    )

    result = normalize_index_membership_intervals(raw, "000300.SH")

    assert result["known_at"].tolist() == [
        pd.Timestamp("2021-01-04", tz="UTC"),
        pd.Timestamp("2021-01-06", tz="UTC"),
    ]
    assert result["known_at_source"].tolist() == [
        "announcement_date",
        "effective_date_fallback",
    ]


def test_weight_observations_reconstruct_non_overlapping_intervals() -> None:
    raw = pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": "600001.SH",
                "trade_date": "20210129",
                "weight": 0.4,
            },
            {
                "index_code": "000300.SH",
                "con_code": "600001.SH",
                "trade_date": "20210226",
                "weight": 0.5,
            },
            {
                "index_code": "000300.SH",
                "con_code": "000002.SZ",
                "trade_date": "20210129",
                "weight": 0.2,
            },
            {
                "index_code": "000300.SH",
                "con_code": "000002.SZ",
                "trade_date": "20210226",
                "weight": 0.2,
            },
            {
                "index_code": "000300.SH",
                "con_code": "000002.SZ",
                "trade_date": "20210331",
                "weight": 0.2,
            },
            {
                "index_code": "000300.SH",
                "con_code": "000003.SZ",
                "trade_date": "20210331",
                "weight": 0.1,
            },
        ]
    )

    result = reconstruct_weight_membership(raw, "000300.SH")
    first = result.loc[result["security_id"] == "CN:SSE:600001"].iloc[0]

    assert first["effective_from"] == pd.Timestamp("2021-01-29")
    assert first["effective_to"] == pd.Timestamp("2021-03-30")
    assert first["known_at_source"] == "effective_date_fallback"


def test_daily_bars_keep_unadjusted_prices_and_normalize_units() -> None:
    raw = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "trade_date": "20210104",
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.1,
                "pre_close": 9.8,
                "vol": 100.0,
                "amount": 200.0,
            }
        ]
    )

    result = normalize_daily_bars(raw)

    assert result.loc[0, "close"] == 10.1
    assert result.loc[0, "volume_shares"] == 10_000.0
    assert result.loc[0, "amount_cny"] == 200_000.0
    assert "adj_factor" not in result.columns


def test_adjustment_factor_must_be_positive() -> None:
    raw = pd.DataFrame(
        [{"ts_code": "600001.SH", "trade_date": "20210104", "adj_factor": 0}]
    )

    with pytest.raises(ValueError, match="positive"):
        normalize_adjustment_factors(raw)


def test_suspension_known_at_never_precedes_announcement() -> None:
    raw = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "suspend_date": "20210105",
                "resume_date": "20210108",
                "ann_date": "20210104",
                "suspend_type": "盘中停牌",
                "suspend_reason": "重大事项",
            }
        ]
    )

    result = normalize_suspensions(raw)

    assert result.loc[0, "known_at"] == pd.Timestamp("2021-01-04", tz="UTC")
    assert result.loc[0, "effective_from"] == pd.Timestamp("2021-01-05")
    assert result.loc[0, "effective_to"] == pd.Timestamp("2021-01-07")


def test_security_id_rejects_unknown_exchange() -> None:
    with pytest.raises(ValueError, match="exchange"):
        to_security_id("600001.XX")
