from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from alpha_lab.research_data.config import (
    ResearchDataConfig,
    load_research_data_config,
)

ROOT = Path(__file__).resolve().parents[2]


def test_repository_research_data_config_is_bounded() -> None:
    config = load_research_data_config(ROOT / "config")

    assert config.schema_version == 1
    assert config.dataset_id == "csi300_point_in_time"
    assert config.index_code == "000300.SH"
    assert config.start_date == date(2020, 1, 1)
    assert config.end_date == date(2026, 7, 11)
    assert config.membership_method == "interval_or_weight_observation"
    assert config.source.provider == "tushare"
    assert config.source.maximum_concurrency == 4


def test_config_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="end_date"):
        ResearchDataConfig.model_validate(
            {
                "schema_version": 1,
                "dataset_id": "csi300_point_in_time",
                "index_code": "000300.SH",
                "start_date": "2021-01-02",
                "end_date": "2021-01-01",
                "membership_method": "interval_or_weight_observation",
                "stock_statuses": ["L", "D", "P", "G"],
                "calendar_exchange": "SSE",
                "maximum_symbols": 1000,
                "source": {
                    "provider": "tushare",
                    "request_timeout_seconds": 30,
                    "max_attempts": 3,
                    "retry_delay_seconds": 2,
                    "request_interval_seconds": 0.2,
                },
                "endpoints": {
                    "security_master": "stock_basic",
                    "trading_calendar": "trade_cal",
                    "membership_primary": "index_member_all",
                    "membership_fallback": "index_weight",
                    "daily_bar": "daily",
                    "adjustment_factor": "adj_factor",
                    "suspension": "suspend_d",
                    "name_history": "namechange",
                },
            }
        )


def test_config_rejects_unbounded_index_code() -> None:
    document = {
        "schema_version": 1,
        "dataset_id": "csi500_point_in_time",
        "index_code": "000905.SH",
        "start_date": "2020-01-01",
        "end_date": "2026-07-11",
        "membership_method": "interval_or_weight_observation",
        "stock_statuses": ["L", "D", "P", "G"],
        "calendar_exchange": "SSE",
        "maximum_symbols": 1000,
        "source": {
            "provider": "tushare",
            "request_timeout_seconds": 30,
            "max_attempts": 3,
            "retry_delay_seconds": 2,
            "request_interval_seconds": 0.2,
        },
        "endpoints": {
            "security_master": "stock_basic",
            "trading_calendar": "trade_cal",
            "membership_primary": "index_member_all",
            "membership_fallback": "index_weight",
            "daily_bar": "daily",
            "adjustment_factor": "adj_factor",
            "suspension": "suspend_d",
            "name_history": "namechange",
        },
    }

    with pytest.raises(ValueError, match="000300.SH"):
        ResearchDataConfig.model_validate(document)
