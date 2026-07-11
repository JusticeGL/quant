from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from alpha_lab.data.config import DataSourceConfig
from alpha_lab.data.providers.baostock_provider import BaostockProvider


def source_config() -> DataSourceConfig:
    return DataSourceConfig(
        provider="akshare",
        endpoint="stock_zh_a_hist",
        fallback_provider="baostock",
        period="daily",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
        adjust="",
        request_interval_seconds=0,
        retry_delay_seconds=0,
    )


def test_baostock_fallback_has_its_own_immutable_cache(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetcher(**kwargs: str) -> pd.DataFrame:
        calls.append(kwargs["symbol"])
        return pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "code": "sz.000333",
                    "open": "54.63",
                    "high": "55.05",
                    "low": "54.20",
                    "close": "54.56",
                    "volume": "23905697",
                    "amount": "1305162003.76",
                    "adjustflag": "3",
                    "tradestatus": "1",
                    "isST": "0",
                }
            ]
        )

    provider = BaostockProvider(tmp_path, fetcher=fetcher, sleep=lambda _: None)
    first = provider.load_range("000333", source_config())
    second = provider.load_range("000333", source_config())

    assert calls == ["000333"]
    assert first.network_requests == 1
    assert second.network_requests == 0
    assert second.cache_hits == 1
    assert second.artifacts[0].provider == "baostock"
    assert "raw/baostock" in second.artifacts[0].parquet_path.as_posix()
