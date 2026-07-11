from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from alpha_lab.data.config import DataSourceConfig
from alpha_lab.data.providers.akshare_provider import AkshareProvider

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "akshare_stock_zh_a_hist.csv"
)


def source_config(end_date: date = date(2024, 1, 3)) -> DataSourceConfig:
    return DataSourceConfig(
        provider="akshare",
        endpoint="stock_zh_a_hist",
        period="daily",
        start_date=date(2024, 1, 1),
        end_date=end_date,
        adjust="",
        max_attempts=2,
        retry_delay_seconds=0,
        request_interval_seconds=0,
    )


def test_exact_repeat_uses_immutable_raw_cache(tmp_path: Path) -> None:
    calls: list[dict[str, str]] = []

    def fetcher(**kwargs: str) -> pd.DataFrame:
        calls.append(kwargs)
        return pd.read_csv(FIXTURE, dtype={"股票代码": str})

    provider = AkshareProvider(tmp_path, fetcher=fetcher, sleep=lambda _: None)
    first = provider.load_range("600519", source_config())
    raw_bytes = first.artifacts[0].parquet_path.read_bytes()
    second = provider.load_range("600519", source_config())

    assert len(calls) == 1
    assert first.network_requests == 1
    assert calls[0]["timeout"] == 15.0
    assert second.network_requests == 0
    assert second.cache_hits == 1
    assert second.artifacts[0].parquet_path.read_bytes() == raw_bytes
    assert second.frame["日期"].max() == "2024-01-03"


def test_extended_range_fetches_only_the_missing_tail(tmp_path: Path) -> None:
    calls: list[dict[str, str]] = []

    def fetcher(**kwargs: str) -> pd.DataFrame:
        calls.append(kwargs)
        return pd.read_csv(FIXTURE, dtype={"股票代码": str})

    provider = AkshareProvider(tmp_path, fetcher=fetcher, sleep=lambda _: None)
    provider.load_range("600519", source_config())
    updated = provider.load_range("600519", source_config(date(2024, 1, 5)))

    assert len(calls) == 2
    assert calls[1]["start_date"] == "20240104"
    assert calls[1]["end_date"] == "20240105"
    assert updated.network_requests == 1
    assert len(updated.artifacts) == 2


def test_transient_failure_is_retried(tmp_path: Path) -> None:
    attempts = 0

    def fetcher(**_: str) -> pd.DataFrame:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary upstream failure")
        return pd.read_csv(FIXTURE, dtype={"股票代码": str})

    provider = AkshareProvider(tmp_path, fetcher=fetcher, sleep=lambda _: None)
    result = provider.load_range("600519", source_config())

    assert attempts == 2
    assert result.network_requests == 1
