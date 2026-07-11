from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_lab.data.config import load_phase1_config
from alpha_lab.data.pipeline import run_ingestion
from alpha_lab.data.providers.akshare_provider import AkshareProvider
from alpha_lab.data.providers.baostock_provider import BaostockProvider

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "akshare_stock_zh_a_hist.csv"


def test_pipeline_repeat_uses_cache_and_preserves_snapshot_identity(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fetcher(**kwargs: str) -> pd.DataFrame:
        calls.append(kwargs["symbol"])
        frame = pd.read_csv(FIXTURE, dtype={"股票代码": str})
        frame["股票代码"] = kwargs["symbol"]
        return frame

    data_root = tmp_path / "data"
    provider = AkshareProvider(data_root, fetcher=fetcher, sleep=lambda _: None)

    first = run_ingestion(ROOT / "config", data_root, provider=provider)
    second = run_ingestion(ROOT / "config", data_root, provider=provider)

    assert len(calls) == 10
    assert first.network_requests == 10
    assert second.network_requests == 0
    assert second.cache_hits == 10
    assert first.snapshot.snapshot_id == second.snapshot.snapshot_id
    assert first.snapshot.manifest_sha256 == second.snapshot.manifest_sha256
    assert first.snapshot.quality_status == "warning"


def test_pipeline_uses_one_auditable_fallback_source_for_the_whole_snapshot(
    tmp_path: Path,
) -> None:
    primary_calls = 0
    fallback_calls: list[str] = []

    def failing_primary(**_: str) -> pd.DataFrame:
        nonlocal primary_calls
        primary_calls += 1
        raise ConnectionError("primary unavailable")

    def fallback_fetcher(**kwargs: str) -> pd.DataFrame:
        fallback_calls.append(kwargs["symbol"])
        code = kwargs["symbol"]
        market = "sh" if code.startswith("6") else "sz"
        return pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "code": f"{market}.{code}",
                    "open": "10.00",
                    "high": "10.20",
                    "low": "9.90",
                    "close": "10.10",
                    "volume": "1000000",
                    "amount": "10050000",
                    "adjustflag": "3",
                    "tradestatus": "1",
                    "isST": "0",
                }
            ]
        )

    data_root = tmp_path / "data"
    primary = AkshareProvider(data_root, fetcher=failing_primary, sleep=lambda _: None)
    fallback = BaostockProvider(
        data_root, fetcher=fallback_fetcher, sleep=lambda _: None
    )

    first = run_ingestion(
        ROOT / "config",
        data_root,
        provider=primary,
        fallback_provider=fallback,
    )
    calls_after_first_run = primary_calls
    second = run_ingestion(
        ROOT / "config",
        data_root,
        provider=primary,
        fallback_provider=fallback,
    )

    assert first.selected_provider == "baostock"
    assert "primary unavailable" in (first.fallback_reason or "")
    assert first.network_requests == 10
    expected_codes = [
        item.code for item in load_phase1_config(ROOT / "config").universe.symbols
    ]
    assert fallback_calls == expected_codes
    assert primary_calls == calls_after_first_run
    assert second.network_requests == 0
    assert second.cache_hits == 10
    assert first.snapshot.snapshot_id == second.snapshot.snapshot_id
