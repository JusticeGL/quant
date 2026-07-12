from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from alpha_lab.research_data.config import TushareSourceConfig
from alpha_lab.research_data.provider import (
    TushareProvider,
    TushareProviderError,
)


class FakeTransport:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object], float]] = []

    def __call__(
        self, url: str, payload: dict[str, object], timeout: float
    ) -> dict[str, Any]:
        self.calls.append((url, payload, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, dict)
        return response


def _source(*, max_attempts: int = 3) -> TushareSourceConfig:
    return TushareSourceConfig(
        provider="tushare",
        request_timeout_seconds=30,
        max_attempts=max_attempts,
        retry_delay_seconds=0,
        request_interval_seconds=0,
    )


def _success() -> dict[str, object]:
    return {
        "code": 0,
        "msg": None,
        "data": {
            "fields": ["ts_code", "name"],
            "items": [["600000.SH", "浦发银行"]],
        },
    }


def test_query_caches_response_without_secret(tmp_path: Path) -> None:
    token = "local-secret-token"
    transport = FakeTransport([_success()])
    provider = TushareProvider(
        tmp_path,
        _source(),
        token=token,
        http_url="https://example.test",
        transport=transport,
        sleep=lambda _: None,
    )

    first = provider.query("stock_basic", {"list_status": "L"}, ("ts_code", "name"))
    second = provider.query("stock_basic", {"list_status": "L"}, ("ts_code", "name"))

    assert len(transport.calls) == 1
    assert first.network_requests == 1
    assert second.network_requests == 0
    assert second.cache_hits == 1
    assert first.artifact.sha256 == second.artifact.sha256
    assert first.frame.to_dict("records") == [
        {"ts_code": "600000.SH", "name": "浦发银行"}
    ]
    metadata = first.artifact.metadata_path.read_text(encoding="utf-8")
    assert token not in metadata
    assert token not in str(first.artifact.parquet_path)
    document = json.loads(metadata)
    assert document["params"] == {"list_status": "L"}
    assert document["http_host"] == "example.test"
    assert "token" not in document


def test_nonzero_provider_code_is_not_empty_data(tmp_path: Path) -> None:
    token = "local-secret-token"
    transport = FakeTransport(
        [{"code": 2002, "msg": f"permission denied {token}", "data": None}]
    )
    provider = TushareProvider(
        tmp_path,
        _source(),
        token=token,
        http_url="https://example.test",
        transport=transport,
        sleep=lambda _: None,
    )

    with pytest.raises(TushareProviderError, match=r"code=2002.*\*\*\*") as caught:
        provider.query("suspend_d", {}, ("ts_code", "suspend_date"))

    assert token not in str(caught.value)
    assert not list((tmp_path / "raw").rglob("*.parquet"))


def test_query_retries_transport_error_then_succeeds(tmp_path: Path) -> None:
    transport = FakeTransport([ConnectionError("temporary"), _success()])
    provider = TushareProvider(
        tmp_path,
        _source(max_attempts=2),
        token="secret",
        http_url="https://example.test",
        transport=transport,
        sleep=lambda _: None,
    )

    result = provider.query("stock_basic", {"list_status": "L"}, ("ts_code", "name"))

    assert result.network_requests == 1
    assert len(transport.calls) == 2


def test_tampered_raw_cache_stops_reuse(tmp_path: Path) -> None:
    transport = FakeTransport([_success()])
    provider = TushareProvider(
        tmp_path,
        _source(),
        token="secret",
        http_url="https://example.test",
        transport=transport,
        sleep=lambda _: None,
    )
    result = provider.query("stock_basic", {"list_status": "L"}, ("ts_code", "name"))
    result.artifact.parquet_path.write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        provider.query("stock_basic", {"list_status": "L"}, ("ts_code", "name"))


def test_returned_fields_must_match_request(tmp_path: Path) -> None:
    transport = FakeTransport(
        [
            {
                "code": 0,
                "msg": None,
                "data": {"fields": ["ts_code"], "items": [["600000.SH"]]},
            }
        ]
    )
    provider = TushareProvider(
        tmp_path,
        _source(),
        token="secret",
        http_url="https://example.test",
        transport=transport,
        sleep=lambda _: None,
    )

    with pytest.raises(TushareProviderError, match="returned fields"):
        provider.query("stock_basic", {"list_status": "L"}, ("ts_code", "name"))


def test_extra_returned_fields_are_ignored(tmp_path: Path) -> None:
    transport = FakeTransport(
        [
            {
                "code": 0,
                "msg": None,
                "data": {
                    "fields": ["ts_code", "name", "unused"],
                    "items": [["600000.SH", "浦发银行", "extra"]],
                },
            }
        ]
    )
    provider = TushareProvider(
        tmp_path,
        _source(),
        token="secret",
        http_url="https://example.test",
        transport=transport,
        sleep=lambda _: None,
    )

    result = provider.query("stock_basic", {"list_status": "L"}, ("ts_code", "name"))

    assert result.frame.columns.tolist() == ["ts_code", "name"]
    assert result.frame.iloc[0].to_dict() == {
        "ts_code": "600000.SH",
        "name": "浦发银行",
    }


def test_token_and_https_endpoint_are_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="TUSHARE_TOKEN"):
        TushareProvider(
            tmp_path,
            _source(),
            token="",
            http_url="https://example.test",
        )
    with pytest.raises(ValueError, match="HTTPS"):
        TushareProvider(
            tmp_path,
            _source(),
            token="secret",
            http_url="http://example.test",
        )
