from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from alpha_lab.research_data.pipeline import run_research_data_pipeline
from alpha_lab.research_data.provider import (
    TushareArtifact,
    TushareProviderError,
    TushareQueryResult,
)

ROOT = Path(__file__).resolve().parents[2]


class FixtureProvider:
    def __init__(self, data_root: Path, *, fail_membership: bool = False) -> None:
        self.data_root = data_root
        self.fail_membership = fail_membership
        self.calls: list[tuple[str, dict[str, object]]] = []

    def query(
        self,
        api_name: str,
        params: dict[str, object],
        fields: tuple[str, ...],
    ) -> TushareQueryResult:
        self.calls.append((api_name, dict(params)))
        if self.fail_membership and api_name in {"index_member_all", "index_weight"}:
            raise TushareProviderError(f"permission denied for {api_name}")
        frame = self._response(api_name, params, fields)
        identity = hashlib.sha256(
            f"{api_name}:{sorted(params.items())}:{fields}".encode()
        ).hexdigest()
        path = self.data_root / "raw" / "fixture" / f"{identity}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            frame.to_parquet(path, index=False)
            path.with_suffix(".json").write_text("{}\n", encoding="utf-8")
        artifact = TushareArtifact(
            api_name=api_name,
            request_sha256=identity,
            parquet_path=path,
            metadata_path=path.with_suffix(".json"),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            row_count=len(frame),
            params=dict(params),
            fields=fields,
            ingested_at="2026-07-11T00:00:00Z",
        )
        return TushareQueryResult(
            frame=frame,
            artifact=artifact,
            cache_hits=1 if path.exists() else 0,
            network_requests=0,
        )

    @staticmethod
    def _response(
        api_name: str, params: dict[str, object], fields: tuple[str, ...]
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]]
        if api_name == "stock_basic":
            rows = (
                [
                    {
                        "ts_code": "600001.SH",
                        "symbol": "600001",
                        "name": "示例一",
                        "market": "主板",
                        "exchange": "SSE",
                        "curr_type": "CNY",
                        "list_status": "L",
                        "list_date": "20100101",
                        "delist_date": None,
                    },
                    {
                        "ts_code": "000001.SZ",
                        "symbol": "000001",
                        "name": "示例二",
                        "market": "主板",
                        "exchange": "SZSE",
                        "curr_type": "CNY",
                        "list_status": "L",
                        "list_date": "20100101",
                        "delist_date": None,
                    },
                ]
                if params["list_status"] == "L"
                else []
            )
        elif api_name == "trade_cal":
            rows = [
                {
                    "exchange": "SSE",
                    "cal_date": "20210104",
                    "is_open": "1",
                    "pretrade_date": "20201231",
                }
            ]
        elif api_name == "index_member_all":
            rows = [
                {
                    "index_code": "000300.SH",
                    "con_code": "600001.SH",
                    "in_date": "20200101",
                    "out_date": None,
                    "ann_date": "20191231",
                    "weight": 0.5,
                },
                {
                    "index_code": "000300.SH",
                    "con_code": "000001.SZ",
                    "in_date": "20200101",
                    "out_date": None,
                    "ann_date": "20191231",
                    "weight": 0.5,
                },
            ]
        elif api_name == "daily":
            assert "ts_code" in params
            rows = [
                {
                    "ts_code": params["ts_code"],
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
        elif api_name == "adj_factor":
            rows = [
                {
                    "ts_code": params["ts_code"],
                    "trade_date": "20210104",
                    "adj_factor": 1.2,
                }
            ]
        elif api_name == "suspend_d":
            rows = []
        elif api_name == "namechange":
            rows = [
                {
                    "ts_code": params["ts_code"],
                    "name": "示例名称",
                    "start_date": "20200101",
                    "end_date": None,
                    "ann_date": "20191231",
                    "change_reason": "初始名称",
                }
            ]
        else:
            raise AssertionError(f"unexpected endpoint: {api_name}")
        return pd.DataFrame(rows, columns=list(fields))


class ConcurrentFixtureProvider(FixtureProvider):
    def __init__(self, data_root: Path) -> None:
        super().__init__(data_root)
        self.symbol_thread_ids: set[int] = set()
        self.thread_lock = threading.Lock()

    def query(
        self,
        api_name: str,
        params: dict[str, object],
        fields: tuple[str, ...],
    ) -> TushareQueryResult:
        if api_name in {"daily", "adj_factor", "suspend_d", "namechange"}:
            with self.thread_lock:
                self.symbol_thread_ids.add(threading.get_ident())
            time.sleep(0.02)
        return super().query(api_name, params, fields)


def test_pipeline_queries_only_historical_member_union(tmp_path: Path) -> None:
    provider = FixtureProvider(tmp_path)

    first = run_research_data_pipeline(ROOT / "config", tmp_path, provider=provider)
    second = run_research_data_pipeline(ROOT / "config", tmp_path, provider=provider)

    daily_calls = [params for name, params in provider.calls if name == "daily"]
    assert {params["ts_code"] for params in daily_calls} == {
        "600001.SH",
        "000001.SZ",
    }
    assert all("ts_code" in params for params in daily_calls)
    assert first.snapshot.snapshot_id == second.snapshot.snapshot_id
    assert first.historical_symbol_count == 2
    assert first.membership_method == "index_member_all"
    assert first.snapshot.quality_status == "pass"


def test_pipeline_queries_complete_name_history_without_date_bounds(
    tmp_path: Path,
) -> None:
    provider = FixtureProvider(tmp_path)

    run_research_data_pipeline(ROOT / "config", tmp_path, provider=provider)

    name_calls = [params for name, params in provider.calls if name == "namechange"]
    assert {params["ts_code"] for params in name_calls} == {
        "600001.SH",
        "000001.SZ",
    }
    assert all(set(params) == {"ts_code"} for params in name_calls)


def test_pipeline_preserves_nullable_st_but_complete_suspension_false(
    tmp_path: Path,
) -> None:
    result = run_research_data_pipeline(
        ROOT / "config", tmp_path, provider=FixtureProvider(tmp_path)
    )
    status_files = list(result.snapshot.snapshot_dir.glob("daily_status/**/*.parquet"))
    status = pd.concat([pd.read_parquet(path) for path in status_files])

    assert status["is_suspended"].tolist() == [False, False]
    assert status["is_st"].tolist() == [False, False]


def test_pipeline_does_not_publish_without_membership_capability(
    tmp_path: Path,
) -> None:
    provider = FixtureProvider(tmp_path, fail_membership=True)

    with pytest.raises(TushareProviderError, match="index_weight"):
        run_research_data_pipeline(ROOT / "config", tmp_path, provider=provider)

    assert not list((tmp_path / "manifests").glob("p5-*/manifest.json"))


def test_pipeline_loads_symbols_with_bounded_concurrency(tmp_path: Path) -> None:
    provider = ConcurrentFixtureProvider(tmp_path)

    run_research_data_pipeline(ROOT / "config", tmp_path, provider=provider)

    assert len(provider.symbol_thread_ids) == 2
