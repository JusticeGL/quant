from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import time
import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import baostock as bs
import pandas as pd

from alpha_lab.data.config import DataSourceConfig
from alpha_lab.data.providers.base import (
    RawArtifact,
    RawRange,
    canonical_json,
    file_sha256,
    intervals_overlap,
    missing_intervals,
)

Fetcher = Callable[..., pd.DataFrame]
Sleeper = Callable[[float], None]

BAOSTOCK_FIELDS = (
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "adjustflag",
    "tradestatus",
    "isST",
)


class BaostockProvider:
    def __init__(
        self,
        data_root: Path,
        *,
        fetcher: Fetcher | None = None,
        sleep: Sleeper = time.sleep,
    ) -> None:
        self.data_root = data_root
        self.fetcher = fetcher or self._query_history
        self.sleep = sleep

    def load_range(self, symbol: str, source: DataSourceConfig) -> RawRange:
        existing = self._matching_artifacts(symbol, source)
        cached = tuple(
            artifact
            for artifact in existing
            if intervals_overlap(
                artifact.requested_start,
                artifact.requested_end,
                source.start_date,
                source.end_date,
            )
        )
        uncovered = missing_intervals(
            source.start_date,
            source.end_date,
            [(item.requested_start, item.requested_end) for item in cached],
        )
        created = [
            self._fetch_artifact(symbol, source, interval_start, interval_end)
            for interval_start, interval_end in uncovered
        ]
        artifacts = tuple(
            sorted(
                (*cached, *created),
                key=lambda item: (
                    item.requested_start,
                    item.requested_end,
                    item.sha256,
                ),
            )
        )
        return RawRange(
            frame=self._combine_frames(artifacts, source.start_date, source.end_date),
            artifacts=artifacts,
            cache_hits=len(cached),
            network_requests=len(created),
        )

    def has_cached_coverage(self, symbol: str, source: DataSourceConfig) -> bool:
        artifacts = self._matching_artifacts(symbol, source)
        uncovered = missing_intervals(
            source.start_date,
            source.end_date,
            [(item.requested_start, item.requested_end) for item in artifacts],
        )
        return not uncovered

    def _raw_dir(self, symbol: str) -> Path:
        return (
            self.data_root / "raw" / "baostock" / "query_history_k_data_plus" / symbol
        )

    def _matching_artifacts(
        self, symbol: str, source: DataSourceConfig
    ) -> tuple[RawArtifact, ...]:
        raw_dir = self._raw_dir(symbol)
        if not raw_dir.exists():
            return ()
        artifacts: list[RawArtifact] = []
        for metadata_path in sorted(raw_dir.glob("*.json")):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            params = metadata.get("params", {})
            if not self._base_params_match(params, symbol, source):
                continue
            parquet_path = metadata_path.with_suffix(".parquet")
            if not parquet_path.is_file():
                raise RuntimeError(f"raw cache is incomplete: {parquet_path}")
            actual_sha = file_sha256(parquet_path)
            if actual_sha != metadata.get("sha256"):
                raise RuntimeError(f"raw cache checksum mismatch: {parquet_path}")
            artifacts.append(
                RawArtifact(
                    provider="baostock",
                    endpoint="query_history_k_data_plus",
                    symbol=symbol,
                    parquet_path=parquet_path,
                    metadata_path=metadata_path,
                    sha256=actual_sha,
                    row_count=int(metadata["row_count"]),
                    requested_start=date.fromisoformat(params["start_date"]),
                    requested_end=date.fromisoformat(params["end_date"]),
                    ingested_at=str(metadata["ingested_at"]),
                )
            )
        return tuple(artifacts)

    @staticmethod
    def _base_params_match(
        params: dict[str, Any], symbol: str, source: DataSourceConfig
    ) -> bool:
        return (
            params.get("symbol") == symbol
            and params.get("period") == source.period
            and params.get("adjust") == source.adjust
        )

    def _fetch_artifact(
        self,
        symbol: str,
        source: DataSourceConfig,
        start: date,
        end: date,
    ) -> RawArtifact:
        params = {
            "symbol": symbol,
            "period": source.period,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "adjust": source.adjust,
        }
        request_document = {
            "provider": "baostock",
            "endpoint": "query_history_k_data_plus",
            "params": params,
        }
        request_hash = hashlib.sha256(
            canonical_json(request_document).encode("utf-8")
        ).hexdigest()
        raw_dir = self._raw_dir(symbol)
        parquet_path = raw_dir / f"{request_hash}.parquet"
        metadata_path = raw_dir / f"{request_hash}.json"

        if parquet_path.exists() or metadata_path.exists():
            if not (parquet_path.is_file() and metadata_path.is_file()):
                raise RuntimeError(
                    f"raw cache is incomplete for request {request_hash}"
                )
            matches = self._matching_artifacts(symbol, source)
            return next(item for item in matches if item.parquet_path == parquet_path)

        error: Exception | None = None
        for attempt in range(1, source.max_attempts + 1):
            try:
                frame = self.fetcher(
                    symbol=symbol,
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                    adjust=source.adjust,
                )
                break
            except Exception as caught:  # noqa: BLE001 - provider errors are unstable
                error = caught
                if attempt == source.max_attempts:
                    raise RuntimeError(
                        f"Baostock failed after {source.max_attempts} attempts for "
                        f"{symbol}: {type(caught).__name__}: {caught}"
                    ) from caught
                self.sleep(source.retry_delay_seconds * attempt)
        else:  # pragma: no cover
            raise RuntimeError("unreachable Baostock retry state") from error

        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Baostock endpoint must return a pandas DataFrame")
        raw_dir.mkdir(parents=True, exist_ok=True)
        nonce = uuid.uuid4().hex
        temp_parquet = raw_dir / f".{request_hash}.{nonce}.tmp.parquet"
        temp_metadata = raw_dir / f".{request_hash}.{nonce}.tmp.json"
        try:
            frame.to_parquet(
                temp_parquet,
                index=False,
                engine="pyarrow",
                compression="zstd",
            )
            sha256 = file_sha256(temp_parquet)
            ingested_at = datetime.now(UTC).isoformat()
            metadata = {
                "schema_version": 1,
                **request_document,
                "baostock_version": importlib.metadata.version("baostock"),
                "ingested_at": ingested_at,
                "row_count": len(frame),
                "sha256": sha256,
            }
            temp_metadata.write_text(f"{canonical_json(metadata)}\n", encoding="utf-8")
            os.replace(temp_parquet, parquet_path)
            os.replace(temp_metadata, metadata_path)
        finally:
            temp_parquet.unlink(missing_ok=True)
            temp_metadata.unlink(missing_ok=True)
        self.sleep(source.request_interval_seconds)
        return RawArtifact(
            provider="baostock",
            endpoint="query_history_k_data_plus",
            symbol=symbol,
            parquet_path=parquet_path,
            metadata_path=metadata_path,
            sha256=sha256,
            row_count=len(frame),
            requested_start=start,
            requested_end=end,
            ingested_at=ingested_at,
        )

    @staticmethod
    def _query_history(
        *, symbol: str, start_date: str, end_date: str, adjust: str
    ) -> pd.DataFrame:
        market_symbol = _to_baostock_symbol(symbol)
        adjust_flag = {"": "3", "qfq": "2", "hfq": "1"}[adjust]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            login_result = bs.login()
        if login_result.error_code != "0":
            raise ConnectionError(f"Baostock login failed: {login_result.error_msg}")
        try:
            result = bs.query_history_k_data_plus(
                market_symbol,
                ",".join(BAOSTOCK_FIELDS),
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag=adjust_flag,
            )
            if result.error_code != "0":
                raise ConnectionError(f"Baostock query failed: {result.error_msg}")
            rows: list[list[str]] = []
            while result.next():
                rows.append(result.get_row_data())
            return pd.DataFrame(rows, columns=list(BAOSTOCK_FIELDS))
        finally:
            with contextlib.redirect_stdout(output):
                bs.logout()

    @staticmethod
    def _combine_frames(
        artifacts: tuple[RawArtifact, ...], start: date, end: date
    ) -> pd.DataFrame:
        if not artifacts:
            return pd.DataFrame()
        frame = pd.concat(
            [pd.read_parquet(item.parquet_path) for item in artifacts],
            ignore_index=True,
        )
        if frame.empty:
            return frame
        if "date" not in frame.columns:
            raise ValueError("Baostock response is missing date")
        dates = pd.to_datetime(frame["date"], errors="raise")
        frame = frame.loc[(dates.dt.date >= start) & (dates.dt.date <= end)].copy()
        return frame.drop_duplicates().sort_values("date").reset_index(drop=True)


def _to_baostock_symbol(symbol: str) -> str:
    if symbol.startswith("6"):
        return f"sh.{symbol}"
    if symbol.startswith(("0", "3")):
        return f"sz.{symbol}"
    raise ValueError(f"Baostock fallback does not support symbol: {symbol}")
