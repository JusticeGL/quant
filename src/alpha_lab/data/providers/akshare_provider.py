from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import time
import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import akshare as ak
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


class AkshareProvider:
    def __init__(
        self,
        data_root: Path,
        *,
        fetcher: Fetcher | None = None,
        sleep: Sleeper = time.sleep,
    ) -> None:
        self.data_root = data_root
        self.fetcher = fetcher or ak.stock_zh_a_hist
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
        missing = missing_intervals(
            source.start_date,
            source.end_date,
            [(item.requested_start, item.requested_end) for item in cached],
        )

        created = [
            self._fetch_artifact(symbol, source, interval_start, interval_end)
            for interval_start, interval_end in missing
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
        frame = self._combine_frames(artifacts, source.start_date, source.end_date)
        return RawRange(
            frame=frame,
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
        return self.data_root / "raw" / "akshare" / "stock_zh_a_hist" / symbol

    def _matching_artifacts(
        self, symbol: str, source: DataSourceConfig
    ) -> tuple[RawArtifact, ...]:
        raw_dir = self._raw_dir(symbol)
        if not raw_dir.exists():
            return ()

        artifacts: list[RawArtifact] = []
        for metadata_path in sorted(raw_dir.glob("*.json")):
            with metadata_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
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
                    provider="akshare",
                    endpoint="stock_zh_a_hist",
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
            "provider": source.provider,
            "endpoint": source.endpoint,
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
                    period=source.period,
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    adjust=source.adjust,
                    timeout=source.request_timeout_seconds,
                )
                break
            except Exception as caught:  # noqa: BLE001 - provider errors are unstable
                error = caught
                if attempt == source.max_attempts:
                    raise RuntimeError(
                        f"AKShare failed after {source.max_attempts} attempts for "
                        f"{symbol}: {type(caught).__name__}: {caught}"
                    ) from caught
                self.sleep(source.retry_delay_seconds * attempt)
        else:  # pragma: no cover - loop either succeeds or raises
            raise RuntimeError("unreachable AKShare retry state") from error

        if not isinstance(frame, pd.DataFrame):
            raise TypeError("AKShare endpoint must return a pandas DataFrame")

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
                "akshare_version": importlib.metadata.version("akshare"),
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
            provider="akshare",
            endpoint="stock_zh_a_hist",
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
    def _combine_frames(
        artifacts: tuple[RawArtifact, ...], start: date, end: date
    ) -> pd.DataFrame:
        if not artifacts:
            return pd.DataFrame()
        frames = [pd.read_parquet(item.parquet_path) for item in artifacts]
        frame = pd.concat(frames, ignore_index=True)
        if frame.empty:
            return frame
        if "日期" not in frame.columns:
            raise ValueError("AKShare response is missing 日期")
        dates = pd.to_datetime(frame["日期"], errors="raise")
        frame = frame.loc[(dates.dt.date >= start) & (dates.dt.date <= end)].copy()
        frame.loc[:, "日期"] = pd.to_datetime(frame["日期"]).dt.strftime("%Y-%m-%d")
        return frame.drop_duplicates().sort_values("日期").reset_index(drop=True)
