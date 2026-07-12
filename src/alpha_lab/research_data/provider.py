from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

from alpha_lab.data.providers.base import canonical_json, file_sha256
from alpha_lab.research_data.config import TushareSourceConfig

Transport = Callable[[str, dict[str, Any], float], dict[str, Any]]
Sleeper = Callable[[float], None]


class TushareProviderError(RuntimeError):
    """A valid Tushare response could not satisfy the requested contract."""


@dataclass(frozen=True)
class TushareArtifact:
    api_name: str
    request_sha256: str
    parquet_path: Path
    metadata_path: Path
    sha256: str
    row_count: int
    params: dict[str, object]
    fields: tuple[str, ...]
    ingested_at: str


@dataclass(frozen=True)
class TushareQueryResult:
    frame: pd.DataFrame
    artifact: TushareArtifact
    cache_hits: int
    network_requests: int


def _requests_transport(
    url: str, payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    document = response.json()
    if not isinstance(document, dict):
        raise TypeError("Tushare response must be a JSON object")
    return document


class TushareProvider:
    def __init__(
        self,
        data_root: Path,
        source: TushareSourceConfig,
        *,
        token: str,
        http_url: str,
        transport: Transport = _requests_transport,
        sleep: Sleeper = time.sleep,
    ) -> None:
        if not token.strip():
            raise ValueError("TUSHARE_TOKEN is required for live provider calls")
        parsed = urlparse(http_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("TUSHARE_HTTP_URL must be an absolute HTTPS URL")
        self.data_root = data_root
        self.source = source
        self.token = token
        self.http_url = http_url.rstrip("/")
        self.http_host = parsed.hostname
        self.transport = transport
        self.sleep = sleep

    def query(
        self,
        api_name: str,
        params: Mapping[str, object],
        fields: tuple[str, ...],
    ) -> TushareQueryResult:
        if not api_name or not fields or len(fields) != len(set(fields)):
            raise ValueError("api_name and unique requested fields are required")
        normalized_params = _json_mapping(params)
        request_document = {
            "schema_version": 1,
            "provider": "tushare",
            "http_host": self.http_host,
            "api_name": api_name,
            "params": normalized_params,
            "fields": list(fields),
        }
        request_hash = hashlib.sha256(
            canonical_json(request_document).encode("utf-8")
        ).hexdigest()
        raw_dir = self.data_root / "raw" / "tushare" / api_name
        parquet_path = raw_dir / f"{request_hash}.parquet"
        metadata_path = raw_dir / f"{request_hash}.json"
        if parquet_path.exists() or metadata_path.exists():
            artifact, frame = self._load_cached(
                api_name,
                normalized_params,
                fields,
                request_hash,
                parquet_path,
                metadata_path,
            )
            return TushareQueryResult(
                frame=frame,
                artifact=artifact,
                cache_hits=1,
                network_requests=0,
            )

        payload: dict[str, Any] = {
            "api_name": api_name,
            "token": self.token,
            "params": normalized_params,
            "fields": ",".join(fields),
        }
        document = self._request_with_retries(api_name, payload)
        frame = self._response_frame(api_name, document, fields)
        artifact = self._write_artifact(
            request_document,
            request_hash,
            frame,
            parquet_path,
            metadata_path,
        )
        self.sleep(self.source.request_interval_seconds)
        return TushareQueryResult(
            frame=frame,
            artifact=artifact,
            cache_hits=0,
            network_requests=1,
        )

    def _request_with_retries(
        self, api_name: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.source.max_attempts + 1):
            try:
                return self.transport(
                    self.http_url,
                    payload,
                    self.source.request_timeout_seconds,
                )
            except Exception as error:  # noqa: BLE001 - transport errors vary
                last_error = error
                if attempt == self.source.max_attempts:
                    message = self._redact(str(error))
                    raise RuntimeError(
                        f"Tushare transport failed for {api_name} after "
                        f"{attempt} attempts: {type(error).__name__}: {message}"
                    ) from error
                self.sleep(self.source.retry_delay_seconds * attempt)
        raise RuntimeError("unreachable Tushare retry state") from last_error

    def _response_frame(
        self,
        api_name: str,
        document: dict[str, Any],
        requested_fields: tuple[str, ...],
    ) -> pd.DataFrame:
        code = document.get("code")
        if code != 0:
            message = self._redact(str(document.get("msg") or "provider error"))
            raise TushareProviderError(
                f"Tushare {api_name} failed: code={code}, message={message}"
            )
        data = document.get("data")
        if not isinstance(data, dict):
            raise TushareProviderError(f"Tushare {api_name} response has no data")
        returned_fields = data.get("fields")
        items = data.get("items")
        if (
            not isinstance(returned_fields, list)
            or not all(isinstance(field, str) for field in returned_fields)
            or len(returned_fields) != len(set(returned_fields))
            or not set(requested_fields).issubset(returned_fields)
        ):
            raise TushareProviderError(
                f"Tushare {api_name} returned fields {returned_fields!r}; "
                f"required {list(requested_fields)!r}"
            )
        if not isinstance(items, list):
            raise TushareProviderError(f"Tushare {api_name} items must be a list")
        try:
            frame = pd.DataFrame(items, columns=returned_fields)
            return frame.loc[:, list(requested_fields)]
        except (TypeError, ValueError) as error:
            raise TushareProviderError(
                f"Tushare {api_name} row shape does not match returned fields"
            ) from error

    def _write_artifact(
        self,
        request_document: dict[str, object],
        request_hash: str,
        frame: pd.DataFrame,
        parquet_path: Path,
        metadata_path: Path,
    ) -> TushareArtifact:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        nonce = uuid.uuid4().hex
        temporary_parquet = parquet_path.parent / f".{request_hash}.{nonce}.parquet"
        temporary_metadata = parquet_path.parent / f".{request_hash}.{nonce}.json"
        try:
            frame.to_parquet(
                temporary_parquet,
                index=False,
                engine="pyarrow",
                compression="zstd",
            )
            sha256 = file_sha256(temporary_parquet)
            ingested_at = datetime.now(UTC).isoformat()
            metadata = {
                **request_document,
                "request_sha256": request_hash,
                "tushare_version": importlib.metadata.version("tushare"),
                "ingested_at": ingested_at,
                "row_count": len(frame),
                "sha256": sha256,
            }
            temporary_metadata.write_text(
                f"{canonical_json(metadata)}\n", encoding="utf-8"
            )
            os.replace(temporary_parquet, parquet_path)
            os.replace(temporary_metadata, metadata_path)
        finally:
            temporary_parquet.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)
        return self._artifact_from_metadata(metadata_path, parquet_path)

    def _load_cached(
        self,
        api_name: str,
        params: dict[str, object],
        fields: tuple[str, ...],
        request_hash: str,
        parquet_path: Path,
        metadata_path: Path,
    ) -> tuple[TushareArtifact, pd.DataFrame]:
        if not parquet_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"raw cache is incomplete for request {request_hash}")
        artifact = self._artifact_from_metadata(metadata_path, parquet_path)
        if (
            artifact.api_name != api_name
            or artifact.params != params
            or artifact.fields != fields
            or artifact.request_sha256 != request_hash
        ):
            raise RuntimeError(f"raw cache identity mismatch: {metadata_path}")
        actual_sha = file_sha256(parquet_path)
        if actual_sha != artifact.sha256:
            raise RuntimeError(f"raw cache checksum mismatch: {parquet_path}")
        frame = pd.read_parquet(parquet_path)
        if len(frame) != artifact.row_count or tuple(frame.columns) != fields:
            raise RuntimeError(f"raw cache schema mismatch: {parquet_path}")
        return artifact, frame

    @staticmethod
    def _artifact_from_metadata(
        metadata_path: Path, parquet_path: Path
    ) -> TushareArtifact:
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
        return TushareArtifact(
            api_name=str(document["api_name"]),
            request_sha256=str(document["request_sha256"]),
            parquet_path=parquet_path,
            metadata_path=metadata_path,
            sha256=str(document["sha256"]),
            row_count=int(document["row_count"]),
            params=_json_mapping(document["params"]),
            fields=tuple(str(item) for item in document["fields"]),
            ingested_at=str(document["ingested_at"]),
        )

    def _redact(self, message: str) -> str:
        return message.replace(self.token, "***")


def _json_mapping(value: Mapping[str, object] | object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("Tushare params must be a mapping")
    encoded = json.loads(canonical_json(dict(value)))
    if not isinstance(encoded, dict):
        raise TypeError("Tushare params must serialize to an object")
    return {str(key): item for key, item in encoded.items()}
