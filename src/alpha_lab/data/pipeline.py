from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import pandas as pd

from alpha_lab.data.config import DataSourceConfig, load_phase1_config
from alpha_lab.data.normalize import normalize_akshare_daily, normalize_baostock_daily
from alpha_lab.data.providers.akshare_provider import AkshareProvider
from alpha_lab.data.providers.baostock_provider import BaostockProvider
from alpha_lab.data.providers.base import RawArtifact, RawRange
from alpha_lab.data.snapshot import RawInput, SnapshotResult, materialize_snapshot


class RangeProvider(Protocol):
    def load_range(self, symbol: str, source: DataSourceConfig) -> RawRange: ...

    def has_cached_coverage(self, symbol: str, source: DataSourceConfig) -> bool: ...


@dataclass(frozen=True)
class IngestionResult:
    snapshot: SnapshotResult
    network_requests: int
    cache_hits: int
    raw_artifact_count: int
    selected_provider: str
    fallback_reason: str | None


def run_ingestion(
    config_dir: Path,
    data_root: Path,
    *,
    end_date: date | None = None,
    provider: RangeProvider | None = None,
    fallback_provider: RangeProvider | None = None,
) -> IngestionResult:
    config = load_phase1_config(config_dir)
    source = config.source
    if end_date is not None:
        source = source.model_copy(update={"end_date": end_date})
        source = DataSourceConfig.model_validate(source.model_dump())

    active_provider: RangeProvider = provider or AkshareProvider(data_root)
    active_fallback = fallback_provider
    if (
        provider is None
        and active_fallback is None
        and source.fallback_provider == "baostock"
    ):
        active_fallback = BaostockProvider(data_root)

    symbols = [item.code for item in config.universe.symbols]
    selected_provider = "akshare"
    fallback_reason: str | None = None
    observed_ranges: list[RawRange] = []
    if active_fallback is not None and all(
        active_fallback.has_cached_coverage(symbol, source) for symbol in symbols
    ):
        selected_provider = "baostock"
        fallback_reason = "complete fallback cache already exists"
        selected_ranges = [
            active_fallback.load_range(symbol, source) for symbol in symbols
        ]
    else:
        try:
            selected_ranges = []
            for symbol in symbols:
                raw_range = active_provider.load_range(symbol, source)
                observed_ranges.append(raw_range)
                selected_ranges.append(raw_range)
            observed_ranges = []
        except RuntimeError as primary_error:
            if active_fallback is None:
                raise
            selected_provider = "baostock"
            fallback_reason = str(primary_error)
            selected_ranges = [
                active_fallback.load_range(symbol, source) for symbol in symbols
            ]

    normalized: list[pd.DataFrame] = []
    raw_inputs: list[RawInput] = []
    network_requests = sum(item.network_requests for item in observed_ranges)
    network_requests += sum(item.network_requests for item in selected_ranges)
    cache_hits = sum(item.cache_hits for item in observed_ranges)
    cache_hits += sum(item.cache_hits for item in selected_ranges)

    for universe_symbol, raw_range in zip(
        config.universe.symbols, selected_ranges, strict=True
    ):
        for artifact in raw_range.artifacts:
            part = _read_artifact_slice(artifact, source)
            raw_inputs.append(
                RawInput(
                    provider=artifact.provider,
                    endpoint=artifact.endpoint,
                    symbol=artifact.symbol,
                    path=artifact.parquet_path,
                    sha256=artifact.sha256,
                    row_count=artifact.row_count,
                    requested_start=artifact.requested_start.isoformat(),
                    requested_end=artifact.requested_end.isoformat(),
                )
            )
            if part.empty:
                continue
            normalizer = (
                normalize_akshare_daily
                if artifact.provider == "akshare"
                else normalize_baostock_daily
            )
            normalized.append(
                normalizer(
                    part,
                    symbol=universe_symbol.code,
                    ingested_at=artifact.ingested_at,
                )
            )

    if not normalized:
        raise RuntimeError("data providers returned no rows for the configured sample")
    frame = pd.concat(normalized, ignore_index=True)
    snapshot = materialize_snapshot(
        data_root,
        frame,
        source=source,
        universe=config.universe,
        raw_inputs=raw_inputs,
    )
    return IngestionResult(
        snapshot=snapshot,
        network_requests=network_requests,
        cache_hits=cache_hits,
        raw_artifact_count=len(raw_inputs),
        selected_provider=selected_provider,
        fallback_reason=fallback_reason,
    )


def _read_artifact_slice(
    artifact: RawArtifact, source: DataSourceConfig
) -> pd.DataFrame:
    frame = pd.read_parquet(artifact.parquet_path)
    if frame.empty:
        return frame
    date_field = "日期" if artifact.provider == "akshare" else "date"
    if date_field not in frame.columns:
        raise ValueError(
            f"raw artifact is missing {date_field}: {artifact.parquet_path}"
        )
    dates = pd.to_datetime(frame[date_field], errors="raise")
    start = max(source.start_date, artifact.requested_start)
    end = min(source.end_date, artifact.requested_end)
    return frame.loc[(dates.dt.date >= start) & (dates.dt.date <= end)].copy()
