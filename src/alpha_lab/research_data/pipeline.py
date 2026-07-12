from __future__ import annotations

import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd

from alpha_lab.research_data.config import (
    ResearchDataConfig,
    load_research_data_config,
)
from alpha_lab.research_data.contracts import ResearchTables
from alpha_lab.research_data.normalize import (
    normalize_adjustment_factors,
    normalize_daily_bars,
    normalize_index_membership_intervals,
    normalize_name_history,
    normalize_security_master,
    normalize_suspensions,
    normalize_trading_calendar,
    reconstruct_weight_membership,
)
from alpha_lab.research_data.provider import (
    TushareArtifact,
    TushareProvider,
    TushareProviderError,
    TushareQueryResult,
)
from alpha_lab.research_data.snapshot import (
    ResearchSnapshotResult,
    materialize_research_snapshot,
)

STOCK_BASIC_FIELDS = (
    "ts_code",
    "symbol",
    "name",
    "market",
    "exchange",
    "list_status",
    "list_date",
)
CALENDAR_FIELDS = ("exchange", "cal_date", "is_open", "pretrade_date")
MEMBERSHIP_FIELDS = (
    "index_code",
    "con_code",
    "in_date",
    "out_date",
    "ann_date",
    "weight",
)
WEIGHT_FIELDS = ("index_code", "con_code", "trade_date", "weight")
DAILY_FIELDS = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "vol",
    "amount",
)
ADJUSTMENT_FIELDS = ("ts_code", "trade_date", "adj_factor")
SUSPENSION_FIELDS = (
    "ts_code",
    "trade_date",
    "suspend_timing",
    "suspend_type",
)
NAME_FIELDS = (
    "ts_code",
    "name",
    "start_date",
    "end_date",
    "ann_date",
    "change_reason",
)


class QueryProvider(Protocol):
    def query(
        self,
        api_name: str,
        params: Mapping[str, object],
        fields: tuple[str, ...],
    ) -> TushareQueryResult: ...


@dataclass(frozen=True)
class CapabilityReport:
    provider: str
    index_code: str
    capabilities: dict[str, dict[str, object]]
    membership_method: str | None


@dataclass(frozen=True)
class ResearchIngestionResult:
    snapshot: ResearchSnapshotResult
    network_requests: int
    cache_hits: int
    raw_artifact_count: int
    historical_symbol_count: int
    membership_method: str


def probe_research_data(
    config_dir: Path,
    data_root: Path,
    *,
    provider: QueryProvider | None = None,
) -> CapabilityReport:
    config = load_research_data_config(config_dir)
    active = provider or _provider_from_environment(data_root, config)
    capabilities: dict[str, dict[str, object]] = {}
    listed = _probe_query(
        active,
        config.endpoints.security_master,
        {"list_status": "L"},
        STOCK_BASIC_FIELDS,
        capabilities,
    )
    _probe_query(
        active,
        config.endpoints.trading_calendar,
        {
            "exchange": config.calendar_exchange,
            "start_date": config.start_date.strftime("%Y%m%d"),
            "end_date": config.start_date.strftime("%Y%m%d"),
        },
        CALENDAR_FIELDS,
        capabilities,
    )
    membership_method: str | None = None
    try:
        _probe_query(
            active,
            config.endpoints.membership_primary,
            {"index_code": config.index_code},
            MEMBERSHIP_FIELDS,
            capabilities,
        )
        membership_method = "index_member_all"
    except (RuntimeError, TushareProviderError):
        _probe_query(
            active,
            config.endpoints.membership_fallback,
            {
                "index_code": config.index_code,
                "start_date": config.start_date.strftime("%Y%m%d"),
                "end_date": config.start_date.strftime("%Y%m%d"),
            },
            WEIGHT_FIELDS,
            capabilities,
        )
        membership_method = "index_weight_observation"
    if not listed.frame.empty:
        ts_code = str(listed.frame.iloc[0]["ts_code"])
        for api_name, fields in (
            (config.endpoints.daily_bar, DAILY_FIELDS),
            (config.endpoints.adjustment_factor, ADJUSTMENT_FIELDS),
            (config.endpoints.suspension, SUSPENSION_FIELDS),
            (config.endpoints.name_history, NAME_FIELDS),
        ):
            _probe_query(
                active,
                api_name,
                {
                    "ts_code": ts_code,
                    "start_date": config.start_date.strftime("%Y%m%d"),
                    "end_date": config.start_date.strftime("%Y%m%d"),
                },
                fields,
                capabilities,
            )
    return CapabilityReport(
        provider="tushare",
        index_code=config.index_code,
        capabilities=capabilities,
        membership_method=membership_method,
    )


def run_research_data_pipeline(
    config_dir: Path,
    data_root: Path,
    *,
    end_date: date | None = None,
    provider: QueryProvider | None = None,
) -> ResearchIngestionResult:
    config = load_research_data_config(config_dir)
    if end_date is not None:
        config = ResearchDataConfig.model_validate(
            config.model_copy(update={"end_date": end_date}).model_dump(mode="json")
        )
    active = provider or _provider_from_environment(data_root, config)
    results: list[TushareQueryResult] = []

    security_results = [
        active.query(
            config.endpoints.security_master,
            {"list_status": status},
            STOCK_BASIC_FIELDS,
        )
        for status in config.stock_statuses
    ]
    results.extend(security_results)
    security_raw = pd.concat(
        [item.frame for item in security_results], ignore_index=True
    ).drop_duplicates("ts_code", keep="last")
    latest_ingested = max(item.artifact.ingested_at for item in security_results)
    security_master = normalize_security_master(
        security_raw, ingested_at=latest_ingested
    )

    calendar_result = active.query(
        config.endpoints.trading_calendar,
        {
            "exchange": config.calendar_exchange,
            "start_date": config.start_date.strftime("%Y%m%d"),
            "end_date": config.end_date.strftime("%Y%m%d"),
        },
        CALENDAR_FIELDS,
    )
    results.append(calendar_result)
    trading_calendar = normalize_trading_calendar(calendar_result.frame)

    membership, membership_results, membership_method = _load_membership(active, config)
    results.extend(membership_results)
    member_codes = sorted(
        {
            _security_id_to_ts_code(value)
            for value in membership["security_id"].astype(str)
        }
    )
    if not member_codes:
        raise ValueError("historical CSI 300 membership is empty")
    if len(member_codes) > config.maximum_symbols:
        raise ValueError(
            f"historical member union exceeds maximum_symbols: {len(member_codes)}"
        )
    master_codes = set(security_master["ts_code"].astype(str))
    missing_master = sorted(set(member_codes) - master_codes)
    if missing_master:
        raise ValueError(
            f"membership securities missing from stock_basic: {missing_master}"
        )

    daily_parts: list[pd.DataFrame] = []
    adjustment_parts: list[pd.DataFrame] = []
    suspension_parts: list[pd.DataFrame] = []
    name_parts: list[pd.DataFrame] = []
    date_params = {
        "start_date": config.start_date.strftime("%Y%m%d"),
        "end_date": config.end_date.strftime("%Y%m%d"),
    }
    worker_count = min(config.source.maximum_concurrency, len(member_codes))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        symbol_results = executor.map(
            lambda ts_code: _query_security_history(
                active, config, ts_code, date_params
            ),
            member_codes,
        )
        for selected in symbol_results:
            daily_result, adjustment_result, suspension_result, name_result = selected
            results.extend(selected)
            daily_parts.append(normalize_daily_bars(daily_result.frame))
            adjustment_parts.append(
                normalize_adjustment_factors(adjustment_result.frame)
            )
            suspension_parts.append(normalize_suspensions(suspension_result.frame))
            name_parts.append(normalize_name_history(name_result.frame))

    daily_bar = pd.concat(daily_parts, ignore_index=True)
    adjustment_factor = pd.concat(adjustment_parts, ignore_index=True)
    suspension = pd.concat(suspension_parts, ignore_index=True)
    name_history = pd.concat(name_parts, ignore_index=True)
    daily_status = _build_daily_status(daily_bar, name_history, suspension)
    tables = ResearchTables(
        security_master=security_master,
        security_name_history=name_history,
        trading_calendar=trading_calendar,
        index_membership=membership,
        daily_bar=daily_bar,
        adjustment_factor=adjustment_factor,
        suspension=suspension,
        daily_status=daily_status,
    )
    artifacts = _unique_artifacts(results)
    snapshot = materialize_research_snapshot(data_root, config, tables, artifacts)
    return ResearchIngestionResult(
        snapshot=snapshot,
        network_requests=sum(item.network_requests for item in results),
        cache_hits=sum(item.cache_hits for item in results),
        raw_artifact_count=len(artifacts),
        historical_symbol_count=len(member_codes),
        membership_method=membership_method,
    )


def _query_security_history(
    provider: QueryProvider,
    config: ResearchDataConfig,
    ts_code: str,
    date_params: dict[str, str],
) -> tuple[
    TushareQueryResult,
    TushareQueryResult,
    TushareQueryResult,
    TushareQueryResult,
]:
    params = {"ts_code": ts_code, **date_params}
    return (
        provider.query(config.endpoints.daily_bar, params, DAILY_FIELDS),
        provider.query(config.endpoints.adjustment_factor, params, ADJUSTMENT_FIELDS),
        provider.query(config.endpoints.suspension, params, SUSPENSION_FIELDS),
        provider.query(config.endpoints.name_history, params, NAME_FIELDS),
    )


def _load_membership(
    provider: QueryProvider, config: ResearchDataConfig
) -> tuple[pd.DataFrame, list[TushareQueryResult], str]:
    try:
        primary = provider.query(
            config.endpoints.membership_primary,
            {"index_code": config.index_code},
            MEMBERSHIP_FIELDS,
        )
        if primary.frame.empty:
            raise TushareProviderError("index_member_all returned no rows")
        return (
            normalize_index_membership_intervals(primary.frame, config.index_code),
            [primary],
            "index_member_all",
        )
    except (RuntimeError, TushareProviderError) as primary_error:
        fallback_results: list[TushareQueryResult] = []
        for start, end in _month_intervals(config.start_date, config.end_date):
            result = provider.query(
                config.endpoints.membership_fallback,
                {
                    "index_code": config.index_code,
                    "start_date": start.strftime("%Y%m%d"),
                    "end_date": end.strftime("%Y%m%d"),
                },
                WEIGHT_FIELDS,
            )
            if len(result.frame) >= 5000:
                raise RuntimeError(
                    f"index_weight may be truncated at row limit for {start}..{end}"
                ) from primary_error
            fallback_results.append(result)
        raw = pd.concat([item.frame for item in fallback_results], ignore_index=True)
        if raw.empty:
            raise TushareProviderError(
                "index_weight returned no rows"
            ) from primary_error
        return (
            reconstruct_weight_membership(raw, config.index_code),
            fallback_results,
            "index_weight_observation",
        )


def _build_daily_status(
    daily_bar: pd.DataFrame,
    name_history: pd.DataFrame,
    suspension: pd.DataFrame,
) -> pd.DataFrame:
    status = daily_bar.loc[:, ["trade_date", "security_id"]].copy()
    status["is_suspended"] = pd.Series(False, index=status.index, dtype="boolean")
    status["is_st"] = pd.Series(pd.NA, index=status.index, dtype="boolean")
    trade_known = pd.to_datetime(status["trade_date"]).dt.tz_localize("UTC")
    for row in suspension.itertuples(index=False):
        end = (
            pd.Timestamp.max.normalize()
            if pd.isna(row.effective_to)
            else pd.Timestamp(row.effective_to)
        )
        mask = (
            (status["security_id"] == row.security_id)
            & (status["trade_date"] >= row.effective_from)
            & (status["trade_date"] <= end)
            & (trade_known >= row.known_at)
        )
        status.loc[mask, "is_suspended"] = True
    for row in name_history.itertuples(index=False):
        end = (
            pd.Timestamp.max.normalize()
            if pd.isna(row.effective_to)
            else pd.Timestamp(row.effective_to)
        )
        mask = (
            (status["security_id"] == row.security_id)
            & (status["trade_date"] >= row.effective_from)
            & (status["trade_date"] <= end)
            & (trade_known >= row.known_at)
        )
        status.loc[mask, "is_st"] = bool(row.is_st)
    status["known_at"] = trade_known
    status["source"] = "tushare.suspend_d+tushare.namechange"
    return status.sort_values(["trade_date", "security_id"], kind="stable").reset_index(
        drop=True
    )


def _probe_query(
    provider: QueryProvider,
    api_name: str,
    params: dict[str, object],
    fields: tuple[str, ...],
    capabilities: dict[str, dict[str, object]],
) -> TushareQueryResult:
    try:
        result = provider.query(api_name, params, fields)
    except Exception as error:
        capabilities[api_name] = {
            "available": False,
            "error_type": type(error).__name__,
            "message": str(error),
        }
        raise
    capabilities[api_name] = {
        "available": True,
        "row_count": len(result.frame),
        "field_count": len(result.frame.columns),
    }
    return result


def _provider_from_environment(
    data_root: Path, config: ResearchDataConfig
) -> TushareProvider:
    return TushareProvider(
        data_root,
        config.source,
        token=os.environ.get("TUSHARE_TOKEN", ""),
        http_url=os.environ.get("TUSHARE_HTTP_URL", "https://api.tushare.pro"),
    )


def _month_intervals(start: date, end: date) -> list[tuple[date, date]]:
    intervals: list[tuple[date, date]] = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        next_month = (
            date(cursor.year + 1, 1, 1)
            if cursor.month == 12
            else date(cursor.year, cursor.month + 1, 1)
        )
        intervals.append((max(start, cursor), min(end, next_month - timedelta(days=1))))
        cursor = next_month
    return intervals


def _security_id_to_ts_code(security_id: str) -> str:
    _, exchange, symbol = security_id.split(":")
    suffix = {"SSE": "SH", "SZSE": "SZ", "BSE": "BJ"}.get(exchange)
    if suffix is None:
        raise ValueError(f"unsupported security exchange: {exchange}")
    return f"{symbol}.{suffix}"


def _unique_artifacts(results: list[TushareQueryResult]) -> list[TushareArtifact]:
    values = {item.artifact.request_sha256: item.artifact for item in results}
    return [values[key] for key in sorted(values)]
