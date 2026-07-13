from __future__ import annotations

import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from alpha_lab.data.providers.base import file_sha256
from alpha_lab.research_data.config import TushareSourceConfig
from alpha_lab.research_data.normalize import to_security_id
from alpha_lab.research_data.provider import (
    TushareArtifact,
    TushareProvider,
    TushareQueryResult,
)
from alpha_lab.robustness.config import RobustnessConfig, load_robustness_config
from alpha_lab.robustness.contracts import ExposureTables

DAILY_BASIC_FIELDS = ("ts_code", "trade_date", "total_mv", "circ_mv")
INDEX_CLASSIFY_FIELDS = (
    "index_code",
    "industry_name",
    "level",
    "industry_code",
    "src",
)
INDUSTRY_MEMBER_FIELDS = (
    "l1_code",
    "l1_name",
    "l2_code",
    "l2_name",
    "l3_code",
    "l3_name",
    "ts_code",
    "name",
    "in_date",
    "out_date",
    "is_new",
)
DAILY_BASIC_ROW_LIMIT = 6000
INDUSTRY_MEMBER_ROW_LIMIT = 5000


class QueryProvider(Protocol):
    def query(
        self,
        api_name: str,
        params: Mapping[str, object],
        fields: tuple[str, ...],
    ) -> TushareQueryResult: ...


def normalize_market_cap(
    raw: pd.DataFrame,
    *,
    row_limit: int = DAILY_BASIC_ROW_LIMIT,
    expected_ts_code: str | None = None,
) -> pd.DataFrame:
    _require_fields(raw, DAILY_BASIC_FIELDS, "daily_basic")
    _reject_row_limit(raw, row_limit, "daily_basic")
    if (
        expected_ts_code is not None
        and not raw["ts_code"].astype(str).eq(expected_ts_code).all()
    ):
        raise ValueError("daily_basic response contains an unexpected security")
    frame = pd.DataFrame(
        {
            "trade_date": _dates(raw["trade_date"], required=True),
            "security_id": raw["ts_code"].map(to_security_id),
            "total_market_cap_cny": pd.to_numeric(
                raw["total_mv"], errors="coerce"
            ).astype("float64")
            * 10_000.0,
            "float_market_cap_cny": pd.to_numeric(
                raw["circ_mv"], errors="coerce"
            ).astype("float64")
            * 10_000.0,
        }
    )
    numeric = frame[["total_market_cap_cny", "float_market_cap_cny"]]
    if numeric.isna().any().any() or (~np.isfinite(numeric)).any().any():
        raise ValueError("daily_basic market cap must be finite")
    if (numeric <= 0).any().any():
        raise ValueError("daily_basic market cap must be positive")
    if frame.duplicated(["trade_date", "security_id"]).any():
        raise ValueError("daily_basic returned duplicate observation keys")
    frame["known_at"] = _utc_dates(frame["trade_date"])
    frame["source"] = "tushare.daily_basic"
    return frame.sort_values(["trade_date", "security_id"], kind="stable").reset_index(
        drop=True
    )


def normalize_industry_definition(raw: pd.DataFrame) -> pd.DataFrame:
    _require_fields(raw, INDEX_CLASSIFY_FIELDS, "index_classify")
    selected = raw.loc[
        raw["src"].astype("string").eq("SW2021")
        & raw["level"].astype("string").str.upper().eq("L1")
    ].copy()
    if selected.empty:
        raise ValueError("index_classify returned no SW2021 level-one industries")
    result = pd.DataFrame(
        {
            "industry_id": "CN:SW2021:" + selected["index_code"].astype("string"),
            "source_index_code": selected["index_code"].astype("string"),
            "industry_code": selected["industry_code"].astype("string"),
            "industry_name": selected["industry_name"].astype("string"),
            "level": "L1",
            "classification_standard": "SW2021",
            "source": "tushare.index_classify",
        }
    )
    if result[["industry_id", "source_index_code", "industry_name"]].isna().any().any():
        raise ValueError("index_classify contains missing required fields")
    if result["industry_id"].duplicated().any():
        raise ValueError("index_classify returned duplicate SW2021 industry keys")
    return result.sort_values("industry_id", kind="stable").reset_index(drop=True)


def normalize_industry_membership(
    raw: pd.DataFrame,
    allowed_l1_codes: set[str],
    *,
    row_limit: int = INDUSTRY_MEMBER_ROW_LIMIT,
    expected_l1_code: str | None = None,
) -> pd.DataFrame:
    _require_fields(raw, INDUSTRY_MEMBER_FIELDS, "index_member_all")
    _reject_row_limit(raw, row_limit, "index_member_all")
    if (
        expected_l1_code is not None
        and not raw["l1_code"].astype(str).eq(expected_l1_code).all()
    ):
        raise ValueError("index_member_all response contains an unexpected industry")
    codes = set(raw["l1_code"].dropna().astype(str))
    unknown = sorted(codes - allowed_l1_codes)
    if unknown:
        raise ValueError(
            f"industry membership contains unknown SW2021 codes: {unknown}"
        )
    effective_from = _dates(raw["in_date"], required=True)
    effective_to = _dates(raw["out_date"], required=False)
    result = pd.DataFrame(
        {
            "industry_id": "CN:SW2021:" + raw["l1_code"].astype("string"),
            "security_id": raw["ts_code"].map(to_security_id),
            "effective_from": effective_from,
            "effective_to": effective_to,
            "announced_at": pd.Series(
                pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]"
            ),
            "known_at": _utc_dates(effective_from),
            "known_at_source": "effective_date_fallback",
            "source": "tushare.index_member_all",
        }
    )
    invalid = result["effective_to"].notna() & (
        result["effective_to"] < result["effective_from"]
    )
    if invalid.any():
        raise ValueError("industry membership effective_to precedes effective_from")
    if result.duplicated(["security_id", "industry_id", "effective_from"]).any():
        raise ValueError("industry membership returned duplicate interval keys")
    if _interval_overlap_count(result):
        raise ValueError("industry membership intervals overlap")
    return result.sort_values(
        ["security_id", "effective_from", "industry_id"], kind="stable"
    ).reset_index(drop=True)


def industry_as_of(intervals: pd.DataFrame, as_of: date) -> pd.DataFrame:
    if intervals.empty:
        return intervals.copy()
    instant = pd.Timestamp(as_of, tz="UTC")
    day = pd.Timestamp(as_of)
    selected = intervals.loc[
        (pd.to_datetime(intervals["effective_from"]) <= day)
        & (
            intervals["effective_to"].isna()
            | (pd.to_datetime(intervals["effective_to"]) >= day)
        )
        & (pd.to_datetime(intervals["known_at"], utc=True) <= instant)
    ]
    return selected.reset_index(drop=True)


def validate_exposure_tables(
    tables: ExposureTables, known_security_ids: set[str]
) -> dict[str, object]:
    duplicate_count = sum(
        int(frame.duplicated(keys).sum())
        for frame, keys in (
            (tables.market_cap, ["trade_date", "security_id"]),
            (tables.industry_definition, ["industry_id"]),
            (
                tables.industry_membership,
                ["security_id", "industry_id", "effective_from"],
            ),
        )
        if set(keys).issubset(frame.columns)
    )
    referenced = pd.concat(
        [
            tables.market_cap.get("security_id", pd.Series(dtype="string")),
            tables.industry_membership.get("security_id", pd.Series(dtype="string")),
        ],
        ignore_index=True,
    )
    unknown_security = int((~referenced.isin(known_security_ids)).sum())
    definitions = set(tables.industry_definition.get("industry_id", []))
    unknown_industry = int(
        (
            ~tables.industry_membership.get(
                "industry_id", pd.Series(dtype="string")
            ).isin(definitions)
        ).sum()
    )
    overlap_count = _interval_overlap_count(tables.industry_membership)
    cap = tables.market_cap.reindex(
        columns=["total_market_cap_cny", "float_market_cap_cny"]
    ).apply(pd.to_numeric, errors="coerce")
    invalid_cap = int((cap.isna() | ~np.isfinite(cap) | (cap <= 0)).sum().sum())
    checks = {
        "empty_required_table": _quality_check(
            sum(
                frame.empty
                for frame in (
                    tables.market_cap,
                    tables.industry_definition,
                    tables.industry_membership,
                )
            )
        ),
        "duplicate_keys": _quality_check(duplicate_count),
        "industry_interval_overlap": _quality_check(overlap_count),
        "unknown_security_reference": _quality_check(unknown_security),
        "unknown_industry_reference": _quality_check(unknown_industry),
        "invalid_market_cap": _quality_check(invalid_cap),
    }
    status = "error" if any(item["count"] for item in checks.values()) else "pass"
    return {
        "schema_version": 1,
        "policy": "phase6_exposure_quality_v1",
        "status": status,
        "summary": {
            "market_cap_count": len(tables.market_cap),
            "industry_definition_count": len(tables.industry_definition),
            "industry_membership_count": len(tables.industry_membership),
        },
        "checks": checks,
    }


def probe_exposure_capabilities(config_dir: Path, data_dir: Path) -> dict[str, object]:
    config, _ = load_robustness_config(config_dir / "robustness.yaml")
    provider = _provider_from_environment(data_dir, config)
    phase5 = _load_phase5_tables(data_dir, config)
    codes = _historical_ts_codes(phase5["membership"])
    if not codes:
        raise ValueError("Phase 5 historical CSI 300 membership is empty")
    definition = provider.query(
        config.exposure_source.endpoints.industry_classification,
        {"level": "L1", "src": config.exposure_source.classification_standard},
        INDEX_CLASSIFY_FIELDS,
    )
    normalized = normalize_industry_definition(definition.frame)
    first_industry = str(normalized.iloc[0]["source_index_code"])
    member = provider.query(
        config.exposure_source.endpoints.industry_membership,
        {"l1_code": first_industry},
        INDUSTRY_MEMBER_FIELDS,
    )
    first_date = config.warmup.start.strftime("%Y%m%d")
    market = provider.query(
        config.exposure_source.endpoints.market_cap,
        {"ts_code": codes[0], "start_date": first_date, "end_date": first_date},
        DAILY_BASIC_FIELDS,
    )
    _reject_row_limit(member.frame, INDUSTRY_MEMBER_ROW_LIMIT, "index_member_all")
    _reject_row_limit(market.frame, DAILY_BASIC_ROW_LIMIT, "daily_basic")
    return {
        "provider": "tushare",
        "classification_standard": "SW2021",
        "bounded_probe": True,
        "industry_count": len(normalized),
        "sample_industry": first_industry,
        "sample_security": codes[0],
        "capabilities": {
            "daily_basic": {"row_count": len(market.frame)},
            "index_classify": {"row_count": len(definition.frame)},
            "index_member_all": {"row_count": len(member.frame)},
        },
    }


def acquire_exposure_tables(
    data_dir: Path,
    config: RobustnessConfig,
    provider: QueryProvider,
) -> tuple[ExposureTables, list[TushareArtifact]]:
    phase5 = _load_phase5_tables(data_dir, config)
    ts_codes = _historical_ts_codes(phase5["membership"])
    if not ts_codes:
        raise ValueError("Phase 5 historical CSI 300 membership is empty")
    start = config.warmup.start.strftime("%Y%m%d")
    end = config.walk_forward_folds[-1].end.strftime("%Y%m%d")

    definition_result = provider.query(
        config.exposure_source.endpoints.industry_classification,
        {"level": "L1", "src": config.exposure_source.classification_standard},
        INDEX_CLASSIFY_FIELDS,
    )
    definitions = normalize_industry_definition(definition_result.frame)
    first_code = str(definitions.iloc[0]["source_index_code"])
    probe_member = provider.query(
        config.exposure_source.endpoints.industry_membership,
        {"l1_code": first_code},
        INDUSTRY_MEMBER_FIELDS,
    )
    probe_market = provider.query(
        config.exposure_source.endpoints.market_cap,
        {"ts_code": ts_codes[0], "start_date": start, "end_date": start},
        DAILY_BASIC_FIELDS,
    )
    _reject_row_limit(probe_member.frame, INDUSTRY_MEMBER_ROW_LIMIT, "index_member_all")
    _reject_row_limit(probe_market.frame, DAILY_BASIC_ROW_LIMIT, "daily_basic")

    worker_count = min(config.exposure_source.maximum_concurrency, len(ts_codes))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        market_results = list(
            executor.map(
                lambda code: provider.query(
                    config.exposure_source.endpoints.market_cap,
                    {"ts_code": code, "start_date": start, "end_date": end},
                    DAILY_BASIC_FIELDS,
                ),
                ts_codes,
            )
        )
    industry_codes = sorted(definitions["source_index_code"].astype(str))
    member_results = [
        probe_member
        if code == first_code
        else provider.query(
            config.exposure_source.endpoints.industry_membership,
            {"l1_code": code},
            INDUSTRY_MEMBER_FIELDS,
        )
        for code in industry_codes
    ]
    market = pd.concat(
        [
            normalize_market_cap(item.frame, expected_ts_code=code)
            for code, item in zip(ts_codes, market_results, strict=True)
        ],
        ignore_index=True,
    )
    membership = pd.concat(
        [
            normalize_industry_membership(
                item.frame,
                set(industry_codes),
                expected_l1_code=code,
            )
            for code, item in zip(industry_codes, member_results, strict=True)
        ],
        ignore_index=True,
    )
    if _interval_overlap_count(membership):
        raise ValueError("industry membership intervals overlap")
    tables = ExposureTables(market, definitions, membership)
    artifacts = _unique_artifacts(
        [definition_result, probe_market, *market_results, *member_results]
    )
    return tables, artifacts


def _provider_from_environment(
    data_dir: Path, config: RobustnessConfig
) -> TushareProvider:
    source = config.exposure_source
    return TushareProvider(
        data_dir,
        TushareSourceConfig.model_validate(
            {
                "provider": source.provider,
                "maximum_concurrency": source.maximum_concurrency,
                "request_timeout_seconds": source.request_timeout_seconds,
                "max_attempts": source.max_attempts,
                "retry_delay_seconds": source.retry_delay_seconds,
                "request_interval_seconds": source.request_interval_seconds,
            }
        ),
        token=os.environ.get("TUSHARE_TOKEN", ""),
        http_url=os.environ.get("TUSHARE_HTTP_URL", "https://api.tushare.pro"),
    )


def _load_phase5_tables(
    data_dir: Path, config: RobustnessConfig
) -> dict[str, pd.DataFrame]:
    manifest_path = data_dir / "manifests" / config.phase5_snapshot_id / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Phase 5 manifest is missing: {manifest_path}")
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("snapshot_id") != config.phase5_snapshot_id:
        raise ValueError("Phase 5 manifest identity mismatch")
    artifacts = {str(item["name"]): item for item in manifest.get("artifacts", [])}
    result: dict[str, pd.DataFrame] = {}
    for key, name in (
        ("security", "security_master.parquet"),
        ("membership", "index_membership.parquet"),
    ):
        item = artifacts.get(name)
        if item is None:
            if key == "membership":
                result[key] = pd.DataFrame(columns=["security_id"])
                continue
            raise ValueError(f"Phase 5 manifest missing {name}")
        path = data_dir / str(item["path"])
        if not path.is_file() or file_sha256(path) != str(item["sha256"]):
            raise ValueError(f"Phase 5 {name} checksum mismatch")
        result[key] = pd.read_parquet(path)
    return result


def _historical_ts_codes(membership: pd.DataFrame) -> list[str]:
    return sorted(
        _security_id_to_ts_code(value) for value in set(membership["security_id"])
    )


def _security_id_to_ts_code(value: object) -> str:
    _, exchange, symbol = str(value).split(":")
    suffix = {"SSE": "SH", "SZSE": "SZ", "BSE": "BJ"}[exchange]
    return f"{symbol}.{suffix}"


def _unique_artifacts(results: list[TushareQueryResult]) -> list[TushareArtifact]:
    return sorted(
        {item.artifact.request_sha256: item.artifact for item in results}.values(),
        key=lambda item: (item.api_name, item.request_sha256),
    )


def _require_fields(raw: pd.DataFrame, fields: tuple[str, ...], label: str) -> None:
    missing = sorted(set(fields) - set(raw.columns))
    if missing:
        raise ValueError(f"{label} missing required fields: {missing}")


def _reject_row_limit(raw: pd.DataFrame, row_limit: int, label: str) -> None:
    if len(raw) >= row_limit:
        raise RuntimeError(
            f"{label} may be truncated at provider row limit {row_limit}"
        )


def _dates(values: pd.Series, *, required: bool) -> pd.Series:
    cleaned = values.replace("", pd.NA)
    result = pd.to_datetime(cleaned, format="%Y%m%d", errors="coerce")
    if required and result.isna().any():
        raise ValueError("required date field contains missing or invalid values")
    if (cleaned.notna() & result.isna()).any():
        raise ValueError("date field contains invalid values")
    return result


def _utc_dates(values: pd.Series) -> pd.Series:
    result = pd.to_datetime(values, errors="coerce")
    if result.dt.tz is None:
        return result.dt.tz_localize("UTC")
    return result.dt.tz_convert("UTC")


def _interval_overlap_count(frame: pd.DataFrame) -> int:
    required = {"security_id", "effective_from", "effective_to"}
    if frame.empty or not required.issubset(frame.columns):
        return 0
    count = 0
    for _, group in frame.groupby("security_id", sort=False):
        prior_end: pd.Timestamp | None = None
        for row in group.sort_values("effective_from", kind="stable").itertuples():
            start = pd.Timestamp(row.effective_from)
            end = (
                pd.Timestamp.max.normalize()
                if pd.isna(row.effective_to)
                else pd.Timestamp(row.effective_to)
            )
            if prior_end is not None and start <= prior_end:
                count += 1
            prior_end = end if prior_end is None else max(prior_end, end)
    return count


def _quality_check(count: int) -> dict[str, object]:
    return {
        "severity": "error",
        "status": "pass" if count == 0 else "fail",
        "count": count,
    }
