from __future__ import annotations

import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
INDEX_CLASSIFY_L2_FIELDS = (
    "index_code",
    "industry_name",
    "level",
    "industry_code",
    "parent_code",
    "src",
)
INDEX_CLASSIFY_L3_FIELDS = INDEX_CLASSIFY_L2_FIELDS
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


@dataclass(frozen=True)
class HistoricalTaxonomyBridge:
    l2_to_stable_l1: Mapping[str, str]
    l3_to_stable_l1: Mapping[str, str]
    source_version: str = "SW2014"
    target_version: str = "SW2021"


def normalize_market_cap(
    raw: pd.DataFrame,
    *,
    row_limit: int = DAILY_BASIC_ROW_LIMIT,
    expected_ts_code: str | None = None,
    expected_start_date: date | None = None,
    expected_end_date: date | None = None,
) -> pd.DataFrame:
    _require_fields(raw, DAILY_BASIC_FIELDS, "daily_basic")
    _reject_row_limit(raw, row_limit, "daily_basic")
    if expected_ts_code is not None and raw.empty:
        raise ValueError(f"daily_basic returned empty response for {expected_ts_code}")
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
    if (
        expected_start_date is not None
        and (frame["trade_date"].dt.date < expected_start_date).any()
    ):
        raise ValueError("daily_basic response precedes requested start_date")
    if (
        expected_end_date is not None
        and (frame["trade_date"].dt.date > expected_end_date).any()
    ):
        raise ValueError("daily_basic response exceeds requested end_date")
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


def normalize_industry_l2_mapping(
    raw: pd.DataFrame, definitions: pd.DataFrame
) -> dict[str, str]:
    _require_fields(raw, INDEX_CLASSIFY_L2_FIELDS, "index_classify L2")
    selected = raw.loc[
        raw["src"].astype("string").eq("SW2021")
        & raw["level"].astype("string").str.upper().eq("L2")
    ].copy()
    if selected.empty:
        raise ValueError("index_classify returned no SW2021 level-two industries")
    required = selected[["index_code", "industry_code", "parent_code"]].replace(
        "", pd.NA
    )
    if required.isna().any().any():
        raise ValueError("index_classify L2 contains missing hierarchy fields")
    l1 = definitions[["industry_code", "source_index_code"]].replace("", pd.NA)
    if l1.isna().any().any() or l1["industry_code"].duplicated().any():
        raise ValueError("SW2021 L1 industry_code hierarchy is incomplete or ambiguous")
    parent_to_l1 = dict(
        zip(
            l1["industry_code"].astype(str),
            l1["source_index_code"].astype(str),
            strict=True,
        )
    )
    unknown = sorted(set(required["parent_code"].astype(str)) - set(parent_to_l1))
    if unknown:
        raise ValueError(f"index_classify L2 contains unknown L1 parent: {unknown}")
    pairs = pd.DataFrame(
        {
            "l2_code": required["industry_code"].astype(str),
            "l1_code": required["parent_code"].astype(str).map(parent_to_l1),
        }
    )
    if (
        required["index_code"].astype(str).duplicated().any()
        or pairs["l2_code"].duplicated().any()
    ):
        raise ValueError("index_classify contains ambiguous L2 parent mapping")
    return dict(zip(pairs["l2_code"], pairs["l1_code"], strict=True))


def normalize_historical_taxonomy_bridge(
    raw_l1: pd.DataFrame,
    raw_l2: pd.DataFrame,
    raw_l3: pd.DataFrame,
    definitions: pd.DataFrame,
) -> HistoricalTaxonomyBridge:
    """Bridge historical SW codes through stable L1 index codes, never names."""
    levels: dict[str, pd.DataFrame] = {}
    for level, raw, fields in (
        ("L1", raw_l1, INDEX_CLASSIFY_FIELDS),
        ("L2", raw_l2, INDEX_CLASSIFY_L2_FIELDS),
        ("L3", raw_l3, INDEX_CLASSIFY_L3_FIELDS),
    ):
        _require_fields(raw, fields, f"index_classify SW2014 {level}")
        selected = raw.loc[
            raw["src"].astype("string").eq("SW2014")
            & raw["level"].astype("string").str.upper().eq(level)
        ].copy()
        required = ["index_code", "industry_code"]
        if level != "L1":
            required.append("parent_code")
        values = selected[required].replace("", pd.NA)
        if values.empty or values.isna().any().any():
            raise ValueError(f"SW2014 {level} taxonomy is incomplete")
        if (
            values["index_code"].astype(str).duplicated().any()
            or values["industry_code"].astype(str).duplicated().any()
        ):
            raise ValueError(f"SW2014 {level} taxonomy is ambiguous")
        levels[level] = values.astype(str)

    stable_targets = definitions["source_index_code"].astype(str)
    if stable_targets.duplicated().any():
        raise ValueError("SW2021 stable L1 index code is ambiguous")
    l1_by_industry = dict(
        zip(
            levels["L1"]["industry_code"],
            levels["L1"]["index_code"],
            strict=True,
        )
    )
    l2_to_l1: dict[str, str] = {}
    for row in levels["L2"].itertuples(index=False):
        stable_l1 = l1_by_industry.get(str(row.parent_code))
        if stable_l1 is None:
            raise ValueError("SW2014 L2 has an unknown L1 parent")
        l2_to_l1[str(row.industry_code)] = stable_l1
    l3_to_l1: dict[str, str] = {}
    for row in levels["L3"].itertuples(index=False):
        stable_l1 = l2_to_l1.get(str(row.parent_code))
        if stable_l1 is None:
            raise ValueError("SW2014 L3 has an unknown L2 parent")
        l3_to_l1[str(row.index_code)] = stable_l1
    return HistoricalTaxonomyBridge(l2_to_l1, l3_to_l1)


def normalize_industry_membership(
    raw: pd.DataFrame,
    allowed_l1_codes: set[str],
    *,
    row_limit: int = INDUSTRY_MEMBER_ROW_LIMIT,
    expected_l1_code: str | None = None,
    allow_empty_expected_l1: bool = False,
) -> pd.DataFrame:
    _require_fields(raw, INDUSTRY_MEMBER_FIELDS, "index_member_all")
    _reject_row_limit(raw, row_limit, "index_member_all")
    if expected_l1_code is not None and raw.empty and not allow_empty_expected_l1:
        raise ValueError(
            f"index_member_all returned empty response for {expected_l1_code}"
        )
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
            "taxonomy_mapping_source": raw.get(
                "taxonomy_mapping_source",
                pd.Series("direct_l1_code", index=raw.index),
            ).astype("string"),
            "taxonomy_source_version": raw.get(
                "taxonomy_source_version",
                pd.Series("SW2021", index=raw.index),
            ).astype("string"),
            "taxonomy_target_version": "SW2021",
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


def normalize_industry_membership_backfill(
    raw: pd.DataFrame,
    l2_to_l1: Mapping[str, str],
    allowed_l1_codes: set[str],
    *,
    expected_ts_code: str,
    row_limit: int = INDUSTRY_MEMBER_ROW_LIMIT,
    allow_empty: bool = False,
    historical_bridge: HistoricalTaxonomyBridge | None = None,
) -> pd.DataFrame:
    _require_fields(raw, INDUSTRY_MEMBER_FIELDS, "index_member_all ts_code backfill")
    _reject_row_limit(raw, row_limit, "index_member_all ts_code backfill")
    if raw.empty:
        if allow_empty:
            return normalize_industry_membership(raw, allowed_l1_codes)
        raise ValueError(
            f"index_member_all returned empty backfill response for {expected_ts_code}"
        )
    ts_codes = raw["ts_code"].replace("", pd.NA)
    if ts_codes.isna().any() or not ts_codes.astype(str).eq(expected_ts_code).all():
        raise ValueError(
            "index_member_all backfill contains an unexpected or empty security"
        )
    l2_codes = raw["l2_code"].replace("", pd.NA)
    if l2_codes.isna().any() and historical_bridge is None:
        raise ValueError("index_member_all backfill contains missing L2 code")
    mapped = l2_codes.astype("string").map(l2_to_l1).astype("string")
    mapping_source = pd.Series("sw2021_l2_parent", index=raw.index, dtype="string")
    source_version = pd.Series("SW2021", index=raw.index, dtype="string")
    unresolved = mapped.isna()
    if unresolved.any() and historical_bridge is not None:
        historical_l2 = (
            l2_codes.astype("string")
            .map(historical_bridge.l2_to_stable_l1)
            .astype("string")
        )
        l3_codes = raw["l3_code"].replace("", pd.NA).astype("string")
        historical_l3 = l3_codes.map(historical_bridge.l3_to_stable_l1).astype("string")
        conflict = (
            unresolved
            & historical_l2.notna()
            & historical_l3.notna()
            & historical_l2.ne(historical_l3)
        )
        if conflict.any():
            raise ValueError("historical L2 and L3 taxonomy paths conflict")
        historical = historical_l3.fillna(historical_l2)
        mapped.loc[unresolved] = historical.loc[unresolved]
        both = unresolved & historical_l2.notna() & historical_l3.notna()
        only_l3 = unresolved & historical_l3.notna() & ~historical_l2.notna()
        mapping_source.loc[both] = "sw2014_l3_l2_l1_bridge"
        mapping_source.loc[only_l3] = "sw2014_l3_l2_l1_bridge"
        mapping_source.loc[unresolved & ~only_l3 & ~both] = "sw2014_l2_l1_bridge"
        source_version.loc[unresolved] = historical_bridge.source_version
    if mapped.isna().any():
        missing = sorted(set(l2_codes.astype("string")[mapped.isna()].dropna()))
        raise ValueError(f"index_member_all backfill has unmapped L2 code: {missing}")
    invalid_historical_target = source_version.eq("SW2014") & ~mapped.isin(
        allowed_l1_codes
    )
    if invalid_historical_target.any():
        unstable = sorted(set(mapped[invalid_historical_target].astype(str)))
        raise ValueError(
            f"used historical L1 is absent from SW2021 definitions: {unstable}"
        )
    supplied = raw["l1_code"].replace("", pd.NA).astype("string")
    conflicts = supplied.notna() & supplied.ne(mapped.astype("string"))
    if conflicts.any():
        raise ValueError("index_member_all L1 conflicts with audited L2 parent mapping")
    recovered = raw.copy()
    recovered["l1_code"] = mapped
    recovered["taxonomy_mapping_source"] = mapping_source
    recovered["taxonomy_source_version"] = source_version
    return normalize_industry_membership(recovered, allowed_l1_codes)


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
    tables: ExposureTables,
    known_security_ids: set[str],
    *,
    expected_security_ids: set[str] | None = None,
    expected_industry_ids: set[str] | None = None,
    expected_market_observations: pd.DataFrame | None = None,
    minimum_temporal_coverage: float | None = None,
    minimum_industry_observation_coverage: float = 1.0,
    market_start_date: date | None = None,
    market_end_date: date | None = None,
) -> dict[str, object]:
    if not 0 < minimum_industry_observation_coverage <= 1:
        raise ValueError("minimum industry observation coverage must be in (0, 1]")
    expected_security_ids = expected_security_ids or set(
        tables.market_cap.attrs.get("expected_security_ids", [])
    )
    expected_industry_ids = expected_industry_ids or set(
        tables.industry_membership.attrs.get("expected_industry_ids", [])
    )
    if expected_market_observations is None:
        stored_observations = tables.market_cap.attrs.get(
            "expected_market_observations"
        )
        expected_market_observations = (
            stored_observations
            if isinstance(stored_observations, pd.DataFrame)
            else pd.DataFrame(columns=["trade_date", "security_id"])
        )
    if minimum_temporal_coverage is None:
        minimum_temporal_coverage = float(
            tables.market_cap.attrs.get("minimum_temporal_coverage", 1.0)
        )
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
    market_dates = pd.to_datetime(
        tables.market_cap.get("trade_date", pd.Series(dtype="datetime64[ns]")),
        errors="coerce",
    )
    out_of_scope_market_cap = 0
    if market_start_date is not None:
        out_of_scope_market_cap += int((market_dates.dt.date < market_start_date).sum())
    if market_end_date is not None:
        out_of_scope_market_cap += int((market_dates.dt.date > market_end_date).sum())
    observed_security_ids = set(tables.market_cap.get("security_id", []))
    observed_industry_ids = set(tables.industry_membership.get("industry_id", []))
    observed_membership_security_ids = set(
        tables.industry_membership.get("security_id", [])
    )
    missing_security_coverage = len(expected_security_ids - observed_security_ids)
    missing_membership_security_coverage = len(
        expected_security_ids - observed_membership_security_ids
    )
    missing_industry_coverage = len(expected_industry_ids - observed_industry_ids)
    explicit_empty_industry_ids = set(
        tables.industry_membership.attrs.get("explicit_empty_industry_ids", [])
    )
    expected_keys = _observation_keys(expected_market_observations)
    observed_keys = _observation_keys(tables.market_cap)
    covered_keys = expected_keys & observed_keys
    expected_observation_count = len(expected_keys)
    observed_observation_count = len(covered_keys)
    temporal_coverage_ratio = (
        observed_observation_count / expected_observation_count
        if expected_observation_count
        else 1.0
    )
    missing_observations = expected_observation_count - observed_observation_count
    insufficient_temporal_coverage = (
        missing_observations
        if temporal_coverage_ratio < minimum_temporal_coverage
        else 0
    )
    expected_by_security: dict[str, int] = {}
    covered_by_security: dict[str, int] = {}
    for trade_date, security_id in expected_keys:
        expected_by_security[security_id] = expected_by_security.get(security_id, 0) + 1
        if (trade_date, security_id) in covered_keys:
            covered_by_security[security_id] = (
                covered_by_security.get(security_id, 0) + 1
            )
    undercovered_security = sum(
        covered_by_security.get(security_id, 0) / expected_count
        < minimum_temporal_coverage
        for security_id, expected_count in expected_by_security.items()
    )
    industry_expected, industry_matched, missing_industry_security_ids = (
        _industry_observation_coverage(
            expected_market_observations, tables.industry_membership
        )
    )
    missing_industry_observations = industry_expected - industry_matched
    industry_coverage_ratio = (
        industry_matched / industry_expected if industry_expected else 1.0
    )
    insufficient_industry_coverage = (
        missing_industry_observations
        if industry_coverage_ratio < minimum_industry_observation_coverage
        else 0
    )
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
        "missing_security_coverage": _quality_check(missing_security_coverage),
        "missing_membership_security_coverage": _quality_check(
            missing_membership_security_coverage, severity="warning"
        ),
        "missing_industry_coverage": _quality_check(
            missing_industry_coverage, severity="warning"
        ),
        "missing_industry_observations": _quality_check(
            missing_industry_observations, severity="warning"
        ),
        "insufficient_industry_observation_coverage": _quality_check(
            insufficient_industry_coverage
        ),
        "explicit_empty_industry_definition": {
            "severity": "info",
            "status": "pass",
            "count": len(explicit_empty_industry_ids),
        },
        "insufficient_temporal_coverage": _quality_check(
            insufficient_temporal_coverage
        ),
        "undercovered_security": _quality_check(undercovered_security),
        "market_cap_out_of_scope": _quality_check(out_of_scope_market_cap),
    }
    status = (
        "error"
        if any(
            item["count"] and item["severity"] == "error" for item in checks.values()
        )
        else "warning"
        if any(
            item["count"] and item["severity"] == "warning" for item in checks.values()
        )
        else "pass"
    )
    return {
        "schema_version": 1,
        "policy": "phase6_exposure_quality_v1",
        "status": status,
        "summary": {
            "market_cap_count": len(tables.market_cap),
            "industry_definition_count": len(tables.industry_definition),
            "industry_membership_count": len(tables.industry_membership),
            "expected_security_count": len(expected_security_ids),
            "expected_industry_count": len(expected_industry_ids),
            "explicit_empty_industry_count": len(explicit_empty_industry_ids),
            "expected_observation_count": expected_observation_count,
            "observed_observation_count": observed_observation_count,
            "temporal_coverage_ratio": round(temporal_coverage_ratio, 12),
            "minimum_temporal_coverage": minimum_temporal_coverage,
            "industry_expected_observation_count": industry_expected,
            "industry_matched_observation_count": industry_matched,
            "industry_missing_observation_count": missing_industry_observations,
            "industry_observation_coverage_ratio": round(industry_coverage_ratio, 12),
            "minimum_industry_observation_coverage": (
                minimum_industry_observation_coverage
            ),
            "missing_industry_security_count": len(missing_industry_security_ids),
            "missing_industry_security_ids": missing_industry_security_ids,
            "historical_taxonomy_bridge_count": int(
                tables.industry_membership.get(
                    "taxonomy_source_version", pd.Series(dtype="string")
                )
                .astype("string")
                .eq("SW2014")
                .sum()
            ),
        },
        "checks": checks,
    }


def probe_exposure_capabilities(config_dir: Path, data_dir: Path) -> dict[str, object]:
    config, _ = load_robustness_config(config_dir / "robustness.yaml")
    provider = _provider_from_environment(data_dir, config)
    phase5 = _load_phase5_tables(data_dir, config)
    sample_date, sample_security_id = _probe_observation(phase5["observations"])
    sample_code = _security_id_to_ts_code(sample_security_id)
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
    normalize_industry_membership(
        member.frame,
        set(normalized["source_index_code"].astype(str)),
        expected_l1_code=first_industry,
        allow_empty_expected_l1=True,
    )
    first_date = sample_date.strftime("%Y%m%d")
    market = provider.query(
        config.exposure_source.endpoints.market_cap,
        {"ts_code": sample_code, "start_date": first_date, "end_date": first_date},
        DAILY_BASIC_FIELDS,
    )
    normalize_market_cap(
        market.frame,
        expected_ts_code=sample_code,
        expected_start_date=sample_date,
        expected_end_date=sample_date,
    )
    return {
        "provider": "tushare",
        "classification_standard": "SW2021",
        "bounded_probe": True,
        "industry_count": len(normalized),
        "sample_industry": first_industry,
        "sample_security": sample_code,
        "sample_trade_date": sample_date.isoformat(),
        "capabilities": {
            "daily_basic": {"row_count": len(market.frame)},
            "index_classify": {"row_count": len(definition.frame)},
            "index_member_all": {"row_count": len(member.frame)},
        },
        "sample_industry_explicitly_empty": member.frame.empty,
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
    end = config.test.end.strftime("%Y%m%d")
    sample_date, sample_security_id = _probe_observation(phase5["observations"])
    sample_code = _security_id_to_ts_code(sample_security_id)

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
    normalize_industry_membership(
        probe_member.frame,
        set(definitions["source_index_code"].astype(str)),
        expected_l1_code=first_code,
        allow_empty_expected_l1=True,
    )
    probe_market = provider.query(
        config.exposure_source.endpoints.market_cap,
        {
            "ts_code": sample_code,
            "start_date": sample_date.strftime("%Y%m%d"),
            "end_date": sample_date.strftime("%Y%m%d"),
        },
        DAILY_BASIC_FIELDS,
    )
    normalize_market_cap(
        probe_market.frame,
        expected_ts_code=sample_code,
        expected_start_date=sample_date,
        expected_end_date=sample_date,
    )

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
            normalize_market_cap(
                item.frame,
                expected_ts_code=code,
                expected_start_date=config.warmup.start,
                expected_end_date=config.test.end,
            )
            for code, item in zip(ts_codes, market_results, strict=True)
        ],
        ignore_index=True,
    )
    expected_security_ids = {to_security_id(code) for code in ts_codes}
    explicit_empty_industry_ids = {
        f"CN:SW2021:{code}"
        for code, item in zip(industry_codes, member_results, strict=True)
        if item.frame.empty
    }
    normalized_l1 = [
        normalize_industry_membership(
            item.frame,
            set(industry_codes),
            expected_l1_code=code,
            allow_empty_expected_l1=True,
        )
        for code, item in zip(industry_codes, member_results, strict=True)
    ]
    membership = pd.concat(normalized_l1, ignore_index=True)
    membership = membership.loc[
        membership["security_id"].isin(expected_security_ids)
    ].copy()
    missing_ts_codes = sorted(
        _security_id_to_ts_code(security_id)
        for security_id in expected_security_ids - set(membership["security_id"])
    )
    hierarchy_result: TushareQueryResult | None = None
    historical_taxonomy_results: list[TushareQueryResult] = []
    backfill_results: list[TushareQueryResult] = []
    if missing_ts_codes:
        hierarchy_result = provider.query(
            config.exposure_source.endpoints.industry_classification,
            {"level": "L2", "src": config.exposure_source.classification_standard},
            INDEX_CLASSIFY_L2_FIELDS,
        )
        l2_to_l1 = normalize_industry_l2_mapping(hierarchy_result.frame, definitions)
        backfill_workers = min(
            config.exposure_source.maximum_concurrency, len(missing_ts_codes)
        )
        with ThreadPoolExecutor(max_workers=backfill_workers) as executor:
            backfill_results = list(
                executor.map(
                    lambda code: provider.query(
                        config.exposure_source.endpoints.industry_membership,
                        {"ts_code": code},
                        INDUSTRY_MEMBER_FIELDS,
                    ),
                    missing_ts_codes,
                )
            )
        requires_historical_bridge = any(
            not item.frame.empty
            and (
                item.frame["l2_code"].replace("", pd.NA).isna().any()
                or not item.frame["l2_code"]
                .replace("", pd.NA)
                .dropna()
                .astype(str)
                .isin(l2_to_l1)
                .all()
            )
            for item in backfill_results
        )
        historical_bridge: HistoricalTaxonomyBridge | None = None
        if requires_historical_bridge:
            for level, fields in (
                ("L1", INDEX_CLASSIFY_FIELDS),
                ("L2", INDEX_CLASSIFY_L2_FIELDS),
                ("L3", INDEX_CLASSIFY_L3_FIELDS),
            ):
                historical_taxonomy_results.append(
                    provider.query(
                        config.exposure_source.endpoints.industry_classification,
                        {"level": level, "src": "SW2014"},
                        fields,
                    )
                )
            historical_bridge = normalize_historical_taxonomy_bridge(
                historical_taxonomy_results[0].frame,
                historical_taxonomy_results[1].frame,
                historical_taxonomy_results[2].frame,
                definitions,
            )
        backfill = pd.concat(
            [
                normalize_industry_membership_backfill(
                    item.frame,
                    l2_to_l1,
                    set(industry_codes),
                    expected_ts_code=code,
                    allow_empty=True,
                    historical_bridge=historical_bridge,
                )
                for code, item in zip(missing_ts_codes, backfill_results, strict=True)
            ],
            ignore_index=True,
        )
        membership = pd.concat([membership, backfill], ignore_index=True)
    membership = membership.sort_values(
        ["security_id", "effective_from", "industry_id"], kind="stable"
    ).reset_index(drop=True)
    if membership.duplicated(["security_id", "industry_id", "effective_from"]).any():
        raise ValueError("industry membership returned duplicate interval keys")
    expected_industry_ids = set(membership["industry_id"])
    market.attrs["expected_security_ids"] = sorted(expected_security_ids)
    market.attrs["expected_market_observations"] = phase5["observations"]
    market.attrs["minimum_temporal_coverage"] = config.minimum_fold_coverage
    membership.attrs["expected_industry_ids"] = sorted(expected_industry_ids)
    membership.attrs["explicit_empty_industry_ids"] = sorted(
        explicit_empty_industry_ids
    )
    if _interval_overlap_count(membership):
        raise ValueError("industry membership intervals overlap")
    tables = ExposureTables(market, definitions, membership)
    quality = validate_exposure_tables(
        tables,
        set(phase5["security"]["security_id"].astype(str)),
        expected_security_ids=expected_security_ids,
        expected_industry_ids=expected_industry_ids,
        expected_market_observations=phase5["observations"],
        minimum_temporal_coverage=config.minimum_fold_coverage,
        minimum_industry_observation_coverage=(
            config.minimum_industry_observation_coverage
        ),
        market_start_date=config.warmup.start,
        market_end_date=config.test.end,
    )
    if quality["status"] == "error":
        raise ValueError("exposure acquisition coverage gates failed")
    artifact_results = [
        definition_result,
        probe_market,
        *market_results,
        *member_results,
        *([hierarchy_result] if hierarchy_result is not None else []),
        *historical_taxonomy_results,
        *backfill_results,
    ]
    artifacts = _unique_artifacts(artifact_results)
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
    return _load_phase5_exposure_context(
        data_dir,
        config.phase5_snapshot_id,
        config.warmup.start,
        config.test.end,
    )


def _load_phase5_exposure_context(
    data_dir: Path,
    snapshot_id: str,
    start_date: date,
    end_date: date,
) -> dict[str, pd.DataFrame]:
    manifest_path = data_dir / "manifests" / snapshot_id / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Phase 5 manifest is missing: {manifest_path}")
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("snapshot_id") != snapshot_id
        or manifest.get("snapshot_type") != "research_market"
    ):
        raise ValueError("Phase 5 manifest identity mismatch")
    artifacts = {str(item["name"]): item for item in manifest.get("artifacts", [])}
    result: dict[str, pd.DataFrame] = {}
    for key, name in (
        ("security", "security_master.parquet"),
        ("membership", "index_membership.parquet"),
        ("universe", "universe_dates.parquet"),
    ):
        item = artifacts.get(name)
        if item is None:
            raise ValueError(f"Phase 5 manifest missing {name}")
        path = data_dir / str(item["path"])
        if not path.is_file() or file_sha256(path) != str(item["sha256"]):
            raise ValueError(f"Phase 5 {name} checksum mismatch")
        result[key] = pd.read_parquet(path)
    daily_parts: list[pd.DataFrame] = []
    for name, item in sorted(artifacts.items()):
        if not name.startswith("daily_bar/"):
            continue
        path = data_dir / str(item["path"])
        if not path.is_file() or file_sha256(path) != str(item["sha256"]):
            raise ValueError(f"Phase 5 {name} checksum mismatch")
        daily_parts.append(pd.read_parquet(path, columns=["trade_date", "security_id"]))
    if not daily_parts:
        raise ValueError("Phase 5 manifest missing daily_bar observations")
    daily = pd.concat(daily_parts, ignore_index=True).drop_duplicates()
    universe = result["universe"].rename(columns={"as_of_date": "trade_date"})
    observations = universe[["trade_date", "security_id"]].merge(
        daily, on=["trade_date", "security_id"], how="inner"
    )
    observations["trade_date"] = pd.to_datetime(
        observations["trade_date"], errors="raise"
    ).dt.normalize()
    observations = observations.loc[
        (observations["trade_date"].dt.date >= start_date)
        & (observations["trade_date"].dt.date <= end_date)
    ]
    result["observations"] = observations.sort_values(
        ["trade_date", "security_id"], kind="stable"
    ).reset_index(drop=True)
    return result


def _historical_ts_codes(membership: pd.DataFrame) -> list[str]:
    return sorted(
        _security_id_to_ts_code(value) for value in set(membership["security_id"])
    )


def _probe_observation(observations: pd.DataFrame) -> tuple[date, str]:
    if observations.empty:
        raise ValueError("Phase 5 has no verified open active daily observations")
    first = observations.sort_values(["trade_date", "security_id"], kind="stable").iloc[
        0
    ]
    return pd.Timestamp(first["trade_date"]).date(), str(first["security_id"])


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


def _quality_check(count: int, *, severity: str = "error") -> dict[str, object]:
    return {
        "severity": severity,
        "status": "pass" if count == 0 else "fail",
        "count": count,
    }


def _observation_keys(frame: pd.DataFrame) -> set[tuple[pd.Timestamp, str]]:
    if frame.empty or not {"trade_date", "security_id"}.issubset(frame.columns):
        return set()
    dates = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
    return set(zip(dates, frame["security_id"].astype(str), strict=True))


def _industry_observation_coverage(
    observations: pd.DataFrame, memberships: pd.DataFrame
) -> tuple[int, int, list[str]]:
    """Measure point-in-time industry matches over Phase 5 market observations."""
    keys = sorted(_observation_keys(observations), key=lambda item: (item[0], item[1]))
    if not keys:
        return 0, 0, []
    expected = pd.DataFrame(keys, columns=["trade_date", "security_id"])
    required = {"security_id", "effective_from", "effective_to", "known_at"}
    if memberships.empty or not required.issubset(memberships.columns):
        missing_ids = sorted(expected["security_id"].unique().tolist())
        return len(expected), 0, missing_ids
    intervals = memberships[list(required)].copy()
    intervals["effective_from"] = pd.to_datetime(
        intervals["effective_from"], errors="raise"
    ).dt.normalize()
    intervals["effective_to"] = pd.to_datetime(
        intervals["effective_to"], errors="coerce"
    ).dt.normalize()
    intervals["known_at"] = pd.to_datetime(
        intervals["known_at"], errors="raise", utc=True
    )
    joined = expected.merge(intervals, on="security_id", how="left")
    trade = joined["trade_date"]
    valid = (
        (joined["effective_from"] <= trade)
        & (joined["effective_to"].isna() | (trade <= joined["effective_to"]))
        & (joined["known_at"].dt.date <= trade.dt.date)
    )
    matched = joined.loc[valid, ["trade_date", "security_id"]].drop_duplicates()
    matched_keys = set(matched.itertuples(index=False, name=None))
    missing_keys = set(keys) - matched_keys
    missing_ids = sorted({str(item[1]) for item in missing_keys})
    return len(keys), len(matched_keys), missing_ids
