from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alpha_lab.research_data.config import ResearchDataConfig
from alpha_lab.research_data.contracts import ResearchTables


def build_research_quality_report(
    tables: ResearchTables, config: ResearchDataConfig
) -> dict[str, Any]:
    duplicate_count = sum(
        _duplicate_count(frame, keys)
        for frame, keys in (
            (tables.security_master, ["security_id"]),
            (
                tables.security_name_history,
                ["security_id", "effective_from"],
            ),
            (
                tables.index_membership,
                ["index_id", "security_id", "effective_from"],
            ),
            (tables.daily_bar, ["trade_date", "security_id"]),
            (
                tables.adjustment_factor,
                ["trade_date", "security_id", "factor_type"],
            ),
            (tables.daily_status, ["trade_date", "security_id"]),
        )
    )
    membership_overlap = _interval_overlap_count(
        tables.index_membership, ["index_id", "security_id"]
    )
    name_overlap = _interval_overlap_count(
        tables.security_name_history, ["security_id"]
    )
    suspension_overlap = _interval_overlap_count(tables.suspension, ["security_id"])
    known_security_ids = set(tables.security_master.get("security_id", []))
    unknown_references = sum(
        int((~frame["security_id"].isin(known_security_ids)).sum())
        for frame in (
            tables.index_membership,
            tables.daily_bar,
            tables.adjustment_factor,
            tables.suspension,
            tables.daily_status,
        )
        if "security_id" in frame.columns
    )
    lifecycle_violations = _membership_lifecycle_violations(
        tables.security_master, tables.index_membership
    )
    adjustment = pd.to_numeric(
        tables.adjustment_factor.get("adj_factor", pd.Series(dtype=float)),
        errors="coerce",
    )
    invalid_adjustment = int((~np.isfinite(adjustment) | (adjustment <= 0)).sum())
    nullable_status = sum(
        int(tables.daily_status[column].isna().sum())
        for column in ("is_suspended", "is_st")
        if column in tables.daily_status.columns
    )
    checks: dict[str, dict[str, object]] = {
        "duplicate_keys": _check("error", duplicate_count),
        "membership_overlap": _check("error", membership_overlap),
        "name_history_overlap": _check("error", name_overlap),
        "suspension_overlap": _check("error", suspension_overlap),
        "unknown_security_reference": _check("error", unknown_references),
        "membership_lifecycle_violation": _check("error", lifecycle_violations),
        "invalid_adjustment_factor": _check("error", invalid_adjustment),
        "nullable_status": _check("warning", nullable_status),
    }
    has_error = any(
        item["severity"] == "error" and item["count"] != 0 for item in checks.values()
    )
    has_warning = any(
        item["severity"] == "warning" and item["count"] != 0 for item in checks.values()
    )
    status = "error" if has_error else "warning" if has_warning else "pass"
    delisted = tables.security_master.get(
        "delist_date", pd.Series(dtype="datetime64[ns]")
    )
    return {
        "schema_version": 1,
        "policy": "phase5_point_in_time_quality_v1",
        "status": status,
        "scope": {
            "index_code": config.index_code,
            "start_date": config.start_date.isoformat(),
            "end_date": config.end_date.isoformat(),
        },
        "summary": {
            "security_count": len(tables.security_master),
            "delisted_security_count": int(delisted.notna().sum()),
            "membership_interval_count": len(tables.index_membership),
            "daily_bar_count": len(tables.daily_bar),
            "adjustment_factor_count": len(tables.adjustment_factor),
            "daily_status_count": len(tables.daily_status),
        },
        "checks": checks,
    }


def _check(severity: str, count: int) -> dict[str, object]:
    return {
        "severity": severity,
        "status": "pass" if count == 0 else "fail",
        "count": count,
    }


def _duplicate_count(frame: pd.DataFrame, keys: list[str]) -> int:
    if frame.empty or not set(keys).issubset(frame.columns):
        return 0
    return int(frame.duplicated(keys).sum())


def _interval_overlap_count(frame: pd.DataFrame, groups: list[str]) -> int:
    required = {*groups, "effective_from", "effective_to"}
    if frame.empty or not required.issubset(frame.columns):
        return 0
    count = 0
    group_key: str | list[str] = groups[0] if len(groups) == 1 else groups
    for _, group in frame.groupby(group_key, dropna=False, sort=False):
        ordered = group.sort_values("effective_from", kind="stable")
        previous_end: pd.Timestamp | None = None
        for row in ordered.itertuples(index=False):
            start = pd.Timestamp(row.effective_from)
            end_value = row.effective_to
            end = (
                pd.Timestamp.max.normalize()
                if pd.isna(end_value)
                else pd.Timestamp(end_value)
            )
            if previous_end is not None and start <= previous_end:
                count += 1
            previous_end = end if previous_end is None else max(previous_end, end)
    return count


def _membership_lifecycle_violations(
    security: pd.DataFrame, membership: pd.DataFrame
) -> int:
    required_security = {"security_id", "list_date", "delist_date"}
    required_membership = {"security_id", "effective_from", "effective_to"}
    if (
        security.empty
        or membership.empty
        or not required_security.issubset(security.columns)
        or not required_membership.issubset(membership.columns)
    ):
        return 0
    merged = membership.merge(
        security[list(required_security)],
        on="security_id",
        how="left",
        validate="many_to_one",
    )
    before_listing = pd.to_datetime(merged["effective_from"]) < pd.to_datetime(
        merged["list_date"]
    )
    after_delisting = merged["delist_date"].notna() & (
        merged["effective_to"].isna()
        | (
            pd.to_datetime(merged["effective_to"], errors="coerce")
            > pd.to_datetime(merged["delist_date"], errors="coerce")
        )
    )
    return int((before_listing | after_delisting).sum())
