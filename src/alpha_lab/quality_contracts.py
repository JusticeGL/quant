from __future__ import annotations

from typing import Any, cast

_PHASE5_CHECK_SEVERITIES = {
    "duplicate_keys": "error",
    "membership_overlap": "error",
    "name_history_overlap": "error",
    "suspension_overlap": "error",
    "unknown_security_reference": "error",
    "membership_lifecycle_violation": "error",
    "invalid_adjustment_factor": "error",
    "nullable_status": "warning",
    "missing_delist_date": "warning",
}

_EXPOSURE_CHECK_SEVERITIES = {
    "empty_required_table": "error",
    "duplicate_keys": "error",
    "industry_interval_overlap": "error",
    "unknown_security_reference": "error",
    "unknown_industry_reference": "error",
    "invalid_market_cap": "error",
    "missing_security_coverage": "error",
    "missing_membership_security_coverage": "warning",
    "missing_industry_coverage": "warning",
    "missing_industry_observations": "warning",
    "insufficient_industry_observation_coverage": "error",
    "explicit_empty_industry_definition": "info",
    "insufficient_temporal_coverage": "error",
    "undercovered_security": "error",
    "market_cap_out_of_scope": "error",
}


def phase5_quality_failures(
    quality: object,
    manifest: dict[str, Any],
    expected_scope: dict[str, object],
) -> list[str]:
    if not isinstance(quality, dict) or set(quality) != {
        "schema_version",
        "policy",
        "status",
        "scope",
        "summary",
        "checks",
    }:
        return ["quality_schema"]
    if (
        quality.get("schema_version") != 1
        or quality.get("policy") != "phase5_point_in_time_quality_v1"
        or not isinstance(quality.get("scope"), dict)
        or not isinstance(quality.get("summary"), dict)
        or not isinstance(quality.get("checks"), dict)
    ):
        return ["quality_schema"]

    failures: list[str] = []
    checks = quality["checks"]
    check_failure = set(checks) != set(_PHASE5_CHECK_SEVERITIES)
    error_counts: list[int] = []
    warning_counts: list[int] = []
    if not check_failure:
        for name, expected_severity in _PHASE5_CHECK_SEVERITIES.items():
            item = checks[name]
            if not _valid_check(item, expected_severity):
                check_failure = True
                break
            count = item["count"]
            (error_counts if expected_severity == "error" else warning_counts).append(
                count
            )
    if check_failure:
        failures.append("quality_checks")
    derived_status = (
        "error"
        if any(error_counts) or check_failure
        else "warning"
        if any(warning_counts)
        else "pass"
    )
    if (
        quality.get("status") != derived_status
        or quality.get("status") != manifest.get("quality_status")
        or quality.get("status") == "error"
    ):
        failures.append("quality_status")

    scope = quality["scope"]
    if (
        set(scope) != {"index_code", "start_date", "end_date"}
        or not all(isinstance(scope.get(key), str) for key in scope)
        or scope != manifest.get("scope")
        or scope != expected_scope
    ):
        failures.append("quality_scope")

    artifacts = _artifacts_by_name(manifest)
    expected_counts = {
        "security_count": _artifact_row_count(artifacts, "security_master.parquet"),
        "membership_interval_count": _artifact_row_count(
            artifacts, "index_membership.parquet"
        ),
        "daily_bar_count": _partition_row_count(artifacts, "daily_bar/"),
        "adjustment_factor_count": _partition_row_count(
            artifacts, "adjustment_factor/"
        ),
        "daily_status_count": _partition_row_count(artifacts, "daily_status/"),
    }
    summary = quality["summary"]
    summary_keys = {*expected_counts, "delisted_security_count"}
    if (
        set(summary) != summary_keys
        or any(summary.get(key) != value for key, value in expected_counts.items())
        or not _nonnegative_int(summary.get("delisted_security_count"))
        or summary.get("delisted_security_count", 0) > summary.get("security_count", -1)
        or summary != manifest.get("summary")
    ):
        failures.append("quality_row_counts")
    return failures


def exposure_quality_failures(quality: object, manifest: dict[str, Any]) -> list[str]:
    if not isinstance(quality, dict) or set(quality) != {
        "schema_version",
        "policy",
        "status",
        "summary",
        "checks",
    }:
        return ["quality_schema"]
    if (
        quality.get("schema_version") != 1
        or quality.get("policy") != "phase6_exposure_quality_v1"
        or not isinstance(quality.get("summary"), dict)
        or not isinstance(quality.get("checks"), dict)
    ):
        return ["quality_schema"]

    checks = quality["checks"]
    check_failure = set(checks) != set(_EXPOSURE_CHECK_SEVERITIES)
    error_counts: list[int] = []
    warning_counts: list[int] = []
    if not check_failure:
        for name, item in checks.items():
            severity = _EXPOSURE_CHECK_SEVERITIES[name]
            if not _valid_check(item, severity, allow_positive=severity == "info"):
                check_failure = True
                break
            if severity == "error":
                error_counts.append(item["count"])
            elif severity == "warning":
                warning_counts.append(item["count"])
    failures = ["quality_checks"] if check_failure else []
    derived_status = (
        "error"
        if any(error_counts) or check_failure
        else "warning"
        if any(warning_counts)
        else "pass"
    )
    if (
        quality.get("status") != derived_status
        or quality.get("status") != manifest.get("quality_status")
        or quality.get("status") == "error"
    ):
        failures.append("quality_status")

    summary = quality["summary"]
    artifacts = _artifacts_by_name(manifest)
    expected_summary_counts = {
        "market_cap_count": _partition_row_count(artifacts, "market_cap/"),
        "industry_definition_count": _artifact_row_count(
            artifacts, "industry_definition.parquet"
        ),
        "industry_membership_count": _artifact_row_count(
            artifacts, "industry_membership.parquet"
        ),
        "industry_membership_pretest_count": _artifact_row_count(
            artifacts, "industry_membership_pretest.parquet"
        ),
    }
    expected_observations = summary.get("expected_observation_count")
    observed_observations = summary.get("observed_observation_count")
    expected_security_count = summary.get("expected_security_count")
    expected_industry_count = summary.get("expected_industry_count")
    reported_ratio = summary.get("temporal_coverage_ratio")
    coverage_scope = manifest.get("coverage_scope")
    coverage_minimum = (
        coverage_scope.get("minimum_temporal_coverage")
        if isinstance(coverage_scope, dict)
        else None
    )
    industry_coverage_minimum = (
        coverage_scope.get("minimum_industry_observation_coverage")
        if isinstance(coverage_scope, dict)
        else None
    )
    derived_ratio = (
        observed_observations / expected_observations
        if _nonnegative_int(expected_observations)
        and expected_observations > 0
        and _nonnegative_int(observed_observations)
        else 1.0
    )
    count_keys = (
        "expected_security_count",
        "expected_industry_count",
        "expected_observation_count",
        "observed_observation_count",
        "industry_expected_observation_count",
        "industry_matched_observation_count",
        "industry_missing_observation_count",
        "missing_industry_security_count",
    )
    if (
        set(summary)
        != {
            *expected_summary_counts,
            *count_keys,
            "temporal_coverage_ratio",
            "minimum_temporal_coverage",
            "explicit_empty_industry_count",
            "industry_observation_coverage_ratio",
            "minimum_industry_observation_coverage",
            "missing_industry_security_ids",
            "historical_taxonomy_bridge_count",
        }
        or any(
            summary.get(key) != value for key, value in expected_summary_counts.items()
        )
        or any(not _nonnegative_int(summary.get(key)) for key in count_keys)
        or not _unit_number(summary.get("temporal_coverage_ratio"))
        or not _unit_number(summary.get("minimum_temporal_coverage"))
        or not _unit_number(summary.get("industry_observation_coverage_ratio"))
        or not _unit_number(summary.get("minimum_industry_observation_coverage"))
        or summary.get("minimum_temporal_coverage") != coverage_minimum
        or summary.get("minimum_industry_observation_coverage")
        != industry_coverage_minimum
        or (
            _nonnegative_int(expected_industry_count)
            and expected_industry_count
            > expected_summary_counts["industry_definition_count"]
        )
        or not _nonnegative_int(summary.get("explicit_empty_industry_count"))
        or (
            _nonnegative_int(observed_observations)
            and observed_observations > expected_summary_counts["market_cap_count"]
        )
        or summary.get("observed_observation_count", 0)
        > summary.get("expected_observation_count", 0)
        or (
            _nonnegative_int(expected_security_count)
            and _nonnegative_int(expected_observations)
            and expected_security_count > expected_observations
        )
        or not isinstance(reported_ratio, (int, float))
        or isinstance(reported_ratio, bool)
        or abs(float(reported_ratio) - derived_ratio) > 1e-12
        or summary.get("industry_missing_observation_count")
        != summary.get("industry_expected_observation_count", 0)
        - summary.get("industry_matched_observation_count", 0)
        or not isinstance(summary.get("missing_industry_security_ids"), list)
        or summary.get("missing_industry_security_ids")
        != sorted(set(summary.get("missing_industry_security_ids", [])))
        or any(
            not isinstance(value, str)
            for value in summary.get("missing_industry_security_ids", [])
        )
        or summary.get("missing_industry_security_count")
        != len(summary.get("missing_industry_security_ids", []))
        or not _nonnegative_int(summary.get("historical_taxonomy_bridge_count"))
        or summary.get("historical_taxonomy_bridge_count", 0)
        > expected_summary_counts["industry_membership_count"]
        or abs(
            float(summary.get("industry_observation_coverage_ratio", -1))
            - (
                summary.get("industry_matched_observation_count", 0)
                / summary.get("industry_expected_observation_count", 1)
                if summary.get("industry_expected_observation_count", 0)
                else 1.0
            )
        )
        > 1e-12
    ):
        failures.append("quality_row_counts")
    return failures


def _valid_check(value: object, severity: str, *, allow_positive: bool = False) -> bool:
    if not isinstance(value, dict) or set(value) != {"severity", "status", "count"}:
        return False
    count = value.get("count")
    return (
        value.get("severity") == severity
        and _nonnegative_int(count)
        and value.get("status") == ("pass" if allow_positive or count == 0 else "fail")
    )


def _artifacts_by_name(manifest: dict[str, Any]) -> dict[object, dict[str, Any]]:
    return {
        item.get("name"): item
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict)
    }


def _artifact_row_count(artifacts: dict[object, dict[str, Any]], name: str) -> int:
    artifact = artifacts.get(name)
    if not isinstance(artifact, dict):
        return -1
    value = artifact.get("row_count")
    return cast(int, value) if _nonnegative_int(value) else -1


def _partition_row_count(artifacts: dict[object, dict[str, Any]], prefix: str) -> int:
    counts = [
        item.get("row_count")
        for name, item in artifacts.items()
        if isinstance(name, str) and name.startswith(prefix)
    ]
    if not counts or any(not _nonnegative_int(value) for value in counts):
        return -1
    total = 0
    for value in counts:
        total += cast(int, value)
    return total


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _unit_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0 <= value <= 1
    )
