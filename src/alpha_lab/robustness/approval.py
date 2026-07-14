from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import yaml

from alpha_lab.database import catalog
from alpha_lab.evaluation.config import load_evaluation_config
from alpha_lab.factors.contract import FactorMetadata
from alpha_lab.robustness import report as robustness_report
from alpha_lab.robustness.config import load_robustness_config
from alpha_lab.robustness.freeze import validate_freeze
from alpha_lab.robustness.report import _markdown
from alpha_lab.robustness.walk_forward import evaluate_gates

REQUEST_SCHEMA_VERSION = 1
APPROVAL_SCHEMA_VERSION = 1
_GATE_NAMES = {
    "direction_consistency",
    "fold_coverage",
    "double_cost_direction",
    "industry_neutral_retention",
}
_ROBUSTNESS_ARTIFACTS = (
    "walk_forward.json",
    "cost_sensitivity.json",
    "exposure_report.json",
    "robustness_report.md",
)
_EXECUTION_CONFIGS = (
    "baseline.yaml",
    "costs.yaml",
    "factor_evaluation.yaml",
    "factor_registry.yaml",
    "robustness.yaml",
    "splits.yaml",
)
_EXECUTION_ROOT_RESOURCES = (
    "Dockerfile",
    "compose.yaml",
    "pyproject.toml",
    "uv.lock",
)


def create_test_request(freeze_path: Path) -> Path:
    freeze_dir = freeze_path.parent
    if freeze_dir.is_symlink():
        raise ValueError("freeze directory is untrusted")
    repo_root = freeze_path.parents[3]
    freeze = validate_freeze(freeze_path, repo_root / "config", repo_root / "data")
    if freeze_path.name != "freeze.json" or freeze.get("freeze_id") != freeze_dir.name:
        raise ValueError("freeze layout or identity is invalid")
    expected_freeze = (
        repo_root / "experiments" / "phase6" / str(freeze["freeze_id"]) / "freeze.json"
    )
    if freeze_path != expected_freeze:
        raise ValueError("freeze path is outside the repository experiments root")
    freeze_sha256 = _sha256(freeze_path)
    artifacts = {
        name: {
            "path": name,
            "sha256": _sha256(_required_regular_file(freeze_dir / name, freeze_dir)),
        }
        for name in _ROBUSTNESS_ARTIFACTS
    }
    gates = _validate_robustness_artifacts(
        freeze_dir, freeze, freeze_sha256, repo_root / "config"
    )
    _replay_pretest_evidence(freeze_path, repo_root)
    test = freeze.get("test")
    if test != {
        "access": "human_approval_only",
        "start": "2026-01-01",
        "end": "2026-07-11",
    }:
        raise ValueError("freeze locked-test boundary is invalid")
    identity = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "status": "test_requested",
        "freeze_id": freeze["freeze_id"],
        "freeze_sha256": freeze_sha256,
        "robustness_artifacts": artifacts,
        "execution_bundle": _execution_bundle(repo_root, freeze),
        "gates": gates,
        "locked_test": test,
    }
    digest = _digest(identity)
    document = {**identity, "request_id": f"request-{digest}"}
    path = freeze_dir / "test_request.json"
    _write_immutable(path, canonical_bytes(document), "test request")
    _register_test_request(
        repo_root / "data" / "metadata.duckdb", freeze, document, path
    )
    return path


def approve_test_request(
    request_path: Path, approver: str, confirmed_freeze_sha256: str
) -> Path:
    _validate_approver(approver)
    request = validate_test_request(request_path)
    _validate_request_files(request_path, request)
    repo_root = request_path.parents[3]
    expected_request = (
        repo_root
        / "experiments"
        / "phase6"
        / str(request["freeze_id"])
        / "test_request.json"
    )
    if request_path != expected_request:
        raise ValueError("test request is outside the repository experiments root")
    validate_execution_bundle_files(request, repo_root)
    if confirmed_freeze_sha256 != request["freeze_sha256"]:
        raise PermissionError("freeze confirmation does not match the test request")
    approval_dir = request_path.parent / "approvals"
    if approval_dir.is_symlink():
        raise ValueError("approval directory is untrusted")
    if approval_dir.exists():
        for current in sorted(approval_dir.glob("*.json")):
            try:
                document = validate_approval(current)
            except ValueError as error:
                raise RuntimeError(f"immutable approval differs: {current}") from error
            if (
                document["request_id"] == request["request_id"]
                and document["request_sha256"] == _sha256(request_path)
                and document["freeze_sha256"] == confirmed_freeze_sha256
                and document["approver"] == approver
            ):
                _register_test_approval(
                    repo_root / "data" / "metadata.duckdb",
                    request,
                    document,
                    current,
                )
                return current
    identity = {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "status": "approved",
        "request_id": request["request_id"],
        "request_sha256": _sha256(request_path),
        "freeze_id": request["freeze_id"],
        "freeze_sha256": confirmed_freeze_sha256,
        "approver": approver,
        "approved_at": _utc_now(),
    }
    digest = _digest(identity)
    document = {**identity, "approval_id": f"approval-{digest}"}
    path = approval_dir / f"approval-{digest}.json"
    _write_immutable(path, canonical_bytes(document), "approval")
    _register_test_approval(
        repo_root / "data" / "metadata.duckdb", request, document, path
    )
    return path


def validate_test_request(path: Path) -> dict[str, Any]:
    document = _read_canonical_json(path, "test request")
    required = {
        "schema_version",
        "status",
        "request_id",
        "freeze_id",
        "freeze_sha256",
        "robustness_artifacts",
        "execution_bundle",
        "gates",
        "locked_test",
    }
    identity = {key: value for key, value in document.items() if key != "request_id"}
    if (
        set(document) != required
        or document.get("schema_version") != REQUEST_SCHEMA_VERSION
        or document.get("status") != "test_requested"
        or document.get("request_id") != f"request-{_digest(identity)}"
        or not _freeze_id(document.get("freeze_id"))
        or not _sha(document.get("freeze_sha256"))
        or document.get("locked_test")
        != {
            "access": "human_approval_only",
            "start": "2026-01-01",
            "end": "2026-07-11",
        }
        or not isinstance(document.get("gates"), dict)
        or set(document["gates"]) != _GATE_NAMES
        or any(value is not True for value in document["gates"].values())
    ):
        raise ValueError("test request schema or identity is invalid")
    artifacts = document.get("robustness_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(_ROBUSTNESS_ARTIFACTS):
        raise ValueError("test request robustness artifacts are invalid")
    for name, reference in artifacts.items():
        if (
            not isinstance(reference, dict)
            or set(reference) != {"path", "sha256"}
            or reference.get("path") != name
            or not _sha(reference.get("sha256"))
        ):
            raise ValueError("test request robustness artifact reference is invalid")
    _validate_execution_bundle_schema(document.get("execution_bundle"))
    if path.name != "test_request.json" or path.parent.name != document["freeze_id"]:
        raise ValueError("test request path does not match its freeze identity")
    return document


def validate_approval(path: Path) -> dict[str, Any]:
    document = _read_canonical_json(path, "approval")
    required = {
        "schema_version",
        "status",
        "approval_id",
        "request_id",
        "request_sha256",
        "freeze_id",
        "freeze_sha256",
        "approver",
        "approved_at",
    }
    identity = {key: value for key, value in document.items() if key != "approval_id"}
    if (
        set(document) != required
        or document.get("schema_version") != APPROVAL_SCHEMA_VERSION
        or document.get("status") != "approved"
        or document.get("approval_id") != f"approval-{_digest(identity)}"
        or not re.fullmatch(r"request-[0-9a-f]{64}", str(document.get("request_id")))
        or not _sha(document.get("request_sha256"))
        or not _freeze_id(document.get("freeze_id"))
        or not _sha(document.get("freeze_sha256"))
        or not _timestamp(document.get("approved_at"))
    ):
        raise ValueError("approval schema or identity is invalid")
    _validate_approver(document.get("approver"))
    if (
        path.name != f"{document['approval_id']}.json"
        or path.parent.name != "approvals"
        or path.parent.parent.name != document["freeze_id"]
    ):
        raise ValueError("approval path does not match its identity")
    return document


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def _read_canonical_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or untrusted")
    try:
        content = path.read_bytes()
        value = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is malformed") from error
    try:
        canonical = canonical_bytes(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} contains unsupported values") from error
    if not isinstance(value, dict) or content != canonical:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_immutable(path: Path, content: bytes, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError(f"immutable {label} differs: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise RuntimeError(f"immutable {label} differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise RuntimeError(f"immutable {label} differs: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _required_regular_file(path: Path, root: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required robustness artifact is missing: {path.name}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("robustness artifact escapes freeze directory") from error
    return path


def _validate_request_files(path: Path, request: dict[str, Any]) -> None:
    freeze_dir = path.parent
    freeze_path = freeze_dir / "freeze.json"
    if (
        freeze_path.is_symlink()
        or not freeze_path.is_file()
        or _sha256(freeze_path) != request["freeze_sha256"]
    ):
        raise ValueError("test request freeze hash has drifted")
    for name, reference in request["robustness_artifacts"].items():
        artifact = freeze_dir / name
        if (
            artifact.is_symlink()
            or not artifact.is_file()
            or _sha256(artifact) != reference["sha256"]
        ):
            raise ValueError(f"test request robustness hash has drifted: {name}")


def _validate_robustness_artifacts(
    freeze_dir: Path,
    freeze: dict[str, Any],
    freeze_sha256: str,
    config_dir: Path,
) -> dict[str, bool]:
    walk = _read_canonical_json(freeze_dir / "walk_forward.json", "walk-forward")
    costs = _read_canonical_json(
        freeze_dir / "cost_sensitivity.json", "cost sensitivity"
    )
    exposure = _read_canonical_json(
        freeze_dir / "exposure_report.json", "exposure report"
    )
    for label, document in (
        ("walk-forward", walk),
        ("cost sensitivity", costs),
        ("exposure report", exposure),
    ):
        _require_finite_numbers(document, label)
    common_keys = {
        "schema_version",
        "freeze_id",
        "factor_id",
        "freeze_sha256",
        "robustness_policy_sha256",
        "evaluation_policy_sha256",
        "dependencies",
        "orientation",
        "test_accessed",
    }
    if set(walk) != common_keys | {"folds", "gates", "passed"}:
        raise ValueError("walk-forward report schema is invalid")
    if set(costs) != common_keys | {"scenarios"}:
        raise ValueError("cost-sensitivity report schema is invalid")
    if set(exposure) != common_keys | {"size", "industry", "missing"}:
        raise ValueError("exposure report schema is invalid")
    common = {key: walk[key] for key in common_keys}
    if any(
        {key: document[key] for key in common_keys} != common
        for document in (costs, exposure)
    ):
        raise ValueError("robustness report dependency linkage is inconsistent")
    factor = freeze.get("factor", {})
    snapshots = freeze.get("snapshots", {})
    policies = freeze.get("policies", {})
    expected_dependencies = {
        "phase5_manifest_sha256": snapshots["phase5"]["manifest_sha256"],
        "exposure_manifest_sha256": snapshots["exposure"]["manifest_sha256"],
        "cost_policy_sha256": policies["costs"]["sha256"],
        "factor_source_sha256": factor["source_sha256"],
        "factor_metadata_sha256": factor["metadata_sha256"],
    }
    if (
        common["schema_version"] != 1
        or common["freeze_id"] != freeze.get("freeze_id")
        or common["freeze_sha256"] != freeze_sha256
        or common["factor_id"] != factor.get("factor_id")
        or common["robustness_policy_sha256"] != policies["robustness"]["sha256"]
        or common["dependencies"] != expected_dependencies
        or common["evaluation_policy_sha256"]
        != load_evaluation_config(config_dir / "factor_evaluation.yaml")[1]
        or common["orientation"]
        != {
            "candidate_direction": -1,
            "score_formula": "score=standardize(winsorize(value*direction))",
            "direction_consistency_source": "oriented_mean_rank_ic_positive",
        }
        or common["test_accessed"] is not False
    ):
        raise ValueError("robustness report freeze or dependency link is invalid")
    config, policy_sha256 = load_robustness_config(config_dir / "robustness.yaml")
    if policy_sha256 != common["robustness_policy_sha256"]:
        raise ValueError("robustness report policy has drifted")
    folds = walk["folds"]
    expected_folds = config.walk_forward_folds
    if not isinstance(folds, list) or len(folds) != len(expected_folds):
        raise ValueError("walk-forward report must contain five folds")
    fold_keys = {
        "fold_id",
        "start",
        "end",
        "input_row_count",
        "valid_row_count",
        "coverage",
        "mean_ic",
        "mean_rank_ic",
        "icir",
        "rank_icir",
        "group_returns",
        "factor_turnover",
        "direction_consistent",
    }
    for actual, expected in zip(folds, expected_folds, strict=True):
        input_rows = actual.get("input_row_count") if isinstance(actual, dict) else None
        valid_rows = actual.get("valid_row_count") if isinstance(actual, dict) else None
        expected_coverage = (
            float(valid_rows / input_rows)
            if _nonnegative_int(input_rows)
            and _nonnegative_int(valid_rows)
            and input_rows > 0
            and valid_rows <= input_rows
            else None
        )
        if (
            not isinstance(actual, dict)
            or set(actual) != fold_keys
            or actual["fold_id"] != expected.fold_id
            or actual["start"] != expected.start.isoformat()
            or actual["end"] != expected.end.isoformat()
            or expected_coverage is None
            or not _finite_number(actual["coverage"])
            or not math.isclose(
                float(actual["coverage"]), expected_coverage, rel_tol=0.0, abs_tol=1e-15
            )
            or any(
                actual[key] is not None and not _finite_number(actual[key])
                for key in (
                    "mean_ic",
                    "mean_rank_ic",
                    "icir",
                    "rank_icir",
                    "factor_turnover",
                )
            )
            or not isinstance(actual["group_returns"], dict)
            or not actual["group_returns"]
            or any(
                not isinstance(key, str)
                or not key.isdigit()
                or not _finite_number(value)
                for key, value in actual["group_returns"].items()
            )
            or actual["direction_consistent"]
            is not (
                actual["mean_rank_ic"] is not None and float(actual["mean_rank_ic"]) > 0
            )
        ):
            raise ValueError("walk-forward fold diagnostics are invalid")
    scenarios = costs["scenarios"]
    expected_scenarios = {str(float(value)) for value in config.cost_multipliers}
    if not isinstance(scenarios, dict) or set(scenarios) != expected_scenarios:
        raise ValueError("cost-sensitivity scenarios are invalid")
    metric_keys = {
        "initial_cash",
        "final_nav",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "trade_count",
        "total_fees",
        "turnover_ratio",
    }
    constraint_keys = {
        "blocked_suspend",
        "blocked_limit",
        "blocked_st",
        "blocked_unknown_status",
        "blocked_t_plus_one",
        "blocked_lot_or_cash",
        "inferred_price_limit_checks",
    }
    for scenario in scenarios.values():
        if (
            not isinstance(scenario, dict)
            or set(scenario) != {"metrics", "folds"}
            or not isinstance(scenario["metrics"], dict)
            or set(scenario["metrics"]) != {"total_return"}
            or not _finite_number(scenario["metrics"]["total_return"])
            or not isinstance(scenario["folds"], list)
            or any(
                not isinstance(item, dict)
                or set(item) != {"fold_id", "metrics", "constraints"}
                or not isinstance(item["metrics"], dict)
                or set(item["metrics"]) != metric_keys
                or any(
                    value is not None and not _finite_number(value)
                    for value in item["metrics"].values()
                )
                or not isinstance(item["constraints"], dict)
                or set(item["constraints"]) != constraint_keys
                or any(
                    not _nonnegative_int(value)
                    for value in item["constraints"].values()
                )
                for item in scenario["folds"]
            )
            or [item.get("fold_id") for item in scenario["folds"]]
            != [fold.fold_id for fold in expected_folds]
        ):
            raise ValueError("cost-sensitivity diagnostics are invalid")
        aggregate = 1.0
        for item in scenario["folds"]:
            total_return = item["metrics"]["total_return"]
            if not _finite_number(total_return):
                raise ValueError("cost-sensitivity fold return is invalid")
            aggregate *= 1.0 + float(total_return)
        if not math.isclose(
            float(scenario["metrics"]["total_return"]),
            aggregate - 1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("cost-sensitivity aggregate is inconsistent")
    if not isinstance(exposure["industry"], dict) or set(exposure["industry"]) != {
        "joined_rows",
        "original_joined_rows",
        "original_rank_ic",
        "neutral_rank_ic",
        "abs_rank_ic_retention",
        "minimum_group_size",
        "by_industry",
        "mean_score_dispersion",
    }:
        raise ValueError("exposure diagnostics are invalid")
    if not isinstance(exposure["size"], dict) or set(exposure["size"]) != {
        "joined_rows",
        "correlation",
        "risk_threshold",
        "risk_flag",
        "uses",
        "method",
        "daily",
        "yearly",
    }:
        raise ValueError("exposure diagnostics are invalid")
    if not isinstance(exposure["missing"], dict) or set(exposure["missing"]) != {
        "size_rows",
        "industry_rows",
    }:
        raise ValueError("exposure diagnostics are invalid")
    _validate_exposure_details(exposure)
    derived = evaluate_gates(folds, costs, exposure, config)
    if (
        walk["gates"] != derived
        or walk["passed"] is not all(derived.values())
        or not all(derived.values())
    ):
        raise PermissionError("robustness gate has not passed for this freeze")
    markdown = freeze_dir / "robustness_report.md"
    if markdown.read_bytes() != _markdown(walk, exposure):
        raise ValueError("robustness Markdown is not derived from bound reports")
    return derived


def _validate_exposure_details(exposure: dict[str, Any]) -> None:
    size = exposure["size"]
    industry = exposure["industry"]
    missing = exposure["missing"]
    if (
        not all(_nonnegative_int(size[key]) for key in ("joined_rows",))
        or not all(
            _nonnegative_int(missing[key]) for key in ("size_rows", "industry_rows")
        )
        or size["uses"] != "log(total_market_cap_cny)"
        or size["method"] != "daily_cross_sectional_spearman"
        or not _finite_number(size["risk_threshold"])
        or size["correlation"] is not None
        and not _finite_number(size["correlation"])
        or size["risk_flag"]
        is not (
            size["correlation"] is not None
            and abs(float(size["correlation"])) > float(size["risk_threshold"])
        )
        or not isinstance(size["daily"], list)
        or any(
            not isinstance(item, dict)
            or set(item) != {"date", "correlation", "count"}
            or not isinstance(item["date"], str)
            or not _finite_number(item["correlation"])
            or not _nonnegative_int(item["count"])
            for item in size["daily"]
        )
        or not isinstance(size["yearly"], dict)
        or any(
            not str(key).isdigit() or not _finite_number(value)
            for key, value in size["yearly"].items()
        )
    ):
        raise ValueError("size exposure diagnostics are invalid")
    numeric_optional = (
        "original_rank_ic",
        "neutral_rank_ic",
        "abs_rank_ic_retention",
        "mean_score_dispersion",
    )
    if (
        not all(
            _nonnegative_int(industry[key])
            for key in ("joined_rows", "original_joined_rows", "minimum_group_size")
        )
        or industry["joined_rows"] != industry["original_joined_rows"]
        or any(
            industry[key] is not None and not _finite_number(industry[key])
            for key in numeric_optional
        )
        or not isinstance(industry["by_industry"], list)
        or any(
            not isinstance(item, dict)
            or set(item) != {"industry_id", "mean_score", "observations"}
            or not isinstance(item["industry_id"], str)
            or not item["industry_id"]
            or not _finite_number(item["mean_score"])
            or not _nonnegative_int(item["observations"])
            for item in industry["by_industry"]
        )
    ):
        raise ValueError("industry exposure diagnostics are invalid")
    original = industry["original_rank_ic"]
    neutral = industry["neutral_rank_ic"]
    expected_retention = (
        abs(float(neutral)) / abs(float(original))
        if original not in (None, 0.0) and neutral is not None
        else None
    )
    actual_retention = industry["abs_rank_ic_retention"]
    if (
        expected_retention is None
        and actual_retention is not None
        or expected_retention is not None
        and (
            actual_retention is None
            or not math.isclose(
                float(actual_retention),
                expected_retention,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
    ):
        raise ValueError("industry-neutral retention is inconsistent")


def _require_finite_numbers(value: object, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for item in value.values():
            _require_finite_numbers(item, label)
    elif isinstance(value, list):
        for item in value:
            _require_finite_numbers(item, label)


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _execution_bundle(repo_root: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    del freeze
    config_root = repo_root / "config"
    config = {
        f"config/{name}": _sha256(
            _required_regular_file(config_root / name, config_root)
        )
        for name in sorted(_EXECUTION_CONFIGS)
    }
    paths = _execution_code_paths(repo_root)
    code = {
        name: _sha256(_required_regular_file(repo_root / name, repo_root))
        for name in paths
    }
    resource_paths = _execution_resource_paths(repo_root)
    resources = {
        name: _sha256(_required_regular_file(repo_root / name, repo_root))
        for name in resource_paths
    }
    return {"configs": config, "code": code, "resources": resources}


def _execution_code_paths(repo_root: Path) -> list[str]:
    source_root = repo_root / "src" / "alpha_lab"
    return sorted(
        path.relative_to(repo_root).as_posix() for path in source_root.rglob("*.py")
    )


def _execution_resource_paths(repo_root: Path) -> list[str]:
    sql_root = repo_root / "src" / "alpha_lab" / "database" / "sql"
    package_resources = [
        path.relative_to(repo_root).as_posix()
        for pattern_root, suffix in ((sql_root, "*.sql"), (repo_root / "src", "*.yaml"))
        for path in pattern_root.rglob(suffix)
    ]
    return sorted([*package_resources, *_EXECUTION_ROOT_RESOURCES])


def _validate_execution_bundle_schema(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "configs",
        "code",
        "resources",
    }:
        raise ValueError("test request execution bundle is invalid")
    for section in ("configs", "code", "resources"):
        items = value[section]
        if (
            not isinstance(items, dict)
            or not items
            or list(items) != sorted(items)
            or any(
                not isinstance(path, str) or not _sha(sha)
                for path, sha in items.items()
            )
        ):
            raise ValueError("test request execution bundle is invalid")


def validate_execution_bundle_files(request: dict[str, Any], repo_root: Path) -> None:
    bundle = request["execution_bundle"]
    _validate_execution_bundle_schema(bundle)
    expected_configs = {f"config/{name}" for name in _EXECUTION_CONFIGS}
    if set(bundle["configs"]) != expected_configs:
        raise ValueError("test request execution config bundle is incomplete")
    expected_code = set(_execution_code_paths(repo_root))
    if set(bundle["code"]) != expected_code:
        raise ValueError("test request execution code bundle is incomplete")
    expected_resources = set(_execution_resource_paths(repo_root))
    if set(bundle["resources"]) != expected_resources:
        raise ValueError("test request execution resource bundle is incomplete")
    for section in ("configs", "code", "resources"):
        for relative, expected_sha256 in bundle[section].items():
            path = _required_regular_file(repo_root / relative, repo_root)
            if _sha256(path) != expected_sha256:
                raise ValueError(f"test request execution bundle drift: {relative}")


def _replay_pretest_evidence(freeze_path: Path, repo_root: Path) -> None:
    experiments_root = repo_root / "experiments"
    experiments_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".pretest-replay-", dir=experiments_root
    ) as temporary_name:
        replay_experiments = Path(temporary_name)
        replay_dir = replay_experiments / "phase6" / freeze_path.parent.name
        replay_dir.mkdir(parents=True)
        replay_freeze = replay_dir / "freeze.json"
        shutil.copyfile(freeze_path, replay_freeze)
        robustness_report.evaluate_frozen_candidate(
            replay_freeze,
            repo_root / "config",
            repo_root / "data",
            replay_experiments,
        )
        for name in _ROBUSTNESS_ARTIFACTS:
            original = _required_regular_file(
                freeze_path.parent / name, freeze_path.parent
            )
            replayed = _required_regular_file(replay_dir / name, replay_dir)
            original_bytes = original.read_bytes()
            replayed_bytes = replayed.read_bytes()
            if (
                original_bytes != replayed_bytes
                or hashlib.sha256(original_bytes).hexdigest()
                != hashlib.sha256(replayed_bytes).hexdigest()
            ):
                raise ValueError(f"robustness evidence replay mismatch: {name}")


def _register_test_request(
    database_path: Path,
    freeze: dict[str, Any],
    request: dict[str, Any],
    request_path: Path,
) -> None:
    factor = freeze["factor"]
    repo_root = request_path.parents[3]
    metadata = FactorMetadata.model_validate(
        yaml.safe_load(
            (repo_root / factor["metadata_path"]).read_text(encoding="utf-8")
        )
    )
    version_id = f"{factor['factor_id']}-{str(factor['source_sha256'])[:20]}"
    with catalog._catalog_write_lock(database_path):
        catalog.initialize_database(database_path)
        with duckdb.connect(str(database_path)) as connection:
            connection.execute("BEGIN TRANSACTION")
            try:
                connection.execute(
                    """
                    INSERT INTO research.factor_definition
                        (factor_id, name, family, description)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (factor_id) DO NOTHING
                    """,
                    [
                        factor["factor_id"],
                        metadata.name,
                        metadata.family,
                        metadata.hypothesis,
                    ],
                )
                connection.execute(
                    """
                    INSERT INTO research.factor_version
                        (factor_version_id, factor_id, formula,
                         implementation_path, code_sha256, metadata_sha256,
                         lookback, direction)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (factor_version_id) DO NOTHING
                    """,
                    [
                        version_id,
                        factor["factor_id"],
                        metadata.formula,
                        factor["source_path"],
                        factor["source_sha256"],
                        factor["metadata_sha256"],
                        metadata.lookback,
                        metadata.direction,
                    ],
                )
                version_rows = connection.execute(
                    """
                    SELECT factor_id, formula, implementation_path,
                           code_sha256, metadata_sha256, lookback, direction
                    FROM research.factor_version WHERE factor_version_id = ?
                    """,
                    [version_id],
                ).fetchall()
                expected_version = (
                    factor["factor_id"],
                    metadata.formula,
                    factor["source_path"],
                    factor["source_sha256"],
                    factor["metadata_sha256"],
                    metadata.lookback,
                    metadata.direction,
                )
                if len(version_rows) != 1 or tuple(version_rows[0]) != expected_version:
                    raise RuntimeError("factor version catalog registration conflict")
                manifest_artifact_id = hashlib.sha256(
                    f"freeze|{freeze['freeze_id']}|{request['freeze_sha256']}".encode()
                ).hexdigest()
                connection.execute(
                    """
                    INSERT INTO research.factor_freeze
                        (freeze_id, freeze_sha256, factor_version_id,
                         phase5_snapshot_id, exposure_snapshot_id,
                         robustness_policy_sha256, cost_policy_sha256,
                         code_commit, test_start, test_end, manifest_artifact_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS DATE),
                            CAST(? AS DATE), ?)
                    ON CONFLICT (freeze_id) DO NOTHING
                    """,
                    [
                        freeze["freeze_id"],
                        request["freeze_sha256"],
                        version_id,
                        freeze["snapshots"]["phase5"]["snapshot_id"],
                        freeze["snapshots"]["exposure"]["snapshot_id"],
                        freeze["policies"]["robustness"]["sha256"],
                        freeze["policies"]["costs"]["sha256"],
                        freeze["git_commit"],
                        request["locked_test"]["start"],
                        request["locked_test"]["end"],
                        manifest_artifact_id,
                    ],
                )
                freeze_rows = connection.execute(
                    """
                    SELECT freeze_sha256, factor_version_id,
                           phase5_snapshot_id, exposure_snapshot_id,
                           robustness_policy_sha256, cost_policy_sha256,
                           code_commit, CAST(test_start AS VARCHAR),
                           CAST(test_end AS VARCHAR), status
                    FROM research.factor_freeze WHERE freeze_id = ?
                    """,
                    [freeze["freeze_id"]],
                ).fetchall()
                expected_freeze = (
                    request["freeze_sha256"],
                    version_id,
                    freeze["snapshots"]["phase5"]["snapshot_id"],
                    freeze["snapshots"]["exposure"]["snapshot_id"],
                    freeze["policies"]["robustness"]["sha256"],
                    freeze["policies"]["costs"]["sha256"],
                    freeze["git_commit"],
                    request["locked_test"]["start"],
                    request["locked_test"]["end"],
                    "frozen",
                )
                if len(freeze_rows) != 1 or tuple(freeze_rows[0]) != expected_freeze:
                    raise RuntimeError("factor freeze catalog registration conflict")
                connection.execute(
                    """
                    INSERT INTO research.test_request
                        (request_id, request_sha256, freeze_id, freeze_sha256,
                         robustness_report_sha256, test_start, test_end)
                    VALUES (?, ?, ?, ?, ?, CAST(? AS DATE), CAST(? AS DATE))
                    ON CONFLICT (request_id) DO NOTHING
                    """,
                    [
                        request["request_id"],
                        _sha256(request_path),
                        request["freeze_id"],
                        request["freeze_sha256"],
                        request["robustness_artifacts"]["robustness_report.md"][
                            "sha256"
                        ],
                        request["locked_test"]["start"],
                        request["locked_test"]["end"],
                    ],
                )
                _require_registered_request(connection, request, request_path)
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise


def _register_test_approval(
    database_path: Path,
    request: dict[str, Any],
    approval: dict[str, Any],
    approval_path: Path,
) -> None:
    with catalog._catalog_write_lock(database_path):
        catalog.initialize_database(database_path)
        with duckdb.connect(str(database_path)) as connection:
            connection.execute("BEGIN TRANSACTION")
            try:
                connection.execute(
                    """
                    INSERT INTO research.test_approval
                        (approval_id, approval_sha256, request_id, freeze_id,
                         confirmed_freeze_sha256, test_start, test_end,
                         approver, approved_at)
                    VALUES (?, ?, ?, ?, ?, CAST(? AS DATE), CAST(? AS DATE),
                            ?, CAST(? AS TIMESTAMPTZ))
                    ON CONFLICT (approval_id) DO NOTHING
                    """,
                    [
                        approval["approval_id"],
                        _sha256(approval_path),
                        request["request_id"],
                        request["freeze_id"],
                        request["freeze_sha256"],
                        request["locked_test"]["start"],
                        request["locked_test"]["end"],
                        approval["approver"],
                        approval["approved_at"],
                    ],
                )
                row = connection.execute(
                    """
                    SELECT approval_sha256, request_id, freeze_id,
                           confirmed_freeze_sha256,
                           CAST(test_start AS VARCHAR), CAST(test_end AS VARCHAR),
                           approver, status
                    FROM research.test_approval WHERE approval_id = ?
                    """,
                    [approval["approval_id"]],
                ).fetchall()
                expected = (
                    _sha256(approval_path),
                    request["request_id"],
                    request["freeze_id"],
                    request["freeze_sha256"],
                    request["locked_test"]["start"],
                    request["locked_test"]["end"],
                    approval["approver"],
                    "approved",
                )
                if len(row) != 1 or tuple(row[0]) != expected:
                    raise RuntimeError("test approval catalog registration conflict")
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise


def _require_registered_request(
    connection: duckdb.DuckDBPyConnection,
    request: dict[str, Any],
    request_path: Path,
) -> None:
    row = connection.execute(
        """
        SELECT request_sha256, freeze_id, freeze_sha256,
               robustness_report_sha256, CAST(test_start AS VARCHAR),
               CAST(test_end AS VARCHAR), status
        FROM research.test_request WHERE request_id = ?
        """,
        [request["request_id"]],
    ).fetchall()
    expected = (
        _sha256(request_path),
        request["freeze_id"],
        request["freeze_sha256"],
        request["robustness_artifacts"]["robustness_report.md"]["sha256"],
        request["locked_test"]["start"],
        request["locked_test"]["end"],
        "test_requested",
    )
    if len(row) != 1 or tuple(row[0]) != expected:
        raise RuntimeError("test request catalog registration conflict")


def _validate_approver(value: object) -> None:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or len(value) > 128
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("approver must be a non-empty printable identity")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _freeze_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"freeze-[0-9a-f]{64}", value) is not None
    )
