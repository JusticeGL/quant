from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import pytest

from alpha_lab.evaluation.config import load_evaluation_config
from alpha_lab.robustness import final_test
from alpha_lab.robustness.approval import (
    approve_test_request,
    create_test_request,
    validate_approval,
    validate_test_request,
)
from alpha_lab.robustness.config import load_robustness_config
from alpha_lab.robustness.freeze import _cost_policy_sha256
from alpha_lab.robustness.report import _markdown

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _isolate_full_freeze_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep approval unit tests focused while preserving freeze identity checks."""

    def validate(path: Path, _config: Path, _data: Path) -> dict[str, Any]:
        document = json.loads(path.read_text(encoding="utf-8"))
        payload = {
            key: value
            for key, value in document.items()
            if key not in {"freeze_id", "identity_sha256"}
        }
        identity = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            document.get("identity_sha256") != identity
            or document.get("freeze_id") != f"freeze-{identity}"
            or path.parent.name != f"freeze-{identity}"
        ):
            raise ValueError("freeze identity mismatch")
        return {**document, "healthy": True, "freeze_sha256": _sha256(path)}

    monkeypatch.setattr("alpha_lab.robustness.approval.validate_freeze", validate)


def test_request_rejects_invalid_freeze_identity(tmp_path: Path) -> None:
    freeze = _freeze_fixture(tmp_path)
    document = json.loads(freeze.read_text(encoding="utf-8"))
    document["identity_sha256"] = "0" * 64
    _write_json(freeze, document)
    with pytest.raises(ValueError, match="freeze identity"):
        create_test_request(freeze)


def test_request_requires_all_pretest_gates(tmp_path: Path) -> None:
    freeze = _freeze_fixture(tmp_path, passed=False)
    with pytest.raises(PermissionError, match="robustness gate"):
        create_test_request(freeze)


@pytest.mark.parametrize(
    ("artifact", "mutation"),
    [
        ("cost_sensitivity.json", lambda _: {}),
        (
            "walk_forward.json",
            lambda value: {
                **value,
                "folds": [{**fold, "coverage": 0.0} for fold in value["folds"]],
            },
        ),
        (
            "exposure_report.json",
            lambda value: {
                **value,
                "industry": {
                    **value["industry"],
                    "abs_rank_ic_retention": 0.0,
                },
            },
        ),
    ],
)
def test_request_rejects_empty_or_self_reported_gate_inputs(
    tmp_path: Path,
    artifact: str,
    mutation: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    freeze = _freeze_fixture(tmp_path)
    path = freeze.parent / artifact
    value = json.loads(path.read_text(encoding="utf-8"))
    _write_json(path, mutation(value))
    with pytest.raises((ValueError, PermissionError)):
        create_test_request(freeze)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_request_rejects_non_finite_cost_gate_evidence(
    tmp_path: Path, invalid: float
) -> None:
    freeze = _freeze_fixture(tmp_path)
    path = freeze.parent / "cost_sensitivity.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["scenarios"]["2.0"]["metrics"]["total_return"] = invalid
    _write_json(path, document)
    with pytest.raises(ValueError, match="unsupported|non-finite"):
        create_test_request(freeze)


@pytest.mark.parametrize(
    ("artifact", "mutate"),
    [
        (
            "walk_forward.json",
            lambda value: value["folds"][0].update({"valid_row_count": 7}),
        ),
        (
            "cost_sensitivity.json",
            lambda value: value["scenarios"]["1.0"]["metrics"].update(
                {"total_return": 0.1}
            ),
        ),
        (
            "exposure_report.json",
            lambda value: value["industry"].update({"abs_rank_ic_retention": 0.9}),
        ),
    ],
)
def test_request_rejects_coherently_republished_inconsistent_evidence(
    tmp_path: Path,
    artifact: str,
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    freeze = _freeze_fixture(tmp_path)
    path = freeze.parent / artifact
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    _write_json(path, document)
    if artifact == "walk_forward.json":
        (freeze.parent / "robustness_report.md").write_bytes(
            _markdown(
                document,
                json.loads(
                    (freeze.parent / "exposure_report.json").read_text(encoding="utf-8")
                ),
            )
        )
    with pytest.raises((ValueError, PermissionError)):
        create_test_request(freeze)


def test_request_is_hash_linked_canonical_and_idempotent(tmp_path: Path) -> None:
    freeze = _freeze_fixture(tmp_path)
    first = create_test_request(freeze)
    original = first.read_bytes()
    second = create_test_request(freeze)
    document = validate_test_request(first)

    assert second == first
    assert first.read_bytes() == original
    assert re.fullmatch(r"request-[0-9a-f]{64}", document["request_id"])
    assert document["status"] == "test_requested"
    assert document["locked_test"] == {
        "access": "human_approval_only",
        "end": "2026-07-11",
        "start": "2026-01-01",
    }
    assert all(document["gates"].values())


def test_request_refuses_existing_conflicting_bytes(tmp_path: Path) -> None:
    freeze = _freeze_fixture(tmp_path)
    path = create_test_request(freeze)
    path.write_text("conflict\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="immutable test request"):
        create_test_request(freeze)


@pytest.mark.parametrize("approver", ["", "   ", "bad\nname", "x" * 129])
def test_approval_rejects_malformed_approver(tmp_path: Path, approver: str) -> None:
    request = create_test_request(_freeze_fixture(tmp_path))
    with pytest.raises(ValueError, match="approver"):
        approve_test_request(request, approver, "a" * 64)


def test_approval_requires_exact_freeze_confirmation(tmp_path: Path) -> None:
    request = create_test_request(_freeze_fixture(tmp_path))
    with pytest.raises(PermissionError, match="freeze confirmation"):
        approve_test_request(request, "Human Reviewer", "0" * 64)


@pytest.mark.parametrize("name", ["freeze.json", "robustness_report.md"])
def test_approval_rejects_request_input_hash_drift(tmp_path: Path, name: str) -> None:
    request = create_test_request(_freeze_fixture(tmp_path))
    confirmation = validate_test_request(request)["freeze_sha256"]
    (request.parent / name).write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash has drifted"):
        approve_test_request(request, "Human Reviewer", confirmation)


@pytest.mark.parametrize(
    "relative",
    [
        "config/factor_evaluation.yaml",
        "src/alpha_lab/evaluation/metrics.py",
        "src/alpha_lab/data/normalize.py",
        "src/alpha_lab/database/catalog.py",
        "src/alpha_lab/database/sql/003_robustness.sql",
        "src/alpha_lab/robustness/approval.py",
        "src/alpha_lab/robustness/final_test.py",
    ],
)
def test_approval_rejects_execution_config_or_code_drift(
    tmp_path: Path, relative: str
) -> None:
    request = create_test_request(_freeze_fixture(tmp_path))
    confirmation = validate_test_request(request)["freeze_sha256"]
    (tmp_path / relative).write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="execution bundle drift"):
        approve_test_request(request, "Human Reviewer", confirmation)


def test_approval_is_hash_linked_and_idempotent(tmp_path: Path) -> None:
    request = create_test_request(_freeze_fixture(tmp_path))
    requested = validate_test_request(request)
    confirmation = requested["freeze_sha256"]
    first = approve_test_request(request, "Human Reviewer", confirmation)
    second = approve_test_request(request, "Human Reviewer", confirmation)
    approval = validate_approval(first)

    assert second == first
    assert approval["status"] == "approved"
    assert approval["request_id"] == requested["request_id"]
    assert approval["request_sha256"] == _sha256(request)
    assert approval["freeze_sha256"] == confirmation
    with duckdb.connect(
        str(tmp_path / "data/metadata.duckdb"), read_only=True
    ) as connection:
        counts = connection.execute(
            """
            SELECT
                (SELECT count(*) FROM research.factor_freeze),
                (SELECT count(*) FROM research.test_request),
                (SELECT count(*) FROM research.test_approval)
            """
        ).fetchone()
    assert counts == (1, 1, 1)


def test_real_schema_v3_anchor_validates_approval_then_request(tmp_path: Path) -> None:
    request_path = create_test_request(_freeze_fixture(tmp_path))
    confirmation = validate_test_request(request_path)["freeze_sha256"]
    approval_path = approve_test_request(request_path, "Human Reviewer", confirmation)
    state: dict[str, Any] = {
        "approval_path": approval_path,
        "experiments_dir": tmp_path / "experiments",
        "data_dir": tmp_path / "data",
        "config_dir": tmp_path / "config",
    }
    state.update(final_test._validate_approval(state))
    state.update(final_test._validate_request(state))
    assert state["request_id"] == validate_test_request(request_path)["request_id"]
    assert state["catalog_database"] == tmp_path / "data/metadata.duckdb"


def test_wrong_request_admin_tuple_fails_before_locked_read(tmp_path: Path) -> None:
    request_path = create_test_request(_freeze_fixture(tmp_path))
    confirmation = validate_test_request(request_path)["freeze_sha256"]
    approval_path = approve_test_request(request_path, "Human Reviewer", confirmation)
    database = tmp_path / "data/metadata.duckdb"
    with duckdb.connect(str(database)) as connection:
        approval_row = connection.execute(
            "SELECT * FROM research.test_approval"
        ).fetchone()
        assert approval_row is not None
        connection.execute("DELETE FROM research.test_approval")
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            """
            UPDATE research.test_request
            SET robustness_report_sha256 = repeat('0', 64)
            """
        )
    with duckdb.connect(str(database)) as connection:
        placeholders = ", ".join("?" for _ in approval_row)
        connection.execute(
            f"INSERT INTO research.test_approval VALUES ({placeholders})",
            list(approval_row),
        )
    state: dict[str, Any] = {
        "approval_path": approval_path,
        "experiments_dir": tmp_path / "experiments",
        "data_dir": tmp_path / "data",
        "config_dir": tmp_path / "config",
    }
    state.update(final_test._validate_approval(state))
    with pytest.raises(PermissionError, match="request administrative"):
        final_test._validate_request(state)


@pytest.mark.parametrize("corruption", ["missing", "migration_sha", "wrong_approver"])
def test_final_approval_requires_exact_admin_catalog_before_read(
    tmp_path: Path, corruption: str
) -> None:
    request = create_test_request(_freeze_fixture(tmp_path))
    confirmation = validate_test_request(request)["freeze_sha256"]
    approval_path = approve_test_request(request, "Human Reviewer", confirmation)
    database = tmp_path / "data/metadata.duckdb"
    if corruption == "missing":
        database.unlink()
    else:
        with duckdb.connect(str(database)) as connection:
            if corruption == "migration_sha":
                connection.execute(
                    """
                    UPDATE meta.schema_migration
                    SET sha256 = repeat('0', 64) WHERE version = 3
                    """
                )
            else:
                connection.execute(
                    "UPDATE research.test_approval SET approver = 'forged'"
                )
    with pytest.raises(PermissionError, match="catalog"):
        final_test._validate_approval(
            {
                "approval_path": approval_path,
                "experiments_dir": tmp_path / "experiments",
                "data_dir": tmp_path / "data",
            }
        )


def test_coherently_resigned_approval_without_admin_record_is_rejected(
    tmp_path: Path,
) -> None:
    request = create_test_request(_freeze_fixture(tmp_path))
    confirmation = validate_test_request(request)["freeze_sha256"]
    original = approve_test_request(request, "Human Reviewer", confirmation)
    document = validate_approval(original)
    identity = {
        **{key: value for key, value in document.items() if key != "approval_id"},
        "approver": "Forged Reviewer",
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    ).hexdigest()
    forged = {**identity, "approval_id": f"approval-{digest}"}
    path = original.parent / f"approval-{digest}.json"
    _write_json(path, forged)
    with pytest.raises(PermissionError, match="catalog"):
        final_test._validate_approval(
            {
                "approval_path": path,
                "experiments_dir": tmp_path / "experiments",
                "data_dir": tmp_path / "data",
            }
        )


def test_approval_refuses_conflicting_existing_bytes(tmp_path: Path) -> None:
    request = create_test_request(_freeze_fixture(tmp_path))
    confirmation = validate_test_request(request)["freeze_sha256"]
    path = approve_test_request(request, "Human Reviewer", confirmation)
    path.write_text("conflict\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="immutable approval"):
        approve_test_request(request, "Human Reviewer", confirmation)


def _freeze_fixture(tmp_path: Path, *, passed: bool = True) -> Path:
    shutil.copytree(ROOT / "config", tmp_path / "config")
    shutil.copytree(
        ROOT / "src/alpha_lab/factors/candidates",
        tmp_path / "src/alpha_lab/factors/candidates",
    )
    execution_paths = [
        "src/alpha_lab/baseline/backtest.py",
        "src/alpha_lab/baseline/config.py",
        "src/alpha_lab/data/normalize.py",
        "src/alpha_lab/database/catalog.py",
        "src/alpha_lab/evaluation/config.py",
        "src/alpha_lab/evaluation/metrics.py",
        "src/alpha_lab/factors/contract.py",
        "src/alpha_lab/factors/registry.py",
        "src/alpha_lab/factors/candidates/F1002.py",
        "src/alpha_lab/factors/candidates/F1002.yaml",
        "src/alpha_lab/robustness/approval.py",
        "src/alpha_lab/robustness/config.py",
        "src/alpha_lab/robustness/exposures.py",
        "src/alpha_lab/robustness/final_test.py",
        "src/alpha_lab/robustness/freeze.py",
        "src/alpha_lab/robustness/io.py",
        "src/alpha_lab/robustness/report.py",
        "src/alpha_lab/robustness/walk_forward.py",
    ]
    for relative in execution_paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    shutil.copytree(
        ROOT / "src/alpha_lab/database/sql", tmp_path / "src/alpha_lab/database/sql"
    )
    payload = {
        "schema_version": 1,
        "factor": {
            "factor_id": "F1002",
            "source_path": "src/alpha_lab/factors/candidates/F1002.py",
            "source_sha256": _sha256(
                tmp_path / "src/alpha_lab/factors/candidates/F1002.py"
            ),
            "metadata_path": "src/alpha_lab/factors/candidates/F1002.yaml",
            "metadata_sha256": _sha256(
                tmp_path / "src/alpha_lab/factors/candidates/F1002.yaml"
            ),
        },
        "snapshots": {
            "phase5": {
                "snapshot_id": "p5-" + "4" * 20,
                "manifest_path": "manifests/p5-" + "4" * 20 + "/manifest.json",
                "manifest_sha256": "5" * 64,
            },
            "exposure": {
                "snapshot_id": "p6x-" + "6" * 20,
                "manifest_path": "manifests/p6x-" + "6" * 20 + "/manifest.json",
                "manifest_sha256": "7" * 64,
                "capability_id": "pretest-" + "8" * 20,
                "capability_sha256": "9" * 64,
            },
        },
        "policies": {
            "robustness": {
                "path": "config/robustness.yaml",
                "sha256": load_robustness_config(tmp_path / "config/robustness.yaml")[
                    1
                ],
            },
            "costs": {
                "path": "config/costs.yaml",
                "sha256": _cost_policy_sha256(tmp_path / "config/costs.yaml"),
            },
        },
        "test": {
            "access": "human_approval_only",
            "start": "2026-01-01",
            "end": "2026-07-11",
        },
        "git_commit": "d" * 40,
    }
    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    freeze_id = f"freeze-{identity}"
    root = tmp_path / "experiments" / "phase6" / freeze_id
    root.mkdir(parents=True)
    freeze = {**payload, "freeze_id": freeze_id, "identity_sha256": identity}
    _write_json(root / "freeze.json", freeze)
    config, robustness_sha256 = load_robustness_config(
        tmp_path / "config/robustness.yaml"
    )
    common = {
        "schema_version": 1,
        "freeze_id": freeze_id,
        "freeze_sha256": _sha256(root / "freeze.json"),
        "factor_id": "F1002",
        "robustness_policy_sha256": robustness_sha256,
        "evaluation_policy_sha256": load_evaluation_config(
            tmp_path / "config/factor_evaluation.yaml"
        )[1],
        "dependencies": {
            "phase5_manifest_sha256": "5" * 64,
            "exposure_manifest_sha256": "7" * 64,
            "cost_policy_sha256": freeze["policies"]["costs"]["sha256"],
            "factor_source_sha256": freeze["factor"]["source_sha256"],
            "factor_metadata_sha256": freeze["factor"]["metadata_sha256"],
        },
        "orientation": {
            "candidate_direction": -1,
            "score_formula": "score=standardize(winsorize(value*direction))",
            "direction_consistency_source": "oriented_mean_rank_ic_positive",
        },
        "test_accessed": False,
    }
    gates = {
        "direction_consistency": passed,
        "fold_coverage": True,
        "double_cost_direction": True,
        "industry_neutral_retention": True,
    }
    folds = [
        {
            "fold_id": fold.fold_id,
            "start": fold.start.isoformat(),
            "end": fold.end.isoformat(),
            "input_row_count": 10,
            "valid_row_count": 8,
            "coverage": 0.8,
            "mean_ic": 0.01,
            "mean_rank_ic": 0.01 if value else -0.01,
            "icir": 0.1,
            "rank_icir": 0.1,
            "group_returns": {"1": -0.01, "2": 0.01},
            "factor_turnover": 0.1,
            "direction_consistent": value,
        }
        for value, fold in zip(
            [passed, True, True, True, True],
            config.walk_forward_folds,
            strict=True,
        )
    ]
    walk = {**common, "folds": folds, "gates": gates, "passed": passed}
    cost = {
        **common,
        "scenarios": {
            str(float(multiplier)): {
                "metrics": {"total_return": (1.01**5) - 1.0},
                "folds": [
                    {
                        "fold_id": fold.fold_id,
                        "metrics": {
                            "initial_cash": 1000000.0,
                            "final_nav": 1010000.0,
                            "total_return": 0.01,
                            "annualized_return": 0.01,
                            "annualized_volatility": 0.1,
                            "sharpe": 0.1,
                            "max_drawdown": -0.01,
                            "trade_count": 1,
                            "total_fees": 1.0,
                            "turnover_ratio": 0.1,
                        },
                        "constraints": {
                            "blocked_suspend": 0,
                            "blocked_limit": 0,
                            "blocked_st": 0,
                            "blocked_unknown_status": 0,
                            "blocked_t_plus_one": 0,
                            "blocked_lot_or_cash": 0,
                            "inferred_price_limit_checks": 0,
                        },
                    }
                    for fold in config.walk_forward_folds
                ],
            }
            for multiplier in config.cost_multipliers
        },
    }
    exposure = {
        **common,
        "size": {
            "joined_rows": 10,
            "correlation": 0.1,
            "risk_threshold": 0.3,
            "risk_flag": False,
            "uses": "log(total_market_cap_cny)",
            "method": "daily_cross_sectional_spearman",
            "daily": [],
            "yearly": {},
        },
        "industry": {
            "joined_rows": 10,
            "original_joined_rows": 10,
            "original_rank_ic": 0.1,
            "neutral_rank_ic": 0.06,
            "abs_rank_ic_retention": 0.6,
            "minimum_group_size": 2,
            "by_industry": [],
            "mean_score_dispersion": 0.0,
        },
        "missing": {"size_rows": 0, "industry_rows": 0},
    }
    _write_json(root / "walk_forward.json", walk)
    _write_json(root / "cost_sensitivity.json", cost)
    _write_json(root / "exposure_report.json", exposure)
    (root / "robustness_report.md").write_bytes(_markdown(walk, exposure))
    return root / "freeze.json"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
