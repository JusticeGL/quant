from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from alpha_lab.baseline.backtest import run_topk_backtest
from alpha_lab.baseline.config import load_phase2_config
from alpha_lab.evaluation.config import load_evaluation_config
from alpha_lab.evaluation.metrics import calculate_factor_metrics, prepare_factor_values
from alpha_lab.factors.contract import validate_factor_output
from alpha_lab.factors.registry import FactorRegistry
from alpha_lab.robustness.approval import (
    _write_immutable,
    canonical_bytes,
    validate_approval,
    validate_test_request,
)
from alpha_lab.robustness.config import WalkForwardFold, load_robustness_config
from alpha_lab.robustness.exposures import calculate_exposures
from alpha_lab.robustness.freeze import (
    _canonical_bytes,
    _cost_policy_sha256,
    _trusted_file,
    _validate_freeze_document_schema,
)
from alpha_lab.robustness.io import _market_contract
from alpha_lab.robustness.walk_forward import (
    backtest_predictions,
    build_fold_labels,
    scale_costs,
)


def run_final_test(
    approval_path: Path,
    config_dir: Path,
    data_dir: Path,
    experiments_dir: Path,
) -> Path:
    """Validate the complete approval chain before any locked partition read."""
    state: dict[str, Any] = {
        "approval_path": approval_path,
        "config_dir": config_dir,
        "data_dir": data_dir,
        "experiments_dir": experiments_dir,
    }
    for validator in (
        _validate_approval,
        _validate_request,
        _validate_freeze,
        _validate_candidate,
        _validate_policy,
        _validate_cost,
        _validate_phase5,
        _validate_exposure,
    ):
        state.update(validator(state))

    test = state["test"]
    run_identity = {
        "schema_version": 1,
        "approval_id": state["approval_id"],
        "approval_sha256": state["approval_sha256"],
        "request_id": state["request_id"],
        "request_sha256": state["request_sha256"],
        "freeze_id": state["freeze_id"],
        "freeze_sha256": state["freeze_sha256"],
        "locked_test": test,
    }
    test_run_id = f"final-{hashlib.sha256(canonical_bytes(run_identity)).hexdigest()}"
    output_dir = (
        experiments_dir
        / "phase6"
        / state["freeze_id"]
        / "final"
        / test_run_id
    )
    try:
        market = _read_locked_market(
            data_dir,
            state["snapshots"]["phase5"]["snapshot_id"],
            date.fromisoformat(test["start"]),
            date.fromisoformat(test["end"]),
        )
        evaluation = _evaluate_locked_test(market, state)
    except Exception as error:
        failed = {
            **run_identity,
            "test_run_id": test_run_id,
            "status": "test_failed",
            "test_accessed": True,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        _publish_final(output_dir, failed)
        raise
    result = {
        **run_identity,
        "test_run_id": test_run_id,
        "status": "test_completed",
        "test_accessed": True,
        "result": evaluation,
    }
    return _publish_final(output_dir, result)


def _publish_final(output_dir: Path, result: dict[str, Any]) -> Path:
    result_path = output_dir / "result.json"
    _write_immutable(
        result_path,
        canonical_bytes(result),
        "final-test artifact",
    )
    _write_immutable(
        output_dir / "report.md",
        _final_markdown(result),
        "final-test artifact",
    )
    return result_path


def _validate_approval(state: dict[str, Any]) -> dict[str, Any]:
    path = state["approval_path"]
    if not isinstance(path, Path) or not path.is_file():
        raise PermissionError("explicit final-test approval is missing")
    try:
        _trusted_file(path, state["experiments_dir"], "approval artifact")
        document = validate_approval(path)
    except ValueError as error:
        raise PermissionError("explicit final-test approval is invalid") from error
    return {
        **document,
        "approval_sha256": _sha256(path),
    }


def _validate_request(state: dict[str, Any]) -> dict[str, Any]:
    approval_path = state["approval_path"]
    request_path = approval_path.parent.parent / "test_request.json"
    _trusted_file(request_path, state["experiments_dir"], "test request")
    document = validate_test_request(request_path)
    if (
        document["request_id"] != state["request_id"]
        or _sha256(request_path) != state["request_sha256"]
        or document["freeze_id"] != state["freeze_id"]
        or document["freeze_sha256"] != state["freeze_sha256"]
    ):
        raise ValueError("request hash link does not match approval")
    freeze_dir = request_path.parent
    for name, reference in document["robustness_artifacts"].items():
        path = freeze_dir / name
        if (
            path.is_symlink()
            or not path.is_file()
            or _sha256(path) != reference["sha256"]
        ):
            raise ValueError(f"request robustness artifact drift: {name}")
    return {
        "request_path": request_path,
        "request_sha256": _sha256(request_path),
        "locked_test": document["locked_test"],
    }


def _validate_freeze(state: dict[str, Any]) -> dict[str, Any]:
    path = state["request_path"].parent / "freeze.json"
    _trusted_file(path, state["experiments_dir"], "freeze artifact")
    freeze = _read_json(path, "freeze")
    expected_keys = {
        "schema_version",
        "factor",
        "snapshots",
        "policies",
        "test",
        "git_commit",
        "freeze_id",
        "identity_sha256",
    }
    if (
        set(freeze) != expected_keys
        or freeze.get("schema_version") != 1
        or path.read_bytes() != _canonical_bytes(freeze, newline=True)
    ):
        raise ValueError("freeze canonical schema is invalid")
    _validate_freeze_document_schema(freeze)
    payload = {
        key: value
        for key, value in freeze.items()
        if key not in {"freeze_id", "identity_sha256"}
    }
    identity_sha256 = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if (
        freeze.get("identity_sha256") != identity_sha256
        or freeze.get("freeze_id") != f"freeze-{identity_sha256}"
    ):
        raise ValueError("freeze content identity is invalid")
    freeze["freeze_sha256"] = _sha256(path)
    if (
        freeze["freeze_id"] != state["freeze_id"]
        or freeze["freeze_sha256"] != state["freeze_sha256"]
        or freeze["test"] != state["locked_test"]
        or path.parent
        != state["experiments_dir"] / "phase6" / str(freeze["freeze_id"])
    ):
        raise ValueError("freeze hash link or locked range does not match request")
    return freeze


def _validate_candidate(state: dict[str, Any]) -> dict[str, Any]:
    root = state["config_dir"].parent
    factor = state["factor"]
    for key in ("source", "metadata"):
        path = root / factor[f"{key}_path"]
        _trusted_file(path, root, f"candidate {key}")
        if (
            path.is_symlink()
            or not path.is_file()
            or _sha256(path) != factor[f"{key}_sha256"]
        ):
            raise ValueError(f"candidate {key} hash drift")
    return {}


def _validate_policy(state: dict[str, Any]) -> dict[str, Any]:
    path = state["config_dir"] / "robustness.yaml"
    _trusted_file(path, state["config_dir"], "robustness policy")
    config, sha256 = load_robustness_config(path)
    if sha256 != state["policies"]["robustness"]["sha256"]:
        raise ValueError("robustness policy hash drift")
    if config.test.model_dump(mode="json") != state["test"]:
        raise ValueError("locked test range drift")
    return {"robustness_config": config}


def _validate_cost(state: dict[str, Any]) -> dict[str, Any]:
    path = state["config_dir"] / "costs.yaml"
    _trusted_file(path, state["config_dir"], "cost policy")
    if _cost_policy_sha256(path) != state["policies"]["costs"]["sha256"]:
        raise ValueError("cost policy hash drift")
    return {}


def _validate_phase5(state: dict[str, Any]) -> dict[str, Any]:
    reference = state["snapshots"]["phase5"]
    path = state["data_dir"] / reference["manifest_path"]
    _trusted_file(path, state["data_dir"], "Phase 5 manifest")
    if (
        path.is_symlink()
        or not path.is_file()
        or _sha256(path) != reference["manifest_sha256"]
    ):
        raise ValueError("Phase 5 manifest hash drift")
    return {"phase5_manifest": _read_json(path, "Phase 5 manifest")}


def _validate_exposure(state: dict[str, Any]) -> dict[str, Any]:
    reference = state["snapshots"]["exposure"]
    path = state["data_dir"] / reference["manifest_path"]
    _trusted_file(path, state["data_dir"], "exposure manifest")
    if (
        path.is_symlink()
        or not path.is_file()
        or _sha256(path) != reference["manifest_sha256"]
    ):
        raise ValueError("exposure manifest hash drift")
    manifest = _read_json(path, "exposure manifest")
    capability = path.parent / "pretest_capability.json"
    _trusted_file(capability, state["data_dir"], "pre-test capability")
    if (
        capability.is_symlink()
        or not capability.is_file()
        or _sha256(capability) != reference["capability_sha256"]
    ):
        raise ValueError("exposure capability hash drift")
    return {"exposure_manifest": manifest}


def _read_locked_market(
    data_dir: Path, snapshot_id: str, start: date, end: date
) -> pd.DataFrame:
    manifest = _read_json(
        data_dir / "manifests" / snapshot_id / "manifest.json",
        "Phase 5 manifest",
    )
    frames: dict[str, list[pd.DataFrame]] = {
        "daily_bar": [],
        "adjustment_factor": [],
        "daily_status": [],
    }
    patterns = {
        name: re.compile(rf"{name}/year=2026/part[.]parquet") for name in frames
    }
    for item in manifest.get("artifacts", []):
        name = str(item.get("name"))
        dataset = next(
            (key for key, pattern in patterns.items() if pattern.fullmatch(name)),
            None,
        )
        if dataset is None:
            continue
        path = _verified_snapshot_artifact(data_dir, item, "research", snapshot_id)
        frame = pd.read_parquet(path)
        frame["trade_date"] = pd.to_datetime(
            frame["trade_date"], errors="raise"
        ).dt.normalize()
        frames[dataset].append(
            frame.loc[frame["trade_date"].dt.date.between(start, end)].copy()
        )
    daily = _concat_required(frames["daily_bar"], "locked daily_bar")
    adjustment = (
        pd.concat(frames["adjustment_factor"], ignore_index=True)
        if frames["adjustment_factor"]
        else pd.DataFrame()
    )
    status = (
        pd.concat(frames["daily_status"], ignore_index=True)
        if frames["daily_status"]
        else pd.DataFrame()
    )
    return _market_contract(daily, adjustment, status)


def _read_locked_exposures(
    state: dict[str, Any], start: date, end: date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = state["exposure_manifest"]
    data_dir = state["data_dir"]
    snapshot_id = state["snapshots"]["exposure"]["snapshot_id"]
    market_item = next(
        item
        for item in manifest["artifacts"]
        if item["name"] == "market_cap/year=2026/part.parquet"
    )
    industry_item = next(
        item
        for item in manifest["artifacts"]
        if item["name"] == "industry_membership.parquet"
    )
    market = pd.read_parquet(
        _verified_snapshot_artifact(data_dir, market_item, "exposures", snapshot_id)
    )
    industry = pd.read_parquet(
        _verified_snapshot_artifact(data_dir, industry_item, "exposures", snapshot_id)
    )
    market["trade_date"] = pd.to_datetime(
        market["trade_date"], errors="raise"
    ).dt.normalize()
    market = market.loc[market["trade_date"].dt.date.between(start, end)].copy()
    return market, industry


def _evaluate_locked_test(market: pd.DataFrame, state: dict[str, Any]) -> dict[str, Any]:
    config_dir = state["config_dir"]
    robustness = state["robustness_config"]
    evaluation, evaluation_sha256 = load_evaluation_config(
        config_dir / "factor_evaluation.yaml"
    )
    phase2 = load_phase2_config(config_dir)
    factor_id = state["factor"]["factor_id"]
    registry = FactorRegistry(
        config_dir.parent / "src" / "alpha_lab" / "factors" / "candidates",
        config_dir / "factor_registry.yaml",
    )
    candidate = registry.get(factor_id)
    raw_scores = validate_factor_output(candidate, market)
    scores = prepare_factor_values(raw_scores, candidate.metadata.direction, evaluation)
    test = state["test"]
    fold = WalkForwardFold(
        fold_id="wf_2026",
        start=date.fromisoformat(test["start"]),
        end=date.fromisoformat(test["end"]),
    )
    labels = build_fold_labels(market, fold)
    evaluated = scores.merge(
        labels,
        on=["trade_date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    metrics = calculate_factor_metrics(
        evaluated,
        expected_rows=int(
            market[["trade_date", "instrument"]].drop_duplicates().shape[0]
        ),
        group_count=evaluation.group_count,
        annualization_days=evaluation.annualization_days,
    )
    predictions = backtest_predictions(evaluated)
    costs: dict[str, Any] = {}
    for multiplier in robustness.cost_multipliers:
        result = run_topk_backtest(
            predictions,
            market,
            strategy=phase2.baseline.strategy,
            costs=scale_costs(phase2.costs, multiplier),
            annualization_days=evaluation.annualization_days,
            allowed_end=fold.end,
        )
        costs[str(float(multiplier))] = {
            "metrics": result.metrics,
            "constraints": result.constraints,
        }
    market_cap, industries = _read_locked_exposures(state, fold.start, fold.end)
    exposures = calculate_exposures(
        evaluated[["trade_date", "instrument", "score"]],
        market_cap,
        industries,
        evaluated[["trade_date", "instrument", "label"]],
        size_risk_threshold=robustness.size_correlation_risk_threshold,
    )
    return {
        "status": "completed",
        "factor_id": factor_id,
        "evaluation_policy_sha256": evaluation_sha256,
        "metrics": metrics,
        "cost_sensitivity": costs,
        "exposures": exposures,
    }


def _verified_snapshot_artifact(
    data_dir: Path,
    item: dict[str, Any],
    root: str,
    snapshot_id: str,
) -> Path:
    name = str(item.get("name"))
    expected = Path(root) / snapshot_id / name
    path = data_dir / expected
    current = path
    while current != data_dir:
        if current.is_symlink():
            raise ValueError(f"locked artifact path contains symlink: {name}")
        if current.parent == current:
            raise ValueError(f"locked artifact escapes data directory: {name}")
        current = current.parent
    if (
        item.get("path") != expected.as_posix()
        or path.is_symlink()
        or not path.is_file()
        or _sha256(path) != item.get("sha256")
    ):
        raise ValueError(f"locked artifact hash or path drift: {name}")
    return path


def _concat_required(frames: list[pd.DataFrame], label: str) -> pd.DataFrame:
    if not frames:
        raise ValueError(f"{label} partition is missing")
    result = pd.concat(frames, ignore_index=True)
    if result.empty:
        raise ValueError(f"{label} is empty")
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is malformed") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _final_markdown(result: dict[str, Any]) -> bytes:
    payload = result.get("result", {})
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    lines = [
        "# Phase 6 Locked Final Test",
        "",
        f"Test run: `{result['test_run_id']}`",
        f"Freeze: `{result['freeze_id']}`",
        f"Status: `{result['status']}`",
        "Locked test accessed: `true`",
        "",
        "## Metrics",
        "",
        f"```json\n{json.dumps(metrics, ensure_ascii=False, sort_keys=True, default=str)}\n```",
        "",
        *(
            [f"Failure: `{result['error']['type']}: {result['error']['message']}`", ""]
            if result["status"] == "test_failed"
            else []
        ),
        "Results are reported without promotion or suppression, including "
        "unfavorable outcomes.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
