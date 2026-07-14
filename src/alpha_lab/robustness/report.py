from __future__ import annotations

import hashlib
import json
import os
import tempfile
from filecmp import cmp
from pathlib import Path
from typing import Any

import pandas as pd

from alpha_lab.baseline.backtest import BacktestResult, run_topk_backtest
from alpha_lab.baseline.config import load_phase2_config
from alpha_lab.evaluation.config import load_evaluation_config
from alpha_lab.evaluation.metrics import (
    calculate_factor_metrics,
    prepare_factor_values,
)
from alpha_lab.factors.contract import validate_factor_output
from alpha_lab.factors.registry import FactorRegistry
from alpha_lab.robustness.config import load_robustness_config
from alpha_lab.robustness.contracts import RobustnessResult
from alpha_lab.robustness.exposures import calculate_exposures
from alpha_lab.robustness.freeze import validate_freeze
from alpha_lab.robustness.io import read_pretest_exposures, read_pretest_market
from alpha_lab.robustness.walk_forward import (
    backtest_predictions,
    build_fold_labels,
    evaluate_gates,
    scale_costs,
)


def evaluate_frozen_candidate(
    freeze_path: Path,
    config_dir: Path,
    data_dir: Path,
    experiments_dir: Path,
) -> RobustnessResult:
    """Evaluate a validated freeze using pre-test readers only."""
    freeze = validate_freeze(freeze_path, config_dir, data_dir)
    robustness, policy_sha256 = load_robustness_config(config_dir / "robustness.yaml")
    phase2 = load_phase2_config(config_dir)
    evaluation, evaluation_sha256 = load_evaluation_config(
        config_dir / "factor_evaluation.yaml"
    )
    factor = freeze.get("factor")
    snapshots = freeze.get("snapshots")
    policies = freeze.get("policies")
    if (
        not isinstance(factor, dict)
        or not isinstance(snapshots, dict)
        or not isinstance(policies, dict)
    ):
        raise ValueError("validated freeze has malformed dependencies")
    factor_id = str(factor["factor_id"])
    if factor_id not in robustness.factor_ids:
        raise PermissionError("freeze candidate is not allowed by robustness policy")
    phase5 = snapshots.get("phase5")
    exposure = snapshots.get("exposure")
    if not isinstance(phase5, dict) or not isinstance(exposure, dict):
        raise ValueError("validated freeze snapshot dependencies are malformed")
    costs_policy = policies.get("costs")
    if not isinstance(costs_policy, dict):
        raise ValueError("validated freeze cost dependency is malformed")

    safe_end_before = robustness.test.start
    market = read_pretest_market(
        data_dir, str(exposure["snapshot_id"]), safe_end_before
    )
    market_cap, industries = read_pretest_exposures(
        data_dir, str(exposure["snapshot_id"]), safe_end_before
    )
    market = market.loc[
        market["trade_date"].dt.date.between(
            robustness.warmup.start, robustness.walk_forward_folds[-1].end
        )
    ].copy()
    if market.empty:
        raise ValueError("pre-test market is empty")

    registry = FactorRegistry(
        config_dir.parent / "src" / "alpha_lab" / "factors" / "candidates",
        config_dir / "factor_registry.yaml",
    )
    candidate = registry.get(factor_id)
    # This is intentionally the sole candidate compute call for all five folds.
    raw_scores = validate_factor_output(candidate, market)
    scores = prepare_factor_values(raw_scores, candidate.metadata.direction, evaluation)

    fold_reports: list[dict[str, Any]] = []
    cost_folds: dict[str, list[dict[str, Any]]] = {
        _multiplier_key(value): [] for value in robustness.cost_multipliers
    }
    exposure_scores: list[pd.DataFrame] = []
    exposure_labels: list[pd.DataFrame] = []
    output_dir = freeze_path.parent
    if output_dir != experiments_dir / "phase6" / str(freeze["freeze_id"]):
        raise ValueError("freeze does not belong to the requested experiments root")

    for fold in robustness.walk_forward_folds:
        fold_market = market.loc[
            market["trade_date"].dt.date.between(fold.start, fold.end)
        ].copy()
        fold_scores = scores.loc[
            scores["trade_date"].dt.date.between(fold.start, fold.end)
        ].copy()
        labels = build_fold_labels(fold_market, fold)
        evaluated = fold_scores.merge(
            labels,
            on=["trade_date", "instrument"],
            how="left",
            validate="one_to_one",
        )
        expected_rows = int(
            fold_market[["trade_date", "instrument"]].drop_duplicates().shape[0]
        )
        metrics = calculate_factor_metrics(
            evaluated,
            expected_rows=expected_rows,
            group_count=evaluation.group_count,
            annualization_days=evaluation.annualization_days,
        )
        summary = {
            "fold_id": fold.fold_id,
            "start": fold.start.isoformat(),
            "end": fold.end.isoformat(),
            "input_row_count": expected_rows,
            "valid_row_count": metrics["valid_row_count"],
            "coverage": metrics["coverage"],
            "mean_ic": metrics["mean_ic"],
            "mean_rank_ic": metrics["mean_rank_ic"],
            "icir": metrics["icir"],
            "rank_icir": metrics["rank_icir"],
            "group_returns": metrics["group_returns"],
            "factor_turnover": metrics["factor_turnover"],
            "direction_consistent": metrics["mean_rank_ic"] is not None
            and float(metrics["mean_rank_ic"]) > 0.0,
        }
        fold_reports.append(summary)
        predictions = backtest_predictions(evaluated)
        exposure_scores.append(evaluated[["trade_date", "instrument", "score"]].copy())
        exposure_labels.append(evaluated[["trade_date", "instrument", "label"]].copy())
        _write_parquet_immutable(
            output_dir / "large" / fold.fold_id / "predictions.parquet", predictions
        )
        for multiplier in robustness.cost_multipliers:
            result = run_topk_backtest(
                predictions,
                fold_market,
                strategy=phase2.baseline.strategy,
                costs=scale_costs(phase2.costs, multiplier),
                annualization_days=evaluation.annualization_days,
                allowed_end=fold.end,
            )
            _store_backtest(output_dir, fold.fold_id, multiplier, result)
            cost_folds[_multiplier_key(multiplier)].append(
                {
                    "fold_id": fold.fold_id,
                    "metrics": result.metrics,
                    "constraints": result.constraints,
                }
            )

    cost_report = _cost_report(cost_folds)
    exposure_report = calculate_exposures(
        pd.concat(exposure_scores, ignore_index=True),
        market_cap,
        industries,
        pd.concat(exposure_labels, ignore_index=True),
        size_risk_threshold=robustness.size_correlation_risk_threshold,
    )
    gates = evaluate_gates(fold_reports, cost_report, exposure_report, robustness)
    passed = all(gates.values())
    common = {
        "schema_version": 1,
        "freeze_id": freeze["freeze_id"],
        "factor_id": factor_id,
        "freeze_sha256": freeze["freeze_sha256"],
        "robustness_policy_sha256": policy_sha256,
        "evaluation_policy_sha256": evaluation_sha256,
        "dependencies": {
            "phase5_manifest_sha256": phase5["manifest_sha256"],
            "exposure_manifest_sha256": exposure["manifest_sha256"],
            "cost_policy_sha256": costs_policy["sha256"],
            "factor_source_sha256": factor["source_sha256"],
            "factor_metadata_sha256": factor["metadata_sha256"],
        },
        "orientation": {
            "candidate_direction": candidate.metadata.direction,
            "score_formula": "score=standardize(winsorize(value*direction))",
            "direction_consistency_source": "oriented_mean_rank_ic_positive",
        },
        "test_accessed": False,
    }
    walk_document = {**common, "folds": fold_reports, "gates": gates, "passed": passed}
    cost_document = {**common, **cost_report}
    exposure_document = {**common, **exposure_report}
    _write_json_immutable(output_dir / "walk_forward.json", walk_document)
    _write_json_immutable(output_dir / "cost_sensitivity.json", cost_document)
    _write_json_immutable(output_dir / "exposure_report.json", exposure_document)
    report_path = output_dir / "robustness_report.md"
    _write_immutable(report_path, _markdown(walk_document, exposure_document))
    report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
    return RobustnessResult(
        freeze_id=str(freeze["freeze_id"]),
        output_dir=output_dir,
        report_path=report_path,
        report_sha256=report_sha256,
        passed=passed,
    )


def _cost_report(folds: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    scenarios: dict[str, dict[str, Any]] = {}
    for key, items in folds.items():
        returns = [float(item["metrics"]["total_return"]) for item in items]
        aggregate = 1.0
        for value in returns:
            aggregate *= 1.0 + value
        scenarios[key] = {
            "metrics": {"total_return": aggregate - 1.0},
            "folds": items,
        }
    return {"scenarios": scenarios}


def _store_backtest(
    output_dir: Path, fold_id: str, multiplier: float, result: BacktestResult
) -> None:
    root = output_dir / "large" / fold_id / f"cost_{_multiplier_key(multiplier)}x"
    _write_parquet_immutable(root / "nav.parquet", result.daily)
    _write_parquet_immutable(root / "trades.parquet", result.trades)


def _write_parquet_immutable(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        _fsync_file(temporary)
        _publish_no_clobber(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_immutable(path: Path, value: object) -> None:
    content = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    ).encode("utf-8")
    _write_immutable(path, content)


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_no_clobber(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_no_clobber(temporary: Path, path: Path) -> None:
    try:
        os.link(temporary, path)
    except FileExistsError:
        if not cmp(temporary, path, shallow=False):
            raise RuntimeError(
                f"immutable robustness artifact differs: {path.name}"
            ) from None
    else:
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _markdown(walk: dict[str, Any], exposure: dict[str, Any]) -> bytes:
    gate_lines = [
        f"- {key}: {'PASS' if value else 'FAIL'}"
        for key, value in sorted(walk["gates"].items())
    ]
    size = exposure["size"]
    lines = [
        "# Phase 6 Robustness Report",
        "",
        f"Freeze: `{walk['freeze_id']}`",
        f"Factor: `{walk['factor_id']}`",
        f"Pre-test result: `{'PASS' if walk['passed'] else 'FAIL'}`",
        "Locked test accessed: `false`",
        "",
        "## Gates",
        "",
        *gate_lines,
        "",
        "## Size risk",
        "",
        f"Correlation: `{size['correlation']}`",
        f"Risk flag: `{str(size['risk_flag']).lower()}`",
        "",
        "Passing permits only a separate test request; it does not authorize "
        "test access.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _multiplier_key(value: float) -> str:
    return str(float(value))
