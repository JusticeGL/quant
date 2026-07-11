from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from alpha_lab.baseline.backtest import BacktestResult, run_topk_backtest
from alpha_lab.baseline.config import CostConfig, CostRule, load_phase2_config
from alpha_lab.evaluation.config import EvaluationConfig, load_evaluation_config
from alpha_lab.evaluation.leakage import audit_factor
from alpha_lab.evaluation.metrics import (
    build_forward_labels,
    calculate_factor_metrics,
    factor_correlations,
    prepare_factor_values,
)
from alpha_lab.factors.contract import validate_factor_output
from alpha_lab.factors.registry import FactorRegistry


@dataclass(frozen=True)
class FactorEvaluationResult:
    factor_id: str
    run_id: str
    output_dir: Path
    result_path: Path
    values_path: Path
    result_sha256: str
    eligible_for_review: bool


def evaluate_factor(
    factor_id: str,
    config_dir: Path,
    data_dir: Path,
    output_root: Path,
    *,
    snapshot_id: str | None = None,
) -> FactorEvaluationResult:
    phase2 = load_phase2_config(config_dir)
    evaluation, evaluation_hash = load_evaluation_config(
        config_dir / "factor_evaluation.yaml"
    )
    registry = FactorRegistry(
        Path(__file__).parents[1] / "factors" / "candidates",
        config_dir / "factor_registry.yaml",
    )
    candidate = registry.get(factor_id)
    selected_snapshot = snapshot_id or _latest_snapshot(data_dir)
    snapshot_manifest = _read_json(
        data_dir / "manifests" / selected_snapshot / "manifest.json"
    )
    if snapshot_manifest["snapshot_id"] != selected_snapshot:
        raise ValueError("snapshot manifest identity mismatch")

    split = phase2.splits
    pretest_end = split.test.start - timedelta(days=1)
    market = pd.read_parquet(
        data_dir / "silver" / selected_snapshot / "daily.parquet",
        filters=[("trade_date", "<", pd.Timestamp(split.test.start))],
    )
    market["trade_date"] = pd.to_datetime(market["trade_date"]).dt.normalize()
    if market.empty or market["trade_date"].dt.date.max() > pretest_end:
        raise ValueError("pre-test market filter failed")
    factor_market = market.loc[market["trade_date"].dt.date <= split.validation.end]

    prepared: dict[str, pd.DataFrame] = {}
    leakage_reports: dict[str, dict[str, Any]] = {}
    for item in registry.all():
        report = audit_factor(item, factor_market)
        leakage_reports[item.metadata.factor_id] = report.to_dict()
        if report.passed:
            values = validate_factor_output(item, factor_market)
            prepared[item.metadata.factor_id] = prepare_factor_values(
                values, item.metadata.direction, evaluation
            )
    leakage = leakage_reports[factor_id]
    if factor_id not in prepared:
        raise ValueError(
            f"factor {factor_id} failed leakage audit: "
            f"{json.dumps(leakage, sort_keys=True)}"
        )

    target_values = prepared[factor_id]
    labels = build_forward_labels(market)
    evaluated = target_values.merge(
        labels,
        on=["trade_date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    validation_mask = evaluated["trade_date"].dt.date.between(
        split.validation.start, split.validation.end
    )
    validation = evaluated.loc[validation_mask].copy()
    expected_rows = int(
        factor_market.loc[
            factor_market["trade_date"].dt.date.between(
                split.validation.start, split.validation.end
            ),
            ["trade_date", "instrument"],
        ]
        .drop_duplicates()
        .shape[0]
    )
    metrics = calculate_factor_metrics(
        validation,
        expected_rows=expected_rows,
        group_count=evaluation.group_count,
        annualization_days=evaluation.annualization_days,
    )
    comparison_values = {
        item_id: frame.loc[
            frame["trade_date"].dt.date.between(
                split.validation.start, split.validation.end
            )
        ]
        for item_id, frame in prepared.items()
        if item_id != factor_id
    }
    correlations = factor_correlations(validation, comparison_values)
    accepted_correlations = [
        abs(value)
        for item_id, value in correlations.items()
        if item_id in registry.accepted_factor_ids and value is not None
    ]
    max_accepted_correlation = (
        max(accepted_correlations) if accepted_correlations else None
    )

    predictions = validation[["trade_date", "instrument", "score", "label"]].rename(
        columns={"trade_date": "datetime"}
    )
    cost_sensitivity = _cost_sensitivity(
        predictions,
        market,
        phase2.costs,
        phase2.baseline.strategy,
        evaluation.annualization_days,
        pretest_end,
    )
    checks = _promotion_checks(
        metrics,
        leakage,
        max_accepted_correlation,
        cost_sensitivity,
        evaluation,
    )
    eligible = all(checks.values())
    git = _git_identity()
    run_identity = {
        "factor_id": factor_id,
        "factor_source_sha256": candidate.source_sha256,
        "factor_metadata_sha256": candidate.metadata_sha256,
        "snapshot_id": selected_snapshot,
        "evaluation_sha256": evaluation_hash,
        "split_sha256": phase2.split_sha256,
        "cost_sha256": phase2.cost_sha256,
        "git_commit": git["commit"],
    }
    identity_hash = _canonical_hash(run_identity)
    run_id = f"factor-{factor_id.lower()}-{identity_hash[:20]}"
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=output_root))
    try:
        values_path = temporary / "factor_values.parquet"
        validation.to_parquet(values_path, index=False)
        result: dict[str, Any] = {
            "schema_version": 1,
            "phase": 3,
            "run_id": run_id,
            "factor": candidate.metadata.model_dump(mode="json"),
            "factor_source_sha256": candidate.source_sha256,
            "factor_metadata_sha256": candidate.metadata_sha256,
            "implementation_path": (
                f"src/alpha_lab/factors/candidates/{candidate.metadata.factor_id}.py"
            ),
            "data_snapshot_id": selected_snapshot,
            "data_research_eligible": snapshot_manifest["universe"][
                "research_eligible"
            ],
            "evaluation_policy_id": evaluation.policy_id,
            "evaluation_config_sha256": evaluation_hash,
            "split_policy_sha256": phase2.split_sha256,
            "cost_policy_sha256": phase2.cost_sha256,
            "git": git,
            "test_access": {
                "locked": True,
                "accessed": False,
                "reported": False,
            },
            "leakage": leakage,
            "metrics": metrics,
            "correlations": correlations,
            "max_accepted_factor_correlation": max_accepted_correlation,
            "exposures": {
                "industry": {
                    "status": "unavailable",
                    "reason": "point-in-time industry field is absent",
                },
                "size": {
                    "status": "unavailable",
                    "reason": "point-in-time market-cap field is absent",
                },
            },
            "topk_cost_sensitivity": cost_sensitivity,
            "promotion_checks": checks,
            "eligible_for_review": eligible,
            "decision": "not_decided",
            "engineering_only": True,
            "limitations": [
                "The fixed sample is survivorship-biased and research_eligible=false.",
                "Industry and market-cap exposure cannot be estimated from "
                "Phase 1 data.",
                "This result is eligible only for human review, never automatic "
                "acceptance.",
            ],
        }
        result_path = temporary / "factor_result.json"
        result_path.write_text(
            json.dumps(
                result, ensure_ascii=False, indent=2, sort_keys=True, default=str
            )
            + "\n",
            encoding="utf-8",
        )
        result_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
        destination = output_root / run_id
        if destination.exists():
            existing_hash = hashlib.sha256(
                (destination / "factor_result.json").read_bytes()
            ).hexdigest()
            if existing_hash != result_hash:
                raise RuntimeError(f"immutable factor run differs: {run_id}")
        else:
            os.replace(temporary, destination)
        finalized = FactorEvaluationResult(
            factor_id=factor_id,
            run_id=run_id,
            output_dir=destination,
            result_path=destination / "factor_result.json",
            values_path=destination / "factor_values.parquet",
            result_sha256=result_hash,
            eligible_for_review=eligible,
        )
        from alpha_lab.database.catalog import record_factor_evaluation

        record_factor_evaluation(
            data_dir / "metadata.duckdb",
            config_dir,
            data_dir,
            finalized.result_path,
        )
        return finalized
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _cost_sensitivity(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    costs: CostConfig,
    strategy: Any,
    annualization_days: int,
    allowed_end: date,
) -> dict[str, Any]:
    scenarios: dict[str, BacktestResult] = {}
    for name, multiplier in (("zero", 0.0), ("base", 1.0), ("double", 2.0)):
        scenarios[name] = run_topk_backtest(
            predictions,
            market,
            strategy=strategy,
            costs=_scaled_costs(costs, multiplier),
            annualization_days=annualization_days,
            allowed_end=allowed_end,
        )
    zero_return = float(scenarios["zero"].metrics["total_return"] or 0.0)
    base_return = float(scenarios["base"].metrics["total_return"] or 0.0)
    sign_stable = not (
        (zero_return > 0 and base_return <= 0) or (zero_return < 0 and base_return >= 0)
    )
    return {
        "scenarios": {
            key: {
                "metrics": value.metrics,
                "constraints": value.constraints,
            }
            for key, value in scenarios.items()
        },
        "base_cost_sign_stable": sign_stable,
    }


def _scaled_costs(costs: CostConfig, multiplier: float) -> CostConfig:
    rules = [
        CostRule(
            effective_from=rule.effective_from,
            effective_to=rule.effective_to,
            commission_rate=rule.commission_rate * multiplier,
            minimum_commission=rule.minimum_commission * multiplier,
            stamp_duty_rate_buy=rule.stamp_duty_rate_buy * multiplier,
            stamp_duty_rate_sell=rule.stamp_duty_rate_sell * multiplier,
            transfer_fee_rate_buy=rule.transfer_fee_rate_buy * multiplier,
            transfer_fee_rate_sell=rule.transfer_fee_rate_sell * multiplier,
            commission_assumption=rule.commission_assumption,
            sources=rule.sources,
        )
        for rule in costs.rules
    ]
    return costs.model_copy(update={"rules": rules})


def _promotion_checks(
    metrics: dict[str, Any],
    leakage: dict[str, Any],
    max_accepted_correlation: float | None,
    cost_sensitivity: dict[str, Any],
    config: EvaluationConfig,
) -> dict[str, bool]:
    threshold = config.thresholds
    consistency = metrics["stability"]["direction_consistency"]
    return {
        "coverage": float(metrics["coverage"]) >= threshold.minimum_coverage,
        "rank_ic": metrics["mean_rank_ic"] is not None
        and abs(float(metrics["mean_rank_ic"])) >= threshold.minimum_abs_rank_ic,
        "icir": metrics["icir"] is not None
        and abs(float(metrics["icir"])) >= threshold.minimum_abs_icir,
        "direction_consistency": consistency is not None
        and float(consistency) >= threshold.minimum_direction_consistency,
        "accepted_correlation": max_accepted_correlation is None
        or max_accepted_correlation <= threshold.maximum_abs_accepted_correlation,
        "leakage": bool(leakage["passed"]) if threshold.require_leakage_pass else True,
        "cost_sign_stability": bool(cost_sensitivity["base_cost_sign_stable"])
        if threshold.require_cost_sign_stability
        else True,
    }


def _latest_snapshot(data_dir: Path) -> str:
    path = data_dir / "state" / "latest_snapshot.txt"
    if not path.is_file():
        raise ValueError("no latest snapshot; run make data-bootstrap first")
    return path.read_text(encoding="utf-8").strip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _git_identity() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
