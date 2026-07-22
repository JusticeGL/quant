from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from alpha_lab.baseline.config import CostConfig, CostRule
from alpha_lab.robustness.config import RobustnessConfig, WalkForwardFold


def build_fold_labels(market: pd.DataFrame, fold: WalkForwardFold) -> pd.DataFrame:
    """Build next-open labels without allowing an entry or exit outside a fold."""
    required = {"trade_date", "instrument", "open"}
    missing = sorted(required - set(market.columns))
    if missing:
        raise ValueError(f"market data is missing label fields: {missing}")
    bounded = market.copy()
    bounded["trade_date"] = pd.to_datetime(
        bounded["trade_date"], errors="raise"
    ).dt.normalize()
    bounded = bounded.loc[
        bounded["trade_date"].dt.date.between(fold.start, fold.end)
    ].sort_values(["instrument", "trade_date"], kind="stable")
    if bounded.duplicated(["trade_date", "instrument"]).any():
        raise ValueError("market data has duplicate label keys")
    grouped = bounded.groupby("instrument", sort=False)
    entry_date = grouped["trade_date"].shift(-1)
    exit_date = grouped["trade_date"].shift(-2)
    open_price = pd.to_numeric(bounded["open"], errors="coerce")
    entry = open_price.groupby(bounded["instrument"], sort=False).shift(-1)
    exit_price = open_price.groupby(bounded["instrument"], sort=False).shift(-2)
    label = (exit_price / entry - 1.0).where(np.isfinite(exit_price / entry - 1.0))
    return (
        bounded[["trade_date", "instrument"]]
        .assign(entry_date=entry_date, exit_date=exit_date, label=label)
        .reset_index(drop=True)
    )


def scale_costs(costs: CostConfig, multiplier: float) -> CostConfig:
    if multiplier < 0 or not np.isfinite(multiplier):
        raise ValueError("cost multiplier must be finite and non-negative")
    fields = (
        "commission_rate",
        "minimum_commission",
        "stamp_duty_rate_buy",
        "stamp_duty_rate_sell",
        "transfer_fee_rate_buy",
        "transfer_fee_rate_sell",
    )
    rules: list[CostRule] = []
    for rule in costs.rules:
        update = {field: getattr(rule, field) * multiplier for field in fields}
        rules.append(rule.model_copy(update=update))
    return costs.model_copy(update={"rules": rules})


def backtest_predictions(evaluated: pd.DataFrame) -> pd.DataFrame:
    required = {
        "trade_date",
        "instrument",
        "score",
        "label",
        "entry_date",
        "exit_date",
    }
    if missing := sorted(required - set(evaluated.columns)):
        raise ValueError(f"evaluated rows are missing backtest fields: {missing}")
    valid = evaluated.dropna(
        subset=["score", "label", "entry_date", "exit_date"]
    ).copy()
    valid["datetime"] = pd.to_datetime(valid.pop("trade_date"), errors="raise")
    valid["entry_date"] = pd.to_datetime(valid["entry_date"], errors="raise")
    valid["exit_date"] = pd.to_datetime(valid["exit_date"], errors="raise")
    if not (
        (valid["datetime"] < valid["entry_date"])
        & (valid["entry_date"] < valid["exit_date"])
    ).all():
        raise ValueError("backtest prediction dates are not strictly ordered")
    return valid[
        ["datetime", "instrument", "score", "label", "entry_date", "exit_date"]
    ].reset_index(drop=True)


def evaluate_gates(
    folds: list[dict[str, Any]],
    cost_sensitivity: dict[str, Any],
    exposures: dict[str, Any],
    config: RobustnessConfig,
) -> dict[str, bool]:
    """Apply only the four approved Phase 6 pre-test gates."""
    consistent = sum(item.get("direction_consistent") is True for item in folds)
    coverage_values = [_finite_float(item.get("coverage")) for item in folds]
    coverage = (
        len(folds) == len(config.walk_forward_folds)
        and all(value is not None for value in coverage_values)
        and all(
            value >= config.minimum_fold_coverage
            for value in coverage_values
            if value is not None
        )
    )
    base_return = _scenario_total_return(cost_sensitivity, 1.0)
    double_return = _scenario_total_return(cost_sensitivity, 2.0)
    no_reversal = (
        base_return is not None
        and double_return is not None
        and not (
            (base_return > 0.0 and double_return <= 0.0)
            or (base_return < 0.0 and double_return >= 0.0)
        )
    )
    retention = _finite_float(
        exposures.get("industry", {}).get("abs_rank_ic_retention")
    )
    return {
        "direction_consistency": consistent
        >= config.minimum_direction_consistent_folds,
        "fold_coverage": coverage,
        "double_cost_direction": no_reversal,
        "industry_neutral_retention": retention is not None
        and retention >= config.minimum_industry_neutral_ic_retention,
    }


def _scenario_total_return(report: dict[str, Any], multiplier: float) -> float | None:
    scenarios = report.get("scenarios", {})
    for key in (str(multiplier), f"{multiplier:g}"):
        value = scenarios.get(key)
        if isinstance(value, dict):
            metrics = value.get("metrics", value)
            result = metrics.get("total_return") if isinstance(metrics, dict) else None
            return _finite_float(result)
    return None


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
