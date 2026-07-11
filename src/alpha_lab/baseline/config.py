from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LabelConfig(StrictModel):
    name: str
    expression: str
    execution_delay_days: int = Field(ge=1)
    holding_days: int = Field(ge=1)


class ModelConfig(StrictModel):
    objective: Literal["regression_l2"]
    n_estimators: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    num_leaves: int = Field(ge=2)
    max_depth: int
    min_child_samples: int = Field(ge=1)
    subsample: float = Field(gt=0, le=1)
    colsample_bytree: float = Field(gt=0, le=1)
    reg_alpha: float = Field(ge=0)
    reg_lambda: float = Field(ge=0)
    n_jobs: int = Field(ge=1)


class StrategyConfig(StrictModel):
    top_k: int = Field(ge=1)
    initial_cash: float = Field(gt=0)
    lot_size: int = Field(ge=1)
    exclude_st: bool
    infer_missing_price_limits: bool
    main_board_limit_ratio: float = Field(gt=0, lt=1)
    st_limit_ratio: float = Field(gt=0, lt=1)
    price_limit_tolerance: float = Field(ge=0, lt=0.1)


class BaselineConfig(StrictModel):
    schema_version: Literal[1]
    experiment_name: str
    random_seed: int
    feature_set: Literal["Alpha158"]
    label: LabelConfig
    model: ModelConfig
    strategy: StrategyConfig
    annualization_days: int = Field(ge=1)
    engineering_only: Literal[True]


class DateRange(StrictModel):
    start: date
    end: date

    @model_validator(mode="after")
    def ordered(self) -> DateRange:
        if self.end < self.start:
            raise ValueError("range end must not precede start")
        return self


class LockedTestRange(DateRange):
    locked: Literal[True]
    access: Literal["human_approval_only"]


class SplitConfig(StrictModel):
    schema_version: Literal[1]
    policy_id: str
    locked: Literal[True]
    engineering_only: Literal[True]
    train: DateRange
    validation: DateRange
    test: LockedTestRange
    notes: str

    @model_validator(mode="after")
    def disjoint_and_ordered(self) -> SplitConfig:
        if not (self.train.end < self.validation.start <= self.validation.end):
            raise ValueError("train and validation ranges must be ordered and disjoint")
        if self.validation.end >= self.test.start:
            raise ValueError("validation must end before locked test begins")
        return self


class CostRule(StrictModel):
    effective_from: date
    effective_to: date | None
    commission_rate: float = Field(ge=0)
    minimum_commission: float = Field(ge=0)
    stamp_duty_rate_buy: float = Field(ge=0)
    stamp_duty_rate_sell: float = Field(ge=0)
    transfer_fee_rate_buy: float = Field(ge=0)
    transfer_fee_rate_sell: float = Field(ge=0)
    commission_assumption: bool
    sources: dict[str, str]

    @model_validator(mode="after")
    def ordered(self) -> CostRule:
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("cost effective_to must not precede effective_from")
        return self


class CostConfig(StrictModel):
    schema_version: Literal[1]
    policy_id: str
    locked: Literal[True]
    currency: Literal["CNY"]
    rules: list[CostRule] = Field(min_length=1)
    notes: str

    @model_validator(mode="after")
    def rules_are_ordered(self) -> CostConfig:
        starts = [rule.effective_from for rule in self.rules]
        if starts != sorted(starts) or len(set(starts)) != len(starts):
            raise ValueError(
                "cost rules must have unique ascending effective_from dates"
            )
        return self

    def rule_for(self, value: date) -> CostRule:
        matches = [
            rule
            for rule in self.rules
            if rule.effective_from <= value
            and (rule.effective_to is None or value <= rule.effective_to)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one cost rule for {value}, got {len(matches)}"
            )
        return matches[0]


class Phase2Config(StrictModel):
    baseline: BaselineConfig
    splits: SplitConfig
    costs: CostConfig
    config_sha256: str
    split_sha256: str
    cost_sha256: str


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_phase2_config(config_dir: Path) -> Phase2Config:
    baseline_raw = _load_yaml(config_dir / "baseline.yaml")
    splits_raw = _load_yaml(config_dir / "splits.yaml")
    costs_raw = _load_yaml(config_dir / "costs.yaml")
    return Phase2Config(
        baseline=BaselineConfig.model_validate(baseline_raw),
        splits=SplitConfig.model_validate(splits_raw),
        costs=CostConfig.model_validate(costs_raw),
        config_sha256=_canonical_hash(
            {"baseline": baseline_raw, "splits": splits_raw, "costs": costs_raw}
        ),
        split_sha256=_canonical_hash(splits_raw),
        cost_sha256=_canonical_hash(costs_raw),
    )
