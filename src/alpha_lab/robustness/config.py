from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alpha_lab.baseline.config import DateRange, LockedTestRange


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExposureEndpoints(StrictModel):
    market_cap: Literal["daily_basic"]
    industry_classification: Literal["index_classify"]
    industry_membership: Literal["index_member_all"]


class ExposureSourceConfig(StrictModel):
    provider: Literal["tushare"]
    classification_standard: Literal["SW2021"]
    maximum_concurrency: int = Field(ge=1, le=4)
    request_timeout_seconds: float = Field(gt=0, le=120)
    max_attempts: int = Field(ge=1, le=10)
    retry_delay_seconds: float = Field(ge=0, le=60)
    request_interval_seconds: float = Field(ge=0, le=10)
    endpoints: ExposureEndpoints


class WalkForwardFold(StrictModel):
    fold_id: str = Field(pattern=r"^wf_[0-9]{4}$")
    start: date
    end: date

    @model_validator(mode="after")
    def ordered(self) -> WalkForwardFold:
        if self.end < self.start:
            raise ValueError("fold end must not precede start")
        return self


class RobustnessConfig(StrictModel):
    schema_version: Literal[1]
    policy_id: str
    phase5_snapshot_id: str
    factor_ids: list[Literal["F1002", "F1003"]]
    warmup: DateRange
    walk_forward_folds: list[WalkForwardFold]
    test: LockedTestRange
    cost_multipliers: list[float]
    minimum_fold_coverage: float = Field(ge=0, le=1)
    minimum_direction_consistent_folds: int = Field(ge=1)
    minimum_industry_neutral_ic_retention: float = Field(ge=0, le=1)
    size_correlation_risk_threshold: float = Field(ge=0, le=1)
    exposure_source: ExposureSourceConfig

    @model_validator(mode="after")
    def validate_locked_policy(self) -> RobustnessConfig:
        if self.test.start != date(2026, 1, 1) or self.test.end != date(2026, 7, 11):
            raise ValueError("Phase 6 locked test range must remain fixed")
        if self.factor_ids != ["F1002", "F1003"]:
            raise ValueError("Phase 6 candidates must be F1002 and F1003")
        if self.cost_multipliers != [0.5, 1.0, 1.5, 2.0]:
            raise ValueError("Phase 6 cost multipliers are locked")
        if self.warmup.end >= self.test.start:
            raise ValueError("warm-up must end before the locked test boundary")
        if len(self.walk_forward_folds) != 5:
            raise ValueError("Phase 6 requires five walk-forward folds")

        previous_end = self.warmup.end
        for fold in self.walk_forward_folds:
            if fold.start <= previous_end:
                raise ValueError("walk-forward folds must be ordered and disjoint")
            if fold.end >= self.test.start:
                raise ValueError("walk-forward fold crosses locked test boundary")
            previous_end = fold.end
        if self.minimum_direction_consistent_folds > len(self.walk_forward_folds):
            raise ValueError("consistent fold minimum exceeds available folds")
        return self


def load_robustness_config(path: Path) -> tuple[RobustnessConfig, str]:
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    config = RobustnessConfig.model_validate(raw)
    canonical = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return config, hashlib.sha256(canonical).hexdigest()
