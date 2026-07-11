from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_coverage: float = Field(ge=0, le=1)
    minimum_abs_rank_ic: float = Field(ge=0)
    minimum_abs_icir: float = Field(ge=0)
    minimum_direction_consistency: float = Field(ge=0, le=1)
    maximum_abs_accepted_correlation: float = Field(ge=0, le=1)
    require_leakage_pass: bool
    require_cost_sign_stability: bool


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    policy_id: str
    locked: Literal[True]
    engineering_only: Literal[True]
    group_count: int = Field(ge=2, le=10)
    winsor_quantile_lower: float = Field(ge=0, lt=0.5)
    winsor_quantile_upper: float = Field(gt=0.5, le=1)
    standardize: bool
    annualization_days: int = Field(ge=1)
    thresholds: Thresholds
    notes: str

    @model_validator(mode="after")
    def quantiles_are_ordered(self) -> EvaluationConfig:
        if self.winsor_quantile_lower >= self.winsor_quantile_upper:
            raise ValueError("winsor quantiles must be ordered")
        return self


def load_evaluation_config(path: Path) -> tuple[EvaluationConfig, str]:
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = EvaluationConfig.model_validate(raw)
    encoded = json.dumps(
        raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return config, hashlib.sha256(encoded).hexdigest()
