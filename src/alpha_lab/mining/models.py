from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alpha_lab.factors.contract import FactorMetadata


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Hypothesis(StrictModel):
    schema_version: Literal[1]
    run_id: str = Field(min_length=3)
    round_number: int = Field(ge=1)
    factor_id: str = Field(pattern=r"^F[0-9]{4}$")
    title: str = Field(min_length=5)
    hypothesis: str = Field(min_length=10)
    rationale: str = Field(min_length=10)
    primary_change: Literal["new_factor", "operator", "window", "combination"]
    changed_variable: str = Field(min_length=2)
    parent_factor_ids: list[str]
    inputs: list[str] = Field(min_length=1)
    lookback: int = Field(ge=1)
    direction: Literal[-1, 1]
    family: str = Field(min_length=2)
    formula: str = Field(min_length=3)
    expected_effect: str = Field(min_length=5)
    falsification_criteria: list[str] = Field(min_length=1)
    created_at: datetime


class CandidateProposal(StrictModel):
    schema_version: Literal[1]
    hypothesis: Hypothesis
    metadata: FactorMetadata
    source_code: str = Field(min_length=40)

    @model_validator(mode="after")
    def hypothesis_matches_metadata(self) -> CandidateProposal:
        hypothesis = self.hypothesis
        metadata = self.metadata
        comparisons = {
            "factor_id": (hypothesis.factor_id, metadata.factor_id),
            "hypothesis": (hypothesis.hypothesis, metadata.hypothesis),
            "formula": (hypothesis.formula, metadata.formula),
            "inputs": (hypothesis.inputs, metadata.inputs),
            "lookback": (hypothesis.lookback, metadata.lookback),
            "direction": (hypothesis.direction, metadata.direction),
            "family": (hypothesis.family, metadata.family),
            "parent_factor_ids": (
                hypothesis.parent_factor_ids,
                metadata.parent_factor_ids,
            ),
        }
        mismatches = [key for key, pair in comparisons.items() if pair[0] != pair[1]]
        if mismatches:
            raise ValueError(f"hypothesis/metadata mismatch: {mismatches}")
        if metadata.status != "candidate":
            raise ValueError("mined factor metadata status must be candidate")
        return self


class MiningDecision(StrictModel):
    schema_version: Literal[1]
    run_id: str
    round_number: int = Field(ge=1)
    factor_id: str = Field(pattern=r"^F[0-9]{4}$")
    decision: Literal["ACCEPT", "REJECT", "ERROR"]
    rationale: str = Field(min_length=5)
    passed_checks: list[str]
    failed_checks: list[str]
    eligible_for_review: bool
    human_approval_required: Literal[True]
    factor_result_sha256: str | None


class MiningConfig(StrictModel):
    schema_version: Literal[1]
    policy_id: str
    default_rounds: int = Field(ge=1)
    maximum_rounds: int = Field(ge=1)
    candidate_id_minimum: int = Field(ge=1, le=9999)
    candidate_id_maximum: int = Field(ge=1, le=9999)
    maximum_lookback: int = Field(ge=1)
    allowed_primary_changes: list[
        Literal["new_factor", "operator", "window", "combination"]
    ]
    require_human_approval_for_acceptance: Literal[True]
    notes: str

    @model_validator(mode="after")
    def ranges_are_ordered(self) -> MiningConfig:
        if self.maximum_rounds < self.default_rounds:
            raise ValueError("maximum_rounds must cover default_rounds")
        if self.candidate_id_maximum < self.candidate_id_minimum:
            raise ValueError("candidate ID range is invalid")
        if len(self.allowed_primary_changes) != len(set(self.allowed_primary_changes)):
            raise ValueError("allowed_primary_changes must be unique")
        return self
