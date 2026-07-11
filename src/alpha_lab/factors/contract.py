from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

FACTOR_KEY_COLUMNS = ("trade_date", "instrument")
FACTOR_OUTPUT_COLUMNS = (*FACTOR_KEY_COLUMNS, "value")
ALLOWED_INPUTS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "adj_factor",
        "suspend",
        "limit_up",
        "limit_down",
        "is_st",
        "list_date",
        "delist_date",
    }
)


class FactorMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factor_id: str = Field(pattern=r"^F[0-9]{4}$")
    name: str = Field(min_length=3)
    hypothesis: str = Field(min_length=10)
    formula: str = Field(min_length=3)
    inputs: list[str] = Field(min_length=1)
    lookback: int = Field(ge=1)
    direction: Literal[-1, 1]
    family: str = Field(min_length=2)
    author: str = Field(min_length=2)
    parent_factor_ids: list[str]
    created_at: datetime
    status: Literal["reference", "candidate", "accepted", "rejected"]

    @field_validator("inputs")
    @classmethod
    def inputs_are_known_and_unique(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - ALLOWED_INPUTS)
        if unknown:
            raise ValueError(f"unknown factor inputs: {unknown}")
        if len(value) != len(set(value)):
            raise ValueError("factor inputs must be unique")
        return value

    @field_validator("parent_factor_ids")
    @classmethod
    def parent_ids_are_valid(cls, value: list[str]) -> list[str]:
        invalid = [item for item in value if not item.startswith("F")]
        if invalid:
            raise ValueError(f"invalid parent factor IDs: {invalid}")
        return value


FactorFunction = Callable[[pd.DataFrame], pd.DataFrame]


class FactorCandidate:
    def __init__(
        self,
        metadata: FactorMetadata,
        compute: FactorFunction,
        source_path: Path,
        metadata_path: Path,
        source_sha256: str,
        metadata_sha256: str,
    ) -> None:
        self.metadata = metadata
        self.compute = compute
        self.source_path = source_path
        self.metadata_path = metadata_path
        self.source_sha256 = source_sha256
        self.metadata_sha256 = metadata_sha256


def validate_factor_output(
    candidate: FactorCandidate, market: pd.DataFrame
) -> pd.DataFrame:
    columns = [*FACTOR_KEY_COLUMNS, *candidate.metadata.inputs]
    missing = sorted(set(columns) - set(market.columns))
    if missing:
        raise ValueError(
            f"{candidate.metadata.factor_id} missing declared inputs: {missing}"
        )
    restricted_input = market.loc[:, columns].copy(deep=True)
    before = pd.util.hash_pandas_object(restricted_input, index=True).sum()
    output = candidate.compute(restricted_input)
    after = pd.util.hash_pandas_object(restricted_input, index=True).sum()
    if before != after:
        raise ValueError(f"{candidate.metadata.factor_id} mutated its input")
    if not isinstance(output, pd.DataFrame):
        raise TypeError(f"{candidate.metadata.factor_id} must return a DataFrame")
    if list(output.columns) != list(FACTOR_OUTPUT_COLUMNS):
        raise ValueError(
            f"{candidate.metadata.factor_id} output columns must be "
            f"{list(FACTOR_OUTPUT_COLUMNS)}"
        )
    if output.duplicated(list(FACTOR_KEY_COLUMNS)).any():
        raise ValueError(f"{candidate.metadata.factor_id} returned duplicate keys")
    input_keys = pd.MultiIndex.from_frame(restricted_input[list(FACTOR_KEY_COLUMNS)])
    output_keys = pd.MultiIndex.from_frame(output[list(FACTOR_KEY_COLUMNS)])
    if not output_keys.isin(input_keys).all():
        raise ValueError(f"{candidate.metadata.factor_id} returned unknown keys")
    numeric = pd.to_numeric(output["value"], errors="coerce")
    non_numeric = output["value"].notna() & numeric.isna()
    if non_numeric.any():
        raise ValueError(f"{candidate.metadata.factor_id} returned non-numeric values")
    finite = numeric.dropna().map(
        lambda item: bool(pd.notna(item) and abs(item) < float("inf"))
    )
    if not finite.all():
        raise ValueError(f"{candidate.metadata.factor_id} returned infinite values")
    result = output.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
    result["value"] = numeric.astype(float)
    return result.sort_values(list(FACTOR_KEY_COLUMNS), kind="stable").reset_index(
        drop=True
    )
