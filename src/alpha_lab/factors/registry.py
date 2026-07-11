from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alpha_lab.factors.contract import (
    FactorCandidate,
    FactorFunction,
    FactorMetadata,
)


class RegistryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    reference_factor_ids: list[str]
    accepted_factor_ids: list[str]
    rejected_factor_ids: list[str]
    notes: str

    @model_validator(mode="after")
    def statuses_are_disjoint(self) -> RegistryConfig:
        values = [
            *self.reference_factor_ids,
            *self.accepted_factor_ids,
            *self.rejected_factor_ids,
        ]
        if len(values) != len(set(values)):
            raise ValueError("factor registry statuses must be disjoint")
        return self


class FactorRegistry:
    def __init__(self, candidates_dir: Path, registry_path: Path) -> None:
        self.candidates_dir = candidates_dir
        self.registry_path = registry_path
        document = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        self.config = RegistryConfig.model_validate(document)
        self._candidates = self._load_candidates()
        configured = {
            *self.config.reference_factor_ids,
            *self.config.accepted_factor_ids,
            *self.config.rejected_factor_ids,
        }
        missing = configured - set(self._candidates)
        unlisted = set(self._candidates) - configured
        invalid_unlisted = sorted(
            factor_id
            for factor_id in unlisted
            if self._candidates[factor_id].metadata.status != "candidate"
        )
        if missing or invalid_unlisted:
            raise ValueError(
                "factor registry status mismatch: "
                f"missing={sorted(missing)}, non_candidate_unlisted={invalid_unlisted}"
            )

    def all(self) -> list[FactorCandidate]:
        return [self._candidates[key] for key in sorted(self._candidates)]

    def get(self, factor_id: str) -> FactorCandidate:
        try:
            return self._candidates[factor_id]
        except KeyError as error:
            raise KeyError(f"unknown factor ID: {factor_id}") from error

    @property
    def accepted_factor_ids(self) -> frozenset[str]:
        return frozenset(self.config.accepted_factor_ids)

    def _load_candidates(self) -> dict[str, FactorCandidate]:
        candidates: dict[str, FactorCandidate] = {}
        names: set[str] = set()
        for metadata_path in sorted(
            self.candidates_dir.glob("F[0-9][0-9][0-9][0-9].yaml")
        ):
            raw: Any = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
            metadata = FactorMetadata.model_validate(raw)
            if metadata_path.stem != metadata.factor_id:
                raise ValueError(f"factor filename does not match ID: {metadata_path}")
            source_path = metadata_path.with_suffix(".py")
            if not source_path.is_file():
                raise ValueError(f"factor implementation is missing: {source_path}")
            if metadata.factor_id in candidates or metadata.name in names:
                raise ValueError(f"duplicate factor ID or name: {metadata.factor_id}")
            compute = _load_compute(metadata.factor_id, source_path)
            candidates[metadata.factor_id] = FactorCandidate(
                metadata=metadata,
                compute=compute,
                source_path=source_path,
                metadata_path=metadata_path,
                source_sha256=_sha256(source_path),
                metadata_sha256=_sha256(metadata_path),
            )
            names.add(metadata.name)
        if not candidates:
            raise ValueError(f"no factor candidates found in {self.candidates_dir}")
        return candidates


def _load_compute(factor_id: str, source_path: Path) -> FactorFunction:
    module_name = f"alpha_lab_factor_{factor_id.lower()}_{_sha256(source_path)[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load factor module: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compute = getattr(module, "compute", None)
    if not callable(compute):
        raise TypeError(f"factor module must define callable compute(): {source_path}")
    return compute  # type: ignore[no-any-return]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
