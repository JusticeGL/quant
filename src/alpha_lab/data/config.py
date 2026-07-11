from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class DataSourceConfig(BaseModel):
    provider: Literal["akshare"]
    endpoint: Literal["stock_zh_a_hist"]
    fallback_provider: Literal["baostock"] | None = "baostock"
    period: Literal["daily"] = "daily"
    start_date: date
    end_date: date
    adjust: Literal["", "qfq", "hfq"] = ""
    request_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    max_attempts: int = Field(default=3, ge=1, le=10)
    retry_delay_seconds: float = Field(default=2.0, ge=0)
    request_interval_seconds: float = Field(default=0.5, ge=0)

    @model_validator(mode="after")
    def validate_date_range(self) -> DataSourceConfig:
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self


class UniverseSymbol(BaseModel):
    code: str = Field(pattern=r"^\d{6}$")
    name: str = Field(min_length=1)


class UniverseConfig(BaseModel):
    sample_id: str = Field(min_length=1)
    as_of: date
    membership_basis: str = Field(min_length=1)
    research_eligible: Literal[False]
    disclaimer: str = Field(min_length=1)
    symbols: list[UniverseSymbol] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_symbols_and_disclaimer(self) -> UniverseConfig:
        codes = [item.code for item in self.symbols]
        if len(codes) != len(set(codes)):
            raise ValueError("universe symbols must be unique")
        if "survivorship-biased" not in self.disclaimer.lower():
            raise ValueError("Phase 1 disclaimer must state survivorship-biased")
        return self


class Phase1Config(BaseModel):
    source: DataSourceConfig
    universe: UniverseConfig


def _load_yaml(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_phase1_config(config_dir: Path) -> Phase1Config:
    return Phase1Config(
        source=DataSourceConfig.model_validate(
            _load_yaml(config_dir / "data_sources.yaml")
        ),
        universe=UniverseConfig.model_validate(
            _load_yaml(config_dir / "universe.yaml")
        ),
    )
