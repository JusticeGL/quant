from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TushareSourceConfig(StrictModel):
    provider: Literal["tushare"]
    maximum_concurrency: int = Field(default=1, ge=1, le=4)
    request_timeout_seconds: float = Field(gt=0, le=120)
    max_attempts: int = Field(ge=1, le=10)
    retry_delay_seconds: float = Field(ge=0, le=60)
    request_interval_seconds: float = Field(ge=0, le=10)


class ResearchEndpoints(StrictModel):
    security_master: Literal["stock_basic"]
    trading_calendar: Literal["trade_cal"]
    membership_primary: Literal["index_member_all"]
    membership_fallback: Literal["index_weight"]
    daily_bar: Literal["daily"]
    adjustment_factor: Literal["adj_factor"]
    suspension: Literal["suspend_d"]
    name_history: Literal["namechange"]


class ResearchDataConfig(StrictModel):
    schema_version: Literal[1]
    dataset_id: str = Field(min_length=3)
    index_code: Literal["000300.SH"]
    start_date: date
    end_date: date
    membership_method: Literal["interval_or_weight_observation"]
    stock_statuses: list[Literal["L", "D", "P", "G"]] = Field(min_length=1)
    calendar_exchange: Literal["SSE"]
    maximum_symbols: int = Field(ge=300, le=2000)
    source: TushareSourceConfig
    endpoints: ResearchEndpoints

    @model_validator(mode="after")
    def validate_scope(self) -> ResearchDataConfig:
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        if self.index_code != "000300.SH":
            raise ValueError("Phase 5 is bounded to 000300.SH")
        if len(self.stock_statuses) != len(set(self.stock_statuses)):
            raise ValueError("stock_statuses must be unique")
        return self


def load_research_data_config(config_dir: Path) -> ResearchDataConfig:
    path = config_dir / "research_data.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ResearchDataConfig.model_validate(document)
