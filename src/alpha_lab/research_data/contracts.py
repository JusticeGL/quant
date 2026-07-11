from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ResearchTables:
    security_master: pd.DataFrame
    security_name_history: pd.DataFrame
    trading_calendar: pd.DataFrame
    index_membership: pd.DataFrame
    daily_bar: pd.DataFrame
    adjustment_factor: pd.DataFrame
    suspension: pd.DataFrame
    daily_status: pd.DataFrame


SECURITY_MASTER_KEY = ("security_id",)
NAME_HISTORY_KEY = ("security_id", "effective_from")
MEMBERSHIP_KEY = ("index_id", "security_id", "effective_from")
DAILY_BAR_KEY = ("trade_date", "security_id")
ADJUSTMENT_FACTOR_KEY = ("trade_date", "security_id", "factor_type")
DAILY_STATUS_KEY = ("trade_date", "security_id")
