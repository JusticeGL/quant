from __future__ import annotations

import pandas as pd


def compute(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["instrument", "trade_date"], kind="stable")
    value = ordered.groupby("instrument")["close"].shift(-1)
    return ordered[["trade_date", "instrument"]].assign(value=value)
