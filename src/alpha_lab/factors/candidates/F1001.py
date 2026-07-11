from __future__ import annotations

import numpy as np
import pandas as pd


def compute(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["instrument", "trade_date"], kind="stable")
    open_price = pd.to_numeric(ordered["open"], errors="coerce")
    close = pd.to_numeric(ordered["close"], errors="coerce")
    previous_close = close.groupby(ordered["instrument"], sort=False).shift(1)
    raw = open_price / previous_close - 1.0
    value = raw.groupby(ordered["instrument"], sort=False).transform(
        lambda series: series.rolling(window=5, min_periods=5).mean()
    )
    value = value.where(np.isfinite(value))
    return ordered[["trade_date", "instrument"]].assign(value=value)
