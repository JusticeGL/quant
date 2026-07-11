from __future__ import annotations

import numpy as np
import pandas as pd


def compute(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["instrument", "trade_date"], kind="stable")
    close = pd.to_numeric(ordered["close"], errors="coerce")
    amount = pd.to_numeric(ordered["amount"], errors="coerce")
    lagged_close = close.groupby(ordered["instrument"], sort=False).shift(10)
    momentum = close / lagged_close - 1.0
    average_amount = amount.groupby(ordered["instrument"], sort=False).transform(
        lambda series: series.rolling(window=10, min_periods=10).mean()
    )
    value = momentum / np.log1p(average_amount)
    value = value.where(np.isfinite(value))
    return ordered[["trade_date", "instrument"]].assign(value=value)
