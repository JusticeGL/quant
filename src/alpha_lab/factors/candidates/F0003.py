from __future__ import annotations

import numpy as np
import pandas as pd


def compute(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["instrument", "trade_date"], kind="stable")
    close = pd.to_numeric(ordered["close"], errors="coerce")
    returns = close.groupby(ordered["instrument"], sort=False).pct_change(
        fill_method=None
    )
    value = returns.groupby(ordered["instrument"], sort=False).transform(
        lambda series: series.rolling(window=20, min_periods=20).std(ddof=1)
    )
    value = value.where(np.isfinite(value))
    return ordered[["trade_date", "instrument"]].assign(value=value)
