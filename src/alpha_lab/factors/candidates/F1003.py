from __future__ import annotations

import numpy as np
import pandas as pd


def compute(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["instrument", "trade_date"], kind="stable")
    high = pd.to_numeric(ordered["high"], errors="coerce")
    low = pd.to_numeric(ordered["low"], errors="coerce")
    close = pd.to_numeric(ordered["close"], errors="coerce")
    raw = (high - low) / close
    value = raw.groupby(ordered["instrument"], sort=False).transform(
        lambda series: series.rolling(window=10, min_periods=10).mean()
    )
    value = value.where(np.isfinite(value))
    return ordered[["trade_date", "instrument"]].assign(value=value)
