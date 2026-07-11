from __future__ import annotations

import numpy as np
import pandas as pd


def compute(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["instrument", "trade_date"], kind="stable")
    volume = pd.to_numeric(ordered["volume"], errors="coerce")
    lagged = volume.groupby(ordered["instrument"], sort=False).shift(1)
    baseline = lagged.groupby(ordered["instrument"], sort=False).transform(
        lambda series: series.rolling(window=10, min_periods=10).mean()
    )
    value = volume / baseline - 1.0
    value = value.where(np.isfinite(value))
    return ordered[["trade_date", "instrument"]].assign(value=value)
