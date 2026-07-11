from __future__ import annotations

import numpy as np
import pandas as pd


def compute(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["instrument", "trade_date"], kind="stable")
    high = pd.to_numeric(ordered["high"], errors="coerce")
    low = pd.to_numeric(ordered["low"], errors="coerce")
    close = pd.to_numeric(ordered["close"], errors="coerce")
    spread = high - low
    raw = ((close - low) / spread - 0.5).where(spread > 0.0)
    value = raw.groupby(ordered["instrument"], sort=False).transform(
        lambda series: series.rolling(window=5, min_periods=5).mean()
    )
    value = value.where(np.isfinite(value))
    return ordered[["trade_date", "instrument"]].assign(value=value)
