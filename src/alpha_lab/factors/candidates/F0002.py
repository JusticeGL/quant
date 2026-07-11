from __future__ import annotations

import numpy as np
import pandas as pd


def compute(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["instrument", "trade_date"], kind="stable")
    close = pd.to_numeric(ordered["close"], errors="coerce")
    lagged = close.groupby(ordered["instrument"], sort=False).shift(5)
    value = close / lagged - 1.0
    value = value.where(np.isfinite(value))
    return ordered[["trade_date", "instrument"]].assign(value=value)
