from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def analyze_signals(
    predictions: pd.DataFrame, annualization_days: int
) -> dict[str, Any]:
    required = {"datetime", "instrument", "score", "label"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions are missing columns: {missing}")

    valid = predictions.dropna(subset=["score", "label"]).copy()
    daily_rows: list[dict[str, object]] = []
    grouped_spreads: list[float] = []
    for trade_date, part in valid.groupby("datetime", sort=True):
        if len(part) < 2:
            continue
        ic = float(part["score"].corr(part["label"], method="pearson"))
        rank_ic = float(part["score"].corr(part["label"], method="spearman"))
        daily_rows.append(
            {
                "date": pd.Timestamp(trade_date).date().isoformat(),
                "ic": _finite_or_none(ic),
                "rank_ic": _finite_or_none(rank_ic),
                "count": len(part),
            }
        )
        ranks = part["score"].rank(method="first")
        bucket_count = min(5, len(part))
        buckets = pd.qcut(ranks, q=bucket_count, labels=False)
        returns = part.assign(bucket=buckets).groupby("bucket")["label"].mean()
        if len(returns) >= 2:
            grouped_spreads.append(float(returns.iloc[-1] - returns.iloc[0]))

    ic_values = np.asarray(
        [row["ic"] for row in daily_rows if row["ic"] is not None], dtype=float
    )
    rank_values = np.asarray(
        [row["rank_ic"] for row in daily_rows if row["rank_ic"] is not None],
        dtype=float,
    )
    coverage = float(len(valid) / len(predictions)) if len(predictions) else 0.0
    error = valid["score"].to_numpy(dtype=float) - valid["label"].to_numpy(dtype=float)
    return {
        "row_count": len(predictions),
        "valid_row_count": len(valid),
        "date_count": int(valid["datetime"].nunique()),
        "instrument_count": int(valid["instrument"].nunique()),
        "coverage": coverage,
        "mean_ic": _mean_or_none(ic_values),
        "mean_rank_ic": _mean_or_none(rank_values),
        "icir": _information_ratio(ic_values, annualization_days),
        "rank_icir": _information_ratio(rank_values, annualization_days),
        "positive_ic_ratio": (
            float(np.mean(ic_values > 0)) if ic_values.size else None
        ),
        "mean_top_bottom_spread": (
            float(np.mean(grouped_spreads)) if grouped_spreads else None
        ),
        "rmse": float(math.sqrt(np.mean(np.square(error)))) if error.size else None,
        "daily": daily_rows,
    }


def _information_ratio(values: np.ndarray, annualization_days: int) -> float | None:
    if values.size < 2:
        return None
    standard_deviation = float(np.std(values, ddof=1))
    if standard_deviation == 0:
        return None
    return float(np.mean(values) / standard_deviation * math.sqrt(annualization_days))


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if values.size else None


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None
