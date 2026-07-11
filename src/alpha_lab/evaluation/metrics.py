from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from alpha_lab.evaluation.config import EvaluationConfig


def prepare_factor_values(
    values: pd.DataFrame, direction: int, config: EvaluationConfig
) -> pd.DataFrame:
    frame = values.copy()
    frame["oriented_value"] = pd.to_numeric(frame["value"], errors="coerce") * direction

    def transform(series: pd.Series[Any]) -> pd.Series[Any]:
        valid = series.dropna()
        if valid.empty:
            return series.astype(float)
        lower = float(valid.quantile(config.winsor_quantile_lower))
        upper = float(valid.quantile(config.winsor_quantile_upper))
        clipped = series.clip(lower=lower, upper=upper).astype(float)
        if not config.standardize:
            return clipped
        deviation = float(clipped.std(ddof=0))
        if deviation == 0 or not math.isfinite(deviation):
            return pd.Series(np.nan, index=series.index, dtype=float)
        return (clipped - float(clipped.mean())) / deviation

    frame["score"] = frame.groupby("trade_date", sort=True)["oriented_value"].transform(
        transform
    )
    return frame


def build_forward_labels(market: pd.DataFrame) -> pd.DataFrame:
    ordered = market.sort_values(["instrument", "trade_date"], kind="stable").copy()
    open_price = pd.to_numeric(ordered["open"], errors="coerce")
    grouped = open_price.groupby(ordered["instrument"], sort=False)
    entry = grouped.shift(-1)
    exit_price = grouped.shift(-2)
    label = exit_price / entry - 1.0
    label = label.where(np.isfinite(label))
    return ordered[["trade_date", "instrument"]].assign(label=label)


def calculate_factor_metrics(
    evaluated: pd.DataFrame,
    *,
    expected_rows: int,
    group_count: int,
    annualization_days: int,
) -> dict[str, Any]:
    valid = evaluated.dropna(subset=["score", "label"]).copy()
    daily: list[dict[str, object]] = []
    grouped_rows: list[dict[str, object]] = []
    for trade_date, part in valid.groupby("trade_date", sort=True):
        if len(part) < 2:
            continue
        ic = _finite_or_none(float(part["score"].corr(part["label"], method="pearson")))
        rank_ic = _finite_or_none(
            float(part["score"].corr(part["label"], method="spearman"))
        )
        daily.append(
            {
                "date": pd.Timestamp(trade_date).date().isoformat(),
                "ic": ic,
                "rank_ic": rank_ic,
                "count": len(part),
            }
        )
        groups = min(group_count, len(part))
        ranks = part["score"].rank(method="first")
        buckets = pd.qcut(ranks, q=groups, labels=False) + 1
        for group, value in (
            part.assign(group=buckets).groupby("group")["label"].mean().items()
        ):
            grouped_rows.append(
                {
                    "date": pd.Timestamp(trade_date).date().isoformat(),
                    "group": int(group),
                    "return": float(value),
                }
            )

    ic_values = np.asarray(
        [item["ic"] for item in daily if item["ic"] is not None], dtype=float
    )
    rank_values = np.asarray(
        [item["rank_ic"] for item in daily if item["rank_ic"] is not None],
        dtype=float,
    )
    group_frame = pd.DataFrame(grouped_rows)
    group_returns: dict[str, float] = {}
    monotonicity: float | None = None
    top_bottom: float | None = None
    if not group_frame.empty:
        means = group_frame.groupby("group")["return"].mean().sort_index()
        group_returns = {str(int(key)): float(value) for key, value in means.items()}
        if len(means) >= 2:
            monotonicity = _finite_or_none(
                float(
                    pd.Series(means.index, dtype=float).corr(
                        pd.Series(means.to_numpy(), dtype=float), method="spearman"
                    )
                )
            )
            top_bottom = float(means.iloc[-1] - means.iloc[0])

    stability = _stability(daily)
    distribution = _distribution(evaluated["value"])
    return {
        "expected_row_count": expected_rows,
        "valid_row_count": len(valid),
        "coverage": float(len(valid) / expected_rows) if expected_rows else 0.0,
        "date_count": int(valid["trade_date"].nunique()),
        "instrument_count": int(valid["instrument"].nunique()),
        "mean_ic": _mean(ic_values),
        "ic_std": _std(ic_values),
        "icir": _information_ratio(ic_values, annualization_days),
        "positive_ic_ratio": float(np.mean(ic_values > 0)) if ic_values.size else None,
        "mean_rank_ic": _mean(rank_values),
        "rank_ic_std": _std(rank_values),
        "rank_icir": _information_ratio(rank_values, annualization_days),
        "group_returns": group_returns,
        "group_monotonicity": monotonicity,
        "top_minus_bottom_return": top_bottom,
        "factor_turnover": _factor_turnover(valid),
        "stability": stability,
        "distribution": distribution,
        "daily": daily,
        "group_daily": grouped_rows,
    }


def factor_correlations(
    target: pd.DataFrame, comparisons: dict[str, pd.DataFrame]
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    base = target[["trade_date", "instrument", "score"]].rename(
        columns={"score": "target"}
    )
    for factor_id, frame in sorted(comparisons.items()):
        merged = base.merge(
            frame[["trade_date", "instrument", "score"]].rename(
                columns={"score": "other"}
            ),
            on=["trade_date", "instrument"],
            how="inner",
            validate="one_to_one",
        ).dropna(subset=["target", "other"])
        result[factor_id] = (
            _finite_or_none(float(merged["target"].corr(merged["other"])))
            if len(merged) >= 2
            else None
        )
    return result


def _factor_turnover(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    ranked = frame.copy()
    ranked["rank"] = ranked.groupby("trade_date")["score"].rank(pct=True)
    matrix = ranked.pivot(index="trade_date", columns="instrument", values="rank")
    changes = matrix.sort_index().diff().abs().stack(future_stack=True).dropna()
    return float(changes.mean()) if len(changes) else None


def _stability(daily: list[dict[str, object]]) -> dict[str, Any]:
    frame = pd.DataFrame(daily)
    if frame.empty:
        return {
            "monthly": {},
            "yearly": {},
            "rolling_subperiods": [],
            "direction_consistency": None,
            "regime": {"status": "unavailable", "reason": "no regime policy"},
        }
    frame["date"] = pd.to_datetime(frame["date"])
    valid = frame.dropna(subset=["rank_ic"]).copy()
    monthly = valid.groupby(valid["date"].dt.to_period("M"))["rank_ic"].mean()
    yearly = valid.groupby(valid["date"].dt.year)["rank_ic"].mean()
    index_chunks = np.array_split(np.arange(len(valid)), min(3, len(valid)))
    chunks = [valid.iloc[index] for index in index_chunks if len(index)]
    subperiods = [float(chunk["rank_ic"].mean()) for chunk in chunks]
    overall = float(valid["rank_ic"].mean()) if len(valid) else 0.0
    if not subperiods or overall == 0:
        consistency = None
    else:
        consistency = float(np.mean([value * overall > 0 for value in subperiods]))
    return {
        "monthly": {str(key): float(value) for key, value in monthly.items()},
        "yearly": {str(key): float(value) for key, value in yearly.items()},
        "rolling_subperiods": subperiods,
        "direction_consistency": consistency,
        "regime": {
            "status": "unavailable",
            "reason": "bull/bear regime policy is not defined in Phase 3",
        },
    }


def _distribution(values: pd.Series[Any]) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.replace([np.inf, -np.inf], np.nan)
    valid = finite.dropna()
    quantiles = valid.quantile([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    return {
        "missing_rate": float(finite.isna().mean()) if len(finite) else 1.0,
        "infinite_count": int(np.isinf(numeric.to_numpy(dtype=float)).sum()),
        "quantiles": {str(key): float(value) for key, value in quantiles.items()},
    }


def _information_ratio(values: np.ndarray, annualization_days: int) -> float | None:
    deviation = _std(values)
    if deviation is None or deviation == 0:
        return None
    return float(np.mean(values) / deviation * math.sqrt(annualization_days))


def _mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if values.size else None


def _std(values: np.ndarray) -> float | None:
    return float(np.std(values, ddof=1)) if values.size >= 2 else None


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None
