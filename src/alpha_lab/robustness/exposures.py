from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def calculate_exposures(
    scores: pd.DataFrame,
    market_cap: pd.DataFrame,
    industries: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    size_risk_threshold: float = 0.30,
) -> dict[str, object]:
    if not 0 <= size_risk_threshold <= 1:
        raise ValueError("size risk threshold must be between zero and one")
    base = _evaluation_rows(scores, labels)
    cap = _prepare_cap(market_cap)
    sized = base.merge(
        cap,
        on=["trade_date", "security_id"],
        how="left",
        validate="many_to_one",
    )
    cap_known = sized["cap_known_at"].dt.date <= sized["trade_date"].dt.date
    sized.loc[~cap_known.fillna(False), "total_market_cap_cny"] = np.nan
    sized["log_total_market_cap"] = np.log(sized["total_market_cap_cny"])
    valid_size = sized.dropna(subset=["score", "log_total_market_cap"]).copy()
    size_daily = _daily_correlations(
        valid_size, "score", "log_total_market_cap", method="spearman"
    )
    size_corr = (
        float(np.mean([item["correlation"] for item in size_daily]))
        if size_daily
        else None
    )
    size_yearly: dict[str, float] = {}
    if size_daily:
        daily_frame = pd.DataFrame(size_daily)
        daily_frame["year"] = pd.to_datetime(daily_frame["date"]).dt.year
        size_yearly = {
            str(int(year)): float(value)
            for year, value in daily_frame.groupby("year")["correlation"].mean().items()
        }

    industry_joined = _join_industry(base, industries)
    valid_industry = industry_joined.dropna(
        subset=["score", "label", "industry_id"]
    ).copy()
    industry_summary = (
        valid_industry.groupby("industry_id", sort=True)["score"]
        .agg(observations="size", mean_score="mean")
        .reset_index()
    )
    by_industry = [
        {
            "industry_id": str(row.industry_id),
            "mean_score": float(row.mean_score),
            "observations": int(row.observations),
        }
        for row in industry_summary.itertuples(index=False)
    ]
    dispersion = (
        float(industry_summary["mean_score"].std(ddof=0))
        if len(industry_summary) >= 2
        else None
    )
    sizes = valid_industry.groupby(["trade_date", "industry_id"], sort=False)[
        "score"
    ].transform("count")
    comparable = valid_industry.loc[sizes >= 2].copy()
    comparable["neutral_score"] = comparable.groupby(
        ["trade_date", "industry_id"], sort=False
    )["score"].transform(_standardize)
    comparable = comparable.dropna(subset=["neutral_score"])
    original_ic = _mean_daily_correlation(
        comparable, "score", "label", method="spearman"
    )
    neutral_ic = _mean_daily_correlation(
        comparable, "neutral_score", "label", method="spearman"
    )
    retention = (
        abs(neutral_ic) / abs(original_ic)
        if neutral_ic is not None and original_ic not in (None, 0.0)
        else None
    )
    industry_input_rows = int(len(base.dropna(subset=["score", "label"])))
    industry_matched_rows = int(len(valid_industry))
    industry_excluded_rows = industry_input_rows - industry_matched_rows
    industry_coverage = (
        industry_matched_rows / industry_input_rows if industry_input_rows else 1.0
    )
    return {
        "size": {
            "joined_rows": int(len(valid_size)),
            "correlation": size_corr,
            "risk_threshold": size_risk_threshold,
            "risk_flag": size_corr is not None and abs(size_corr) > size_risk_threshold,
            "uses": "log(total_market_cap_cny)",
            "method": "daily_cross_sectional_spearman",
            "daily": size_daily,
            "yearly": size_yearly,
        },
        "industry": {
            "input_rows": industry_input_rows,
            "matched_rows": industry_matched_rows,
            "excluded_rows": industry_excluded_rows,
            "coverage": industry_coverage,
            "joined_rows": int(len(comparable)),
            "original_joined_rows": int(len(comparable)),
            "original_rank_ic": original_ic,
            "neutral_rank_ic": neutral_ic,
            "abs_rank_ic_retention": retention,
            "minimum_group_size": 2,
            "by_industry": by_industry,
            "mean_score_dispersion": dispersion,
        },
        "missing": {
            "size_rows": int(len(base) - len(valid_size)),
            "industry_rows": int(len(base) - len(valid_industry)),
        },
    }


def _evaluation_rows(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    required_scores = {"trade_date", "instrument", "score"}
    required_labels = {"trade_date", "instrument", "label"}
    if missing := sorted(required_scores - set(scores.columns)):
        raise ValueError(f"scores are missing columns: {missing}")
    if missing := sorted(required_labels - set(labels.columns)):
        raise ValueError(f"labels are missing columns: {missing}")
    left = scores[list(required_scores)].copy()
    right = labels[list(required_labels)].copy()
    for frame in (left, right):
        frame["trade_date"] = pd.to_datetime(
            frame["trade_date"], errors="raise"
        ).dt.normalize()
    if (
        left.duplicated(["trade_date", "instrument"]).any()
        or right.duplicated(["trade_date", "instrument"]).any()
    ):
        raise ValueError("scores and labels must have unique keys")
    base = left.merge(
        right,
        on=["trade_date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    base["security_id"] = base["instrument"].map(_security_id)
    return base


def _prepare_cap(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "trade_date",
        "security_id",
        "total_market_cap_cny",
        "known_at",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"market cap is missing columns: {missing}")
    result = frame[list(required)].copy()
    result["trade_date"] = pd.to_datetime(
        result["trade_date"], errors="raise"
    ).dt.normalize()
    result["cap_known_at"] = pd.to_datetime(
        result.pop("known_at"), errors="raise", utc=True
    )
    result["total_market_cap_cny"] = pd.to_numeric(
        result["total_market_cap_cny"], errors="coerce"
    ).where(lambda value: value > 0)
    if result.duplicated(["trade_date", "security_id"]).any():
        raise ValueError("market cap has duplicate exact-date keys")
    return result


def _join_industry(base: pd.DataFrame, industries: pd.DataFrame) -> pd.DataFrame:
    required = {
        "industry_id",
        "security_id",
        "effective_from",
        "effective_to",
        "known_at",
    }
    if missing := sorted(required - set(industries.columns)):
        raise ValueError(f"industries are missing columns: {missing}")
    intervals = industries[list(required)].copy()
    intervals["effective_from"] = pd.to_datetime(
        intervals["effective_from"], errors="raise"
    ).dt.normalize()
    intervals["effective_to"] = pd.to_datetime(
        intervals["effective_to"], errors="coerce"
    ).dt.normalize()
    intervals["industry_known_at"] = pd.to_datetime(
        intervals.pop("known_at"), errors="raise", utc=True
    )
    joined = base.merge(intervals, on="security_id", how="left")
    trade = joined["trade_date"]
    valid = (
        (joined["effective_from"] <= trade)
        & (joined["effective_to"].isna() | (trade <= joined["effective_to"]))
        & (joined["industry_known_at"].dt.date <= trade.dt.date)
    )
    joined = joined.loc[valid].copy()
    if joined.duplicated(["trade_date", "instrument"]).any():
        raise ValueError("multiple point-in-time industries match one score row")
    return joined


def _standardize(values: pd.Series[Any]) -> pd.Series[Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    deviation = float(numeric.std(ddof=0))
    if deviation == 0.0 or not math.isfinite(deviation):
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (numeric - float(numeric.mean())) / deviation


def _mean_daily_correlation(
    frame: pd.DataFrame, left: str, right: str, *, method: str
) -> float | None:
    rows = _daily_correlations(frame, left, right, method=method)
    return float(np.mean([item["correlation"] for item in rows])) if rows else None


def _daily_correlations(
    frame: pd.DataFrame, left: str, right: str, *, method: str
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for _, part in frame.groupby("trade_date", sort=True):
        valid = part[[left, right]].dropna()
        if len(valid) < 2:
            continue
        value = float(valid[left].corr(valid[right], method=method))
        if math.isfinite(value):
            values.append(
                {
                    "date": pd.Timestamp(part["trade_date"].iloc[0]).date().isoformat(),
                    "correlation": value,
                    "count": int(len(valid)),
                }
            )
    return values


def _security_id(instrument: object) -> str:
    value = str(instrument)
    exchange = {"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}.get(value[:2])
    if exchange is None or len(value) != 8 or not value[2:].isdigit():
        raise ValueError(f"unsupported instrument: {value}")
    return f"CN:{exchange}:{value[2:]}"
