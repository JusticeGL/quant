from __future__ import annotations

from datetime import date

import pandas as pd


def universe_as_of(
    securities: pd.DataFrame,
    membership: pd.DataFrame,
    as_of: date,
    *,
    index_id: str = "CN:INDEX:000300.SH",
) -> pd.DataFrame:
    security_required = {"security_id", "list_date", "delist_date"}
    membership_required = {
        "index_id",
        "security_id",
        "effective_from",
        "effective_to",
        "known_at",
    }
    missing_security = sorted(security_required - set(securities.columns))
    missing_membership = sorted(membership_required - set(membership.columns))
    if missing_security or missing_membership:
        raise ValueError(
            "universe input columns are incomplete: "
            f"security={missing_security}, membership={missing_membership}"
        )
    effective_date = pd.Timestamp(as_of)
    known_time = effective_date.tz_localize("UTC")
    known_at = pd.to_datetime(membership["known_at"], utc=True, errors="raise")
    selected = membership.loc[
        (membership["index_id"] == index_id)
        & (pd.to_datetime(membership["effective_from"]) <= effective_date)
        & (
            membership["effective_to"].isna()
            | (pd.to_datetime(membership["effective_to"]) >= effective_date)
        )
        & (known_at <= known_time)
    ].copy()
    merged = selected.merge(
        securities,
        on="security_id",
        how="inner",
        validate="many_to_one",
        suffixes=("_membership", "_security"),
    )
    list_date = pd.to_datetime(merged["list_date"], errors="raise")
    delist_date = pd.to_datetime(merged["delist_date"], errors="coerce")
    merged = merged.loc[
        (list_date <= effective_date)
        & (delist_date.isna() | (delist_date >= effective_date))
    ].copy()
    if merged["security_id"].duplicated().any():
        raise ValueError(
            "overlapping membership intervals produce duplicate securities"
        )
    merged.insert(0, "as_of_date", effective_date)
    return merged.sort_values("security_id", kind="stable").reset_index(drop=True)
