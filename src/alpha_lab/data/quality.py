from __future__ import annotations

from typing import Any, cast

import pandas as pd

from alpha_lab.data.normalize import MINIMUM_COLUMNS

STATUS_FIELDS = (
    "adj_factor",
    "suspend",
    "limit_up",
    "limit_down",
    "is_st",
    "list_date",
    "delist_date",
)
CORE_VALUE_FIELDS = ("open", "high", "low", "close", "volume", "amount")


def build_quality_report(
    frame: pd.DataFrame, *, expected_instruments: set[str] | None = None
) -> dict[str, Any]:
    missing_columns = sorted(set(MINIMUM_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"normalized data is missing columns: {missing_columns}")

    duplicate_mask = frame.duplicated(["trade_date", "instrument"], keep=False)
    invalid_ohlc = (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
        | (frame["volume"] < 0)
        | (frame["amount"] < 0)
    )
    missing_status_fields = [
        field for field in STATUS_FIELDS if frame[field].isna().all()
    ]
    missing_core = {
        field: int(frame[field].isna().sum())
        for field in CORE_VALUE_FIELDS
        if frame[field].isna().any()
    }
    observed_instruments = set(frame["instrument"].astype(str).unique())
    missing_instruments = sorted((expected_instruments or set()) - observed_instruments)

    union_dates = pd.DatetimeIndex(sorted(frame["trade_date"].dropna().unique()))
    missing_dates: dict[str, list[str]] = {}
    for instrument, part in frame.groupby("instrument", sort=True):
        present = pd.DatetimeIndex(part["trade_date"].dropna().unique())
        absent = union_dates.difference(present)
        missing_dates[str(instrument)] = [item.strftime("%Y-%m-%d") for item in absent]

    duplicate_rows = frame.loc[duplicate_mask, ["trade_date", "instrument"]]
    invalid_rows = frame.loc[invalid_ohlc, ["trade_date", "instrument"]]
    has_errors = bool(
        duplicate_mask.any()
        or invalid_ohlc.any()
        or missing_core
        or missing_instruments
    )
    status = "error" if has_errors else "warning" if missing_status_fields else "ok"

    return {
        "schema_version": 1,
        "status": status,
        "row_count": len(frame),
        "instrument_count": int(frame["instrument"].nunique()),
        "date_range": {
            "start": _format_date(frame["trade_date"].min()),
            "end": _format_date(frame["trade_date"].max()),
        },
        "missing_rates": {
            field: round(float(frame[field].isna().mean()), 8)
            for field in MINIMUM_COLUMNS
        },
        "missing_core_values": missing_core,
        "missing_instruments": missing_instruments,
        "missing_status_fields": missing_status_fields,
        "missing_dates_by_instrument": missing_dates,
        "duplicates": {
            "count": int(duplicate_mask.sum()),
            "keys": _key_records(duplicate_rows),
        },
        "invalid_rows": {
            "count": int(invalid_ohlc.sum()),
            "keys": _key_records(invalid_rows),
        },
    }


def _format_date(value: object) -> str | None:
    if pd.isna(value):
        return None
    return cast(str, pd.Timestamp(value).strftime("%Y-%m-%d"))


def _key_records(frame: pd.DataFrame, limit: int = 20) -> list[dict[str, str]]:
    return [
        {
            "trade_date": pd.Timestamp(row.trade_date).strftime("%Y-%m-%d"),
            "instrument": str(row.instrument),
        }
        for row in frame.head(limit).itertuples(index=False)
    ]
