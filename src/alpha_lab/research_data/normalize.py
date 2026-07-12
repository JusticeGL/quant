from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

EXCHANGE_BY_SUFFIX = {"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}


def to_security_id(ts_code: str) -> str:
    parts = str(ts_code).split(".")
    if len(parts) != 2 or len(parts[0]) != 6 or not parts[0].isdigit():
        raise ValueError(f"invalid Tushare security code: {ts_code}")
    try:
        exchange = EXCHANGE_BY_SUFFIX[parts[1]]
    except KeyError as error:
        raise ValueError(f"unsupported Tushare exchange: {parts[1]}") from error
    return f"CN:{exchange}:{parts[0]}"


def normalize_security_master(raw: pd.DataFrame, *, ingested_at: str) -> pd.DataFrame:
    required = (
        "ts_code",
        "symbol",
        "name",
        "market",
        "exchange",
        "list_status",
        "list_date",
    )
    _require_columns(raw, required, "stock_basic")
    eligible = (
        raw["ts_code"].astype("string").str.fullmatch(r"[0-9]{6}\.(SH|SZ|BJ)", na=False)
    )
    excluded_count = int((~eligible).sum())
    selected = raw.loc[eligible].copy()
    frame = selected.loc[:, list(required)].copy()
    currency = selected.get("curr_type", pd.Series("CNY", index=selected.index))
    delist_date = selected.get("delist_date", pd.Series(pd.NA, index=selected.index))
    frame["curr_type"] = currency.replace("", pd.NA).fillna("CNY")
    frame["delist_date"] = delist_date
    frame.insert(0, "security_id", frame["ts_code"].map(to_security_id))
    frame["list_date"] = _date_series(frame["list_date"], required=True)
    frame["delist_date"] = _date_series(frame["delist_date"], required=False)
    frame["board"] = frame.pop("market").astype("string")
    frame["currency"] = frame.pop("curr_type").astype("string")
    frame["known_at"] = pd.Timestamp(ingested_at)
    if frame["known_at"].dt.tz is None:
        frame["known_at"] = frame["known_at"].dt.tz_localize("UTC")
    else:
        frame["known_at"] = frame["known_at"].dt.tz_convert("UTC")
    frame["source"] = "tushare.stock_basic"
    if frame["security_id"].duplicated().any():
        raise ValueError("stock_basic returned duplicate security_id")
    invalid = frame["delist_date"].notna() & (frame["delist_date"] < frame["list_date"])
    if invalid.any():
        raise ValueError("delist_date must be on or after list_date")
    result = frame.sort_values("security_id", kind="stable").reset_index(drop=True)
    result.attrs["excluded_non_a_share_count"] = excluded_count
    return result


def normalize_name_history(raw: pd.DataFrame) -> pd.DataFrame:
    required = (
        "ts_code",
        "name",
        "start_date",
        "end_date",
        "ann_date",
        "change_reason",
    )
    _require_columns(raw, required, "namechange")
    frame = raw.loc[:, list(required)].copy()
    frame.insert(0, "security_id", frame.pop("ts_code").map(to_security_id))
    frame["effective_from"] = _date_series(frame.pop("start_date"), required=True)
    frame["effective_to"] = _date_series(frame.pop("end_date"), required=False)
    announcement = _date_series(frame.pop("ann_date"), required=False)
    known_date = announcement.fillna(frame["effective_from"])
    frame["announced_at"] = _utc_date(announcement)
    frame["known_at"] = _utc_date(known_date)
    frame["known_at_source"] = np.where(
        announcement.notna(), "announcement_date", "effective_date_fallback"
    )
    compact_name = (
        frame["name"].astype("string").str.upper().str.replace(" ", "", regex=False)
    )
    frame["is_st"] = compact_name.str.match(r"^(?:\*ST|ST|S\*ST|SST)").astype("boolean")
    frame["source"] = "tushare.namechange"
    _validate_interval_order(frame, "name history")
    return frame.sort_values(
        ["security_id", "effective_from"], kind="stable"
    ).reset_index(drop=True)


def normalize_index_membership_intervals(
    raw: pd.DataFrame, index_code: str
) -> pd.DataFrame:
    required = ("index_code", "con_code", "in_date", "out_date")
    _require_columns(raw, required, "index_member_all")
    frame = raw.copy()
    if not (frame["index_code"].astype(str) == index_code).all():
        raise ValueError("index membership response contains an unexpected index")
    result = pd.DataFrame(
        {
            "index_id": f"CN:INDEX:{index_code}",
            "security_id": frame["con_code"].map(to_security_id),
            "effective_from": _date_series(frame["in_date"], required=True),
            "effective_to": _date_series(frame["out_date"], required=False),
        }
    )
    announcement = _date_series(
        frame.get("ann_date", pd.Series(pd.NA, index=frame.index)), required=False
    )
    known_date = announcement.fillna(result["effective_from"])
    result["announced_at"] = _utc_date(announcement)
    result["known_at"] = _utc_date(known_date)
    result["known_at_source"] = np.where(
        announcement.notna(), "announcement_date", "effective_date_fallback"
    )
    result["weight"] = pd.to_numeric(
        frame.get("weight", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    result["membership_method"] = "index_member_all"
    result["source"] = "tushare.index_member_all"
    _validate_interval_order(result, "index membership")
    return result.sort_values(
        ["security_id", "effective_from"], kind="stable"
    ).reset_index(drop=True)


def reconstruct_weight_membership(raw: pd.DataFrame, index_code: str) -> pd.DataFrame:
    required = ("index_code", "con_code", "trade_date", "weight")
    _require_columns(raw, required, "index_weight")
    frame = raw.loc[:, list(required)].copy()
    if not (frame["index_code"].astype(str) == index_code).all():
        raise ValueError("index weight response contains an unexpected index")
    frame["trade_date"] = _date_series(frame["trade_date"], required=True)
    if frame.duplicated(["trade_date", "con_code"]).any():
        raise ValueError("index_weight returned duplicate observation keys")
    observation_dates = sorted(frame["trade_date"].unique())
    date_positions = {value: index for index, value in enumerate(observation_dates)}
    records: list[dict[str, object]] = []
    for con_code, group in frame.groupby("con_code", sort=True):
        ordered = group.sort_values("trade_date", kind="stable")
        positions = [date_positions[value] for value in ordered["trade_date"]]
        segment_start = 0
        for offset in range(1, len(positions) + 1):
            boundary = (
                offset == len(positions)
                or positions[offset] != positions[offset - 1] + 1
            )
            if not boundary:
                continue
            first_row = ordered.iloc[segment_start]
            last_position = positions[offset - 1]
            effective_to = (
                pd.Timestamp(observation_dates[last_position + 1]) - timedelta(days=1)
                if last_position + 1 < len(observation_dates)
                else pd.NaT
            )
            effective_from = pd.Timestamp(first_row["trade_date"])
            records.append(
                {
                    "index_id": f"CN:INDEX:{index_code}",
                    "security_id": to_security_id(str(con_code)),
                    "effective_from": effective_from,
                    "effective_to": effective_to,
                    "announced_at": pd.NaT,
                    "known_at": effective_from.tz_localize("UTC"),
                    "known_at_source": "effective_date_fallback",
                    "weight": float(first_row["weight"]),
                    "membership_method": "index_weight_observation",
                    "source": "tushare.index_weight",
                }
            )
            segment_start = offset
    return (
        pd.DataFrame(records)
        .sort_values(["security_id", "effective_from"], kind="stable")
        .reset_index(drop=True)
    )


def normalize_daily_bars(raw: pd.DataFrame) -> pd.DataFrame:
    required = (
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "vol",
        "amount",
    )
    _require_columns(raw, required, "daily")
    frame = pd.DataFrame(
        {
            "trade_date": _date_series(raw["trade_date"], required=True),
            "security_id": raw["ts_code"].map(to_security_id),
        }
    )
    for field in ("open", "high", "low", "close", "pre_close"):
        frame[field] = pd.to_numeric(raw[field], errors="coerce").astype("float64")
    frame["volume_shares"] = (
        pd.to_numeric(raw["vol"], errors="coerce").astype("float64") * 100.0
    )
    frame["amount_cny"] = (
        pd.to_numeric(raw["amount"], errors="coerce").astype("float64") * 1000.0
    )
    frame["known_at"] = _utc_date(frame["trade_date"])
    frame["source"] = "tushare.daily"
    _reject_duplicate_keys(frame, ["trade_date", "security_id"], "daily")
    return frame.sort_values(["trade_date", "security_id"], kind="stable").reset_index(
        drop=True
    )


def normalize_adjustment_factors(raw: pd.DataFrame) -> pd.DataFrame:
    required = ("ts_code", "trade_date", "adj_factor")
    _require_columns(raw, required, "adj_factor")
    frame = pd.DataFrame(
        {
            "trade_date": _date_series(raw["trade_date"], required=True),
            "security_id": raw["ts_code"].map(to_security_id),
            "factor_type": "tushare_adj",
            "adj_factor": pd.to_numeric(raw["adj_factor"], errors="coerce").astype(
                "float64"
            ),
        }
    )
    if (~np.isfinite(frame["adj_factor"]) | (frame["adj_factor"] <= 0)).any():
        raise ValueError("adjustment factors must be finite and positive")
    frame["known_at"] = _utc_date(frame["trade_date"])
    frame["source"] = "tushare.adj_factor"
    _reject_duplicate_keys(
        frame, ["trade_date", "security_id", "factor_type"], "adj_factor"
    )
    return frame.sort_values(["trade_date", "security_id"], kind="stable").reset_index(
        drop=True
    )


def normalize_suspensions(raw: pd.DataFrame) -> pd.DataFrame:
    required = ("ts_code", "trade_date", "suspend_timing", "suspend_type")
    _require_columns(raw, required, "suspend_d")
    trade_date = _date_series(raw["trade_date"], required=True)
    frame = pd.DataFrame(
        {
            "security_id": raw["ts_code"].map(to_security_id),
            "effective_from": trade_date,
            "effective_to": trade_date,
            "resume_date": pd.NaT,
        }
    )
    frame["announced_at"] = pd.Series(
        pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"
    )
    frame["known_at"] = _utc_date(trade_date)
    frame["known_at_source"] = "effective_date_fallback"
    frame["suspend_type"] = raw["suspend_type"].astype("string")
    frame["suspend_reason"] = raw["suspend_timing"].astype("string")
    frame["source"] = "tushare.suspend_d"
    _validate_interval_order(frame, "suspension")
    return frame.sort_values(
        ["security_id", "effective_from"], kind="stable"
    ).reset_index(drop=True)


def normalize_trading_calendar(raw: pd.DataFrame) -> pd.DataFrame:
    required = ("exchange", "cal_date", "is_open", "pretrade_date")
    _require_columns(raw, required, "trade_cal")
    frame = pd.DataFrame(
        {
            "exchange": raw["exchange"].astype("string"),
            "calendar_date": _date_series(raw["cal_date"], required=True),
            "is_open": raw["is_open"].astype(str).map({"1": True, "0": False}),
            "previous_open_date": _date_series(raw["pretrade_date"], required=False),
        }
    )
    frame["source"] = "tushare.trade_cal"
    _reject_duplicate_keys(frame, ["exchange", "calendar_date"], "trade_cal")
    return frame.sort_values(["calendar_date", "exchange"], kind="stable").reset_index(
        drop=True
    )


def _require_columns(raw: pd.DataFrame, required: tuple[str, ...], source: str) -> None:
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"Tushare {source} response is missing columns: {missing}")


def _date_series(values: pd.Series, *, required: bool) -> pd.Series:
    cleaned = values.replace("", pd.NA)
    result = pd.to_datetime(cleaned, format="%Y%m%d", errors="coerce").dt.normalize()
    if required and result.isna().any():
        raise ValueError("required date field contains missing or invalid values")
    invalid = cleaned.notna() & result.isna()
    if invalid.any():
        raise ValueError("date field contains invalid values")
    return result


def _utc_date(values: pd.Series) -> pd.Series:
    result = pd.to_datetime(values, errors="coerce")
    if result.dt.tz is None:
        return result.dt.tz_localize("UTC")
    return result.dt.tz_convert("UTC")


def _validate_interval_order(frame: pd.DataFrame, label: str) -> None:
    invalid = frame["effective_to"].notna() & (
        frame["effective_to"] < frame["effective_from"]
    )
    if invalid.any():
        raise ValueError(f"{label} effective_to precedes effective_from")


def _reject_duplicate_keys(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    if frame.duplicated(columns).any():
        raise ValueError(f"{label} returned duplicate keys")
