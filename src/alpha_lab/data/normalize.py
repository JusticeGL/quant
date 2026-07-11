from __future__ import annotations

import pandas as pd

MINIMUM_COLUMNS = (
    "trade_date",
    "instrument",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "adj_factor",
    "suspend",
    "limit_up",
    "limit_down",
    "is_st",
    "list_date",
    "delist_date",
    "source",
    "ingested_at",
)

AKSHARE_COLUMN_MAP = {
    "日期": "trade_date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
}

BAOSTOCK_COLUMN_MAP = {
    "date": "trade_date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
}


def to_qlib_instrument(symbol: str) -> str:
    if len(symbol) != 6 or not symbol.isdigit():
        raise ValueError(f"unsupported A-share symbol: {symbol}")
    if symbol.startswith(("6", "68")):
        return f"SH{symbol}"
    if symbol.startswith(("0", "3")):
        return f"SZ{symbol}"
    if symbol.startswith(("4", "8")):
        return f"BJ{symbol}"
    raise ValueError(f"cannot infer exchange for symbol: {symbol}")


def normalize_akshare_daily(
    raw: pd.DataFrame, *, symbol: str, ingested_at: str
) -> pd.DataFrame:
    missing = sorted(set(AKSHARE_COLUMN_MAP) - set(raw.columns))
    if missing:
        raise ValueError(f"AKShare response is missing columns: {missing}")

    frame = raw.loc[:, list(AKSHARE_COLUMN_MAP)].rename(columns=AKSHARE_COLUMN_MAP)
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"], errors="raise"
    ).dt.normalize()
    for field in ("open", "high", "low", "close", "volume", "amount"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce").astype("float64")
    # Eastmoney reports A-share volume in hands; normalize to shares.
    frame["volume"] = frame["volume"] * 100.0

    frame.insert(1, "instrument", to_qlib_instrument(symbol))
    frame["adj_factor"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    for field in ("suspend", "limit_up", "limit_down", "is_st"):
        frame[field] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    for field in ("list_date", "delist_date"):
        frame[field] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    frame["source"] = "akshare.stock_zh_a_hist"
    frame["ingested_at"] = pd.to_datetime(ingested_at, utc=True)

    return (
        frame.loc[:, list(MINIMUM_COLUMNS)]
        .sort_values(["trade_date", "instrument"], kind="stable")
        .reset_index(drop=True)
    )


def normalize_baostock_daily(
    raw: pd.DataFrame, *, symbol: str, ingested_at: str
) -> pd.DataFrame:
    status_columns = {"tradestatus", "isST"}
    missing = sorted((set(BAOSTOCK_COLUMN_MAP) | status_columns) - set(raw.columns))
    if missing:
        raise ValueError(f"Baostock response is missing columns: {missing}")

    frame = raw.loc[:, list(BAOSTOCK_COLUMN_MAP)].rename(columns=BAOSTOCK_COLUMN_MAP)
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"], errors="raise"
    ).dt.normalize()
    for field in ("open", "high", "low", "close", "volume", "amount"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce").astype("float64")

    frame.insert(1, "instrument", to_qlib_instrument(symbol))
    frame["adj_factor"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    frame["suspend"] = (
        raw["tradestatus"].astype(str).map({"1": False, "0": True}).astype("boolean")
    )
    for field in ("limit_up", "limit_down"):
        frame[field] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    frame["is_st"] = (
        raw["isST"].astype(str).map({"1": True, "0": False}).astype("boolean")
    )
    for field in ("list_date", "delist_date"):
        frame[field] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    frame["source"] = "baostock.query_history_k_data_plus"
    frame["ingested_at"] = pd.to_datetime(ingested_at, utc=True)

    return (
        frame.loc[:, list(MINIMUM_COLUMNS)]
        .sort_values(["trade_date", "instrument"], kind="stable")
        .reset_index(drop=True)
    )
