from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D

from alpha_lab.data.config import load_phase1_config
from alpha_lab.data.normalize import normalize_akshare_daily
from alpha_lab.data.qlib_export import export_qlib
from alpha_lab.data.snapshot import RawInput, materialize_snapshot

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "akshare_stock_zh_a_hist.csv"


def sample_frame() -> pd.DataFrame:
    raw = pd.read_csv(FIXTURE, dtype={"股票代码": str})
    first = normalize_akshare_daily(
        raw,
        symbol="600519",
        ingested_at="2026-07-10T00:00:00+00:00",
    )
    second_raw = raw.copy()
    second_raw.loc[:, "股票代码"] = "000001"
    second_raw.loc[:, "开盘"] = second_raw["开盘"] / 100
    second_raw.loc[:, "收盘"] = second_raw["收盘"] / 100
    second_raw.loc[:, "最高"] = second_raw["最高"] / 100
    second_raw.loc[:, "最低"] = second_raw["最低"] / 100
    second = normalize_akshare_daily(
        second_raw.iloc[1:],
        symbol="000001",
        ingested_at="2026-07-10T00:00:00+00:00",
    )
    return pd.concat([first, second], ignore_index=True)


def test_snapshot_and_qlib_export_are_deterministic_and_loadable(
    tmp_path: Path,
) -> None:
    config = load_phase1_config(ROOT / "config")
    raw_file = tmp_path / "source.parquet"
    sample_frame().to_parquet(raw_file, index=False)
    raw_input = RawInput(
        provider="akshare",
        endpoint="stock_zh_a_hist",
        symbol="sample",
        path=raw_file,
        sha256=hashlib.sha256(raw_file.read_bytes()).hexdigest(),
        row_count=5,
        requested_start="2024-01-01",
        requested_end="2024-06-30",
    )

    first = materialize_snapshot(
        tmp_path / "data",
        sample_frame(),
        source=config.source,
        universe=config.universe,
        raw_inputs=[raw_input],
    )
    second = materialize_snapshot(
        tmp_path / "data",
        sample_frame(),
        source=config.source,
        universe=config.universe,
        raw_inputs=[raw_input],
    )

    assert first.snapshot_id == second.snapshot_id
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.silver_path.read_bytes() == second.silver_path.read_bytes()

    export_path = tmp_path / "qlib" / first.snapshot_id
    export_one = export_qlib(first.silver_path, export_path, first.snapshot_id)
    export_two = export_qlib(first.silver_path, export_path, first.snapshot_id)

    assert export_one.content_sha256 == export_two.content_sha256
    assert (export_path / "calendars" / "day.txt").is_file()
    assert (export_path / "instruments" / "all.txt").is_file()

    qlib.init(provider_uri=str(export_path), region=REG_CN)
    loaded = D.features(
        ["SH600519", "SZ000001"],
        ["$close"],
        start_time="2024-01-02",
        end_time="2024-01-04",
        freq="day",
    )
    assert loaded.loc[("SH600519", pd.Timestamp("2024-01-02")), "$close"] == 1710.0
    assert ("SZ000001", pd.Timestamp("2024-01-02")) not in loaded.index
    assert loaded.loc[("SZ000001", pd.Timestamp("2024-01-03")), "$close"] == 17.05
