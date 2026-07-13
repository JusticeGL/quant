from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import pytest

from alpha_lab.robustness import io as robustness_io
from alpha_lab.robustness.io import read_pretest_exposures, read_pretest_market

LOCKED_START = date(2026, 1, 1)


@pytest.mark.parametrize("reader", [read_pretest_market, read_pretest_exposures])
@pytest.mark.parametrize("end_before", [LOCKED_START, date(2026, 1, 2)])
def test_pretest_reader_rejects_locked_boundary_before_any_data_reader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reader: Any,
    end_before: date,
) -> None:
    opened: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> object:
        opened.append("opened")
        raise AssertionError("locked data reader must not be called")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(pq, "read_table", forbidden)
    monkeypatch.setattr(duckdb, "connect", forbidden)

    with pytest.raises(PermissionError, match="locked test"):
        reader(tmp_path, "missing-snapshot", end_before)

    assert opened == []


def test_pretest_market_opens_only_canonical_pre2026_parts_and_filters_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_id = "p5-fixture"
    artifacts = [
        _parquet_artifact(
            tmp_path,
            snapshot_id,
            "daily_bar/year=2025/part.parquet",
            pd.DataFrame(
                [
                    _bar("2025-12-30"),
                    _bar("2025-12-31"),
                    _bar("2026-01-02"),
                ]
            ),
            root="research",
        ),
        _parquet_artifact(
            tmp_path,
            snapshot_id,
            "adjustment_factor/year=2025/part.parquet",
            pd.DataFrame(
                [
                    _adjustment("2025-12-30"),
                    _adjustment("2025-12-31"),
                    _adjustment("2026-01-02"),
                ]
            ),
            root="research",
        ),
        _parquet_artifact(
            tmp_path,
            snapshot_id,
            "daily_status/year=2025/part.parquet",
            pd.DataFrame(
                [
                    _status("2025-12-30"),
                    _status("2025-12-31"),
                    _status("2026-01-02"),
                ]
            ),
            root="research",
        ),
        _parquet_artifact(
            tmp_path,
            snapshot_id,
            "daily_bar/year=2026/part.parquet",
            pd.DataFrame([_bar("2026-01-02")]),
            root="research",
        ),
        _parquet_artifact(
            tmp_path,
            snapshot_id,
            "adjustment_factor/year=2026/part.parquet",
            pd.DataFrame([_adjustment("2026-01-02")]),
            root="research",
        ),
        _parquet_artifact(
            tmp_path,
            snapshot_id,
            "daily_status/year=2026/part.parquet",
            pd.DataFrame([_status("2026-01-02")]),
            root="research",
        ),
    ]
    _write_manifest(tmp_path, snapshot_id, "research_market", artifacts)
    real_reader = pd.read_parquet
    opened: list[str] = []

    def guarded(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        value = str(path)
        opened.append(value)
        if "year=2026" in value:
            raise AssertionError("locked partition opened")
        return real_reader(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded)

    result = read_pretest_market(tmp_path, snapshot_id, date(2025, 12, 31))

    assert result["trade_date"].dt.date.tolist() == [date(2025, 12, 30)]
    assert result.loc[0, "instrument"] == "SH600000"
    assert result.loc[0, "volume"] == 10_000.0
    assert result.loc[0, "amount"] == 101_000.0
    assert result.loc[0, "adj_factor"] == 1.2
    assert bool(result.loc[0, "suspend"]) is False
    assert pd.isna(result.loc[0, "limit_up"])
    assert pd.isna(result.loc[0, "limit_down"])
    assert len(opened) == 3
    assert all("year=2025" in path for path in opened)


@pytest.mark.parametrize(
    ("name", "path"),
    [
        (
            "daily_bar/year=2025/copy.parquet",
            "research/p5-fixture/daily_bar/year=2025/copy.parquet",
        ),
        (
            "daily_bar/year=2025/part.parquet",
            "research/p5-fixture/daily_bar/year=2026/part.parquet",
        ),
    ],
)
def test_pretest_market_rejects_extra_or_relabelled_parts_before_parquet_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    path: str,
) -> None:
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not parquet")
    artifact = {
        "name": name,
        "path": path,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "row_count": 1,
    }
    _write_manifest(tmp_path, "p5-fixture", "research_market", [artifact])
    opened = False

    def forbidden(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal opened
        opened = True
        raise AssertionError("invalid part must not be opened")

    monkeypatch.setattr(pd, "read_parquet", forbidden)

    with pytest.raises(ValueError, match="canonical"):
        read_pretest_market(tmp_path, "p5-fixture", date(2025, 12, 31))

    assert opened is False


def test_pretest_market_rejects_canonical_path_symlinked_to_locked_partition(
    tmp_path: Path,
) -> None:
    snapshot_id = "p5-fixture"
    locked = (
        tmp_path / "research" / snapshot_id / "daily_bar" / "year=2026" / "part.parquet"
    )
    locked.parent.mkdir(parents=True)
    pd.DataFrame([_bar("2026-01-02")]).to_parquet(locked, index=False)
    apparent_pretest = (
        tmp_path / "research" / snapshot_id / "daily_bar" / "year=2025" / "part.parquet"
    )
    apparent_pretest.parent.mkdir(parents=True)
    apparent_pretest.symlink_to(locked)
    artifact = {
        "name": "daily_bar/year=2025/part.parquet",
        "path": "research/p5-fixture/daily_bar/year=2025/part.parquet",
        "sha256": hashlib.sha256(locked.read_bytes()).hexdigest(),
        "row_count": 1,
    }
    _write_manifest(tmp_path, snapshot_id, "research_market", [artifact])

    with pytest.raises(ValueError, match="symlink"):
        read_pretest_market(tmp_path, snapshot_id, date(2025, 12, 31))


@pytest.mark.parametrize(
    "dataset", ["daily_bar", "adjustment_factor", "daily_status", "market_cap"]
)
@pytest.mark.parametrize("bad_known_at", ["missing", None, "not-a-timestamp"])
def test_each_dated_dataset_requires_parseable_non_null_known_at(
    tmp_path: Path,
    dataset: str,
    bad_known_at: object,
) -> None:
    reader, snapshot_id = _dated_snapshot_with_known_at(tmp_path, dataset, bad_known_at)

    with pytest.raises(ValueError, match="known_at"):
        reader(tmp_path, snapshot_id, date(2025, 12, 31))


@pytest.mark.parametrize(
    "security_id",
    [
        "CN:SSE:1",
        "CN:SSE:000001",
        "CN:SZSE:600000",
        "CN:BSE:600000",
    ],
)
def test_market_adapter_rejects_noncanonical_exchange_or_code(
    security_id: str,
) -> None:
    with pytest.raises(ValueError, match="unsupported security ID"):
        robustness_io._to_instrument(security_id)


def test_pretest_exposures_skip_locked_market_cap_and_bound_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_id = "p6x-fixture"
    artifacts = [
        _parquet_artifact(
            tmp_path,
            snapshot_id,
            "market_cap/year=2025/part.parquet",
            pd.DataFrame(
                [
                    _market_cap("2025-12-30"),
                    _market_cap("2025-12-31"),
                    _market_cap("2026-01-02"),
                ]
            ),
            root="exposures",
        ),
        _parquet_artifact(
            tmp_path,
            snapshot_id,
            "market_cap/year=2026/part.parquet",
            pd.DataFrame([_market_cap("2026-01-02")]),
            root="exposures",
        ),
        _parquet_artifact(
            tmp_path,
            snapshot_id,
            "industry_membership.parquet",
            pd.DataFrame(
                [
                    _membership("2025-01-01T00:00:00Z"),
                    _membership("2026-01-01T00:00:00Z", industry_id="SW:late"),
                ]
            ),
            root="exposures",
        ),
    ]
    _write_manifest(tmp_path, snapshot_id, "point_in_time_exposure", artifacts)
    real_reader = pd.read_parquet
    opened: list[str] = []

    def guarded(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        value = str(path)
        opened.append(value)
        if "year=2026" in value:
            raise AssertionError("locked partition opened")
        return real_reader(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded)

    market_cap, membership = read_pretest_exposures(
        tmp_path, snapshot_id, date(2025, 12, 31)
    )

    assert market_cap["trade_date"].dt.date.tolist() == [date(2025, 12, 30)]
    assert membership["industry_id"].tolist() == ["SW:bank"]
    assert membership.loc[0, "effective_to"] == pd.Timestamp("2025-12-30")
    assert all("year=2026" not in path for path in opened)


def _bar(trade_date: str) -> dict[str, object]:
    return {
        "trade_date": pd.Timestamp(trade_date),
        "security_id": "CN:SSE:600000",
        "open": 10.0,
        "high": 10.2,
        "low": 9.9,
        "close": 10.1,
        "pre_close": 9.8,
        "volume_shares": 10_000.0,
        "amount_cny": 101_000.0,
        "known_at": pd.Timestamp(trade_date, tz="UTC"),
        "source": "fixture.daily",
    }


def _adjustment(trade_date: str) -> dict[str, object]:
    return {
        "trade_date": pd.Timestamp(trade_date),
        "security_id": "CN:SSE:600000",
        "factor_type": "tushare_adj",
        "adj_factor": 1.2,
        "known_at": pd.Timestamp(trade_date, tz="UTC"),
        "source": "fixture.adj_factor",
    }


def _status(trade_date: str) -> dict[str, object]:
    return {
        "trade_date": pd.Timestamp(trade_date),
        "security_id": "CN:SSE:600000",
        "is_suspended": False,
        "is_st": False,
        "known_at": pd.Timestamp(trade_date, tz="UTC"),
    }


def _market_cap(trade_date: str) -> dict[str, object]:
    return {
        "trade_date": pd.Timestamp(trade_date),
        "security_id": "CN:SSE:600000",
        "total_market_cap_cny": 1_000_000.0,
        "float_market_cap_cny": 800_000.0,
        "known_at": pd.Timestamp(trade_date, tz="UTC"),
        "source": "fixture.daily_basic",
    }


def _membership(known_at: str, *, industry_id: str = "SW:bank") -> dict[str, object]:
    return {
        "industry_id": industry_id,
        "security_id": "CN:SSE:600000",
        "effective_from": pd.Timestamp("2025-01-01"),
        "effective_to": pd.Timestamp("2026-12-31"),
        "known_at": pd.Timestamp(known_at),
        "provenance": "fixture",
    }


def _dated_snapshot_with_known_at(
    data_dir: Path,
    target_dataset: str,
    bad_known_at: object,
) -> tuple[Any, str]:
    frames = {
        "daily_bar": pd.DataFrame([_bar("2025-12-30")]),
        "adjustment_factor": pd.DataFrame([_adjustment("2025-12-30")]),
        "daily_status": pd.DataFrame([_status("2025-12-30")]),
        "market_cap": pd.DataFrame([_market_cap("2025-12-30")]),
    }
    target = frames[target_dataset]
    if bad_known_at == "missing":
        frames[target_dataset] = target.drop(columns="known_at")
    else:
        frames[target_dataset]["known_at"] = target["known_at"].astype(object)
        frames[target_dataset].loc[0, "known_at"] = bad_known_at
    if target_dataset == "market_cap":
        snapshot_id = "p6x-fixture"
        artifacts = [
            _parquet_artifact(
                data_dir,
                snapshot_id,
                "market_cap/year=2025/part.parquet",
                frames["market_cap"],
                root="exposures",
            ),
            _parquet_artifact(
                data_dir,
                snapshot_id,
                "industry_membership.parquet",
                pd.DataFrame([_membership("2025-01-01T00:00:00Z")]),
                root="exposures",
            ),
        ]
        _write_manifest(data_dir, snapshot_id, "point_in_time_exposure", artifacts)
        return read_pretest_exposures, snapshot_id
    snapshot_id = "p5-fixture"
    artifacts = [
        _parquet_artifact(
            data_dir,
            snapshot_id,
            f"{dataset}/year=2025/part.parquet",
            frames[dataset],
            root="research",
        )
        for dataset in ("daily_bar", "adjustment_factor", "daily_status")
    ]
    _write_manifest(data_dir, snapshot_id, "research_market", artifacts)
    return read_pretest_market, snapshot_id


def _parquet_artifact(
    data_dir: Path,
    snapshot_id: str,
    name: str,
    frame: pd.DataFrame,
    *,
    root: str,
) -> dict[str, object]:
    relative = Path(root) / snapshot_id / name
    path = data_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return {
        "name": name,
        "path": relative.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "row_count": len(frame),
    }


def _write_manifest(
    data_dir: Path,
    snapshot_id: str,
    snapshot_type: str,
    artifacts: list[dict[str, object]],
) -> None:
    path = data_dir / "manifests" / snapshot_id / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "snapshot_id": snapshot_id,
        "snapshot_type": snapshot_type,
        "artifacts": artifacts,
    }
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
