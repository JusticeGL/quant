from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.research_data.config import load_research_data_config
from alpha_lab.research_data.contracts import ResearchTables
from alpha_lab.research_data.provider import TushareArtifact
from alpha_lab.research_data.snapshot import materialize_research_snapshot

ROOT = Path(__file__).resolve().parents[2]


def _tables() -> ResearchTables:
    security = pd.DataFrame(
        [
            {
                "security_id": "CN:SSE:600001",
                "ts_code": "600001.SH",
                "symbol": "600001",
                "name": "示例退市",
                "exchange": "SSE",
                "list_status": "D",
                "list_date": pd.Timestamp("2010-01-01"),
                "delist_date": pd.Timestamp("2022-12-31"),
            }
        ]
    )
    membership = pd.DataFrame(
        [
            {
                "index_id": "CN:INDEX:000300.SH",
                "security_id": "CN:SSE:600001",
                "effective_from": pd.Timestamp("2021-01-01"),
                "effective_to": pd.Timestamp("2021-12-31"),
                "known_at": pd.Timestamp("2021-01-01", tz="UTC"),
                "weight": 0.5,
            }
        ]
    )
    return ResearchTables(
        security_master=security,
        security_name_history=pd.DataFrame(
            columns=["security_id", "effective_from", "effective_to", "is_st"]
        ),
        trading_calendar=pd.DataFrame(
            [
                {
                    "exchange": "SSE",
                    "calendar_date": pd.Timestamp("2021-01-04"),
                    "is_open": True,
                    "previous_open_date": pd.Timestamp("2020-12-31"),
                    "source": "tushare.trade_cal",
                }
            ]
        ),
        index_membership=membership,
        daily_bar=pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2021-01-04"),
                    "security_id": "CN:SSE:600001",
                    "open": 10.0,
                    "high": 10.2,
                    "low": 9.9,
                    "close": 10.1,
                    "pre_close": 9.8,
                    "volume_shares": 10_000.0,
                    "amount_cny": 200_000.0,
                    "known_at": pd.Timestamp("2021-01-04", tz="UTC"),
                    "source": "tushare.daily",
                }
            ]
        ),
        adjustment_factor=pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2021-01-04"),
                    "security_id": "CN:SSE:600001",
                    "factor_type": "tushare_adj",
                    "adj_factor": 1.2,
                    "known_at": pd.Timestamp("2021-01-04", tz="UTC"),
                    "source": "tushare.adj_factor",
                }
            ]
        ),
        suspension=pd.DataFrame(
            columns=["security_id", "effective_from", "effective_to"]
        ),
        daily_status=pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2021-01-04"),
                    "security_id": "CN:SSE:600001",
                    "is_suspended": pd.NA,
                    "is_st": pd.NA,
                    "known_at": pd.Timestamp("2021-01-04", tz="UTC"),
                }
            ]
        ),
    )


def _raw_input(tmp_path: Path) -> TushareArtifact:
    raw_path = tmp_path / "raw" / "source.parquet"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(b"raw-fixture")
    metadata_path = raw_path.with_suffix(".json")
    metadata_path.write_text("{}\n", encoding="utf-8")
    return TushareArtifact(
        api_name="fixture",
        request_sha256="a" * 64,
        parquet_path=raw_path,
        metadata_path=metadata_path,
        sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        row_count=1,
        params={},
        fields=("fixture",),
        ingested_at="2026-07-11T00:00:00Z",
    )


def test_snapshot_identity_and_artifact_hashes_are_stable(tmp_path: Path) -> None:
    config = load_research_data_config(ROOT / "config")
    raw_inputs = [_raw_input(tmp_path)]

    first = materialize_research_snapshot(tmp_path, config, _tables(), raw_inputs)
    second = materialize_research_snapshot(tmp_path, config, _tables(), raw_inputs)

    assert first.snapshot_id.startswith("p5-")
    assert first.snapshot_id == second.snapshot_id
    assert first.manifest_sha256 == second.manifest_sha256
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["summary"]["delisted_security_count"] == 1
    assert manifest["quality_status"] == "warning"
    assert all(not item["path"].startswith("/") for item in manifest["artifacts"])
    snapshot_dir = tmp_path / "research" / first.snapshot_id
    assert (snapshot_dir / "security_master.parquet").is_file()
    assert (snapshot_dir / "daily_bar" / "year=2021" / "part.parquet").is_file()
    assert (snapshot_dir / "universe_dates.parquet").is_file()


def test_snapshot_refuses_quality_error(tmp_path: Path) -> None:
    tables = _tables()
    tables.adjustment_factor.loc[0, "adj_factor"] = 0.0

    with pytest.raises(ValueError, match="quality"):
        materialize_research_snapshot(
            tmp_path,
            load_research_data_config(ROOT / "config"),
            tables,
            [_raw_input(tmp_path)],
        )


def test_tampered_published_snapshot_is_not_silently_reused(tmp_path: Path) -> None:
    config = load_research_data_config(ROOT / "config")
    raw_inputs = [_raw_input(tmp_path)]
    first = materialize_research_snapshot(tmp_path, config, _tables(), raw_inputs)
    artifact = first.snapshot_dir / "security_master.parquet"
    artifact.write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="immutable snapshot"):
        materialize_research_snapshot(tmp_path, config, _tables(), raw_inputs)
