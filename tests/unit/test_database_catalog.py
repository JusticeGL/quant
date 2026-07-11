from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from alpha_lab.database.catalog import (
    EXPECTED_TABLES,
    check_database,
    initialize_database,
    sync_repository_metadata,
)

ROOT = Path(__file__).resolve().parents[2]


def test_database_initialization_is_versioned_idempotent_and_constrained(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "data" / "metadata.duckdb"

    first = initialize_database(database_path)
    second = initialize_database(database_path)

    assert first.schema_version == 1
    assert second.schema_version == 1
    assert first.migration_sha256 == second.migration_sha256

    with duckdb.connect(str(database_path)) as connection:
        tables = {
            f"{schema}.{table}"
            for schema, table in connection.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                """
            ).fetchall()
        }
        migration_count = connection.execute(
            "SELECT count(*) FROM meta.schema_migration"
        ).fetchone()[0]
        contract_count = connection.execute(
            "SELECT count(*) FROM meta.dataset_contract"
        ).fetchone()[0]

        assert tables >= EXPECTED_TABLES
        assert migration_count == 1
        assert contract_count >= 8

        connection.execute(
            """
            INSERT INTO ref.security
                (security_id, asset_type, exchange, currency, lot_size)
            VALUES ('CN:SH:600519', 'stock', 'SSE', 'CNY', 100)
            """
        )
        with pytest.raises(duckdb.ConstraintException):
            connection.execute(
                """
                INSERT INTO ref.security
                    (security_id, asset_type, exchange, currency, lot_size)
                VALUES ('CN:SH:600519', 'stock', 'SSE', 'CNY', 100)
                """
            )


def test_daily_bar_macro_reads_typed_external_parquet(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.duckdb"
    parquet_path = tmp_path / "daily.parquet"
    pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2024-01-02"),
                "instrument": "SH600519",
                "open": 1700.0,
                "high": 1720.0,
                "low": 1695.0,
                "close": 1710.0,
                "volume": 2_500_000.0,
                "amount": 4_275_000_000.0,
                "adj_factor": None,
                "suspend": None,
                "limit_up": None,
                "limit_down": None,
                "is_st": None,
                "list_date": None,
                "delist_date": None,
                "source": "akshare.stock_zh_a_hist",
                "ingested_at": pd.Timestamp("2026-07-10", tz="UTC"),
            }
        ]
    ).to_parquet(parquet_path, index=False)
    initialize_database(database_path)

    escaped_path = str(parquet_path).replace("'", "''")
    with duckdb.connect(str(database_path), read_only=True) as connection:
        row = connection.execute(
            f"SELECT * FROM market.read_daily_bar('{escaped_path}')"
        ).fetchone()
        columns = [item[0] for item in connection.description]

    assert columns[:4] == ["trade_date", "instrument", "open", "high"]
    assert str(row[0]) == "2024-01-02"
    assert row[1] == "SH600519"
    assert row[6] == 2_500_000.0


def test_manifest_and_universe_sync_populates_catalog(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    snapshot_id = "p1-test-snapshot"
    silver_path = data_root / "silver" / snapshot_id / "daily.parquet"
    silver_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [{"trade_date": pd.Timestamp("2024-01-02"), "instrument": "SH600519"}]
    ).to_parquet(silver_path, index=False)
    silver_sha = hashlib.sha256(silver_path.read_bytes()).hexdigest()

    report_path = data_root / "manifests" / snapshot_id / "quality_report.json"
    report_path.parent.mkdir(parents=True)
    quality_report = {
        "status": "warning",
        "duplicates": {"count": 0, "keys": []},
        "invalid_rows": {"count": 0, "keys": []},
        "missing_instruments": [],
        "missing_status_fields": ["adj_factor"],
    }
    report_path.write_text(json.dumps(quality_report), encoding="utf-8")
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "identity_sha256": "a" * 64,
        "source": {"provider": "akshare", "endpoint": "stock_zh_a_hist"},
        "universe": {"sample_id": "liquid_a_share_engineering_sample_10"},
        "summary": {
            "row_count": 1,
            "instrument_count": 1,
            "date_start": "2024-01-02",
            "date_end": "2024-01-02",
            "quality_status": "warning",
        },
        "raw_inputs": [],
        "artifacts": {
            "silver": {
                "path": f"silver/{snapshot_id}/daily.parquet",
                "sha256": silver_sha,
            },
            "quality_report": {
                "path": f"manifests/{snapshot_id}/quality_report.json",
                "sha256": report_sha,
            },
        },
    }
    manifest_path = report_path.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    state_path = data_root / "state" / "latest_snapshot.txt"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(f"{snapshot_id}\n", encoding="utf-8")

    database_path = data_root / "metadata.duckdb"
    initialize_database(database_path)
    first = sync_repository_metadata(database_path, ROOT / "config", data_root)
    second = sync_repository_metadata(database_path, ROOT / "config", data_root)
    report = check_database(database_path, data_root)

    assert first.snapshots_synced == 1
    assert second.snapshots_synced == 1
    assert report["schema_version"] == 1
    assert report["security_count"] == 10
    assert report["snapshot_count"] == 1
    assert report["artifact_count"] == 2
    assert report["latest_snapshot_id"] == snapshot_id
    assert report["missing_artifact_files"] == []
