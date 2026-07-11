from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

from alpha_lab.database.catalog import (
    initialize_database,
    sync_research_snapshot,
)


def _artifact(data_root: Path, relative: str, frame: pd.DataFrame) -> dict[str, object]:
    path = data_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return {
        "name": Path(relative).name,
        "path": relative,
        "format": "parquet",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "row_count": len(frame),
    }


def test_database_applies_two_immutable_migrations(tmp_path: Path) -> None:
    result = initialize_database(tmp_path / "metadata.duckdb")
    second = initialize_database(tmp_path / "metadata.duckdb")

    assert result.schema_version == 2
    assert result.migration_sha256 == second.migration_sha256
    with duckdb.connect(str(result.database_path), read_only=True) as connection:
        migrations = connection.execute(
            "SELECT version, name FROM meta.schema_migration ORDER BY version"
        ).fetchall()
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

    assert migrations == [(1, "initial_catalog"), (2, "research_data")]
    assert "meta.provider_capability" in tables
    assert "ref.security_name_history" in tables


def test_research_snapshot_sync_is_idempotent(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    snapshot_id = "p5-catalog-test"
    prefix = f"research/{snapshot_id}"
    security = pd.DataFrame(
        [
            {
                "security_id": "CN:SSE:600001",
                "ts_code": "600001.SH",
                "symbol": "600001",
                "name": "*ST示例",
                "exchange": "SSE",
                "board": "主板",
                "currency": "CNY",
                "list_status": "D",
                "list_date": pd.Timestamp("2010-01-01"),
                "delist_date": pd.Timestamp("2022-12-31"),
                "known_at": pd.Timestamp("2020-01-01", tz="UTC"),
            }
        ]
    )
    names = pd.DataFrame(
        [
            {
                "security_id": "CN:SSE:600001",
                "name": "*ST示例",
                "is_st": True,
                "effective_from": pd.Timestamp("2021-01-01"),
                "effective_to": pd.Timestamp("2021-12-31"),
                "announced_at": pd.Timestamp("2020-12-31", tz="UTC"),
                "known_at": pd.Timestamp("2020-12-31", tz="UTC"),
            }
        ]
    )
    membership = pd.DataFrame(
        [
            {
                "index_id": "CN:INDEX:000300.SH",
                "security_id": "CN:SSE:600001",
                "effective_from": pd.Timestamp("2020-01-01"),
                "effective_to": pd.Timestamp("2021-12-31"),
                "announced_at": pd.Timestamp("2019-12-31", tz="UTC"),
                "known_at": pd.Timestamp("2019-12-31", tz="UTC"),
                "weight": 0.5,
            }
        ]
    )
    artifacts = [
        _artifact(data_root, f"{prefix}/security_master.parquet", security),
        _artifact(data_root, f"{prefix}/security_name_history.parquet", names),
        _artifact(data_root, f"{prefix}/index_membership.parquet", membership),
    ]
    manifest = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "snapshot_type": "research_market",
        "identity_sha256": "a" * 64,
        "quality_status": "pass",
        "source": {"provider": "tushare", "credential_redacted": True},
        "scope": {
            "index_code": "000300.SH",
            "start_date": "2020-01-01",
            "end_date": "2022-12-31",
        },
        "summary": {
            "security_count": 1,
            "delisted_security_count": 1,
            "membership_interval_count": 1,
            "daily_bar_count": 0,
            "adjustment_factor_count": 0,
            "daily_status_count": 0,
        },
        "raw_inputs": [],
        "artifacts": artifacts,
    }
    manifest_path = data_root / "manifests" / snapshot_id / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest, default=str), encoding="utf-8")
    database_path = data_root / "metadata.duckdb"

    sync_research_snapshot(database_path, data_root, manifest_path)
    sync_research_snapshot(database_path, data_root, manifest_path)

    with duckdb.connect(str(database_path), read_only=True) as connection:
        counts = connection.execute(
            """
            SELECT
                (SELECT count(*) FROM meta.dataset_snapshot),
                (SELECT count(*) FROM ref.security),
                (SELECT count(*) FROM ref.security_lifecycle),
                (SELECT count(*) FROM ref.security_name_history),
                (SELECT count(*) FROM ref.index_definition),
                (SELECT count(*) FROM ref.index_membership_history),
                (SELECT count(*) FROM meta.artifact)
            """
        ).fetchone()

    assert counts == (1, 1, 1, 1, 1, 1, 3)
