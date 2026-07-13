from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from alpha_lab.database import catalog
from alpha_lab.research_data.provider import TushareArtifact
from alpha_lab.robustness.config import load_robustness_config
from alpha_lab.robustness.contracts import ExposureTables
from alpha_lab.robustness.exposure_data import (
    normalize_industry_definition,
    normalize_industry_membership,
    normalize_market_cap,
)
from alpha_lab.robustness.exposure_snapshot import materialize_exposure_snapshot

ROOT = Path(__file__).resolve().parents[2]
ROBUSTNESS_TABLES = {
    "ref.industry_definition",
    "ref.industry_membership_history",
    "research.factor_freeze",
    "research.test_request",
    "research.test_approval",
    "research.final_test_run",
}


def test_database_applies_robustness_migration(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.duckdb"

    first = catalog.initialize_database(database_path)
    second = catalog.initialize_database(database_path)

    assert first.schema_version == 3
    assert second.schema_version == 3
    assert first.migration_sha256 == second.migration_sha256
    with duckdb.connect(str(database_path), read_only=True) as connection:
        migrations = connection.execute(
            "SELECT version, name, sha256 FROM meta.schema_migration ORDER BY version"
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

    migration_sql = (
        ROOT / "src" / "alpha_lab" / "database" / "sql" / "003_robustness.sql"
    ).read_bytes()
    assert [(version, name) for version, name, _ in migrations] == [
        (1, "initial_catalog"),
        (2, "research_data"),
        (3, "robustness_catalog"),
    ]
    assert migrations[-1][2] == hashlib.sha256(migration_sql).hexdigest()
    assert tables >= ROBUSTNESS_TABLES


def test_robustness_tables_enforce_sha256_and_foreign_keys(tmp_path: Path) -> None:
    database_path = catalog.initialize_database(
        tmp_path / "metadata.duckdb"
    ).database_path

    with duckdb.connect(str(database_path)) as connection:
        with pytest.raises(duckdb.ConstraintException):
            connection.execute(
                """
                INSERT INTO ref.industry_definition
                    (definition_id, exposure_snapshot_id, industry_id,
                     source_index_code, industry_name, level,
                     classification_standard, source, source_artifact_id)
                VALUES ('not-a-sha256', 'p6x-fixture', 'CN:SW2021:801010.SI',
                        '801010.SI', '农林牧渔', 'L1', 'SW2021',
                        'tushare.index_classify', ?)
                """,
                ["a" * 64],
            )
        with pytest.raises(duckdb.ConstraintException):
            connection.execute(
                """
                INSERT INTO ref.industry_membership_history
                    (membership_id, exposure_snapshot_id, definition_id,
                     security_id, effective_from, known_at,
                     known_at_source, source, source_artifact_id)
                VALUES (?, 'p6x-fixture', ?, 'CN:SSE:600000', DATE '2021-01-01',
                        TIMESTAMPTZ '2021-01-01 00:00:00+00',
                        'effective_date_fallback', 'tushare.index_member_all', ?)
                """,
                ["b" * 64, "c" * 64, "d" * 64],
            )


def test_exposure_snapshot_sync_is_bulk_and_idempotent(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    exposure = _materialize_exposure_fixture(data_dir)
    database_path = data_dir / "metadata.duckdb"

    catalog.sync_exposure_snapshot(database_path, data_dir, exposure.manifest_path)
    first = _exposure_counts(database_path, exposure.snapshot_id)
    catalog.sync_exposure_snapshot(database_path, data_dir, exposure.manifest_path)
    second = _exposure_counts(database_path, exposure.snapshot_id)

    assert first == second == (1, 1, 1, 2, 1)
    with duckdb.connect(str(database_path), read_only=True) as connection:
        market_table_count = connection.execute(
            """
            SELECT count(*)
            FROM information_schema.tables
            WHERE table_schema = 'market'
              AND table_name = 'exposure_market_cap'
            """
        ).fetchone()[0]
        membership = connection.execute(
            """
            SELECT length(membership_id), known_at_source, effective_from,
                   effective_to
            FROM ref.industry_membership_history
            """
        ).fetchone()

    assert market_table_count == 0
    assert membership == (
        64,
        "effective_date_fallback",
        pd.Timestamp("2021-01-01").date(),
        None,
    )


def test_exposure_snapshot_sync_rejects_tampered_artifact(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    exposure = _materialize_exposure_fixture(data_dir)
    manifest = json.loads(exposure.manifest_path.read_text(encoding="utf-8"))
    definition = next(
        item
        for item in manifest["artifacts"]
        if item["name"] == "industry_definition.parquet"
    )
    (data_dir / definition["path"]).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="invalid exposure snapshot"):
        catalog.sync_exposure_snapshot(
            data_dir / "metadata.duckdb", data_dir, exposure.manifest_path
        )


def test_exposure_snapshot_sync_rejects_detached_manifest_reference(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    exposure = _materialize_exposure_fixture(data_dir)
    detached = tmp_path / "detached-manifest.json"
    detached.write_bytes(exposure.manifest_path.read_bytes())

    with pytest.raises(ValueError, match="canonical manifest path"):
        catalog.sync_exposure_snapshot(data_dir / "metadata.duckdb", data_dir, detached)


def test_applied_robustness_migration_checksum_is_immutable(tmp_path: Path) -> None:
    database_path = catalog.initialize_database(
        tmp_path / "metadata.duckdb"
    ).database_path
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            "UPDATE meta.schema_migration SET sha256 = ? WHERE version = 3",
            ["0" * 64],
        )

    with pytest.raises(RuntimeError, match="migration hash differs"):
        catalog.initialize_database(database_path)


def _exposure_counts(database_path: Path, snapshot_id: str) -> tuple[int, ...]:
    with duckdb.connect(str(database_path), read_only=True) as connection:
        return connection.execute(
            """
            SELECT
                (SELECT count(*) FROM ref.industry_definition
                 WHERE exposure_snapshot_id = ?),
                (SELECT count(*) FROM ref.industry_membership_history
                 WHERE exposure_snapshot_id = ?),
                (SELECT count(*) FROM meta.dataset_snapshot
                 WHERE snapshot_id = ? AND snapshot_type = 'point_in_time_exposure'),
                (SELECT count(*) FROM meta.artifact
                 WHERE dataset_name = 'market.exposure_market_cap'),
                (SELECT count(*) FROM meta.quality_result
                 WHERE snapshot_id = ? AND dataset_name = 'research.exposure_snapshot')
            """,
            [snapshot_id, snapshot_id, snapshot_id, snapshot_id],
        ).fetchone()


def _materialize_exposure_fixture(data_dir: Path):
    phase5_manifest = _write_phase5_fixture(data_dir)
    config, policy_sha256 = load_robustness_config(ROOT / "config" / "robustness.yaml")
    return materialize_exposure_snapshot(
        data_dir,
        config,
        policy_sha256,
        phase5_manifest,
        _exposure_tables(),
        [_raw_input(data_dir)],
    )


def _exposure_tables() -> ExposureTables:
    definitions = normalize_industry_definition(
        pd.DataFrame(
            [["801010.SI", "农林牧渔", "L1", "110000", "SW2021"]],
            columns=["index_code", "industry_name", "level", "industry_code", "src"],
        )
    )
    membership = normalize_industry_membership(
        pd.DataFrame(
            [
                [
                    "801010.SI",
                    "农林牧渔",
                    "",
                    "",
                    "",
                    "",
                    "600000.SH",
                    "浦发银行",
                    "20210101",
                    "",
                    "Y",
                ]
            ],
            columns=[
                "l1_code",
                "l1_name",
                "l2_code",
                "l2_name",
                "l3_code",
                "l3_name",
                "ts_code",
                "name",
                "in_date",
                "out_date",
                "is_new",
            ],
        ),
        {"801010.SI"},
    )
    market = normalize_market_cap(
        pd.DataFrame(
            [
                ["600000.SH", "20210104", 123.4, 100.0],
                ["600000.SH", "20260710", 130.0, 110.0],
            ],
            columns=["ts_code", "trade_date", "total_mv", "circ_mv"],
        )
    )
    return ExposureTables(market, definitions, membership)


def _write_phase5_fixture(data_dir: Path) -> Path:
    snapshot_id = "p5-ecaa6e8aeae6b9f8fb25"
    research = data_dir / "research" / snapshot_id
    security = pd.DataFrame(
        [
            {
                "security_id": "CN:SSE:600000",
                "ts_code": "600000.SH",
                "symbol": "600000",
                "name": "浦发银行",
                "exchange": "SSE",
                "board": "主板",
                "currency": "CNY",
                "list_status": "L",
                "list_date": pd.Timestamp("1999-11-10"),
                "delist_date": pd.NaT,
                "known_at": pd.Timestamp("2020-01-01", tz="UTC"),
            }
        ]
    )
    names = pd.DataFrame(
        [
            {
                "security_id": "CN:SSE:600000",
                "name": "浦发银行",
                "is_st": False,
                "effective_from": pd.Timestamp("1999-11-10"),
                "effective_to": pd.NaT,
                "announced_at": pd.Timestamp("1999-11-10", tz="UTC"),
                "known_at": pd.Timestamp("1999-11-10", tz="UTC"),
            }
        ]
    )
    membership = pd.DataFrame(
        [
            {
                "index_id": "CN:INDEX:000300.SH",
                "security_id": "CN:SSE:600000",
                "effective_from": pd.Timestamp("2020-01-01"),
                "effective_to": pd.NaT,
                "announced_at": pd.Timestamp("2019-12-31", tz="UTC"),
                "known_at": pd.Timestamp("2019-12-31", tz="UTC"),
                "weight": 0.5,
                "membership_method": "fixture",
            }
        ]
    )
    universe = pd.DataFrame(
        [
            {"as_of_date": pd.Timestamp("2021-01-04"), "security_id": "CN:SSE:600000"},
            {"as_of_date": pd.Timestamp("2026-07-10"), "security_id": "CN:SSE:600000"},
        ]
    )
    daily = universe.rename(columns={"as_of_date": "trade_date"})
    frames = {
        "security_master.parquet": security,
        "security_name_history.parquet": names,
        "index_membership.parquet": membership,
        "universe_dates.parquet": universe,
        "daily_bar/year=2021/part.parquet": daily.iloc[:1],
        "daily_bar/year=2026/part.parquet": daily.iloc[1:],
    }
    artifacts: list[dict[str, object]] = []
    for name, frame in frames.items():
        path = research / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        artifacts.append(
            {
                "name": name,
                "path": path.relative_to(data_dir).as_posix(),
                "format": "parquet",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "row_count": len(frame),
            }
        )
    manifest = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "snapshot_type": "research_market",
        "identity_sha256": "9" * 64,
        "quality_status": "pass",
        "source": {"provider": "tushare", "credential_redacted": True},
        "scope": {
            "index_code": "000300.SH",
            "start_date": "2020-01-01",
            "end_date": "2026-07-11",
        },
        "summary": {
            "security_count": 1,
            "daily_bar_count": 2,
        },
        "raw_inputs": [],
        "artifacts": artifacts,
    }
    path = data_dir / "manifests" / snapshot_id / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _raw_input(data_dir: Path) -> TushareArtifact:
    path = data_dir / "raw" / "tushare" / "fixture" / "request.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    metadata = path.with_suffix(".json")
    metadata.write_text("{}\n", encoding="utf-8")
    return TushareArtifact(
        api_name="fixture",
        request_sha256="a" * 64,
        parquet_path=path,
        metadata_path=metadata,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        row_count=1,
        params={},
        fields=("fixture",),
        ingested_at="2026-07-13T00:00:00Z",
    )
