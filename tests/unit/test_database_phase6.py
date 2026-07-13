from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from alpha_lab.database import catalog
from alpha_lab.research_data.provider import TushareArtifact
from alpha_lab.robustness import exposure_snapshot
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


def test_test_request_rejects_a_different_locked_range(tmp_path: Path) -> None:
    database_path = catalog.initialize_database(
        tmp_path / "metadata.duckdb"
    ).database_path
    with duckdb.connect(str(database_path)) as connection:
        _seed_factor_freezes(connection)

        with pytest.raises(duckdb.ConstraintException):
            _insert_test_request(
                connection,
                request_id="request-wrong-range",
                freeze_id="freeze-one",
                freeze_sha256="1" * 64,
                test_start="2026-01-02",
                test_end="2026-07-11",
            )


def test_test_approval_rejects_a_wrong_confirmed_freeze_hash(
    tmp_path: Path,
) -> None:
    database_path = catalog.initialize_database(
        tmp_path / "metadata.duckdb"
    ).database_path
    with duckdb.connect(str(database_path)) as connection:
        _seed_factor_freezes(connection)
        _insert_test_request(connection)

        with pytest.raises(duckdb.ConstraintException):
            _insert_test_approval(connection, confirmed_freeze_sha256="2" * 64)


def test_final_test_run_rejects_a_different_freeze(tmp_path: Path) -> None:
    database_path = catalog.initialize_database(
        tmp_path / "metadata.duckdb"
    ).database_path
    with duckdb.connect(str(database_path)) as connection:
        _seed_factor_freezes(connection)
        _insert_test_request(connection)
        _insert_test_approval(connection)

        with pytest.raises(duckdb.ConstraintException):
            _insert_final_test_run(
                connection,
                freeze_id="freeze-two",
                freeze_sha256="2" * 64,
            )


def test_final_test_run_rejects_a_different_locked_range(tmp_path: Path) -> None:
    database_path = catalog.initialize_database(
        tmp_path / "metadata.duckdb"
    ).database_path
    with duckdb.connect(str(database_path)) as connection:
        _seed_factor_freezes(connection)
        _insert_test_request(connection)
        _insert_test_approval(connection)

        with pytest.raises(duckdb.ConstraintException):
            _insert_final_test_run(connection, test_end="2026-07-10")


def test_robustness_state_chain_accepts_the_exact_freeze_identity(
    tmp_path: Path,
) -> None:
    database_path = catalog.initialize_database(
        tmp_path / "metadata.duckdb"
    ).database_path
    with duckdb.connect(str(database_path)) as connection:
        _seed_factor_freezes(connection)
        _insert_test_request(connection)
        _insert_test_approval(connection)
        _insert_final_test_run(connection)

        counts = connection.execute(
            """
            SELECT
                (SELECT count(*) FROM research.test_request),
                (SELECT count(*) FROM research.test_approval),
                (SELECT count(*) FROM research.final_test_run)
            """
        ).fetchone()

    assert counts == (1, 1, 1)


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


@pytest.mark.parametrize(
    "target",
    ["market_cap", "raw", "quality", "industry", "phase5"],
)
def test_exposure_sync_rejects_post_validation_file_tamper_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    data_dir = tmp_path / "data"
    exposure = _materialize_exposure_fixture(data_dir)
    database_path = data_dir / "metadata.duckdb"
    real_validate = exposure_snapshot.validate_exposure_snapshot

    def validate_then_tamper(root: Path, snapshot_id: str) -> dict[str, object]:
        result = real_validate(root, snapshot_id)
        path = _post_validation_target(data_dir, exposure.manifest_path, target)
        if target in {"market_cap", "industry", "phase5"}:
            frame = pd.read_parquet(path)
            frame.to_parquet(path, index=False, compression="gzip")
        else:
            path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(
        exposure_snapshot,
        "validate_exposure_snapshot",
        validate_then_tamper,
    )

    with pytest.raises(ValueError, match="checksum"):
        catalog.sync_exposure_snapshot(database_path, data_dir, exposure.manifest_path)

    assert _transactional_catalog_counts(database_path) == (0, 0, 0, 0)


def test_exposure_sync_rejects_post_validation_manifest_path_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    exposure = _materialize_exposure_fixture(data_dir)
    database_path = data_dir / "metadata.duckdb"
    real_validate = exposure_snapshot.validate_exposure_snapshot

    def validate_then_escape(root: Path, snapshot_id: str) -> dict[str, object]:
        result = real_validate(root, snapshot_id)
        manifest = json.loads(exposure.manifest_path.read_text(encoding="utf-8"))
        manifest["raw_inputs"][0]["path"] = "../../escaped.parquet"
        exposure.manifest_path.write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(
        exposure_snapshot,
        "validate_exposure_snapshot",
        validate_then_escape,
    )

    with pytest.raises(ValueError, match="manifest changed|escapes data_dir"):
        catalog.sync_exposure_snapshot(database_path, data_dir, exposure.manifest_path)

    assert _transactional_catalog_counts(database_path) == (0, 0, 0, 0)


def test_catalog_artifact_resolution_rejects_path_escape(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    escaped = tmp_path / "escaped.parquet"
    escaped.write_bytes(b"fixture")

    with pytest.raises(ValueError, match="escapes data_dir"):
        catalog._resolve_catalog_artifact(data_dir, "../escaped.parquet")


def test_exposure_sync_rolls_back_a_forced_mid_transaction_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    exposure = _materialize_exposure_fixture(data_dir)
    database_path = data_dir / "metadata.duckdb"

    def fail_after_parent_sync(*args: object, **kwargs: object) -> None:
        raise RuntimeError("forced exposure catalog failure")

    monkeypatch.setattr(catalog, "_sync_exposure_manifest", fail_after_parent_sync)

    with pytest.raises(RuntimeError, match="forced exposure catalog failure"):
        catalog.sync_exposure_snapshot(database_path, data_dir, exposure.manifest_path)

    assert _transactional_catalog_counts(database_path) == (0, 0, 0, 0)


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


def _seed_factor_freezes(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        INSERT INTO research.factor_definition
            (factor_id, name, family, description)
        VALUES ('F1002', 'phase6-fixture-factor', 'liquidity', 'fixture')
        """
    )
    connection.execute(
        """
        INSERT INTO research.factor_version
            (factor_version_id, factor_id, formula, implementation_path,
             code_sha256, metadata_sha256, lookback, direction)
        VALUES ('F1002-fixture', 'F1002', 'fixture', 'fixture.py', ?, ?, 20, 1)
        """,
        ["a" * 64, "b" * 64],
    )
    for freeze_id, freeze_sha256 in (
        ("freeze-one", "1" * 64),
        ("freeze-two", "2" * 64),
    ):
        connection.execute(
            """
            INSERT INTO research.factor_freeze
                (freeze_id, freeze_sha256, factor_version_id,
                 phase5_snapshot_id, exposure_snapshot_id,
                 robustness_policy_sha256, cost_policy_sha256, code_commit,
                 test_start, test_end, manifest_artifact_id)
            VALUES (?, ?, 'F1002-fixture', 'p5-fixture', 'p6x-fixture',
                    ?, ?, 'c0ffee', DATE '2026-01-01', DATE '2026-07-11', ?)
            """,
            [freeze_id, freeze_sha256, "c" * 64, "d" * 64, "e" * 64],
        )


def _insert_test_request(
    connection: duckdb.DuckDBPyConnection,
    *,
    request_id: str = "request-one",
    freeze_id: str = "freeze-one",
    freeze_sha256: str = "1" * 64,
    test_start: str = "2026-01-01",
    test_end: str = "2026-07-11",
) -> None:
    connection.execute(
        """
        INSERT INTO research.test_request
            (request_id, request_sha256, freeze_id, freeze_sha256,
             robustness_report_sha256, test_start, test_end)
        VALUES (?, ?, ?, ?, ?, CAST(? AS DATE), CAST(? AS DATE))
        """,
        [
            request_id,
            "3" * 64,
            freeze_id,
            freeze_sha256,
            "4" * 64,
            test_start,
            test_end,
        ],
    )


def _insert_test_approval(
    connection: duckdb.DuckDBPyConnection,
    *,
    approval_id: str = "approval-one",
    request_id: str = "request-one",
    freeze_id: str = "freeze-one",
    confirmed_freeze_sha256: str = "1" * 64,
    test_start: str = "2026-01-01",
    test_end: str = "2026-07-11",
) -> None:
    connection.execute(
        """
        INSERT INTO research.test_approval
            (approval_id, approval_sha256, request_id, freeze_id,
             confirmed_freeze_sha256, test_start, test_end, approver)
        VALUES (?, ?, ?, ?, ?, CAST(? AS DATE), CAST(? AS DATE), 'reviewer')
        """,
        [
            approval_id,
            "5" * 64,
            request_id,
            freeze_id,
            confirmed_freeze_sha256,
            test_start,
            test_end,
        ],
    )


def _insert_final_test_run(
    connection: duckdb.DuckDBPyConnection,
    *,
    approval_id: str = "approval-one",
    request_id: str = "request-one",
    freeze_id: str = "freeze-one",
    freeze_sha256: str = "1" * 64,
    test_start: str = "2026-01-01",
    test_end: str = "2026-07-11",
) -> None:
    connection.execute(
        """
        INSERT INTO research.final_test_run
            (test_run_id, run_sha256, approval_id, request_id, freeze_id,
             freeze_sha256, result_artifact_id, report_artifact_id, status,
             test_start, test_end, started_at, finished_at)
        VALUES ('final-one', ?, ?, ?, ?, ?, ?, ?, 'success',
                CAST(? AS DATE), CAST(? AS DATE), current_timestamp,
                current_timestamp)
        """,
        [
            "6" * 64,
            approval_id,
            request_id,
            freeze_id,
            freeze_sha256,
            "7" * 64,
            "8" * 64,
            test_start,
            test_end,
        ],
    )


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


def _transactional_catalog_counts(database_path: Path) -> tuple[int, ...]:
    with duckdb.connect(str(database_path), read_only=True) as connection:
        return connection.execute(
            """
            SELECT
                (SELECT count(*) FROM meta.dataset_snapshot),
                (SELECT count(*) FROM ref.security),
                (SELECT count(*) FROM meta.artifact),
                (SELECT count(*) FROM meta.repository_state
                 WHERE key = 'latest_exposure_snapshot_id')
            """
        ).fetchone()


def _post_validation_target(
    data_dir: Path,
    manifest_path: Path,
    target: str,
) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if target == "market_cap":
        item = next(
            artifact
            for artifact in manifest["artifacts"]
            if artifact["name"].startswith("market_cap/")
        )
    elif target == "raw":
        item = manifest["raw_inputs"][0]
    elif target == "quality":
        item = manifest["quality_report"]
    elif target == "industry":
        item = next(
            artifact
            for artifact in manifest["artifacts"]
            if artifact["name"] == "industry_membership.parquet"
        )
    elif target == "phase5":
        phase5_path = (
            data_dir / "manifests" / manifest["phase5_snapshot_id"] / "manifest.json"
        )
        phase5 = json.loads(phase5_path.read_text(encoding="utf-8"))
        item = next(
            artifact
            for artifact in phase5["artifacts"]
            if artifact["name"] == "security_master.parquet"
        )
    else:
        raise AssertionError(f"unknown tamper target: {target}")
    return data_dir / item["path"]


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
