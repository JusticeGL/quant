from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from alpha_lab.data.config import load_phase1_config
from alpha_lab.data.normalize import to_qlib_instrument

SCHEMA_VERSION = 2
MIGRATIONS = (
    (1, "initial_catalog", "001_initial.sql"),
    (2, "research_data", "002_research_data.sql"),
)

EXPECTED_TABLES = {
    "meta.schema_migration",
    "meta.data_source",
    "meta.ingestion_run",
    "meta.artifact",
    "meta.dataset_snapshot",
    "meta.snapshot_artifact",
    "meta.quality_result",
    "meta.dataset_contract",
    "meta.repository_state",
    "meta.provider_capability",
    "ref.exchange",
    "ref.security",
    "ref.security_identifier_history",
    "ref.security_lifecycle",
    "ref.security_name_history",
    "ref.trading_calendar",
    "ref.industry_classification",
    "ref.industry_node",
    "ref.security_industry_history",
    "ref.index_definition",
    "ref.index_membership_history",
    "market.corporate_action",
    "fundamental.metric_definition",
    "fundamental.filing_catalog",
    "fundamental.fact_artifact",
    "policy.policy_version",
    "policy.price_limit_rule",
    "policy.cost_rule",
    "research.universe_definition",
    "research.universe_membership",
    "research.factor_definition",
    "research.factor_version",
    "research.experiment_run",
    "research.experiment_metric",
    "research.experiment_decision",
    "research.backtest_run",
}

DATASET_CONTRACTS: tuple[dict[str, Any], ...] = (
    {
        "dataset_name": "ref.security_master",
        "storage_layer": "research",
        "primary_key_columns": ["security_id"],
        "partition_columns": [],
        "required_columns": [
            "security_id",
            "ts_code",
            "exchange",
            "list_status",
            "list_date",
            "delist_date",
            "known_at",
        ],
        "point_in_time_column": "known_at",
        "description": "Point-in-time A-share security master with delisted rows.",
    },
    {
        "dataset_name": "ref.security_name_history",
        "storage_layer": "research",
        "primary_key_columns": ["security_id", "effective_from"],
        "partition_columns": [],
        "required_columns": [
            "security_id",
            "name",
            "is_st",
            "effective_from",
            "effective_to",
            "known_at",
        ],
        "point_in_time_column": "known_at",
        "description": "Historical security names and ST intervals.",
    },
    {
        "dataset_name": "ref.index_membership",
        "storage_layer": "research",
        "primary_key_columns": ["index_id", "security_id", "effective_from"],
        "partition_columns": [],
        "required_columns": [
            "index_id",
            "security_id",
            "effective_from",
            "effective_to",
            "known_at",
            "membership_method",
        ],
        "point_in_time_column": "known_at",
        "description": "Point-in-time dynamic index membership intervals.",
    },
    {
        "dataset_name": "research.universe_dates",
        "storage_layer": "research",
        "primary_key_columns": ["as_of_date", "index_id", "security_id"],
        "partition_columns": [],
        "required_columns": ["as_of_date", "index_id", "security_id"],
        "point_in_time_column": "as_of_date",
        "description": "Materialized historically knowable universe membership.",
    },
    {
        "dataset_name": "market.daily_bar",
        "storage_layer": "silver",
        "primary_key_columns": ["trade_date", "instrument"],
        "partition_columns": ["year", "exchange"],
        "required_columns": [
            "trade_date",
            "instrument",
            "open",
            "high",
            "low",
            "close",
            "volume_shares",
            "amount_cny",
            "source",
            "ingested_at",
        ],
        "point_in_time_column": "trade_date",
        "description": "Unadjusted canonical A-share daily bars.",
    },
    {
        "dataset_name": "market.adjustment_factor",
        "storage_layer": "silver",
        "primary_key_columns": ["trade_date", "security_id", "factor_type"],
        "partition_columns": ["year"],
        "required_columns": [
            "trade_date",
            "security_id",
            "adj_factor",
            "factor_type",
            "base_date",
            "known_at",
        ],
        "point_in_time_column": "known_at",
        "description": "Versioned adjustment factors; adjusted prices are derived.",
    },
    {
        "dataset_name": "market.daily_basic",
        "storage_layer": "silver",
        "primary_key_columns": ["trade_date", "security_id"],
        "partition_columns": ["year"],
        "required_columns": [
            "trade_date",
            "security_id",
            "turnover_rate",
            "total_market_value",
            "float_market_value",
            "known_at",
        ],
        "point_in_time_column": "known_at",
        "description": "Daily valuation, shares, turnover, and market capitalization.",
    },
    {
        "dataset_name": "market.daily_status",
        "storage_layer": "silver",
        "primary_key_columns": ["trade_date", "security_id"],
        "partition_columns": ["year"],
        "required_columns": [
            "trade_date",
            "security_id",
            "is_suspended",
            "is_st",
            "limit_up_price",
            "limit_down_price",
            "known_at",
        ],
        "point_in_time_column": "known_at",
        "description": "Nullable observed A-share status and tradability fields.",
    },
    {
        "dataset_name": "market.corporate_action",
        "storage_layer": "silver",
        "primary_key_columns": ["action_id"],
        "partition_columns": ["announcement_year"],
        "required_columns": [
            "action_id",
            "security_id",
            "action_type",
            "announcement_date",
            "known_at",
        ],
        "point_in_time_column": "known_at",
        "description": "Corporate actions with announcement and effective dates.",
    },
    {
        "dataset_name": "fundamental.financial_fact",
        "storage_layer": "fundamentals",
        "primary_key_columns": ["filing_id", "metric_code", "scope"],
        "partition_columns": ["report_year", "statement_type"],
        "required_columns": [
            "filing_id",
            "metric_code",
            "value",
            "unit",
            "scope",
            "known_at",
        ],
        "point_in_time_column": "known_at",
        "description": "Long-form point-in-time financial statement facts.",
    },
    {
        "dataset_name": "research.factor_value",
        "storage_layer": "research",
        "primary_key_columns": ["trade_date", "instrument", "factor_version_id"],
        "partition_columns": ["factor_version_id", "year"],
        "required_columns": [
            "trade_date",
            "instrument",
            "factor_version_id",
            "value",
            "data_snapshot_id",
        ],
        "point_in_time_column": "trade_date",
        "description": "Immutable factor values tied to one data snapshot.",
    },
    {
        "dataset_name": "research.backtest_daily",
        "storage_layer": "research",
        "primary_key_columns": ["backtest_id", "trade_date"],
        "partition_columns": ["backtest_id"],
        "required_columns": [
            "backtest_id",
            "trade_date",
            "nav",
            "return",
            "turnover",
            "cost",
        ],
        "point_in_time_column": "trade_date",
        "description": "Daily backtest summary; positions and trades remain artifacts.",
    },
)


@dataclass(frozen=True)
class InitializationResult:
    database_path: Path
    schema_version: int
    migration_sha256: str


@dataclass(frozen=True)
class SyncResult:
    securities_synced: int
    snapshots_synced: int
    artifacts_synced: int
    quality_results_synced: int


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _migration_sql(filename: str) -> str:
    resource = resources.files("alpha_lab.database.sql").joinpath(filename)
    return resource.read_text(encoding="utf-8")


def initialize_database(database_path: Path) -> InitializationResult:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    latest_sha256 = ""

    with duckdb.connect(str(database_path)) as connection:
        connection.execute("CREATE SCHEMA IF NOT EXISTS meta")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS meta.schema_migration (
                version INTEGER PRIMARY KEY,
                name VARCHAR NOT NULL,
                sha256 VARCHAR NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
            )
            """
        )
        for version, name, filename in MIGRATIONS:
            sql = _migration_sql(filename)
            migration_sha256 = hashlib.sha256(sql.encode("utf-8")).hexdigest()
            latest_sha256 = migration_sha256
            existing = connection.execute(
                "SELECT name, sha256 FROM meta.schema_migration WHERE version = ?",
                [version],
            ).fetchone()
            if existing is not None:
                if str(existing[0]) != name or str(existing[1]) != migration_sha256:
                    raise RuntimeError(
                        "applied database migration hash differs from packaged "
                        f"migration {version}"
                    )
                continue
            connection.execute("BEGIN TRANSACTION")
            try:
                connection.execute(sql)
                connection.execute(
                    """
                    INSERT INTO meta.schema_migration (version, name, sha256)
                    VALUES (?, ?, ?)
                    """,
                    [version, name, migration_sha256],
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

        connection.execute("BEGIN TRANSACTION")
        try:
            _seed_catalog(connection)
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    return InitializationResult(
        database_path=database_path,
        schema_version=SCHEMA_VERSION,
        migration_sha256=latest_sha256,
    )


def _seed_catalog(connection: duckdb.DuckDBPyConnection) -> None:
    sources = (
        (
            "akshare",
            "akshare",
            "stock_zh_a_hist",
            "akshare",
            importlib.metadata.version("akshare"),
            "Research use; upstream interfaces can change.",
            10,
        ),
        (
            "baostock",
            "baostock",
            "query_history_k_data_plus",
            "baostock",
            importlib.metadata.version("baostock"),
            "Public fallback data source; verify source terms before redistribution.",
            20,
        ),
        (
            "tushare",
            "tushare",
            "rest_api",
            "tushare",
            importlib.metadata.version("tushare"),
            "Optional authenticated research data source; redistribution terms apply.",
            5,
        ),
    )
    connection.executemany(
        """
        INSERT INTO meta.data_source
            (source_id, provider, endpoint, package_name, package_version,
             license_note, priority)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (source_id) DO NOTHING
        """,
        sources,
    )
    connection.executemany(
        """
        INSERT INTO ref.exchange (exchange_code, name)
        VALUES (?, ?)
        ON CONFLICT (exchange_code) DO NOTHING
        """,
        (
            ("SSE", "Shanghai Stock Exchange"),
            ("SZSE", "Shenzhen Stock Exchange"),
            ("BSE", "Beijing Stock Exchange"),
        ),
    )
    for contract in DATASET_CONTRACTS:
        connection.execute(
            """
            INSERT INTO meta.dataset_contract
                (dataset_name, storage_layer, storage_format,
                 primary_key_columns, partition_columns, required_columns,
                 point_in_time_column, description, schema_version)
            VALUES (?, ?, 'parquet', ?, ?, ?, ?, ?, ?)
            ON CONFLICT (dataset_name) DO UPDATE SET
                storage_layer = excluded.storage_layer,
                storage_format = excluded.storage_format,
                primary_key_columns = excluded.primary_key_columns,
                partition_columns = excluded.partition_columns,
                required_columns = excluded.required_columns,
                point_in_time_column = excluded.point_in_time_column,
                description = excluded.description,
                schema_version = excluded.schema_version,
                updated_at = excluded.updated_at
            """,
            [
                contract["dataset_name"],
                contract["storage_layer"],
                _canonical_json(contract["primary_key_columns"]),
                _canonical_json(contract["partition_columns"]),
                _canonical_json(contract["required_columns"]),
                contract["point_in_time_column"],
                contract["description"],
                SCHEMA_VERSION,
            ],
        )


def sync_repository_metadata(
    database_path: Path, config_dir: Path, data_root: Path
) -> SyncResult:
    initialize_database(database_path)
    config = load_phase1_config(config_dir)
    universe_document = config.universe.model_dump(mode="json")
    universe_sha256 = hashlib.sha256(
        _canonical_json(universe_document).encode("utf-8")
    ).hexdigest()

    securities_synced = 0
    snapshots_synced = 0
    artifact_ids: set[str] = set()
    quality_results_synced = 0

    with duckdb.connect(str(database_path)) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            for symbol in config.universe.symbols:
                _sync_security(
                    connection, symbol.code, symbol.name, config.universe.as_of
                )
                securities_synced += 1

            connection.execute(
                """
                INSERT INTO research.universe_definition
                    (universe_id, name, description, construction_rule,
                     survivorship_free, research_eligible, config_sha256)
                VALUES (?, ?, ?, ?, false, ?, ?)
                ON CONFLICT (universe_id) DO NOTHING
                """,
                [
                    config.universe.sample_id,
                    config.universe.sample_id,
                    config.universe.disclaimer,
                    _canonical_json(universe_document),
                    config.universe.research_eligible,
                    universe_sha256,
                ],
            )
            for symbol in config.universe.symbols:
                security_id, _, _ = _security_identity(symbol.code)
                membership_id = hashlib.sha256(
                    f"{config.universe.sample_id}|{security_id}|{config.universe.as_of}".encode()
                ).hexdigest()
                connection.execute(
                    """
                    INSERT INTO research.universe_membership
                        (membership_id, universe_id, security_id, effective_from)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (membership_id) DO NOTHING
                    """,
                    [
                        membership_id,
                        config.universe.sample_id,
                        security_id,
                        config.universe.as_of,
                    ],
                )

            for manifest_path in sorted(
                (data_root / "manifests").glob("*/manifest.json")
            ):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("snapshot_type") == "research_market":
                    snapshot_artifacts, quality_count = _sync_research_manifest(
                        connection, data_root, manifest
                    )
                else:
                    snapshot_artifacts, quality_count = _sync_manifest(
                        connection, data_root, manifest
                    )
                artifact_ids.update(snapshot_artifacts)
                quality_results_synced += quality_count
                snapshots_synced += 1

            latest_path = data_root / "state" / "latest_snapshot.txt"
            if latest_path.is_file():
                connection.execute(
                    """
                    INSERT INTO meta.repository_state (key, value)
                    VALUES ('latest_snapshot_id', ?)
                    ON CONFLICT (key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                    """,
                    [latest_path.read_text(encoding="utf-8").strip()],
                )
            latest_research = data_root / "state" / "latest_research_snapshot.txt"
            if latest_research.is_file():
                connection.execute(
                    """
                    INSERT INTO meta.repository_state (key, value)
                    VALUES ('latest_research_snapshot_id', ?)
                    ON CONFLICT (key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                    """,
                    [latest_research.read_text(encoding="utf-8").strip()],
                )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    return SyncResult(
        securities_synced=securities_synced,
        snapshots_synced=snapshots_synced,
        artifacts_synced=len(artifact_ids),
        quality_results_synced=quality_results_synced,
    )


def sync_research_snapshot(
    database_path: Path, data_root: Path, manifest_path: Path
) -> None:
    """Synchronize one immutable Phase 5 snapshot into the DuckDB catalog."""
    with _catalog_write_lock(database_path):
        initialize_database(database_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("snapshot_type") != "research_market":
            raise ValueError("manifest is not a Phase 5 research snapshot")
        with duckdb.connect(str(database_path)) as connection:
            connection.execute("BEGIN TRANSACTION")
            try:
                _sync_research_manifest(connection, data_root, manifest)
                connection.execute(
                    """
                    INSERT INTO meta.repository_state (key, value)
                    VALUES ('latest_research_snapshot_id', ?)
                    ON CONFLICT (key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                    """,
                    [manifest["snapshot_id"]],
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise


def _sync_research_manifest(
    connection: duckdb.DuckDBPyConnection,
    data_root: Path,
    manifest: dict[str, Any],
) -> tuple[set[str], int]:
    snapshot_id = str(manifest["snapshot_id"])
    summary = manifest["summary"]
    scope = manifest["scope"]
    quality_status = str(manifest["quality_status"])
    snapshot_status = "invalid" if quality_status == "error" else "valid"
    connection.execute(
        """
        INSERT INTO meta.dataset_snapshot
            (snapshot_id, snapshot_type, status, identity_sha256,
             schema_version, config_sha256, source_config, universe_config,
             row_count, security_count, start_date, end_date, quality_status)
        VALUES (?, 'research_market', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (snapshot_id) DO NOTHING
        """,
        [
            snapshot_id,
            snapshot_status,
            manifest["identity_sha256"],
            int(manifest["schema_version"]),
            manifest["identity_sha256"],
            _canonical_json(manifest["source"]),
            _canonical_json(scope),
            int(summary.get("daily_bar_count", 0)),
            int(summary["security_count"]),
            scope["start_date"],
            scope["end_date"],
            quality_status,
        ],
    )

    artifact_ids: set[str] = set()
    artifact_by_name: dict[str, str] = {}
    for artifact in manifest.get("artifacts", []):
        name = str(artifact["name"])
        dataset_name = _research_dataset_name(name)
        artifact_id = _upsert_artifact(
            connection,
            data_root,
            layer="research",
            dataset_name=dataset_name,
            relative_path=str(artifact["path"]),
            artifact_format=str(artifact.get("format", "parquet")),
            sha256=str(artifact["sha256"]),
            schema_version=int(manifest["schema_version"]),
            source_id="tushare",
            row_count=int(artifact["row_count"]),
            min_event_date=scope["start_date"],
            max_event_date=scope["end_date"],
        )
        _link_artifact(connection, snapshot_id, artifact_id, dataset_name)
        artifact_ids.add(artifact_id)
        artifact_by_name[name] = artifact_id

    for raw_input in manifest.get("raw_inputs", []):
        artifact_id = _upsert_artifact(
            connection,
            data_root,
            layer="raw",
            dataset_name=f"tushare.{raw_input['api_name']}",
            relative_path=str(raw_input["path"]),
            artifact_format="parquet",
            sha256=str(raw_input["sha256"]),
            schema_version=int(manifest["schema_version"]),
            source_id="tushare",
            row_count=int(raw_input["row_count"]),
            min_event_date=scope["start_date"],
            max_event_date=scope["end_date"],
        )
        _link_artifact(
            connection, snapshot_id, artifact_id, f"tushare.{raw_input['api_name']}"
        )
        artifact_ids.add(artifact_id)

    security_path = _research_artifact_path(
        data_root, manifest, "security_master.parquet"
    )
    name_path = _research_artifact_path(
        data_root, manifest, "security_name_history.parquet"
    )
    membership_path = _research_artifact_path(
        data_root, manifest, "index_membership.parquet"
    )
    security = pd.read_parquet(security_path)
    names = pd.read_parquet(name_path)
    membership = pd.read_parquet(membership_path)
    security_artifact = artifact_by_name.get("security_master.parquet")
    name_artifact = artifact_by_name.get("security_name_history.parquet")
    membership_artifact = artifact_by_name.get("index_membership.parquet")
    _sync_research_securities(connection, security, security_artifact)
    _sync_security_names(connection, names, name_artifact)
    _sync_index_membership(connection, membership, membership_artifact)
    quality_count = _sync_research_quality(connection, data_root, snapshot_id, manifest)
    return artifact_ids, quality_count


def _sync_research_securities(
    connection: duckdb.DuckDBPyConnection,
    frame: pd.DataFrame,
    source_artifact_id: str | None,
) -> None:
    listing_status = {
        "L": "listed",
        "D": "delisted",
        "P": "suspended_listing",
        "G": "prelisted",
    }
    for row in frame.to_dict("records"):
        security_id = str(row["security_id"])
        connection.execute(
            """
            INSERT INTO ref.security
                (security_id, asset_type, exchange, board, currency, lot_size)
            VALUES (?, 'stock', ?, ?, ?, 100)
            ON CONFLICT (security_id) DO NOTHING
            """,
            [
                security_id,
                row["exchange"],
                row.get("board"),
                row.get("currency") or "CNY",
            ],
        )
        known_at = row.get("known_at") or pd.Timestamp.now(tz="UTC")
        for identifier_type, identifier_value in (
            ("tushare_code", row.get("ts_code")),
            ("symbol", row.get("symbol")),
            ("name", row.get("name")),
        ):
            if identifier_value is None:
                continue
            identifier_id = hashlib.sha256(
                f"{security_id}|{identifier_type}|{identifier_value}|{row['list_date']}".encode()
            ).hexdigest()
            connection.execute(
                """
                INSERT INTO ref.security_identifier_history
                    (identifier_id, security_id, identifier_type, identifier_value,
                     valid_from, known_at, source_artifact_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (identifier_id) DO NOTHING
                """,
                [
                    identifier_id,
                    security_id,
                    identifier_type,
                    str(identifier_value),
                    row["list_date"],
                    known_at,
                    source_artifact_id,
                ],
            )
        connection.execute(
            """
            INSERT INTO ref.security_lifecycle
                (security_id, list_date, delist_date, listing_status,
                 known_at, source_artifact_id)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (security_id) DO UPDATE SET
                list_date = excluded.list_date,
                delist_date = excluded.delist_date,
                listing_status = excluded.listing_status,
                known_at = excluded.known_at,
                source_artifact_id = excluded.source_artifact_id
            """,
            [
                security_id,
                row["list_date"],
                row.get("delist_date"),
                listing_status.get(str(row.get("list_status")), "unknown"),
                known_at,
                source_artifact_id,
            ],
        )


def _sync_security_names(
    connection: duckdb.DuckDBPyConnection,
    frame: pd.DataFrame,
    source_artifact_id: str | None,
) -> None:
    for row in frame.to_dict("records"):
        name_id = hashlib.sha256(
            f"{row['security_id']}|{row['effective_from']}|{row['name']}".encode()
        ).hexdigest()
        connection.execute(
            """
            INSERT INTO ref.security_name_history
                (name_history_id, security_id, security_name, is_st,
                 effective_from, effective_to, announced_at, known_at,
                 source_artifact_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (name_history_id) DO NOTHING
            """,
            [
                name_id,
                row["security_id"],
                row["name"],
                row.get("is_st"),
                row["effective_from"],
                row.get("effective_to"),
                row.get("announced_at"),
                row["known_at"],
                source_artifact_id,
            ],
        )


def _sync_index_membership(
    connection: duckdb.DuckDBPyConnection,
    frame: pd.DataFrame,
    source_artifact_id: str | None,
) -> None:
    connection.execute(
        """
        INSERT INTO ref.index_definition
            (index_id, index_code, index_name, exchange, provider)
        VALUES ('CN:INDEX:000300.SH', '000300.SH', '沪深300', 'SSE', 'tushare')
        ON CONFLICT (index_id) DO NOTHING
        """
    )
    for row in frame.to_dict("records"):
        membership_id = hashlib.sha256(
            f"{row['index_id']}|{row['security_id']}|{row['effective_from']}".encode()
        ).hexdigest()
        connection.execute(
            """
            INSERT INTO ref.index_membership_history
                (membership_id, index_id, security_id, effective_from,
                 effective_to, announced_at, known_at, weight,
                 membership_method, source_artifact_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (membership_id) DO NOTHING
            """,
            [
                membership_id,
                row["index_id"],
                row["security_id"],
                row["effective_from"],
                row.get("effective_to"),
                row.get("announced_at"),
                row.get("known_at"),
                row.get("weight"),
                row.get("membership_method"),
                source_artifact_id,
            ],
        )


def _sync_research_quality(
    connection: duckdb.DuckDBPyConnection,
    data_root: Path,
    snapshot_id: str,
    manifest: dict[str, Any],
) -> int:
    quality = manifest.get("quality_report")
    if not quality:
        return 0
    path = data_root / str(quality["path"])
    if not path.is_file():
        return 0
    report = json.loads(path.read_text(encoding="utf-8"))
    for name, check in report.get("checks", {}).items():
        count = int(check["count"])
        connection.execute(
            """
            INSERT INTO meta.quality_result
                (snapshot_id, dataset_name, check_name, severity, status,
                 observed_value, threshold_value, affected_rows, details)
            VALUES (?, 'research.point_in_time', ?, ?, ?, ?, 0, ?, ?)
            ON CONFLICT (snapshot_id, dataset_name, check_name) DO UPDATE SET
                severity = excluded.severity,
                status = excluded.status,
                observed_value = excluded.observed_value,
                affected_rows = excluded.affected_rows,
                details = excluded.details
            """,
            [
                snapshot_id,
                name,
                check["severity"],
                check["status"],
                count,
                count,
                _canonical_json(check),
            ],
        )
    return len(report.get("checks", {}))


def _research_artifact_path(
    data_root: Path, manifest: dict[str, Any], name: str
) -> Path:
    for artifact in manifest.get("artifacts", []):
        if artifact.get("name") == name:
            path = data_root / str(artifact["path"])
            if not path.is_file():
                raise ValueError(f"research snapshot artifact is missing: {path}")
            return path
    raise ValueError(f"research snapshot manifest is missing artifact: {name}")


def _research_dataset_name(name: str) -> str:
    prefix = name.split("/", maxsplit=1)[0].removesuffix(".parquet")
    return {
        "security_master": "ref.security_master",
        "security_name_history": "ref.security_name_history",
        "trading_calendar": "ref.trading_calendar",
        "index_membership": "ref.index_membership",
        "suspension": "market.suspension",
        "universe_dates": "research.universe_dates",
        "daily_bar": "market.daily_bar",
        "adjustment_factor": "market.adjustment_factor",
        "daily_status": "market.daily_status",
    }.get(prefix, f"research.{prefix}")


def _sync_security(
    connection: duckdb.DuckDBPyConnection, code: str, name: str, as_of: date
) -> None:
    security_id, exchange, board = _security_identity(code)
    connection.execute(
        """
        INSERT INTO ref.security
            (security_id, asset_type, exchange, board, currency, lot_size)
        VALUES (?, 'stock', ?, ?, 'CNY', 100)
        ON CONFLICT (security_id) DO NOTHING
        """,
        [security_id, exchange, board],
    )
    identifiers = (
        ("symbol", code),
        ("qlib_code", to_qlib_instrument(code)),
        ("name", name),
    )
    for identifier_type, identifier_value in identifiers:
        identifier_id = hashlib.sha256(
            f"{security_id}|{identifier_type}|{identifier_value}|{as_of}".encode()
        ).hexdigest()
        connection.execute(
            """
            INSERT INTO ref.security_identifier_history
                (identifier_id, security_id, identifier_type, identifier_value,
                 valid_from)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (identifier_id) DO NOTHING
            """,
            [identifier_id, security_id, identifier_type, identifier_value, as_of],
        )
    connection.execute(
        """
        INSERT INTO ref.security_lifecycle (security_id, listing_status)
        VALUES (?, 'unknown')
        ON CONFLICT (security_id) DO NOTHING
        """,
        [security_id],
    )


def _security_identity(code: str) -> tuple[str, str, str]:
    if code.startswith("6"):
        exchange = "SSE"
        board = "star" if code.startswith("688") else "main"
    elif code.startswith(("0", "3")):
        exchange = "SZSE"
        board = "chinext" if code.startswith("3") else "main"
    elif code.startswith(("4", "8")):
        exchange = "BSE"
        board = "bse"
    else:
        raise ValueError(f"unsupported A-share code: {code}")
    return f"CN:{exchange}:{code}", exchange, board


def _sync_manifest(
    connection: duckdb.DuckDBPyConnection,
    data_root: Path,
    manifest: dict[str, Any],
) -> tuple[set[str], int]:
    snapshot_id = str(manifest["snapshot_id"])
    summary = manifest["summary"]
    quality_status = str(summary["quality_status"])
    snapshot_status = "invalid" if quality_status == "error" else "valid"
    connection.execute(
        """
        INSERT INTO meta.dataset_snapshot
            (snapshot_id, snapshot_type, status, identity_sha256,
             schema_version, config_sha256, source_config, universe_config,
             row_count, security_count, start_date, end_date, quality_status)
        VALUES (?, 'market', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (snapshot_id) DO NOTHING
        """,
        [
            snapshot_id,
            snapshot_status,
            manifest["identity_sha256"],
            int(manifest["schema_version"]),
            manifest["identity_sha256"],
            _canonical_json(manifest["source"]),
            _canonical_json(manifest["universe"]),
            int(summary["row_count"]),
            int(summary["instrument_count"]),
            summary["date_start"],
            summary["date_end"],
            quality_status,
        ],
    )

    artifact_ids: set[str] = set()
    for raw_input in manifest.get("raw_inputs", []):
        artifact_id = _upsert_artifact(
            connection,
            data_root,
            layer="raw",
            dataset_name="market.daily_bar_raw",
            relative_path=str(raw_input["path"]),
            artifact_format="parquet",
            sha256=str(raw_input["sha256"]),
            schema_version=int(manifest["schema_version"]),
            source_id=str(raw_input.get("provider") or "") or None,
            row_count=int(raw_input["row_count"]),
            min_event_date=raw_input.get("requested_start"),
            max_event_date=raw_input.get("requested_end"),
        )
        _link_artifact(connection, snapshot_id, artifact_id, "market.daily_bar_raw")
        artifact_ids.add(artifact_id)

    for artifact_name, artifact in manifest.get("artifacts", {}).items():
        layer = "report" if artifact_name == "quality_report" else artifact_name
        dataset_name = (
            "meta.quality_report"
            if artifact_name == "quality_report"
            else f"market.daily_bar_{artifact_name}"
        )
        artifact_id = _upsert_artifact(
            connection,
            data_root,
            layer=layer,
            dataset_name=dataset_name,
            relative_path=str(artifact["path"]),
            artifact_format="json" if artifact_name == "quality_report" else "parquet",
            sha256=str(artifact["sha256"]),
            schema_version=int(manifest["schema_version"]),
            source_id=None,
            row_count=(
                int(summary["row_count"])
                if artifact_name in {"bronze", "silver"}
                else None
            ),
            min_event_date=summary["date_start"],
            max_event_date=summary["date_end"],
        )
        _link_artifact(connection, snapshot_id, artifact_id, dataset_name)
        artifact_ids.add(artifact_id)

    qlib_manifest_path = data_root / "qlib" / snapshot_id / "export_manifest.json"
    if qlib_manifest_path.is_file():
        qlib_manifest = json.loads(qlib_manifest_path.read_text(encoding="utf-8"))
        artifact_id = _upsert_artifact(
            connection,
            data_root,
            layer="qlib",
            dataset_name="qlib.file_storage",
            relative_path=f"qlib/{snapshot_id}",
            artifact_format="qlib-file-storage",
            sha256=str(qlib_manifest["content_sha256"]),
            schema_version=int(qlib_manifest["schema_version"]),
            source_id=None,
            row_count=int(summary["row_count"]),
            min_event_date=summary["date_start"],
            max_event_date=summary["date_end"],
        )
        _link_artifact(connection, snapshot_id, artifact_id, "qlib.file_storage")
        artifact_ids.add(artifact_id)

    quality_count = _sync_quality_results(connection, data_root, snapshot_id, manifest)
    return artifact_ids, quality_count


def _upsert_artifact(
    connection: duckdb.DuckDBPyConnection,
    data_root: Path,
    *,
    layer: str,
    dataset_name: str,
    relative_path: str,
    artifact_format: str,
    sha256: str,
    schema_version: int,
    source_id: str | None,
    row_count: int | None,
    min_event_date: str | None,
    max_event_date: str | None,
) -> str:
    artifact_id = hashlib.sha256(
        f"{layer}|{relative_path}|{sha256}".encode()
    ).hexdigest()
    artifact_path = data_root / relative_path
    file_size = artifact_path.stat().st_size if artifact_path.is_file() else None
    connection.execute(
        """
        INSERT INTO meta.artifact
            (artifact_id, source_id, layer, dataset_name, relative_path,
             format, sha256, file_size_bytes, row_count, min_event_date,
             max_event_date, schema_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (artifact_id) DO NOTHING
        """,
        [
            artifact_id,
            source_id,
            layer,
            dataset_name,
            relative_path,
            artifact_format,
            sha256,
            file_size,
            row_count,
            min_event_date,
            max_event_date,
            schema_version,
        ],
    )
    return artifact_id


def _link_artifact(
    connection: duckdb.DuckDBPyConnection,
    snapshot_id: str,
    artifact_id: str,
    dataset_name: str,
) -> None:
    connection.execute(
        """
        INSERT INTO meta.snapshot_artifact
            (snapshot_id, artifact_id, dataset_name)
        VALUES (?, ?, ?)
        ON CONFLICT (snapshot_id, artifact_id) DO NOTHING
        """,
        [snapshot_id, artifact_id, dataset_name],
    )


def _sync_quality_results(
    connection: duckdb.DuckDBPyConnection,
    data_root: Path,
    snapshot_id: str,
    manifest: dict[str, Any],
) -> int:
    report_artifact = manifest.get("artifacts", {}).get("quality_report")
    if report_artifact is None:
        return 0
    report_path = data_root / str(report_artifact["path"])
    if not report_path.is_file():
        return 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks = (
        (
            "duplicate_keys",
            "error",
            int(report["duplicates"]["count"]),
            report["duplicates"],
        ),
        (
            "invalid_rows",
            "error",
            int(report["invalid_rows"]["count"]),
            report["invalid_rows"],
        ),
        (
            "missing_instruments",
            "error",
            len(report.get("missing_instruments", [])),
            report.get("missing_instruments", []),
        ),
        (
            "missing_status_fields",
            "warning",
            len(report.get("missing_status_fields", [])),
            report.get("missing_status_fields", []),
        ),
    )
    for check_name, severity, affected_rows, details in checks:
        connection.execute(
            """
            INSERT INTO meta.quality_result
                (snapshot_id, dataset_name, check_name, severity, status,
                 observed_value, threshold_value, affected_rows, details)
            VALUES (?, 'market.daily_bar', ?, ?, ?, ?, 0, ?, ?)
            ON CONFLICT (snapshot_id, dataset_name, check_name) DO UPDATE SET
                severity = excluded.severity,
                status = excluded.status,
                observed_value = excluded.observed_value,
                affected_rows = excluded.affected_rows,
                details = excluded.details
            """,
            [
                snapshot_id,
                check_name,
                severity,
                "pass" if affected_rows == 0 else "fail",
                affected_rows,
                affected_rows,
                _canonical_json(details),
            ],
        )
    return len(checks)


@contextmanager
def _catalog_write_lock(database_path: Path) -> Iterator[None]:
    lock_path = database_path.with_suffix(f"{database_path.suffix}.lockdir")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 60.0
    while True:
        try:
            lock_path.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for DuckDB catalog lock: {lock_path}"
                ) from None
            time.sleep(0.1)
    try:
        yield
    finally:
        lock_path.rmdir()


def record_baseline_run(
    database_path: Path,
    config_dir: Path,
    data_root: Path,
    manifest_path: Path,
) -> None:
    with _catalog_write_lock(database_path):
        _record_baseline_run_unlocked(
            database_path, config_dir, data_root, manifest_path
        )


def _record_baseline_run_unlocked(
    database_path: Path,
    config_dir: Path,
    data_root: Path,
    manifest_path: Path,
) -> None:
    """Record one successful Phase 2 run and its artifacts in the catalog."""
    from alpha_lab.baseline.config import load_phase2_config

    sync_repository_metadata(database_path, config_dir, data_root)
    phase1 = load_phase1_config(config_dir)
    phase2 = load_phase2_config(config_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = str(manifest["run_id"])
    snapshot_id = str(manifest["data_snapshot_id"])
    output_dir = manifest_path.parent

    with duckdb.connect(str(database_path)) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            for policy_id, policy_type, policy_hash in (
                (phase2.splits.policy_id, "split", phase2.split_sha256),
                (phase2.costs.policy_id, "cost", phase2.cost_sha256),
            ):
                connection.execute(
                    """
                    INSERT INTO policy.policy_version
                        (policy_id, policy_type, version, config_sha256, locked)
                    VALUES (?, ?, '1', ?, true)
                    ON CONFLICT (policy_id) DO NOTHING
                    """,
                    [policy_id, policy_type, policy_hash],
                )

            for index, rule in enumerate(phase2.costs.rules):
                for side in ("buy", "sell"):
                    stamp = (
                        rule.stamp_duty_rate_buy
                        if side == "buy"
                        else rule.stamp_duty_rate_sell
                    )
                    transfer = (
                        rule.transfer_fee_rate_buy
                        if side == "buy"
                        else rule.transfer_fee_rate_sell
                    )
                    connection.execute(
                        """
                        INSERT INTO policy.cost_rule
                            (rule_id, policy_id, market, side, effective_from,
                             effective_to, commission_rate, minimum_commission,
                             stamp_duty_rate, transfer_fee_rate)
                        VALUES (?, ?, 'CN_A_SHARE', ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (rule_id) DO NOTHING
                        """,
                        [
                            f"{phase2.costs.policy_id}-{index}-{side}",
                            phase2.costs.policy_id,
                            side,
                            rule.effective_from,
                            rule.effective_to,
                            rule.commission_rate,
                            rule.minimum_commission,
                            stamp,
                            transfer,
                        ],
                    )

            connection.execute(
                """
                INSERT INTO research.experiment_run
                    (experiment_id, data_snapshot_id, universe_id,
                     split_policy_sha256, cost_policy_sha256, code_commit,
                     random_seed, status, started_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'success',
                        current_timestamp, current_timestamp)
                ON CONFLICT (experiment_id) DO NOTHING
                """,
                [
                    run_id,
                    snapshot_id,
                    phase1.universe.sample_id,
                    phase2.split_sha256,
                    phase2.cost_sha256,
                    str(manifest["git"]["commit"]),
                    phase2.baseline.random_seed,
                ],
            )

            metrics = {
                **{
                    f"signal_{key}": value
                    for key, value in manifest["signal_analysis"].items()
                    if key != "daily"
                },
                **{
                    f"backtest_{key}": value
                    for key, value in manifest["backtest"]["metrics"].items()
                },
            }
            for name, value in metrics.items():
                if value is None or not isinstance(value, (int, float)):
                    continue
                connection.execute(
                    """
                    INSERT INTO research.experiment_metric
                        (experiment_id, split_name, metric_name, metric_value)
                    VALUES (?, 'validation', ?, ?)
                    ON CONFLICT
                        (experiment_id, split_name, metric_name, period)
                    DO NOTHING
                    """,
                    [run_id, name, float(value)],
                )

            artifact_ids: dict[str, str] = {}
            artifact_specs = {
                "predictions.parquet": ("research", "baseline.signal", "parquet"),
                "backtest_daily.parquet": (
                    "research",
                    "baseline.backtest_daily",
                    "parquet",
                ),
                "trades.parquet": ("research", "baseline.trades", "parquet"),
                "lightgbm_model.txt": ("research", "baseline.model", "lightgbm"),
                "run_manifest.json": ("report", "baseline.manifest", "json"),
                "baseline_report.md": ("report", "baseline.report", "markdown"),
                "baseline_report.html": ("report", "baseline.report", "html"),
            }
            for name, (layer, dataset_name, artifact_format) in artifact_specs.items():
                path = output_dir / name
                relative_path = Path(
                    os.path.relpath(path.resolve(), start=data_root.resolve())
                ).as_posix()
                artifact_ids[name] = _upsert_artifact(
                    connection,
                    data_root,
                    layer=layer,
                    dataset_name=dataset_name,
                    relative_path=relative_path,
                    artifact_format=artifact_format,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    schema_version=1,
                    source_id=None,
                    row_count=None,
                    min_event_date=None,
                    max_event_date=None,
                )

            connection.execute(
                """
                INSERT INTO research.backtest_run
                    (backtest_id, experiment_id, data_snapshot_id,
                     signal_artifact_id, position_artifact_id,
                     trade_artifact_id, report_artifact_id, status, summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'success', ?)
                ON CONFLICT (backtest_id) DO NOTHING
                """,
                [
                    f"{run_id}-topk",
                    run_id,
                    snapshot_id,
                    artifact_ids["predictions.parquet"],
                    artifact_ids["backtest_daily.parquet"],
                    artifact_ids["trades.parquet"],
                    artifact_ids["baseline_report.html"],
                    _canonical_json(manifest["backtest"]),
                ],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise


def record_factor_evaluation(
    database_path: Path,
    config_dir: Path,
    data_root: Path,
    result_path: Path,
) -> None:
    with _catalog_write_lock(database_path):
        _record_factor_evaluation_unlocked(
            database_path, config_dir, data_root, result_path
        )


def record_mining_decision(
    database_path: Path,
    experiment_id: str,
    policy_version: str,
    decision_path: Path,
) -> None:
    """Record one immutable Phase 4 recommendation for an evaluated experiment."""
    from alpha_lab.mining.models import MiningDecision

    decision = MiningDecision.model_validate_json(
        decision_path.read_text(encoding="utf-8")
    )
    reason = decision.model_dump(mode="json")
    reason_json = _canonical_json(reason)
    with _catalog_write_lock(database_path):
        initialize_database(database_path)
        with duckdb.connect(str(database_path)) as connection:
            connection.execute(
                """
                INSERT INTO research.experiment_decision
                    (experiment_id, decision, reason, policy_version)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (experiment_id) DO NOTHING
                """,
                [
                    experiment_id,
                    decision.decision.lower(),
                    reason_json,
                    policy_version,
                ],
            )
            stored = connection.execute(
                """
                SELECT decision, reason, policy_version
                FROM research.experiment_decision
                WHERE experiment_id = ?
                """,
                [experiment_id],
            ).fetchone()
            if stored is None:
                raise RuntimeError(
                    f"mining decision was not recorded for {experiment_id}"
                )
            expected = (decision.decision.lower(), reason_json, policy_version)
            actual = (
                str(stored[0]),
                _canonical_json(json.loads(str(stored[1]))),
                str(stored[2]),
            )
            if actual != expected:
                raise RuntimeError(
                    f"mining decision is immutable for experiment {experiment_id}"
                )


def _record_factor_evaluation_unlocked(
    database_path: Path,
    config_dir: Path,
    data_root: Path,
    result_path: Path,
) -> None:
    """Record a Phase 3 factor version, evaluation, and artifacts."""
    sync_repository_metadata(database_path, config_dir, data_root)
    phase1 = load_phase1_config(config_dir)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    factor = result["factor"]
    factor_id = str(factor["factor_id"])
    run_id = str(result["run_id"])
    version_id = f"{factor_id}-{str(result['factor_source_sha256'])[:20]}"

    with duckdb.connect(str(database_path)) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(
                """
                INSERT INTO policy.policy_version
                    (policy_id, policy_type, version, config_sha256, locked)
                VALUES (?, 'factor_evaluation', '1', ?, true)
                ON CONFLICT (policy_id) DO NOTHING
                """,
                [
                    result["evaluation_policy_id"],
                    result["evaluation_config_sha256"],
                ],
            )
            connection.execute(
                """
                INSERT INTO research.factor_definition
                    (factor_id, name, family, description)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (factor_id) DO NOTHING
                """,
                [
                    factor_id,
                    factor["name"],
                    factor["family"],
                    factor["hypothesis"],
                ],
            )
            connection.execute(
                """
                INSERT INTO research.factor_version
                    (factor_version_id, factor_id, formula,
                     implementation_path, code_sha256, metadata_sha256,
                     lookback, direction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (factor_version_id) DO NOTHING
                """,
                [
                    version_id,
                    factor_id,
                    factor["formula"],
                    result["implementation_path"],
                    result["factor_source_sha256"],
                    result["factor_metadata_sha256"],
                    factor["lookback"],
                    factor["direction"],
                ],
            )
            connection.execute(
                """
                INSERT INTO research.experiment_run
                    (experiment_id, factor_version_id, data_snapshot_id,
                     universe_id, split_policy_sha256, cost_policy_sha256,
                     code_commit, status, started_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'success',
                        current_timestamp, current_timestamp)
                ON CONFLICT (experiment_id) DO NOTHING
                """,
                [
                    run_id,
                    version_id,
                    result["data_snapshot_id"],
                    phase1.universe.sample_id,
                    result["split_policy_sha256"],
                    result["cost_policy_sha256"],
                    result["git"]["commit"],
                ],
            )

            metrics = {
                key: value
                for key, value in result["metrics"].items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            metrics["eligible_for_review"] = float(result["eligible_for_review"])
            metrics["leakage_passed"] = float(result["leakage"]["passed"])
            base_metrics = result["topk_cost_sensitivity"]["scenarios"]["base"][
                "metrics"
            ]
            for key, value in base_metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[f"topk_{key}"] = value
            for key, value in result["correlations"].items():
                if value is not None:
                    metrics[f"correlation_{key}"] = value
            for name, value in metrics.items():
                connection.execute(
                    """
                    INSERT INTO research.experiment_metric
                        (experiment_id, split_name, metric_name, metric_value)
                    VALUES (?, 'validation', ?, ?)
                    ON CONFLICT
                        (experiment_id, split_name, metric_name, period)
                    DO NOTHING
                    """,
                    [run_id, name, float(value)],
                )

            artifact_ids: dict[str, str] = {}
            for name, dataset_name, artifact_format in (
                ("factor_values.parquet", "factor.values", "parquet"),
                ("factor_result.json", "factor.result", "json"),
            ):
                path = result_path.parent / name
                relative_path = Path(
                    os.path.relpath(path.resolve(), start=data_root.resolve())
                ).as_posix()
                artifact_ids[name] = _upsert_artifact(
                    connection,
                    data_root,
                    layer="research" if name.endswith(".parquet") else "report",
                    dataset_name=dataset_name,
                    relative_path=relative_path,
                    artifact_format=artifact_format,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    schema_version=1,
                    source_id=None,
                    row_count=(
                        int(result["metrics"]["valid_row_count"])
                        if name.endswith(".parquet")
                        else None
                    ),
                    min_event_date=None,
                    max_event_date=None,
                )
            connection.execute(
                """
                INSERT INTO research.backtest_run
                    (backtest_id, experiment_id, data_snapshot_id,
                     signal_artifact_id, report_artifact_id, status, summary)
                VALUES (?, ?, ?, ?, ?, 'success', ?)
                ON CONFLICT (backtest_id) DO NOTHING
                """,
                [
                    f"{run_id}-topk",
                    run_id,
                    result["data_snapshot_id"],
                    artifact_ids["factor_values.parquet"],
                    artifact_ids["factor_result.json"],
                    _canonical_json(result["topk_cost_sensitivity"]),
                ],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise


def check_database(database_path: Path, data_root: Path) -> dict[str, Any]:
    if not database_path.is_file():
        raise FileNotFoundError(f"DuckDB catalog does not exist: {database_path}")
    with duckdb.connect(str(database_path), read_only=True) as connection:
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
        migration = connection.execute(
            """
            SELECT version, sha256
            FROM meta.schema_migration
            ORDER BY version DESC
            LIMIT 1
            """
        ).fetchone()
        latest = connection.execute(
            "SELECT value FROM meta.repository_state WHERE key = 'latest_snapshot_id'"
        ).fetchone()
        artifact_paths = connection.execute(
            "SELECT relative_path FROM meta.artifact ORDER BY relative_path"
        ).fetchall()
        missing_artifacts = [
            str(path)
            for (path,) in artifact_paths
            if not (data_root / str(path)).exists()
        ]
        report: dict[str, Any] = {
            "database_path": str(database_path),
            "schema_version": int(migration[0]) if migration else None,
            "migration_sha256": str(migration[1]) if migration else None,
            "missing_tables": sorted(EXPECTED_TABLES - tables),
            "data_source_count": _scalar_count(connection, "meta.data_source"),
            "dataset_contract_count": _scalar_count(
                connection, "meta.dataset_contract"
            ),
            "security_count": _scalar_count(connection, "ref.security"),
            "snapshot_count": _scalar_count(connection, "meta.dataset_snapshot"),
            "artifact_count": _scalar_count(connection, "meta.artifact"),
            "quality_result_count": _scalar_count(connection, "meta.quality_result"),
            "latest_snapshot_id": str(latest[0]) if latest else None,
            "missing_artifact_files": missing_artifacts,
            "logical_orphans": _logical_orphans(connection),
        }
        report["healthy"] = (
            not report["missing_tables"]
            and not missing_artifacts
            and not any(report["logical_orphans"].values())
        )
        return report


def _scalar_count(connection: duckdb.DuckDBPyConnection, table: str) -> int:
    row = connection.execute(f"SELECT count(*) FROM {table}").fetchone()
    if row is None:
        raise RuntimeError(f"count query returned no row for {table}")
    return int(row[0])


def _logical_orphans(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    queries = {
        "universe_membership_without_security": """
            SELECT count(*)
            FROM research.universe_membership AS membership
            LEFT JOIN ref.security AS security USING (security_id)
            WHERE security.security_id IS NULL
        """,
        "corporate_action_without_security": """
            SELECT count(*)
            FROM market.corporate_action AS action
            LEFT JOIN ref.security AS security USING (security_id)
            WHERE security.security_id IS NULL
        """,
        "filing_without_security": """
            SELECT count(*)
            FROM fundamental.filing_catalog AS filing
            LEFT JOIN ref.security AS security USING (security_id)
            WHERE security.security_id IS NULL
        """,
        "fact_artifact_without_artifact": """
            SELECT count(*)
            FROM fundamental.fact_artifact AS fact
            LEFT JOIN meta.artifact AS artifact USING (artifact_id)
            WHERE artifact.artifact_id IS NULL
        """,
        "experiment_without_snapshot": """
            SELECT count(*)
            FROM research.experiment_run AS experiment
            LEFT JOIN meta.dataset_snapshot AS snapshot
              ON experiment.data_snapshot_id = snapshot.snapshot_id
            WHERE snapshot.snapshot_id IS NULL
        """,
        "backtest_without_snapshot": """
            SELECT count(*)
            FROM research.backtest_run AS backtest
            LEFT JOIN meta.dataset_snapshot AS snapshot
              ON backtest.data_snapshot_id = snapshot.snapshot_id
            WHERE snapshot.snapshot_id IS NULL
        """,
    }
    results: dict[str, int] = {}
    for name, query in queries.items():
        row = connection.execute(query).fetchone()
        if row is None:
            raise RuntimeError(f"logical orphan query returned no row: {name}")
        results[name] = int(row[0])
    return results
