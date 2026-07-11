from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from alpha_lab.database.catalog import (
    EXPECTED_TABLES,
    check_database,
    initialize_database,
    record_baseline_run,
    record_factor_evaluation,
    record_mining_decision,
    sync_repository_metadata,
)

ROOT = Path(__file__).resolve().parents[2]


def test_database_initialization_is_versioned_idempotent_and_constrained(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "data" / "metadata.duckdb"

    first = initialize_database(database_path)
    second = initialize_database(database_path)

    assert first.schema_version == 2
    assert second.schema_version == 2
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
        assert migration_count == 2
        assert contract_count >= 12

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
    assert report["schema_version"] == 2
    assert report["security_count"] == 10
    assert report["snapshot_count"] == 1
    assert report["artifact_count"] == 2
    assert report["latest_snapshot_id"] == snapshot_id
    assert report["missing_artifact_files"] == []


def test_baseline_run_registration_is_idempotent(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    database_path = data_root / "metadata.duckdb"
    initialize_database(database_path)
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            INSERT INTO meta.dataset_snapshot
                (snapshot_id, snapshot_type, status, identity_sha256,
                 schema_version)
            VALUES ('p1-baseline-test', 'market', 'valid', ?, 1)
            """,
            ["a" * 64],
        )

    output_dir = tmp_path / "artifacts" / "baseline" / "run-test"
    output_dir.mkdir(parents=True)
    for name in (
        "predictions.parquet",
        "backtest_daily.parquet",
        "trades.parquet",
        "lightgbm_model.txt",
        "baseline_report.md",
        "baseline_report.html",
    ):
        (output_dir / name).write_text(name, encoding="utf-8")
    manifest = {
        "run_id": "run-test",
        "data_snapshot_id": "p1-baseline-test",
        "git": {"commit": "b" * 40},
        "signal_analysis": {"mean_ic": 0.1, "daily": []},
        "backtest": {"metrics": {"total_return": 0.02}, "constraints": {}},
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    record_baseline_run(database_path, ROOT / "config", data_root, manifest_path)
    record_baseline_run(database_path, ROOT / "config", data_root, manifest_path)

    with duckdb.connect(str(database_path), read_only=True) as connection:
        counts = connection.execute(
            """
            SELECT
                (SELECT count(*) FROM research.experiment_run),
                (SELECT count(*) FROM research.backtest_run),
                (SELECT count(*) FROM research.experiment_metric),
                (SELECT count(*) FROM policy.policy_version),
                (SELECT count(*) FROM policy.cost_rule),
                (SELECT count(*) FROM meta.artifact)
            """
        ).fetchone()

    assert counts == (1, 1, 2, 2, 2, 7)


def test_factor_evaluation_registration_is_idempotent(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    database_path = data_root / "metadata.duckdb"
    initialize_database(database_path)
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            INSERT INTO meta.dataset_snapshot
                (snapshot_id, snapshot_type, status, identity_sha256,
                 schema_version)
            VALUES ('p1-factor-test', 'market', 'valid', ?, 1)
            """,
            ["c" * 64],
        )

    output_dir = tmp_path / "artifacts" / "factors" / "factor-f0001-test"
    output_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"trade_date": pd.Timestamp("2024-01-02"), "instrument": "A", "value": 1.0}]
    ).to_parquet(output_dir / "factor_values.parquet", index=False)
    result = {
        "run_id": "factor-f0001-test",
        "factor": {
            "factor_id": "F0001",
            "name": "momentum_20d",
            "family": "momentum",
            "hypothesis": "A sufficiently long test hypothesis.",
            "formula": "close / Ref(close, 20) - 1",
            "lookback": 21,
            "direction": 1,
        },
        "factor_source_sha256": "d" * 64,
        "factor_metadata_sha256": "e" * 64,
        "implementation_path": "src/alpha_lab/factors/candidates/F0001.py",
        "evaluation_policy_id": "phase3-test-policy",
        "evaluation_config_sha256": "f" * 64,
        "data_snapshot_id": "p1-factor-test",
        "split_policy_sha256": "1" * 64,
        "cost_policy_sha256": "2" * 64,
        "git": {"commit": "3" * 40},
        "metrics": {"valid_row_count": 1, "coverage": 1.0},
        "eligible_for_review": False,
        "leakage": {"passed": True},
        "correlations": {"F0002": 0.2},
        "topk_cost_sensitivity": {
            "scenarios": {"base": {"metrics": {"total_return": 0.01}}}
        },
    }
    result_path = output_dir / "factor_result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    def record(_: int) -> None:
        record_factor_evaluation(database_path, ROOT / "config", data_root, result_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(record, range(2)))

    with duckdb.connect(str(database_path), read_only=True) as connection:
        counts = connection.execute(
            """
            SELECT
                (SELECT count(*) FROM research.factor_definition),
                (SELECT count(*) FROM research.factor_version),
                (SELECT count(*) FROM research.experiment_run),
                (SELECT count(*) FROM research.backtest_run),
                (SELECT count(*) FROM research.experiment_metric),
                (SELECT count(*) FROM meta.artifact)
            """
        ).fetchone()

    assert counts == (1, 1, 1, 1, 6, 2)


def test_mining_decision_registration_is_idempotent(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.duckdb"
    initialize_database(database_path)
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            INSERT INTO meta.dataset_snapshot
                (snapshot_id, snapshot_type, status, identity_sha256, schema_version)
            VALUES ('p1-decision-test', 'market', 'valid', ?, 1)
            """,
            ["a" * 64],
        )
        connection.execute(
            """
            INSERT INTO ref.security
                (security_id, asset_type, exchange, currency, lot_size)
            VALUES ('CN:SH:600519', 'stock', 'SSE', 'CNY', 100)
            """
        )
        connection.execute(
            """
            INSERT INTO research.universe_definition
                (universe_id, name, construction_rule, survivorship_free,
                 research_eligible, config_sha256)
            VALUES ('u-test', 'test', '{}', false, false, ?)
            """,
            ["9" * 64],
        )
        connection.execute(
            """
            INSERT INTO research.experiment_run
                (experiment_id, data_snapshot_id, universe_id,
                 split_policy_sha256, code_commit, status)
            VALUES ('factor-f1000-test', 'p1-decision-test', 'u-test', ?, ?, 'success')
            """,
            ["b" * 64, "c" * 40],
        )

    decision_path = tmp_path / "decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "phase4-test",
                "round_number": 1,
                "factor_id": "F1000",
                "decision": "REJECT",
                "rationale": "Fixed promotion checks did not all pass.",
                "passed_checks": ["leakage"],
                "failed_checks": ["ic"],
                "eligible_for_review": False,
                "human_approval_required": True,
                "factor_result_sha256": "d" * 64,
            }
        ),
        encoding="utf-8",
    )

    record_mining_decision(
        database_path,
        "factor-f1000-test",
        "phase4_factor_mining_v1",
        decision_path,
    )
    record_mining_decision(
        database_path,
        "factor-f1000-test",
        "phase4_factor_mining_v1",
        decision_path,
    )

    with duckdb.connect(str(database_path), read_only=True) as connection:
        row = connection.execute(
            """
            SELECT decision, reason->>'factor_id', reason->>'human_approval_required',
                   policy_version
            FROM research.experiment_decision
            """
        ).fetchone()

    assert row == ("reject", "F1000", "true", "phase4_factor_mining_v1")
