# DuckDB Catalog and Parquet Storage Design

## Physical boundary

The research platform does not store large market, fundamental, factor, position,
or trade facts inside one database file. Those datasets are immutable Parquet
artifacts. `data/metadata.duckdb` stores their catalog, versions, relationships,
quality results, reference data, policies, and experiment summaries.

Qlib remains a derived representation generated from one fixed silver snapshot.
It is never the source of truth.

## Schemas

| Schema | Purpose |
|---|---|
| `meta` | Migrations, providers, ingestion runs, artifacts, snapshots, quality and dataset contracts |
| `ref` | Exchanges, securities, identifier history, lifecycle, calendars, point-in-time industries and indices |
| `market` | Corporate-action catalog and typed Parquet readers for market facts |
| `fundamental` | Filing and metric catalogs plus typed Parquet readers for financial facts |
| `policy` | Versioned price-limit, cost and other locked research policies |
| `research` | Universes, factor versions, experiments, robustness freezes, approval gates and final-test catalogs |

The initial migration creates the core catalog. Phase 5 adds the immutable
`002_research_data.sql` migration for provider capabilities and security-name
history. Phase 6 adds the immutable `003_robustness.sql` migration for exposure
metadata and the hash-linked freeze, request, approval and final-test state
catalog. Initialization seeds external dataset contracts, exchange and provider
reference rows, and typed table macros for reading Parquet facts.

## Dataset contracts

The following facts remain external Parquet datasets:

- `market.daily_bar`
- `market.adjustment_factor`
- `market.daily_basic`
- `market.exposure_market_cap`
- `market.daily_status`
- `market.corporate_action`
- `fundamental.financial_fact`
- `research.factor_value`
- `research.backtest_daily`

Each contract records its primary key, partition columns, required fields,
point-in-time column and schema version. New files must be registered in
`meta.artifact`; immutable snapshots reference them through
`meta.snapshot_artifact`.

## Point-in-time rules

- A stable internal `security_id` is the identity; symbols and names are
  versioned identifiers.
- Unknown status is `NULL`, not `false` or zero.
- Security, industry and index membership use effective date ranges.
- Fundamental data uses `announcement_date` and `known_at`, not only report
  period.
- A snapshot is immutable and identified by configuration, schema and artifact
  hashes.
- Large experiment payloads stay in Parquet; DuckDB stores paths, hashes and
  summaries.
- Phase 6 industry definitions and point-in-time membership intervals are small
  normalized reference records in DuckDB. Daily total and float market-cap
  values remain partitioned Parquet artifacts.

## DuckDB constraint boundary

DuckDB 1.5.4 enforces primary keys, unique constraints, checks and foreign keys
within one schema, but does not support foreign keys across schemas. Phase 6
therefore uses physical foreign keys from industry membership to industry
definition/security inside `ref`, and from approvals/final-test runs through the
state chain inside `research`. Snapshot and artifact references that cross into
`meta` are explicit logical foreign keys backed by immutable SHA256 identifiers.
`make db-check` validates the catalog boundary and fails if an orphan is present.

## Migrations

Versioned SQL files live under `src/alpha_lab/database/sql/`. An applied
migration is recorded with its SHA256 in `meta.schema_migration`. Applied files
must never be edited: schema changes require a new numbered migration.

Phase 5 uses `002_research_data.sql`; Phase 6 uses
`003_robustness.sql`. `001_initial.sql` and `002_research_data.sql` retain their
recorded SHA256 values. Dynamic index membership continues to use
`ref.index_membership_history`; Phase 6 industry exposure intervals use
`ref.industry_membership_history`. Large daily bars, adjustment factors, status,
market-cap exposure facts and materialized universe dates remain external
Parquet linked through `meta.artifact` and `meta.snapshot_artifact`.

Initialization is idempotent. A packaged migration whose SHA256 differs from an
already applied version is rejected.

## Phase 6 exposure catalog synchronization

`sync_exposure_snapshot(database_path, data_dir, manifest_path)` accepts only a
Task 2 manifest at its canonical `data/manifests/p6x-*/manifest.json` path. It
revalidates the manifest identity, Phase 5 dependency, quality report, artifact
layout and every SHA256 before opening a catalog transaction. It registers the
Phase 5 parent and all exposure/raw/quality artifacts, then bulk inserts industry
definitions and membership history with `INSERT ... SELECT ... ON CONFLICT`.
Repeating the same sync is idempotent. Artifact paths outside `data_dir`, hash
drift and unknown references are rejected; no daily market-cap rows are copied
into DuckDB.

The normalized state tables are:

- `research.factor_freeze`: exact candidate/data/policy hashes and locked range.
- `research.test_request`: one robustness-passed request linked to its freeze.
- `research.test_approval`: named human approval linked to the exact request and
  confirmed freeze hash.
- `research.final_test_run`: immutable final result metadata linked to both the
  approval and freeze.

## Commands

```bash
make db-init
make db-check
```

`db-init` creates the catalog, seeds contracts and reference rows, then syncs
repository universe configuration, manifests, quality reports and Qlib export
metadata. `db-check` is read-only and verifies required tables, logical foreign
keys and artifact paths.

The generated `data/metadata.duckdb` is local state and is excluded from Git.
