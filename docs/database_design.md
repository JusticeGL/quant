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
| `ref` | Exchanges, securities, identifier history, lifecycle, calendars, industries and indices |
| `market` | Corporate-action catalog and typed Parquet readers for market facts |
| `fundamental` | Filing and metric catalogs plus typed Parquet readers for financial facts |
| `policy` | Versioned price-limit, cost and other locked research policies |
| `research` | Universes, factor versions, experiment metrics, decisions and backtest catalogs |

The initial migration creates 34 base tables, eight external dataset contracts,
three exchange records, two data-source records, and typed table macros for
reading Parquet facts.

## Dataset contracts

The following facts remain external Parquet datasets:

- `market.daily_bar`
- `market.adjustment_factor`
- `market.daily_basic`
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

## DuckDB constraint boundary

DuckDB 1.5.4 enforces primary keys, unique constraints, checks and foreign keys
within one schema, but does not support foreign keys across schemas. Cross-domain
relationships are therefore explicit logical foreign keys. `make db-check`
validates them for universes, corporate actions, filings, experiments and
backtests, and fails if an orphan is present.

## Migrations

Versioned SQL files live under `src/alpha_lab/database/sql/`. An applied
migration is recorded with its SHA256 in `meta.schema_migration`. Applied files
must never be edited: schema changes require a new numbered migration.

Initialization is idempotent. A packaged migration whose SHA256 differs from an
already applied version is rejected.

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
