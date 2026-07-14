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
definition/security inside `ref`. Inside `research`, each state carries the
same freeze ID, freeze SHA256 and locked test range. Composite candidate keys
and foreign keys bind request to freeze, approval to request, and final-test run
to approval/request; a mismatched hash, freeze, or range is rejected by DuckDB.
Snapshot and artifact references that cross into `meta` are explicit logical
foreign keys backed by immutable SHA256 identifiers. `make db-check` validates
the catalog boundary and fails if an orphan is present.

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
runs the complete Task 2 semantic validation and records the validated manifest
SHA256. Inside the catalog lock and transaction, it re-reads the canonical
manifest, requires the SHA256 to be unchanged, then computes the current SHA256
from every exposure, raw, quality, industry and required Phase 5 file. The
computed digest, not an unchecked declared value, is passed to artifact
registration. Only after those checks does it register the Phase 5 parent and
artifacts and bulk insert industry definitions/membership with
`INSERT ... SELECT ... ON CONFLICT`.

Both canonical `industry_membership.parquet` and its fixed-cutoff
`industry_membership_pretest.parquet` derivative are registered and covered by
the transaction's TOCTOU seal. Only the canonical full artifact populates
`ref.industry_membership_history`; the pre-test derivative remains an external
artifact and is never duplicated into a DuckDB fact table.

The canonical capability is registered as `meta.pretest_data_capability` (JSON
in the report layer) and its physical bytes participate in the final pre-commit
TOCTOU seal. It is metadata only and does not duplicate market facts.

Small Phase 5 reference artifacts, their quality report and Phase 6 industry
artifacts are hashed and parsed from the same in-memory byte buffer, so derived
catalog rows cannot come from bytes different from the verified digest. After
all catalog rows and the latest-state pointer are written, the final fallible
business operation before `COMMIT` re-hashes both canonical manifests and every
exposure, raw, quality, industry and Phase 5 dependency. Any drift rolls back
the transaction.

Repeating the same sync is idempotent. Post-validation mutation, artifact paths
outside `data_dir`, hash drift and unknown references are rejected. Any failure
rolls back Phase 5/exposure rows and the latest-state update together. No daily
market-cap rows are copied into DuckDB. The catalog lock coordinates catalog
writers; a non-cooperating external file writer is detected when the pre-commit
seal observes its change, but cannot be serialized by that lock. Operationally,
`data/raw` remains immutable and append-only, and snapshot dependencies must not
be overwritten.

The normalized state tables are:

- `research.factor_freeze`: exact candidate/data/policy hashes and locked range;
  exposes a unique `(freeze_id, freeze_sha256, test_start, test_end)` identity.
- `research.test_request`: carries and composite-references that exact freeze
  identity.
- `research.test_approval`: carries the request, freeze, confirmed freeze hash
  and range and composite-references the exact request identity.
- `research.final_test_run`: carries approval, request, freeze, hash and range
  and composite-references the exact approved identity.

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
