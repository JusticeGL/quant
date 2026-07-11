# Phase 5 Research-Grade A-Share Data Design

## Objective and scope

Phase 5 builds a point-in-time research dataset for the CSI 300 index
(`000300.SH`) from 2020-01-01 through the configured current end date. It must
reconstruct the investable universe on sampled historical dates, retain
delisted securities, represent ST and suspension intervals, keep unadjusted
prices separate from adjustment factors, and prevent records from becoming
visible before their announcement or conservative known date.

The phase does not run factor mining, access the locked test split, download
the full A-share daily market, or connect to a broker. Tushare is the primary
enhanced provider. AKShare remains an explicit daily-price fallback only when
equivalent provenance and fields can be preserved.

## Deployment boundary

Research data lives in a focused `alpha_lab.research_data` package with its own
CLI and Make targets. A Compose `data` service uses the same locked Linux
Python 3.11 image but exposes only data commands and mounts the data volume.
This creates a separately deployable service boundary without splitting the
repository or duplicating dependency locks.

The existing Phase 1 data pipeline remains intact. Phase 5 writes new snapshot
types and never mutates prior raw, bronze, silver, Qlib, manifest, or database
migration artifacts.

## Credentials and transport

`TUSHARE_TOKEN` and `TUSHARE_HTTP_URL` are read from the local ignored `.env`.
Neither value is written to logs, exceptions, cache keys, sidecars, manifests,
tests, Git, or generated reports. The provider sends HTTPS POST requests using
the Tushare REST envelope (`api_name`, `params`, and `fields`) and injects the
token only at transport time.

The client uses bounded retries, a request interval, timeouts, and immutable
request caching. Tushare non-zero response codes are typed failures. Permission
errors are reported per capability and are never converted to empty datasets.
The custom HTTP endpoint is configuration, not a hard-coded application
constant.

## Provider capabilities

The Tushare adapter exposes typed retrieval operations for:

- `stock_basic` across listed, delisted, paused, and approved-not-trading
  statuses for security master and lifecycle dates;
- historical CSI 300 membership through an interval-capable endpoint when
  available, otherwise through dated `index_weight` observations reconstructed
  conservatively into membership intervals;
- `trade_cal` for the SSE/SZSE research calendar;
- unadjusted `daily` bars for the union of historical CSI 300 members;
- `adj_factor` stored separately from daily prices;
- `suspend_d` for suspension and resumption events;
- `namechange` for historical names and ST intervals.

The provider probes capabilities before a build. The snapshot manifest records
the selected endpoint, requested fields, row limits, returned coverage, and any
approved fallback. If no provider field supplies an announcement timestamp,
the record receives a conservative `known_at` equal to its effective date and
the manifest marks the source as `effective_date_fallback`. Such records are
never made visible earlier than that date.

## Immutable storage layout

Each network response is stored under:

```text
data/raw/tushare/<api_name>/<request_sha256>.parquet
data/raw/tushare/<api_name>/<request_sha256>.json
```

The sidecar contains the API name, non-secret parameters, requested fields,
provider package version, fetch time, row count, response schema, and Parquet
SHA256. The request identity excludes the token but includes the configured
provider identity, API name, canonical parameters, and fields. Existing files
must match their recorded hash before reuse.

One research snapshot is written atomically under:

```text
data/research/<snapshot_id>/
├── security_master.parquet
├── security_name_history.parquet
├── trading_calendar.parquet
├── index_membership.parquet
├── daily_bar/year=<YYYY>/part.parquet
├── adjustment_factor/year=<YYYY>/part.parquet
├── daily_status/year=<YYYY>/part.parquet
└── universe_dates.parquet
data/manifests/<snapshot_id>/
├── manifest.json
└── quality_report.json
```

Large facts remain immutable Parquet. DuckDB stores catalog rows, policies,
reference intervals, artifact identities, and snapshot relationships rather
than duplicate daily facts.

## Canonical contracts

### Security master

The primary key is `security_id`, normalized as `CN:<exchange>:<symbol>`.
Required lifecycle fields are `list_date`, nullable `delist_date`,
`list_status`, `exchange`, `board`, `currency`, `known_at`, and source artifact
identity. Delisted securities remain present after their last daily bar.

### Index membership

The primary interval key is `(index_id, security_id, effective_from)`.
Intervals carry nullable `effective_to`, nullable provider `announced_at`,
mandatory `known_at`, nullable weight, membership source method, and source
artifact identity. Intervals may not overlap for the same index and security.

The universe on date `D` includes a security only when:

```text
effective_from <= D
and (effective_to is null or D <= effective_to)
and known_at <= D
and list_date <= D
and (delist_date is null or D <= delist_date)
```

### Name and ST history

Name intervals use `(security_id, effective_from)` and retain the exact source
name and reason. `is_st` is derived only from names carrying the ST family of
prefixes during the effective interval. No current name is backfilled into
earlier dates.

### Daily status

The primary key is `(trade_date, security_id)`. It contains nullable
`is_suspended`, `is_st`, suspension type, resumption date, and `known_at`.
Missing events remain null when provider coverage is incomplete; absence of a
row is not interpreted as `false` unless the capability report proves complete
coverage for that date and security.

### Prices and adjustment factors

Daily bars remain unadjusted and use the existing market field semantics.
Adjustment factors have `(trade_date, security_id, factor_type)` as key and
must be finite and positive. Derived forward/backward-adjusted values are
computed views tied to one snapshot and never overwrite raw or canonical
unadjusted prices.

## Snapshot pipeline

The build proceeds deterministically:

1. Load Phase 5 configuration and validate the date range and index code.
2. Probe provider capabilities without persisting secrets.
3. Fetch and cache the security master, index metadata, calendar, and
   membership history.
4. Derive the union of securities that were CSI 300 members during the range.
5. Fetch only that union's daily bars, adjustment factors, suspension events,
   and name history, resuming from verified raw cache intervals.
6. Normalize endpoint outputs independently and build point-in-time intervals.
7. Materialize partitioned Parquet into a temporary snapshot directory.
8. Run quality and point-in-time gates.
9. Write the manifest, atomically publish the snapshot, and synchronize its
   small metadata into DuckDB.

The snapshot identity is a canonical hash of schema versions, Phase 5 config,
all raw artifact hashes, normalization policy, and membership reconstruction
method. Fetch timestamps and absolute host paths do not affect identity.

## Database evolution

The applied `001_initial.sql` remains byte-for-byte unchanged. A new numbered
migration advances the schema version and adds only structures missing from the
current design, including security-name history and provider capability audit
records. Existing security lifecycle, index membership, artifact, snapshot,
and universe tables are reused. Applied migration SHA256 values remain
immutable.

Catalog synchronization is idempotent and protected by the existing
cross-container atomic directory lock. External Parquet readers expose typed
research tables without copying daily facts into DuckDB.

## Commands

Stable Phase 5 entry points are:

```text
make research-data-probe
make research-data-bootstrap
make research-data-update END_DATE=<YYYY-MM-DD>
make research-data-validate SNAPSHOT=<snapshot_id>
make universe-asof DATE=<YYYY-MM-DD> SNAPSHOT=<snapshot_id>
```

`probe` performs small capability calls and reports access without building a
snapshot. `bootstrap` builds 2020-01-01 through the configured end date.
`update` appends missing raw intervals and creates a new immutable snapshot.
`universe-asof` is read-only and prints the historically valid CSI 300 members
and their lifecycle/status provenance.

## Failure handling and recovery

Every request is cached independently. Network, permission, schema, row-limit,
and quality failures stop publication but preserve verified raw artifacts for
resume. Temporary normalized output is never promoted after a failed gate.
Fallback requires explicit configuration and equivalent field semantics; it is
recorded per artifact and cannot silently mix incompatible price adjustment or
status conventions inside one canonical dataset.

## Quality gates and acceptance tests

The implementation must include tests proving:

- two sampled dates reconstruct the expected membership using effective and
  known dates;
- a membership announced after a sampled date is invisible on that date;
- a delisted member remains in security master and historical universes but is
  excluded after delisting;
- ST and suspension intervals do not leak backward and missing coverage stays
  nullable;
- adjustment factors are positive, unique, and kept separate from prices;
- membership and lifecycle intervals do not overlap or violate date order;
- repeated provider requests reuse immutable cache without network calls;
- tampered raw sidecars or Parquet hashes stop the build;
- the same inputs create the same snapshot identity and Parquet content hashes;
- database migration and catalog synchronization are idempotent;
- token and configured credentials never appear in tracked files or artifacts.

Formal verification runs only through the Linux Python 3.11 container with
`make lint`, `make test`, and `make smoke`. A bounded live bootstrap is accepted
only after the fixture-driven suite passes; provider permission gaps remain
explicit unresolved capabilities rather than fabricated data.
