# Phase 6 robustness and final-test gate

Phase 6 is a research-only workflow for F1002 and F1003. It adds point-in-time
size and SW2021 industry exposures, fixed pre-test robustness checks, and an
explicit human approval boundary around the locked test. It never connects to
a broker or trading account, and its reports are not investment advice.

## Immutable flow and calendar

The state machine is:

```text
Phase 5 snapshot -> p6x exposure snapshot -> candidate freeze
-> robustness reports -> test request -> human approval -> final result
```

Every transition pins SHA256 identities. Existing artifacts may be reused only
when their bytes are identical; changed inputs require a new identity. Large
Parquet predictions, NAV, trades, raw provider cache, DuckDB, credentials, and
model outputs remain ignored. Git may track only the named compact JSON and
Markdown audit records.

The exact locked calendar is:

- warm-up: 2020-01-01 through 2020-12-31;
- five walk-forward folds: 2021-01-01 through 2025-12-31, one calendar year per
  fold;
- locked test: 2026-01-01 through 2026-07-11.

Pre-test readers reject a requested range reaching 2026-01-01 before opening a
Parquet file. A warning is a disclosed quality or exposure risk that does not
by itself fail a fixed gate; it is never silently converted to a pass. Gate
failures prevent test-request creation.

## Commands and mandatory stop

All formal Python execution uses Linux Python 3.11 through Make and Docker.
Apple Silicon uses Docker's native ARM64 platform.

```bash
make exposure-probe
make exposure-bootstrap
make robustness-freeze ID=F1002
make robustness-eval FREEZE=freeze-<sha256>
make test-request FREEZE=freeze-<sha256>
```

Automation stops here. A human must inspect `walk_forward.json`,
`cost_sensitivity.json`, `exposure_report.json`, `robustness_report.md`, the
freeze hash, and the request identity. Only that human may issue:

```bash
make test-approve REQUEST=request-<sha256> APPROVER=<human> CONFIRM=<freeze-sha256>
make final-test APPROVAL=approval-<sha256>
```

`ID`, `FREEZE`, `REQUEST`, `APPROVER`, `CONFIRM`, and `APPROVAL` are checked by
Make before Python starts. F1002 and F1003 require separate freezes, requests,
approvals, and results. Approval of a plan, provider, cost rule, commit, or
earlier phase does not grant final-test access.

## Provider failure and recovery

`make exposure-probe` performs bounded capability checks. Bootstrap reuses
successful immutable raw provider cache entries. If the provider is unavailable,
rate-limited, truncated, or returns a wrong schema, keep the cache, diagnose the
reported exception type and upstream status, then retry the same command. Do
not delete raw files, lower quality thresholds, change the calendar, or switch
architecture. CLI errors are credential-safe structured JSON and intentionally
do not repeat provider exception messages. Tokens stay only in local `.env`.

An exposure snapshot is published only after schema, row limit, point-in-time,
coverage, checksum, and quality validation. `exposure-bootstrap` then syncs the
validated manifest into `data/metadata.duckdb`; a freeze cannot trust an
uncataloged capability.

Historical industry coverage has a user-approved minimum of 98%. It is
calculated over every expected Phase 5 `(trade_date, security_id)` observation
using membership intervals valid and known on that date. Missing industry
securities and observations are reported as deterministic warnings at or above
the threshold; below it publication fails closed. Such observations still take
part in factor and cost evaluation, but are explicitly excluded from the
industry-neutral calculation. Empty L1 responses are disclosed separately as
information; duplicate, overlap, unknown-reference, and market-cap checks are
not relaxed.

Some historical memberships use codes that predate SW2021. When a non-empty
row cannot be resolved by the SW2021 L2 hierarchy, bootstrap obtains SW2014
L1/L2/L3 dictionaries from the same immutable Tushare cache and follows only
their code hierarchy. An L3 and L2 path must agree, and the resulting SW2014 L1
index code must uniquely exist in the SW2021 L1 dictionary. No industry-name
matching or hand-written security/code exception is allowed. The Parquet row
records source/target taxonomy and mapping provenance; quality reports include
the bridge count. Any missing, ambiguous, unstable or conflicting link still
stops publication.

## Evidence and warning semantics

Robustness evaluation computes the candidate once and uses the fixed five folds.
It reports 0.5x, 1.0x, 1.5x, and 2.0x costs, direction consistency, at least 70%
fold coverage, industry-neutral IC retention, and size correlation risk. A size
correlation over the configured threshold is a warning/risk disclosure, not a
hidden gate rewrite. No result is accepted from a single return curve.

The compact allowlist is limited to `freeze.json`, the three JSON robustness
reports, `robustness_report.md`, `test_request.json`, approval JSON, and final
`result.json`/`report.md`. Failed experiments and failed final-test audit records
are retained; they are never overwritten.

## Security model

The schema-v3 DuckDB catalog is the administrative trust anchor for the exact
exposure snapshot, capability, freeze, request, approval, and final run. The
final loader validates the complete hash-linked chain before opening locked
data. This is a cooperative threat model: it protects against accidental drift,
stale/mismatched artifacts, and ordinary workflow bypass. A same-permission
actor that deliberately rewrites or replaces DuckDB is outside the cooperative
threat model; resisting that actor requires external signatures or independently
enforced read-only storage.
