# Phase 6 warning pre-test capability fix report

Date: 2026-07-15

## Scope

- Fixed the pre-test catalog anchor to accept an exposure manifest quality status
  of `pass` or `warning` and require the DuckDB snapshot row to match that exact
  status.
- Kept the capability's own safe-artifact integrity status as `pass`; an
  exposure-level coverage warning does not weaken capability hash, namespace,
  row-count, or catalog checks.
- Rejected `error`, unknown, non-string, and missing manifest statuses before
  opening any safe Parquet artifact.
- Added both `pass`/`warning` catalog mismatch directions to the fail-closed
  coverage.
- Did not modify data, migrations, coverage policy, freeze state, or evaluation
  outputs.

## TDD evidence

The new real-warning integration assertion failed before the implementation at
the hard-coded `pass` catalog tuple with:

```text
ValueError: pre-test capability catalog snapshot anchor mismatch
```

After the fix, the focused Docker selection completed with:

```text
29 passed in 5.50s
```

The focused selection covered the real warning snapshot, the existing pass
snapshot regression, invalid manifest statuses, both catalog status mismatch
directions, and all existing catalog-anchor corruptions.

## Completion gates

- `make lint`: passed; Ruff checks and formatting passed, mypy reported no
  issues in 68 source files.
- `make test`: passed; `395 passed in 31.97s`.
- `make smoke`: passed on Linux aarch64 with Python 3.11.15; qlib 0.9.7,
  akshare 1.18.64, DuckDB 1.5.4, PyArrow 24.0.0, and LightGBM 4.6.0 imported.

## Main-data read-only validation

The main data directory was mounted read-only into the Linux container.
`validate_pretest_capability` succeeded without freezing or evaluating:

```json
{"artifact_count": 26, "capability_id": "pretest-4d59c5d86c5c22f5f044", "quality_status": "warning", "snapshot_id": "p6x-123df2b10b84b5020b29"}
```

## Unresolved issues

None in this fix scope. Freeze and robustness evaluation remain intentionally
unexecuted for the parent Phase 6 workflow.
