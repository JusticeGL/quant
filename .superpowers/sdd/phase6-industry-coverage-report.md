# Phase 6 industry observation coverage report

Date: 2026-07-14

## Decision

The user explicitly approved a minimum 98% historical industry-observation
coverage threshold with a disclosed warning for unmatched history. This does
not authorize the locked final test.

## Implementation

- Added `minimum_industry_observation_coverage: 0.98` to the strict Phase 6
  configuration; values outside `(0, 1]` are rejected.
- Coverage is computed from unique Phase 5 expected
  `(trade_date, security_id)` observations and membership intervals satisfying
  both effective dates and `known_at` on each date.
- Quality reports include expected, matched and missing observation counts,
  coverage ratio, deterministic missing-security IDs/count, and explicit empty
  L1 information. Missing industry data is a warning at or above 98%; below
  98% is an error. Existing duplicate, overlap, unknown-reference, market-cap
  and temporal-coverage gates remain strict.
- Empty per-security backfill responses are retained as explicit missing data;
  every non-empty backfill still requires an audited L2-to-L1 mapping.
- Exposure reports disclose industry input, matched, excluded rows and coverage.
  Industry-neutral IC uses matched rows only; factor and cost evaluation retain
  all otherwise valid rows. Test-request replay validates the expanded schema.

## Verification

- Focused Docker tests: `40 passed` before the full regression run.
- Snapshot/freeze/request focused Docker tests: `168 passed`.
- `make test`: `386 passed in 75.75s`.
- `make lint`: Ruff checks passed, 112 files formatted, mypy passed for 68
  source files.
- `make smoke`: Linux aarch64, Python 3.11.15; qlib 0.9.7, akshare 1.18.64,
  duckdb 1.5.4, pyarrow 24.0.0, lightgbm 4.6.0 imported successfully.
- `git diff --check`: passed.

## Real cached bootstrap

After commit `6510c47`, the required bootstrap was run with
`/private/tmp/quant-phase6-data.yaml` and `../../.env`. It stopped before
publication, so no `p6x-*` snapshot or manifest hash was produced and an
idempotent second publication run was not applicable.

The fail-closed cause was not the approved 98% threshold: one non-empty
per-security backfill (`000708.SZ`) contains L2 industry code `230100`, while
the cached SW2021 L2 hierarchy has no audited parent mapping for that code.
The implementation intentionally did not infer an L1, discard the row, relax
hierarchy validation, delete raw cache, or continue to freeze. The raw cache is
preserved. Resolving this requires an explicit data-source/historical taxonomy
decision before bootstrap can be retried.
