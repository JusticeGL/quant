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

The real cached exposure bootstrap and idempotent second run are performed
after this implementation commit; their snapshot identity and actual counts
will be appended only if publication succeeds.
