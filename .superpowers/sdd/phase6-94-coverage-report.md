# Phase 6 — 94% historical industry coverage decision

## Decision and scope

On 2026-07-15 the user explicitly selected option 2: lower only the minimum
point-in-time industry-observation coverage from 98% to 94% and continue. The
decision followed a real-cache result of 445,199 matches out of 473,103 Phase 5
observations (94.1019186097%), leaving 27,904 unmatched observations across 74
securities. Ten securities had no membership interval; the remaining gaps were
at the beginning of otherwise populated histories.

This is a bounded acceptance of diagnostic coverage risk. Missing industry
observations remain deterministic warnings with complete expected, matched,
missing, ratio, security-count and sorted missing-security-ID fields in the
quality report. A ratio below 94% remains an error and blocks publication.
Duplicate intervals, overlaps, unknown references, market-cap quality, fold
coverage, direction consistency, cost scenarios, neutral-IC retention and size
risk gates are unchanged.

Unmatched observations remain in the primary factor, cost and backtest inputs.
Only industry-neutral diagnostics exclude unmatched observations. No current
industry value is backfilled into history.

## Verification

- TDD RED: policy test failed because the committed configuration still read
  `0.98` (1 failed, 43 passed).
- Focused GREEN: 44 passed, including exact 94% warning behavior and 93%
  fail-closed behavior.
- `make lint`: Ruff, format check and mypy passed.
- `make test`: 390 passed.
- `make smoke`: Linux aarch64, Python 3.11.15; qlib 0.9.7, akshare 1.18.64,
  DuckDB 1.5.4, PyArrow 24.0.0 and LightGBM 4.6.0 imported.

Real-cache publication and its idempotent replay are recorded below after the
policy commit.
