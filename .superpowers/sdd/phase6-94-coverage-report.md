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

## Real-cache bootstrap result

After policy commit `1378735`, `make exposure-bootstrap` was run with
`/private/tmp/quant-phase6-data.yaml` and the repository-local environment file.
The builder wrote immutable snapshot `p6x-123df2b10b84b5020b29`. Its stored
quality report is a warning, not an error, and preserves the complete result:

- expected observations: 473,103;
- matched observations: 445,199;
- missing observations: 27,904;
- coverage: 94.1019186097%;
- missing-security count: 74, with all 74 sorted identifiers retained in the
  quality report;
- temporal coverage: 100%;
- insufficient-industry-coverage error count: zero;
- warning counts: 27,904 missing observations and 10 securities with no
  membership interval.

The command then stopped during the independent catalog validation. A direct
container diagnostic reproduced the sanitized failure set
`["quality_row_counts", "quality_recomputed"]`. Consequently the snapshot was
not synchronized into DuckDB and must not yet be treated as a trusted pre-test
capability. This is a new post-publication quality-recomputation blocker, not a
failure of the approved 94% floor.

Per the fail-closed rule, no idempotent replay, candidate freeze, robustness
evaluation, approval, or final-test access was attempted. The immutable raw
cache and the written snapshot were left untouched for diagnosis.

## Catalog validation resolution

Commit `0732791` fixed two independent validation assumptions without changing
the published snapshot or a migration:

- the recomputation now takes the authenticated `expected_industry_ids` from
  the physical membership Parquet metadata, instead of treating every industry
  definition (including explicit empty responses) as an expected membership;
- `observed_observation_count` is the intersection with expected Phase 5
  observations and may therefore be smaller than the complete physical market
  cap body. It may never exceed that body, and the subsequent independent
  physical recomputation still requires every stored count, ratio, severity,
  status, empty-response count and taxonomy-bridge count to match exactly.

TDD reproduced both failures. The new integration case catalogs a warning
snapshot while preserving warning severity/status, an explicit-empty count,
and a market-cap body larger than the expected-observation intersection.
Tampered observed counts remain rejected as `quality_recomputed`.

Verification after the fix:

- focused snapshot/quality/catalog suite: 95 passed;
- `make lint`: passed;
- `make test`: 391 passed;
- `make smoke`: Linux aarch64 and Python 3.11.15; all five required imports
  passed with the versions recorded above;
- direct validation of the immutable real snapshot: healthy, zero failures,
  warning status, 722 checked files, unchanged manifest SHA256
  `7ef6377858c1eb859147e09feaf32b834e8a3343dbbae44e73bb49654b41183e`.

Two normal real-cache `exposure-bootstrap` replays both returned the same
snapshot ID and manifest SHA256 with warning status and synchronized DuckDB
successfully. A separate instrumented cache replay counted 711 queries, 711
cache hits and exactly zero network requests, again producing the same identity.
The catalog contains exactly one snapshot row, one quality row and one linked
pre-test capability for this identity; `latest_exposure_snapshot_id` points to
it. No freeze, evaluation, approval, or final test was run.
