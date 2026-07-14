# Phase 6 Robustness and Final Test Design

## Objective and scope

Phase 6 evaluates frozen versions of `F1002` and `F1003` on the research-grade
CSI 300 dataset. It adds point-in-time industry and market-cap exposures,
calendar walk-forward evaluation, cost sensitivity, and a human approval gate
for the locked final test. It does not change factor formulas, Phase 3/4
evaluation policy, the Phase 5 snapshot, or any existing applied migration.

The Phase 5 snapshot `p5-ecaa6e8aeae6b9f8fb25` remains immutable. Phase 6
creates separate exposure, freeze, robustness, approval, and final-test
artifacts. No workflow promotes a factor automatically or connects to a broker.

## Architecture

### Exposure data boundary

`alpha_lab.robustness.exposure_data` owns the Phase 6 exposure dataset. It uses
the existing secret-safe Tushare provider and immutable raw request cache.

- `daily_basic` supplies daily `total_mv` and `circ_mv`. Tushare values are
  normalized from ten-thousand CNY to CNY. Exposure analysis uses
  `log(total_market_cap_cny)` and retains the untransformed values for audit.
- `index_classify` supplies the SW2021 industry dictionary.
- `index_member_all` supplies historical constituent intervals for SW2021
industries. Industry membership requires
`effective_from <= D <= effective_to` and `known_at <= D`.
- The canonical `industry_membership.parquet` retains the complete approved
  history for catalog and future approved-test use. Pre-test code can open only
  `industry_membership_pretest.parquet`, a manifest-bound deterministic view
  containing rows known and effective before 2026-01-01 with interval ends
  clipped to 2025-12-31. Snapshot validation re-derives and byte-checks this
  view from the canonical history.
- The relay must return every required field. Extra fields may be discarded;
  missing fields, truncation, duplicate keys, interval overlaps, unknown
  securities, or inadequate coverage stop publication.
- Current industry values are never used to backfill history.

Task 2 also publishes canonical
`manifests/<p6x-id>/pretest_capability.json`. Its content deliberately contains
no p6x ID or p6x path, so its content-derived identity can be referenced by the
p6x root without a hash cycle. The root identity binds the complete
manifest-relative `{path, sha256, capability_id}` reference. The capability
contains only the fixed `2026-01-01` cutoff, opaque Phase 5 parent ID/manifest
SHA, policy SHA, pre-2026 Phase 5 market partitions, pre-2026 market-cap
partitions, industry definitions, the isolated pre-test membership, and exact
safe-subset quality metadata. It excludes full membership, full quality
reports, raw-cache paths and 2026 counts.

Publication and catalog administration re-derive this capability from the full
roots. Freeze creation, freeze validation and pre-test readers use a pure
capability validator instead: it may parse the p6x root and recompute its
identity as opaque parent metadata, but cannot open either full quality report,
the Phase 5 root, raw cache, a 2026 partition, or full membership. Both readers
are located by the root p6x ID; there is no Phase 5-manifest fallback.

The immutable exposure snapshot ID is `p6x-<identity-prefix>`. Its identity
includes the Phase 5 manifest hash, every raw exposure artifact hash, the
exposure configuration hash, and the schema version. Daily market-cap facts and
industry intervals remain in Parquet. DuckDB stores the snapshot catalog,
industry definitions, interval metadata, and artifact links, not a duplicate of
daily market-cap facts.

The exposure capability probe runs before bulk acquisition. Unsupported relay
capabilities fail explicitly and preserve completed caches.

### Robustness package

`alpha_lab.robustness` is independent from the locked Phase 3 evaluation
package. Its focused modules are:

- `config.py`: strict Phase 6 policy and time-boundary validation.
- `exposure_data.py`: capability probe, normalization, quality, and `p6x-*`
  publication.
- `freeze.py`: immutable candidate-version manifests.
- `walk_forward.py`: fold-local factor metrics, labels, and Top-K backtests.
- `exposures.py`: size and industry exposure metrics and industry-neutral
  scores.
- `approval.py`: test requests, explicit approvals, and hash validation.
- `final_test.py`: locked-test execution and immutable report publication.
- `report.py`: deterministic JSON and Markdown summaries.

F1002 and F1003 are evaluated separately. A freeze pins the candidate source
and metadata SHA256, Phase 5 and exposure snapshot IDs, policy hashes, Git
commit, and declared test range. Any pinned input change invalidates the freeze
and all approvals derived from it.

## Time policy and leakage boundary

The fixed Phase 6 calendar is:

- Warm-up only: 2020-01-01 through 2020-12-31.
- Walk-forward fold 1: 2021-01-01 through 2021-12-31.
- Walk-forward fold 2: 2022-01-01 through 2022-12-31.
- Walk-forward fold 3: 2023-01-01 through 2023-12-31.
- Walk-forward fold 4: 2024-01-01 through 2024-12-31.
- Walk-forward fold 5: 2025-01-01 through 2025-12-31.
- Locked final test: 2026-01-01 through 2026-07-11.

F1002 and F1003 are deterministic rolling factors, so walk-forward means
independent calendar-fold calculation and evaluation, not fictitious model
retraining. Rolling inputs may use observations before a fold starts. Signals,
forward labels, trades, exits, metrics, and reports must remain inside the fold.
The last signal dates without fold-local entry and exit prices are excluded.

Pre-test code filters every market and exposure read to dates before 2026-01-01.
It cannot open, score, aggregate, or report locked-test rows. Tests instrument
the data reader and prove the rejection occurs before a locked Parquet read.

## Robustness evaluation

Each annual fold produces coverage, IC, Rank IC, ICIR, group returns, factor
turnover, direction consistency, and a Top-K backtest using the existing China
A-share cost and execution rules. Results retain the fold date boundaries,
input hashes, and row counts.

Cost sensitivity runs the same signals at `0.5x`, `1.0x`, `1.5x`, and `2.0x`
the standard cost policy. It scales monetary rates and minimum commission
consistently and records all constraints and fees.

Exposure analysis produces:

- daily cross-sectional Spearman correlation between factor score and
  `log(total_market_cap_cny)`;
- annual and aggregate size-correlation distributions;
- industry mean score dispersion and observations by industry;
- scores standardized within point-in-time industries;
- original versus industry-neutral Rank IC and retention ratio.

The locked pre-test gate requires:

- Rank IC direction consistency in at least four of five annual folds;
- coverage of at least 70 percent in every fold;
- no return-direction reversal at `2.0x` costs;
- industry-neutral absolute Rank IC retention of at least 50 percent.

Absolute size correlation above 0.30 is a reported risk flag, not an automatic
failure. Passing the gate permits a test request only; it never accepts or
promotes a factor.

## Freeze, approval, and final-test state machine

The stable workflow is:

```text
candidate -> frozen -> robustness_passed -> test_requested
          -> approved -> test_completed
```

Failure states are immutable reports linked to the last valid state. No command
silently skips a state.

Commands are:

```text
make exposure-probe
make exposure-bootstrap
make robustness-freeze ID=F1002
make robustness-eval FREEZE=<freeze_id>
make test-request FREEZE=<freeze_id>
make test-approve REQUEST=<request_id> APPROVER=<name> CONFIRM=<freeze_hash>
make final-test APPROVAL=<approval_id>
```

F1002 and F1003 have separate freezes, requests, approvals, and test results.
After pre-test evaluation, automation stops. The user must explicitly approve a
specific request and exact freeze hash. Approval of this design is not approval
to access either candidate's final test.

The final-test loader validates the freeze, candidate hashes, data snapshot
hashes, policy hashes, request, approval, and explicit test range before opening
test data. A changed input makes the approval invalid. Repeating the same test
is idempotent; a different result for an existing test run ID is an error.

## Artifacts and immutability

Small auditable artifacts live under:

```text
experiments/phase6/<freeze_id>/
  freeze.json
  walk_forward.json
  cost_sensitivity.json
  exposure_report.json
  robustness_report.md
  test_request.json
  approvals/<approval_id>.json
  final/<test_run_id>/result.json
  final/<test_run_id>/report.md
```

Large daily values, predictions, NAV series, and trades are immutable Parquet
artifacts under the ignored data/experiment storage boundary. Small JSON and
Markdown audit records may be committed. Publication uses temporary paths and
atomic rename. Existing bytes are compared before idempotent reuse and are
never overwritten with different content.

## Error handling and recovery

- Provider errors redact credentials and retain successful raw caches.
- Truncation, missing fields, quality errors, or PIT inconsistencies prevent an
  exposure snapshot from being published.
- A failed robustness fold records no successful state transition and can be
  retried from immutable inputs.
- Missing, stale, mismatched, or malformed approvals fail before test reads.
- A failed final test keeps its failure audit record and never replaces an
  earlier successful report.
- No workflow edits `config/splits.yaml`, `config/costs.yaml`,
  `config/factor_evaluation.yaml`, Phase 3 evaluation code, leakage tests, Phase
  5 manifests, or raw data.

## Testing and acceptance

Tests cover:

- daily market-cap units, SW2021 interval normalization, known-at semantics,
  duplicate/overlap rejection, and deterministic exposure snapshot identity;
- relay capability gaps and cache reuse;
- labels and trades that cannot cross fold boundaries;
- cost multipliers and deterministic fold metrics;
- point-in-time size/industry joins and industry-neutral Rank IC;
- rejection of unfrozen, changed, unapproved, or incorrectly approved
  candidates before any test read;
- immutable and idempotent robustness and final reports;
- F1002 and F1003 pre-test execution from frozen versions;
- native ARM64 Linux Python 3.11 `make lint`, `make test`, and `make smoke`.

Phase 6 acceptance requires a valid `p6x-*` exposure snapshot, immutable freezes
and complete pre-test reports for F1002 and F1003, and a demonstrated approval
gate. Final-test reports are required only for candidates the user explicitly
approves after reviewing pre-test results.
