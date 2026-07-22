# Phase 6 Task 6 fix 7 report

## Scope

Closed the final Task 6 approval-chain review findings without creating a real
test request or approval and without reading the locked 2026 partition.

## Changes

- `create_test_request()` now calls the production `validate_freeze()` before
  hashing or registering any robustness evidence. This enforces canonical
  freeze identity, current dependency hashes, the pre-test capability, and its
  administrative DuckDB anchor.
- Robustness evidence now has a complete strict contract:
  - exact top-level and nested schemas;
  - exact fold identities and dates;
  - row-count/coverage consistency;
  - exact backtest metric and constraint fields;
  - per-fold-to-aggregate cost-return derivation;
  - exposure field shapes, size-risk derivation, and industry-retention
    derivation;
  - common freeze, policy, dependency, orientation, and test-access fields;
  - deterministic Markdown regenerated from the bound JSON reports.
- Every numeric value is required to be finite. Gate evaluation now treats
  NaN and both infinities as failures, including the 2x-cost scenario.
  Canonical JSON writers use `allow_nan=False`.
- The execution bundle now explicitly pins and validates
  `data/normalize.py`, `database/catalog.py`, all three migration SQL resources,
  and the Task 6 approval/final-test modules in addition to the existing
  evaluation closure.
- The wrong-request administrative-anchor regression now temporarily removes
  the child approval row, mutates the otherwise FK-valid request field, and
  restores the approval in separate transactions. It reaches the read-only
  final-test anchor check instead of failing during fixture setup.
- Positive approval fixtures now use a valid content-derived freeze identity
  and byte-faithful Task 5 report structures. Added invalid identity,
  coherently republished inconsistent evidence, execution dependency drift,
  and NaN/Infinity regressions.

## Formal runtime verification

Executed in the Linux Python 3.11 Docker service:

```text
docker compose run --rm research ruff check \
  src/alpha_lab/robustness/approval.py \
  src/alpha_lab/robustness/report.py \
  src/alpha_lab/robustness/walk_forward.py \
  tests/unit/test_test_approval.py \
  tests/unit/test_walk_forward.py
PASS: All checks passed.

docker compose run --rm research pytest -q \
  tests/unit/test_walk_forward.py \
  tests/unit/test_test_approval.py \
  tests/integration/test_final_test_gate.py
PASS: 54 passed in 4.52s.

git diff --check
PASS.
```

The five pre-existing Task 4 catalog-corruption fixtures that DuckDB blocked at
FK or unique constraints were repaired without weakening production schema
constraints. They now temporarily remove and restore legal referencing rows,
or insert a unique second artifact, so every case reaches the application
anchor validation. The expanded run is green:

```text
docker compose run --rm research pytest -q \
  tests/unit/test_candidate_freeze.py \
  tests/integration/test_robustness_pipeline.py \
  tests/unit/test_walk_forward.py \
  tests/unit/test_test_approval.py \
  tests/integration/test_final_test_gate.py
PASS: 133 passed in 14.75s.
```

## Safety

- No applied migration was edited.
- No Task 7 code was added.
- No real test request, human approval, or final-test artifact was created.
- No locked 2026 data was opened.
