# Phase 6 static-check fix report

## Scope

- Base commit: `46f9e48`
- Static-only fixes in the Phase 6 Task 6 implementation and its tests.
- No provider calls, real approval creation, or 2026 locked-partition access.

## Changes

- Applied Ruff's required import ordering, line wrapping, context-manager
  formatting, and formatting of three Phase 6 tests.
- Annotated `_nonnegative_int` as a `TypeGuard[int]` so mypy can safely narrow
  the validated row-count operands without changing runtime behavior.
- Added a narrow `no-untyped-call` suppression for PyArrow's untyped
  `ParquetFile` constructor; the existing metadata row-count check is unchanged.

## Docker commands and results

- `make lint` (initial): failed with 4 Ruff findings in `final_test.py`.
- `docker compose run --rm research ruff check --fix src/alpha_lab/robustness/final_test.py`:
  fixed import ordering and exposed the remaining three manual-format findings.
- `docker compose run --rm research ruff format ...`: formatted the affected
  implementation and Phase 6 test files.
- `make lint` (intermediate): found one PyArrow `no-untyped-call` and three
  row-count narrowing errors.
- `make lint` (final): passed; Ruff check passed, 112 files formatted, mypy
  reported `Success: no issues found in 68 source files`.
- `docker compose run --rm research pytest -q tests/test_phase6_contract.py tests/unit/test_test_approval.py tests/unit/test_candidate_freeze.py tests/unit/test_database_phase6.py tests/integration/test_final_test_gate.py`:
  `152 passed in 26.90s`.
- `git diff --check`: passed.

One first focused-test invocation referenced a nonexistent
`tests/unit/test_pretest_capability.py`; no tests ran in that invocation. The
correct repository test paths were then used for the successful 152-test run.
