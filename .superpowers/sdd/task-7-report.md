# Task 7 implementation report

## Scope delivered

- Added the seven stable Phase 6 Typer commands: `exposure-probe`,
  `exposure-bootstrap`, `robustness-freeze`, `robustness-eval`, `test-request`,
  `test-approve`, and `final-test`.
- Exposure bootstrap catalogs the validated snapshot through
  `sync_exposure_snapshot` before reporting success.
- Request, approval, and final-test commands call the production Task 6 state
  machine. IDs are strict 64-hex identities and resolve to exactly one canonical
  artifact.
- Phase 6 failures emit credential-safe structured JSON containing operation,
  status, and exception type, but not provider exception text.
- Added Make targets. Required variables fail with exit 2 before Docker/Python:
  `ID`, `FREEZE`, `REQUEST`, `APPROVER`, `CONFIRM`, and `APPROVAL`.
- Added the narrow Phase 6 JSON/Markdown Git allowlist. Parquet, unexpected JSON,
  data, DuckDB, credentials, caches, and large outputs remain ignored.
- Added README workflow entry and the full immutable calendar, provider recovery,
  warning semantics, approval pause, research-only notice, and cooperative threat
  model in `docs/phase6_robustness.md`.
- No live provider command, real freeze/evaluation, test request, approval, or
  2026 final test was executed.

## TDD and verification evidence

1. RED:
   `docker compose run --rm research pytest -q tests/test_phase6_contract.py tests/unit/test_cli.py`
   produced 8 failures because all Task 7 interfaces were absent.
2. GREEN after implementation: the same focused suite passed 8 tests; after the
   strict artifact-ID traversal test was added, the final focused suite passed
   9 tests.
3. Full Phase 6 regression:
   `docker compose run --rm research pytest -q tests/test_phase6_contract.py tests/unit/test_cli.py tests/unit/test_database_phase6.py tests/unit/test_exposure_data.py tests/unit/test_factor_exposures.py tests/unit/test_robustness_config.py tests/unit/test_robustness_io.py tests/unit/test_test_approval.py tests/integration/test_exposure_snapshot.py tests/integration/test_final_test_gate.py tests/integration/test_robustness_pipeline.py`
   passed: **188 passed in 17.92s**.
4. Task 7 Ruff and formatting:
   `ruff check` and `ruff format --check` passed for `src/alpha_lab/cli.py`,
   `tests/unit/test_cli.py`, and `tests/test_phase6_contract.py`.
5. Actual Make guard: `make robustness-freeze` without `ID` exited 2 with
   `ID is required` before Docker/Python.
6. Actual `git check-ignore --no-index -q` checks showed selected freeze,
   approval, and final-result paths are trackable (exit 1), while
   `unexpected.json` and `large/predictions.parquet` remain ignored (exit 0).
7. `git diff --check` passed.

## Existing whole-tree gate failures

`make lint` does not currently pass because files delivered before Task 7 have
outstanding static issues outside this task's allowed file scope:

- `src/alpha_lab/robustness/final_test.py`: Ruff import ordering, two E501 lines,
  and one SIM117 finding.
- `src/alpha_lab/robustness/pretest_capability.py`: one mypy `no-untyped-call`.
- `src/alpha_lab/robustness/approval.py`: three mypy optional-operand findings.

Task 7's own files pass Ruff/format. These earlier issues are reported rather
than silently mixed into the Task 7 commit.
