# Task 6 fix8 report

## Outcome

- `create_test_request` now replays the complete production pre-test evaluation in
  a temporary experiments root containing only the copied `freeze.json` before
  it publishes a request.
- The replayed `walk_forward.json`, `cost_sensitivity.json`,
  `exposure_report.json`, and `robustness_report.md` must match the submitted
  artifacts byte-for-byte and by SHA256. A wholly replaced, internally
  self-consistent report set is rejected.
- Temporary replay outputs, including the `large/` tree, are removed by the
  temporary-directory boundary and never enter the formal experiment directory.
- The execution bundle is now a closed, sorted inventory of all
  `src/alpha_lab/**/*.py`, packaged YAML metadata, database SQL migrations,
  `pyproject.toml`, `uv.lock`, `Dockerfile`, `compose.yaml`, and the locked Phase
  6 configuration files. Approval and final-test locked reads retain exact
  namespace and hash validation.
- The missing parenthesis in `tests/unit/test_robustness_io.py` was repaired.
  Its administrative fixture now rejects invalid manifest partition declarations
  before any Parquet read, matching the production catalog trust boundary.
- No approval was created, no 2026 data was read, no migration was changed, and
  no Task 7 work was started.

## Verification executed in Linux Docker (Python 3.11)

- Focused replay/approval/I/O/pipeline suite:
  `75 passed in 12.17s`.
- Expanded Phase 6 suite (exposure snapshot, final-test gate, pipeline, freeze,
  database v3, exposure data/metrics, robustness config/I/O/approval and
  walk-forward): `254 passed in 71.82s`.
- Earlier focused checkpoint after the I/O syntax/fixture repair:
  `64 passed in 11.42s`.
- Ruff check: all checks passed.
- Ruff format check: all five changed source/test files formatted.
- `git diff --check`: passed.

The production replay integration invokes the evaluator three times in total,
records only the exclusive `2026-01-01` reader boundary (six reader calls), and
verifies that no `.pretest-replay-*` directory remains afterward.
