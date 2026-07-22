# Phase 6 Tushare relay shape fix report

## Outcome

- Kept the primary `index_member_all(l1_code=...)` acquisition for every SW2021
  L1 definition.
- An explicitly empty L1 response is now retained as an auditable information
  quality check instead of being converted into a fake membership.
- For Phase 5 historical securities not covered by the L1 responses, acquisition
  queries the same `index_member_all` endpoint once per `ts_code`.
- Backfill rows are recovered only through a separately acquired SW2021 L2
  `index_classify` hierarchy: `l2 industry_code -> parent_code -> L1
  industry_code -> L1 source_index_code`.
- Missing, duplicate, ambiguous, conflicting, empty-security, invalid-date, and
  overlapping results fail closed. No industry is guessed or hard-coded.
- Output membership is limited to the Phase 5 historical security set and must
  cover that entire expected set before publication.
- The L2 classification request and all per-security backfill requests remain
  immutable Tushare raw artifacts and are included in snapshot identity inputs.
- No provider switch, migration change, backtest, approval, or final-test access
  was introduced.

## TDD and verification

The initial focused test run failed at collection because the old implementation
had no L2 hierarchy contract. Tests now cover empty L1 responses, successful
per-security recovery, L2 parent conflicts, unknown parents, exact duplicate L2
rows, empty backfill responses, and complete expected-security coverage.

Executed in the Linux Python 3.11 ARM64 container:

- `docker compose run --rm --build research pytest -q tests/unit/test_exposure_data.py`
  - `27 passed`
- Related exposure snapshot and DuckDB regression suite
  - `86 passed`
- `make lint`
  - Ruff checks passed; 112 files formatted; mypy passed for 68 source files.
- `make test`
  - `380 passed in 37.64s`
- `make smoke`
  - Linux aarch64, Python 3.11.15; qlib 0.9.7, akshare 1.18.64,
    duckdb 1.5.4, pyarrow 24.0.0, lightgbm 4.6.0 imported successfully.
- `git diff --check`
  - clean.

## Real cache/bootstrap result

The real run used the existing main-repository data mount and configured relay:

`make exposure-bootstrap COMPOSE='docker compose --env-file /Users/tanwentao/Desktop/project/quant/.env -f compose.yaml -f /private/tmp/phase6-data-mount.yaml'`

It retained all existing cache files and added the L2 classification artifact plus
the missing per-security `index_member_all(ts_code=...)` raw artifacts. The run
correctly stopped without publishing an exposure snapshot because the relay
returned legitimate empty responses for these ten Phase 5 historical securities:

- `000413.SZ`
- `000627.SZ`
- `000671.SZ`
- `000961.SZ`
- `002411.SZ`
- `600068.SH`
- `600297.SH`
- `600705.SH`
- `600837.SH`
- `601989.SH`

The ten securities represent 5,071 of 473,103 Phase 5 observations (1.0719%).
`801230.SI` is also an explicitly empty L1 definition response and is recorded as
such. Since the task requires complete historical-security coverage, accepting an
incomplete snapshot would violate the contract. Resolving this remaining data gap
requires an explicitly approved additional auditable source/endpoint or an
architectural change to the expected universe; this implementation does neither.
