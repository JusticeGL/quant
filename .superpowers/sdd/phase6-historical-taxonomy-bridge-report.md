# Phase 6 historical taxonomy bridge report

Date: 2026-07-14

## Scope

Resolve historical non-empty Tushare memberships that use pre-SW2021 codes,
without a new provider, name matching, security exceptions, or relaxed quality
gates. This work does not authorize freeze, approval, or final-test access.

## Implementation

- Acquires immutable SW2014 L1/L2/L3 `index_classify` responses only when a
  non-empty backfill cannot use the SW2021 L2 hierarchy.
- Validates the code chain from membership L3 index code and/or L2 industry
  code through SW2014 parents to an L1 index code that must uniquely exist in
  the SW2021 L1 dictionary.
- Requires L3 and L2 paths to agree when both are present. Missing, ambiguous,
  conflicting, or unstable links fail closed.
- Records mapping path, taxonomy source version, taxonomy target version, and
  the number of bridged membership intervals in quality evidence.
- Keeps every existing observation coverage, duplicate, overlap,
  unknown-reference, and market-cap gate unchanged.

## Verification

- Focused Docker suite covering the real `230100` / `850412.SI` chain,
  conflicting paths, unstable L1 targets, acquisition, snapshot and freeze:
  `133 passed`.

- `make test`: `390 passed in 72.14s`.
- `make lint`: Ruff and formatting passed; mypy passed for 68 source files.
- `make smoke`: Linux aarch64, Python 3.11.15; qlib 0.9.7, akshare
  1.18.64, duckdb 1.5.4, pyarrow 24.0.0 and lightgbm 4.6.0 imported.

Real-cache bootstrap and its idempotent replay are recorded after the
implementation commit.
