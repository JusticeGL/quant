# A-Share Alpha Lab Agent Rules

## Scope and runtime

- Implement one specification phase at a time. Do not start a later phase unless the user explicitly requests it.
- Run every formal Python command through `make` or `docker compose`; the supported runtime is Linux with Python 3.11.
- On Apple Silicon, use Docker's native ARM64 platform. Diagnose dependency failures before considering any `linux/amd64` fallback, and document any approved fallback.
- Phase 0 must not download market data, run a backtest, or connect to a broker or trading account.

## Data, security, and reproducibility

- Never modify files under `data/raw`; raw data is immutable and append-only.
- Never commit `.env`, tokens, credentials, data files, caches, model artifacts, or large experiment outputs.
- Keep dependency changes in `pyproject.toml` and regenerate the committed `uv.lock` inside the Linux container with `make lock`.
- Keep large facts in immutable Parquet; DuckDB stores catalog and relational metadata, not a duplicate market-data body.
- Never edit an applied database migration. Add a new numbered migration and preserve the recorded SHA256.
- Treat `config/splits.yaml`, evaluation code, cost rules, leakage tests, and data manifests as locked areas. Propose changes for human review instead of editing them during factor mining.

## Research integrity

- A new factor must include metadata and tests with its implementation.
- Future functions, label leakage, negative shifts, future windows, and filling historical universes with current constituents are prohibited.
- Change only one primary variable in each experiment round.
- Do not accept a factor based on a single return curve; use the fixed structured evaluation policy.
- Do not delete failed experiments. Failures are part of the audit trail.

## Completion gate

- Before reporting completion, run `make lint`, `make test`, and `make smoke` in the Linux container and report the actual results.
