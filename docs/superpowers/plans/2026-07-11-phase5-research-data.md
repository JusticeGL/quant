# Phase 5 Research-Grade A-Share Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a point-in-time CSI 300 research dataset from 2020-01-01 with immutable Tushare caches, historical lifecycle/status/membership, separate adjustment factors, reproducible snapshots, and DuckDB catalog integration.

**Architecture:** A new `alpha_lab.research_data` package owns Phase 5 configuration, transport, normalization, quality, snapshot materialization, and orchestration. It reuses the repository's immutable Parquet and DuckDB catalog patterns while keeping Phase 1 snapshots untouched. A data-only Compose service and CLI provide a separately deployable boundary.

**Tech Stack:** Linux Python 3.11, pandas, PyArrow/Parquet, requests, Pydantic, DuckDB, Typer, Docker Compose, uv, pytest.

## Global Constraints

- Formal Python commands run only through `make` or `docker compose` in Linux Python 3.11.
- Apple Silicon remains native ARM64; no `linux/amd64` fallback.
- Never modify `data/raw`, applied migration `001_initial.sql`, locked evaluation code, leakage tests, split rules, cost rules, or existing manifests.
- Never commit or log `TUSHARE_TOKEN`, `.env`, raw data, snapshots, DuckDB files, caches, or large artifacts.
- Phase 5 scope is CSI 300 (`000300.SH`) from `2020-01-01` through the configured end date, not full-market daily history.
- Daily prices stay unadjusted; adjustment factors are separate immutable facts.
- Missing status coverage remains nullable and is never interpreted as false.
- Every implementation behavior follows a failing-test → minimal implementation → passing-test cycle.

---

### Task 1: Phase 5 configuration and credential boundary

**Files:**
- Create: `config/research_data.yaml`
- Create: `src/alpha_lab/research_data/__init__.py`
- Create: `src/alpha_lab/research_data/config.py`
- Modify: `.env.example`
- Modify: `compose.yaml`
- Test: `tests/unit/test_research_data_config.py`

**Interfaces:**
- Produces: `ResearchDataConfig`, `TushareSourceConfig`, `load_research_data_config(config_dir: Path) -> ResearchDataConfig`.
- Produces environment contract: `TUSHARE_TOKEN` required only by live provider commands; `TUSHARE_HTTP_URL` defaults to `https://api.tushare.pro`.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_repository_research_data_config_is_bounded() -> None:
    config = load_research_data_config(ROOT / "config")
    assert config.index_code == "000300.SH"
    assert config.start_date == date(2020, 1, 1)
    assert config.source.provider == "tushare"


def test_config_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="end_date"):
        ResearchDataConfig.model_validate(
            {
                "schema_version": 1,
                "dataset_id": "csi300_point_in_time",
                "index_code": "000300.SH",
                "start_date": "2021-01-02",
                "end_date": "2021-01-01",
                "membership_method": "interval_or_weight_observation",
                "source": {
                    "provider": "tushare",
                    "request_timeout_seconds": 30,
                    "max_attempts": 3,
                    "retry_delay_seconds": 2,
                    "request_interval_seconds": 0.2,
                },
            }
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `docker compose run --rm research pytest -q tests/unit/test_research_data_config.py`

Expected: import failure because `alpha_lab.research_data.config` does not exist.

- [ ] **Step 3: Implement strict Pydantic configuration**

```python
class TushareSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["tushare"]
    request_timeout_seconds: float = Field(gt=0, le=120)
    max_attempts: int = Field(ge=1, le=10)
    retry_delay_seconds: float = Field(ge=0)
    request_interval_seconds: float = Field(ge=0)


class ResearchDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    dataset_id: str
    index_code: Literal["000300.SH"]
    start_date: date
    end_date: date
    source: TushareSourceConfig
```

Validate date order, known endpoint names, and membership reconstruction mode. Add empty environment names to `.env.example` and pass both variables into the `research` and `data` Compose services without embedding values.

- [ ] **Step 4: Run configuration tests and verify GREEN**

Run: `docker compose run --rm research pytest -q tests/unit/test_research_data_config.py`

Expected: all tests pass.

### Task 2: Secret-safe immutable Tushare REST cache

**Files:**
- Create: `src/alpha_lab/research_data/provider.py`
- Test: `tests/unit/test_tushare_provider.py`

**Interfaces:**
- Produces: `TushareArtifact`, `TushareQueryResult`, `TushareProvider.query(api_name: str, params: Mapping[str, object], fields: tuple[str, ...]) -> TushareQueryResult`.
- Consumes: `TushareSourceConfig`, environment token and HTTP URL.

- [ ] **Step 1: Write failing cache, redaction, and error tests**

```python
def test_query_caches_response_without_secret(tmp_path: Path) -> None:
    transport = FakeTransport(fields=["ts_code"], items=[["600000.SH"]])
    provider = TushareProvider(tmp_path, token="secret", http_url="https://example.test", transport=transport)
    first = provider.query("stock_basic", {"list_status": "L"}, ("ts_code",))
    second = provider.query("stock_basic", {"list_status": "L"}, ("ts_code",))
    assert transport.calls == 1
    assert first.artifact.sha256 == second.artifact.sha256
    assert "secret" not in first.artifact.metadata_path.read_text()


def test_nonzero_provider_code_is_not_empty_data(tmp_path: Path) -> None:
    with pytest.raises(TushareProviderError, match="permission"):
        provider.query("suspend_d", {}, ("ts_code",))
```

Also test incomplete pairs, checksum tampering, canonical request identity, retries, and returned-field mismatch.

- [ ] **Step 2: Run provider tests and verify RED**

Run: `docker compose run --rm research pytest -q tests/unit/test_tushare_provider.py`

Expected: import failure for the missing provider.

- [ ] **Step 3: Implement the minimal provider**

```python
payload = {
    "api_name": api_name,
    "token": self.token,
    "params": dict(params),
    "fields": ",".join(fields),
}
response = self.transport(self.http_url, payload, self.timeout)
```

Canonical cache identity includes provider identity, API name, non-secret params, and fields. Write Parquet and JSON via temporary files followed by `os.replace`; sidecars contain no request headers, token, or full payload. Verify both files and SHA256 before cache reuse.

- [ ] **Step 4: Run provider tests and verify GREEN**

Run: `docker compose run --rm research pytest -q tests/unit/test_tushare_provider.py`

Expected: all tests pass.

### Task 3: Point-in-time normalization and universe reconstruction

**Files:**
- Create: `src/alpha_lab/research_data/contracts.py`
- Create: `src/alpha_lab/research_data/normalize.py`
- Create: `src/alpha_lab/research_data/universe.py`
- Test: `tests/unit/test_research_data_normalize.py`
- Test: `tests/unit/test_universe_asof.py`

**Interfaces:**
- Produces normalization functions for security master, name history, index membership/weights, calendar, daily bars, adjustment factors, and suspensions.
- Produces: `universe_as_of(securities: DataFrame, membership: DataFrame, as_of: date) -> DataFrame`.

- [ ] **Step 1: Write failing point-in-time tests**

```python
def test_membership_announced_later_is_invisible() -> None:
    result = universe_as_of(securities, membership, date(2021, 1, 4))
    assert "CN:SSE:600001" not in set(result["security_id"])


def test_delisted_security_remains_in_historical_universe() -> None:
    before = universe_as_of(securities, membership, date(2021, 6, 1))
    after = universe_as_of(securities, membership, date(2023, 6, 1))
    assert "CN:SSE:600002" in set(before["security_id"])
    assert "CN:SSE:600002" not in set(after["security_id"])
```

Add tests for ST prefixes and intervals, suspension `known_at`, positive adjustment factors, no current-name backfill, duplicate keys, and exchange normalization.

- [ ] **Step 2: Run normalization tests and verify RED**

Run: `docker compose run --rm research pytest -q tests/unit/test_research_data_normalize.py tests/unit/test_universe_asof.py`

Expected: missing module imports.

- [ ] **Step 3: Implement strict canonical frames**

```python
mask = (
    (membership["effective_from"] <= timestamp)
    & (membership["effective_to"].isna() | (membership["effective_to"] >= timestamp))
    & (membership["known_at"] <= timestamp)
)
```

Normalize dates with `errors="raise"`, preserve nullable booleans, reject interval overlaps, derive `is_st` only from effective historical names, and use `effective_date_fallback` when no announcement field exists.

- [ ] **Step 4: Run normalization tests and verify GREEN**

Run: `docker compose run --rm research pytest -q tests/unit/test_research_data_normalize.py tests/unit/test_universe_asof.py`

Expected: all tests pass.

### Task 4: Quality gates and immutable Phase 5 snapshot

**Files:**
- Create: `src/alpha_lab/research_data/quality.py`
- Create: `src/alpha_lab/research_data/snapshot.py`
- Test: `tests/unit/test_research_data_quality.py`
- Test: `tests/integration/test_research_snapshot.py`

**Interfaces:**
- Produces: `build_research_quality_report(tables: ResearchTables, config: ResearchDataConfig) -> dict[str, object]`.
- Produces: `materialize_research_snapshot(data_root: Path, config: ResearchDataConfig, tables: ResearchTables, raw_inputs: Sequence[TushareArtifact]) -> ResearchSnapshotResult`.

- [ ] **Step 1: Write failing quality and reproducibility tests**

```python
def test_snapshot_identity_is_stable(tmp_path: Path) -> None:
    first = materialize_research_snapshot(tmp_path, config, tables, raw_inputs)
    second = materialize_research_snapshot(tmp_path, config, tables, raw_inputs)
    assert first.snapshot_id == second.snapshot_id
    assert first.manifest_sha256 == second.manifest_sha256


def test_overlapping_membership_is_error() -> None:
    report = build_research_quality_report(overlapping_tables, config)
    assert report["status"] == "error"
```

Test delisted retention, nullable coverage, positive factors, membership/lifecycle consistency, artifact hashes, deterministic partitions, and atomic publication.

- [ ] **Step 2: Run snapshot tests and verify RED**

Run: `docker compose run --rm research pytest -q tests/unit/test_research_data_quality.py tests/integration/test_research_snapshot.py`

Expected: missing quality and snapshot modules.

- [ ] **Step 3: Implement quality and snapshot publication**

Write stable sorted Parquet with PyArrow Zstandard compression, one deterministic file per table/year, canonical JSON manifests, and `p5-<identity-prefix>` IDs. Refuse to publish quality status `error`; update only `data/state/latest_research_snapshot.txt` after all immutable files and manifest hashes verify.

- [ ] **Step 4: Run snapshot tests and verify GREEN**

Run: `docker compose run --rm research pytest -q tests/unit/test_research_data_quality.py tests/integration/test_research_snapshot.py`

Expected: all tests pass.

### Task 5: End-to-end research data pipeline and capability probe

**Files:**
- Create: `src/alpha_lab/research_data/pipeline.py`
- Test: `tests/integration/test_research_data_pipeline.py`

**Interfaces:**
- Produces: `probe_research_data(config_dir: Path, data_root: Path, provider: TushareProvider | None = None) -> CapabilityReport`.
- Produces: `run_research_data_pipeline(config_dir: Path, data_root: Path, end_date: date | None = None, provider: TushareProvider | None = None) -> ResearchIngestionResult`.

- [ ] **Step 1: Write failing orchestration and resume tests**

```python
class FixtureProvider:
    def __init__(self, responses: dict[str, pd.DataFrame]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object]]] = []

    def query(
        self,
        api_name: str,
        params: Mapping[str, object],
        fields: tuple[str, ...],
    ) -> TushareQueryResult:
        self.calls.append((api_name, dict(params)))
        return fixture_result(api_name, self.responses[api_name], fields)


def test_pipeline_queries_only_historical_member_union(tmp_path: Path) -> None:
    provider = FixtureProvider(RESPONSES)
    first = run_research_data_pipeline(CONFIG, tmp_path, provider=provider)
    daily_calls = [params for name, params in provider.calls if name == "daily"]
    assert {params["ts_code"] for params in daily_calls} == {
        "600000.SH",
        "000001.SZ",
    }
    assert all("trade_date" not in params for params in daily_calls)
    second = run_research_data_pipeline(CONFIG, tmp_path, provider=provider)
    assert first.snapshot.snapshot_id == second.snapshot.snapshot_id
```

Add one separate test whose fixture provider raises a typed permission error for
`suspend_d`; assert the pipeline stops without publishing a manifest.

- [ ] **Step 2: Run pipeline tests and verify RED**

Run: `docker compose run --rm research pytest -q tests/integration/test_research_data_pipeline.py`

Expected: missing pipeline module.

- [ ] **Step 3: Implement bounded orchestration**

Fetch `stock_basic` statuses first, then membership/calendar, derive the historical union, then query daily/factor/suspension/name endpoints only for that union. Convert endpoint row-limit errors into actionable failures. Pass normalized tables and raw artifacts to the snapshot materializer.

- [ ] **Step 4: Run pipeline tests and verify GREEN**

Run: `docker compose run --rm research pytest -q tests/integration/test_research_data_pipeline.py`

Expected: all tests pass.

### Task 6: Versioned DuckDB migration and Phase 5 catalog sync

**Files:**
- Create: `src/alpha_lab/database/sql/002_research_data.sql`
- Modify: `src/alpha_lab/database/catalog.py`
- Test: `tests/unit/test_database_catalog.py`

**Interfaces:**
- Changes: `initialize_database(database_path: Path) -> InitializationResult` applies ordered packaged migrations and returns schema version 2.
- Produces: `sync_research_snapshot(database_path: Path, data_root: Path, manifest_path: Path) -> None`.

- [ ] **Step 1: Write failing migration tests**

```python
def test_database_applies_two_immutable_migrations(tmp_path: Path) -> None:
    result = initialize_database(tmp_path / "metadata.duckdb")
    assert result.schema_version == 2
    with duckdb.connect(str(result.database_path)) as connection:
        assert connection.execute("select count(*) from meta.schema_migration").fetchone()[0] == 2
```

Add idempotency, stored-hash mismatch, security-name history, provider capability, external reader, and snapshot sync tests.

- [ ] **Step 2: Run database tests and verify RED**

Run: `docker compose run --rm research pytest -q tests/unit/test_database_catalog.py`

Expected: schema version remains 1 and new tables are missing.

- [ ] **Step 3: Implement ordered migration loading and sync**

Discover packaged `[0-9][0-9][0-9]_*.sql`, compute each SHA256, verify every already-applied migration, and apply unapplied migrations in order within transactions. Do not change `001_initial.sql`. Register Phase 5 Parquet artifacts and interval metadata idempotently under the existing catalog write lock.

- [ ] **Step 4: Run database tests and verify GREEN**

Run: `docker compose run --rm research pytest -q tests/unit/test_database_catalog.py`

Expected: all database tests pass with schema version 2.

### Task 7: CLI, Make, Compose, docs, and live bounded verification

**Files:**
- Modify: `src/alpha_lab/cli.py`
- Modify: `Makefile`
- Modify: `compose.yaml`
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `docs/data_dictionary.md`
- Modify: `docs/database_design.md`
- Create: `docs/phase5_research_data.md`
- Create: `tests/test_phase5_contract.py`
- Modify: `tests/unit/test_cli.py`

**Interfaces:**
- Produces CLI commands: `research-data-probe`, `research-data-bootstrap`, `research-data-update`, `research-data-validate`, and `universe-asof`.
- Produces stable Make targets with the same names.

- [ ] **Step 1: Write failing contract and CLI tests**

Assert all five commands, `config/research_data.yaml`, migration 002, separate Compose data service, local environment variable names, and Phase 5 documentation exist.

- [ ] **Step 2: Run contract tests and verify RED**

Run: `docker compose run --rm research pytest -q tests/test_phase5_contract.py tests/unit/test_cli.py`

Expected: commands and files are missing.

- [ ] **Step 3: Implement CLI and documentation**

Commands render structured JSON, redact provider errors, require a token only for network calls, and make validation/universe queries read-only. Docs state the CSI 300/2020 scope, point-in-time formula, custom endpoint risk, capability gaps, data volume, and no performance claims.

- [ ] **Step 4: Run fixture-driven full gates**

Run:

```text
make lint
make test
make smoke
```

Expected: zero lint/type errors, all tests pass, and ARM64 smoke imports remain successful.

- [ ] **Step 5: Store credentials locally and probe capabilities**

Create ignored `.env` with `TUSHARE_TOKEN` and `TUSHARE_HTTP_URL` from the user-provided values. Confirm `git status --ignored` marks it ignored and no tracked diff contains the token. Run `make research-data-probe` and record only redacted capability results.

- [ ] **Step 6: Run bounded live bootstrap or retain explicit capability blocker**

Run: `make research-data-bootstrap`

Expected: a `p5-*` snapshot for CSI 300 history is published and `make research-data-validate SNAPSHOT=<id>` passes. If the provider reports permission or row-limit failures, preserve cache, report the exact endpoint capability as unresolved, and do not fabricate or silently downgrade data.

- [ ] **Step 7: Final verification, commit, push, and CI check**

Verify locked areas are unchanged, `.env` and `data/` are untracked/ignored, run fresh full gates, commit only source/config/tests/docs, push `main` as previously authorized, and wait for GitHub Actions success.
