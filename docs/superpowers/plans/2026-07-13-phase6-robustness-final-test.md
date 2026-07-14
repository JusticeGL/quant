# Phase 6 Robustness and Final Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build point-in-time size and industry exposures, freeze F1002 and F1003, run five pre-test walk-forward folds with cost and exposure analysis, and enforce explicit per-freeze approval before locked-test access.

**Architecture:** A new `alpha_lab.robustness` package consumes the immutable Phase 5 snapshot and creates a separate `p6x-*` exposure snapshot from cached Tushare `daily_basic`, `index_classify`, and `index_member_all` responses. Freeze, robustness, request, approval, and final-test artifacts form a hash-linked immutable state machine; pre-test readers cannot access dates on or after 2026-01-01.

**Tech Stack:** Linux Python 3.11 on native ARM64, pandas, PyArrow/Parquet, DuckDB, Pydantic, Typer, Tushare REST through the existing provider, pytest, Ruff, mypy.

## Global Constraints

- Formal Python commands run only through `make` or `docker compose` in Linux Python 3.11.
- Apple Silicon remains native ARM64; no `linux/amd64` fallback.
- Do not edit `config/splits.yaml`, `config/costs.yaml`, `config/factor_evaluation.yaml`, `src/alpha_lab/evaluation`, `tests/leakage`, applied migrations, Phase 5 manifests, or `data/raw` files.
- Do not change F1002 or F1003 source or metadata during Phase 6.
- Keep the Phase 5 snapshot `p5-ecaa6e8aeae6b9f8fb25` immutable.
- Pre-test execution must reject every read whose requested range reaches 2026-01-01 before opening Parquet.
- Approval of the design does not authorize final-test access; stop after generating pre-test reports and test requests.
- Never commit `.env`, credentials, Parquet, DuckDB, predictions, NAV series, trades, caches, or generated large reports.
- Every behavior change follows failing test, observed RED, minimal implementation, observed GREEN.

---

### Task 1: Strict Phase 6 policy and data contracts

**Files:**
- Create: `config/robustness.yaml`
- Create: `src/alpha_lab/robustness/__init__.py`
- Create: `src/alpha_lab/robustness/config.py`
- Create: `src/alpha_lab/robustness/contracts.py`
- Test: `tests/unit/test_robustness_config.py`

**Interfaces:**
- Produces: `RobustnessConfig`, `ExposureSourceConfig`, `WalkForwardFold`, and `load_robustness_config(path: Path) -> tuple[RobustnessConfig, str]`.
- Produces: `ExposureTables`, `ExposureSnapshotResult`, `FrozenCandidate`, and `RobustnessResult` dataclasses.

- [ ] **Step 1: Write failing strict-policy tests**

```python
def test_phase6_policy_has_locked_calendar_and_candidates() -> None:
    config, _ = load_robustness_config(ROOT / "config" / "robustness.yaml")
    assert config.factor_ids == ["F1002", "F1003"]
    assert len(config.walk_forward_folds) == 5
    assert config.test.start == date(2026, 1, 1)
    assert config.test.end == date(2026, 7, 11)
    assert config.test.access == "human_approval_only"
    assert config.cost_multipliers == [0.5, 1.0, 1.5, 2.0]


def test_phase6_policy_rejects_fold_overlap_with_test() -> None:
    document = yaml.safe_load((ROOT / "config" / "robustness.yaml").read_text())
    document["walk_forward_folds"][-1]["end"] = "2026-01-02"
    with pytest.raises(ValueError, match="test boundary"):
        RobustnessConfig.model_validate(document)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `docker compose run --rm research pytest -q tests/unit/test_robustness_config.py`

Expected: FAIL because `alpha_lab.robustness.config` does not exist.

- [ ] **Step 3: Implement the strict configuration and contracts**

```python
class WalkForwardFold(StrictModel):
    fold_id: str = Field(pattern=r"^wf_[0-9]{4}$")
    start: date
    end: date


class RobustnessConfig(StrictModel):
    schema_version: Literal[1]
    policy_id: str
    phase5_snapshot_id: str
    factor_ids: list[Literal["F1002", "F1003"]]
    warmup: DateRange
    walk_forward_folds: list[WalkForwardFold]
    test: LockedTestRange
    cost_multipliers: list[float]
    minimum_fold_coverage: float
    minimum_direction_consistent_folds: int
    minimum_industry_neutral_ic_retention: float
    size_correlation_risk_threshold: float
    exposure_source: ExposureSourceConfig
```

Write the exact 2020 warm-up, five annual 2021-2025 folds, locked 2026 test, 70 percent coverage, four consistent folds, 50 percent neutral IC retention, 0.30 size-risk threshold, SW2021 classification, and required endpoint names into `config/robustness.yaml`. Hash canonical YAML content after validation.

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `docker compose run --rm research pytest -q tests/unit/test_robustness_config.py`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add config/robustness.yaml src/alpha_lab/robustness tests/unit/test_robustness_config.py
git commit -m "feat: add phase 6 robustness policy"
```

### Task 2: Point-in-time exposure acquisition and immutable snapshot

**Files:**
- Create: `src/alpha_lab/robustness/exposure_data.py`
- Create: `src/alpha_lab/robustness/exposure_snapshot.py`
- Test: `tests/unit/test_exposure_data.py`
- Test: `tests/integration/test_exposure_snapshot.py`

**Interfaces:**
- Consumes: `RobustnessConfig`, `TushareProvider`, Phase 5 manifest and security master.
- Produces: `probe_exposure_capabilities(config_dir: Path, data_dir: Path) -> dict[str, object]`.
- Produces: `build_exposure_snapshot(config_dir: Path, data_dir: Path) -> ExposureSnapshotResult`.
- Produces: `validate_exposure_snapshot(data_dir: Path, snapshot_id: str) -> dict[str, object]`.

- [ ] **Step 1: Write failing normalization and PIT tests**

```python
def test_daily_basic_converts_ten_thousand_cny_to_cny() -> None:
    raw = pd.DataFrame([{
        "ts_code": "600000.SH", "trade_date": "20210104",
        "total_mv": 123.4, "circ_mv": 100.0,
    }])
    result = normalize_market_cap(raw)
    assert result.loc[0, "total_market_cap_cny"] == 1_234_000.0
    assert result.loc[0, "float_market_cap_cny"] == 1_000_000.0
    assert result.loc[0, "known_at"] == pd.Timestamp("2021-01-04", tz="UTC")


def test_industry_asof_uses_effective_and_known_dates() -> None:
    selected = industry_as_of(intervals, date(2021, 1, 4))
    assert "CN:SSE:600000" not in set(selected["security_id"])
```

Also test required fields, positive market cap, SW2021 filtering, duplicate keys, interval overlap, unknown Phase 5 securities, row-limit detection, and no current-industry backfill.

- [ ] **Step 2: Run the tests and verify RED**

Run: `docker compose run --rm research pytest -q tests/unit/test_exposure_data.py`

Expected: FAIL because exposure functions are missing.

- [ ] **Step 3: Implement bounded provider queries and canonical tables**

Use these exact requested fields:

```python
DAILY_BASIC_FIELDS = ("ts_code", "trade_date", "total_mv", "circ_mv")
INDEX_CLASSIFY_FIELDS = (
    "index_code", "industry_name", "level", "industry_code", "src",
)
INDUSTRY_MEMBER_FIELDS = (
    "l1_code", "l1_name", "l2_code", "l2_name", "l3_code", "l3_name",
    "ts_code", "name", "in_date", "out_date", "is_new",
)
```

Probe one bounded date and one SW2021 level-one industry before bulk acquisition. Query `daily_basic` per historical CSI 300 member using the same maximum concurrency of four and immutable request cache. Query the SW2021 dictionary once and each level-one industry membership once. Treat missing announcement dates as `effective_date_fallback` and record that provenance.

- [ ] **Step 4: Implement deterministic `p6x-*` publication**

Write stable Zstandard Parquet:

```text
data/exposures/p6x-*/market_cap/year=YYYY/part.parquet
data/exposures/p6x-*/industry_definition.parquet
data/exposures/p6x-*/industry_membership.parquet
data/exposures/p6x-*/industry_membership_pretest.parquet
data/manifests/p6x-*/quality_report.json
data/manifests/p6x-*/manifest.json
data/state/latest_exposure_snapshot.txt
```

The manifest identity contains the Phase 5 manifest SHA256, policy SHA256, sorted raw request identities, and written artifact hashes. Publish the latest pointer only after artifact and quality validation. Refuse quality status `error`.

- [ ] **Step 5: Run snapshot tests and verify GREEN**

Run: `docker compose run --rm research pytest -q tests/unit/test_exposure_data.py tests/integration/test_exposure_snapshot.py`

Expected: all tests pass, repeated fixture publication has identical snapshot and manifest hashes.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/alpha_lab/robustness/exposure_data.py src/alpha_lab/robustness/exposure_snapshot.py tests/unit/test_exposure_data.py tests/integration/test_exposure_snapshot.py
git commit -m "feat: add point-in-time exposure snapshot"
```

### Task 3: DuckDB schema v3 exposure and approval catalog

**Files:**
- Create: `src/alpha_lab/database/sql/003_robustness.sql`
- Modify: `src/alpha_lab/database/catalog.py`
- Modify: `docs/database_design.md`
- Modify: `docs/data_dictionary.md`
- Test: `tests/unit/test_database_phase6.py`

**Interfaces:**
- Produces: `sync_exposure_snapshot(database_path: Path, data_dir: Path, manifest_path: Path) -> None`.
- Produces catalog tables for industry definitions, industry membership, factor freezes, test requests, approvals, and final-test runs.

- [ ] **Step 1: Write failing migration and idempotency tests**

```python
def test_database_applies_robustness_migration(tmp_path: Path) -> None:
    result = initialize_database(tmp_path / "metadata.duckdb")
    assert result.schema_version == 3
    with duckdb.connect(str(result.database_path), read_only=True) as con:
        tables = {row[0] for row in con.execute(
            "SELECT table_schema || '.' || table_name FROM information_schema.tables"
        ).fetchall()}
    assert "ref.industry_definition" in tables
    assert "ref.industry_membership_history" in tables
    assert "research.factor_freeze" in tables
    assert "research.test_approval" in tables
```

Add a fixture exposure manifest and assert two syncs produce identical counts.

- [ ] **Step 2: Run database tests and verify RED**

Run: `docker compose run --rm research pytest -q tests/unit/test_database_phase6.py`

Expected: FAIL with schema version 2 or missing tables.

- [ ] **Step 3: Add immutable migration 003 and catalog sync**

Create normalized metadata tables with SHA256 identifiers and foreign keys. Register exposure frames and use bulk `INSERT ... SELECT ... ON CONFLICT` operations. Add migration 003 to the ordered migration list and set `SCHEMA_VERSION = 3`; do not edit migrations 001 or 002.

- [ ] **Step 4: Run database tests and verify GREEN**

Run: `docker compose run --rm research pytest -q tests/unit/test_database_phase6.py tests/unit/test_database_phase5.py tests/unit/test_database_catalog.py`

Expected: all database tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/alpha_lab/database/sql/003_robustness.sql src/alpha_lab/database/catalog.py docs/database_design.md docs/data_dictionary.md tests/unit/test_database_phase6.py
git commit -m "feat: catalog phase 6 exposure metadata"
```

### Task 4: Locked readers and immutable candidate freezes

**Files:**
- Create: `src/alpha_lab/robustness/io.py`
- Create: `src/alpha_lab/robustness/freeze.py`
- Test: `tests/unit/test_robustness_io.py`
- Test: `tests/unit/test_candidate_freeze.py`

**Interfaces:**
- Produces: `read_pretest_market(data_dir: Path, capability_snapshot_id: str, end_before: date) -> pd.DataFrame`; the ID is the root p6x snapshot and no Phase 5-manifest fallback is allowed.
- Produces: `read_pretest_exposures(data_dir: Path, snapshot_id: str, end_before: date) -> tuple[pd.DataFrame, pd.DataFrame]`.
- Produces: `freeze_candidate(factor_id: str, config_dir: Path, data_dir: Path, experiments_dir: Path) -> FrozenCandidate`.
- Produces: `validate_freeze(freeze_path: Path, config_dir: Path, data_dir: Path) -> dict[str, object]`.

Task 2 publishes `manifests/<p6x-id>/pretest_capability.json` and binds its
manifest-relative reference into root identity. Task 4/5 freeze and reader
paths validate only that capability and its safe artifact list. Full roots and
full quality reports remain publication/catalog inputs and future approved
Task 6 final-test inputs only.

Before safe artifact access, Task 4/5 also read `data/metadata.duckdb` in
read-only mode and require the exact Task 3 catalog attestation for the p6x root
and unique capability artifact plus passing administrative quality. They then
validate the closed 2020-2025 safe partition set and Parquet footer row counts.
Missing/old/mismatched catalog state fails closed before Parquet access.
The migration ledger is matched exactly by version, name, and packaged SQL
SHA256, including rejection of extras. This is a cooperative publisher check,
not protection from manual database mutation or replacement by an actor with
the same write authority; that stronger threat model needs an external trust
root or enforced read-only storage.

- [ ] **Step 1: Write failing locked-read tests**

```python
def test_pretest_reader_rejects_test_boundary_before_parquet_read(monkeypatch) -> None:
    opened = False
    def forbidden(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal opened
        opened = True
        raise AssertionError("Parquet must not be opened")
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(PermissionError, match="locked test"):
        read_pretest_market(DATA, SNAPSHOT, date(2026, 1, 1))
    assert opened is False
```

Add tests that a freeze rejects F1001, pins F1002 source/metadata and both snapshot hashes, is byte-identical on repeat, and fails after candidate or manifest hash drift.

- [ ] **Step 2: Run freeze tests and verify RED**

Run: `docker compose run --rm research pytest -q tests/unit/test_robustness_io.py tests/unit/test_candidate_freeze.py`

Expected: FAIL because the modules are missing.

- [ ] **Step 3: Implement locked reads and freeze identity**

The reader first validates `end_before < config.test.start`, then resolves only artifact partitions whose year is before 2026 and applies explicit `< 2026-01-01` filters. Convert Phase 5 columns to the factor/backtest contract without reading the test range.

Freeze identity is the SHA256 of canonical JSON containing factor ID, source hash, metadata hash, Phase 5 manifest hash, exposure manifest hash, robustness policy hash, cost policy hash, test boundaries, and Git commit. Write `experiments/phase6/freeze-<hash>/freeze.json` atomically.

- [ ] **Step 4: Run freeze tests and verify GREEN**

Run: `docker compose run --rm research pytest -q tests/unit/test_robustness_io.py tests/unit/test_candidate_freeze.py`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add src/alpha_lab/robustness/io.py src/alpha_lab/robustness/freeze.py tests/unit/test_robustness_io.py tests/unit/test_candidate_freeze.py
git commit -m "feat: freeze phase 6 candidates"
```

### Task 5: Walk-forward, cost sensitivity, and exposure evaluation

**Files:**
- Create: `src/alpha_lab/robustness/walk_forward.py`
- Create: `src/alpha_lab/robustness/exposures.py`
- Create: `src/alpha_lab/robustness/report.py`
- Test: `tests/unit/test_walk_forward.py`
- Test: `tests/unit/test_factor_exposures.py`
- Test: `tests/integration/test_robustness_pipeline.py`

**Interfaces:**
- Produces: `build_fold_labels(market: pd.DataFrame, fold: WalkForwardFold) -> pd.DataFrame`.
- Produces: `scale_costs(costs: CostConfig, multiplier: float) -> CostConfig`.
- Produces: `calculate_exposures(scores: pd.DataFrame, market_cap: pd.DataFrame, industries: pd.DataFrame, labels: pd.DataFrame) -> dict[str, object]`.
- Produces: `evaluate_frozen_candidate(freeze_path: Path, config_dir: Path, data_dir: Path, experiments_dir: Path) -> RobustnessResult`.

- [ ] **Step 1: Write failing fold-boundary and cost tests**

```python
def test_labels_do_not_cross_fold_end() -> None:
    labels = build_fold_labels(market, WalkForwardFold(
        fold_id="wf_2025", start=date(2025, 1, 1), end=date(2025, 12, 31)
    ))
    assert labels["entry_date"].max().date() <= date(2025, 12, 31)
    assert labels["exit_date"].max().date() <= date(2025, 12, 31)


def test_cost_scaling_includes_minimum_commission() -> None:
    doubled = scale_costs(costs, 2.0)
    assert doubled.rules[0].commission_rate == costs.rules[0].commission_rate * 2
    assert doubled.rules[0].minimum_commission == costs.rules[0].minimum_commission * 2
```

- [ ] **Step 2: Write failing PIT exposure tests**

```python
def test_industry_neutral_scores_use_asof_membership() -> None:
    report = calculate_exposures(scores, market_cap, industries, labels)
    assert report["industry"]["joined_rows"] == 4
    assert report["industry"]["neutral_rank_ic"] is not None
    assert report["size"]["risk_flag"] is False
```

Test that no industry interval is joined before `known_at`, missing exposure rows remain missing, and size uses `log(total_market_cap_cny)`.

- [ ] **Step 3: Run evaluation tests and verify RED**

Run: `docker compose run --rm research pytest -q tests/unit/test_walk_forward.py tests/unit/test_factor_exposures.py`

Expected: FAIL because evaluation functions are missing.

- [ ] **Step 4: Implement fold-local metrics and backtests**

Compute each candidate once on warm-up plus pre-test market, but slice scores by fold before labels, metrics, and backtests. Use fold-local next-open labels and pass `allowed_end=fold.end` to `run_topk_backtest`. Produce one result per fold and per cost multiplier. A direction-consistent fold has non-null Rank IC whose sign matches the candidate direction after score orientation.

- [ ] **Step 5: Implement exposure analysis and locked gates**

Join market cap on exact `(trade_date, security_id)`. Join industry where effective and known intervals contain the trade date. Standardize scores inside `(trade_date, industry_id)` groups with at least two valid members. Calculate original and neutral Rank IC on the same joined rows. Apply exactly the four approved gates and the separate 0.30 size-risk flag.

- [ ] **Step 6: Implement immutable reports**

Write canonical `walk_forward.json`, `cost_sensitivity.json`, `exposure_report.json`, and deterministic `robustness_report.md` under the freeze directory. Store large fold predictions, NAV, and trades as ignored Parquet. Refuse differing bytes at an existing path.

- [ ] **Step 7: Run evaluation tests and verify GREEN**

Run: `docker compose run --rm research pytest -q tests/unit/test_walk_forward.py tests/unit/test_factor_exposures.py tests/integration/test_robustness_pipeline.py`

Expected: all fixture folds, exposure reports, gates, and repeat hashes pass.

- [ ] **Step 8: Commit Task 5**

```bash
git add src/alpha_lab/robustness/walk_forward.py src/alpha_lab/robustness/exposures.py src/alpha_lab/robustness/report.py tests/unit/test_walk_forward.py tests/unit/test_factor_exposures.py tests/integration/test_robustness_pipeline.py
git commit -m "feat: evaluate phase 6 robustness"
```

### Task 6: Test requests, explicit approval, and immutable final reports

**Files:**
- Create: `src/alpha_lab/robustness/approval.py`
- Create: `src/alpha_lab/robustness/final_test.py`
- Test: `tests/unit/test_test_approval.py`
- Test: `tests/integration/test_final_test_gate.py`

**Interfaces:**
- Produces: `create_test_request(freeze_path: Path) -> Path`.
- Produces: `approve_test_request(request_path: Path, approver: str, confirmed_freeze_sha256: str) -> Path`.
- Produces: `run_final_test(approval_path: Path, config_dir: Path, data_dir: Path, experiments_dir: Path) -> Path`.

- [ ] **Step 1: Write failing approval-gate tests**

```python
def test_request_requires_pretest_pass(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="robustness gate"):
        create_test_request(failed_freeze_path)


def test_final_test_rejects_missing_approval_before_read(monkeypatch) -> None:
    opened = False
    def forbidden_read(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal opened
        opened = True
        raise AssertionError("locked market must not be read")
    monkeypatch.setattr(final_test, "_read_locked_market", forbidden_read)
    with pytest.raises(PermissionError, match="approval"):
        run_final_test(missing_path, CONFIG, DATA, tmp_path)
    assert opened is False
```

Also test wrong freeze confirmation, changed factor hash, changed policy/snapshot hash, malformed approver, idempotent approval, and immutable final result conflicts.

- [ ] **Step 2: Run approval tests and verify RED**

Run: `docker compose run --rm research pytest -q tests/unit/test_test_approval.py tests/integration/test_final_test_gate.py`

Expected: FAIL because approval and final-test modules are missing.

- [ ] **Step 3: Implement hash-linked request and approval artifacts**

A request contains the freeze ID/hash, robustness report hash, all gate results, exact locked range, and `status=test_requested`. Approval requires a non-empty approver and exact freeze hash, and contains request hash, approval timestamp, and `status=approved`. IDs derive from canonical content excluding their own ID field.

The implementation additionally revalidates the full Task 5 JSON set and
recomputes the gates, pins the effective config/code execution bundle in the
request, and records exact freeze/request/approval tuples in the existing
schema-v3 catalog. The catalog is verified read-only before locked access; no
new migration is required.

- [ ] **Step 4: Implement final-test validation before data access**

Validate approval, request, freeze, candidate, policy, cost, Phase 5, and exposure hashes in that order. Only then read 2026-01-01 through 2026-07-11 partitions. Use the same factor, metrics, cost scenarios, and exposure functions without applying pre-test gates to hide or suppress results. Publish immutable `result.json` and `report.md` even when results are unfavorable.

- [ ] **Step 5: Run approval tests and verify GREEN**

Run: `docker compose run --rm research pytest -q tests/unit/test_test_approval.py tests/integration/test_final_test_gate.py`

Expected: all gate-order, hash-drift, idempotency, and immutable-report tests pass.

- [ ] **Step 6: Commit Task 6**

```bash
git add src/alpha_lab/robustness/approval.py src/alpha_lab/robustness/final_test.py tests/unit/test_test_approval.py tests/integration/test_final_test_gate.py
git commit -m "feat: enforce phase 6 test approval"
```

### Task 7: CLI, Make targets, audit tracking, and documentation

**Files:**
- Modify: `src/alpha_lab/cli.py`
- Modify: `Makefile`
- Modify: `.gitignore`
- Modify: `README.md`
- Create: `docs/phase6_robustness.md`
- Create: `tests/test_phase6_contract.py`
- Modify: `tests/unit/test_cli.py`

**Interfaces:**
- Produces the seven stable commands defined in the approved design.

- [ ] **Step 1: Write failing command-contract tests**

```python
def test_phase6_make_targets_exist() -> None:
    makefile = (ROOT / "Makefile").read_text()
    for target in (
        "exposure-probe", "exposure-bootstrap", "robustness-freeze",
        "robustness-eval", "test-request", "test-approve", "final-test",
    ):
        assert f"\n{target}:" in makefile
```

Assert CLI help lists matching commands and `.gitignore` permits only the named Phase 6 small JSON/Markdown audit files, including approvals, while continuing to ignore Parquet and all large generated outputs.

- [ ] **Step 2: Run contract tests and verify RED**

Run: `docker compose run --rm research pytest -q tests/test_phase6_contract.py tests/unit/test_cli.py`

Expected: FAIL because Phase 6 commands are absent.

- [ ] **Step 3: Add CLI and Make targets**

Use required Make variables and fail before invoking Python when `ID`, `FREEZE`, `REQUEST`, `APPROVER`, `CONFIRM`, or `APPROVAL` is missing. CLI catches provider, validation, DuckDB, and filesystem errors, prints no credentials, and emits structured JSON summaries.

- [ ] **Step 4: Add docs and audit whitelist**

Document the immutable data flow, exact calendar, warning semantics, commands, expected pause before test approval, recovery from provider failures, and the fact that results are research-only. Whitelist only named Phase 6 audit JSON/Markdown files; keep large outputs ignored.

- [ ] **Step 5: Run contract tests and verify GREEN**

Run: `docker compose run --rm research pytest -q tests/test_phase6_contract.py tests/unit/test_cli.py`

Expected: all command and documentation contracts pass.

- [ ] **Step 6: Commit Task 7**

```bash
git add src/alpha_lab/cli.py Makefile .gitignore README.md docs/phase6_robustness.md tests/test_phase6_contract.py tests/unit/test_cli.py
git commit -m "feat: add phase 6 workflow commands"
```

### Task 8: Live exposure snapshot and F1002/F1003 pre-test checkpoint

**Files:**
- Generated and ignored: `data/raw/tushare/*`, `data/exposures/p6x-*`, large Phase 6 Parquet.
- Generated small audit files: `experiments/phase6/freeze-*/freeze.json`, pre-test JSON, and Markdown reports.

**Interfaces:**
- Consumes all prior tasks.
- Produces valid exposure snapshot and pre-test reports, then stops before approval.

- [ ] **Step 1: Run the bounded capability probe**

Run: `make exposure-probe`

Expected: `daily_basic`, `index_classify`, and SW2021 `index_member_all` report required fields and bounded row counts. If any capability fails, diagnose relay shape without switching providers or fabricating exposure data.

- [ ] **Step 2: Build and validate the live exposure snapshot**

Run: `make exposure-bootstrap`

Expected: one immutable `p6x-*` snapshot with non-error quality and DuckDB schema v3 sync. Repeat the command and require zero network requests with identical snapshot and manifest hashes.

- [ ] **Step 3: Freeze both approved candidates**

Run: `make robustness-freeze ID=F1002`

Run: `make robustness-freeze ID=F1003`

Expected: two immutable freeze IDs pinning different candidate hashes but identical policy and data hashes.

- [ ] **Step 4: Run both pre-test robustness evaluations**

Run: `make robustness-eval FREEZE=<F1002-freeze-id>`

Run: `make robustness-eval FREEZE=<F1003-freeze-id>`

Expected: five complete fold results, four cost scenarios, size and industry exposures, deterministic reports, and no test access.

- [ ] **Step 5: Generate requests only for passing freezes**

Run for each passing freeze: `make test-request FREEZE=<freeze-id>`

Expected: request includes the exact freeze hash and reports `test_accessed=false`. A failing freeze receives a documented pre-test report but cannot create a request.

- [ ] **Step 6: Stop for explicit user review**

Report both candidates' fold metrics, cost stability, exposures, gate outcomes, freeze hashes, and request IDs. Do not run `make test-approve` or `make final-test` in this implementation session.

- [ ] **Step 7: Run fresh completion gates**

Run: `make lint`

Run: `make test`

Run: `make smoke`

Expected: all commands exit zero in native ARM64 Linux Python 3.11.

- [ ] **Step 8: Verify repository security and commit the Phase 6 checkpoint**

Confirm `.env` and `data/` are ignored, scan staged content for the live token, verify migrations 001 and 002 hashes are unchanged, and stage only source, configuration, tests, docs, lockfile changes, and selected small audit summaries. Commit and push `main` only after all gates pass and the previously authorized direct-upload workflow remains valid.
