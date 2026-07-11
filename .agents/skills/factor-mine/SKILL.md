---
name: factor-mine
description: Propose and evaluate one auditable A-share factor change inside the repository Phase 4 mining workflow. Use when Codex is asked to run a factor mining round or loop, create a bounded factor hypothesis and candidate, resume an interrupted mining run, or review structured factor results without changing locked evaluation, leakage, split, cost, or data-manifest assets.
---

# Factor Mine

Create exactly one primary factor change per round and hand all measurement to the fixed
repository evaluator.

## Workflow

1. Read `AGENTS.md`, the run's `research_brief.md`, `run_manifest.json`, prior round
   decisions, and current candidate YAML files.
2. Preserve `config/splits.yaml`, `config/costs.yaml`,
   `config/factor_evaluation.yaml`, `src/alpha_lab/evaluation/`,
   `tests/leakage/`, and `data/manifests/` byte-for-byte.
3. Select one unused `Fxxxx` ID in the configured mining range.
4. Propose one change only: a new factor, one operator, one window, or one combination.
   State the changed variable and falsification criteria explicitly.
5. Write one combined proposal matching `schemas/proposal.schema.json`. Keep the Python
   source in `source_code`; do not write metrics or claim performance.
6. Ensure the candidate:
   - reads only declared market fields;
   - returns `trade_date`, `instrument`, `value` in that order;
   - uses only historical shifts and trailing windows;
   - declares `min_periods` for every rolling window;
   - converts infinities to NaN;
   - performs no network, filesystem, subprocess, label, or test-specific access.
7. Run `make mining-round RUN=<run_id>`. Let the fixed evaluator create
   `test_report.json` and `factor_result.json`.
8. Treat `ACCEPT` as a recommendation for human review only. Never update
   `accepted_factor_ids` automatically and never hide REJECT or ERROR rounds.
9. For multiple rounds, use `make mining-loop RUN=<run_id> ROUNDS=5`. Resume the same
   run after interruption; do not delete or renumber prior rounds.

## Integrity rules

- Do not inspect locked test values or tune thresholds to a candidate.
- Do not change more than one primary variable in a round.
- Do not reuse an existing factor ID with different bytes.
- Do not infer unavailable industry, size, regime, suspension, or limit data.
- Stop and record ERROR if a locked-area hash changes or the fixed evaluator fails.
