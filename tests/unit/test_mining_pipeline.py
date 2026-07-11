from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from alpha_lab.mining.models import CandidateProposal
from alpha_lab.mining.pipeline import (
    initialize_mining_run,
    run_mining_loop,
)

ROOT = Path(__file__).resolve().parents[2]


def _source(window: int) -> str:
    return f"""from __future__ import annotations

import numpy as np
import pandas as pd


def compute(market: pd.DataFrame) -> pd.DataFrame:
    ordered = market.sort_values(["instrument", "trade_date"], kind="stable")
    returns = ordered.groupby("instrument", sort=False)["close"].pct_change()
    values = returns.groupby(ordered["instrument"], sort=False).transform(
        lambda series: series.rolling(window={window}, min_periods={window}).mean()
    )
    values = values.where(np.isfinite(values))
    result = ordered.loc[:, ["trade_date", "instrument"]].copy()
    result["value"] = values
    return result
"""


def _proposal(run_id: str, round_number: int, factor_id: str) -> dict[str, object]:
    created_at = datetime(2026, 7, 11, tzinfo=UTC).isoformat()
    hypothesis = "Recent average close returns may persist over a short horizon."
    formula = "Mean(PctChange(close, 1), 5)"
    shared = {
        "factor_id": factor_id,
        "hypothesis": hypothesis,
        "formula": formula,
        "inputs": ["close"],
        "lookback": 6,
        "direction": 1,
        "family": "momentum",
        "parent_factor_ids": [],
    }
    return {
        "schema_version": 1,
        "hypothesis": {
            "schema_version": 1,
            "run_id": run_id,
            "round_number": round_number,
            "title": "Short close momentum",
            "rationale": "This isolates one bounded trailing return transformation.",
            "primary_change": "new_factor",
            "changed_variable": "factor_formula",
            "expected_effect": "Positive validation rank information coefficient.",
            "falsification_criteria": ["Fails any fixed promotion check."],
            "created_at": created_at,
            **shared,
        },
        "metadata": {
            "name": f"mined_close_momentum_{factor_id.lower()}",
            "author": "codex",
            "created_at": created_at,
            "status": "candidate",
            **shared,
        },
        "source_code": _source(5),
    }


def _sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    config = repo / "config"
    data = repo / "data"
    candidates = repo / "src" / "alpha_lab" / "factors" / "candidates"
    leakage = repo / "tests" / "leakage"
    evaluation = repo / "src" / "alpha_lab" / "evaluation"
    for path in (config, candidates, leakage, evaluation):
        path.mkdir(parents=True)

    for name in (
        "mining.yaml",
        "factor_registry.yaml",
        "splits.yaml",
        "costs.yaml",
        "factor_evaluation.yaml",
    ):
        shutil.copy2(ROOT / "config" / name, config / name)
    for path in (ROOT / "src" / "alpha_lab" / "factors" / "candidates").glob(
        "F000[1-3].*"
    ):
        shutil.copy2(path, candidates / path.name)
    (evaluation / "pipeline.py").write_text("LOCKED = True\n", encoding="utf-8")
    (leakage / "test_guard.py").write_text("LOCKED = True\n", encoding="utf-8")

    snapshot = "p1-mining-test"
    manifest_dir = data / "manifests" / snapshot
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps({"snapshot_id": snapshot}), encoding="utf-8"
    )
    (manifest_dir / "quality_report.json").write_text(
        json.dumps({"status": "warning"}), encoding="utf-8"
    )
    state = data / "state" / "latest_snapshot.txt"
    state.parent.mkdir(parents=True)
    state.write_text(f"{snapshot}\n", encoding="utf-8")
    monkeypatch.setattr(
        "alpha_lab.mining.pipeline._git_identity",
        lambda _: {"commit": "a" * 40, "dirty": False},
    )
    return repo, data


def test_proposal_rejects_metadata_that_does_not_match_hypothesis() -> None:
    document = _proposal("mining-test", 1, "F1000")
    document["metadata"]["lookback"] = 7  # type: ignore[index]

    with pytest.raises(ValueError, match="hypothesis/metadata mismatch"):
        CandidateProposal.model_validate(document)


def test_five_round_loop_retains_evaluator_errors_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, data = _sandbox(tmp_path, monkeypatch)
    experiments = repo / "experiments"
    proposals = repo / "proposals"
    proposals.mkdir()
    for round_number in range(1, 6):
        factor_id = f"F{999 + round_number:04d}"
        (proposals / f"round_{round_number:04d}.json").write_text(
            json.dumps(_proposal("mining-test", round_number, factor_id)),
            encoding="utf-8",
        )

    calls = 0

    def fail_evaluation(*_: object, **__: object) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("simulated evaluator interruption")

    monkeypatch.setattr("alpha_lab.mining.pipeline.evaluate_factor", fail_evaluation)
    results = run_mining_loop(
        "mining-test",
        5,
        repo_root=repo,
        config_dir=repo / "config",
        data_dir=data,
        experiments_dir=experiments,
        artifacts_dir=repo / "artifacts",
        proposals_dir=proposals,
    )

    assert calls == 5
    assert [item.decision for item in results] == ["ERROR"] * 5
    run_dir = experiments / "mining-test"
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["completed_rounds"] == 5
    assert manifest["decision_counts"] == {"ACCEPT": 0, "REJECT": 0, "ERROR": 5}
    for round_number in range(1, 6):
        round_dir = run_dir / f"round_{round_number:04d}"
        assert (round_dir / "hypothesis.json").is_file()
        assert (round_dir / "candidate").is_dir()
        assert (round_dir / "test_report.json").is_file()
        assert (round_dir / "factor_result.json").is_file()
        assert (round_dir / "decision.json").is_file()
    assert (run_dir / "final_report.md").is_file()


def test_locked_area_change_blocks_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, data = _sandbox(tmp_path, monkeypatch)
    experiments = repo / "experiments"
    initialize_mining_run(
        "mining-lock-test",
        1,
        repo_root=repo,
        config_dir=repo / "config",
        data_dir=data,
        experiments_dir=experiments,
    )
    (repo / "config" / "splits.yaml").write_text("changed: true\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="locked areas changed"):
        run_mining_loop(
            "mining-lock-test",
            1,
            repo_root=repo,
            config_dir=repo / "config",
            data_dir=data,
            experiments_dir=experiments,
            artifacts_dir=repo / "artifacts",
        )
