from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from alpha_lab.evaluation.leakage import inspect_factor_source
from alpha_lab.evaluation.pipeline import evaluate_factor
from alpha_lab.factors.registry import FactorRegistry
from alpha_lab.mining.models import (
    CandidateProposal,
    MiningConfig,
    MiningDecision,
)

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$")


@dataclass(frozen=True)
class MiningRoundResult:
    run_id: str
    round_number: int
    factor_id: str
    decision: str
    round_dir: Path
    decision_path: Path


def initialize_mining_run(
    run_id: str,
    requested_rounds: int,
    *,
    repo_root: Path,
    config_dir: Path,
    data_dir: Path,
    experiments_dir: Path,
) -> Path:
    config, config_hash = _load_mining_config(config_dir / "mining.yaml")
    _validate_run_id(run_id)
    if not 1 <= requested_rounds <= config.maximum_rounds:
        raise ValueError(
            f"requested rounds must be between 1 and {config.maximum_rounds}"
        )
    run_dir = experiments_dir / run_id
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if int(manifest["requested_rounds"]) != requested_rounds:
            raise ValueError("existing run uses a different requested_rounds value")
        return run_dir

    snapshot_id = _latest_snapshot(data_dir)
    snapshot_manifest = data_dir / "manifests" / snapshot_id / "manifest.json"
    if not snapshot_manifest.is_file():
        raise ValueError(f"snapshot manifest is missing: {snapshot_manifest}")
    locked_hashes = _locked_area_hashes(repo_root, config_dir, data_dir, snapshot_id)
    registry = FactorRegistry(
        repo_root / "src" / "alpha_lab" / "factors" / "candidates",
        config_dir / "factor_registry.yaml",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "proposals").mkdir(exist_ok=True)
    git = _git_identity(repo_root)
    manifest = {
        "schema_version": 1,
        "phase": 4,
        "run_id": run_id,
        "status": "active",
        "requested_rounds": requested_rounds,
        "completed_rounds": 0,
        "current_round": None,
        "decision_counts": {"ACCEPT": 0, "REJECT": 0, "ERROR": 0},
        "data_snapshot_id": snapshot_id,
        "mining_policy_id": config.policy_id,
        "mining_config_sha256": config_hash,
        "git": git,
        "locked_area_hashes": locked_hashes,
        "rounds": [],
    }
    _atomic_json(manifest_path, manifest)
    brief = _research_brief(run_id, snapshot_id, requested_rounds, registry)
    _atomic_text(run_dir / "research_brief.md", brief)
    return run_dir


def run_mining_round(
    run_id: str,
    *,
    repo_root: Path,
    config_dir: Path,
    data_dir: Path,
    experiments_dir: Path,
    artifacts_dir: Path,
    proposal_path: Path | None = None,
) -> MiningRoundResult:
    run_dir = experiments_dir / run_id
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        config, _ = _load_mining_config(config_dir / "mining.yaml")
        initialize_mining_run(
            run_id,
            config.default_rounds,
            repo_root=repo_root,
            config_dir=config_dir,
            data_dir=data_dir,
            experiments_dir=experiments_dir,
        )
    with _run_lock(run_dir):
        return _run_mining_round_locked(
            run_id,
            repo_root=repo_root,
            config_dir=config_dir,
            data_dir=data_dir,
            experiments_dir=experiments_dir,
            artifacts_dir=artifacts_dir,
            proposal_path=proposal_path,
        )


def _run_mining_round_locked(
    run_id: str,
    *,
    repo_root: Path,
    config_dir: Path,
    data_dir: Path,
    experiments_dir: Path,
    artifacts_dir: Path,
    proposal_path: Path | None,
) -> MiningRoundResult:
    run_dir = experiments_dir / run_id
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest["status"] == "complete":
        raise ValueError(f"mining run is already complete: {run_id}")
    _assert_locked_areas(repo_root, config_dir, data_dir, manifest)
    config, _ = _load_mining_config(config_dir / "mining.yaml")
    current_round = manifest.get("current_round")
    round_number = (
        int(current_round)
        if current_round is not None
        else int(manifest["completed_rounds"]) + 1
    )
    if round_number > int(manifest["requested_rounds"]):
        raise ValueError("all requested rounds are already complete")
    round_dir = run_dir / f"round_{round_number:04d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    manifest["current_round"] = round_number
    _atomic_json(manifest_path, manifest)

    stored_proposal = round_dir / "candidate" / "candidate.json"
    selected_proposal = proposal_path or (
        run_dir / "proposals" / f"round_{round_number:04d}.json"
    )
    if stored_proposal.is_file():
        proposal = CandidateProposal.model_validate(_read_json(stored_proposal))
    else:
        proposal = CandidateProposal.model_validate(_read_json(selected_proposal))
        _validate_proposal(proposal, run_id, round_number, config)
        _stage_proposal(round_dir, proposal)

    factor_id = proposal.metadata.factor_id
    decision_path = round_dir / "decision.json"
    if decision_path.is_file():
        decision = MiningDecision.model_validate(_read_json(decision_path))
        _record_round_decision(data_dir, config, round_dir, decision_path)
        _finalize_manifest_round(manifest_path, round_dir, decision)
        return MiningRoundResult(
            run_id=run_id,
            round_number=round_number,
            factor_id=factor_id,
            decision=decision.decision,
            round_dir=round_dir,
            decision_path=decision_path,
        )

    try:
        static_issues = inspect_factor_source(
            round_dir / "candidate" / f"{factor_id}.py",
            set(proposal.metadata.inputs),
        )
        if static_issues:
            issue_text = "; ".join(
                f"{issue.code}@{issue.line}" for issue in static_issues
            )
            raise ValueError(f"candidate failed static audit: {issue_text}")
        _publish_candidate(repo_root, proposal)
        evaluation = evaluate_factor(
            factor_id,
            config_dir,
            data_dir,
            artifacts_dir / "factors",
            snapshot_id=str(manifest["data_snapshot_id"]),
        )
        factor_result = _read_json(evaluation.result_path)
        _atomic_json(round_dir / "factor_result.json", factor_result)
        locked_after = _assert_locked_areas(repo_root, config_dir, data_dir, manifest)
        test_report = {
            "schema_version": 1,
            "factor_id": factor_id,
            "passed": bool(factor_result["leakage"]["passed"]),
            "static_issues": factor_result["leakage"]["static_issues"],
            "prefix_invariant": factor_result["leakage"]["prefix_invariant"],
            "future_perturbation_invariant": factor_result["leakage"][
                "future_perturbation_invariant"
            ],
            "locked_areas_unchanged": locked_after,
        }
        _atomic_json(round_dir / "test_report.json", test_report)
        decision = _decision_from_result(
            run_id, round_number, factor_result, evaluation.result_sha256
        )
    except Exception as error:
        error_result = {
            "schema_version": 1,
            "phase": 4,
            "status": "error",
            "factor_id": factor_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        _atomic_json(round_dir / "factor_result.json", error_result)
        _atomic_json(
            round_dir / "test_report.json",
            {
                "schema_version": 1,
                "factor_id": factor_id,
                "passed": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        decision = MiningDecision(
            schema_version=1,
            run_id=run_id,
            round_number=round_number,
            factor_id=factor_id,
            decision="ERROR",
            rationale=f"Round failed and was retained: {type(error).__name__}: {error}",
            passed_checks=[],
            failed_checks=["execution"],
            eligible_for_review=False,
            human_approval_required=True,
            factor_result_sha256=_sha256(round_dir / "factor_result.json"),
        )
    _atomic_json(decision_path, decision.model_dump(mode="json"))
    _record_round_decision(data_dir, config, round_dir, decision_path)
    _finalize_manifest_round(manifest_path, round_dir, decision)
    return MiningRoundResult(
        run_id=run_id,
        round_number=round_number,
        factor_id=factor_id,
        decision=decision.decision,
        round_dir=round_dir,
        decision_path=decision_path,
    )


def run_mining_loop(
    run_id: str,
    rounds: int,
    *,
    repo_root: Path,
    config_dir: Path,
    data_dir: Path,
    experiments_dir: Path,
    artifacts_dir: Path,
    proposals_dir: Path | None = None,
) -> list[MiningRoundResult]:
    initialize_mining_run(
        run_id,
        rounds,
        repo_root=repo_root,
        config_dir=config_dir,
        data_dir=data_dir,
        experiments_dir=experiments_dir,
    )
    results: list[MiningRoundResult] = []
    while True:
        manifest = _read_json(experiments_dir / run_id / "run_manifest.json")
        if int(manifest["completed_rounds"]) >= rounds:
            break
        next_round = int(manifest["completed_rounds"]) + 1
        proposal = (
            proposals_dir / f"round_{next_round:04d}.json"
            if proposals_dir is not None
            else None
        )
        results.append(
            run_mining_round(
                run_id,
                repo_root=repo_root,
                config_dir=config_dir,
                data_dir=data_dir,
                experiments_dir=experiments_dir,
                artifacts_dir=artifacts_dir,
                proposal_path=proposal,
            )
        )
    render_mining_report(run_id, experiments_dir)
    return results


def render_mining_report(run_id: str, experiments_dir: Path) -> Path:
    run_dir = experiments_dir / run_id
    manifest = _read_json(run_dir / "run_manifest.json")
    rows = []
    for item in manifest["rounds"]:
        rows.append(
            f"| {item['round_number']} | {item['factor_id']} | "
            f"{item['decision']} | `{item['factor_result_sha256']}` |"
        )
    report = f"""# Factor Mining Run {run_id}

> Engineering research only. ACCEPT means eligible for human review, not approved.

- Status: `{manifest["status"]}`
- Snapshot: `{manifest["data_snapshot_id"]}`
- Requested rounds: {manifest["requested_rounds"]}
- Completed rounds: {manifest["completed_rounds"]}
- Decisions: `{json.dumps(manifest["decision_counts"], sort_keys=True)}`
- Locked areas unchanged: `true`

| Round | Factor | Recommendation | Result SHA256 |
|---:|---|---|---|
{os.linesep.join(rows)}

The locked test set was not accessed. Failed and rejected rounds remain in this
directory.
"""
    path = run_dir / "final_report.md"
    _atomic_text(path, report)
    return path


def _stage_proposal(round_dir: Path, proposal: CandidateProposal) -> None:
    candidate_dir = round_dir / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        round_dir / "hypothesis.json",
        proposal.hypothesis.model_dump(mode="json"),
    )
    _atomic_json(candidate_dir / "candidate.json", proposal.model_dump(mode="json"))
    source = proposal.source_code.rstrip() + "\n"
    compile(source, f"<{proposal.metadata.factor_id}>", "exec")
    _atomic_text(candidate_dir / f"{proposal.metadata.factor_id}.py", source)
    metadata_text = yaml.safe_dump(
        proposal.metadata.model_dump(mode="json"),
        allow_unicode=True,
        sort_keys=False,
    )
    _atomic_text(candidate_dir / f"{proposal.metadata.factor_id}.yaml", metadata_text)


def _publish_candidate(repo_root: Path, proposal: CandidateProposal) -> None:
    candidate_dir = repo_root / "src" / "alpha_lab" / "factors" / "candidates"
    factor_id = proposal.metadata.factor_id
    source = (proposal.source_code.rstrip() + "\n").encode()
    metadata = yaml.safe_dump(
        proposal.metadata.model_dump(mode="json"),
        allow_unicode=True,
        sort_keys=False,
    ).encode()
    _write_immutable(candidate_dir / f"{factor_id}.py", source)
    _write_immutable(candidate_dir / f"{factor_id}.yaml", metadata)


def _decision_from_result(
    run_id: str,
    round_number: int,
    result: dict[str, Any],
    result_sha256: str,
) -> MiningDecision:
    checks = {
        str(key): bool(value) for key, value in result["promotion_checks"].items()
    }
    passed = sorted(key for key, value in checks.items() if value)
    failed = sorted(key for key, value in checks.items() if not value)
    eligible = bool(result["eligible_for_review"])
    decision: Literal["ACCEPT", "REJECT"] = "ACCEPT" if eligible else "REJECT"
    rationale = (
        "All fixed promotion checks passed; recommend human review only."
        if eligible
        else f"Fixed promotion checks failed: {', '.join(failed)}."
    )
    return MiningDecision(
        schema_version=1,
        run_id=run_id,
        round_number=round_number,
        factor_id=str(result["factor"]["factor_id"]),
        decision=decision,
        rationale=rationale,
        passed_checks=passed,
        failed_checks=failed,
        eligible_for_review=eligible,
        human_approval_required=True,
        factor_result_sha256=result_sha256,
    )


def _finalize_manifest_round(
    manifest_path: Path, round_dir: Path, decision: MiningDecision
) -> None:
    manifest = _read_json(manifest_path)
    if any(
        int(item["round_number"]) == decision.round_number
        for item in manifest["rounds"]
    ):
        return
    factor_result_path = round_dir / "factor_result.json"
    manifest["rounds"].append(
        {
            "round_number": decision.round_number,
            "factor_id": decision.factor_id,
            "decision": decision.decision,
            "factor_result_sha256": _sha256(factor_result_path),
            "decision_sha256": _sha256(round_dir / "decision.json"),
        }
    )
    manifest["completed_rounds"] = len(manifest["rounds"])
    manifest["current_round"] = None
    manifest["decision_counts"][decision.decision] += 1
    if int(manifest["completed_rounds"]) >= int(manifest["requested_rounds"]):
        manifest["status"] = "complete"
    _atomic_json(manifest_path, manifest)


def _record_round_decision(
    data_dir: Path,
    config: MiningConfig,
    round_dir: Path,
    decision_path: Path,
) -> None:
    factor_result = _read_json(round_dir / "factor_result.json")
    experiment_id = factor_result.get("run_id")
    if experiment_id is None:
        return
    from alpha_lab.database.catalog import record_mining_decision

    record_mining_decision(
        data_dir / "metadata.duckdb",
        str(experiment_id),
        config.policy_id,
        decision_path,
    )


def _validate_proposal(
    proposal: CandidateProposal,
    run_id: str,
    round_number: int,
    config: MiningConfig,
) -> None:
    hypothesis = proposal.hypothesis
    if hypothesis.run_id != run_id or hypothesis.round_number != round_number:
        raise ValueError("proposal run_id/round_number does not match current round")
    number = int(hypothesis.factor_id.removeprefix("F"))
    if not config.candidate_id_minimum <= number <= config.candidate_id_maximum:
        raise ValueError("candidate factor ID is outside the mining range")
    if hypothesis.lookback > config.maximum_lookback:
        raise ValueError("candidate lookback exceeds mining policy")
    if hypothesis.primary_change not in config.allowed_primary_changes:
        raise ValueError("primary change is not allowed by mining policy")
    if len(hypothesis.changed_variable.split(",")) != 1:
        raise ValueError("each round must change exactly one primary variable")


def _locked_area_hashes(
    repo_root: Path, config_dir: Path, data_dir: Path, snapshot_id: str
) -> dict[str, str]:
    paths = [
        config_dir / "splits.yaml",
        config_dir / "costs.yaml",
        config_dir / "factor_evaluation.yaml",
        data_dir / "manifests" / snapshot_id / "manifest.json",
        data_dir / "manifests" / snapshot_id / "quality_report.json",
    ]
    paths.extend(sorted((repo_root / "src" / "alpha_lab" / "evaluation").glob("*.py")))
    paths.extend(sorted((repo_root / "tests" / "leakage").glob("*.py")))
    return {
        path.resolve().relative_to(repo_root.resolve()).as_posix(): _sha256(path)
        for path in paths
        if path.is_file()
    }


def _assert_locked_areas(
    repo_root: Path,
    config_dir: Path,
    data_dir: Path,
    manifest: dict[str, Any],
) -> bool:
    current = _locked_area_hashes(
        repo_root, config_dir, data_dir, str(manifest["data_snapshot_id"])
    )
    expected = manifest["locked_area_hashes"]
    if current != expected:
        changed = sorted(set(current) | set(expected))
        changed = [key for key in changed if current.get(key) != expected.get(key)]
        raise RuntimeError(f"locked areas changed during mining: {changed}")
    return True


def _research_brief(
    run_id: str,
    snapshot_id: str,
    rounds: int,
    registry: FactorRegistry,
) -> str:
    factors = "\n".join(
        f"- {item.metadata.factor_id}: {item.metadata.name} "
        f"({item.metadata.family}, {item.metadata.status})"
        for item in registry.all()
    )
    return f"""# Research Brief: {run_id}

- Snapshot: `{snapshot_id}`
- Budget: {rounds} rounds
- Test access: prohibited
- One primary change per round
- ACCEPT means human-review recommendation only

## Current factor library

{factors}
"""


@contextmanager
def _run_lock(run_dir: Path) -> Iterator[None]:
    lock_dir = run_dir / ".round.lockdir"
    deadline = time.monotonic() + 60.0
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for mining lock: {lock_dir}"
                ) from None
            time.sleep(0.1)
    try:
        yield
    finally:
        lock_dir.rmdir()


def _load_mining_config(path: Path) -> tuple[MiningConfig, str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = MiningConfig.model_validate(raw)
    return config, _canonical_hash(raw)


def _validate_run_id(run_id: str) -> None:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run ID must contain only letters, digits, dot, dash, underscore"
        )


def _latest_snapshot(data_dir: Path) -> str:
    path = data_dir / "state" / "latest_snapshot.txt"
    if not path.is_file():
        raise ValueError("no latest snapshot; run make data-bootstrap first")
    return path.read_text(encoding="utf-8").strip()


def _git_identity(repo_root: Path) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"JSON file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    content = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n"
    )
    _atomic_text(path, content)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"candidate artifact is immutable: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
