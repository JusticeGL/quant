from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_lab.baseline.analysis import analyze_signals
from alpha_lab.baseline.backtest import run_topk_backtest
from alpha_lab.baseline.config import Phase2Config, load_phase2_config
from alpha_lab.baseline.features import alpha158_definition, load_alpha158_dataset
from alpha_lab.baseline.model import fit_and_predict
from alpha_lab.baseline.report import render_reports


@dataclass(frozen=True)
class BaselineResult:
    run_id: str
    output_dir: Path
    manifest_path: Path
    markdown_report_path: Path
    html_report_path: Path
    reproducibility_sha256: str


def run_baseline(
    config_dir: Path,
    data_dir: Path,
    output_root: Path,
    *,
    snapshot_id: str | None = None,
) -> BaselineResult:
    config = load_phase2_config(config_dir)
    selected_snapshot = snapshot_id or _latest_snapshot(data_dir)
    snapshot_manifest = _read_json(
        data_dir / "manifests" / selected_snapshot / "manifest.json"
    )
    qlib_manifest = _read_json(
        data_dir / "qlib" / selected_snapshot / "export_manifest.json"
    )
    if snapshot_manifest.get("snapshot_id") != selected_snapshot:
        raise ValueError("snapshot manifest identity mismatch")
    if qlib_manifest.get("snapshot_id") != selected_snapshot:
        raise ValueError("Qlib export snapshot identity mismatch")
    _validate_protocol(config, snapshot_manifest)

    git = _git_identity()
    git_commit = str(git["commit"])
    run_id = (
        f"baseline-{selected_snapshot.removeprefix('p1-')[:10]}-"
        f"{config.config_sha256[:10]}-{git_commit[:10]}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=output_root))
    try:
        result = _execute(
            config,
            data_dir,
            selected_snapshot,
            snapshot_manifest,
            qlib_manifest,
            git,
            run_id,
            temporary,
        )
        destination = output_root / run_id
        if destination.exists():
            existing = _read_json(destination / "run_manifest.json")
            if existing.get("reproducibility_sha256") != result.reproducibility_sha256:
                raise RuntimeError(
                    f"immutable run {run_id} produced a different reproducibility hash"
                )
        else:
            os.replace(temporary, destination)
        finalized = BaselineResult(
            run_id=run_id,
            output_dir=destination,
            manifest_path=destination / "run_manifest.json",
            markdown_report_path=destination / "baseline_report.md",
            html_report_path=destination / "baseline_report.html",
            reproducibility_sha256=result.reproducibility_sha256,
        )
        from alpha_lab.database.catalog import record_baseline_run

        record_baseline_run(
            data_dir / "metadata.duckdb",
            config_dir,
            data_dir,
            finalized.manifest_path,
        )
        return finalized
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _execute(
    config: Phase2Config,
    data_dir: Path,
    snapshot_id: str,
    snapshot_manifest: dict[str, Any],
    qlib_manifest: dict[str, Any],
    git: dict[str, object],
    run_id: str,
    output_dir: Path,
) -> BaselineResult:
    split = config.splits
    access_end = split.test.start - timedelta(days=1)
    dataset = load_alpha158_dataset(
        data_dir / "qlib" / snapshot_id,
        start_time=split.train.start.isoformat(),
        end_time=split.validation.end.isoformat(),
        label_expression=config.baseline.label.expression,
    )
    _, feature_names = alpha158_definition()
    market = pd.read_parquet(data_dir / "silver" / snapshot_id / "daily.parquet")
    market_dates = sorted(pd.to_datetime(market["trade_date"]).dt.date.unique())
    outcome_steps = (
        config.baseline.label.execution_delay_days + config.baseline.label.holding_days
    )
    train = _select_observations(
        dataset,
        start=split.train.start,
        signal_end=split.train.end,
        outcome_end=split.train.end,
        market_dates=market_dates,
        outcome_steps=outcome_steps,
    )
    validation = _select_observations(
        dataset,
        start=split.validation.start,
        signal_end=split.validation.end,
        outcome_end=access_end,
        market_dates=market_dates,
        outcome_steps=outcome_steps,
    )
    if train[feature_names].notna().sum().sum() == 0:
        raise ValueError("Alpha158 training features are entirely missing")

    model, scores = fit_and_predict(
        train,
        validation,
        feature_names,
        config.baseline.model,
        config.baseline.random_seed,
    )
    predictions = validation[["datetime", "instrument", "LABEL"]].rename(
        columns={"LABEL": "label"}
    )
    predictions = predictions.assign(score=scores)[
        ["datetime", "instrument", "score", "label"]
    ]
    signal_analysis = analyze_signals(predictions, config.baseline.annualization_days)

    backtest = run_topk_backtest(
        predictions,
        market,
        strategy=config.baseline.strategy,
        costs=config.costs,
        annualization_days=config.baseline.annualization_days,
        allowed_end=access_end,
    )

    predictions_path = output_dir / "predictions.parquet"
    daily_path = output_dir / "backtest_daily.parquet"
    trades_path = output_dir / "trades.parquet"
    model_path = output_dir / "lightgbm_model.txt"
    predictions.to_parquet(predictions_path, index=False)
    backtest.daily.to_parquet(daily_path, index=False)
    backtest.trades.to_parquet(trades_path, index=False)
    if model.booster_ is None:
        raise RuntimeError("LightGBM model has no fitted booster")
    model.booster_.save_model(str(model_path))

    reproducibility_payload = {
        "data_snapshot_id": snapshot_id,
        "qlib_content_sha256": qlib_manifest["content_sha256"],
        "config_sha256": config.config_sha256,
        "random_seed": config.baseline.random_seed,
        "predictions": [
            {
                "datetime": row.datetime.date().isoformat(),
                "instrument": row.instrument,
                "score": _rounded(row.score),
                "label": _rounded(row.label),
            }
            for row in predictions.itertuples(index=False)
        ],
        "signal_summary": {
            key: _rounded(value)
            for key, value in signal_analysis.items()
            if key != "daily"
        },
        "backtest_metrics": {
            key: _rounded(value) for key, value in backtest.metrics.items()
        },
        "constraint_counts": backtest.constraints,
    }
    reproducibility_sha256 = _canonical_hash(reproducibility_payload)
    artifacts = {
        path.name: {"sha256": _file_hash(path), "bytes": path.stat().st_size}
        for path in [predictions_path, daily_path, trades_path, model_path]
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "phase": 2,
        "run_id": run_id,
        "experiment_name": config.baseline.experiment_name,
        "status": "success",
        "engineering_only": True,
        "data_snapshot_id": snapshot_id,
        "data_research_eligible": snapshot_manifest["universe"]["research_eligible"],
        "qlib_content_sha256": qlib_manifest["content_sha256"],
        "config_sha256": config.config_sha256,
        "split_policy_id": split.policy_id,
        "split_policy_sha256": config.split_sha256,
        "cost_policy_id": config.costs.policy_id,
        "cost_policy_sha256": config.cost_sha256,
        "git": git,
        "random_seed": config.baseline.random_seed,
        "feature_set": config.baseline.feature_set,
        "feature_count": len(feature_names),
        "label": config.baseline.label.model_dump(mode="json"),
        "model": config.baseline.model.model_dump(mode="json"),
        "strategy": config.baseline.strategy.model_dump(mode="json"),
        "splits": {
            "train": split.train.model_dump(mode="json"),
            "validation": split.validation.model_dump(mode="json"),
            "test": {"locked": True, "accessed": False, "reported": False},
        },
        "row_counts": {"train": len(train), "validation": len(validation)},
        "signal_analysis": signal_analysis,
        "backtest": {
            "metrics": backtest.metrics,
            "constraints": backtest.constraints,
        },
        "reproducibility_sha256": reproducibility_sha256,
        "limitations": [
            "The fixed 10-stock sample is survivorship-biased and marked "
            "research_eligible=false.",
            "Only 2024 H1 is available; this is an engineering split, not the "
            "long-horizon protocol.",
            "Adjustment factors, listing dates, and delisting dates are absent.",
            "Price-limit flags are absent and are conservatively inferred from "
            "open versus prior close.",
            "Commission and minimum commission are explicit engineering assumptions.",
            "The locked test range was not loaded, scored, evaluated, or reported.",
            "Results are research/education artifacts and do not indicate future "
            "performance.",
        ],
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    markdown_path, html_path = render_reports(manifest, output_dir)
    return BaselineResult(
        run_id=run_id,
        output_dir=output_dir,
        manifest_path=manifest_path,
        markdown_report_path=markdown_path,
        html_report_path=html_path,
        reproducibility_sha256=reproducibility_sha256,
    )


def _validate_protocol(config: Phase2Config, snapshot_manifest: dict[str, Any]) -> None:
    summary = snapshot_manifest["summary"]
    available_start = pd.Timestamp(summary["date_start"]).date()
    available_end = pd.Timestamp(summary["date_end"]).date()
    if config.splits.train.start < available_start:
        raise ValueError("train starts before the selected snapshot")
    if config.splits.test.end > available_end:
        raise ValueError("locked test ends after the selected snapshot")
    if config.baseline.strategy.top_k >= int(summary["instrument_count"]):
        raise ValueError("top_k must be smaller than the available instrument count")
    if not config.baseline.engineering_only or not config.splits.engineering_only:
        raise ValueError("the Phase 1 snapshot may only run an engineering baseline")


def _select_observations(
    dataset: pd.DataFrame,
    *,
    start: date,
    signal_end: date,
    outcome_end: date,
    market_dates: list[date],
    outcome_steps: int,
) -> pd.DataFrame:
    if outcome_steps < 1:
        raise ValueError("outcome_steps must be positive")
    position = {value: index for index, value in enumerate(market_dates)}
    eligible_dates = {
        value
        for value in market_dates
        if start <= value <= signal_end
        and position[value] + outcome_steps < len(market_dates)
        and market_dates[position[value] + outcome_steps] <= outcome_end
    }
    selected = dataset.loc[dataset["datetime"].dt.date.isin(eligible_dates)]
    return selected.dropna(subset=["LABEL"]).copy()


def _latest_snapshot(data_dir: Path) -> str:
    path = data_dir / "state" / "latest_snapshot.txt"
    if not path.is_file():
        raise ValueError("no latest snapshot; run make data-bootstrap first")
    return path.read_text(encoding="utf-8").strip()


def _git_identity() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required artifact does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rounded(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return round(number, 12) if np.isfinite(number) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value
