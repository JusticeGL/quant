from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from alpha_lab.baseline.backtest import BacktestResult
from alpha_lab.robustness import report
from alpha_lab.robustness.approval import _replay_pretest_evidence
from alpha_lab.robustness.report import evaluate_frozen_candidate

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "relative",
    [
        "large/wf_2025/predictions.parquet",
        "large/wf_2025/cost_1.0x/nav.parquet",
        "large/wf_2025/cost_1.0x/trades.parquet",
    ],
)
def test_large_artifacts_are_idempotent_and_refuse_conflicts(
    tmp_path: Path, relative: str
) -> None:
    path = tmp_path / relative
    original = pd.DataFrame({"value": [1.0]})
    report._write_parquet_immutable(path, original)
    first = path.read_bytes()

    report._write_parquet_immutable(path, original)
    assert path.read_bytes() == first

    with pytest.raises(RuntimeError, match="immutable robustness artifact differs"):
        report._write_parquet_immutable(path, pd.DataFrame({"value": [2.0]}))
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


@pytest.mark.parametrize(
    "name",
    [
        "walk_forward.json",
        "cost_sensitivity.json",
        "exposure_report.json",
        "robustness_report.md",
    ],
)
def test_small_artifacts_are_idempotent_and_refuse_conflicts(
    tmp_path: Path, name: str
) -> None:
    path = tmp_path / name
    report._write_immutable(path, b"first\n")
    report._write_immutable(path, b"first\n")

    with pytest.raises(RuntimeError, match="immutable robustness artifact differs"):
        report._write_immutable(path, b"second\n")
    assert path.read_bytes() == b"first\n"
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_pipeline_computes_once_uses_fold_bounds_and_is_repeatable(
    tmp_path: Path, monkeypatch: Any
) -> None:
    experiments = tmp_path / "experiments"
    freeze_dir = experiments / "phase6" / "freeze-fixture"
    freeze_dir.mkdir(parents=True)
    freeze_path = freeze_dir / "freeze.json"
    freeze_path.write_text("{}\n", encoding="utf-8")
    market = _market()
    cap, industry = _exposures(market)
    reads: list[date] = []
    computes = 0
    allowed_ends: list[date] = []

    monkeypatch.setattr(
        report,
        "validate_freeze",
        lambda *_: {
            "freeze_id": "freeze-fixture",
            "freeze_sha256": "a" * 64,
            "factor": {
                "factor_id": "F1002",
                "source_sha256": "b" * 64,
                "metadata_sha256": "c" * 64,
            },
            "snapshots": {
                "phase5": {
                    "snapshot_id": "p5-fixture",
                    "manifest_sha256": "d" * 64,
                },
                "exposure": {
                    "snapshot_id": "p6x-fixture",
                    "manifest_sha256": "e" * 64,
                },
            },
            "policies": {"costs": {"sha256": "f" * 64}},
        },
    )

    def market_reader(*_: object) -> pd.DataFrame:
        end_before = _[-1]
        assert isinstance(end_before, date)
        reads.append(end_before)
        assert end_before == date(2026, 1, 1)
        return market.copy()

    def exposure_reader(*_: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        end_before = _[-1]
        assert isinstance(end_before, date)
        reads.append(end_before)
        assert end_before == date(2026, 1, 1)
        return cap.copy(), industry.copy()

    real_validate = report.validate_factor_output

    def compute_once(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal computes
        computes += 1
        return real_validate(*args, **kwargs)  # type: ignore[arg-type]

    def fake_backtest(*args: object, **kwargs: object) -> BacktestResult:
        allowed_end = kwargs["allowed_end"]
        assert isinstance(allowed_end, date)
        allowed_ends.append(allowed_end)
        predictions = args[0]
        assert pd.to_datetime(predictions["datetime"]).dt.date.max() <= allowed_end
        daily = pd.DataFrame([{"trade_date": allowed_end, "nav": 1.01, "fees": 0.1}])
        trades = pd.DataFrame(
            columns=["trade_date", "instrument", "side", "shares", "price", "fee"]
        )
        return BacktestResult(
            daily=daily,
            trades=trades,
            metrics={"total_return": 0.01},
            constraints={},
        )

    monkeypatch.setattr(report, "read_pretest_market", market_reader)
    monkeypatch.setattr(report, "read_pretest_exposures", exposure_reader)
    monkeypatch.setattr(report, "validate_factor_output", compute_once)
    monkeypatch.setattr(report, "run_topk_backtest", fake_backtest)

    first = evaluate_frozen_candidate(
        freeze_path, ROOT / "config", tmp_path / "data", experiments
    )
    original = {
        name: (freeze_dir / name).read_bytes()
        for name in (
            "walk_forward.json",
            "cost_sensitivity.json",
            "exposure_report.json",
            "robustness_report.md",
        )
    }
    second = evaluate_frozen_candidate(
        freeze_path, ROOT / "config", tmp_path / "data", experiments
    )
    shutil.copytree(ROOT / "config", tmp_path / "config")
    shutil.copytree(
        ROOT / "src/alpha_lab/factors/candidates",
        tmp_path / "src/alpha_lab/factors/candidates",
    )
    _replay_pretest_evidence(freeze_path, tmp_path)

    assert computes == 3  # exactly once per complete evaluation invocation/replay
    assert len(allowed_ends) == 60  # five folds x four costs x three invocations
    assert sorted(set(allowed_ends)) == [
        date(year, 12, 31) for year in range(2021, 2026)
    ]
    assert reads == [date(2026, 1, 1)] * 6
    assert not list(experiments.glob(".pretest-replay-*"))
    assert first.report_sha256 == second.report_sha256
    assert all(
        (freeze_dir / name).read_bytes() == content
        for name, content in original.items()
    )
    walk = json.loads((freeze_dir / "walk_forward.json").read_text())
    assert walk["test_accessed"] is False
    assert walk["dependencies"] == {
        "cost_policy_sha256": "f" * 64,
        "exposure_manifest_sha256": "e" * 64,
        "factor_metadata_sha256": "c" * 64,
        "factor_source_sha256": "b" * 64,
        "phase5_manifest_sha256": "d" * 64,
    }
    assert walk["orientation"] == {
        "candidate_direction": -1,
        "direction_consistency_source": "oriented_mean_rank_ic_positive",
        "score_formula": "score=standardize(winsorize(value*direction))",
    }


def test_pipeline_refuses_conflicting_report_bytes(
    tmp_path: Path, monkeypatch: Any
) -> None:
    experiments = tmp_path / "experiments"
    freeze_dir = experiments / "phase6" / "freeze-fixture"
    freeze_dir.mkdir(parents=True)
    freeze_path = freeze_dir / "freeze.json"
    freeze_path.write_text("{}\n", encoding="utf-8")
    (freeze_dir / "walk_forward.json").write_text("conflict\n", encoding="utf-8")
    monkeypatch.setattr(
        report,
        "validate_freeze",
        lambda *_: {
            "freeze_id": "freeze-fixture",
            "freeze_sha256": "a" * 64,
            "factor": {
                "factor_id": "F1002",
                "source_sha256": "b" * 64,
                "metadata_sha256": "c" * 64,
            },
            "snapshots": {
                "phase5": {
                    "snapshot_id": "p5-fixture",
                    "manifest_sha256": "d" * 64,
                },
                "exposure": {
                    "snapshot_id": "p6x-fixture",
                    "manifest_sha256": "e" * 64,
                },
            },
            "policies": {"costs": {"sha256": "f" * 64}},
        },
    )
    market = _market()
    cap, industry = _exposures(market)
    monkeypatch.setattr(report, "read_pretest_market", lambda *_: market.copy())
    monkeypatch.setattr(
        report, "read_pretest_exposures", lambda *_: (cap.copy(), industry.copy())
    )
    monkeypatch.setattr(
        report,
        "run_topk_backtest",
        lambda *_, **kwargs: BacktestResult(
            daily=pd.DataFrame([{"trade_date": kwargs["allowed_end"], "nav": 1.0}]),
            trades=pd.DataFrame(),
            metrics={"total_return": 0.01},
            constraints={},
        ),
    )

    try:
        evaluate_frozen_candidate(
            freeze_path, ROOT / "config", tmp_path / "data", experiments
        )
    except RuntimeError as error:
        assert "immutable robustness artifact differs" in str(error)
    else:
        raise AssertionError("conflicting report must be rejected")


@pytest.mark.parametrize("factor_id", ["F1002", "F1003"])
def test_real_candidates_and_backtester_cover_all_locked_cost_folds(
    tmp_path: Path, monkeypatch: Any, factor_id: str
) -> None:
    experiments = tmp_path / "experiments"
    freeze_dir = experiments / "phase6" / f"freeze-{factor_id.lower()}"
    freeze_dir.mkdir(parents=True)
    freeze_path = freeze_dir / "freeze.json"
    freeze_path.write_text("{}\n", encoding="utf-8")
    market = _market()
    cap, industry = _exposures(market)
    candidates = ROOT / "src" / "alpha_lab" / "factors" / "candidates"

    monkeypatch.setattr(
        report,
        "validate_freeze",
        lambda *_: {
            "freeze_id": f"freeze-{factor_id.lower()}",
            "freeze_sha256": "a" * 64,
            "factor": {
                "factor_id": factor_id,
                "source_sha256": hashlib.sha256(
                    (candidates / f"{factor_id}.py").read_bytes()
                ).hexdigest(),
                "metadata_sha256": hashlib.sha256(
                    (candidates / f"{factor_id}.yaml").read_bytes()
                ).hexdigest(),
            },
            "snapshots": {
                "phase5": {
                    "snapshot_id": "p5-fixture",
                    "manifest_sha256": "d" * 64,
                },
                "exposure": {
                    "snapshot_id": "p6x-fixture",
                    "manifest_sha256": "e" * 64,
                },
            },
            "policies": {"costs": {"sha256": "f" * 64}},
        },
    )
    monkeypatch.setattr(
        report,
        "read_pretest_market",
        lambda _data, _snapshot, end_before: (
            market.copy()
            if end_before == date(2026, 1, 1)
            else (_ for _ in ()).throw(AssertionError("wrong boundary"))
        ),
    )
    monkeypatch.setattr(
        report,
        "read_pretest_exposures",
        lambda _data, _snapshot, end_before: (
            (cap.copy(), industry.copy())
            if end_before == date(2026, 1, 1)
            else (_ for _ in ()).throw(AssertionError("wrong boundary"))
        ),
    )

    evaluate_frozen_candidate(
        freeze_path, ROOT / "config", tmp_path / "data", experiments
    )

    costs = json.loads((freeze_dir / "cost_sensitivity.json").read_text())
    assert set(costs["scenarios"]) == {"0.5", "1.0", "1.5", "2.0"}
    for scenario in costs["scenarios"].values():
        assert [fold["fold_id"] for fold in scenario["folds"]] == [
            f"wf_{year}" for year in range(2021, 2026)
        ]
    for year in range(2021, 2026):
        predictions = pd.read_parquet(
            freeze_dir / "large" / f"wf_{year}" / "predictions.parquet"
        )
        assert predictions["label"].notna().all()
        assert predictions["entry_date"].notna().all()
        assert predictions["exit_date"].notna().all()
        assert pd.to_datetime(predictions["exit_date"]).dt.date.max() <= date(
            year, 12, 31
        )
        trades = pd.read_parquet(
            freeze_dir / "large" / f"wf_{year}" / "cost_1.0x" / "trades.parquet"
        )
        if not trades.empty:
            assert (
                pd.to_datetime(trades["trade_date"]).dt.date.max()
                <= pd.to_datetime(predictions["exit_date"]).dt.date.max()
            )


def _market() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in range(2020, 2026):
        dates = pd.bdate_range(f"{year}-01-04", periods=15)
        for number, instrument in enumerate(("SH600000", "SH600001", "SZ000001")):
            for index, trade_date in enumerate(dates):
                price = 10.0 + number + index * 0.01
                rows.append(
                    {
                        "trade_date": trade_date,
                        "instrument": instrument,
                        "open": price,
                        "high": price * 1.01,
                        "low": price * 0.99,
                        "close": price,
                        "volume": float(10_000 + index * (number + 1)),
                        "amount": price * 10_000,
                        "adj_factor": 1.0,
                        "suspend": False,
                        "limit_up": False,
                        "limit_down": False,
                        "is_st": False,
                        "list_date": pd.NaT,
                        "delist_date": pd.NaT,
                    }
                )
    return pd.DataFrame(rows)


def _exposures(market: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = {
        "SH600000": "CN:SSE:600000",
        "SH600001": "CN:SSE:600001",
        "SZ000001": "CN:SZSE:000001",
    }
    cap = market[["trade_date", "instrument"]].copy()
    cap["security_id"] = cap.pop("instrument").map(mapping)
    cap["total_market_cap_cny"] = cap["security_id"].map(
        {
            value: float(1_000_000 * (index + 1))
            for index, value in enumerate(mapping.values())
        }
    )
    cap["known_at"] = pd.to_datetime(cap["trade_date"], utc=True)
    industry = pd.DataFrame(
        {
            "industry_id": ["A", "A", "A"],
            "security_id": list(mapping.values()),
            "effective_from": pd.to_datetime(["2020-01-01"] * 3),
            "effective_to": [pd.NaT] * 3,
            "known_at": pd.to_datetime(["2020-01-01T00:00:00Z"] * 3),
        }
    )
    return cap, industry
