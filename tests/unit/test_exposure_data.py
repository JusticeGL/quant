from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.robustness.config import load_robustness_config
from alpha_lab.robustness.contracts import ExposureTables
from alpha_lab.robustness.exposure_data import (
    _load_phase5_tables,
    industry_as_of,
    normalize_industry_definition,
    normalize_industry_membership,
    normalize_market_cap,
    validate_exposure_tables,
)

ROOT = Path(__file__).resolve().parents[2]


def test_daily_basic_converts_ten_thousand_cny_to_cny() -> None:
    raw = pd.DataFrame(
        [
            {
                "ts_code": "600000.SH",
                "trade_date": "20210104",
                "total_mv": 123.4,
                "circ_mv": 100.0,
            }
        ]
    )

    result = normalize_market_cap(raw)

    assert result.loc[0, "total_market_cap_cny"] == 1_234_000.0
    assert result.loc[0, "float_market_cap_cny"] == 1_000_000.0
    assert result.loc[0, "known_at"] == pd.Timestamp("2021-01-04", tz="UTC")


def test_market_cap_requires_fields_positive_values_and_unique_keys() -> None:
    with pytest.raises(ValueError, match="required fields"):
        normalize_market_cap(pd.DataFrame([{"ts_code": "600000.SH"}]))
    invalid = pd.DataFrame(
        [["600000.SH", "20210104", 0, 1]],
        columns=["ts_code", "trade_date", "total_mv", "circ_mv"],
    )
    with pytest.raises(ValueError, match="positive"):
        normalize_market_cap(invalid)
    duplicate = pd.concat([invalid.assign(total_mv=1), invalid.assign(total_mv=2)])
    with pytest.raises(ValueError, match="duplicate"):
        normalize_market_cap(duplicate)


def test_market_cap_rejects_response_for_another_security() -> None:
    raw = pd.DataFrame(
        [["600001.SH", "20210104", 1, 1]],
        columns=["ts_code", "trade_date", "total_mv", "circ_mv"],
    )
    with pytest.raises(ValueError, match="unexpected security"):
        normalize_market_cap(raw, expected_ts_code="600000.SH")


def test_industry_definition_filters_to_sw2021_level_one() -> None:
    raw = pd.DataFrame(
        [
            ["801010.SI", "农林牧渔", "L1", "110000", "SW2021"],
            ["801020.SI", "采掘", "L1", "210000", "SW"],
            ["801030.SI", "二级", "L2", "110100", "SW2021"],
        ],
        columns=["index_code", "industry_name", "level", "industry_code", "src"],
    )

    result = normalize_industry_definition(raw)

    assert result["industry_id"].tolist() == ["CN:SW2021:801010.SI"]
    assert result["classification_standard"].tolist() == ["SW2021"]


def test_industry_membership_uses_effective_date_as_known_date() -> None:
    raw = _membership_raw()

    result = normalize_industry_membership(raw, {"801010.SI"})

    assert result.loc[0, "known_at"] == pd.Timestamp("2021-01-05", tz="UTC")
    assert result.loc[0, "known_at_source"] == "effective_date_fallback"


def test_industry_asof_uses_effective_and_known_dates() -> None:
    intervals = normalize_industry_membership(_membership_raw(), {"801010.SI"})

    selected = industry_as_of(intervals, date(2021, 1, 4))

    assert "CN:SSE:600000" not in set(selected["security_id"])


def test_industry_asof_does_not_backfill_current_industry() -> None:
    intervals = normalize_industry_membership(_membership_raw(), {"801010.SI"})

    assert industry_as_of(intervals, date(2020, 12, 31)).empty


def test_membership_rejects_unknown_dictionary_codes_and_overlaps() -> None:
    with pytest.raises(ValueError, match="unknown SW2021"):
        normalize_industry_membership(_membership_raw(), {"801020.SI"})
    overlapping = pd.concat(
        [
            _membership_raw(),
            _membership_raw().assign(in_date="20210110", out_date="20210120"),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="overlap"):
        normalize_industry_membership(overlapping, {"801010.SI"})


def test_membership_rejects_response_for_another_requested_industry() -> None:
    with pytest.raises(ValueError, match="unexpected industry"):
        normalize_industry_membership(
            _membership_raw(),
            {"801010.SI"},
            expected_l1_code="801020.SI",
        )


def test_quality_rejects_unknown_phase5_security() -> None:
    tables = ExposureTables(
        market_cap=normalize_market_cap(
            pd.DataFrame(
                [["600001.SH", "20210104", 1, 1]],
                columns=["ts_code", "trade_date", "total_mv", "circ_mv"],
            )
        ),
        industry_definition=normalize_industry_definition(_definition_raw()),
        industry_membership=normalize_industry_membership(
            _membership_raw(), {"801010.SI"}
        ),
    )

    report = validate_exposure_tables(tables, {"CN:SSE:600000"})

    assert report["status"] == "error"
    assert report["checks"]["unknown_security_reference"]["count"] == 1


def test_quality_rejects_empty_exposure_tables() -> None:
    empty = ExposureTables(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    report = validate_exposure_tables(empty, {"CN:SSE:600000"})

    assert report["status"] == "error"
    assert report["checks"]["empty_required_table"]["count"] == 3


def test_provider_row_limit_is_rejected() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["600000.SH"] * 6000,
            "trade_date": pd.date_range("2000-01-01", periods=6000).strftime("%Y%m%d"),
            "total_mv": [1.0] * 6000,
            "circ_mv": [1.0] * 6000,
        }
    )
    with pytest.raises(RuntimeError, match="row limit"):
        normalize_market_cap(raw, row_limit=6000)


def test_phase5_membership_checksum_is_verified(tmp_path: Path) -> None:
    config, _ = load_robustness_config(ROOT / "config" / "robustness.yaml")
    snapshot = tmp_path / "research" / config.phase5_snapshot_id
    snapshot.mkdir(parents=True)
    security = snapshot / "security_master.parquet"
    membership = snapshot / "index_membership.parquet"
    pd.DataFrame([{"security_id": "CN:SSE:600000"}]).to_parquet(security)
    pd.DataFrame([{"security_id": "CN:SSE:600000"}]).to_parquet(membership)
    manifest_dir = tmp_path / "manifests" / config.phase5_snapshot_id
    manifest_dir.mkdir(parents=True)
    manifest = {
        "snapshot_id": config.phase5_snapshot_id,
        "artifacts": [
            {
                "name": path.name,
                "path": path.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in (security, membership)
        ],
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest))
    membership.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checksum"):
        _load_phase5_tables(tmp_path, config)


def _definition_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [["801010.SI", "农林牧渔", "L1", "110000", "SW2021"]],
        columns=["index_code", "industry_name", "level", "industry_code", "src"],
    )


def _membership_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "l1_code": "801010.SI",
                "l1_name": "农林牧渔",
                "l2_code": "801011.SI",
                "l2_name": "种植业",
                "l3_code": "850111.SI",
                "l3_name": "种子",
                "ts_code": "600000.SH",
                "name": "浦发银行",
                "in_date": "20210105",
                "out_date": "20210131",
                "is_new": "N",
            }
        ]
    )
