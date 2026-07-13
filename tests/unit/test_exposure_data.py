from __future__ import annotations

import hashlib
import json
import threading
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.research_data.provider import TushareArtifact, TushareQueryResult
from alpha_lab.robustness.config import load_robustness_config
from alpha_lab.robustness.contracts import ExposureTables
from alpha_lab.robustness.exposure_data import (
    DAILY_BASIC_FIELDS,
    INDEX_CLASSIFY_FIELDS,
    INDUSTRY_MEMBER_FIELDS,
    _load_phase5_tables,
    acquire_exposure_tables,
    industry_as_of,
    normalize_industry_definition,
    normalize_industry_membership,
    normalize_market_cap,
    probe_exposure_capabilities,
    validate_exposure_tables,
)

ROOT = Path(__file__).resolve().parents[2]
_COVERAGE_DATES = (
    "20200102",
    "20210104",
    "20220104",
    "20250102",
    "20260710",
)


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


def test_market_cap_rejects_empty_requested_security_response() -> None:
    empty = pd.DataFrame(columns=["ts_code", "trade_date", "total_mv", "circ_mv"])

    with pytest.raises(ValueError, match="empty.*600000.SH"):
        normalize_market_cap(empty, expected_ts_code="600000.SH")


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


def test_membership_rejects_empty_requested_industry_response() -> None:
    empty = pd.DataFrame(columns=_membership_raw().columns)

    with pytest.raises(ValueError, match="empty.*801010.SI"):
        normalize_industry_membership(
            empty,
            {"801010.SI"},
            expected_l1_code="801010.SI",
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


def test_quality_rejects_missing_expected_security_and_industry_coverage() -> None:
    tables = ExposureTables(
        market_cap=normalize_market_cap(
            pd.DataFrame(
                [["600000.SH", "20210104", 1, 1]],
                columns=["ts_code", "trade_date", "total_mv", "circ_mv"],
            )
        ),
        industry_definition=normalize_industry_definition(_definition_raw()),
        industry_membership=normalize_industry_membership(
            _membership_raw(), {"801010.SI"}
        ),
    )

    report = validate_exposure_tables(
        tables,
        {"CN:SSE:600000", "CN:SSE:600001"},
        expected_security_ids={"CN:SSE:600000", "CN:SSE:600001"},
        expected_industry_ids={
            "CN:SW2021:801010.SI",
            "CN:SW2021:801020.SI",
        },
    )

    assert report["status"] == "error"
    assert report["checks"]["missing_security_coverage"]["count"] == 1
    assert report["checks"]["missing_industry_coverage"]["count"] == 1


def test_quality_rejects_severe_temporal_gaps_despite_every_entity_present() -> None:
    tables = ExposureTables(
        market_cap=normalize_market_cap(
            pd.DataFrame(
                [
                    ["600000.SH", "20200102", 1, 1],
                    ["600001.SH", "20200102", 1, 1],
                ],
                columns=DAILY_BASIC_FIELDS,
            )
        ),
        industry_definition=normalize_industry_definition(_definition_raw()),
        industry_membership=normalize_industry_membership(
            _membership_raw(), {"801010.SI"}
        ),
    )
    expected = pd.DataFrame(
        [
            {"trade_date": date, "security_id": security_id}
            for security_id in ("CN:SSE:600000", "CN:SSE:600001")
            for date in pd.date_range("2020-01-02", periods=5, freq="B")
        ]
    )

    report = validate_exposure_tables(
        tables,
        {"CN:SSE:600000", "CN:SSE:600001"},
        expected_market_observations=expected,
        minimum_temporal_coverage=0.70,
    )

    assert report["status"] == "error"
    assert report["summary"]["expected_observation_count"] == 10
    assert report["summary"]["observed_observation_count"] == 2
    assert report["summary"]["temporal_coverage_ratio"] == 0.2
    assert report["checks"]["insufficient_temporal_coverage"]["count"] == 8
    assert report["checks"]["undercovered_security"]["count"] == 2


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
                "name": path.relative_to(snapshot).as_posix(),
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


def test_probe_uses_exact_bounded_queries_and_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _phase5_provider_fixture(tmp_path)
    provider = _FakeProvider(
        tmp_path,
        [_definition_raw(), _membership_raw(), _market_raw()],
    )
    monkeypatch.setattr(
        "alpha_lab.robustness.exposure_data._provider_from_environment",
        lambda *_: provider,
    )

    report = probe_exposure_capabilities(ROOT / "config", tmp_path)

    assert report["bounded_probe"] is True
    assert report["sample_trade_date"] == "2020-01-02"
    assert report["sample_security"] == "600000.SH"
    assert provider.calls == [
        (
            "index_classify",
            {"level": "L1", "src": "SW2021"},
            INDEX_CLASSIFY_FIELDS,
        ),
        ("index_member_all", {"l1_code": "801010.SI"}, INDUSTRY_MEMBER_FIELDS),
        (
            "daily_basic",
            {
                "ts_code": "600000.SH",
                "start_date": "20200102",
                "end_date": "20200102",
            },
            DAILY_BASIC_FIELDS,
        ),
    ]


def test_probe_rejects_wrong_industry_before_daily_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _phase5_provider_fixture(tmp_path)
    provider = _FakeProvider(
        tmp_path,
        [
            _definition_raw(),
            _membership_raw().assign(l1_code="801020.SI"),
            _market_raw(),
        ],
    )
    monkeypatch.setattr(
        "alpha_lab.robustness.exposure_data._provider_from_environment",
        lambda *_: provider,
    )

    with pytest.raises(ValueError, match="unexpected industry"):
        probe_exposure_capabilities(ROOT / "config", tmp_path)

    assert len(provider.calls) == 2


def test_probe_rejects_wrong_daily_security(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _phase5_provider_fixture(tmp_path)
    provider = _FakeProvider(
        tmp_path,
        [
            _definition_raw(),
            _membership_raw(),
            _market_raw().assign(ts_code="600001.SH"),
        ],
    )
    monkeypatch.setattr(
        "alpha_lab.robustness.exposure_data._provider_from_environment",
        lambda *_: provider,
    )

    with pytest.raises(ValueError, match="unexpected security"):
        probe_exposure_capabilities(ROOT / "config", tmp_path)

    assert len(provider.calls) == 3


def test_acquisition_queries_each_security_and_l1_with_bounded_concurrency(
    tmp_path: Path,
) -> None:
    _phase5_provider_fixture(tmp_path, security_ids=["CN:SSE:600000", "CN:SSE:600001"])
    config, _ = load_robustness_config(ROOT / "config" / "robustness.yaml")
    provider = _AcquisitionFakeProvider(tmp_path)

    tables, artifacts = acquire_exposure_tables(tmp_path, config, provider)

    assert set(tables.market_cap["security_id"]) == {
        "CN:SSE:600000",
        "CN:SSE:600001",
    }
    assert set(tables.industry_membership["industry_id"]) == {
        "CN:SW2021:801010.SI",
        "CN:SW2021:801020.SI",
    }
    assert provider.max_active_bulk == 2
    assert provider.max_active_bulk <= 4
    assert len(artifacts) == 6
    assert [call[0] for call in provider.calls[:3]] == [
        "index_classify",
        "index_member_all",
        "daily_basic",
    ]
    assert provider.calls[-1][0] == "index_member_all"
    assert [call[0] for call in provider.calls].count("index_classify") == 1
    member_calls = [call for call in provider.calls if call[0] == "index_member_all"]
    assert {call[1]["l1_code"] for call in member_calls} == {
        "801010.SI",
        "801020.SI",
    }
    assert all(call[2] == INDUSTRY_MEMBER_FIELDS for call in member_calls)
    market_calls = [call for call in provider.calls if call[0] == "daily_basic"]
    assert len(market_calls) == 3
    assert all(call[2] == DAILY_BASIC_FIELDS for call in market_calls)
    assert {
        (call[1]["ts_code"], call[1]["start_date"], call[1]["end_date"])
        for call in market_calls
    } == {
        ("600000.SH", "20200102", "20200102"),
        ("600000.SH", "20200101", "20260711"),
        ("600001.SH", "20200101", "20260711"),
    }
    quality = validate_exposure_tables(
        tables,
        {"CN:SSE:600000", "CN:SSE:600001"},
    )
    assert quality["status"] == "pass"
    assert quality["summary"]["expected_observation_count"] == 10
    assert quality["summary"]["observed_observation_count"] == 10
    assert quality["summary"]["temporal_coverage_ratio"] == 1.0


def test_acquisition_rejects_one_row_per_entity_over_long_horizon(
    tmp_path: Path,
) -> None:
    _phase5_provider_fixture(tmp_path, security_ids=["CN:SSE:600000", "CN:SSE:600001"])
    config, _ = load_robustness_config(ROOT / "config" / "robustness.yaml")
    provider = _AcquisitionFakeProvider(tmp_path, sparse_bulk=True)

    with pytest.raises(ValueError, match="coverage"):
        acquire_exposure_tables(tmp_path, config, provider)


@pytest.mark.parametrize(
    ("empty_security", "empty_industry", "message"),
    [
        ("600001.SH", None, "empty response for 600001.SH"),
        (None, "801020.SI", "empty response for 801020.SI"),
    ],
)
def test_acquisition_does_not_mask_an_empty_requested_response(
    tmp_path: Path,
    empty_security: str | None,
    empty_industry: str | None,
    message: str,
) -> None:
    _phase5_provider_fixture(tmp_path, security_ids=["CN:SSE:600000", "CN:SSE:600001"])
    config, _ = load_robustness_config(ROOT / "config" / "robustness.yaml")
    provider = _AcquisitionFakeProvider(
        tmp_path,
        empty_security=empty_security,
        empty_industry=empty_industry,
    )

    with pytest.raises(ValueError, match=message):
        acquire_exposure_tables(tmp_path, config, provider)


def _definition_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [["801010.SI", "农林牧渔", "L1", "110000", "SW2021"]],
        columns=["index_code", "industry_name", "level", "industry_code", "src"],
    )


def _market_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [["600000.SH", "20200102", 1, 1]],
        columns=DAILY_BASIC_FIELDS,
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


def _phase5_provider_fixture(
    tmp_path: Path, *, security_ids: list[str] | None = None
) -> None:
    config, _ = load_robustness_config(ROOT / "config" / "robustness.yaml")
    security_ids = security_ids or ["CN:SSE:600000"]
    snapshot = tmp_path / "research" / config.phase5_snapshot_id
    snapshot.mkdir(parents=True)
    security = snapshot / "security_master.parquet"
    membership = snapshot / "index_membership.parquet"
    frame = pd.DataFrame({"security_id": security_ids})
    frame.to_parquet(security)
    frame.to_parquet(membership)
    universe = snapshot / "universe_dates.parquet"
    daily = snapshot / "daily_bar" / "part.parquet"
    daily.parent.mkdir(parents=True)
    observations = pd.DataFrame(
        [
            {
                "as_of_date": pd.Timestamp(value),
                "trade_date": pd.Timestamp(value),
                "security_id": security_id,
            }
            for value in _COVERAGE_DATES
            for security_id in security_ids
        ]
    )
    observations[["as_of_date", "security_id"]].to_parquet(universe)
    pd.concat(
        [
            observations[["trade_date", "security_id"]],
            pd.DataFrame(
                [
                    {
                        "trade_date": pd.Timestamp("2020-01-01"),
                        "security_id": security_ids[0],
                    },
                    {
                        "trade_date": pd.Timestamp("2020-01-02"),
                        "security_id": "CN:SSE:600999",
                    },
                ]
            ),
        ],
        ignore_index=True,
    ).to_parquet(daily)
    manifest_dir = tmp_path / "manifests" / config.phase5_snapshot_id
    manifest_dir.mkdir(parents=True)
    manifest = {
        "snapshot_id": config.phase5_snapshot_id,
        "snapshot_type": "research_market",
        "artifacts": [
            {
                "name": path.relative_to(snapshot).as_posix(),
                "path": path.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in (security, membership, universe, daily)
        ],
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest))


class _FakeProvider:
    def __init__(self, root: Path, frames: list[pd.DataFrame]) -> None:
        self.root = root
        self.frames = frames
        self.calls: list[tuple[str, dict[str, object], tuple[str, ...]]] = []

    def query(
        self, api_name: str, params: dict[str, object], fields: tuple[str, ...]
    ) -> TushareQueryResult:
        self.calls.append((api_name, dict(params), fields))
        frame = self.frames.pop(0)
        request_sha256 = f"{len(self.calls):064x}"
        path = self.root / "fake" / f"{request_sha256}.parquet"
        metadata = path.with_suffix(".json")
        return TushareQueryResult(
            frame=frame,
            artifact=TushareArtifact(
                api_name=api_name,
                request_sha256=request_sha256,
                parquet_path=path,
                metadata_path=metadata,
                sha256="a" * 64,
                row_count=len(frame),
                params=dict(params),
                fields=fields,
                ingested_at="2026-07-13T00:00:00Z",
            ),
            cache_hits=1,
            network_requests=0,
        )


class _AcquisitionFakeProvider:
    def __init__(
        self,
        root: Path,
        *,
        empty_security: str | None = None,
        empty_industry: str | None = None,
        sparse_bulk: bool = False,
    ) -> None:
        self.root = root
        self.calls: list[tuple[str, dict[str, object], tuple[str, ...]]] = []
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(2)
        self.active_bulk = 0
        self.max_active_bulk = 0
        self.empty_security = empty_security
        self.empty_industry = empty_industry
        self.sparse_bulk = sparse_bulk

    def query(
        self, api_name: str, params: dict[str, object], fields: tuple[str, ...]
    ) -> TushareQueryResult:
        with self.lock:
            self.calls.append((api_name, dict(params), fields))
            request_number = len(self.calls)
        is_bulk = api_name == "daily_basic" and params.get("start_date") != params.get(
            "end_date"
        )
        if is_bulk:
            with self.lock:
                self.active_bulk += 1
                self.max_active_bulk = max(self.max_active_bulk, self.active_bulk)
            self.barrier.wait(timeout=5)
        try:
            frame = self._frame(api_name, params)
        finally:
            if is_bulk:
                with self.lock:
                    self.active_bulk -= 1
        request_sha256 = f"{request_number:064x}"
        path = self.root / "fake" / f"{request_sha256}.parquet"
        return TushareQueryResult(
            frame=frame,
            artifact=TushareArtifact(
                api_name=api_name,
                request_sha256=request_sha256,
                parquet_path=path,
                metadata_path=path.with_suffix(".json"),
                sha256="a" * 64,
                row_count=len(frame),
                params=dict(params),
                fields=fields,
                ingested_at="2026-07-13T00:00:00Z",
            ),
            cache_hits=1,
            network_requests=0,
        )

    def _frame(self, api_name: str, params: dict[str, object]) -> pd.DataFrame:
        if api_name == "index_classify":
            return pd.concat(
                [
                    _definition_raw(),
                    _definition_raw().assign(
                        index_code="801020.SI",
                        industry_code="120000",
                        industry_name="采掘",
                    ),
                ],
                ignore_index=True,
            )
        if api_name == "index_member_all":
            l1_code = str(params["l1_code"])
            if l1_code == self.empty_industry:
                return pd.DataFrame(columns=INDUSTRY_MEMBER_FIELDS)
            return _membership_raw().assign(
                l1_code=l1_code,
                ts_code=("600000.SH" if l1_code == "801010.SI" else "600001.SH"),
            )
        if api_name == "daily_basic":
            if (
                params.get("start_date") != params.get("end_date")
                and params["ts_code"] == self.empty_security
            ):
                return pd.DataFrame(columns=DAILY_BASIC_FIELDS)
            dates = (
                (_COVERAGE_DATES[:1] if self.sparse_bulk else _COVERAGE_DATES)
                if params.get("start_date") != params.get("end_date")
                else (str(params["start_date"]),)
            )
            return pd.DataFrame(
                [[str(params["ts_code"]), value, 1, 1] for value in dates],
                columns=DAILY_BASIC_FIELDS,
            )
        raise AssertionError(f"unexpected endpoint: {api_name}")
