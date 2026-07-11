from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from alpha_lab.data.config import DataSourceConfig, load_phase1_config

ROOT = Path(__file__).resolve().parents[2]


def test_repository_phase1_config_is_fixed_and_explicitly_biased() -> None:
    config = load_phase1_config(ROOT / "config")

    assert config.source.provider == "akshare"
    assert config.source.endpoint == "stock_zh_a_hist"
    assert config.source.start_date == date(2024, 1, 1)
    assert config.source.end_date == date(2024, 6, 30)
    assert len(config.universe.symbols) == 10
    assert config.universe.research_eligible is False
    assert "survivorship-biased" in config.universe.disclaimer


def test_source_config_rejects_an_inverted_date_range() -> None:
    with pytest.raises(ValueError, match="end_date"):
        DataSourceConfig(
            provider="akshare",
            endpoint="stock_zh_a_hist",
            period="daily",
            start_date=date(2024, 2, 1),
            end_date=date(2024, 1, 1),
            adjust="",
        )


def test_universe_codes_are_strings_in_yaml() -> None:
    document = yaml.safe_load((ROOT / "config" / "universe.yaml").read_text())

    assert all(isinstance(item["code"], str) for item in document["symbols"])
