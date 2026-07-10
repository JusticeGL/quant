"""Phase 0 import and runtime smoke check."""

from __future__ import annotations

import platform
from importlib.metadata import version

import akshare
import duckdb
import lightgbm
import pyarrow
import qlib

_IMPORTED_MODULES = (qlib, akshare, duckdb, pyarrow, lightgbm)
_DISTRIBUTIONS = {
    "qlib": "pyqlib",
    "akshare": "akshare",
    "duckdb": "duckdb",
    "pyarrow": "pyarrow",
    "lightgbm": "lightgbm",
}


def collect_report() -> dict[str, str]:
    """Return the formal runtime identity and required package versions."""
    package_versions = {
        name: version(distribution) for name, distribution in _DISTRIBUTIONS.items()
    }
    return {
        "system": platform.system(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        **package_versions,
    }


def main() -> int:
    """Print the smoke report after all required imports have succeeded."""
    report = collect_report()
    for label, value in report.items():
        print(f"{label}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
