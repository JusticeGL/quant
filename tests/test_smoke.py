from __future__ import annotations

import contextlib
import io
import unittest

from alpha_lab.smoke import collect_report, main


class SmokeReportTest(unittest.TestCase):
    def test_report_identifies_linux_python_and_required_packages(self) -> None:
        report = collect_report()

        self.assertEqual(report["system"], "Linux")
        self.assertTrue(report["python"].startswith("3.11."))
        for package in ("qlib", "akshare", "duckdb", "pyarrow", "lightgbm"):
            self.assertTrue(report[package])

    def test_main_prints_platform_and_package_versions(self) -> None:
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        rendered = output.getvalue()
        for label in (
            "system",
            "platform",
            "machine",
            "python",
            "qlib",
            "akshare",
            "duckdb",
            "pyarrow",
            "lightgbm",
        ):
            self.assertIn(f"{label}:", rendered)


if __name__ == "__main__":
    unittest.main()
