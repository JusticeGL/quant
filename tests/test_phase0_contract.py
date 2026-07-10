from __future__ import annotations

import pathlib
import tomllib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class PhaseZeroContractTest(unittest.TestCase):
    def test_required_files_exist(self) -> None:
        required = {
            ".env.example",
            ".gitignore",
            "AGENTS.md",
            "Dockerfile",
            "Makefile",
            "README.md",
            "compose.yaml",
            "pyproject.toml",
            "uv.lock",
        }

        missing = sorted(path for path in required if not (ROOT / path).is_file())

        self.assertEqual(missing, [])

    def test_python_and_phase_zero_dependencies_are_declared(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle)["project"]

        self.assertEqual(project["requires-python"], ">=3.11,<3.12")
        names = {
            requirement.split()[0].lower() for requirement in project["dependencies"]
        }
        self.assertTrue({"pyqlib", "akshare", "duckdb", "pyarrow", "lightgbm"} <= names)

    def test_qlib_source_is_pinned_for_linux_arm64(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as handle:
            config = tomllib.load(handle)
            dependencies = config["project"]["dependencies"]

        qlib_requirement = next(
            requirement
            for requirement in dependencies
            if requirement.startswith("pyqlib")
        )
        self.assertEqual(
            qlib_requirement,
            "pyqlib @ git+https://github.com/microsoft/qlib.git@"
            "da920b7f954f48ab1bb64117c976710de198373e",
        )

    def test_build_backend_allows_pinned_qlib_source_reference(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as handle:
            config = tomllib.load(handle)

        self.assertTrue(config["tool"]["hatch"]["metadata"]["allow-direct-references"])

    def test_dockerfile_uses_python_311_without_amd64_override(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("python:3.11", dockerfile)
        self.assertNotIn("linux/amd64", dockerfile)

    def test_dockerfile_does_not_require_external_frontend_image(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertFalse(dockerfile.startswith("# syntax="))

    def test_docker_context_excludes_macos_metadata(self) -> None:
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

        self.assertIn("**/.DS_Store", dockerignore.splitlines())

    def test_dockerfile_caches_dependencies_before_copying_source(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        dependency_sync = "uv sync --frozen --all-groups --no-install-project"

        dependency_sync_position = dockerfile.index(dependency_sync)
        source_copy_position = dockerfile.index("COPY src ./src")
        project_sync_position = dockerfile.index(
            "uv sync --frozen --all-groups", source_copy_position
        )

        self.assertLess(dependency_sync_position, source_copy_position)
        self.assertLess(source_copy_position, project_sync_position)

    def test_dockerfile_caches_dependencies_before_copying_readme(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        dependency_sync_position = dockerfile.index(
            "uv sync --frozen --all-groups --no-install-project"
        )
        readme_copy_position = dockerfile.index("COPY README.md ./")

        self.assertLess(dependency_sync_position, readme_copy_position)

    def test_makefile_exposes_phase_zero_targets(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

        for target in ("build", "lock", "smoke", "lint", "test"):
            self.assertRegex(makefile, rf"(?m)^{target}:")


if __name__ == "__main__":
    unittest.main()
