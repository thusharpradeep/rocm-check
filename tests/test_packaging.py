from __future__ import annotations

import os
import subprocess
import sys
from importlib import resources
from pathlib import Path

from rocm_check import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_both_console_scripts() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'rocm_check = "rocm_check.cli:main"' in text
    assert 'rocm-check = "rocm_check.cli:main"' in text


def test_pyproject_uses_single_source_version() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'dynamic = ["version"]' in text
    assert 'version = {attr = "rocm_check.__version__"}' in text


def test_package_data_is_available_from_source_tree() -> None:
    package = resources.files("rocm_check")

    for relative in (
        "rules.yaml",
        "templates/report.html",
        "static/report.css",
        "static/report.js",
    ):
        assert package.joinpath(relative).is_file(), relative


def test_module_entry_point_runs_from_another_directory(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )

    completed = subprocess.run(
        [sys.executable, "-m", "rocm_check", "--version"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )

    assert completed.returncode == 0
    assert __version__ in completed.stdout
