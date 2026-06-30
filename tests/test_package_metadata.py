from __future__ import annotations

import tomllib
from pathlib import Path

import ficelle


def test_runtime_version_matches_project_metadata():
    metadata = tomllib.loads(Path("pyproject.toml").read_text())

    assert ficelle.__version__ == metadata["project"]["version"]
