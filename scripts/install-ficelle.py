#!/usr/bin/env python3
"""Install a source checkout into Ficelle's dedicated local runtime."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV = Path.home() / ".local" / "share" / "ficelle" / "venv"


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--python", default=os.getenv("FICELLE_PYTHON") or sys.executable)
    parser.add_argument("--venv", default=os.getenv("FICELLE_VENV") or str(DEFAULT_VENV))
    parser.add_argument("--dry-run", action="store_true")
    bootstrap, setup_args = parser.parse_known_args(argv)
    venv = Path(bootstrap.venv).expanduser().resolve()
    python = venv_python(venv)
    if not python.exists():
        command = [bootstrap.python, "-m", "venv", str(venv)]
        if bootstrap.dry_run:
            print(f"DRY RUN: {' '.join(command)}")
        else:
            created = subprocess.run(command, check=False)
            if created.returncode != 0:
                return created.returncode
    command = [
        str(python),
        "-m",
        "ficelle.install",
        "--python",
        str(python),
        "--package",
        str(ROOT),
        *setup_args,
    ]
    if bootstrap.dry_run:
        command.append("--dry-run")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        return subprocess.run(command, check=False, env=env).returncode
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle).get("project", {})
    requirements = project.get("dependencies", []) if isinstance(project, dict) else []
    installed = subprocess.run(
        [str(python), "-m", "pip", "install", *requirements],
        check=False,
    )
    if installed.returncode != 0:
        return installed.returncode
    installed = subprocess.run(
        [str(python), "-m", "pip", "install", "-e", "--no-deps", str(ROOT)],
        check=False,
    )
    if installed.returncode != 0:
        return installed.returncode
    checked = subprocess.run([str(python), "-m", "pip", "check"], check=False)
    if checked.returncode != 0:
        return checked.returncode
    command.insert(command.index("--package"), "--skip-package")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
