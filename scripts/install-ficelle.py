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
# Runs the checkout's installer in an interpreter that may not hold Ficelle yet. The path
# lives in this process only: a PYTHONPATH would leak into the installer's own `ficelle`
# children and make them import the checkout instead of the package just installed.
CHECKOUT_INSTALLER = (
    "import sys; sys.path.insert(0, sys.argv.pop(1)); "
    "from ficelle.install import main; raise SystemExit(main())"
)


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--python", default=os.getenv("FICELLE_PYTHON") or sys.executable)
    parser.add_argument("--venv", default=os.getenv("FICELLE_VENV") or str(DEFAULT_VENV))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    bootstrap, setup_args = parser.parse_known_args(argv)
    venv = Path(bootstrap.venv).expanduser().resolve()
    python = venv_python(venv)
    read_only = bootstrap.dry_run or bootstrap.preflight_only
    if not python.exists():
        command = [bootstrap.python, "-m", "venv", str(venv)]
        if bootstrap.dry_run:
            print(f"DRY RUN: {' '.join(command)}")
        elif not bootstrap.preflight_only:
            created = subprocess.run(command, check=False)
            if created.returncode != 0:
                return created.returncode
    # A caller-supplied package (a wheel, a spec) is installed by ficelle.install itself:
    # the editable checkout below would otherwise be installed and the artifact skipped.
    custom_package = any(arg == "--package" or arg.startswith("--package=") for arg in setup_args)
    command = [
        str(python),
        "-m",
        "ficelle.install",
        "--python",
        str(python),
        *([] if custom_package else ["--package", str(ROOT)]),
        *setup_args,
    ]
    if read_only:
        # Neither flag may touch the owned venv, which may not exist yet: run the checkout's
        # installer with this interpreter and keep the venv python as the target to inspect.
        command[0] = str(python) if python.exists() else sys.executable
        if bootstrap.dry_run:
            command.append("--dry-run")
        if bootstrap.preflight_only:
            command.append("--preflight-only")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        return subprocess.run(command, check=False, env=env).returncode
    if not custom_package:
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
            [str(python), "-m", "pip", "install", "--no-deps", "-e", str(ROOT)],
            check=False,
        )
        if installed.returncode != 0:
            return installed.returncode
        checked = subprocess.run([str(python), "-m", "pip", "check"], check=False)
        if checked.returncode != 0:
            return checked.returncode
        command.insert(command.index("--package"), "--skip-package")
    else:
        command[1:3] = ["-c", CHECKOUT_INSTALLER, str(ROOT / "src")]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
