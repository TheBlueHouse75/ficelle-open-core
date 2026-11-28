#!/usr/bin/env python3
"""Install a source checkout into Ficelle's dedicated local runtime."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
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


def run_python(
    interpreter: str, arguments: list[str], *, env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    child_env = dict(os.environ if env is None else env, PYTHONSAFEPATH="1")
    if child_env.get("FICELLE_HOME"):
        child_env["FICELLE_HOME"] = os.path.abspath(Path(child_env["FICELLE_HOME"]).expanduser())
    with tempfile.TemporaryDirectory(prefix="ficelle-install-") as cwd:
        Path(cwd).chmod(0o700)
        return subprocess.run([interpreter, "-P", *arguments], check=False, cwd=cwd, env=child_env)


def caller_package_reference(value: str) -> str:
    # Preserve package specifications and URLs when the installer leaves the caller's cwd.
    if "://" in value or value.startswith(("git+", "hg+", "svn+", "bzr+")):
        return value
    path = Path(value).expanduser()
    if value in {".", ".."} or "/" in value or value.endswith((".whl", ".zip", ".tar.gz")) or path.exists():
        return os.path.abspath(path)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--python", default=os.getenv("FICELLE_PYTHON") or sys.executable)
    parser.add_argument("--venv", default=os.getenv("FICELLE_VENV") or str(DEFAULT_VENV))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--package", type=caller_package_reference)
    parser.add_argument("--ficelle-home", type=Path)
    parser.add_argument("--hermes-home", type=Path)
    bootstrap, setup_args = parser.parse_known_args(argv)
    for flag, value in (("--ficelle-home", bootstrap.ficelle_home), ("--hermes-home", bootstrap.hermes_home)):
        if value is not None:
            setup_args.extend((flag, os.path.abspath(value.expanduser())))
    venv = Path(bootstrap.venv).expanduser().resolve()
    python = venv_python(venv)
    read_only = bootstrap.dry_run or bootstrap.preflight_only
    if not python.exists():
        interpreter = str(Path(bootstrap.python).expanduser())
        interpreter = os.path.abspath(shutil.which(interpreter) or interpreter)
        venv_args = ["-m", "venv", str(venv)]
        if bootstrap.dry_run:
            print(f"DRY RUN: {' '.join([interpreter, '-P', *venv_args])}")
        elif not bootstrap.preflight_only:
            created = run_python(interpreter, venv_args)
            if created.returncode != 0:
                return created.returncode
    # A caller-supplied package (a wheel, a spec) is installed by ficelle.install itself:
    # the editable checkout below would otherwise be installed and the artifact skipped.
    custom_package = bootstrap.package is not None
    installer_args = [
        "--python",
        str(python),
        "--package",
        bootstrap.package if custom_package else str(ROOT),
        *setup_args,
    ]
    if read_only:
        # Neither flag may touch the owned venv, which may not exist yet: run the checkout's
        # installer with this interpreter and keep the venv python as the target to inspect.
        interpreter = str(python) if python.exists() else os.path.abspath(sys.executable)
        if bootstrap.dry_run:
            installer_args.append("--dry-run")
        if bootstrap.preflight_only:
            installer_args.append("--preflight-only")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        return run_python(interpreter, ["-m", "ficelle.install", *installer_args], env=env).returncode
    if not custom_package:
        with (ROOT / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle).get("project", {})
        requirements = project.get("dependencies", []) if isinstance(project, dict) else []
        installed = run_python(str(python), ["-m", "pip", "install", *requirements])
        if installed.returncode != 0:
            return installed.returncode
        installed = run_python(str(python), ["-m", "pip", "install", "--no-deps", "-e", str(ROOT)])
        if installed.returncode != 0:
            return installed.returncode
        checked = run_python(str(python), ["-m", "pip", "check"])
        if checked.returncode != 0:
            return checked.returncode
        return run_python(str(python), ["-m", "ficelle.install", "--skip-package", *installer_args]).returncode
    return run_python(str(python), ["-c", CHECKOUT_INSTALLER, str(ROOT / "src"), *installer_args]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
