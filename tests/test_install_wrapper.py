from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install-ficelle.py"
SPEC = importlib.util.spec_from_file_location("ficelle_install_wrapper", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
install_wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(install_wrapper)


def test_source_wrapper_orders_editable_pip_options_before_the_checkout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(install_wrapper.subprocess, "run", run)

    result = install_wrapper.main(["--venv", str(tmp_path / "venv"), "--non-interactive"])

    assert result == 0
    assert calls[1] == [
        str(python),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "-e",
        str(install_wrapper.ROOT),
    ]


def _record_runs(monkeypatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(install_wrapper.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("flag", ["--preflight-only", "--dry-run"])
@pytest.mark.parametrize("venv_exists", [True, False])
def test_source_wrapper_read_only_modes_never_create_or_modify_the_owned_venv(
    monkeypatch, tmp_path: Path, flag: str, venv_exists: bool
) -> None:
    venv = tmp_path / "venv"
    python = install_wrapper.venv_python(venv)
    if venv_exists:
        python.parent.mkdir(parents=True)
        python.touch()
    calls = _record_runs(monkeypatch)

    assert install_wrapper.main(["--venv", str(venv), flag, "--skip-service"]) == 0

    # A missing venv python must not be executed: that was a FileNotFoundError.
    assert len(calls) == 1
    assert calls[0][0] == (str(python) if venv_exists else sys.executable)
    assert calls[0][1:3] == ["-m", "ficelle.install"]
    assert flag in calls[0]
    assert not {"pip", "venv"} & set(calls[0])


def test_source_wrapper_installs_a_caller_supplied_package_instead_of_the_checkout(
    monkeypatch, tmp_path: Path
) -> None:
    python = install_wrapper.venv_python(tmp_path / "venv")
    python.parent.mkdir(parents=True)
    python.touch()
    calls = _record_runs(monkeypatch)

    result = install_wrapper.main(
        ["--venv", str(tmp_path / "venv"), "--package", "dist/ficelle_router-1.0-py3-none-any.whl", "--no-editable"]
    )

    assert result == 0
    assert len(calls) == 1
    assert calls[0].count("--package") == 1
    assert "dist/ficelle_router-1.0-py3-none-any.whl" in calls[0]
    assert "--skip-package" not in calls[0]


def test_source_wrapper_runs_the_checkout_installer_when_the_fresh_venv_lacks_ficelle(
    monkeypatch, tmp_path: Path
) -> None:
    # A caller-supplied package skips the editable install, so the fresh venv holds no
    # `ficelle` module yet: the installer itself must still come from the checkout.
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    python = install_wrapper.venv_python(venv)
    run = subprocess.run
    calls = _record_runs(monkeypatch)

    assert install_wrapper.main(["--venv", str(venv), "--package", "x.whl", "--skip-service"]) == 0

    command = calls[-1]
    probe = command[: command.index("--python")] + ["--help"]
    assert probe[0] == str(python)
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = run(probe, capture_output=True, text=True, check=False, cwd=tmp_path, env=env)
    assert result.returncode == 0, result.stderr
    assert "--package" in result.stdout
