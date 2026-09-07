from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


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
