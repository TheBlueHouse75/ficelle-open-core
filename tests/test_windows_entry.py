from __future__ import annotations

import os
import sys

import pytest

from ficelle import windows_entry


def test_windows_entry_maps_assignments_to_service_environment():
    assert windows_entry.service_environment_from_assignments(
        [
            "FICELLE_HOME=C:\\Users\\cyril\\.ficelle",
            "HERMES_HOME=C:\\Users\\cyril\\.hermes",
        ]
    ) == {
        "FICELLE_HOME": "C:\\Users\\cyril\\.ficelle",
        "HERMES_HOME": "C:\\Users\\cyril\\.hermes",
    }


def test_windows_entry_rejects_malformed_or_homeless_assignments():
    with pytest.raises(SystemExit):
        windows_entry.service_environment_from_assignments(["FICELLE_HOME"])
    with pytest.raises(SystemExit):
        windows_entry.service_environment_from_assignments(["FICELLE_HOME="])
    with pytest.raises(SystemExit):
        windows_entry.service_environment_from_assignments(["HERMES_HOME=C:\\x"])


def test_windows_entry_main_sets_environment_redirects_stdio_and_serves(tmp_path, monkeypatch):
    ficelle_home = tmp_path / ".ficelle"
    runtime_dir = tmp_path / "legacy-runtime"
    observed: dict[str, object] = {}

    def fake_main_args(argv):
        observed["argv"] = argv
        observed["FICELLE_HOME"] = os.environ.get("FICELLE_HOME")
        observed["FICELLE_RUNTIME_DIR"] = os.environ.get("FICELLE_RUNTIME_DIR")
        print("server line")
        return 0

    # Register originals with monkeypatch so teardown restores the test environment.
    monkeypatch.setenv("FICELLE_HOME", "before")
    monkeypatch.setenv("FICELLE_RUNTIME_DIR", "before")
    monkeypatch.setattr("ficelle.router.main_args", fake_main_args)

    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        assert (
            windows_entry.main(
                [
                    f"FICELLE_HOME={ficelle_home}",
                    f"FICELLE_RUNTIME_DIR={runtime_dir}",
                ]
            )
            == 0
        )
    finally:
        if sys.stdout is not original_stdout:
            sys.stdout.close()
        if sys.stderr is not original_stderr:
            sys.stderr.close()
        sys.stdout, sys.stderr = original_stdout, original_stderr

    assert observed["argv"] == ["--serve"]
    assert observed["FICELLE_HOME"] == str(ficelle_home)
    assert observed["FICELLE_RUNTIME_DIR"] == str(runtime_dir)
    assert "server line" in (ficelle_home / "logs" / "ficelle.log").read_text(encoding="utf-8")
    assert (ficelle_home / "logs" / "ficelle.error.log").exists()
