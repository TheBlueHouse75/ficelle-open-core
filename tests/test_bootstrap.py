from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap-ficelle.py"
spec = importlib.util.spec_from_file_location("ficelle_bootstrap", BOOTSTRAP_PATH)
assert spec is not None and spec.loader is not None
bootstrap = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bootstrap
spec.loader.exec_module(bootstrap)


def make_options(tmp_path, **overrides):
    values = {
        "wheel_url": str(tmp_path / "ficelle_router-0.1.2-py3-none-any.whl"),
        "license_key": None,
        "sha256": None,
        "python": None,
        "hermes_home": tmp_path / ".hermes",
        "configure_hermes": True,
        "backup_existing": True,
        "skip_service": False,
        "skip_smoke": False,
        "dry_run": False,
        "keep_wheel": False,
    }
    values.update(overrides)
    return bootstrap.BootstrapOptions(**values)


def test_redact_url_hides_query_tokens():
    assert bootstrap.redact_url("https://example.test/wheel?token=secret") == "https://example.test/wheel?[REDACTED]"
    assert bootstrap.redact_url("https://example.test/wheel") == "https://example.test/wheel"


def test_setup_command_configures_hermes_by_default(tmp_path):
    wheel = tmp_path / "ficelle_router-0.1.2-py3-none-any.whl"
    options = make_options(tmp_path)

    command = bootstrap.setup_command(options, Path("/tmp/hermes-python"), wheel)

    assert command[:3] == ["/tmp/hermes-python", "-m", "ficelle.install"]
    assert "--skip-package" in command
    assert "--configure-hermes" in command
    assert "--hermes-home" in command
    assert str(tmp_path / ".hermes") in command


def test_setup_command_can_disable_config_and_backups(tmp_path):
    wheel = tmp_path / "ficelle_router-0.1.2-py3-none-any.whl"
    options = make_options(tmp_path, configure_hermes=False, backup_existing=False, skip_service=True, skip_smoke=True, dry_run=True)

    command = bootstrap.setup_command(options, Path("/tmp/hermes-python"), wheel)

    assert "--configure-hermes" not in command
    assert "--no-backup" in command
    assert "--skip-service" in command
    assert "--skip-smoke" in command
    assert "--dry-run" in command


def test_install_wheel_falls_back_to_uv_when_pip_is_missing(monkeypatch, tmp_path):
    calls = []
    wheel = tmp_path / "ficelle_router-0.1.2-py3-none-any.whl"
    wheel.write_text("placeholder")
    options = make_options(tmp_path)

    def fake_run(command, *, dry_run, env=None):
        calls.append((command, env.get("HERMES_HOME") if env else None))
        if command[:3] == ["/tmp/hermes-python", "-m", "pip"]:
            return bootstrap.CommandResult(command, 1, stderr="/tmp/hermes-python: No module named pip")
        return bootstrap.CommandResult(command, 0)

    monkeypatch.setattr(bootstrap, "run_command", fake_run)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: "/opt/homebrew/bin/uv" if name == "uv" else None)

    bootstrap.install_wheel(options, Path("/tmp/hermes-python"), wheel)

    assert calls == [
        (["/tmp/hermes-python", "-m", "pip", "install", "--force-reinstall", str(wheel)], str(tmp_path / ".hermes")),
        (["uv", "pip", "install", "--python", "/tmp/hermes-python", "--force-reinstall", str(wheel)], str(tmp_path / ".hermes")),
    ]


def test_download_local_wheel_verifies_sha256_path(tmp_path):
    source = tmp_path / "source" / "ficelle_router-0.1.2-py3-none-any.whl"
    source.parent.mkdir()
    source.write_bytes(b"wheel-bytes")
    options = make_options(tmp_path, wheel_url=str(source))

    downloaded = bootstrap.download_wheel(options, tmp_path / "download")

    assert downloaded.read_bytes() == b"wheel-bytes"
    assert downloaded.name == source.name


def test_sha256_verification_rejects_mismatch(tmp_path):
    wheel = tmp_path / "ficelle_router-0.1.2-py3-none-any.whl"
    wheel.write_bytes(b"wheel-bytes")

    try:
        bootstrap.verify_sha256(wheel, "0" * 64)
    except SystemExit as exc:
        assert "SHA256 mismatch" in str(exc)
    else:
        raise AssertionError("expected checksum mismatch")


def test_detect_hermes_python_uses_explicit_python(monkeypatch, tmp_path):
    options = make_options(tmp_path, python="/tmp/hermes-python")
    monkeypatch.setattr(bootstrap, "python_version_result", lambda python: bootstrap.CommandResult([str(python)], 0, "3.11.14\n"))

    assert bootstrap.detect_hermes_python(options) == Path("/tmp/hermes-python")
