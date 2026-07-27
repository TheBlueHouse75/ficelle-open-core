from __future__ import annotations

import importlib.util
import io
import sys
from email.message import Message
from pathlib import Path


BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap-ficelle.py"
spec = importlib.util.spec_from_file_location("ficelle_bootstrap", BOOTSTRAP_PATH)
assert spec is not None and spec.loader is not None
bootstrap = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bootstrap
spec.loader.exec_module(bootstrap)


class _WheelResponse(io.BytesIO):
    def __init__(self, filename: str):
        super().__init__(b"wheel-bytes")
        self.headers = Message()
        self.headers.add_header(
            "Content-Disposition",
            "attachment",
            filename=filename,
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


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
        (["uv", "pip", "install", "--python", "/tmp/hermes-python", "--reinstall", str(wheel)], str(tmp_path / ".hermes")),
    ]


def test_download_local_wheel_verifies_sha256_path(tmp_path):
    source = tmp_path / "source" / "ficelle_router-0.1.2-py3-none-any.whl"
    source.parent.mkdir()
    source.write_bytes(b"wheel-bytes")
    options = make_options(tmp_path, wheel_url=str(source))

    downloaded = bootstrap.download_wheel(options, tmp_path / "download")

    assert downloaded.read_bytes() == b"wheel-bytes"
    assert downloaded.name == source.name


def test_download_local_wheel_dry_run_allows_nonexistent_path(tmp_path):
    source = tmp_path / "not-built-yet" / "ficelle_pro-1.0-py3-none-any.whl"
    options = make_options(
        tmp_path,
        wheel_url=str(source),
        dry_run=True,
    )

    downloaded = bootstrap.download_wheel(options, tmp_path / "download")

    assert downloaded == tmp_path / "download" / source.name


def test_default_wheel_endpoint_uses_valid_fallback_filename():
    assert bootstrap.wheel_filename_from_url(bootstrap.DEFAULT_WHEEL_URL) == (
        "ficelle_pro-0-py3-none-any.whl"
    )


def test_wheel_filename_rejects_non_version_placeholder():
    assert (
        bootstrap.safe_wheel_filename("ficelle_pro-latest-py3-none-any.whl")
        is None
    )


def test_wheel_filename_rejects_invalid_versions():
    for version in ("1..2", "1!", "1+", "1_2"):
        assert bootstrap.safe_wheel_filename(
            f"ficelle_pro-{version}-py3-none-any.whl"
        ) is None


def test_remote_download_honors_valid_content_disposition(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bootstrap,
        "open_same_origin",
        lambda request, timeout: _WheelResponse(
            "ficelle_pro-1.2.3-py3-none-any.whl"
        ),
    )
    options = make_options(
        tmp_path,
        wheel_url=bootstrap.DEFAULT_WHEEL_URL,
        license_key="FICL-SECRET",
    )

    downloaded = bootstrap.download_wheel(options, tmp_path / "download")

    assert downloaded.name == "ficelle_pro-1.2.3-py3-none-any.whl"
    assert downloaded.read_bytes() == b"wheel-bytes"


def test_remote_download_rejects_unsafe_advertised_filename(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bootstrap,
        "open_same_origin",
        lambda request, timeout: _WheelResponse("../../not-a-wheel.whl"),
    )
    options = make_options(tmp_path, wheel_url=bootstrap.DEFAULT_WHEEL_URL)

    downloaded = bootstrap.download_wheel(options, tmp_path / "download")

    assert downloaded.name == bootstrap.FALLBACK_WHEEL_FILENAME
    assert downloaded.parent == tmp_path / "download"


def test_remote_download_refuses_insecure_non_loopback_url(monkeypatch, tmp_path):
    called = False

    def open_url(request, timeout):
        nonlocal called
        called = True
        return _WheelResponse("ficelle_pro-1.0-py3-none-any.whl")

    monkeypatch.setattr(bootstrap, "open_same_origin", open_url)
    options = make_options(
        tmp_path,
        wheel_url="http://packages.example.test/ficelle_pro-1.0-py3-none-any.whl",
        license_key="FICL-SECRET",
    )

    try:
        bootstrap.download_wheel(options, tmp_path / "download")
    except SystemExit as exc:
        assert "insecure URL" in str(exc)
    else:
        raise AssertionError("expected an insecure remote URL to be refused")
    assert called is False


def test_remote_download_allows_loopback_http(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bootstrap,
        "open_same_origin",
        lambda request, timeout: _WheelResponse(
            "ficelle_pro-1.0-py3-none-any.whl"
        ),
    )
    options = make_options(
        tmp_path,
        wheel_url="http://127.0.0.1:8799/wheel",
    )

    downloaded = bootstrap.download_wheel(options, tmp_path / "download")

    assert downloaded.name == "ficelle_pro-1.0-py3-none-any.whl"


def test_redirect_handler_refuses_cross_origin_with_license_key():
    handler = bootstrap.SameOriginRedirectHandler()
    request = bootstrap.urllib.request.Request(
        bootstrap.DEFAULT_WHEEL_URL,
        headers={"Authorization": "Bearer FICL-SECRET"},
    )

    try:
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://cdn.example.invalid/ficelle_pro-1.0-py3-none-any.whl",
        )
    except bootstrap.urllib.error.URLError as exc:
        assert "cross-origin" in str(exc.reason)
    else:
        raise AssertionError("expected a cross-origin redirect to be refused")


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


def test_run_license_activation_passes_key_via_env_not_argv(monkeypatch, tmp_path):
    options = make_options(tmp_path, license_key="SK-SECRET-999")
    captured = {}

    def fake_run(command, *, dry_run, env=None):
        captured["command"] = list(command)
        captured["env"] = env
        return bootstrap.CommandResult(list(command), 0)

    monkeypatch.setattr(bootstrap, "run_command", fake_run)
    bootstrap.run_license_activation(options, Path("/tmp/hermes-python"))

    assert captured["command"] == [str(Path("/tmp/hermes-python")), "-m", "ficelle", "license", "activate"]
    assert "SK-SECRET-999" not in " ".join(captured["command"])  # never on the command line (it is echoed)
    assert captured["env"]["FICELLE_LICENSE_KEY"] == "SK-SECRET-999"  # passed through the environment


def test_run_license_activation_skips_without_key(monkeypatch, tmp_path):
    options = make_options(tmp_path, license_key=None)
    calls = []
    monkeypatch.setattr(bootstrap, "run_command", lambda *a, **k: calls.append(1) or bootstrap.CommandResult([], 0))
    bootstrap.run_license_activation(options, Path("/tmp/hermes-python"))
    assert calls == []  # no license key → nothing to activate


def test_run_license_activation_is_best_effort_on_failure(monkeypatch, tmp_path, capsys):
    options = make_options(tmp_path, license_key="SK-x")
    monkeypatch.setattr(bootstrap, "run_command", lambda *a, **k: bootstrap.CommandResult([], 1))
    bootstrap.run_license_activation(options, Path("/tmp/hermes-python"))  # must not raise
    assert "activation did not complete" in capsys.readouterr().out
