"""Tests for the in-app Pro installer engine (ficelle.pro_install) and the `install-pro` CLI.

Core-only: pro_install never imports ficelle_pro, so these run without the closed pack. Download
and pip are injected, so nothing hits the network or actually installs anything.
"""
from __future__ import annotations

import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest

from ficelle import pro_install
from ficelle.install import CommandResult


@pytest.fixture(autouse=True)
def isolate_install_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pro_install, "install_status_path", lambda: tmp_path / "pro-install-status.json")
    monkeypatch.setattr(pro_install, "install_lock_path", lambda: tmp_path / "pro-install.lock")


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self, size: int = -1) -> bytes:
        if not self._data:
            return b""
        if size < 0:
            data, self._data = self._data, b""
            return data
        data, self._data = self._data[:size], self._data[size:]
        return data


def write_pro_wheel(path: Path, core_requirement: str | None = None) -> None:
    requirement = core_requirement or (
        f"ficelle-router=={pro_install.CORE_VERSION}"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "ficelle_pro-0.1.3.dist-info/METADATA",
            "Metadata-Version: 2.4\n"
            "Name: ficelle-pro\n"
            "Version: 0.1.3\n"
            f"Requires-Dist: {requirement}\n"
            "Requires-Dist: cryptography>=42\n",
        )


class _FakeHeaders:
    """Stands in for urllib's response.headers (an email.message.Message with get_filename)."""

    def __init__(self, filename: str | None) -> None:
        self._filename = filename

    def get_filename(self) -> str | None:
        return self._filename


class _FakeResponseWithHeaders(_FakeResponse):
    def __init__(self, data: bytes, filename: str | None) -> None:
        super().__init__(data)
        self.headers = _FakeHeaders(filename)


def test_download_pro_wheel_sends_bearer_key_and_saves(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def opener(request: object, timeout: float) -> _FakeResponse:
        captured["auth"] = request.get_header("Authorization")  # type: ignore[attr-defined]
        return _FakeResponse(b"WHEEL-BYTES")

    wheel = pro_install.download_pro_wheel("FICL-SECRET-1", tmp_path, opener=opener)
    assert wheel.exists() and wheel.read_bytes() == b"WHEEL-BYTES"
    assert captured["auth"] == "Bearer FICL-SECRET-1"


def test_download_pro_wheel_refuses_insecure_remote_url(tmp_path: Path, monkeypatch) -> None:
    # A wheel is pip-installed (arbitrary code) — a plain-http remote URL (MITM-tamperable) is refused.
    monkeypatch.setenv("FICELLE_WHEEL_URL", "http://evil.example/wheel")
    called = False

    def opener(request: object, timeout: float) -> "_FakeResponse":
        nonlocal called
        called = True
        return _FakeResponse(b"x")

    try:
        pro_install.download_pro_wheel("FICL-x", tmp_path, opener=opener)
    except pro_install.ProInstallError as exc:
        assert "insecure" in str(exc)
    else:
        raise AssertionError("expected ProInstallError for an insecure URL")
    assert called is False  # never even opened the connection
    # https remote and http-on-loopback (the dev mock) are both accepted.
    monkeypatch.setenv("FICELLE_WHEEL_URL", "https://install.ficelle.ai/wheel")
    pro_install.download_pro_wheel("FICL-x", tmp_path, opener=opener)
    monkeypatch.setenv("FICELLE_WHEEL_URL", "http://127.0.0.1:8799/wheel")
    pro_install.download_pro_wheel("FICL-x", tmp_path, opener=opener)


def test_download_pro_wheel_never_leaks_key_on_failure(tmp_path: Path) -> None:
    def opener(request: object, timeout: float) -> _FakeResponse:
        raise RuntimeError("boom while fetching FICL-SECRET-1")

    try:
        pro_install.download_pro_wheel("FICL-SECRET-1", tmp_path, opener=opener)
    except pro_install.ProInstallError as exc:
        assert "FICL-SECRET-1" not in str(exc)  # only the exception type name is surfaced
        assert "RuntimeError" in str(exc)
    else:
        raise AssertionError("expected ProInstallError")


def test_install_pro_wheel_invokes_pip_with_current_interpreter(tmp_path: Path) -> None:
    wheel = tmp_path / "ficelle_pro-latest.whl"
    write_pro_wheel(wheel)
    seen: list[list[str]] = []

    def runner(command: list[str]) -> int:
        seen.append(command)
        return 0

    pro_install.install_pro_wheel(wheel, runner=runner)
    assert seen == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "cryptography>=42",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            str(wheel),
        ],
    ]


def test_install_pro_wheel_raises_when_pip_fails(tmp_path: Path) -> None:
    wheel = tmp_path / "w.whl"
    write_pro_wheel(wheel)
    try:
        pro_install.install_pro_wheel(wheel, runner=lambda command: 1)
    except pro_install.ProInstallError:
        pass
    else:
        raise AssertionError("expected ProInstallError on pip failure")


def test_install_pro_wheel_falls_back_to_uv_when_pip_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    wheel = tmp_path / "ficelle_pro-1-py3-none-any.whl"
    write_pro_wheel(wheel)
    commands: list[list[str]] = []

    monkeypatch.setattr(pro_install, "_uv_executable", lambda: "/opt/homebrew/bin/uv")

    def runner(command: list[str]) -> CommandResult | int:
        commands.append(command)
        if command[0] == sys.executable:
            return CommandResult(command, 1, stderr=f"{sys.executable}: No module named pip")
        return 0

    pro_install.install_pro_wheel(wheel, runner=runner)

    assert commands[0] == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "cryptography>=42",
    ]
    assert commands[1] == [
        "/opt/homebrew/bin/uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "cryptography>=42",
    ]
    assert commands[2][:4] == [sys.executable, "-m", "pip", "install"]
    assert commands[3] == [
        "/opt/homebrew/bin/uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--reinstall",
        "--no-deps",
        str(wheel),
    ]


def test_uv_executable_honors_absolute_override_with_minimal_path(tmp_path: Path, monkeypatch) -> None:
    uv = tmp_path / "bin" / "uv"
    uv.parent.mkdir()
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o700)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("FICELLE_UV_BIN", str(uv))
    monkeypatch.setattr(pro_install.shutil, "which", lambda _name: None)

    assert pro_install._uv_executable() == str(uv)


def test_uv_executable_prefers_explicit_override_over_path(tmp_path: Path, monkeypatch) -> None:
    override = tmp_path / "override" / "uv"
    discovered = tmp_path / "path" / "uv"
    for executable in (override, discovered):
        executable.parent.mkdir()
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o700)
    monkeypatch.setenv("FICELLE_UV_BIN", str(override))
    monkeypatch.setattr(pro_install.shutil, "which", lambda _name: str(discovered))

    assert pro_install._uv_executable() == str(override)


def test_detached_install_command_uses_systemd_scope(tmp_path: Path, monkeypatch) -> None:
    systemd_run = tmp_path / "systemd-run"
    systemd_run.write_text("#!/bin/sh\nexit 0\n")
    systemd_run.chmod(0o700)
    monkeypatch.setenv("INVOCATION_ID", "systemd-service")
    monkeypatch.setattr(
        pro_install.shutil,
        "which",
        lambda name: str(systemd_run) if name == "systemd-run" else None,
    )

    command = pro_install.detached_install_command()

    assert command[:5] == [
        str(systemd_run),
        "--user",
        "--scope",
        "--quiet",
        "--collect",
    ]
    assert command[5].startswith("--unit=ficelle-pro-install-")
    assert command[6:] == [sys.executable, "-m", "ficelle", "install-pro"]


def test_install_status_roundtrip_is_private(tmp_path: Path) -> None:
    pro_install.write_install_status("failed", "Package install failed.")

    assert pro_install.read_install_status()["status"] == "failed"
    assert pro_install.read_install_status()["message"] == "Package install failed."
    assert pro_install.install_status_path().stat().st_mode & 0o777 == 0o600


def test_install_status_concurrent_writes_remain_valid() -> None:
    statuses = ["queued", "installing", "restarting", "installed", "failed"]

    with ThreadPoolExecutor(max_workers=len(statuses)) as executor:
        list(executor.map(pro_install.write_install_status, statuses))

    assert pro_install.read_install_status()["status"] in statuses


def test_queue_install_rejects_a_second_active_attempt() -> None:
    pro_install.queue_install()

    with pytest.raises(pro_install.ProInstallInProgress):
        pro_install.queue_install()

    assert pro_install.read_install_status()["status"] == "queued"


def test_queue_install_recovers_an_abandoned_stale_attempt(monkeypatch) -> None:
    pro_install.write_install_status("installing")
    status = pro_install.read_install_status()
    monkeypatch.setattr(
        pro_install.time,
        "time",
        lambda: (
            float(status["updated_at"])
            + pro_install.ACTIVE_INSTALL_STATUS_TTL_SECONDS
            + 1
        ),
    )

    pro_install.queue_install()

    assert pro_install.read_install_status()["status"] == "queued"


def test_queue_install_rejects_while_real_installer_lock_is_held() -> None:
    pro_install.write_install_status("installed")

    with pro_install.exclusive_install_lock():
        with pytest.raises(pro_install.ProInstallInProgress):
            pro_install.queue_install()

    assert pro_install.read_install_status()["status"] == "installed"


def test_install_status_failure_is_explicit(monkeypatch) -> None:
    def fail_write(_path, _payload):
        raise OSError("disk unavailable")

    monkeypatch.setattr(pro_install, "atomic_write_json", fail_write)

    with pytest.raises(OSError):
        pro_install.write_install_status("installing")
    assert pro_install.best_effort_write_install_status("failed") is False


def test_install_pro_downloads_then_pips_that_wheel(monkeypatch) -> None:
    installed: dict[str, str] = {}

    def opener(request: object, timeout: float) -> _FakeResponse:
        return _FakeResponse(b"WHEEL")

    def runner(command: list[str]) -> int:
        installed["wheel"] = command[-1]  # the last arg is the wheel path
        return 0

    monkeypatch.setattr(
        pro_install,
        "verify_pro_core_compatibility",
        lambda wheel: None,
    )
    pro_install.install_pro("FICL-x", opener=opener, runner=runner)
    # Regression: the download used to be saved as the invalid "ficelle_pro-latest.whl", which pip
    # rejects ("Invalid wheel filename"); it must now be a valid wheel name.
    assert installed["wheel"].endswith("-py3-none-any.whl")


def test_install_pro_verifies_pinned_checksum(monkeypatch) -> None:
    import hashlib

    body = b"WHEEL-CONTENT"
    installed: list[list[str]] = []

    def opener(request: object, timeout: float) -> "_FakeResponse":
        return _FakeResponse(body)

    def runner(command: list[str]) -> int:
        installed.append(command)
        return 0

    monkeypatch.setattr(
        pro_install,
        "verify_pro_core_compatibility",
        lambda wheel: None,
    )
    # A matching pin installs.
    monkeypatch.setenv("FICELLE_WHEEL_SHA256", hashlib.sha256(body).hexdigest())
    pro_install.install_pro("FICL-x", opener=opener, runner=runner)
    assert len(installed) == 2
    assert installed[0][-1] == "cryptography>=42"
    assert installed[1][-1].endswith("-py3-none-any.whl")
    # A wrong pin is refused BEFORE pip runs.
    monkeypatch.setenv("FICELLE_WHEEL_SHA256", "0" * 64)
    installed.clear()
    try:
        pro_install.install_pro("FICL-x", opener=opener, runner=runner)
    except pro_install.ProInstallError as exc:
        assert "checksum" in str(exc)
    else:
        raise AssertionError("expected ProInstallError on checksum mismatch")
    assert installed == []  # pip never ran on the tampered wheel


def test_pro_wheel_must_match_the_installed_core(tmp_path: Path) -> None:
    compatible = tmp_path / "ficelle_pro-0.1.3-py3-none-any.whl"
    write_pro_wheel(compatible)
    pro_install.verify_pro_core_compatibility(compatible)

    incompatible = tmp_path / "ficelle_pro-0.2.0-py3-none-any.whl"
    write_pro_wheel(incompatible, "ficelle-router==0.2.0")
    with pytest.raises(pro_install.ProInstallError, match="incompatible"):
        pro_install.verify_pro_core_compatibility(incompatible)


def test_install_pro_requires_a_key() -> None:
    try:
        pro_install.install_pro("")
    except pro_install.ProInstallError:
        pass
    else:
        raise AssertionError("expected ProInstallError for an empty key")


def test_pro_wheel_url_honors_env_override(monkeypatch) -> None:
    monkeypatch.delenv("FICELLE_WHEEL_URL", raising=False)
    assert pro_install.pro_wheel_url() == pro_install.DEFAULT_WHEEL_URL
    monkeypatch.setenv("FICELLE_WHEEL_URL", "http://127.0.0.1:9/wheel")
    assert pro_install.pro_wheel_url() == "http://127.0.0.1:9/wheel"


def test_install_pro_cli_reads_env_key_then_restarts(monkeypatch) -> None:
    from ficelle import cli

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        pro_install,
        "install_pro",
        lambda key, **kw: seen.update(
            key=key,
            key_in_install_env="FICELLE_LICENSE_KEY" in os.environ,
        ),
    )

    class _Backend:
        def restart(self) -> int:
            seen["restarted"] = True
            seen["key_in_restart_env"] = "FICELLE_LICENSE_KEY" in os.environ
            return 0

    monkeypatch.setattr(cli, "current_service_backend", lambda *a, **k: _Backend())
    monkeypatch.setattr(
        pro_install,
        "exclusive_install_lock",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setenv("FICELLE_LICENSE_KEY", "FICL-ENV-KEY")
    assert cli.main(["install-pro"]) == 0
    assert seen["key"] == "FICL-ENV-KEY" and seen["restarted"] is True
    assert seen["key_in_install_env"] is False
    assert seen["key_in_restart_env"] is False
    assert pro_install.read_install_status()["status"] == "installed"


def test_install_pro_cli_waits_for_lock_when_queued_by_admin(monkeypatch) -> None:
    from ficelle import cli

    seen: dict[str, object] = {}

    def lock(*, wait: bool = False):
        seen["wait"] = wait
        return nullcontext()

    monkeypatch.setattr(pro_install, "exclusive_install_lock", lock)
    monkeypatch.setattr(pro_install, "install_pro", lambda key, **kwargs: None)
    monkeypatch.setenv("FICELLE_LICENSE_KEY", "FICL-ENV-KEY")
    monkeypatch.setenv("FICELLE_INSTALL_QUEUED", "1")

    class _Backend:
        def restart(self) -> int:
            return 0

    monkeypatch.setattr(cli, "current_service_backend", lambda *args, **kwargs: _Backend())

    assert cli.main(["install-pro"]) == 0
    assert seen["wait"] is True
    assert "FICELLE_INSTALL_QUEUED" not in os.environ


def test_install_pro_cli_without_key_exits_2(monkeypatch) -> None:
    from ficelle import cli

    monkeypatch.delenv("FICELLE_LICENSE_KEY", raising=False)
    assert cli.main(["install-pro"]) == 2
    assert pro_install.read_install_status()["status"] == "failed"


def test_install_pro_cli_persists_restart_failure(monkeypatch) -> None:
    from ficelle import cli

    monkeypatch.setattr(pro_install, "install_pro", lambda key, **kwargs: None)
    monkeypatch.setattr(
        pro_install,
        "exclusive_install_lock",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setenv("FICELLE_LICENSE_KEY", "FICL-ENV-KEY")

    class _Backend:
        def restart(self) -> int:
            raise RuntimeError("service manager unavailable")

    monkeypatch.setattr(cli, "current_service_backend", lambda *args, **kwargs: _Backend())

    assert cli.main(["install-pro"]) == 1
    status = pro_install.read_install_status()
    assert status["status"] == "failed"
    assert "service manager unavailable" not in status["message"]


def test_install_pro_cli_persists_failure_before_releasing_lock(monkeypatch) -> None:
    from ficelle import cli

    events: list[str] = []

    @contextmanager
    def observed_lock(*, wait: bool = False):
        events.append("locked")
        try:
            yield
        finally:
            events.append("unlocked")

    def record_status(status: str, message: str | None = None) -> bool:
        events.append(status)
        return True

    def fail_install(key: str, **kwargs: object) -> None:
        raise pro_install.ProInstallError("install failed")

    monkeypatch.setenv("FICELLE_LICENSE_KEY", "FICL-ENV-KEY")
    monkeypatch.setattr(pro_install, "exclusive_install_lock", observed_lock)
    monkeypatch.setattr(pro_install, "best_effort_write_install_status", record_status)
    monkeypatch.setattr(pro_install, "install_pro", fail_install)

    assert cli.main(["install-pro"]) == 1
    assert events == ["locked", "installing", "failed", "unlocked"]


def test_install_pro_cli_stops_before_install_when_status_cannot_persist(monkeypatch) -> None:
    from ficelle import cli

    def unexpected_install(key: str, **kwargs: object) -> None:
        raise AssertionError("install must not start")

    monkeypatch.setenv("FICELLE_LICENSE_KEY", "FICL-ENV-KEY")
    monkeypatch.setattr(
        pro_install,
        "exclusive_install_lock",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(pro_install, "best_effort_write_install_status", lambda status, message=None: False)
    monkeypatch.setattr(pro_install, "install_pro", unexpected_install)

    assert cli.main(["install-pro"]) == 1


def test_install_pro_lock_rejects_a_concurrent_helper(tmp_path: Path) -> None:
    lock_path = tmp_path / "pro-install.lock"
    with pro_install.exclusive_install_lock(lock_path):
        try:
            with pro_install.exclusive_install_lock(lock_path):
                raise AssertionError("concurrent helper unexpectedly acquired the lock")
        except pro_install.ProInstallInProgress:
            pass


def test_download_saves_under_the_services_advertised_wheel_name(tmp_path: Path) -> None:
    # The service sends the real wheel name via Content-Disposition; the download must keep it,
    # because pip rejects a wheel file whose name isn't a valid wheel filename.
    def opener(request: object, timeout: float) -> _FakeResponseWithHeaders:
        return _FakeResponseWithHeaders(b"WHEEL", "ficelle_pro-1.2.3-py3-none-any.whl")

    wheel = pro_install.download_pro_wheel("FICL-x", tmp_path, opener=opener)
    assert wheel.name == "ficelle_pro-1.2.3-py3-none-any.whl"
    assert wheel.read_bytes() == b"WHEEL"


def test_download_falls_back_to_a_valid_wheel_name_without_a_header(tmp_path: Path) -> None:
    # No usable Content-Disposition (e.g. a bare response) -> a generic but pip-valid wheel name,
    # never the old invalid "ficelle_pro-latest.whl".
    def opener(request: object, timeout: float) -> _FakeResponse:
        return _FakeResponse(b"WHEEL")

    wheel = pro_install.download_pro_wheel("FICL-x", tmp_path, opener=opener)
    assert wheel.name.endswith("-py3-none-any.whl") and wheel.name.count("-") >= 4


def test_download_ignores_a_path_in_the_content_disposition(tmp_path: Path) -> None:
    # A header value must never escape dest_dir: only its last path segment is used.
    def opener(request: object, timeout: float) -> _FakeResponseWithHeaders:
        return _FakeResponseWithHeaders(b"WHEEL", "../../../../tmp/evil-1-py3-none-any.whl")

    wheel = pro_install.download_pro_wheel("FICL-x", tmp_path, opener=opener)
    assert wheel.parent == tmp_path
    assert wheel.name == "evil-1-py3-none-any.whl"


def test_download_falls_back_on_an_unsafe_advertised_name(tmp_path: Path) -> None:
    # A control char/NUL (would crash the write) or an over-long name must degrade to the safe
    # fallback, never propagate to wheel.write_bytes — so the install can't crash on a bad header.
    for bad in ("ficelle_pro-1.0\x00-py3-none-any.whl", "a" * 300 + ".whl", "no-extension", "up.WHL"):

        def opener(request: object, timeout: float, _bad: str = bad) -> _FakeResponseWithHeaders:
            return _FakeResponseWithHeaders(b"WHEEL", _bad)

        wheel = pro_install.download_pro_wheel("FICL-x", tmp_path, opener=opener)
        assert wheel.name == pro_install._FALLBACK_WHEEL_NAME
        assert wheel.exists()  # the write always succeeds under the safe fallback name


def test_download_falls_back_on_a_non_wheel_filename(tmp_path: Path) -> None:
    def opener(request: object, timeout: float) -> _FakeResponseWithHeaders:
        return _FakeResponseWithHeaders(b"WHEEL", "ficelle-pro.whl")

    wheel = pro_install.download_pro_wheel("FICL-x", tmp_path, opener=opener)
    assert wheel.name == pro_install._FALLBACK_WHEEL_NAME


def test_download_falls_back_on_a_malformed_wheel_filename(tmp_path: Path) -> None:
    def opener(request: object, timeout: float) -> _FakeResponseWithHeaders:
        return _FakeResponseWithHeaders(b"WHEEL", "ficelle_pro-1.0-a-b-c-d-e.whl")

    wheel = pro_install.download_pro_wheel("FICL-x", tmp_path, opener=opener)
    assert wheel.name == pro_install._FALLBACK_WHEEL_NAME


def test_redirect_handler_refuses_cross_origin_with_bearer_key() -> None:
    handler = pro_install._SameOriginRedirectHandler()
    request = pro_install.urllib.request.Request(
        "https://install.ficelle.ai/wheel",
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
    except pro_install.ProInstallError as exc:
        assert "cross-origin" in str(exc)
    else:
        raise AssertionError("expected ProInstallError for a cross-origin redirect")


def test_redirect_handler_accepts_an_explicit_default_https_port() -> None:
    handler = pro_install._SameOriginRedirectHandler()
    request = pro_install.urllib.request.Request(
        "https://install.ficelle.ai/wheel",
        headers={"Authorization": "Bearer FICL-SECRET"},
    )
    redirected = handler.redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://install.ficelle.ai:443/ficelle_pro-1.0-py3-none-any.whl",
    )
    assert redirected is not None


def test_download_rejects_an_oversized_wheel(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pro_install, "MAX_PRO_WHEEL_BYTES", 4)

    def opener(request: object, timeout: float) -> _FakeResponse:
        return _FakeResponse(b"12345")

    try:
        pro_install.download_pro_wheel("FICL-x", tmp_path, opener=opener)
    except pro_install.ProInstallError as exc:
        assert "maximum allowed size" in str(exc)
    else:
        raise AssertionError("expected ProInstallError for an oversized wheel")
    assert list(tmp_path.iterdir()) == []
