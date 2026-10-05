from __future__ import annotations

import hashlib
import io
import json
import subprocess
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pytest

import ficelle
from ficelle import license_ops, update
from ficelle.runtime_paths import RuntimePaths


# The manifests below must describe a release *newer* than the installed one, or the
# checker legitimately reports up_to_date. Derive it so a version bump does not turn
# these tests red.
_MAJOR, _MINOR, _PATCH = ficelle.__version__.split(".")
NEXT_VERSION = f"{_MAJOR}.{_MINOR}.{int(_PATCH) + 1}"


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            body, self.body = self.body, b""
            return body
        body, self.body = self.body[:size], self.body[size:]
        return body


@pytest.fixture(autouse=True)
def isolate_update_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(update, "_RUNTIME_PATHS", RuntimePaths.from_env(environ={"FICELLE_HOME": str(tmp_path)}))


@pytest.fixture(autouse=True)
def isolate_license(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read this machine's real entitlement, nor renew it against the real service."""

    def refuse_real_refresh(*_args: object) -> None:
        raise AssertionError("a test reached the real license service")

    monkeypatch.setattr(license_ops, "ENTITLEMENT_PATH", tmp_path / "entitlement.token")
    monkeypatch.setattr(license_ops, "MACHINE_ID_PATH", tmp_path / "machine-id")
    try:
        licensing = license_ops._licensing()
    except license_ops.ProPackUnavailable:
        return
    monkeypatch.setattr(licensing, "_default_poster", refuse_real_refresh)


def compact_manifest(version: str = NEXT_VERSION) -> dict[str, object]:
    return {
        "version": version,
        "channel": "stable",
        "release_url": "https://ficelle.ai/releases/" + version,
        "notes": "Bug fixes",
        "core": {
            "wheel_url": f"https://downloads.ficelle.ai/ficelle_router-{version}-py3-none-any.whl",
            "sha256": "a" * 64,
        },
    }


def test_parse_compact_manifest_validates_artifact_and_version() -> None:
    manifest = update.parse_release_manifest(
        compact_manifest(),
        source_url="https://install.ficelle.ai/api/releases/latest/core",
    )

    assert manifest.version == NEXT_VERSION
    assert manifest.core.distribution == "ficelle-router"
    assert manifest.core.filename == f"ficelle_router-{NEXT_VERSION}-py3-none-any.whl"
    assert manifest.core.sha256 == "a" * 64


def test_parse_github_release_uses_asset_digest() -> None:
    payload = {
        "tag_name": f"v{NEXT_VERSION}",
        "html_url": f"https://github.com/TheBlueHouse75/ficelle-open-core/releases/tag/v{NEXT_VERSION}",
        "body": "Release notes",
        "assets": [
            {
                "name": f"ficelle_router-{NEXT_VERSION}-py3-none-any.whl",
                "browser_download_url": f"https://github.com/TheBlueHouse75/ficelle-open-core/releases/download/v{NEXT_VERSION}/ficelle_router-{NEXT_VERSION}-py3-none-any.whl",
                "digest": "sha256:" + "b" * 64,
            }
        ],
    }

    manifest = update.parse_release_manifest(
        payload,
        source_url=update.DEFAULT_UPDATE_MANIFEST_URL,
    )

    assert manifest.version == NEXT_VERSION
    assert manifest.core.sha256 == "b" * 64


def test_check_for_updates_persists_available_status_without_exposing_artifact_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    update._write_status(
        {
            "pro_wheel_url": "https://install.ficelle.ai/old-pro.whl",
            "pro_sha256": "e" * 64,
            "pro_filename": "ficelle_pro-0.1.3-py3-none-any.whl",
            "pro_authorization": "entitlement",
        }
    )
    payload = json.dumps(compact_manifest()).encode()
    monkeypatch.setattr(update, "manifest_url", lambda: "https://install.ficelle.ai/manifest.json")
    monkeypatch.setattr(update, "_is_pro_installed", lambda: False)

    def opener(_request: object, _timeout: float) -> _Response:
        return _Response(payload)

    status = update.check_for_updates(force=True, opener=opener)

    assert status["status"] == "available"
    assert status["update_available"] is True
    assert status["latest_version"] == NEXT_VERSION
    assert update.public_update_status()["release_notes"] == "Bug fixes"
    assert "core_wheel_url" not in update.public_update_status()
    assert update.public_update_status()["pro_artifact_available"] is False
    assert update.update_status_path().stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("authorization", ["none", "bearer", "entitlement"])
def test_check_for_updates_accepts_available_pro_authorization_without_refresh(
    monkeypatch: pytest.MonkeyPatch, authorization: str,
) -> None:
    payload = compact_manifest()
    payload["pro"] = {
        "wheel_url": f"https://install.ficelle.ai/ficelle_pro-{NEXT_VERSION}-py3-none-any.whl",
        "sha256": "d" * 64,
        "authorization": authorization,
    }
    monkeypatch.setattr(update, "manifest_url", lambda: "https://install.ficelle.ai/manifest.json")
    monkeypatch.setattr(update, "_is_pro_installed", lambda: True)
    monkeypatch.setattr(license_ops, "service_url", lambda: "https://install.ficelle.ai")
    monkeypatch.setattr(license_ops, "is_entitled", lambda: True)
    monkeypatch.setattr(license_ops, "refresh", lambda: pytest.fail("live or managed authorization must not refresh"))
    monkeypatch.setattr(license_ops, "cached_entitlement_token", lambda: "signed-entitlement")
    monkeypatch.setenv("FICELLE_UPDATE_PRO_TOKEN", "managed-update-token")

    status = update.check_for_updates(
        force=True,
        opener=lambda _request, _timeout: _Response(json.dumps(payload).encode()),
    )

    assert status["pro_update_required"] is False
    assert update.public_update_status()["pro_artifact_available"] is True


def _entitlement_pro_manifest() -> bytes:
    payload = compact_manifest()
    payload["pro"] = {
        "wheel_url": f"https://install.ficelle.ai/ficelle_pro-{NEXT_VERSION}-py3-none-any.whl",
        "sha256": "d" * 64,
        "authorization": "entitlement",
    }
    return json.dumps(payload).encode()


@pytest.mark.parametrize("entrypoint", ["check", "apply"])
@pytest.mark.parametrize("refreshed_status", ["active", "canceled"])
def test_lapsed_pro_update_uses_only_an_active_renewed_entitlement(
    monkeypatch: pytest.MonkeyPatch, entrypoint: str, refreshed_status: str,
) -> None:
    """A successful refresh can sign a canceled subscription; neither path may download it."""
    licensing = pytest.importorskip("ficelle_pro.licensing")
    private_key, public_key = licensing.generate_dev_keypair()
    monkeypatch.setattr(licensing, "LICENSE_PUBLIC_KEY_B64", public_key)
    now = 1_000_000.0
    monkeypatch.setattr(licensing.time, "time", lambda: now)
    payload = {
        "product": licensing.PRODUCT,
        "status": "active",
        "machine_activation_id": "review-activation",
        "issued_at": now - 100,
        "expires_at": now - 50,
        "grace_deadline": now - 1,
    }
    licensing.store_entitlement_token(license_ops.ENTITLEMENT_PATH, licensing.sign_payload(payload, private_key))
    payload.update({"status": refreshed_status, "issued_at": now, "expires_at": now + 100, "grace_deadline": now + 200})
    renewed_token = licensing.sign_payload(payload, private_key)
    refresh_calls: list[str] = []

    def poster(url: str, body: dict[str, object], timeout: float) -> tuple[int, dict[str, str]]:
        refresh_calls.append(url)
        assert body["machine_activation_id"] == "review-activation"
        assert "license_key" not in body
        assert timeout == 15.0
        return 200, {"entitlement": renewed_token}

    monkeypatch.setattr(licensing, "_default_poster", poster)
    monkeypatch.setattr(update, "manifest_url", lambda: "https://install.ficelle.ai/manifest.json")
    monkeypatch.setattr(update, "_is_pro_installed", lambda: True)
    monkeypatch.setattr(license_ops, "service_url", lambda: "https://install.ficelle.ai")
    manifest_payload = json.loads(_entitlement_pro_manifest())
    manifest_payload["core"]["sha256"] = hashlib.sha256(b"core-wheel").hexdigest()
    manifest_bytes = json.dumps(manifest_payload).encode()
    downloads: list[str | None] = []

    if entrypoint == "check":
        status = update.check_for_updates(force=True, opener=lambda _request, _timeout: _Response(manifest_bytes))
        assert status["pro_update_required"] is (refreshed_status != "active")
        assert update.check_for_updates(opener=lambda *_args: pytest.fail("cached check must not use the network")) == status
    else:
        manifest = update.parse_release_manifest(manifest_payload, source_url=update.manifest_url())
        update._write_status({**manifest.internal_dict(), "status": "queued", "update_available": True})
        monkeypatch.setattr(update, "_managed_service", lambda: (object(), True))

        def download(request: urllib.request.Request, _timeout: float) -> _Response:
            downloads.append(request.get_header("Authorization"))
            # A bad Pro checksum stops before package or service mutation.
            return _Response(b"core-wheel")

        assert update.apply_update(opener=download) == 1
        status = update.read_update_status()
        assert downloads == ([None, f"Bearer {renewed_token}"] if refreshed_status == "active" else [])
        if refreshed_status == "active":
            assert status["message"] == "release artifact checksum mismatch"

    if refreshed_status != "active":
        assert "not active" in status["message"]
        assert "ficelle license activate" in status["message"]
    assert refresh_calls == ["https://install.ficelle.ai/api/license/refresh"]
    assert license_ops.cached_entitlement_token() == renewed_token
    assert renewed_token not in json.dumps(status)


def test_a_revoked_license_is_named_at_check_time_with_the_way_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """MacBook, 05/10/2026: the activation had been deleted on the service, the check still said
    "update available", and every install ended on `download failed (HTTPError)`."""

    def rejected() -> None:
        raise license_ops.LicenseOperationError(
            "license service rejected the request: license no longer exists; Bearer refresh-secret-123 " + "x" * 300
        )

    monkeypatch.setattr(update, "manifest_url", lambda: "https://install.ficelle.ai/manifest.json")
    monkeypatch.setattr(update, "_is_pro_installed", lambda: True)
    monkeypatch.setattr(license_ops, "service_url", lambda: "https://install.ficelle.ai")
    monkeypatch.setattr(license_ops, "is_entitled", lambda now=None: False)
    monkeypatch.setattr(license_ops, "refresh", rejected)

    status = update.check_for_updates(force=True, opener=lambda _request, _timeout: _Response(_entitlement_pro_manifest()))

    assert status["pro_update_required"] is True
    assert "license no longer exists" in status["message"]
    assert "ficelle license activate" in status["message"]
    assert "refresh-secret-123" not in status["message"]
    assert len(status["message"]) <= 240
    assert status["message"].endswith("then retry the update")
    assert update.apply_update(opener=lambda *_args: pytest.fail("refused license must not download")) == 1
    assert update.read_update_status()["message"] == status["message"]


@pytest.mark.parametrize("entrypoint", ["check", "apply"])
def test_license_refresh_cache_failure_returns_an_actionable_update_status(
    monkeypatch: pytest.MonkeyPatch, entrypoint: str,
) -> None:
    def unwritable() -> None:
        raise PermissionError("private runtime path")

    monkeypatch.setattr(update, "manifest_url", lambda: "https://install.ficelle.ai/manifest.json")
    monkeypatch.setattr(update, "_is_pro_installed", lambda: True)
    monkeypatch.setattr(license_ops, "service_url", lambda: "https://install.ficelle.ai")
    monkeypatch.setattr(license_ops, "is_entitled", lambda: False)
    monkeypatch.setattr(license_ops, "refresh", unwritable)
    if entrypoint == "check":
        status = update.check_for_updates(force=True, opener=lambda *_args: _Response(_entitlement_pro_manifest()))
        assert status["pro_update_required"] is True
    else:
        manifest = update.parse_release_manifest(json.loads(_entitlement_pro_manifest()), source_url=update.manifest_url())
        update._write_status({**manifest.internal_dict(), "status": "queued", "update_available": True})
        assert update.apply_update(opener=lambda *_args: pytest.fail("uncached license must not download")) == 1
        status = update.read_update_status()
    assert "permissions" in status["message"]
    assert "private runtime path" not in status["message"]


@pytest.mark.parametrize("authorization", ["entitlement", "bearer"])
@pytest.mark.parametrize("body_kind", ["json", "truncated"])
def test_a_refused_pro_download_says_why_and_what_to_do(
    tmp_path: Path, authorization: str, body_kind: str,
) -> None:
    credential = "signed-entitlement" if authorization == "entitlement" else "FICL-review-token-secret"
    body = (
        json.dumps({"error": f"unknown update authorization {credential}; sk-reviewSecret1234 " + "x" * 300}).encode()
        if body_kind == "json"
        else b'{"error": "' + b"x" * 4096
    )

    def refuse(request: object, _timeout: float) -> object:
        raise urllib.error.HTTPError(
            "https://install.ficelle.ai/api/releases/latest/wheel",
            404,
            f"Not Found {credential}",
            {},
            io.BytesIO(body),
        )

    pro = update.ReleaseArtifact(
        distribution="ficelle-pro",
        version=NEXT_VERSION,
        url="https://install.ficelle.ai/api/releases/latest/wheel",
        sha256="d" * 64,
        filename=f"ficelle_pro-{NEXT_VERSION}-py3-none-any.whl",
        authorization=authorization,
    )

    with pytest.raises(update.UpdateUnavailable) as refused:
        update._download_artifact(pro, tmp_path, authorization_token=credential, opener=refuse)
    message = str(refused.value)
    assert "HTTP 404" in message
    assert ("unknown update authorization" if body_kind == "json" else "Not Found") in message
    assert credential not in message and "sk-reviewSecret1234" not in message
    assert len(message) <= 240
    assert message.endswith("then retry the update")
    assert not (tmp_path / pro.filename).exists()
    assert refused.value.__cause__.closed
    if authorization == "entitlement":
        assert "ficelle license refresh" in message
        assert "ficelle license activate" in message
    else:
        assert "FICELLE_UPDATE_PRO_TOKEN" in message
        assert "ficelle license" not in message
    # The public Core artifact has no license to blame: it keeps the plain transport error.
    with pytest.raises(update.UpdateError) as failed:
        update._download_artifact(pro, tmp_path, opener=refuse)
    assert not isinstance(failed.value, update.UpdateUnavailable)
    assert str(failed.value) == "release artifact download failed (HTTP 404)"


def test_check_for_updates_blocks_pro_artifact_from_another_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = compact_manifest()
    payload["pro"] = {
        "wheel_url": f"https://downloads.ficelle.ai/ficelle_pro-{NEXT_VERSION}-py3-none-any.whl",
        "sha256": "d" * 64,
        "authorization": "entitlement",
    }
    monkeypatch.setattr(update, "manifest_url", lambda: "https://install.ficelle.ai/manifest.json")
    monkeypatch.setattr(update, "_is_pro_installed", lambda: True)
    monkeypatch.setattr(license_ops, "service_url", lambda: "https://install.ficelle.ai")
    monkeypatch.setattr(license_ops, "cached_entitlement_token", lambda: "signed-entitlement")

    status = update.check_for_updates(
        force=True,
        opener=lambda _request, _timeout: _Response(json.dumps(payload).encode()),
    )

    assert status["pro_update_required"] is True
    assert "origin does not match the license service" in status["message"]


def test_check_failure_without_writable_status_returns_a_safe_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(update, "manifest_url", lambda: "https://install.ficelle.ai/manifest.json")

    def fail_write(_path: Path, _payload: object) -> None:
        raise PermissionError()

    monkeypatch.setattr(update, "atomic_write_json", fail_write)

    status = update.check_for_updates(
        force=True,
        opener=lambda _request, _timeout: _Response(b"not-json"),
    )

    assert status["status"] == "error"
    assert "invalid JSON" in status["message"]


def test_forced_check_does_not_overwrite_an_active_update() -> None:
    update._write_status({"status": "queued", "update_available": True})

    status = update.check_for_updates(
        force=True,
        opener=lambda _request, _timeout: pytest.fail("active update must not be rechecked"),
    )

    assert status["status"] == "queued"


def test_queue_update_preserves_verified_artifact_details(monkeypatch: pytest.MonkeyPatch) -> None:
    update._write_status(
        {
            **compact_manifest()["core"],
            "status": "available",
            "update_available": True,
            "latest_version": "0.1.4",
            "core_wheel_url": compact_manifest()["core"]["wheel_url"],
            "core_sha256": compact_manifest()["core"]["sha256"],
            "core_filename": "ficelle_router-0.1.4-py3-none-any.whl",
        }
    )
    monkeypatch.setattr(update, "_is_pro_installed", lambda: False)

    status = update.queue_update()

    assert status["status"] == "queued"
    queued = update.read_update_status()
    assert queued["core_wheel_url"].startswith("https://downloads.ficelle.ai/")
    assert queued["core_sha256"] == "a" * 64


@pytest.mark.parametrize("authorization", ["entitlement", "bearer"])
def test_install_retry_rechecks_repaired_pro_authorization(
    monkeypatch: pytest.MonkeyPatch, authorization: str,
) -> None:
    payload = json.loads(_entitlement_pro_manifest())
    payload["pro"]["authorization"] = authorization
    monkeypatch.setattr(update, "manifest_url", lambda: "https://install.ficelle.ai/manifest.json")
    monkeypatch.setattr(update, "_open_secure_url", lambda *_args: _Response(json.dumps(payload).encode()))
    monkeypatch.setattr(update, "_is_pro_installed", lambda: True)
    monkeypatch.setattr(license_ops, "service_url", lambda: "https://install.ficelle.ai")
    monkeypatch.setattr(license_ops, "is_entitled", lambda: True)
    monkeypatch.setattr(license_ops, "refresh", lambda: pytest.fail("authorization retry must use the repaired credential"))
    token: list[str | None] = [None]
    monkeypatch.setattr(license_ops, "cached_entitlement_token", lambda: token[0])
    monkeypatch.delenv("FICELLE_UPDATE_PRO_TOKEN", raising=False)
    assert update.check_for_updates(force=True)["pro_update_required"] is True

    token[0] = "repaired-entitlement"
    monkeypatch.setenv("FICELLE_UPDATE_PRO_TOKEN", "repaired-managed-token")
    queued = update.queue_update()

    assert queued["status"] == "queued"
    assert queued["pro_update_required"] is False


def test_download_artifact_verifies_sha256(tmp_path: Path) -> None:
    body = b"wheel-bytes"
    artifact = update.ReleaseArtifact(
        distribution="ficelle-router",
        version="0.1.4",
        url="https://downloads.ficelle.ai/ficelle_router-0.1.4-py3-none-any.whl",
        sha256=hashlib.sha256(body).hexdigest(),
        filename="ficelle_router-0.1.4-py3-none-any.whl",
    )

    path = update._download_artifact(
        artifact,
        tmp_path,
        opener=lambda _request, _timeout: _Response(body),
    )

    assert path.read_bytes() == body


def test_download_artifact_rejects_checksum_mismatch(tmp_path: Path) -> None:
    body = b"tampered"
    artifact = update.ReleaseArtifact(
        distribution="ficelle-router",
        version="0.1.4",
        url="https://downloads.ficelle.ai/ficelle_router-0.1.4-py3-none-any.whl",
        sha256="c" * 64,
        filename="ficelle_router-0.1.4-py3-none-any.whl",
    )

    with pytest.raises(update.UpdateError, match="checksum"):
        update._download_artifact(
            artifact,
            tmp_path,
            opener=lambda _request, _timeout: _Response(body),
        )


def test_pro_artifact_must_target_the_new_core_version(tmp_path: Path) -> None:
    wheel = tmp_path / "ficelle_pro-0.1.4-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "ficelle_pro-0.1.4.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: ficelle-pro\nRequires-Dist: ficelle-router==0.1.3\n",
        )

    with pytest.raises(update.UpdateError, match="incompatible"):
        update._verify_pro_core_compatibility(wheel, core_version="0.1.4")


def test_update_manifest_refuses_insecure_urls() -> None:
    payload = compact_manifest()
    payload["core"] = {
        "wheel_url": "http://downloads.ficelle.ai/ficelle_router-0.1.4-py3-none-any.whl",
        "sha256": "a" * 64,
    }

    with pytest.raises(update.UpdateError, match="HTTPS"):
        update.parse_release_manifest(payload, source_url="https://install.ficelle.ai/manifest.json")


def test_update_manifest_reports_malformed_urls_as_update_errors() -> None:
    payload = compact_manifest()
    payload["release_url"] = "https://[invalid"

    with pytest.raises(update.UpdateError, match="HTTPS"):
        update.parse_release_manifest(payload, source_url="https://install.ficelle.ai/manifest.json")


def test_update_commands_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    command = ["fake-installer"]

    def timeout_run(*_args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(command, float(kwargs["timeout"]))

    monkeypatch.setattr(update.subprocess, "run", timeout_run)
    result = update._run_command(command)

    assert result.returncode == 124
    assert "timed out" in (result.stderr or "")


def test_update_installs_wheels_without_dependency_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = tmp_path / "ficelle_router.whl"
    wheel.write_bytes(b"wheel")
    commands: list[list[str]] = []
    monkeypatch.setattr(
        update,
        "_run_command",
        lambda command: commands.append(command)
        or subprocess.CompletedProcess(command, 0, "", ""),
    )

    update._install_wheel(wheel)

    assert commands == [[
        update.sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        str(wheel),
    ]]


def test_update_dependency_check_falls_back_to_uv_without_pip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[:3] == [update.sys.executable, "-m", "pip"]:
            return subprocess.CompletedProcess(
                command,
                1,
                "",
                "No module named pip",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(update, "_run_command", fake_run)
    monkeypatch.setattr(update, "_uv_executable", lambda: "/usr/local/bin/uv")

    update._pip_check()

    assert commands == [
        [update.sys.executable, "-m", "pip", "check"],
        [
            "/usr/local/bin/uv",
            "pip",
            "check",
            "--python",
            update.sys.executable,
        ],
    ]


def test_update_extracts_only_external_runtime_requirements(tmp_path: Path) -> None:
    wheel = tmp_path / "ficelle_pro.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "ficelle_pro-0.3.7.dist-info/METADATA",
            "\n".join(
                (
                    "Metadata-Version: 2.4",
                    "Name: ficelle-pro",
                    "Version: 0.3.7",
                    "Requires-Dist: ficelle-router==0.3.7",
                    "Requires-Dist: packaging>=24",
                    "Requires-Dist: cryptography>=42; python_version >= '3.11'",
                    "",
                )
            ),
        )

    assert update._wheel_runtime_requirements(wheel) == (
        "packaging>=24",
        "cryptography>=42; python_version >= '3.11'",
    )


def test_update_check_loop_retries_a_failed_check(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    results = iter(({"status": "error"}, {"status": "up_to_date"}))

    class _StopLoop(Exception):
        pass

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) >= 3:
            raise _StopLoop()

    monkeypatch.setattr(update, "spawn_recovery", lambda: False)
    monkeypatch.setattr(update, "check_for_updates", lambda: next(results))
    monkeypatch.setattr(update.time, "sleep", fake_sleep)

    with pytest.raises(_StopLoop):
        update.update_check_loop()

    assert sleeps == [update.UPDATE_CHECK_START_DELAY_SECONDS, update.UPDATE_ERROR_RETRY_SECONDS, update.UPDATE_CHECK_INTERVAL_SECONDS]


def test_update_check_loop_requests_detached_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "marker.json"
    marker.touch()
    recovery_calls: list[bool] = []
    sleeps: list[float] = []

    class _StopLoop(Exception):
        pass

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise _StopLoop()

    monkeypatch.setattr(update, "update_recovery_marker_path", lambda: marker)
    monkeypatch.setattr(update, "spawn_recovery", lambda: recovery_calls.append(True) or True)
    monkeypatch.setattr(update, "recover_interrupted_update", lambda: pytest.fail("recovery must be detached"))
    monkeypatch.setattr(update, "read_update_status", lambda: {"status": "installing"})
    monkeypatch.setattr(update.time, "sleep", fake_sleep)

    with pytest.raises(_StopLoop):
        update.update_check_loop()

    assert recovery_calls == [True, True]
    assert sleeps == [update.UPDATE_CHECK_START_DELAY_SECONDS, update.UPDATE_ERROR_RETRY_SECONDS]


def test_update_package_lock_is_shared_with_pro_installer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lock_platform,
) -> None:
    # Also run on the no-`fcntl` leg: `exclusive_package_install_lock` used to guard its `flock`
    # calls with `if fcntl is not None`, so where `fcntl` is absent it took no lock at all and
    # this sharing was nominal — an update could run straight through a Pro install.
    from ficelle import pro_install

    lock_path = tmp_path / "pro-install.lock"
    monkeypatch.setattr(update, "package_install_lock_path", lambda: lock_path)
    monkeypatch.setattr(pro_install, "install_lock_path", lambda: lock_path)

    with pro_install.exclusive_install_lock():
        with pytest.raises(update.UpdateInProgress):
            with update.exclusive_package_install_lock():
                pass


def test_apply_update_requires_a_managed_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = update.parse_release_manifest(
        compact_manifest(),
        source_url="https://install.ficelle.ai/api/releases/latest/core",
    )
    update._write_status({
        **manifest.internal_dict(),
        "status": "available",
        "update_available": True,
    })
    monkeypatch.setattr(update, "package_install_lock_path", lambda: tmp_path / "pro-install.lock")
    monkeypatch.setattr(update, "_is_pro_installed", lambda: False)
    monkeypatch.setattr(update, "_managed_service", lambda: (object(), False))

    assert update.apply_update() == 1
    status = update.read_update_status()
    assert status["status"] == "failed"
    assert "managed Ficelle service" in status["message"]


def test_interrupted_update_recovery_marker_is_durable_and_restores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    purelib = tmp_path / "purelib"
    backup_dir = tmp_path / "recovery" / "backup"
    original = purelib / "ficelle"
    backup = backup_dir / "0-ficelle"
    original.mkdir(parents=True)
    backup.mkdir(parents=True)
    monkeypatch.setattr(update.sysconfig, "get_paths", lambda: {"purelib": str(purelib)})
    monkeypatch.setattr(update, "update_recovery_dir", lambda: tmp_path / "recovery")
    monkeypatch.setattr(update, "package_install_lock_path", lambda: tmp_path / "pro-install.lock")
    entries = [update._BackupEntry(original, backup)]

    update._write_recovery_marker(entries, target_version="0.1.4")
    restored: list[list[update._BackupEntry]] = []
    restarted: list[bool] = []

    class _Backend:
        def restart(self) -> int:
            restarted.append(True)
            return 0

    monkeypatch.setattr(update, "_restore_installed_packages", lambda value: restored.append(value))
    monkeypatch.setattr(update, "_managed_service", lambda: (_Backend(), True))

    assert update.recover_interrupted_update() is True
    assert restored == [entries]
    assert restarted == [True]
    assert not (tmp_path / "recovery").exists()
