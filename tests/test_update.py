from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

import ficelle
from ficelle import license_ops, update


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
    monkeypatch.setattr(update, "update_status_path", lambda: tmp_path / "update-status.json")


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


def test_check_for_updates_accepts_entitlement_authorized_pro_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = compact_manifest()
    payload["pro"] = {
        "wheel_url": f"https://install.ficelle.ai/ficelle_pro-{NEXT_VERSION}-py3-none-any.whl",
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

    assert status["pro_update_required"] is False
    assert update.public_update_status()["pro_artifact_available"] is True


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
    assert "authorization" in status["message"]


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
