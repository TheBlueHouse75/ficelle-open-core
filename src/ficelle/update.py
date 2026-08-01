"""Release discovery and safe Core update orchestration.

The router is a long-running process and must never replace its own interpreter while it is
serving requests.  This module therefore keeps three concerns separate:

* a small, redacted status file that the admin UI can read without doing network I/O;
* a release checker that understands both Ficelle's release manifest and GitHub Releases;
* a detached helper that downloads, verifies, installs, smoke-tests, and restarts the service.

The Pro pack is deliberately not coupled to a stored license key. A release manifest may
advertise an authenticated Pro artifact, but the caller must provide either a managed token
through ``FICELLE_UPDATE_PRO_TOKEN`` or the already-cached entitlement token. A Core-only
release remains fully independent.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import urllib.request
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from email.parser import Parser
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import urlparse

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows foreground installs
    fcntl = None  # type: ignore[assignment]

from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from ficelle import __version__ as CORE_VERSION
from ficelle.json_store import atomic_write_json, load_json
from ficelle.runtime_paths import RuntimePaths
from ficelle.url_security import uses_secure_http_transport


DEFAULT_UPDATE_MANIFEST_URL = (
    "https://api.github.com/repos/TheBlueHouse75/ficelle-open-core/releases/latest"
)
UPDATE_CHECK_TTL_SECONDS = 24 * 60 * 60
UPDATE_CHECK_START_DELAY_SECONDS = 5
UPDATE_CHECK_INTERVAL_SECONDS = 6 * 60 * 60
UPDATE_ERROR_RETRY_SECONDS = 15 * 60
ACTIVE_UPDATE_STALE_SECONDS = 2 * 60 * 60
UPDATE_COMMAND_TIMEOUT_SECONDS = 10 * 60
MAX_MANIFEST_BYTES = 1_000_000
MAX_WHEEL_BYTES = 256 * 1024 * 1024
ACTIVE_UPDATE_STATUSES = frozenset({"queued", "installing", "restarting"})
_SHA256_LENGTH = 64
_CHECK_LOCK = threading.Lock()
_RUNTIME_PATHS = RuntimePaths.from_env()


class UpdateError(Exception):
    """A key-free update error suitable for CLI and admin status output."""


class UpdateInProgress(UpdateError):
    """Raised when another update is already queued or being applied."""


class UpdateUnavailable(UpdateError):
    """Raised when there is no verified artifact that can be installed."""


@dataclass(frozen=True)
class ReleaseArtifact:
    distribution: str
    version: str
    url: str
    sha256: str
    filename: str
    authorization: str = "none"

    def internal_dict(self, prefix: str) -> dict[str, str]:
        return {
            f"{prefix}_wheel_url": self.url,
            f"{prefix}_sha256": self.sha256,
            f"{prefix}_filename": self.filename,
            f"{prefix}_authorization": self.authorization,
        }


@dataclass(frozen=True)
class ReleaseManifest:
    version: str
    channel: str
    release_url: str
    notes: str
    published_at: str | None
    core: ReleaseArtifact
    pro: ReleaseArtifact | None = None

    def internal_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "latest_version": self.version,
            "channel": self.channel,
            "release_url": self.release_url,
            "release_notes": self.notes,
            "published_at": self.published_at,
            **self.core.internal_dict("core"),
        }
        if self.pro is not None:
            payload.update(self.pro.internal_dict("pro"))
        else:
            payload.update({
                "pro_wheel_url": None,
                "pro_sha256": None,
                "pro_filename": None,
                "pro_authorization": None,
            })
        return payload


def update_status_path() -> Path:
    # Capture the selected service context at import time. The CLI temporarily exposes a
    # persisted FICELLE_HOME while importing runtime modules, then restores the parent env;
    # resolving from os.environ later would silently write the update state to ~/.ficelle.
    return _RUNTIME_PATHS.router_dir / "update-status.json"


def update_lock_path() -> Path:
    return update_status_path().with_suffix(".lock")


def package_install_lock_path() -> Path:
    """Return the lock shared with the standalone Pro installer."""
    return _RUNTIME_PATHS.router_dir / "pro-install.lock"


def update_recovery_dir() -> Path:
    """Return the durable workspace used to recover an interrupted package swap."""
    return _RUNTIME_PATHS.router_dir / "update-recovery"


def update_recovery_marker_path() -> Path:
    return update_recovery_dir() / "marker.json"


def update_recovery_backup_dir() -> Path:
    return update_recovery_dir() / "backup"


def manifest_url() -> str:
    return os.getenv("FICELLE_UPDATE_MANIFEST_URL", DEFAULT_UPDATE_MANIFEST_URL).strip()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_text(value: Any, limit: int = 5000) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _safe_sha256(value: Any) -> str:
    text = str(value or "").strip().lower()
    if len(text) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in text):
        raise UpdateError("release artifact checksum is invalid")
    return text


def _normalized_version(value: Any) -> str:
    raw = str(value or "").strip()
    if raw.startswith(("v", "V")):
        raw = raw[1:]
    try:
        return str(Version(raw))
    except InvalidVersion as exc:
        raise UpdateError("release version is invalid") from exc


def _safe_release_url(value: Any, fallback: str) -> str:
    candidate = str(value or fallback).strip()
    if not candidate:
        return fallback
    if not uses_secure_http_transport(candidate):
        raise UpdateError("release URL must use HTTPS")
    return candidate


def _safe_wheel_filename(value: Any, *, distribution: str) -> str:
    name = str(value or "").strip().replace("\\", "/").split("/")[-1]
    try:
        parsed_distribution, _version, _build, _tags = parse_wheel_filename(name)
    except InvalidWheelFilename as exc:
        raise UpdateError("release artifact filename is invalid") from exc
    if str(parsed_distribution) != distribution:
        raise UpdateError("release artifact distribution is unexpected")
    return name


def _artifact_from_mapping(
    raw: Mapping[str, Any],
    *,
    distribution: str,
    version: str,
    source_url: str,
) -> ReleaseArtifact:
    raw_url = raw.get("wheel_url") or raw.get("url")
    if not raw_url:
        raise UpdateError("release artifact URL is missing")
    url = _safe_release_url(raw_url, source_url)
    filename = str(raw.get("filename") or Path(urlparse(url).path).name)
    filename = _safe_wheel_filename(filename, distribution=distribution)
    try:
        parsed_distribution, parsed_version, _build, _tags = parse_wheel_filename(filename)
    except InvalidWheelFilename as exc:  # pragma: no cover - guarded by _safe_wheel_filename
        raise UpdateError("release artifact filename is invalid") from exc
    if str(parsed_distribution) != distribution or Version(str(parsed_version)) != Version(version):
        raise UpdateError("release artifact version does not match the release")
    digest = raw.get("sha256") or raw.get("digest")
    if isinstance(digest, str) and digest.lower().startswith("sha256:"):
        digest = digest.split(":", 1)[1]
    return ReleaseArtifact(
        distribution=distribution,
        version=version,
        url=url,
        sha256=_safe_sha256(digest),
        filename=filename,
        authorization=str(raw.get("authorization") or "none").strip().lower(),
    )


def _github_artifact(
    assets: list[Mapping[str, Any]],
    *,
    distribution: str,
    version: str,
    source_url: str,
) -> ReleaseArtifact:
    for asset in assets:
        name = str(asset.get("name") or "")
        try:
            safe_name = _safe_wheel_filename(name, distribution=distribution)
            parsed_distribution, parsed_version, _build, _tags = parse_wheel_filename(safe_name)
        except UpdateError:
            continue
        if str(parsed_distribution) != distribution or Version(str(parsed_version)) != Version(version):
            continue
        digest = asset.get("digest") or asset.get("sha256")
        if isinstance(digest, str) and digest.lower().startswith("sha256:"):
            digest = digest.split(":", 1)[1]
        if not digest:
            continue
        download_url = asset.get("browser_download_url")
        if not isinstance(download_url, str) or not download_url.strip():
            continue
        return ReleaseArtifact(
            distribution=distribution,
            version=version,
            url=_safe_release_url(download_url, source_url),
            sha256=_safe_sha256(digest),
            filename=safe_name,
        )
    raise UpdateError(f"{distribution} release artifact is missing a verified SHA-256")


def parse_release_manifest(payload: Mapping[str, Any], *, source_url: str) -> ReleaseManifest:
    """Parse Ficelle's compact manifest or GitHub's latest-release JSON response."""
    if payload.get("tag_name"):
        version = _normalized_version(payload.get("tag_name"))
        raw_assets = payload.get("assets")
        assets = [asset for asset in raw_assets if isinstance(asset, Mapping)] if isinstance(raw_assets, list) else []
        core = _github_artifact(assets, distribution="ficelle-router", version=version, source_url=source_url)
        return ReleaseManifest(
            version=version,
            channel="stable" if not payload.get("prerelease") else "preview",
            release_url=_safe_release_url(payload.get("html_url"), source_url),
            notes=_safe_text(payload.get("body")),
            published_at=_safe_text(payload.get("published_at"), 80) or None,
            core=core,
        )

    version = _normalized_version(payload.get("version"))
    raw_core = payload.get("core")
    core_mapping = raw_core if isinstance(raw_core, Mapping) else payload
    raw_pro = payload.get("pro")
    pro_mapping = raw_pro if isinstance(raw_pro, Mapping) else None
    core = _artifact_from_mapping(
        core_mapping,
        distribution="ficelle-router",
        version=version,
        source_url=source_url,
    )
    pro = (
        _artifact_from_mapping(
            pro_mapping,
            distribution="ficelle-pro",
            version=version,
            source_url=source_url,
        )
        if pro_mapping is not None and (pro_mapping.get("wheel_url") or pro_mapping.get("url"))
        else None
    )
    return ReleaseManifest(
        version=version,
        channel=_safe_text(payload.get("channel"), 32) or "stable",
        release_url=_safe_release_url(payload.get("release_url"), source_url),
        notes=_safe_text(payload.get("notes") or payload.get("release_notes")),
        published_at=_safe_text(payload.get("published_at"), 80) or None,
        core=core,
        pro=pro,
    )


def _open_json_url(
    url: str,
    *,
    opener: Callable[[urllib.request.Request, float], Any] | None = None,
) -> Mapping[str, Any]:
    if not uses_secure_http_transport(url):
        raise UpdateError("update manifest URL must use HTTPS")
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": f"Ficelle/{CORE_VERSION} update-check",
        },
    )
    open_url = opener or _open_secure_url
    try:
        with open_url(request, 20) as response:
            body = response.read(MAX_MANIFEST_BYTES + 1)
    except Exception as exc:
        raise UpdateError(f"update check failed ({type(exc).__name__})") from exc
    if len(body) > MAX_MANIFEST_BYTES:
        raise UpdateError("update manifest is too large")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise UpdateError("update manifest is invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise UpdateError("update manifest must be an object")
    return payload


def _is_pro_installed() -> bool:
    try:
        from ficelle.pro import pro_installed

        return bool(pro_installed())
    except Exception:
        return False


def _same_origin(left: str, right: str) -> bool:
    try:
        left_url = urlparse(left)
        right_url = urlparse(right)
        left_port = left_url.port or (443 if left_url.scheme.lower() == "https" else 80)
        right_port = right_url.port or (443 if right_url.scheme.lower() == "https" else 80)
        return (
            left_url.scheme.lower() == right_url.scheme.lower()
            and (left_url.hostname or "").lower() == (right_url.hostname or "").lower()
            and left_port == right_port
        )
    except ValueError:
        return False


class _SecureRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, *, same_origin_only: bool) -> None:
        super().__init__()
        self.same_origin_only = same_origin_only

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_handle: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> urllib.request.Request | None:
        if not uses_secure_http_transport(new_url):
            raise UpdateError("refusing an insecure redirect for the release artifact")
        if self.same_origin_only and not _same_origin(request.full_url, new_url):
            raise UpdateError("refusing a cross-origin redirect for the authenticated release artifact")
        return super().redirect_request(request, file_handle, code, message, headers, new_url)


def _open_secure_url(
    request: urllib.request.Request,
    timeout: float,
    *,
    same_origin_only: bool = False,
) -> Any:
    return urllib.request.build_opener(
        _SecureRedirectHandler(same_origin_only=same_origin_only)
    ).open(request, timeout=timeout)


def _pro_update_token(artifact: ReleaseArtifact) -> str | None:
    if artifact.authorization == "none":
        return None
    if artifact.authorization == "bearer":
        token = os.getenv("FICELLE_UPDATE_PRO_TOKEN", "").strip()
        if not token:
            raise UpdateUnavailable("the Pro update token is not available on this machine")
        return token
    if artifact.authorization == "entitlement":
        from ficelle import license_ops

        if not _same_origin(artifact.url, license_ops.service_url()):
            raise UpdateUnavailable("the Pro artifact origin does not match the license service")
        token = license_ops.cached_entitlement_token()
        if not token:
            raise UpdateUnavailable("the active Pro entitlement is not available for update")
        return token
    raise UpdateUnavailable("the Pro update authorization mode is unsupported")


def _status_base() -> dict[str, Any]:
    return {
        "status": "not_checked",
        "update_available": False,
        "current_version": CORE_VERSION,
        "latest_version": CORE_VERSION,
        "channel": "stable",
        "release_url": "",
        "release_notes": "",
        "published_at": None,
        "checked_at": None,
        "message": "",
        "pro_update_required": False,
    }


def read_update_status() -> dict[str, Any]:
    path = update_status_path()
    payload = load_json(path, None)
    if not isinstance(payload, dict):
        return _status_base()
    result = _status_base()
    result.update({key: value for key, value in payload.items() if key in {
        "status",
        "update_available",
        "current_version",
        "latest_version",
        "channel",
        "release_url",
        "release_notes",
        "published_at",
        "checked_at",
        "message",
        "pro_update_required",
        "core_wheel_url",
        "core_sha256",
        "core_filename",
        "core_authorization",
        "pro_wheel_url",
        "pro_sha256",
        "pro_filename",
        "pro_authorization",
        "updated_at",
    }})
    result["current_version"] = CORE_VERSION
    return result


def _status_snapshot(
    payload: Mapping[str, Any],
    *,
    existing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = _status_base()
    result.update(existing if existing is not None else read_update_status())
    result.update(payload)
    result["current_version"] = CORE_VERSION
    result["updated_at"] = _now_iso()
    return result


def _write_status(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = _status_snapshot(payload)
    path = update_status_path()
    try:
        atomic_write_json(path, result)
    except OSError as exc:
        raise UpdateError("local update status could not be saved") from exc
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return result


def public_update_status(status: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return only fields safe to expose to the local admin page."""
    status = status or read_update_status()
    public = {
        key: status.get(key)
        for key in (
            "status",
            "update_available",
            "current_version",
            "latest_version",
            "channel",
            "release_url",
            "release_notes",
            "published_at",
            "checked_at",
            "message",
            "pro_update_required",
        )
    }
    public["pro_artifact_available"] = bool(status.get("pro_wheel_url"))
    return public


def _timestamp_age(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (datetime.now(UTC) - parsed).total_seconds())


def update_check_is_due(status: Mapping[str, Any] | None = None) -> bool:
    current = status or read_update_status()
    if str(current.get("status")) in ACTIVE_UPDATE_STATUSES:
        age = _timestamp_age(current.get("updated_at"))
        return age is not None and age >= ACTIVE_UPDATE_STALE_SECONDS
    age = _timestamp_age(current.get("checked_at"))
    ttl = UPDATE_ERROR_RETRY_SECONDS if str(current.get("status")) == "error" else UPDATE_CHECK_TTL_SECONDS
    return age is None or age >= ttl


def _persist_checked_status(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Write a check result without racing a queued/applying update."""
    try:
        with exclusive_update_lock(wait=False):
            current = read_update_status()
            if str(current.get("status")) in ACTIVE_UPDATE_STATUSES and not update_check_is_due(current):
                return current
            return _write_status(payload)
    except UpdateInProgress:
        # The updater owns the lock; its status file is the authoritative snapshot.
        return read_update_status()


def check_for_updates(
    *,
    force: bool = False,
    opener: Callable[[urllib.request.Request, float], Any] | None = None,
) -> dict[str, Any]:
    """Check for a release and persist a redacted, reusable result."""
    with _CHECK_LOCK:
        existing = read_update_status()
        if str(existing.get("status")) in ACTIVE_UPDATE_STATUSES and not update_check_is_due(existing):
            return existing
        if not force and not update_check_is_due(existing):
            return existing
        try:
            source = manifest_url()
            manifest = parse_release_manifest(_open_json_url(source, opener=opener), source_url=source)
            current_version = Version(CORE_VERSION)
            latest_version = Version(manifest.version)
            available = latest_version > current_version
            pro_required = False
            if available and _is_pro_installed():
                if manifest.pro is None:
                    pro_required = True
                else:
                    try:
                        _pro_update_token(manifest.pro)
                    except UpdateUnavailable:
                        pro_required = True
            payload = {
                **manifest.internal_dict(),
                "status": "available" if available else "up_to_date",
                "update_available": available,
                "pro_update_required": pro_required,
                "checked_at": _now_iso(),
                "message": (
                    "A compatible Pro artifact and update authorization are required for this installed Pro build."
                    if pro_required
                    else ""
                ),
            }
            return _persist_checked_status(payload)
        except (UpdateError, InvalidVersion) as exc:
            error_payload = {
                "status": "error",
                "update_available": False,
                "checked_at": _now_iso(),
                "message": _safe_text(str(exc), 240),
            }
            try:
                return _persist_checked_status(error_payload)
            except UpdateError:
                # A read-only or unavailable runtime home must not turn a normal update-check
                # failure into a traceback. The caller can still render this in-memory result.
                return _status_snapshot(error_payload, existing=existing)


def update_check_loop() -> None:
    """Background checker: delayed once at startup, then periodic and non-blocking."""
    if os.getenv("FICELLE_DISABLE_UPDATE_CHECK") == "1":
        return
    if update_recovery_marker_path().exists():
        try:
            spawn_recovery()
        except Exception:
            # The marker remains durable so the next service start or check cycle can retry the
            # detached recovery helper; recovery must not prevent the router from becoming ready.
            pass
    time.sleep(UPDATE_CHECK_START_DELAY_SECONDS)
    while True:
        status = read_update_status()
        recovery_pending = update_recovery_marker_path().exists()
        try:
            if recovery_pending:
                spawn_recovery()
            else:
                status = check_for_updates()
        except Exception:
            # The serving process must never depend on the release endpoint.
            pass
        retry = (
            UPDATE_ERROR_RETRY_SECONDS
            if recovery_pending or str(status.get("status")) == "error"
            else UPDATE_CHECK_INTERVAL_SECONDS
        )
        time.sleep(retry)


@contextmanager
def exclusive_update_lock(*, wait: bool = False) -> Iterator[None]:
    path = update_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            try:
                flags = fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB
                fcntl.flock(lock_file.fileno(), flags)
            except BlockingIOError as exc:
                raise UpdateInProgress("another Ficelle update is already running") from exc
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def exclusive_package_install_lock(*, wait: bool = False) -> Iterator[None]:
    """Share the Pro installer lock so two package swaps cannot overlap."""
    path = package_install_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            try:
                flags = fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB
                fcntl.flock(lock_file.fileno(), flags)
            except BlockingIOError as exc:
                raise UpdateInProgress("another Ficelle package installation is already running") from exc
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def queue_update() -> dict[str, Any]:
    """Reserve one update slot after obtaining a fresh, verified release status."""
    current = read_update_status()
    if str(current.get("status")) in ACTIVE_UPDATE_STATUSES and not update_check_is_due(current):
        raise UpdateInProgress("another Ficelle update is already running")
    if not bool(current.get("update_available")) or not current.get("core_wheel_url"):
        current = check_for_updates(force=True)
    with exclusive_update_lock(wait=False):
        current = read_update_status()
        if str(current.get("status")) in ACTIVE_UPDATE_STATUSES and not update_check_is_due(current):
            raise UpdateInProgress("another Ficelle update is already running")
        if not bool(current.get("update_available")):
            raise UpdateUnavailable(str(current.get("message") or "Ficelle is already up to date"))
        if bool(current.get("pro_update_required")):
            raise UpdateUnavailable(str(current.get("message") or "a compatible Pro artifact is required"))
        return _write_status({"status": "queued", "message": "Update queued."})


def _detached_helper_command(command: list[str], unit_prefix: str) -> list[str]:
    systemd_run = shutil.which("systemd-run") or next(
        (candidate for candidate in ("/usr/bin/systemd-run", "/bin/systemd-run") if Path(candidate).is_file()),
        None,
    )
    if os.getenv("INVOCATION_ID"):
        if not systemd_run:
            raise UpdateError("systemd-run is required to run the Ficelle update helper from the managed service")
        unit = f"{unit_prefix}-{os.getpid()}-{int(time.time())}"
        return [
            systemd_run,
            "--user",
            "--scope",
            "--quiet",
            "--collect",
            f"--unit={unit}",
            *command,
        ]
    return command


def detached_update_command() -> list[str]:
    return _detached_helper_command(
        [sys.executable, "-m", "ficelle", "update", "--apply"],
        "ficelle-update",
    )


def detached_recovery_command() -> list[str]:
    return _detached_helper_command(
        [sys.executable, "-m", "ficelle", "update", "--recover"],
        "ficelle-update-recovery",
    )


def spawn_update() -> dict[str, Any]:
    """Queue and detach the update helper so the daemon can safely restart itself."""
    status = queue_update()
    try:
        subprocess.Popen(
            detached_update_command(),
            env={**os.environ, "FICELLE_UPDATE_QUEUED": "1"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        try:
            _write_status({"status": "failed", "message": "The Ficelle updater could not start."})
        except UpdateError:
            pass
        raise UpdateError("the Ficelle updater could not start") from exc
    return status


def spawn_recovery() -> bool:
    """Start recovery outside the router so the service can restart safely."""
    if not update_recovery_marker_path().exists():
        return False
    subprocess.Popen(
        detached_recovery_command(),
        env={**os.environ, "FICELLE_UPDATE_RECOVERY": "1"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return True


def _uv_executable() -> str | None:
    configured = os.getenv("FICELLE_UV_BIN", "").strip()
    if configured and Path(configured).is_file() and os.access(configured, os.X_OK):
        return configured
    return shutil.which("uv")


def _run_command(
    command: list[str],
    *,
    timeout: float = UPDATE_COMMAND_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            command,
            124,
            stdout="",
            stderr=f"command timed out after {timeout:g} seconds",
        )
    except OSError as exc:
        return subprocess.CompletedProcess(command, 1, stdout="", stderr=f"{type(exc).__name__}: {exc}")


def _install_wheel(wheel: Path, *, no_deps: bool = False) -> None:
    command = [sys.executable, "-m", "pip", "install", "--force-reinstall"]
    if no_deps:
        command.append("--no-deps")
    command.append(str(wheel))
    result = _run_command(command)
    if result.returncode == 0:
        return
    combined = f"{result.stdout}\n{result.stderr}"
    uv = _uv_executable()
    if "No module named pip" not in combined or not uv:
        raise UpdateError("package installation failed")
    uv_command = [uv, "pip", "install", "--python", sys.executable, "--reinstall"]
    if no_deps:
        uv_command.append("--no-deps")
    uv_command.append(str(wheel))
    result = _run_command(uv_command)
    if result.returncode != 0:
        raise UpdateError("package installation failed")


def _download_artifact(
    artifact: ReleaseArtifact,
    destination: Path,
    *,
    authorization_token: str | None = None,
    opener: Callable[[urllib.request.Request, float], Any] | None = None,
) -> Path:
    if not uses_secure_http_transport(artifact.url):
        raise UpdateError("release artifact URL must use HTTPS")
    headers = {
        "Accept": "application/octet-stream",
        "User-Agent": f"Ficelle/{CORE_VERSION} updater",
    }
    if authorization_token:
        headers["Authorization"] = f"Bearer {authorization_token}"
    request = urllib.request.Request(artifact.url, headers=headers)
    if opener is not None:
        open_url = opener
    else:
        def open_artifact(request: urllib.request.Request, timeout: float) -> Any:
            return _open_secure_url(
                request,
                timeout,
                same_origin_only=bool(authorization_token),
            )

        open_url = open_artifact
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / artifact.filename
    digest = hashlib.sha256()
    downloaded = 0
    try:
        with open_url(request, 180) as response, path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                downloaded += len(chunk)
                if downloaded > MAX_WHEEL_BYTES:
                    raise UpdateError("release artifact is too large")
                digest.update(chunk)
                output.write(chunk)
    except UpdateError:
        path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise UpdateError(f"release artifact download failed ({type(exc).__name__})") from exc
    if digest.hexdigest() != artifact.sha256:
        path.unlink(missing_ok=True)
        raise UpdateError("release artifact checksum mismatch")
    return path


@dataclass(frozen=True)
class _BackupEntry:
    original: Path
    backup: Path


def _package_root() -> Path:
    import ficelle

    root = Path(ficelle.__file__ or "").resolve().parent
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    try:
        root.relative_to(purelib)
    except ValueError as exc:
        raise UpdateError("development/editable install detected; run ficelle-setup from the new checkout") from exc
    return purelib


def _installed_package_paths() -> list[Path]:
    purelib = _package_root()
    paths = [purelib / "ficelle"]
    paths.extend(sorted(purelib.glob("ficelle_router-*.dist-info")))
    paths.extend(sorted(purelib.glob("ficelle_router-*.egg-info")))
    if (purelib / "ficelle_pro").exists():
        paths.append(purelib / "ficelle_pro")
    paths.extend(sorted(purelib.glob("ficelle_pro-*.dist-info")))
    paths.extend(sorted(purelib.glob("ficelle_pro-*.egg-info")))
    return [path for path in paths if path.exists()]


def _backup_installed_packages(destination: Path) -> list[_BackupEntry]:
    destination.mkdir(parents=True, exist_ok=True)
    entries: list[_BackupEntry] = []
    for index, original in enumerate(_installed_package_paths()):
        backup = destination / f"{index}-{original.name}"
        if original.is_dir():
            shutil.copytree(original, backup)
        else:
            shutil.copy2(original, backup)
        entries.append(_BackupEntry(original, backup))
    if not entries:
        raise UpdateError("installed Ficelle package files could not be located")
    return entries


def _remove_installed_package_paths() -> None:
    purelib = _package_root()
    candidates = [purelib / "ficelle", purelib / "ficelle_pro"]
    candidates.extend(purelib.glob("ficelle_router-*.dist-info"))
    candidates.extend(purelib.glob("ficelle_router-*.egg-info"))
    candidates.extend(purelib.glob("ficelle_pro-*.dist-info"))
    candidates.extend(purelib.glob("ficelle_pro-*.egg-info"))
    for path in candidates:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _restore_installed_packages(entries: list[_BackupEntry]) -> None:
    _remove_installed_package_paths()
    for entry in entries:
        if entry.backup.is_dir():
            shutil.copytree(entry.backup, entry.original)
        else:
            entry.original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry.backup, entry.original)


def _recovery_entry_payload(entries: list[_BackupEntry]) -> list[dict[str, str]]:
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    backup_dir = update_recovery_backup_dir().resolve()
    payload: list[dict[str, str]] = []
    for entry in entries:
        original = entry.original.resolve()
        backup = entry.backup.resolve()
        try:
            original_relative = original.relative_to(purelib)
            backup_relative = backup.relative_to(backup_dir)
        except ValueError as exc:
            raise UpdateError("update recovery paths are outside the active environment") from exc
        payload.append(
            {
                "original": str(original_relative),
                "backup": str(backup_relative),
            }
        )
    return payload


def _write_recovery_marker(
    entries: list[_BackupEntry],
    *,
    target_version: str,
    phase: str = "prepared",
) -> None:
    marker = update_recovery_marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "phase": phase,
        "target_version": target_version,
        "entries": _recovery_entry_payload(entries),
        "updated_at": _now_iso(),
    }
    atomic_write_json(marker, payload)
    marker.chmod(0o600)


def _read_recovery_marker() -> tuple[str, str, list[_BackupEntry]]:
    marker = update_recovery_marker_path()
    invalid = object()
    payload = load_json(marker, invalid)
    if not isinstance(payload, dict):
        raise UpdateError("the interrupted update recovery marker is invalid")
    phase = str(payload.get("phase") or "")
    if phase not in {"prepared", "committed"}:
        raise UpdateError("the interrupted update recovery phase is invalid")
    target_version = _normalized_version(payload.get("target_version"))
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise UpdateError("the interrupted update recovery backup is incomplete")
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    backup_dir = update_recovery_backup_dir().resolve()
    entries: list[_BackupEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise UpdateError("the interrupted update recovery backup is invalid")
        original_relative = Path(str(raw_entry.get("original") or ""))
        backup_relative = Path(str(raw_entry.get("backup") or ""))
        if (
            original_relative.is_absolute()
            or backup_relative.is_absolute()
            or ".." in original_relative.parts
            or ".." in backup_relative.parts
        ):
            raise UpdateError("the interrupted update recovery paths are invalid")
        original = (purelib / original_relative).resolve()
        backup = (backup_dir / backup_relative).resolve()
        try:
            original.relative_to(purelib)
            backup.relative_to(backup_dir)
        except ValueError as exc:
            raise UpdateError("the interrupted update recovery paths are invalid") from exc
        if not backup.exists():
            raise UpdateError("the interrupted update recovery backup is missing")
        entries.append(_BackupEntry(original, backup))
    return phase, target_version, entries


def _write_recovery_phase(phase: str) -> None:
    marker = update_recovery_marker_path()
    payload = load_json(marker, None)
    if not isinstance(payload, dict):
        raise UpdateError("the interrupted update recovery marker is invalid")
    payload["phase"] = phase
    payload["updated_at"] = _now_iso()
    atomic_write_json(marker, payload)
    marker.chmod(0o600)


def _clear_update_recovery() -> None:
    shutil.rmtree(update_recovery_dir(), ignore_errors=True)


def _recover_interrupted_update_locked() -> bool:
    marker = update_recovery_marker_path()
    if not marker.exists():
        return False
    phase, target_version, entries = _read_recovery_marker()
    if phase == "committed":
        try:
            _write_status({
                "status": "installed",
                "update_available": False,
                "current_version": target_version,
                "message": f"Ficelle {target_version} is installed.",
            })
        except UpdateError:
            pass
        _clear_update_recovery()
        return True

    # The marker is written before stopping the service and before the first pip mutation. If the
    # helper disappears at any point after that, restoring the durable copy is safe and idempotent.
    _restore_installed_packages(entries)
    backend, managed = _managed_service()
    if managed and backend.restart() != 0:
        raise UpdateError("the previous Ficelle update was rolled back but the service did not restart")
    _write_recovery_phase("committed")
    try:
        _write_status({
            "status": "failed",
            "update_available": False,
            "message": "A previous Ficelle update was interrupted; the previous package was restored.",
        })
    except UpdateError:
        pass
    _clear_update_recovery()
    return True


def recover_interrupted_update() -> bool:
    """Recover a package swap whose detached helper disappeared unexpectedly."""
    if not update_recovery_marker_path().exists():
        return False
    try:
        with exclusive_update_lock(wait=False):
            with exclusive_package_install_lock(wait=False):
                return _recover_interrupted_update_locked()
    except UpdateInProgress:
        return False


def _installed_version() -> str:
    result = _run_command([sys.executable, "-c", "import ficelle; print(ficelle.__version__)"])
    if result.returncode != 0:
        raise UpdateError("updated package could not be imported")
    try:
        return _normalized_version(result.stdout.strip())
    except UpdateError as exc:
        raise UpdateError("updated package reported an invalid version") from exc


def _verify_updated_runtime(*, expect_pro: bool) -> None:
    imports = "import ficelle"
    if expect_pro:
        imports += "; import ficelle_pro"
    result = _run_command([sys.executable, "-c", imports])
    if result.returncode != 0:
        raise UpdateError("updated package runtime dependencies could not be imported")


def _verify_pro_core_compatibility(wheel: Path, *, core_version: str) -> None:
    expected = f"ficelle-router=={core_version}".replace(" ", "").lower()
    try:
        with zipfile.ZipFile(wheel) as archive:
            metadata_name = next(
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            )
            metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
    except (OSError, UnicodeError, zipfile.BadZipFile, StopIteration) as exc:
        raise UpdateError("Pro artifact metadata could not be verified") from exc
    requirements = metadata.get_all("Requires-Dist", [])
    normalized = {requirement.replace(" ", "").lower() for requirement in requirements}
    if expected not in normalized:
        raise UpdateError(f"Pro artifact is incompatible with Ficelle Core {core_version}")


def _managed_service() -> tuple[Any, bool]:
    from ficelle import cli

    paths = cli.service_paths()
    managed = paths.plist.exists() or paths.systemd_unit.exists()
    return cli.current_service_backend(), managed


def _pro_artifact_from_status(status: Mapping[str, Any]) -> ReleaseArtifact | None:
    url = str(status.get("pro_wheel_url") or "")
    if not url:
        return None
    version = _normalized_version(status.get("latest_version"))
    filename = _safe_wheel_filename(status.get("pro_filename"), distribution="ficelle-pro")
    parsed_distribution, parsed_version, _build, _tags = parse_wheel_filename(filename)
    if str(parsed_distribution) != "ficelle-pro" or Version(str(parsed_version)) != Version(version):
        raise UpdateError("Pro release artifact version does not match the release")
    return ReleaseArtifact(
        distribution="ficelle-pro",
        version=version,
        url=_safe_release_url(url, url),
        sha256=_safe_sha256(status.get("pro_sha256")),
        filename=filename,
        authorization=str(status.get("pro_authorization") or "none").strip().lower(),
    )


def _apply_update_locked(
    status: Mapping[str, Any],
    *,
    opener: Callable[[urllib.request.Request, float], Any] | None = None,
) -> int:
    if update_recovery_marker_path().exists():
        _recover_interrupted_update_locked()
        status = read_update_status()
    if not bool(status.get("update_available")) or not status.get("core_wheel_url"):
        try:
            _write_status({"status": "failed", "message": "No verified update is queued."})
        except UpdateError:
            pass
        return 1

    target_version = _normalized_version(status.get("latest_version"))
    core_url = str(status.get("core_wheel_url") or "")
    if not core_url:
        raise UpdateUnavailable("the verified Core artifact URL is missing")
    core = ReleaseArtifact(
        distribution="ficelle-router",
        version=target_version,
        url=_safe_release_url(core_url, core_url),
        sha256=_safe_sha256(status.get("core_sha256")),
        filename=_safe_wheel_filename(status.get("core_filename"), distribution="ficelle-router"),
        authorization=str(status.get("core_authorization") or "none"),
    )
    pro_installed = _is_pro_installed()
    pro = _pro_artifact_from_status(status) if pro_installed else None
    if pro_installed and pro is None:
        raise UpdateUnavailable("a compatible Pro artifact is required for this installed Pro build")
    pro_token = _pro_update_token(pro) if pro is not None else None
    backend, managed = _managed_service()
    if not managed:
        raise UpdateUnavailable("verified updates require a managed Ficelle service; run ficelle-setup first")

    _write_status({"status": "installing", "message": "Installing the verified Ficelle update."})
    with tempfile.TemporaryDirectory(prefix="ficelle-update-") as temporary:
        directory = Path(temporary)
        core_wheel = _download_artifact(core, directory / "core", opener=opener)
        pro_wheel = (
            _download_artifact(
                pro,
                directory / "pro",
                authorization_token=pro_token,
                opener=opener,
            )
            if pro is not None
            else None
        )
        if pro_wheel is not None:
            _verify_pro_core_compatibility(pro_wheel, core_version=target_version)
        try:
            # A previous committed marker is normally cleaned during startup recovery. Clean
            # marker-less leftovers as a second guard before creating the next durable backup.
            _clear_update_recovery()
            backup = _backup_installed_packages(update_recovery_backup_dir())
            _write_recovery_marker(backup, target_version=target_version)
        except Exception:
            _clear_update_recovery()
            raise
        rollback_complete = False
        try:
            if backend.stop() != 0:
                raise UpdateError("the Ficelle service could not be stopped safely")
            _install_wheel(core_wheel)
            if pro_wheel is not None:
                _install_wheel(pro_wheel)
            _verify_updated_runtime(expect_pro=pro_wheel is not None)
            if _installed_version() != target_version:
                raise UpdateError("updated package version does not match the release")
            _write_status({"status": "restarting", "message": "Restarting the Ficelle service."})
            if backend.restart() != 0:
                raise UpdateError("the Ficelle service did not become ready after the update")
            _write_recovery_phase("committed")
        except Exception:
            try:
                backend.stop()
                _restore_installed_packages(backup)
                if backend.restart() != 0:
                    raise UpdateError("the previous Ficelle package was restored but the service did not restart")
                rollback_complete = True
            except Exception:
                pass
            if rollback_complete:
                _clear_update_recovery()
            raise
    try:
        _write_status({
            "status": "installed",
            "update_available": False,
            "current_version": target_version,
            "message": f"Ficelle {target_version} is installed.",
        })
    except UpdateError:
        # The package and service are already healthy; a metadata write failure must not report a
        # false install failure after the restart succeeded.
        pass
    _clear_update_recovery()
    return 0


def apply_update(
    *,
    opener: Callable[[urllib.request.Request, float], Any] | None = None,
) -> int:
    """Apply the queued update and return a process-style exit code."""
    try:
        with exclusive_update_lock(wait=True):
            with exclusive_package_install_lock(wait=False):
                return _apply_update_locked(read_update_status(), opener=opener)
    except Exception as exc:
        message = str(exc) if isinstance(exc, (UpdateError, UpdateUnavailable)) else "The Ficelle update failed."
        try:
            _write_status({"status": "failed", "message": _safe_text(message, 240)})
        except UpdateError:
            pass
        return 1
