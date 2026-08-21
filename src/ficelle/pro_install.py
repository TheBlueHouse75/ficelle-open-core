"""Install (or upgrade to) the Ficelle Pro pack from a license key — the engine behind the
in-app unlock (paste your key in the admin, no terminal).

A running core daemon cannot pip-install into its own environment and restart itself: the
LaunchAgent restart boots the service out (killing the daemon) before it relaunches, so a
self-restart would die half-way. Instead ``POST /admin/license/install`` spawns
``ficelle install-pro`` as a DETACHED helper that outlives the restart; this module is that
helper's engine. It downloads the license-gated wheel and pip-installs it into the current
interpreter's environment (which pulls the open core too). The open core keeps its
one-directional independence — this fetches and installs a wheel, it never imports ``ficelle_pro``.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
import zipfile
from contextlib import contextmanager
from email.parser import Parser
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

from packaging.utils import InvalidWheelFilename, parse_wheel_filename

from ficelle import __version__ as CORE_VERSION
from ficelle.install import CommandResult, pip_is_unavailable
from ficelle.json_store import atomic_write_json, load_json
from ficelle.probe_lock import file_lock
from ficelle.runtime_paths import RuntimePaths
from ficelle.url_security import uses_secure_http_transport

_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
MAX_PRO_WHEEL_BYTES = 256 * 1024 * 1024
ACTIVE_INSTALL_STATUS_TTL_SECONDS = 15 * 60
_ACTIVE_INSTALL_STATUSES = frozenset({"queued", "installing", "restarting"})

# The private, license-gated wheel endpoint — same default as the standalone bootstrap script
# (scripts/bootstrap-ficelle.py); overridable for staging/testing via FICELLE_WHEEL_URL.
DEFAULT_WHEEL_URL = f"https://install.ficelle.ai/api/releases/{CORE_VERSION}/wheel"
PRO_RUNTIME_REQUIREMENTS = ("cryptography>=42",)

# pip rejects an invalid wheel filename, so the download is saved under the real wheel name the
# service advertises (Content-Disposition); this generic-but-valid name is the fallback.
_FALLBACK_WHEEL_NAME = "ficelle_pro-0-py3-none-any.whl"


class ProInstallError(Exception):
    """A step of the Pro pack install (download or pip) failed. The message is key-free."""


class ProInstallInProgress(Exception):
    """Raised when another Ficelle Pro installation is already queued or running."""


def install_status_path() -> Path:
    return RuntimePaths.from_env().router_dir / "pro-install-status.json"


def install_status_lock_path() -> Path:
    return install_status_path().with_suffix(".lock")


def install_lock_path() -> Path:
    return RuntimePaths.from_env().router_dir / "pro-install.lock"


def write_install_status(status: str, message: str | None = None) -> None:
    """Persist a key-free install state shared by the detached helper and admin UI."""
    path = install_status_path()
    payload: dict[str, Any] = {"status": status, "updated_at": time.time()}
    if message:
        payload["message"] = message
    atomic_write_json(path, payload)
    path.chmod(0o600)


def best_effort_write_install_status(status: str, message: str | None = None) -> bool:
    """Report status without raising; callers decide whether a missed update is blocking."""
    try:
        write_install_status(status, message)
    except OSError:
        return False
    return True


def read_install_status() -> dict[str, Any]:
    path = install_status_path()
    if not path.exists():
        return {"status": "idle"}
    invalid = object()
    payload = load_json(path, invalid)
    if not isinstance(payload, dict) or not isinstance(payload.get("status"), str):
        return {"status": "failed", "message": "The Ficelle Pro install status could not be read."}
    return {
        key: payload[key]
        for key in ("status", "message", "updated_at")
        if key in payload
    }


def queue_install() -> None:
    """Atomically reserve the single installer slot and publish its queued state."""
    with exclusive_install_lock(install_status_lock_path(), wait=True):
        current = read_install_status()
        status = current.get("status")
        updated_at = current.get("updated_at")
        active_age = (
            time.time() - float(updated_at)
            if isinstance(updated_at, (int, float))
            else 0.0
        )
        if status in _ACTIVE_INSTALL_STATUSES and active_age <= ACTIVE_INSTALL_STATUS_TTL_SECONDS:
            raise ProInstallInProgress("another Ficelle Pro installation is already queued or running")
        # A terminal or stale status can still race with a helper that has not released the
        # actual install lock yet. Probe that lock before publishing a new reservation; the
        # queued helper still waits when a new direct CLI install wins the gap after this probe.
        with exclusive_install_lock():
            pass
        write_install_status("queued")


def _effective_origin(url: str) -> tuple[str, str | None, int | None]:
    parsed = urlparse(url)
    default_port = 443 if parsed.scheme == "https" else 80 if parsed.scheme == "http" else None
    return parsed.scheme.lower(), parsed.hostname, parsed.port or default_port


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Allow redirects only within the wheel endpoint origin so Bearer credentials cannot leak."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        if _effective_origin(req.full_url) != _effective_origin(newurl):
            raise ProInstallError("refusing a cross-origin redirect for the Pro pack download")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_same_origin(request: urllib.request.Request, *, timeout: float) -> Any:
    return urllib.request.build_opener(_SameOriginRedirectHandler()).open(request, timeout=timeout)


def pro_wheel_url() -> str:
    return os.getenv("FICELLE_WHEEL_URL", DEFAULT_WHEEL_URL)


@contextmanager
def exclusive_install_lock(
    lock_path: Path | None = None,
    *,
    wait: bool = False,
) -> Iterator[None]:
    """Acquire an advisory file lock, using the installer lock path by default.

    Callers may supply a separate path for short serialization work; installer callers hold
    the default lock through package installation and service restart. ``wait=True`` waits
    indefinitely; otherwise a lock already held raises ProInstallInProgress.

    Uses ``probe_lock.file_lock``, the tree's one primitive that speaks both ``fcntl.flock`` and
    ``msvcrt.locking``. Importing ``fcntl`` here directly is what broke this module on Windows.
    """
    path = lock_path or install_lock_path()
    with file_lock(path, blocking=wait) as acquired:
        if not acquired:
            raise ProInstallInProgress("another Ficelle Pro installation is already running")
        yield


def _require_secure_url(url: str) -> None:
    """The wheel is pip-installed (arbitrary code), so refuse to fetch it over a channel a
    network attacker could tamper with: require https, except http on loopback for the dev mock.
    """
    if uses_secure_http_transport(url):
        return
    raise ProInstallError("refusing to fetch the Pro pack over an insecure URL (need https, or http on loopback)")


def _wheel_filename(headers: Any) -> str:
    """A valid, path-safe ``.whl`` name for the download — pip rejects an invalid wheel filename.

    Honor the service's advertised name (Content-Disposition, the real ``…-py3-none-any.whl``) but
    keep only its last path segment and require a ``.whl`` suffix, so a header value can never
    escape ``dest_dir``. Fall back to a valid generic name when no usable header is present.
    """
    raw = ""
    get_filename = getattr(headers, "get_filename", None)
    if callable(get_filename):
        try:
            raw = str(get_filename() or "")
        except Exception:  # a malformed header must not abort the install
            raw = ""
    name = raw.strip().replace("\\", "/").split("/")[-1]
    # Require a plain, reasonably short ``.whl`` name; reject a leading dot, control chars / NUL
    # (which would raise on write), and over-long names — falling back so a surprising header can
    # never crash the install rather than just save under the wrong name.
    if name.startswith(".") or not name.isprintable() or len(name) > 128:
        return _FALLBACK_WHEEL_NAME
    try:
        parse_wheel_filename(name)
    except InvalidWheelFilename:
        return _FALLBACK_WHEEL_NAME
    return name


def download_pro_wheel(
    license_key: str, dest_dir: Path, *, opener: Callable[..., Any] | None = None
) -> Path:
    """Download the license-gated Pro wheel into ``dest_dir`` and return its path.

    The key authenticates the download (Bearer) and is never logged; a failure raises
    ProInstallError with a key-free message. ``opener`` is injectable so tests run offline.
    """
    url = pro_wheel_url()
    _require_secure_url(url)
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {license_key}"})
    open_url = opener or _open_same_origin
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open_url(request, timeout=180) as response:
            filename = _wheel_filename(getattr(response, "headers", None))
            wheel = dest_dir / filename
            downloaded = 0
            try:
                with wheel.open("wb") as output:
                    while chunk := response.read(_DOWNLOAD_CHUNK_BYTES):
                        downloaded += len(chunk)
                        if downloaded > MAX_PRO_WHEEL_BYTES:
                            raise ProInstallError("Pro pack download exceeds the maximum allowed size")
                        output.write(chunk)
            except Exception:
                wheel.unlink(missing_ok=True)
                raise
    except ProInstallError:
        raise
    except Exception as exc:  # network/auth/transport — never surface the request body or the key
        raise ProInstallError(f"could not download the Pro pack ({type(exc).__name__})") from exc
    return wheel


def _command_result(command: list[str], result: CommandResult | int) -> CommandResult:
    if isinstance(result, CommandResult):
        return result
    return CommandResult(command=command, returncode=result)


def _resolve_executable(
    name: str,
    candidates: list[str],
    *,
    preferred: list[str] | None = None,
) -> str | None:
    discovered = shutil.which(name)
    for candidate in [*(preferred or []), discovered or "", *candidates]:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _uv_executable() -> str | None:
    return _resolve_executable(
        "uv",
        [
            str(Path.home() / ".local/bin/uv"),
            str(Path.home() / ".cargo/bin/uv"),
            "/opt/homebrew/bin/uv",
            "/usr/local/bin/uv",
        ],
        preferred=[os.getenv("FICELLE_UV_BIN", "").strip()],
    )


def detached_install_command() -> list[str]:
    """Command that survives restarting the managed daemon.

    A POSIX session is enough for launchd, but systemd kills every process left in the
    service cgroup. Under a systemd service, start the helper in its own transient user
    scope; ``systemd-run --scope`` inherits the supplied environment without placing the
    license key on argv.
    """
    command = [sys.executable, "-m", "ficelle", "install-pro"]
    if not os.getenv("INVOCATION_ID"):
        return command
    systemd_run = _resolve_executable("systemd-run", ["/usr/bin/systemd-run"])
    if not systemd_run:
        raise ProInstallError("systemd-run is required to install Ficelle Pro from the managed service")
    unit = f"ficelle-pro-install-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    return [
        systemd_run,
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        f"--unit={unit}",
        *command,
    ]


def install_pro_wheel(
    wheel: Path,
    *,
    runner: Callable[[list[str]], CommandResult | int] | None = None,
) -> None:
    """Install ``wheel`` into the existing Core environment without consulting PyPI."""
    verify_pro_core_compatibility(wheel)
    run = runner or _pip_run
    dependency_command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        *PRO_RUNTIME_REQUIREMENTS,
    ]
    dependency_result = _command_result(
        dependency_command,
        run(dependency_command),
    )
    if dependency_result.returncode != 0:
        if not pip_is_unavailable(dependency_result):
            raise ProInstallError("pip install of the Pro runtime failed")
        uv = _uv_executable()
        if not uv:
            raise ProInstallError("pip is unavailable and uv could not be found")
        uv_dependency_command = [
            uv,
            "pip",
            "install",
            "--python",
            sys.executable,
            *PRO_RUNTIME_REQUIREMENTS,
        ]
        if _command_result(
            uv_dependency_command,
            run(uv_dependency_command),
        ).returncode != 0:
            raise ProInstallError("uv install of the Pro runtime failed")

    pip_command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        str(wheel),
    ]
    pip_result = _command_result(pip_command, run(pip_command))
    if pip_result.returncode == 0:
        return
    if not pip_is_unavailable(pip_result):
        raise ProInstallError("pip install of the Pro pack failed")
    uv = _uv_executable()
    if not uv:
        raise ProInstallError("pip is unavailable and uv could not be found")
    uv_command = [
        uv,
        "pip",
        "install",
        "--python",
        sys.executable,
        "--reinstall",
        "--no-deps",
        str(wheel),
    ]
    uv_result = _command_result(uv_command, run(uv_command))
    if uv_result.returncode != 0:
        raise ProInstallError("uv install of the Pro pack failed")


def verify_pro_core_compatibility(wheel: Path) -> None:
    try:
        with zipfile.ZipFile(wheel) as archive:
            metadata_name = next(
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            )
            metadata = Parser().parsestr(
                archive.read(metadata_name).decode("utf-8")
            )
    except (OSError, UnicodeError, zipfile.BadZipFile, StopIteration) as exc:
        raise ProInstallError(
            "Pro wheel metadata could not be verified"
        ) from exc
    requirements = metadata.get_all("Requires-Dist", [])
    expected = f"ficelle-router=={CORE_VERSION}"
    normalized = {
        requirement.replace(" ", "").lower()
        for requirement in requirements
    }
    if expected not in normalized:
        raise ProInstallError(
            f"Pro wheel is incompatible with Ficelle Core {CORE_VERSION}"
        )


def _pip_run(command: list[str]) -> CommandResult:
    completed = subprocess.run(command, capture_output=True, text=True)
    return CommandResult(command, completed.returncode, completed.stdout, completed.stderr)


def _verify_wheel_checksum(wheel: Path) -> None:
    """Verify the wheel against a pinned SHA-256 before it is executed, when one is provided via
    ``FICELLE_WHEEL_SHA256`` (parity with the standalone bootstrap's ``--sha256``).

    Absent a pin, the trust boundary for the in-app "latest wheel" install is TLS plus the
    license-gated endpoint: a checksum served by the same index would add nothing against an
    endpoint compromise, and no separate trust root is available at first-install time (the signed
    entitlement only exists after activation, which happens after the pack is installed). An
    operator who wants the extra guarantee pins the hash out-of-band via the env var.
    """
    expected = os.getenv("FICELLE_WHEEL_SHA256", "").strip().lower()
    if not expected:
        return
    actual = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if actual != expected:
        raise ProInstallError("Pro pack checksum mismatch — refusing to install")


def install_pro(
    license_key: str,
    *,
    opener: Callable[..., Any] | None = None,
    runner: Callable[[list[str]], CommandResult | int] | None = None,
) -> None:
    """Download + pip-install the Pro pack (no restart). Raises ProInstallError on failure.

    The caller (the ``install-pro`` helper) restarts the service afterwards so the freshly
    installed pack is loaded.
    """
    if not license_key:
        raise ProInstallError("a license key is required to install Ficelle Pro")
    with tempfile.TemporaryDirectory(prefix="ficelle-pro-install-") as tmp:
        wheel = download_pro_wheel(license_key, Path(tmp), opener=opener)
        _verify_wheel_checksum(wheel)
        install_pro_wheel(wheel, runner=runner)
